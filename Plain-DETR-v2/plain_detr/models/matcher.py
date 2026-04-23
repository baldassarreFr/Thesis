# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from plain_detr.util.box_ops import bbox2delta, box_cxcywh_to_xyxy, generalized_box_iou

if TYPE_CHECKING:
    from plain_detr.main import Config


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_bbox_type: str = "l1",
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_bbox_type: This decides how to calculate box loss.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_bbox_type = cost_bbox_type
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # Flatten per calcolare le matrici in batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            # Controllo di sicurezza: se TUTTO il batch è completamente vuoto
            sizes = [len(v["boxes"]) for v in targets]
            if sum(sizes) == 0:
                empty_idx = torch.empty(0, dtype=torch.int64, device=out_prob.device)
                return [(empty_idx, empty_idx) for _ in range(bs)]

            # Concatena label e box
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Costo Classificazione
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Costo L1 Box
            if self.cost_bbox_type == "l1":
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            elif self.cost_bbox_type == "reparam":
                out_delta = outputs["pred_deltas"].flatten(0, 1)
                out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
                tgt_delta = bbox2delta(out_bbox_old, tgt_bbox)
                cost_bbox = torch.cdist(out_delta[:, None], tgt_delta, p=1).squeeze(1)
            else:
                raise NotImplementedError

            # Costo GIoU
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Matrice di costo finale (flattened)
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            # This is a FIX introduced from the original code to handle the case where some images in the batch are completely empty (no targets).
            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                if sizes[i] == 0:
                    # Immagine vuota: restituisce tensori vuoti
                    indices.append((
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    ))
                else:
                    # c[i] estrae SOLO i costi delle query dell'immagine i contro i target dell'immagine i
                    idx_i, idx_j = linear_sum_assignment(c[i])
                    indices.append((
                        torch.as_tensor(idx_i, dtype=torch.int64),
                        torch.as_tensor(idx_j, dtype=torch.int64),
                    ))
            return indices
            

def build_matcher(args: Config):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_bbox_type="l1" if (not args.reparam) else "reparam",
    )
