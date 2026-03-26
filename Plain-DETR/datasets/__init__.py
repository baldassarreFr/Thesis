# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"Dataset loading and evaluation for COCO and ZOD"

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    """This is called in the main.py for evaluation, to get the coco api for evaluation. If the dataset is ZOD, it will return a stub evaluator that can be used for evaluation."""
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if dataset.__class__.__name__ == "ZODDetection":
        return _ZODEvaluatorStub(dataset)
    return None


class _ZODEvaluatorStub:
    """Stub evaluator for ZOD to pass CocoEvaluator init check."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.img_ids = []

    def getImgIds(self):
        return []

    def getCatIds(self):
        return []


def build_dataset(image_set, args):
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic

        return build_coco_panoptic(image_set, args)
    if args.dataset_file == "zod":
        from .zod import build_zod

        return build_zod(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
