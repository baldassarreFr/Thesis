# ------------------------------------------------------------------------
# ZOD Evaluator for Plain-DETR
# Uses pycocotools for COCO-standard evaluation metrics
# ------------------------------------------------------------------------

import json
import os
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def generate_coco_ground_truth(dataset, output_path="zod_ground_truth.json"):
    """Generate COCO format ground truth from ZOD dataset."""
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Vehicle"},
            {"id": 2, "name": "VulnerableVehicle"},
            {"id": 3, "name": "Pedestrian"},
        ],
    }

    annotation_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        # Use orig_size for GT JSON (original image dimensions for COCO format)
        height, width = (
            int(target["orig_size"][0].item()),
            int(target["orig_size"][1].item()),
        )
        image_id = int(target["image_id"].item())

        coco_gt["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
            }
        )

        boxes = target["boxes"]
        labels = target["labels"]

        for i in range(len(boxes)):
            box = boxes[i]
            cx, cy, bw, bh = box.tolist()
            x_min = (cx - bw / 2) * width
            y_min = (cy - bh / 2) * height
            box_width = bw * width
            box_height = bh * height

            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(labels[i].item()) + 1,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco_gt, f)
    print(f"Generated COCO ground truth: {output_path}")
    return output_path


def evaluate_with_coco_metrics(
    dataset, model, postprocessors, output_dir=".", score_threshold=0.05
):
    """Evaluate model on ZOD dataset using COCO metrics.

    Args:
        dataset: ZOD dataset
        model: Plain-DETR model
        postprocessors: Dict with 'bbox' postprocessor
        output_dir: Directory to save results
        score_threshold: Minimum score for predictions
    """
    model.eval()
    device = next(model.parameters()).device

    gt_path = os.path.join(output_dir, "zod_ground_truth.json")
    if not os.path.exists(gt_path):
        generate_coco_ground_truth(dataset, gt_path)

    from util.misc import collate_fn

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    coco_predictions = []

    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            outputs = model(samples)

            # Use postprocessor to get final predictions (same as engine.py)
            # CRITICAL: Use orig_size (original image dims) for denormalization to COCO scale
            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0
            ).to(device)
            results = postprocessors["bbox"](outputs, orig_target_sizes)

            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())

                boxes = result["boxes"]
                labels = result["labels"]
                scores = result["scores"]

                for box, label, score in zip(boxes, labels, scores):
                    if score < score_threshold:
                        continue

                    x_min, y_min, x_max, y_max = box.tolist()
                    box_width = x_max - x_min
                    box_height = y_max - y_min

                    coco_predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label.item())
                            + 1,  # Convert to 1-indexed
                            "bbox": [x_min, y_min, box_width, box_height],
                            "score": float(score.item()),
                        }
                    )

    pred_path = os.path.join(output_dir, "zod_predictions.json")
    pred_data = {
        "predictions": coco_predictions,
        "coordinate_space": "original_4k",
        "original_size": [2168, 3848],
    }
    with open(pred_path, "w") as f:
        json.dump(pred_data, f, indent=2)
    print(f"Generated predictions: {pred_path} ({len(coco_predictions)} boxes)")

    # Save separate file for pycocotools (requires list format, not dict)
    pred_coco_path = os.path.join(output_dir, "zod_predictions_coco.json")
    with open(pred_coco_path, "w") as f:
        json.dump(coco_predictions, f)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_coco_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AR": stats[6],
    }
