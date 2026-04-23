#!/usr/bin/env python3
"""
ZOD Evaluator for Plain-DETR
Uses COCO metrics (pycococotools) for evaluation.
Supports multi-GPU evaluation via DDP.
"""
import json
import os
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from plain_detr.util.misc import all_gather, collate_fn, is_main_process

def generate_coco_ground_truth(dataset, output_path: str):
    """Generate COCO format ground truth bypassing heavy image loading."""
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Pedestrian"},
            {"id": 2, "name": "Vehicle"},
            {"id": 3, "name": "VulnerableVehicle"},
        ],
    }

    annotation_id = 1
    
    # Import needed just for fast image header reading
    from PIL import Image

    for frame_id in dataset.frame_ids:
        # Fast read of image dimensions without loading pixels
        img_dir = dataset.img_folder / frame_id / "camera_front_blur"
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        with Image.open(img_files[0]) as pil_img:
            img_width, img_height = pil_img.size
            
        # The true valid area after our fixed crop
        valid_w = img_width
        valid_h = img_height - 400 - 568  # CROP_TOP and CROP_BOTTOM from zod.py
        image_id = int(frame_id)

        coco_gt["images"].append({"id": image_id, "width": valid_w, "height": valid_h})

        # Directly read raw JSON annotations
        anno_path = dataset.img_folder / frame_id / "annotations" / "object_detection.json"
        with open(anno_path) as f:
            annotations = json.load(f)

        for anno in annotations:
            class_name = anno.get("properties", {}).get("class", "")
            # Mapping class to 1-indexed COCO directly
            class_map = {"Pedestrian": 1, "Vehicle": 2, "VulnerableVehicle": 3}
            
            if class_name not in class_map:
                continue

            coords = anno.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue

            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            
            # Apply the fixed crop shift manually
            x_min_raw = min(xs)
            y_min_raw = min(ys) - 400  # Shift up by CROP_TOP
            x_max_raw = max(xs)
            y_max_raw = max(ys) - 400
            
            # clipping to valid area after crop (just in case some boxes are partially outside)
            x_min = max(0, min(x_min_raw, valid_w))
            y_min = max(0, min(y_min_raw, valid_h))
            x_max = max(0, min(x_max_raw, valid_w))
            y_max = max(0, min(y_max_raw, valid_h))


            # Calculate width and height
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            # Only keep boxes that survived the crop (y_max > 0 and y_min < valid_h)
            if box_w > 0 and box_h > 0:            
                
                coco_gt["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_map[class_name],
                        "bbox": [x_min, y_min, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco_gt, f)

    print(f"Generated fast COCO ground truth: {output_path}")
    return output_path



@torch.no_grad()
def evaluate_with_coco_metrics(
    dataset,
    model,
    postprocessors,
    output_dir: str = ".",
    score_threshold: float = 0.05,
    device: str = "cuda",
    batch_size: int = 2,
    max_samples: int | None = None,
):
    """
    Evaluate model on ZOD dataset using COCO metrics.
    Supports multi-GPU evaluation via DDP.
    """
    model.eval()
    device = torch.device(device)
    
    # 1. Check if we are in Multi-GPU (Distributed) mode
    is_distributed = dist.is_initialized()
    
    # 2. Ground Truth Generation (Only Rank 0 creates the file)
    gt_path = os.path.join(output_dir, "zod_ground_truth.json")
    if is_main_process() and not os.path.exists(gt_path):
        generate_coco_ground_truth(dataset, gt_path)
        
    # Safety barrier: GPUs 1, 2, and 3 stop here and wait for GPU 0 to finish creating the file
    if is_distributed:
        dist.barrier()
        
    # 3. Create DataLoader with the correct Sampler
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
        
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    # 4. ===== All GPUs run inference on their respective data shards =====
    coco_predictions = []
    for idx, (samples, targets) in enumerate(data_loader):
        if max_samples is not None and idx >= max_samples:
            break
            
        samples = samples.to(device)
        outputs = model(samples)
        
        # Use orig_size to scale boxes back to the fixed original coordinates
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        
        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            boxes = result["boxes"]
            labels = result["labels"]
            scores = result["scores"]
            
            for box, label, score in zip(boxes, labels, scores):
                if score.item() > score_threshold:
                    # COCO format requires [x_min, y_min, width, height]
                    xmin, ymin, xmax, ymax = box.tolist()
                    coco_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                    
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label.item()) + 1,
                        "bbox": coco_box,
                        "score": score.item(),
                    })

    # 5. ===== Synchronization: Gather predictions from all GPUs =====
    if is_distributed:
        all_predictions = all_gather(coco_predictions)
    else:
        all_predictions = [coco_predictions]

    # 6. ===== Final Evaluation (ONLY Rank 0 computes the metrics) =====
    if is_main_process():
        # Merge lists from all GPUs into a single flat list
        merged_predictions = []
        for preds in all_predictions:
            merged_predictions.extend(preds)

        # Write the overall predictions JSON file
        pred_coco_path = os.path.join(output_dir, "zod_predictions_coco.json")
        with open(pred_coco_path, "w") as f:
            json.dump(merged_predictions, f)

        # Run pycocotools to calculate metrics
        try:
            coco_gt = COCO(gt_path)
            if len(merged_predictions) == 0:
                print("No predictions found above score threshold.")
                return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
                
            coco_dt = coco_gt.loadRes(pred_coco_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Return the exact metrics that main.py expects for logging
            metrics = {
                "AP": coco_eval.stats[0],
                "AP50": coco_eval.stats[1],
                "AP75": coco_eval.stats[2],
                "AR": coco_eval.stats[8]  # AR for maxDets=100
            }
            return metrics
            
        except Exception as e:
            print(f"COCO Evaluation failed: {e}")
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}

    else:
        # Non-main ranks must return dummy zeros to prevent engine.py from crashing
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}