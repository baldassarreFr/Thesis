#!/usr/bin/env python3
"""
Plain-DETR Inference Script - Visualize model predictions on COCO images
Usage: uv run python -m plain_detr.inference
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from plain_detr.main import Config
from plain_detr.datasets import build_dataset
from plain_detr.models.detr import build as build_model
from plain_detr.util.box_ops import box_iou, box_cxcywh_to_xyxy


COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COLORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 0.5, 0),
    (0.5, 0, 1),
    (0, 1, 0.5),
    (0.5, 0.5, 0.5),
    (0.8, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.2, 0.8),
]


def main():
    print("=" * 60)
    print("Plain-DETR Inference Script (Single-Pass Optimized)")
    print("=" * 60)

    checkpoint = Path("/root/Plain-DETR-v2/exps/dinov3_vit_small_boxrpe/checkpoint.epoch_11.pth")
    output_dir = Path("/root/Plain-DETR-v2/visualizations/COCO")
    num_images = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_save = 10

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    print("\n1. Loading model config...")
    with open("/root/Plain-DETR-v2/exps/dinov3_vit_small_boxrpe/args.json") as f:
        args_dict = json.load(f)
    args_dict["device"] = device
    args = Config(**args_dict)

    print("2. Building model...")
    model, criterion, postprocessors = build_model(args)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print("   Model loaded")

    print("3. Loading COCO val dataset...")
    dataset = build_dataset("val", args)
    print(f"   Dataset: {len(dataset)} images")

    good_dir = output_dir / "good"
    bad_dir = output_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    print("\nPhase 1: Searching for Good and Bad examples in a single pass...")
    good_idxs = []
    bad_idxs = []

    random.seed(42)
    all_idx = list(range(len(dataset)))
    random.shuffle(all_idx)
    selected_idxs = all_idx[:num_images]

    for idx in tqdm(selected_idxs, desc="Analyzing Images"):
        if len(good_idxs) >= max_save and len(bad_idxs) >= max_save:
            print("\nTarget reached! 10 good and 10 bad found.")
            break

        sample, target = dataset[idx]
        inputs = sample.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        orig_size = target["orig_size"].tolist()
        aug_size = target["size"].tolist()
        target_sizes = torch.tensor([aug_size], device=device)
        orig_target_sizes = torch.tensor([orig_size], device=device)
        results = postprocessors["bbox"](outputs, target_sizes, orig_target_sizes)

        if len(results) == 0 or len(results[0]["scores"]) == 0:
            continue

        gt_boxes = target.get("boxes", torch.tensor([])).to(device)
        gt_labels = target.get("labels", torch.tensor([])).to(device)

        if len(gt_boxes) == 0:
            continue

        h, w = target["size"].tolist()
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes_abs = gt_boxes_xyxy * torch.tensor([w, h, w, h], dtype=torch.float32, device=device)

        res = results[0]
        scores = res["scores"]
        labels = res["labels"]
        pred_boxes = res["boxes"]

        # --- STRICT GOOD LOGIC (Perfect Recall) ---
        if len(good_idxs) < max_save:
            matched_gts = set()
            for score, pred_box, pred_label in zip(scores, pred_boxes, labels):
                if score.item() < 0.8:
                    continue

                ious, _ = box_iou(pred_box.unsqueeze(0), gt_boxes_abs)
                max_iou, best_gt_idx = ious.max(dim=1)

                if max_iou.item() >= 0.5 and pred_label.item() == gt_labels[best_gt_idx.item()].item():
                    matched_gts.add(best_gt_idx.item())

            # Only append if ALL ground truth items were found
            if len(matched_gts) == len(gt_boxes):
                good_idxs.append(idx)

        # --- BAD LOGIC (Searching for any severe error) ---
        if len(bad_idxs) < max_save:
            has_error = False
            for score, pred_box, pred_label in zip(scores, pred_boxes, labels):
                if score.item() < 0.5:
                    continue

                ious, _ = box_iou(pred_box.unsqueeze(0), gt_boxes_abs)
                max_iou_val = ious.max(dim=1)[0].item()
                best_gt_idx = ious.max(dim=1)[1].item()

                if max_iou_val >= 0.5 and pred_label.item() != gt_labels[best_gt_idx].item():
                    has_error = True
                    break
                elif max_iou_val < 0.1:
                    has_error = True
                    break

            if has_error:
                bad_idxs.append(idx)

    print("\nPhase 2: Generating visualizations...")

    # Process Good Images
    for i, idx in enumerate(tqdm(good_idxs, desc="Saving Good")):
        sample, target = dataset[idx]
        img_tensor = sample.clone()
        if img_tensor.min() < 0:
            img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(3, 1, 1)
        img_array = (img_tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        gt_boxes = target.get("boxes", torch.tensor([]))
        gt_labels = target.get("labels", torch.tensor([]))
        aug_h, aug_w = target["size"].tolist()
        gt_boxes_abs = box_cxcywh_to_xyxy(gt_boxes) * torch.tensor([aug_w, aug_h, aug_w, aug_h], dtype=torch.float32)

        inputs = sample.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)

        res = postprocessors["bbox"](
            outputs,
            torch.tensor([target["size"].tolist()], device=device),
            torch.tensor([target["orig_size"].tolist()], device=device),
        )[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(pil_img)
        ax2.imshow(pil_img)

        for box, label in zip(gt_boxes_abs, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            ax1.add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=(0, 1, 0), facecolor="none")
            )
            c_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls_{label}"
            # Text color set to black for readability
            ax1.text(
                x1,
                y1 - 5,
                f"GT: {c_name}",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round", facecolor=(0, 1, 0), alpha=0.8),
            )
        ax1.set_title("Ground Truth", fontsize=12)
        ax1.axis("off")

        for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
            if score.item() < 0.5:
                continue
            x1, y1, x2, y2 = box.tolist()
            color = COLORS[label.item() % len(COLORS)]
            ax2.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
            c_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls_{label}"
            # Text color set to black
            ax2.text(
                x1,
                y1 - 5,
                f"{c_name}: {score:.2f}",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )
        ax2.set_title("Predictions", fontsize=12)
        ax2.axis("off")

        plt.tight_layout()
        fig.savefig(good_dir / f"good_{i:03d}.jpg", dpi=150, bbox_inches="tight")
        plt.close()

    # Process Bad Images (Comprehensive view)
    for i, idx in enumerate(tqdm(bad_idxs, desc="Saving Bad")):
        sample, target = dataset[idx]
        img_tensor = sample.clone()
        if img_tensor.min() < 0:
            img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(3, 1, 1)
        img_array = (img_tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        gt_boxes = target.get("boxes", torch.tensor([]))
        gt_labels = target.get("labels", torch.tensor([]))
        aug_h, aug_w = target["size"].tolist()
        gt_boxes_abs = box_cxcywh_to_xyxy(gt_boxes) * torch.tensor([aug_w, aug_h, aug_w, aug_h], dtype=torch.float32)

        # Move GT to device for IoU calculation during plotting
        gt_boxes_abs_device = gt_boxes_abs.to(device)

        inputs = sample.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)

        res = postprocessors["bbox"](
            outputs,
            torch.tensor([target["size"].tolist()], device=device),
            torch.tensor([target["orig_size"].tolist()], device=device),
        )[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(pil_img)
        ax2.imshow(pil_img)

        # Draw GT
        for box, label in zip(gt_boxes_abs, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            ax1.add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=(0, 1, 0), facecolor="none")
            )
            c_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls_{label}"
            ax1.text(
                x1,
                y1 - 5,
                f"GT: {c_name}",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round", facecolor=(0, 1, 0), alpha=0.8),
            )
        ax1.set_title("Ground Truth", fontsize=12)
        ax1.axis("off")

        # Dynamically categorize and draw ALL predictions
        for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
            if score.item() < 0.5:
                continue

            ious, _ = box_iou(box.unsqueeze(0), gt_boxes_abs_device)
            max_iou_val = ious.max(dim=1)[0].item()
            best_gt_idx = ious.max(dim=1)[1].item()

            x1, y1, x2, y2 = box.tolist()

            if max_iou_val >= 0.5 and label.item() == gt_labels[best_gt_idx].item():
                # Correct prediction (Blue)
                color, text = (0.2, 0.6, 1.0), f"OK: {COCO_CLASSES[label]}"
            elif max_iou_val >= 0.5 and label.item() != gt_labels[best_gt_idx].item():
                # Misclassification (Red)
                color, text = (1, 0, 0), f"MIS: {COCO_CLASSES[label]}->{COCO_CLASSES[gt_labels[best_gt_idx].item()]}"
            elif max_iou_val < 0.1:
                # Hallucination (Orange)
                color, text = (1, 0.5, 0), f"HAL: {COCO_CLASSES[label]}"
            else:
                # Poor localization (Yellow)
                color, text = (1, 1, 0), f"LOC: {COCO_CLASSES[label]}"

            ax2.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor="none"))
            ax2.text(
                x1,
                y1 - 5,
                f"{text} ({score:.2f})",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

        ax2.set_title("All Predictions (Categorized)", fontsize=12)
        ax2.axis("off")

        plt.tight_layout()
        fig.savefig(bad_dir / f"bad_{i:03d}.jpg", dpi=150, bbox_inches="tight")
        plt.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
