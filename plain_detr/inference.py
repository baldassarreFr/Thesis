#!/usr/bin/env python3
"""
Plain-DETR Inference Script - Visualize model predictions on COCO images
Usage: python plain_detr/inference.py
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from plain_detr.main import Config
from plain_detr.datasets import build_dataset
from plain_detr.models.detr import build as build_model
from plain_detr.util.box_ops import box_iou


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
    print("Plain-DETR Inference Script")
    print("=" * 60)

    checkpoint = Path("exps/dinov3_vit_small_boxrpe/checkpoint.epoch_11.pth")
    output_dir = Path("visualizations")
    num_images = 200
    device = "cpu"
    max_save = 10

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    # Load args
    print("\n1. Loading model config...")
    with open("exps/dinov3_vit_small_boxrpe/args.json") as f:
        args_dict = json.load(f)
    args_dict["device"] = device
    args = Config(**args_dict)

    # Build model
    print("2. Building model...")
    model, criterion, postprocessors = build_model(args)
    model.eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print("   Model loaded")

    # Build dataset
    print("3. Loading COCO val dataset...")
    dataset = build_dataset("val", args)
    print(f"   Dataset: {len(dataset)} images")

    # Select random images
    print("4. Running inference...")
    all_idx = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(all_idx)
    selected = all_idx[:num_images]

    # Output dirs
    good_dir = output_dir / "good"
    bad_dir = output_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    # Collect good first, then bad - use separate shuffles
    print("   Phase 1: Finding good examples (max_score >= 0.8)")
    good_idxs = []
    random.seed(42)
    all_idx = list(range(len(dataset)))
    random.shuffle(all_idx)

    for idx in all_idx:
        if len(good_idxs) >= max_save:
            break

        sample, target = dataset[idx]
        inputs = sample.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        orig_size = target["size"].tolist()  # augmented size for correct box coordinates
        target_sizes = torch.tensor([orig_size], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)

        if len(results) == 0 or len(results[0]["scores"]) == 0:
            continue

        res = results[0]
        max_score = res["scores"].max().item()

        if max_score >= 0.8:
            good_idxs.append(idx)

    print(f"   Found {len(good_idxs)} good images")

    print("   Phase 2: Finding bad examples (false positives)")
    bad_idxs = []
    random.seed(123)  # Different seed
    all_idx = list(range(len(dataset)))
    random.shuffle(all_idx)

    for idx in all_idx:
        if len(bad_idxs) >= max_save:
            break

        sample, target = dataset[idx]
        inputs = sample.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        orig_size = target["size"].tolist()  # augmented size for correct box coordinates
        target_sizes = torch.tensor([orig_size], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)

        if len(results) == 0 or len(results[0]["scores"]) == 0:
            continue

        res = results[0]
        scores = res["scores"]
        labels = res["labels"]
        pred_boxes = res["boxes"]

        gt_boxes = target.get("boxes", torch.tensor([]))

        has_fp = False
        if len(gt_boxes) > 0:
            for i, (score, box, label) in enumerate(zip(scores, pred_boxes, labels)):
                if 0.1 <= score.item() <= 0.5:
                    ious, _ = box_iou(box.unsqueeze(0), gt_boxes)
                    max_iou = ious.max().item()
                    if max_iou < 0.3:
                        has_fp = True
                        break

        if has_fp or (scores.max().item() >= 0.1 and len(gt_boxes) == 0):
            bad_idxs.append(idx)

    print(f"   Found {len(bad_idxs)} bad images")

    print("   Phase 3: Generating visualizations...")

    # Generate good visualizations
    for i, idx in enumerate(good_idxs):
        sample, target = dataset[idx]
        inputs = sample.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        orig_size = target["size"].tolist()  # augmented size for correct box coordinates
        target_sizes = torch.tensor([orig_size], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)
        res = results[0]

        # Prepare image
        img_tensor = sample.clone()
        if img_tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
        img_tensor = img_tensor.clamp(0, 1)
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Draw boxes with score >= 0.5
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(pil_img)

        boxes = res["boxes"]
        labels = res["labels"]
        scores = res["scores"]

        for box, label, score in zip(boxes, labels, scores):
            if score.item() < 0.5:
                continue
            x1, y1, x2, y2 = box.tolist()
            color = COLORS[label.item() % len(COLORS)]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls_{label}"
            ax.text(
                x1,
                y1 - 5,
                f"{class_name}: {score:.2f}",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

        ax.axis("off")
        plt.tight_layout()
        fig.savefig(good_dir / f"good_{i:03d}.jpg", dpi=150, bbox_inches="tight", facecolor="black")
        plt.close()

    # Generate bad visualizations
    for i, idx in enumerate(bad_idxs):
        sample, target = dataset[idx]
        inputs = sample.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        orig_size = target["size"].tolist()  # augmented size for correct box coordinates
        target_sizes = torch.tensor([orig_size], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)
        res = results[0]

        # Prepare image
        img_tensor = sample.clone()
        if img_tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
        img_tensor = img_tensor.clamp(0, 1)
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Draw all boxes with score >= 0.1 (show false positives)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(pil_img)

        boxes = res["boxes"]
        labels = res["labels"]
        scores = res["scores"]

        for box, label, score in zip(boxes, labels, scores):
            if score.item() < 0.1:
                continue
            x1, y1, x2, y2 = box.tolist()
            color = COLORS[label.item() % len(COLORS)]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls_{label}"
            ax.text(
                x1,
                y1 - 5,
                f"{class_name}: {score:.2f}",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

        ax.axis("off")
        plt.tight_layout()
        fig.savefig(bad_dir / f"bad_{i:03d}.jpg", dpi=150, bbox_inches="tight", facecolor="black")
        plt.close()

    # Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"good": len(good_idxs), "bad": len(bad_idxs)}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done! Good: {len(good_idxs)}, Bad: {len(bad_idxs)}")
    print(f"Saved to: {output_dir}/good/ and {output_dir}/bad/")
    print("=" * 60)


if __name__ == "__main__":
    main()
