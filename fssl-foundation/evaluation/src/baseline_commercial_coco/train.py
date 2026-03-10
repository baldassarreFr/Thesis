#!/usr/bin/env python3
"""Training and evaluation script for COCO-pretrained Faster R-CNN on ZOD

This script trains a Faster R-CNN model with ResNet-50 FPN backbone
pretrained on COCO, and fine-tunes it on ZOD for object detection.

Usage:
    python train.py

Requirements:
    - ZOD dataset at /root/zod-dataset/ (default)
    - Dependencies: torch, torchvision, pycocotools, tqdm
"""

import os
import sys
import json
import argparse
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from zod_dataset import ZODRescaled

torch.manual_seed(42)

# Default configuration
NUM_EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.005
NUM_TRAIN = 1000
NUM_TEST = 300
IMG_SIZE = 640
DATASET_ROOT = "/root/zod-dataset/"


def get_default_output_dir():
    """Get output directory based on the script's parent folder name."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = os.path.basename(script_dir)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    return os.path.join(root_dir, "outputs", experiment_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on ZOD")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DATASET_ROOT,
        help=f"Path to ZOD dataset (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=NUM_TRAIN,
        help=f"Number of training samples (default: {NUM_TRAIN})",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=NUM_TEST,
        help=f"Number of test samples (default: {NUM_TEST})",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=IMG_SIZE,
        help=f"Image size (default: {IMG_SIZE})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-detect from folder name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or get_default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    transform = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.326, 0.323, 0.330], std=[0.113, 0.112, 0.117]),
        ]
    )

    print("Loading data...")
    dataset = ZODRescaled(
        dataset_root=args.dataset_root,
        version="full",
        transform=transform,
        rescaled_size=(args.img_size, args.img_size),
        type="val",
    )

    train_ds = Subset(dataset, list(range(args.num_train)))
    test_ds = Subset(
        dataset, list(range(args.num_train + 200, args.num_train + 200 + args.num_test))
    )

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Vehicle"},
            {"id": 2, "name": "VulnerableVehicle"},
            {"id": 3, "name": "Pedestrian"},
        ],
    }
    ann_id = 1
    for idx in range(len(test_ds)):
        img, target = test_ds[idx]
        h, w = img.shape[1], img.shape[2]
        gt["images"].append({"id": int(target["image_id"]), "width": w, "height": h})
        for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
            x1, y1, x2, y2 = box
            if x1 < -5 or y1 < -5 or x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            gt["annotations"].append(
                {
                    "id": ann_id,
                    "category_id": int(label),
                    "iscrowd": 0,
                    "image_id": int(target["image_id"]),
                    "area": float((x2 - x1) * (y2 - y1)),
                    "bbox": [
                        float(max(0, x1)),
                        float(max(0, y1)),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                }
            )
            ann_id += 1

    gt_file = os.path.join(output_dir, "gt_eval.json")
    with open(gt_file, "w") as f:
        json.dump(gt, f)

    print(f"\n=== COCO pretrained ===")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 4)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=lambda x: tuple(zip(*x)),
        shuffle=True,
        num_workers=2,
    )

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for imgs, tgts in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            imgs = [i.to(device) for i in imgs]
            tgts = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in tgts
            ]
            losses = sum(model(imgs, tgts).values())
            if not torch.isfinite(losses):
                print(f"Warning: non-finite loss at epoch {epoch + 1}, skipping batch")
                continue
            opt.zero_grad()
            losses.backward()
            opt.step()
            loss_sum += losses.item()
        print(f"Epoch {epoch + 1}: loss = {loss_sum / len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "coco_model.pth"))

    print("\nEvaluating...")
    model.eval()
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, collate_fn=lambda x: tuple(zip(*x))
    )

    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Inference"):
            imgs = [i.to(device) for i in imgs]
            out = model(imgs)
            for o in out:
                for box, label, score in zip(o["boxes"], o["labels"], o["scores"]):
                    if score < 0.3:
                        continue
                    x1, y1, x2, y2 = box
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    preds.append(
                        {
                            "image_id": 0,
                            "category_id": int(label),
                            "bbox": [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                            ],
                            "score": float(score),
                        }
                    )

    for i, (img, _) in enumerate(test_loader):
        img_id = int(test_ds[i * args.batch_size][1]["image_id"])
        for j in range(len(img)):
            if i * args.batch_size + j < len(preds):
                preds[i * args.batch_size + j]["image_id"] = img_id

    pred_file = os.path.join(output_dir, "coco_preds.json")
    with open(pred_file, "w") as f:
        json.dump(preds, f)

    coco = COCO(gt_file)
    dt = coco.loadRes(pred_file)
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    print("\n=== COCO Results ===")
    ev.summarize()

    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
