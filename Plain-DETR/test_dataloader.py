#!/usr/bin/env python3
import sys

sys.path.insert(0, "/root/projects/zod")

import torch
from datasets.zod import build_zod


class Args:
    dataset_file = "zod"
    zod_path = "/root/zod-dataset/"
    zod_crop = "none"
    zod_image_size = 800


args = Args()
dataset_val = build_zod("val", args)

print(f"Dataset size: {len(dataset_val)}")

img, target = dataset_val[0]

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image min/max: {img.min():.3f} / {img.max():.3f}")

print(f"\nTarget keys: {target.keys()}")
print(f"Boxes shape: {target['boxes'].shape}")
print(f"Boxes dtype: {target['boxes'].dtype}")
print(f"Boxes sample:\n{target['boxes'][:3]}")

boxes = target["boxes"]
if len(boxes) > 0:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    print(f"\nBox format check (cx, cy, w, h):")
    print(f"  cx range: [{cx.min():.4f}, {cx.max():.4f}]")
    print(f"  cy range: [{cy.min():.4f}, {cy.max():.4f}]")
    print(f"  w range:  [{w.min():.4f}, {w.max():.4f}]")
    print(f"  h range:  [{h.min():.4f}, {h.max():.4f}]")
    all_normalized = (
        (cx >= 0).all()
        and (cx <= 1).all()
        and (cy >= 0).all()
        and (cy <= 1).all()
        and (w >= 0).all()
        and (w <= 1).all()
        and (h >= 0).all()
        and (h <= 1).all()
    )
    print(f"\nAll boxes normalized [0,1]: {all_normalized}")
    print(f"Labels: {target['labels']}")
