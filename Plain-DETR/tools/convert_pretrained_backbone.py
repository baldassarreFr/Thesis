#!/usr/bin/env python3
"""
Convert pretrained backbone weights to Plain-DETR format.

Usage:
    python tools/convert_pretrained_backbone.py \
        --input /path/to/pretrained_checkpoint.pth \
        --output /path/to/converted_weights.pth \
        --backbone resnet50
"""

import argparse
import torch
from pathlib import Path


def convert_dino_resnet(checkpoint):
    """Convert DINO ResNet checkpoint to Plain-DETR format.

    Output format: keys compatible with model.backbone[0].body
    Example: conv1.weight, bn1.weight, layer1.0.conv1.weight, etc.
    """
    if "student" in checkpoint:
        state_dict = checkpoint["student"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        if key.startswith("backbone."):
            key = key[9:]

        if key.startswith("head.") or key == "num_batches_tracked":
            continue

        new_state_dict[key] = value

    return new_state_dict


def convert_dino_vit(checkpoint):
    """Convert DINO ViT checkpoint to Plain-DETR format."""
    if "student" in checkpoint:
        state_dict = checkpoint["student"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        if key.startswith("backbone."):
            key = key[9:]

        new_state_dict[key] = value

    return new_state_dict


def convert_standard_checkpoint(checkpoint, backbone_type):
    """Convert standard torchvision or custom checkpoint."""
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            key = key[9:]

        new_state_dict[key] = value

    return new_state_dict


def convert_checkpoint(input_path, output_path, backbone_type):
    """Main conversion function."""
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    backbone_type = backbone_type.lower()

    if "resnet" in backbone_type:
        converted = convert_dino_resnet(checkpoint)
    elif "vit" in backbone_type or "swin" in backbone_type:
        converted = convert_dino_vit(checkpoint)
    else:
        converted = convert_standard_checkpoint(checkpoint, backbone_type)

    print(f"Converted {len(converted)} keys")

    output_checkpoint = {
        "model": converted,
        "backbone_type": backbone_type,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output_checkpoint, output_path)
    print(f"Saved converted weights to: {output_path}")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert pretrained backbone weights")
    parser.add_argument("--input", required=True, help="Path to input checkpoint")
    parser.add_argument(
        "--output", required=True, help="Path to save converted weights"
    )
    parser.add_argument(
        "--backbone",
        required=True,
        help="Backbone type (e.g., resnet50, vit_small, swin_tiny)",
    )

    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, args.backbone)


if __name__ == "__main__":
    main()
