#!/usr/bin/env python3
"""
This script visualizes the exact PyTorch tensor transformations 
that will be used in the Plain-DETR DataLoader for ZOD frames.
"""

import json
import random
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ZOD_ROOT = Path("/root/zod-dataset")
SINGLE_FRAMES = ZOD_ROOT / "single_frames"
OUTPUT_DIR = Path("/root/Plain-DETR-v2/visualizations/ZOD/ZOD_transformed_frames")
NUM_SAMPLES = 10

# Crop parameters
CROP_TOP = 400          # remove sky
CROP_BOTTOM = 568       # remove ego-vehicle    
RANDOM_CROP_SIZE = 800

VISIBILITY_THRESHOLD = 0.30     # if after the crop an object retains less than 30% of its original area, we discard it


CLASS_COLORS = {0: "blue", 1: "green", 2: "orange"}
ID_TO_NAME = {0: "Pedestrian", 1: "Vehicle", 2: "VulnerableVehicle"}


# ==============================================================================
# 1. THE CORE FUNCTION (To be copied into datasets/zod.py)
# ==============================================================================
def prepare_frame(img_tv, boxes_tv, labels_tensor):
    """   
    Args:
        img_tv: tv_tensors.Image [C, H, W]
        boxes_tv: tv_tensors.BoundingBoxes [N, 4] in absolute XYXY format
        labels_tensor: torch.Tensor [N]
        
    Returns:
        img_cropped: Transformed image tensor
        final_boxes: Transformed bounding boxes tensor
        final_labels: Transformed labels tensor
    """
    
    # Calculate original areas of the bounding boxes BEFORE any crop
    # Area = (x2 - x1) * (y2 - y1)
    orig_areas = (boxes_tv[:, 2] - boxes_tv[:, 0]) * (boxes_tv[:, 3] - boxes_tv[:, 1])


    # FIXED CROP (Remove ego-vehicle and sky)
    valid_h = img_tv.shape[-2] - CROP_TOP - CROP_BOTTOM
    valid_w = img_tv.shape[-1]
    
    img_fixed = v2.functional.crop(img_tv, top=CROP_TOP, left=0, height=valid_h, width=valid_w)
    boxes_fixed = v2.functional.crop(boxes_tv, top=CROP_TOP, left=0, height=valid_h, width=valid_w)


    # RANDOM CROP 800x800
    top, left, h, w = v2.RandomCrop.get_params(img_fixed, (RANDOM_CROP_SIZE, RANDOM_CROP_SIZE))

    img_cropped = v2.functional.crop(img_fixed, top, left, h, w)
    boxes_cropped = v2.functional.crop(boxes_fixed, top, left, h, w)
    
    # BOX FILTERING (based on visibility ratio)
    crop_w = (boxes_cropped[:, 2] - boxes_cropped[:, 0]).clamp(min=0)
    crop_h = (boxes_cropped[:, 3] - boxes_cropped[:, 1]).clamp(min=0)
    crop_areas = crop_w * crop_h
    
    # Keep objects that retain at least 30% of their original area
    # safety check (both dimensions > 15px)
    valid_mask = (orig_areas > 0) & ((crop_areas / orig_areas) >= VISIBILITY_THRESHOLD) & (crop_w > 15) & (crop_h > 15)
    
    final_boxes = boxes_cropped[valid_mask]
    final_labels = labels_tensor[valid_mask]


    # FORMAT CONVERSION
    # [X1, Y1, X2, Y2] absolute pixel values -> [CX, CY, W, H] normalized
    # final_boxes currently contains [X1, Y1, X2, Y2] absolute pixel values (0 to 800)
    
    # 1. Unbind the tensor into 4 separate columns
    x1, y1, x2, y2 = final_boxes.unbind(dim=-1)
    
    # 2. Calculate center coordinates, width, and height
    b_cx = (x1 + x2) / 2.0
    b_cy = (y1 + y2) / 2.0
    b_w = x2 - x1
    b_h = y2 - y1
    
    # 3. Stack them back together and normalize by dividing by the crop size (w, h from Step 2)
    # This guarantees all values are strictly floats between 0.0 and 1.0
    final_boxes_detr = torch.stack([b_cx / w, b_cy / h, b_w / w, b_h / h], dim=-1)
    

    debug_dict = {
        "img_fixed": img_fixed,
        "boxes_fixed": boxes_fixed,
        "crop_top": top,
        "crop_left": left,
        "final_boxes_xyxy": final_boxes # We pass XYXY for easy plotting
    }


    # Return the DETR-ready tensor
    return img_cropped, final_boxes_detr, final_labels, debug_dict

# ==============================================================================
# 2. SHELL HELPERS (Data Loading & Visualization)
# ==============================================================================
def load_annotations(frame_id: str) -> list:
    """Load object detection annotations from ZOD JSON."""
    anno_path = SINGLE_FRAMES / frame_id / "annotations" / "object_detection.json"
    if not anno_path.exists():
        return []

    with open(anno_path) as f:
        annotations = json.load(f)

    valid_classes = {"Pedestrian": 0, "Vehicle": 1, "VulnerableVehicle": 2}
    boxes = []

    for anno in annotations:
        class_name = anno.get("properties", {}).get("class", "")
        if class_name not in valid_classes:
            continue

        coords = anno.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue

        xs = [p[0] for p in coords]     # extract all x-coordinates
        ys = [p[1] for p in coords]     # extract all y-coordinates
        
        # Create a bounding box from the coordinates in the format [x_min, y_min, x_max, y_max]
        boxes.append({
            "xyxy": [min(xs), min(ys), max(xs), max(ys)],
            "class_id": valid_classes[class_name],
            "class_name": class_name,
        })

    return boxes



def load_image(frame_id: str) -> Image.Image:
    """Load the front blur image for a given ZOD frame."""
    img_dir = SINGLE_FRAMES / frame_id / "camera_front_blur"
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        raise FileNotFoundError(f"No image found in {img_dir}")
    return Image.open(img_files[0])


def draw_boxes_on_ax(ax, boxes_tensor, labels_tensor):
    """Draws boxes (XYXY format) on a matplotlib axis."""
    for box, label_id in zip(boxes_tensor.tolist(), labels_tensor.tolist()):
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(label_id, "red")
        class_name = ID_TO_NAME.get(label_id, "Unknown")
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, class_name, fontsize=8, color=color, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))


# ==============================================================================
# 3. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = sorted([f.name for f in SINGLE_FRAMES.iterdir() if f.is_dir()])
    
    if not all_frames:
        print("No frames found. Please check ZOD_ROOT path.")
        return

    selected_frames = random.sample(all_frames, min(NUM_SAMPLES, len(all_frames)))
    print(f"Processing {len(selected_frames)} frames for visualization...")

    for frame_id in selected_frames:
        # PREPARATION
        pil_img = load_image(frame_id)
        boxes_dict = load_annotations(frame_id)
        
        if not boxes_dict:
            continue

        img_width, img_height = pil_img.size
        
        # Convert PIL to tv_tensors.Image
        img_tv = tv_tensors.Image(v2.ToImage()(pil_img))
        
        # Convert list of dicts to tensors 
        boxes_xyxy = torch.tensor([b["xyxy"] for b in boxes_dict], dtype=torch.float32)
        labels = torch.tensor([b["class_id"] for b in boxes_dict], dtype=torch.int64)       
        boxes_tv = tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=(img_height, img_width))


        # Prepare the frame for visualization
        img_cropped, final_boxes_detr, final_labels, debug = prepare_frame(img_tv, boxes_tv, labels)



        # --------
        # 2x2 VISUALIZATION GRID
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Top-Left: Original Image (No boxes)
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title(f"1. Original\n{img_width}x{img_height}", fontsize=10)
        axes[0, 0].axis("off")

        # 2. Top-Right: Original Image + GT
        axes[0, 1].imshow(pil_img)
        axes[0, 1].set_title(f"2. Original + GT\n{len(boxes_tv)} boxes", fontsize=10)
        draw_boxes_on_ax(axes[0, 1], boxes_tv, labels)
        axes[0, 1].axis("off")

        # 3. Bottom-Left: After Fixed Crop
        img_fixed_pil = v2.ToPILImage()(debug["img_fixed"])
        axes[1, 0].imshow(img_fixed_pil)
        axes[1, 0].set_title(f"3. Fixed Crops (Top: {CROP_TOP}, Bot: {CROP_BOTTOM})\n{img_fixed_pil.size[0]}x{img_fixed_pil.size[1]}", fontsize=10)
        draw_boxes_on_ax(axes[1, 0], debug["boxes_fixed"], labels) # Note: labels length matches boxes_fixed before filtering
        axes[1, 0].axis("off")

        # 4. Bottom-Right: Final Random Crop (Using XYXY boxes from debug for plotting)
        img_cropped_pil = v2.ToPILImage()(img_cropped)
        axes[1, 1].imshow(img_cropped_pil)
        axes[1, 1].set_title(f"4. Random 800x800 Crop\ncrop at ({debug['crop_left']}, {debug['crop_top']}) - {len(debug['final_boxes_xyxy'])} boxes kept", fontsize=10)
        draw_boxes_on_ax(axes[1, 1], debug["final_boxes_xyxy"], final_labels)
        axes[1, 1].axis("off")

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{frame_id}_grid.jpg"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved: {save_path.name}")

if __name__ == "__main__":
    main()