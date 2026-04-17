# ------------------------------------------------------------------------
# Plain-DETR
# ZOD Dataset for object detection
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import json
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

if TYPE_CHECKING:
    from plain_detr.main import Config


# ==============================================================================
# CONFIGURATION
# ==============================================================================
ZOD_CLASS_TO_ID = {
    "Pedestrian": 0,
    "Vehicle": 1,
    "VulnerableVehicle": 2,
}
ID_TO_ZOD_CLASS = {v: k for k, v in ZOD_CLASS_TO_ID.items()}

# Crop parameters
CROP_TOP = 400  # remove sky
CROP_BOTTOM = 568  # remove ego-vehicle
RANDOM_CROP_SIZE = 800
VISIBILITY_THRESHOLD = 0.30  # if after the crop an object retains less than 30% of its original area, we discard it


# ==============================================================================
# ZOD Dataset Class
# ==============================================================================
class ZODDetection(torch.utils.data.Dataset):
    """ZOD dataset for object detection."""

    def __init__(
        self,
        img_folder: Path,
        ann_folder: Path,
        split: str = "train",
        transforms=None,
    ):
        """
        Args:
            img_folder: Path to ZOD single_frames directory
            ann_folder: Path to ZOD annotations folder
            split: "train" or "val"
            transforms: Optional transforms to apply after our custom preprocessing
        """
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self.split = split
        self._transforms = transforms

        # Load split from ZOD's trainval JSON
        trainval_path = self.ann_folder.parent / "trainval-frames-full.json"
        with open(trainval_path) as f:
            trainval_data = json.load(f)

        # Get frame IDs for this split (trainval_data[split] is a list of dicts with 'id' field)
        split_list = trainval_data.get(self.split, [])
        if isinstance(split_list, list):
            split_ids = {item["id"] for item in split_list if isinstance(item, dict) and "id" in item}
        else:
            split_ids = set()

        # Filter to only existing frames
        self.frame_ids = sorted([fid for fid in split_ids if (self.img_folder / fid / "camera_front_blur").exists()])

        print(f"Loaded ZOD {split}: {len(self.frame_ids)} frames")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]

        # Load image
        img_dir = self.img_folder / frame_id / "camera_front_blur"
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not img_files:
            raise FileNotFoundError(f"No image found in {img_dir}")

        pil_img = Image.open(img_files[0])
        img_width, img_height = pil_img.size

        # Load annotations
        anno_path = self.img_folder / frame_id / "annotations" / "object_detection.json"
        with open(anno_path) as f:
            annotations = json.load(f)

        # Parse boxes - filter for dynamic objects only
        boxes_xyxy = []
        labels = []

        for anno in annotations:
            class_name = anno.get("properties", {}).get("class", "")
            if class_name not in ZOD_CLASS_TO_ID:
                continue

            coords = anno.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue

            xs = [p[0] for p in coords]  # extract all x-coordinates
            ys = [p[1] for p in coords]  # extract all y-coordinates

            # Create a bounding box from the coordinates in the format [x_min, y_min, x_max, y_max]
            boxes_xyxy.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(ZOD_CLASS_TO_ID[class_name])

        if not boxes_xyxy:
            # Return empty tensors if no valid boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Convert to tv_tensors
        img_tv = tv_tensors.Image(v2.ToImage()(pil_img))
        boxes_tv = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img_height, img_width))

        # Apply preprocessing (crop + visibility filter + normalization)
        is_train = self.split == "train"

        # Calculate dimensions BEFORE crop for orig_size (fixed crop removes top+CROP_BOTTOM)
        valid_h = pil_img.size[1] - CROP_TOP - CROP_BOTTOM
        valid_w = pil_img.size[0]

        img_processed, boxes_processed, labels_processed = preprocess_zod_frame(img_tv, boxes_tv, labels, is_train)

        # Build target dict (same format as COCO)
        target = {
            "boxes": boxes_processed,  # CXCYWH normalized [N, 4]
            "labels": labels_processed,  # [N]
            "image_id": torch.tensor([int(frame_id)], dtype=torch.int64),
            "orig_size": torch.tensor([valid_h, valid_w]),
            "size": torch.tensor([img_processed.shape[-2], img_processed.shape[-1]]),
        }

        # Apply additional transforms if provided (e.g., normalize)
        if self._transforms is not None:
            img_processed, target = self._transforms(img_processed, target)

        return img_processed, target


def preprocess_zod_frame(img_tv, boxes_tv, labels_tensor, is_train=True):
    """
    Preprocess ZOD frame: crops, filtering, normalization.

    Args:
        img_tv: tv_tensors.Image [C, H, W]
        boxes_tv: tv_tensors.BoundingBoxes [N, 4] in absolute XYXY format
        labels_tensor: torch.Tensor [N]
        is_train: If True, apply random crop + visibility filter. If False, apply resize only.

    Returns:
        img_final: Transformed image tensor
        final_boxes: CXCYWH normalized boxes [N, 4]
        final_labels: Filtered labels [N]
    """
    # Calculate original areas BEFORE any crop
    orig_areas = (boxes_tv[:, 2] - boxes_tv[:, 0]).clamp(min=0) * (boxes_tv[:, 3] - boxes_tv[:, 1]).clamp(min=0)

    # FIXED CROP (Remove ego-vehicle and sky)
    valid_h = img_tv.shape[-2] - CROP_TOP - CROP_BOTTOM
    valid_w = img_tv.shape[-1]

    img_fixed = v2.functional.crop(img_tv, top=CROP_TOP, left=0, height=valid_h, width=valid_w)
    boxes_fixed = v2.functional.crop(boxes_tv, top=CROP_TOP, left=0, height=valid_h, width=valid_w)

    if is_train:
        # TRAIN: RANDOM CROP 800x800
        top, left, h, w = v2.RandomCrop.get_params(img_fixed, (RANDOM_CROP_SIZE, RANDOM_CROP_SIZE))
        img_final = v2.functional.crop(img_fixed, top, left, h, w)
        boxes_final = v2.functional.crop(boxes_fixed, top, left, h, w)

        h_final, w_final = img_final.shape[-2], img_final.shape[-1]

        # FIX: Manually clamp the box coordinates to the new image boundaries
        # This ensures that coordinates outside the crop become exactly 0 or max width/height
        boxes_clamped = boxes_final.clone()
        boxes_clamped[:, 0::2] = boxes_clamped[:, 0::2].clamp(min=0, max=w_final)
        boxes_clamped[:, 1::2] = boxes_clamped[:, 1::2].clamp(min=0, max=h_final)

        # BOX FILTERING: Calculate actual visible area using the clamped coordinates
        crop_w = boxes_clamped[:, 2] - boxes_clamped[:, 0]
        crop_h = boxes_clamped[:, 3] - boxes_clamped[:, 1]
        crop_areas = crop_w * crop_h
        
        valid_mask = (
            (orig_areas > 0) & 
            ((crop_areas / orig_areas) >= VISIBILITY_THRESHOLD) & 
            (crop_w > 15) & 
            (crop_h > 15)
        )

        # CRITICAL: Overwrite the boxes with the clamped ones 
        # so the model never sees negative or out-of-bounds coordinates
        boxes_final = boxes_clamped

    else:
        # EVALUATION: Proportional Resize (max_size=2000, size=1200)
        resize_transform = v2.Resize(size=1200, max_size=2000, antialias=True)
        img_final = resize_transform(img_fixed)
        boxes_final = resize_transform(boxes_fixed)
        
        # SAFETY CLAMP FOR EVALUATION: Ensure resized boxes don't exceed image dimensions due to float rounding
        h_final, w_final = img_final.shape[-2], img_final.shape[-1]
        boxes_final[:, 0::2] = boxes_final[:, 0::2].clamp(min=0, max=w_final)
        boxes_final[:, 1::2] = boxes_final[:, 1::2].clamp(min=0, max=h_final)
        
        valid_mask = orig_areas > 0

    final_boxes_xyxy = boxes_final[valid_mask]
    final_labels = labels_tensor[valid_mask]

    h_final, w_final = img_final.shape[-2], img_final.shape[-1]

    # FORMAT CONVERSION: XYXY -> CXCYWH normalized
    if len(final_boxes_xyxy) > 0:
        x1, y1, x2, y2 = final_boxes_xyxy.unbind(dim=-1)
        b_cx = (x1 + x2) / 2.0
        b_cy = (y1 + y2) / 2.0
        b_w = x2 - x1
        b_h = y2 - y1
        final_boxes_cxcywh = torch.stack(
            [b_cx / w_final, b_cy / h_final, b_w / w_final, b_h / h_final], dim=-1
        )  # normalization
    else:
        final_boxes_cxcywh = torch.empty((0, 4), dtype=torch.float32)

    return img_final, final_boxes_cxcywh, final_labels


# ==============================================================================
# Transforms (matching COCO style)
# ==============================================================================
def make_zod_transforms(image_set, args):
    """Create transforms for ZOD dataset."""
    # For ZOD, we don't need additional augments - preprocessing is in the dataset
    # But we can add normalize here if needed
    normalize = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return normalize


# ==============================================================================
# Build function (matching COCO API)
# ==============================================================================
def build(image_set, args: Config):
    """Build ZOD dataset."""
    root = args.data_dir / args.zod_path
    assert root.exists(), f"provided ZOD path {root} does not exist"

    # ZOD uses single_frames folder for images
    img_folder = root / "single_frames"
    ann_folder = root / "single_frames"  # annotations are in each frame folder

    dataset = ZODDetection(
        img_folder=img_folder,
        ann_folder=ann_folder,
        split=image_set,
        transforms=make_zod_transforms(image_set, args),
    )
    return dataset
