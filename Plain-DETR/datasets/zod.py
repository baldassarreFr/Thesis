# ------------------------------------------------------------------------
# Makes the ZOD dataset work with Plain-DETR
# Integrates with evaluation/src/zod_dataset.py
# ------------------------------------------------------------------------

import sys
from pathlib import Path

# Add the fssl-foundation evaluation path to sys.path
fssl_eval_path = (
    Path(__file__).parent.parent.parent / "fssl-foundation" / "evaluation" / "src"
)
if str(fssl_eval_path) not in sys.path:
    sys.path.insert(0, str(fssl_eval_path))

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image

# Import from existing zod_dataset.py
from zod_dataset import ZODROIFar, ZODROIWide, ZODRescaled
from torchvision.transforms import functional as F_transforms


class ZODDetection(Dataset):
    """ZOD dataset for object detection compatible with Plain-DETR."""

    def __init__(
        self,
        root,
        image_set="val",
        transform=None,
        crop_type="none",
        rescaled_size=(448, 800),
    ):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.crop_type = crop_type
        self.rescaled_size = rescaled_size

        target_h, target_w = self.rescaled_size
        zod_size = (target_w, target_h)

        if crop_type == "Far":
            self.dataset = ZODROIFar(
                dataset_root=root,
                type=image_set,
                transform=None,
                rescaled_size=zod_size,
            )
        elif crop_type == "Wide":
            self.dataset = ZODROIWide(
                dataset_root=root,
                type=image_set,
                transform=None,
                rescaled_size=zod_size,
            )
        else:
            self.dataset = ZODRescaled(
                dataset_root=root,
                type=image_set,
                transform=None,
                rescaled_size=zod_size,
            )

        self._indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        actual_idx = self._indices[idx]
        img, target = self.dataset[actual_idx]

        if isinstance(img, Image.Image):
            img = F_transforms.to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        # Resize image to target size - boxes are already resized by ZODRescaled
        img = F_transforms.resize(img, self.rescaled_size)

        h_resized, w_resized = img.shape[1], img.shape[2]
        target["orig_size"] = torch.as_tensor(
            [2168, 3848]
        )  # Original ZOD frame size (H, W)
        target["size"] = torch.as_tensor([h_resized, w_resized])  # Resized dimensions
        target["image_id"] = torch.as_tensor([int(target.get("image_id", idx))])

        boxes = target["boxes"].clone()
        labels = target["labels"].clone()

        # === ROBUST BOX SANITIZATION (vectorized) ===
        exclude_keys = ["image_id", "orig_size", "size"]
        per_box_tensors = {}

        # Safely identify per-box tensors (ignoring metadata)
        for key, value in target.items():
            if (
                key not in exclude_keys
                and isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.shape[0] == boxes.shape[0]
            ):
                per_box_tensors[key] = value.clone()

        # Step 1: Clip coordinates to valid range
        x_min = boxes[:, 0].clamp_(min=0.0, max=w_resized)
        y_min = boxes[:, 1].clamp_(min=0.0, max=h_resized)
        x_max = boxes[:, 2].clamp_(min=0.0, max=w_resized)
        y_max = boxes[:, 3].clamp_(min=0.0, max=h_resized)

        # Step 2: Enforce validity (x_max >= x_min, y_max >= y_min)
        x_max = torch.maximum(x_max, x_min)
        y_max = torch.maximum(y_max, y_min)

        # Step 3: Filter degenerate boxes
        area = (x_max - x_min) * (y_max - y_min)
        valid_mask = area > 1e-3

        # Apply filtering to ALL per-box tensors EXCEPT the boxes we just mathematically fixed
        for key, tensor in per_box_tensors.items():
            if key == "boxes":
                continue
            target[key] = tensor[valid_mask]

        # Explicitly assign the corrected boxes
        target["boxes"] = torch.stack([x_min, y_min, x_max, y_max], dim=-1)[valid_mask]
        # === END SANITIZATION ===

        # Get sanitized boxes for conversion
        boxes = target["boxes"]

        # ZODRescaled already returns boxes in RESIZED coordinates (e.g., 800x448)
        # Just normalize to [0,1] by dividing by resized dimensions
        # Convert from [x_min, y_min, x_max, y_max] to [cx, cy, w, h] normalized [0,1]
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        cx = ((x_min + x_max) / 2) / w_resized  # center x normalized
        cy = ((y_min + y_max) / 2) / h_resized  # center y normalized
        box_w = (x_max - x_min) / w_resized  # width normalized
        box_h = (y_max - y_min) / h_resized  # height normalized

        # Stack back to [cx, cy, w, h] format
        boxes = torch.stack([cx, cy, box_w, box_h], dim=-1)

        target["boxes"] = boxes
        # Convert from 1-indexed to 0-indexed (DETR expects 0=background)
        # Use filtered labels from sanitization, not original labels
        target["labels"] = target["labels"] - 1

        return img, target

    def get_original_image(self, image_id):
        """Load the original high-resolution image from ZOD dataset.

        Args:
            image_id: The image ID (frame_id)

        Returns:
            PIL Image: Original high-res image (3848x2168)
        """
        # Import ZodFrames and Anonymization from the zod package (pip installed)
        # This avoids circular import issues with our local zod.py
        import zod.constants as zod_constants

        actual_idx = (
            self._indices[image_id] if image_id < len(self._indices) else image_id
        )
        frame = self.dataset.frames[actual_idx]
        img_array = frame.info.get_key_camera_frame(
            zod_constants.Anonymization.BLUR
        ).read()
        return Image.fromarray(img_array)


def build_zod(image_set, args):
    """Build ZOD dataset using original ZOD train/val splits.

    Image sets:
        - 'train': Full training split (89,972 frames from ZOD train)
        - 'val': Full validation split (10,023 frames from ZOD val)
        - 'val_finetune': 80% of validation split (~8,018 frames) for fine-tuning
        - 'val_test': 20% of validation split (~2,005 frames) for evaluation

    The 80/20 split uses seed=42 for reproducibility (matching baseline implementation).
    """
    import torch.utils.data

    root = Path(args.zod_path)
    assert root.exists(), f"provided ZOD path {root} does not exist"

    crop_type = getattr(args, "zod_crop", "none")
    # torchvision.resize expects (height, width), so swap: (height, width)
    rescaled_size = (
        getattr(args, "zod_image_height", args.zod_image_size),
        args.zod_image_size,
    )

    # Handle 80/20 split of validation set
    if image_set in ("val_finetune", "val_test"):
        # Load full validation set
        full_val_dataset = ZODDetection(
            root=str(root),
            image_set="val",
            transform=None,
            crop_type=crop_type,
            rescaled_size=rescaled_size,
        )

        # Split into 80% train (finetune) / 20% test
        # Using seed=42 for reproducibility (matching baseline implementation)
        val_size = len(full_val_dataset)
        train_size = int(0.8 * val_size)
        test_size = val_size - train_size

        generator = torch.Generator().manual_seed(42)
        finetune_dataset, test_dataset = torch.utils.data.random_split(
            full_val_dataset, [train_size, test_size], generator=generator
        )

        if image_set == "val_finetune":
            return finetune_dataset
        else:
            return test_dataset

    # Standard splits: 'train' or 'val'
    dataset = ZODDetection(
        root=str(root),
        image_set=image_set,
        transform=None,  # no additional transforms
        crop_type=crop_type,
        rescaled_size=rescaled_size,
    )

    return dataset


def get_coco_api_from_dataset(dataset):
    """ZOD uses a custom evaluator, GT annotations collected during evaluation."""
    return None
