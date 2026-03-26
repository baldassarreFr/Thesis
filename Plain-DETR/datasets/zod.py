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
        target["labels"] = (
            labels - 1
        )  # Convert from 1-indexed to 0-indexed (DETR expects 0=background)

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
    """Build ZOD dataset using original ZOD train/val splits."""
    root = Path(args.zod_path)
    assert root.exists(), f"provided ZOD path {root} does not exist"

    crop_type = getattr(args, "zod_crop", "none")
    # torchvision.resize expects (height, width), so swap: (height, width)
    rescaled_size = (
        getattr(args, "zod_image_height", args.zod_image_size),
        args.zod_image_size,
    )

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
