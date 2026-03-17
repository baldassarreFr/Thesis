import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from zod_dataset import *


class ZODUnscaled(Dataset):
    """Prepares original, unscaled ZOD dataset for object detection. Only 'val' frames are used."""

    def __init__(
        self, dataset_root="/root/zod-dataset/", version="full", transform=None
    ):
        self.transform = transform
        self.class_mapping = {
            "Vehicle": 0,
            "VulnerableVehicle": 1,
            "Pedestrian": 2,
            "Animal": 3,
        }
        zod_frames = ZodFrames(dataset_root, version)

        # consider only val frames
        val_indices = zod_frames.get_split(constants.VAL)
        self.frames = [zod_frames[idx] for idx in val_indices]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = frame.info.get_key_camera_frame(
            Anonymization.BLUR
        ).read()  # change it to tensor
        if self.transform:
            img = self.transform(img)
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        bboxes = [
            annotation.box2d.xyxy
            for annotation in annotations
            if annotation.name in self.class_mapping
        ]

        labels = [
            self.class_mapping[annotation.name]
            for annotation in annotations
            if annotation.name in self.class_mapping
        ]

        # handle empty annotations
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def plot_image_with_bboxes(image, bboxes, output_path="bbox_plot.png"):
    """Plots a ZOD image with bounding boxes."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.tolist()
        width = x_max - x_min
        height = y_max - y_min

        rect = Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title("Original image with Bounding Boxes")
    plt.show()
    plt.savefig(output_path)


def generate_bbox_plots(image_index):
    """plots original + scaled plots with bboxes"""

    # plot original image
    dataset = ZODUnscaled()
    img, target = dataset[image_index]
    bboxes = target["boxes"]
    plot_image_with_bboxes(img, bboxes, f"bbox_original_{image_index}.png")

    # use cropped/rescaled images
    dataset = ZODObjectDetection("/root/zod-dataset/")
    img, target = dataset[image_index]
    bboxes = target["boxes"]
    plot_image_with_bboxes(img, bboxes, f"bbox_scaled_{image_index}.png")


if __name__ == "__main__":
    index = 1
    generate_bbox_plots(index)
