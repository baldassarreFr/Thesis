from main_dino import DataAugmentationDINO
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import utils
import json
import os
import time
import datetime
from pathlib import Path
import math
import sys
import config

from zod_dataset import ZOD


def plot_augmentations(dataset, index):
    img = dataset[index]
    
    # Number of images to plot
    num_images = len(img)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes] 
    
    titles = [f"Global Crop {i+1}" for i in range(min(2, num_images))]
    titles.extend([f"Local Crop {i-1}" for i in range(2, num_images)])
    
    for i, curr_img in enumerate(img):
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        
        # Undo normalization
        img_denorm = inv_normalize(curr_img)
        
        img_reshaped = img_denorm.permute(1, 2, 0).numpy()
        img_reshaped = (img_reshaped - img_reshaped.min()) / (img_reshaped.max() - img_reshaped.min())  # Rescale to [0, 1]
        
        axes[i].imshow(img_reshaped)
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("augmentations.png")


if __name__=="__main__":
    image_index = 1

    transform = DataAugmentationDINO(
        config.global_crops_scale,
        config.local_crops_scale,
        config.local_crops_number
    )

    dataset = ZOD("/mnt/tier2/project/p201222/data/ZOD_clone_2018_scaleout_zenseact", transform=transform)
    # dataset = datasets.ImageFolder("../../../tiny-imagenet-200/train", transform=transform)
    # dataset = datasets.ImageFolder(config.datapath, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    plot_augmentations(dataset, image_index)    