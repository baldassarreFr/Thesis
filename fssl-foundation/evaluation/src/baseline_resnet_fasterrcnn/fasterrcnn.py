import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import config

from load_backbone import *
from zod_dataset import *
from coco_evaluation import *


torch.manual_seed(42)


def create_fasterrcnn(checkpoint_path, num_classes=4):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Load from ZOD pretrained checkpoint
        resnet_model = load_trained_resnet(checkpoint_path)
    else:
        # Use ImageNet pretrained ResNet50
        print("Using ImageNet pretrained ResNet50")
        resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Define which layers from the ResNet model to use as output for the FPN
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

    in_channels_list = [
        256,
        512,
        1024,
        2048,
    ]  # ResNet layers [layer1, layer2, layer3, layer4]

    del resnet_model.fc

    fpn_backbone = BackboneWithFPN(
        resnet_model, return_layers, in_channels_list, out_channels=256
    )

    model = FasterRCNN(backbone=fpn_backbone, num_classes=num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, train_loader, output_path="faster_rcnn.pth", num_epochs=20):
    """trains a given faster r-cnn model for *num_epochs* and saves it to *output_path*"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("running on ", device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)

    model.train()

    loss_per_epoch = []
    global_progress = tqdm(range(0, num_epochs), desc=f"Training")
    for epoch in global_progress:
        epoch_loss = 0

        local_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in local_progress:
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()

            optimizer.step()
        try:
            print(f"loss for epoch {epoch + 1}: {epoch_loss / len(train_loader)}")
            loss_per_epoch.append(epoch_loss / len(train_loader))
        except Exception as e:
            print(e)

    torch.save(model.state_dict(), output_path)


def eval(model, test_dataset, score_threshold=0.5):
    predictions_to_coco(test_dataset, model, score_threshold)


if __name__ == "__main__":
    # Create model with DINO pretrained backbone (from config)
    model = create_fasterrcnn(config.model_checkpoint_path, num_classes=4)
    # model = create_fasterrcnn("models/resnet50_360_frames_sequences.pth", num_classes=4)
    ## random weights
    # model = create_fasterrcnn(None, num_classes=4)

    ## faster r-cnn with dino vit backbone
    # vit = load_trained_vit(dino_checkpoint_path="models/vit_small_200_frames_sequences.pth")
    # backbone = ViTFeatureExtractor(vit)
    # model = create_fasterrcnn(backbone)

    transform = T.Compose(
        [
            T.Resize((1000, 1200)),  # height, width - full size for training
            T.ToTensor(),
            T.Normalize(
                mean=[0.326, 0.323, 0.330],  # ZOD normalization
                std=[0.113, 0.112, 0.117],
            ),
        ]
    )

    dataset = ZODROIFar(
        dataset_root=config.datapath,
        type="val",
        transform=transform,
        rescaled_size=(1200, 1000),
    )
    # subset_indices = list(range(50_000))
    # train_dataset = Subset(train_dataset, subset_indices)

    # val dataset
    # test_dataset = ZODROIWide(dataset_root=config.datapath, type="val", transform=transform, rescaled_size=(1200, 1000))

    # train/test split (80/20 with fixed seed for reproducibility)
    random.seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Quick test subsets (configured in config.py)
    if config.quick_train_subset:
        train_dataset = Subset(train_dataset, list(range(config.quick_train_subset)))
        print(f"QUICK TEST: Using {config.quick_train_subset} training samples")
    if config.quick_test_subset:
        test_dataset = Subset(test_dataset, list(range(config.quick_test_subset)))
        print(f"QUICK TEST: Using {config.quick_test_subset} test samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    if config.train:  # train + evaluate
        print("starting training")
        train(model, train_loader, config.output_path, config.num_epochs)

        print("starting evaluation")
        eval(model, test_dataset, config.score_threshold)

    else:  # only evaluate, load trained model (use output_path file name)
        print("starting evaluation")
        state_dict = torch.load(config.output_path)
        model.load_state_dict(state_dict)
        eval(model, test_dataset, config.score_threshold)
