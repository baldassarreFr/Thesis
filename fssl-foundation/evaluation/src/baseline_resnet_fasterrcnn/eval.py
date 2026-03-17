import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms as T

import config
from coco_evaluation import *
from zod_dataset import *
from fasterrcnn import *

torch.manual_seed(42)


def plot_bboxes_target_predicted(dataset, predictions, index=0):
    img, target = dataset[index]
    img_permuted = img.permute(1, 2, 0)

    bboxes = target["boxes"]
    plot_image_with_bboxes(img_permuted, bboxes, f"target.png")

    bboxes = predictions[index]["boxes"]
    plot_image_with_bboxes(img_permuted, bboxes, f"predictions.png")


if __name__ == "__main__":
    # Load model with ZOD pretrained backbone
    model = create_fasterrcnn(config.model_checkpoint_path, num_classes=4)

    # Use same transforms as training (ZODROIFar expects (1000, 1200) - height, width)
    transform = T.Compose(
        [
            T.Resize((1000, 1200)),  # height, width - matches training
            T.ToTensor(),
            T.Normalize(
                mean=[0.326, 0.323, 0.330],  # ZOD normalization
                std=[0.113, 0.112, 0.117],
            ),
        ]
    )

    # Use ZODROIFar (matching PDF methodology)
    dataset = ZODROIFar(
        dataset_root=config.datapath,
        type="val",
        transform=transform,
        rescaled_size=(1200, 1000),
    )

    # Same 80/20 split with fixed seed
    import random

    random.seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Load trained weights from config.output_path
    print(f"Loading model from {config.output_path}")
    state_dict = torch.load(config.output_path)
    model.load_state_dict(state_dict)
    print("Model loaded successfully")

    # Run evaluation
    print("Evaluating model...")
    predictions = predictions_to_coco(test_dataset, model, config.score_threshold)

    # Plot first prediction
    print("Saving visualization...")
    plot_bboxes_target_predicted(test_dataset, predictions, index=0)

    print("Done! Check eval_results.txt for metrics.")
