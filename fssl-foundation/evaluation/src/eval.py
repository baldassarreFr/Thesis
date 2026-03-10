import torch

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
    # Options for backbone:
    # 1. ZOD pretrained: path to your ZOD checkpoint
    # 2. ImageNet pretrained: None (uses torchvision pretrained ResNet50)
    checkpoint_path = (
        "/root/projects/fssl-foundation/ZODPretraining/src/outputs/checkpoint.pth"
    )

    # Use ZOD pretrained backbone (or None for ImageNet pretrained ResNet50)
    model = create_fasterrcnn(checkpoint_path, num_classes=4)

    transform = T.Compose(
        [
            T.Resize((1000, 700)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.326, 0.323, 0.330],  # ZOD normalization
                std=[0.113, 0.112, 0.117],
            ),
        ]
    )
    dataset = ZODRescaled(
        dataset_root="/home/jovyan/zod-full/",
        version="full",
        transform=transform,
        rescaled_size=(1000, 700),
        type="val",
    )

    # train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # For ZOD pretrained: load checkpoint
    # For ImageNet pretrained: model already has pretrained weights from create_fasterrcnn
    if checkpoint_path:
        print(f"Loading ZOD pretrained model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        # The checkpoint contains student/teacher keys, extract student
        if "student" in state_dict:
            state_dict = state_dict["student"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Using ImageNet pretrained ResNet50 (random detection head)")

    # eval
    print("Evaluating model")
    predictions_to_coco(test_dataset, model)

    # plot predicted bboxes
    # plot_bboxes_target_predicted(dataset, predictions, index=0)
