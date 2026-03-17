# Evaluation Module - Project Structure

This module contains experiments for training and evaluating object detection models on the ZOD (Zenseact Open Dataset).

## Project Structure

```
evaluation/src/
├── config.py                     # Default configuration
├── zod_dataset.py               # Dataset classes (ZODROIFar, ZODROIWide, etc.)
├── coco_evaluation.py          # COCO-format evaluation metrics
├── load_backbone.py            # Backbone loading utilities
├── utils.py                    # Utility functions
├── vision_transformer.py       # ViT models
│
├── baseline_commercial_coco/   # COCO-pretrained Faster R-CNN
│   ├── train.py
│   ├── zod_dataset.py
│   ├── README.md
│   └── output/                 # Model checkpoints and results
│
├── baseline_resnet_fasterrcnn/ # ResNet-50 + Faster R-CNN (DINO pretrained)
│   ├── fasterrcnn.py          # Training script
│   ├── eval.py                # Evaluation script
│   ├── config.py              # Experiment config
│   ├── README.md
│   └── output/                # Model checkpoints and results
│
├── detr/                       # DETR experiment
│   └── detr.py
│
└── visualization/             # Visualization utilities
    ├── plot_bboxes.py
    ├── visualize_attention.py
    └── visualization_wrapper.py
```

## Shared Files (in src/ root)

These files are shared across all experiments:

| File | Description |
|------|-------------|
| `config.py` | Default configuration (paths, epochs, batch_size, etc.) |
| `zod_dataset.py` | Dataset classes: ZODROIFar, ZODROIWide, ZODRescaled |
| `coco_evaluation.py` | COCO evaluation metrics |
| `load_backbone.py` | Backbone loading (ResNet, ViT) |
| `utils.py` | Utility functions |
| `vision_transformer.py` | Vision Transformer implementations |

## Experiments

### baseline_resnet_fasterrcnn

ResNet-50 backbone pretrained on ZOD using DINO + Faster R-CNN detection head.

**Results:** 33.2% AP (target: 21.9%)

**Usage:**
```bash
# Training
cd evaluation/src/baseline_resnet_fasterrcnn
python fasterrcnn.py

# Evaluation (set train = False in config.py first)
python fasterrcnn.py

# Or use eval.py
python eval.py
```

See `baseline_resnet_fasterrcnn/README.md` for details.

### baseline_commercial_coco

COCO-pretrained Faster R-CNN baseline.

**Usage:**
```bash
cd evaluation/src/baseline_commercial_coco
python train.py
```

See `baseline_commercial_coco/README.md` for details.

## Adding New Experiments

To add a new experiment (e.g., ViT + Faster R-CNN):

1. Create a new folder: `evaluation/src/baseline_vit_fasterrcnn/`
2. Copy `fasterrcnn.py` and modify the model architecture
3. Copy `eval.py` and update for the new experiment
4. Create `config.py` with experiment-specific overrides
5. Create `output/` folder for results
6. Run training/evaluation

## Dataset Classes

| Class | Description | ROI Crop |
|-------|-------------|----------|
| `ZODROIFar` | Far ROI crop | top=924, left=1284, 1280×384 |
| `ZODROIWide` | Wide ROI crop | top=428, left=4, 3840×1152 |
| `ZODRescaled` | No crop, just resize | None |

## Configuration

Each experiment can override shared config via its own `config.py`:

| Parameter | Description |
|-----------|-------------|
| `train` | True = train+eval, False = eval only |
| `output_path` | Path to save/load model checkpoint |
| `model_checkpoint_path` | Path to pretrained backbone |
| `datapath` | Path to ZOD dataset |
| `num_epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `num_workers` | DataLoader workers |
| `score_threshold` | Confidence threshold for predictions |

## Notes

- All experiments use ZOD validation set with 80/20 train/test split
- Random seed 42 for reproducibility
- ZOD normalization: mean=[0.326, 0.323, 0.330], std=[0.113, 0.112, 0.117]
