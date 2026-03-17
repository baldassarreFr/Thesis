# Baseline: ZOD-DINO ResNet-50 + Faster R-CNN

## Description
This experiment evaluates the ResNet-50 backbone pretrained on ZOD using DINO (teacher-student) method, combined with a Faster R-CNN detection head initialized from COCO pretrained weights.

The model is fine-tuned end-to-end on ZOD for object detection.

## Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 2 |
| Learning Rate | 0.003 |
| Image Size | 1000x1200 (ROI Far crop) |
| Train Samples | ~8,004 (80% of val set) |
| Test Samples | ~2,005 (20% of val set) |
| Dataset | /root/zod-dataset/ |
| DINO Checkpoint | ZODPretraining/src/outputs/checkpoint.pth |
| ROI Type | ZODROIFar |

## Usage

### Training + Evaluation
```bash
cd evaluation/src/baseline_resnet_fasterrcnn
python fasterrcnn.py
```

This will:
- Load ResNet-50 backbone trained on ZOD using DINO (student weights)
- Initialize detection head from COCO pretrained weights
- Fine-tune end-to-end on ZOD
- Save checkpoint to `output/fasterrcnn.pth`
- Run evaluation and save results to `output/eval_results.txt`

### Evaluation Only
There are two ways to evaluate a trained model:

**Option 1 - Using fasterrcnn.py:**
```bash
# Edit config.py: set train = False
python fasterrcnn.py
```

**Option 2 - Using eval.py (recommended):**
```bash
python eval.py
```
This loads weights from `output/fasterrcnn.pth` and also saves visualization plots.

## Model Architecture
- **Backbone:** ResNet-50 (DINO pretrained on ZOD, student weights)
- **Detection Head:** Faster R-CNN with COCO pretrained head
- **Fine-tuning:** End-to-end (both backbone and head are trained)

## Results
| Metric | Value |
|--------|-------|
| AP@0.50:0.95 | 33.2% |
| AP@0.50 | 61.4% |
| AP@0.75 | 31.1% |
| AR@100 | 41.0% |

Target (PDF baseline): 21.9% AP

## Files
- `fasterrcnn.py` - Main script (training + evaluation, controlled by config.train)
- `eval.py` - Standalone evaluation script (recommended for evaluation only)
- `config.py` - Configuration parameters
- `output/` - Folder containing:
  - `fasterrcnn.pth` - Trained model checkpoint
  - `eval_results.txt` - Evaluation metrics
  - `training.log` - Training logs
  - `coco_preds.json` - Predictions in COCO format
  - `gt_eval.json` - Ground truth in COCO format

## Shared Dependencies
The following files are shared across experiments (in parent directory):
- `zod_dataset.py` - Dataset classes (ZODROIFar, ZODROIWide, etc.)
- `coco_evaluation.py` - COCO-format evaluation
- `load_backbone.py` - Backbone loading utilities
- `utils.py` - Utility functions
- `vision_transformer.py` - ViT models

## Notes
- The original DINO checkpoint is NOT modified, to preserve them for future evaluations
- Uses ZODROIFar crop (matching PDF methodology)
- 80/20 train/test split with random.seed(42) for reproducibility
