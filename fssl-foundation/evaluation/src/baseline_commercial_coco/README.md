# Baseline: Commercial COCO-Pretrained Faster R-CNN

## Description
This baseline uses a Faster R-CNN with ResNet-50 FPN backbone pretrained on COCO, then fine-tuned on ZOD.

**Important:** This is NOT using the ZOD-pretrained ResNet-50 from DINO. This is a trivial baseline.

## Default Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch Size | 8 |
| Learning Rate | 0.005 |
| Train Samples | 1000 |
| Test Samples | 300 |
| Image Size | 640 |
| Dataset | /root/zod-dataset/ |

## Usage
```bash
# Just run (uses all defaults)
python train.py

# Override specific options
python train.py --epochs 20 --batch_size 4
```

The script automatically:
- Uses ZOD dataset at `/root/zod-dataset/` (default)
- Saves outputs to `../../outputs/baseline_commercial_coco/` (based on folder name)

## Results
- **mAP@0.5:** 17.2%
- **mAP@0.5:0.95:** 7.9%

## Notes
This serves as a baseline to compare against the ZOD-DINO pretrained model.
