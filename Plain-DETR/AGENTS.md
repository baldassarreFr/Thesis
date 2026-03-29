# AGENTS.md - Plain-DETR Codebase Guide

## CRITICAL RULES
- **NEVER change formatting** (quotes, commas, indentation, line breaks) of existing code
- Only modify the specific lines needed for the task
- If you need to make changes, ask for approval first
- Do NOT use automated tools that reformat code

## Overview
Plain-DETR is a PyTorch implementation of "DETR Doesn't Need Multi-Scale or Locality Design" (ICCV 2023). This is a research project for object detection using DETR-based models.

## Project Structure
```
Plain-DETR/
├── main.py           # Training/evaluation entry point
├── engine.py         # Train and eval functions
├── models/           # Model definitions (backbone, transformer, detr)
├── datasets/         # Data loading, COCO evaluation
├── util/             # Utilities (box_ops, misc, plotting)
├── configs/          # Configuration shell scripts
├── tools/            # Training/evaluation launch scripts
└── exps/             # Experiment configurations
```

## Commands

### Training (Single Node)
```bash
GPUS_PER_NODE=<num gpus> ./tools/run_dist_launch.sh <num gpus> <path to config>
```

### Training (Multi-node)
```bash
MASTER_ADDR=<master node IP> GPUS_PER_NODE=<num gpus> NODE_RANK=<rank> \
  ./tools/run_dist_launch.sh <num gpus> <path to config>
```

### Evaluation
```bash
<path to config> --eval --resume <path to model checkpoint>
```

### Prepare Pretrained Models
```bash
bash tools/prepare_pt_model.sh
```

### Running a Single Test
This project does not have a formal test suite. To test code changes:
1. Run a short training iteration with reduced epochs/batch size
2. Or run evaluation on a small subset

### Linting/Type Checking
No formal linting setup exists. Follow PEP 8 conventions. Consider using:
```bash
pip install flake8 black isort mypy
flake8 .
black --check .
isort --check .
mypy .
```

### Dependency Installation
```bash
conda create -n plain_detr python=3.8 -y
conda activate plain_detr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports (torch, numpy, etc.)
- Local imports (relative imports within project)
- Use explicit relative imports: `from models import build_model` or `from .module import Class`

### Formatting
- 4 spaces for indentation
- Maximum line length: 100 characters (soft limit)
- Use f-strings for string formatting: `f"value: {value:.3f}"`
- Use double quotes for strings unless containing double quotes
- Add space around operators: `a + b`, not `a+b`

### Type Hints
- Use type hints for function signatures when beneficial
- Common types: `Optional`, `List`, `Dict`, `Iterable`, `Tensor`
- Import from `typing` module

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods/variables: prefix with underscore `_private_method`

### File Headers
Include copyright header in new files:
```python
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
```

### Classes
- Use `super().__init__()` without arguments for Python 3 compatibility
- Document parameters in docstrings

### Error Handling
- Use assertions for debugging: `assert condition, "message"`
- Use try/except for recoverable errors
- Raise specific exceptions with clear messages

### Code Patterns

**Argparse arguments:**
```python
parser.add_argument("--name", default=default_value, type=float/int/str)
parser.add_argument("--flag", action="store_true")  # boolean flags
```

**Model forward pass pattern:**
```python
# Use torch.no_grad() for inference
@torch.no_grad()
def evaluate(model, ...):
    model.eval()
    # evaluation code
```

**Distributed training:**
```python
from torch import distributed as dist
if utils.is_main_process():
    # logging, saving, etc.
```

**Checkpoint saving:**
```python
utils.save_on_master({"model": model.state_dict(), ...}, path)
```

**Logging:**
```python
print(f"Epoch: [{epoch}] loss: {loss_value:.4f}")
# or with metric_logger
metric_logger = utils.MetricLogger(delimiter="  ")
metric_logger.update(loss=loss_value)
```

### Key Utilities (util/misc.py)
- `utils.MetricLogger` - Training metrics logging
- `utils.SmoothedValue` - Moving average for metrics
- `utils.save_on_master` - Save only on main process
- `utils.is_main_process()` - Check if main process
- `utils.get_rank()` - Get process rank
- `utils.reduce_dict()` - Reduce dict across GPUs

### Device Management
```python
device = torch.device(args.device)
model.to(device)
tensor.to(device)
```

### Data Loading
```python
from torch.utils.data import DataLoader
data_loader = DataLoader(dataset, batch_size=..., num_workers=..., pin_memory=True)
```

### Common Gotchas
- Use `torch.no_grad()` during evaluation
- Set model to `.train()` during training, `.eval()` during inference
- Synchronize metrics with `metric_logger.synchronize_between_processes()` in distributed mode
- Use `utils.reduce_dict()` for loss averaging across GPUs
- Handle non-finite losses: `if not math.isfinite(loss_value): sys.exit(1)`

## Environment
- Python 3.8
- PyTorch 1.13.1
- CUDA 11.7
- Key packages: torch, torchvision, timm==0.4.5, mmpycocotools, tqdm, scipy, wandb
