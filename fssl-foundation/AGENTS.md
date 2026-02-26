# AGENTS.md - Development Guidelines

## Projects

- **zod/**: Zenseact Open Dataset (pytest + ruff)
- **fssl-foundation/**: Federated Self-Supervised Learning
- **Plain-DETR/**: DETR implementation

## Commands

### zod (main project)
```bash
cd zod && pip install -e ".[dev]"  # includes ruff, pytest

# Lint/format
ruff check zod/              # check all
ruff check --fix zod/        # auto-fix
ruff format zod/             # format

# Test
pytest                          # all tests
pytest tests/test_file.py       # single file
pytest tests/test_file.py::test_name  # single test
pytest -k "pattern"             # by pattern
pytest -v -x                    # verbose, stop on fail
```

### fssl-foundation
```bash
pip install -r DINOv1/requirements.txt
pip install -r ZODPretraining/requirements.txt
pip install -r evaluation/requirements.txt

cd DINOv1/src && python main_dino.py --data_path /path --output_dir ./output
```

### Plain-DETR
```bash
pip install -r requirements.txt
bash configs/swinv2_small_sup_pt_ape.sh
```

## Code Style

### Imports (ruff isort)
```python
# Stdlib (alphabetical)
import argparse
import os
from pathlib import Path
from typing import Any

# Third-party (alphabetical)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Local
from zod import ZodFrames
from zod.constants import NotAnnotatedError
```

### Formatting
- Line length: 120 chars (zod), ~100-120 (fssl)
- 4 spaces, f-strings, trailing commas

```python
# Good
result = func(
    arg1="value1",
    arg2="value2",
)
```

### Naming
- **Classes**: PascalCase (`ZodFrames`, `DINOLoss`)
- **Functions/vars**: snake_case (`train_dino`, `fix_random_seeds`)
- **Constants**: UPPER_SNAKE_CASE (`NUM_WORKERS`)
- **Private**: underscore prefix (`_internal`)

### Type Hints
```python
def load_weights(model: nn.Module, path: str) -> None:
    """Load pretrained weights."""
    ...
```

### Error Handling
```python
# Assertions for internal checks
assert len(frames) > 0, "Dataset empty"

# Try/except for I/O
try:
    state_dict = torch.load(path)
except FileNotFoundError:
    print(f"Not found: {path}")
    sys.exit(1)

# Check for NaN
if not math.isfinite(loss.item()):
    sys.exit(1)
```

### Docstrings
```python
def scheduler(base: float, final: float, epochs: int) -> np.ndarray:
    """Generate cosine learning rate schedule.
    
    Args:
        base: Initial learning rate.
        final: Final learning rate.
        epochs: Number of epochs.
    Returns:
        Array of learning rates.
    """
    ...
```

### PyTorch
```python
class DINOLoss(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        ...

@torch.no_grad()
def evaluate(model: nn.Module, loader) -> dict:
    ...

# Mixed precision
with torch.cuda.amp.autocast():
    loss = criterion(output, target)
```

### Argument Parsing
```python
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--arch', default='vit_small')
    return parser

if __name__ == '__main__':
    args = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args()
```

### Distributed Training
```python
init_distributed_mode(args)
if is_main_process():
    print(f"Training for {args.epochs} epochs")
```

### Ruff Config (zod/pyproject.toml)
```toml
[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["I", "W", "E"]
```

## Common Issues

**CUDA OOM**: Reduce batch size, use gradient accumulation, enable mixed precision.

**Loss NaN**: Check preprocessing, reduce LR, disable amp.

**Distributed**: Ensure NCCL, check RANK/WORLD_SIZE/LOCAL_RANK, use `dist_url="env://"`.
