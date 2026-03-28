# AGENTS.md

## Project

Plain-DETR — object detection with DETR using single-scale features and global cross-attention (ICCV 2023).

## Stack

- Python 3.12, PyTorch, torchvision, timm
- CLI: cyclopts + pydantic
- Package manager: uv (deps in `pyproject.toml`, lockfile in `uv.lock`)
- Build system: hatchling

## Commands

```bash
uv sync                  # install deps
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
uv run pytest            # test
```

## Code style

- Format and lint with ruff (`line-length = 120`, rules in `pyproject.toml`).
- Type check with ty.
- Use modern type annotations: `list[int]`, `dict[str, Any]`, `Path | None` — never `List`, `Optional`, `Union`.
- Use `pathlib.Path` for all filesystem paths, not `os.path`.
- Use `logging` for output, with f-strings for message formatting (not `%`-style or `.format()`).
- Function parameters should not have default values unless there is a clear reason. Think twice before adding a default.
- Fail early and loudly. Raise exceptions at the first sign of invalid state. Do not silently swallow errors or return fallback values — defensive programming hides bugs.

## Assert vs raise

- Use `assert` for **internal invariants** — conditions that should be impossible if the code is correct. When an assertion fires, we are at fault, not the user, not another library, not the system.
- Use `raise` for **input validation and runtime errors** — conditions that depend on caller-supplied data, configuration, or external state. When an exception is raised, the caller may have made a mistake, or an external system may have failed, and we want to provide a clear error message and allow them to handle it.
- Rule of thumb: if a caller could reasonably trigger the condition through normal (mis)use, `raise` an exception. If the condition implies a bug in our own code, use `assert`.
- Always include informative context in both assert messages and exception messages to facilitate debugging.
- In pytest tests, always use `assert` (never `raise`) for test expectations.

## Project layout

```
plain_detr/          # main package
  main.py            # training/eval entry point (cyclopts CLI)
  benchmark.py       # inference speed benchmarking
  engine.py          # train/eval loops
  datasets/          # COCO data loading, transforms, evaluation
  models/            # DETR model, backbone (SwinV2), decoders, matcher
  util/              # box ops, misc helpers
configs/             # shell scripts with experiment hyperparameters
tools/               # distributed launch scripts, model download
```

## Testing

- Framework: **pytest**.
- No tests exist yet. When adding code, add tests.
