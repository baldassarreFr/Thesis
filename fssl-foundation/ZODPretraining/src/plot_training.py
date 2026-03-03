import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_metrics(log_path: str, output_dir: str | None = None):
    """Plot training metrics from log.txt file.

    Args:
        log_path: Path to the log.txt file
        output_dir: Directory to save plots (default: same as log file)
    """
    log_path = Path(log_path)
    if output_dir is None:
        output_dir = log_path.parent
    else:
        output_dir = Path(output_dir)

    df = pd.read_json(log_path, lines=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["train_lr"], marker="o", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate")
    axes[1].set_yscale("log")
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["train_wd"], marker="o", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Weight Decay")
    axes[2].set_title("Weight Decay")
    axes[2].grid(True)

    plt.tight_layout()
    output_path = output_dir / "training_metrics.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot training metrics")
    parser.add_argument(
        "--log_path", type=str, default="output/log.txt", help="Path to log.txt file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save plots"
    )
    args = parser.parse_args()

    plot_training_metrics(args.log_path, args.output_dir)
