# ------------------------------------------------------------------------
# Visualization utilities for ZOD detection
# ------------------------------------------------------------------------

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


CATEGORY_NAMES = {1: "Vehicle", 2: "VulnerableVehicle", 3: "Pedestrian"}


def visualize_best_predictions(
    output_dir,
    dataset,
    model=None,
    postprocessors=None,
    device=None,
    predictions_in_resized_coords=False,
):
    """Visualize top 4 images with highest confidence predictions.

    Loads predictions from zod_predictions.json, finds top 4 images by sum of
    confidence scores (>0.5 threshold), and draws boxes on original high-res images.

    Args:
        output_dir: Experiment output directory
        dataset: ZODDetection dataset (to access get_original_image)
    """
    output_path = Path(output_dir)
    predictions_file = output_path / "zod_predictions.json"

    if not predictions_file.exists():
        print(f"WARNING: {predictions_file} not found. Skipping visualization.")
        return

    with open(predictions_file) as f:
        data = json.load(f)

    # Handle both old format (just predictions list) and new format (with metadata)
    if isinstance(data, dict) and "predictions" in data:
        predictions = data["predictions"]
        coord_space = data.get("coordinate_space", "original_4k")
    else:
        predictions = data
        coord_space = "original_4k"  # default assumption

    image_scores = {}
    image_boxes = {}
    for pred in predictions:
        if pred["score"] > 0.5:
            img_id = pred["image_id"]
            if img_id not in image_scores:
                image_scores[img_id] = 0
                image_boxes[img_id] = []
            image_scores[img_id] += pred["score"]
            image_boxes[img_id].append(pred)

    top_4_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:4]

    vis_dir = output_path / "output_final_visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Get resize dimensions from dataset if available, otherwise assume 800x448
    # Predictions should be in original 4K coords (3848x2168), but if they're
    # in resized coords (e.g., < 1000), we need to scale them up
    resize_w = getattr(dataset, "rescaled_size", (800, 448))[1]
    resize_h = getattr(dataset, "rescaled_size", (800, 448))[0]

    for rank, (img_id, total_score) in enumerate(top_4_images):
        try:
            orig_img = dataset.get_original_image(img_id)
            if orig_img is None:
                print(
                    f"WARNING: Could not load original image for ID {img_id}: returned None"
                )
                continue
        except Exception as e:
            print(f"WARNING: Could not load original image for ID {img_id}: {e}")
            continue

        draw = ImageDraw.Draw(orig_img)
        boxes = image_boxes.get(img_id, [])

        for box_info in boxes:
            x_min, y_min, w, h = box_info["bbox"]
            x_max = x_min + w
            y_max = y_min + h

            # If predictions are in resized coords, scale to original 4K
            if coord_space == "resized":
                orig_w, orig_h = orig_img.size
                scale_x = orig_w / resize_w
                scale_y = orig_h / resize_h
                x_min = x_min * scale_x
                y_min = y_min * scale_y
                x_max = x_max * scale_x
                y_max = y_max * scale_y

            cat_id = box_info["category_id"]
            score = box_info["score"]
            label = CATEGORY_NAMES.get(cat_id, f"Class{cat_id}")

            color = (
                (0, 255, 0)
                if cat_id == 1
                else (255, 0, 0)
                if cat_id == 2
                else (0, 0, 255)
            )

            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=5,
            )

            text = f"{label}: {score:.2f}"
            draw.text((x_min, y_min - 25), text, fill=color)

        save_path = vis_dir / f"best_pred_{rank + 1}_img{img_id}.jpg"
        orig_img.save(save_path)
        print(f"Saved: {save_path} (total conf: {total_score:.2f})")


def generate_performance_plots(metrics_jsonl, output_dir):
    """Generate training performance curves from JSONL metrics.

    Args:
        metrics_jsonl: Path to training_metrics.jsonl
        output_dir: Output directory for plots
    """
    metrics_path = Path(metrics_jsonl)
    if not metrics_path.exists():
        print(f"WARNING: {metrics_jsonl} not found. Skipping plot generation.")
        return

    data = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        print("WARNING: No data found in metrics file.")
        return

    # Separate train and test metrics
    # Train metrics are available every epoch
    epochs_train = [d["epoch"] for d in data]
    losses = [d.get("train_loss", 0) for d in data]

    # Test metrics are only available when eval was run (based on eval_every)
    eval_data = [d for d in data if "test_AP" in d and d.get("test_AP", 0) > 0]
    epochs_eval = [d["epoch"] for d in eval_data]
    ap = [d.get("test_AP", 0) for d in eval_data]
    ap50 = [d.get("test_AP50", 0) for d in eval_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss for all epochs (train metrics available every epoch)
    ax1.plot(epochs_train, losses, marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss vs Epoch")
    ax1.grid(True)

    # Plot mAP only for epochs where evaluation was run
    if epochs_eval:
        ax2.plot(epochs_eval, ap50, marker="s", label="AP50", color="blue")
        ax2.plot(epochs_eval, ap, marker="^", label="AP", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title("Validation mAP vs Epoch (evaluated epochs only)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    vis_dir = Path(output_dir) / "output_final_visualizations"
    vis_dir.mkdir(exist_ok=True)
    save_path = vis_dir / "training_performance_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")
