import re
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
LOG_FILE = "/root/Plain-DETR-v2/wandb/DINOv3_vit_small_on_COCO_complete_run.log"
SMOOTHING_WEIGHT = 0.5  # Standard TensorBoard smoothing (0.0 to 0.99)
# ---------------------

def smooth_curve(scalars, weight):
    """
    Applies an Exponential Moving Average (EMA) to a list of scalars.
    This replicates the exact smoothing algorithm used by TensorBoard.
    """
    if not scalars:
        return []
        
    last = scalars[0]
    smoothed = []
    for point in scalars:
        # EMA formula
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

global_steps = []
learning_rates = []
raw_losses = []

map_steps = []
map_values = []

print(f"Parsing file {LOG_FILE} line by line (Step-wise)...")

with open(LOG_FILE, "r") as f:
    for line in f:
        # Look for lines printed at each batch:
        match_step = re.search(r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?lr:\s*([0-9\.e\-]+).*?loss:\s*([0-9\.]+)\s*\(([0-9\.]+)\)", line)
        
        if match_step:
            epoch = int(match_step.group(1))
            step_in_epoch = int(match_step.group(2))
            max_steps_per_epoch = int(match_step.group(3))
            
            lr = float(match_step.group(4))
            # We take the raw loss (group 5) because we will apply our own continuous smoothing
            loss_val = float(match_step.group(5)) 
            
            # Calculate the continuous global step
            global_step = (epoch * max_steps_per_epoch) + step_in_epoch
            
            global_steps.append(global_step)
            learning_rates.append(lr)
            raw_losses.append(loss_val)
            
        # Look for mAP lines
        match_map = re.search(r"copypaste: AP AP50 AP75 APs APm APl (\d+\.\d+)", line)
        if match_map and len(global_steps) > 0:
            map_steps.append(global_steps[-1])
            map_values.append(float(match_map.group(1)) * 100)

print(f"Found {len(global_steps)} training steps and {len(map_values)} mAP evaluations.")

# Apply TensorBoard-style continuous smoothing to the raw losses
smoothed_losses = smooth_curve(raw_losses, weight=SMOOTHING_WEIGHT)

# --- PLOTTING ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Learning Rate
ax1.plot(global_steps, learning_rates, color='#ff7f0e', linewidth=2)
ax1.set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
ax1.set_xlabel("Global Steps", fontsize=12)
ax1.set_ylabel("Learning Rate", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Plot 2: Training Loss (Our Custom Smoothed Version)
ax2.plot(global_steps, smoothed_losses, color='#8a2be2', linewidth=2)
ax2.set_title(f"Training Loss (EMA Smoothed, w={SMOOTHING_WEIGHT})", fontsize=14, fontweight='bold')
ax2.set_xlabel("Global Steps", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Plot 3: mAP Evolution
ax3.plot(map_steps, map_values, marker='s', color='#2ca02c', linewidth=2, markersize=8)
ax3.set_title("mAP Evolution (COCO)", fontsize=14, fontweight='bold')
ax3.set_xlabel("Global Steps", fontsize=12)
ax3.set_ylabel("mAP (%)", fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)

# Highlight the final mAP point
if len(map_values) > 0:
    ax3.annotate(f"{map_values[-1]:.1f}%", 
                 xy=(map_steps[-1], map_values[-1]), 
                 xytext=(map_steps[-1] - (max(map_steps)*0.1), map_values[-1]+2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("plots.png", dpi=300)
print("Plot successfully saved as 'plots.png'!")
plt.show()