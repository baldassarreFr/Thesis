train = False
output_path = "fasterrcnn.pth"
backbone = "resnet_supervised"
model_checkpoint_path = (
    "/root/projects/fssl-foundation/ZODPretraining/src/outputs/checkpoint.pth"
)
datapath = "/root/zod-dataset/"
num_epochs = 20
score_threshold = 0.3
batch_size = 2  # Reduced from 4 due to OOM
num_workers = 4  # Reduced from 12 to avoid OOM

# Quick test settings (set to None for full run)
quick_train_subset = None  # None = full training
quick_test_subset = None  # None = full evaluation
