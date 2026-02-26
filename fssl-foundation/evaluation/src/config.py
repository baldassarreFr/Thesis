train = True
output_path = "fasterrcnn.pth"
backbone = "resnet_supervised" # resnet_random, resnet_ssl // vit_random, vit_trained
model_checkpoint_path = None # "trained_backbones/imagenet_50_epochs.pth" # must be provided if resnet_ssl is used 
datapath = "../../../../../mnt/ZOD/"
num_epochs = 20
score_threshold = 0.7