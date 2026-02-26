# find config explanations here: https://docs.google.com/spreadsheets/d/1viVa87YU4pFKX-x7xJTVOfnT6i-cwmqvcR0VCtHsFOU/edit?gid=0#gid=0

# 1) Data preparation/augmentation
datapath = "../../../tiny-imagenet-200/train"
global_crops_scale = (0.14, 1.0)
local_crops_number = 8
local_crops_scale = (0.05, 0.4)
batch_size = 64
num_workers = 4

# 2) DINO
out_dim = 65536 # For complex and large datasets large values (like 65k) work well
use_bn_in_head = False
arch = "resnet" # vit_tiny, vit_small, vit_base, resnet
patch_size = 16 # Using smaller values leads to better performance but requires more memory.
drop_path_rate = 0.1
norm_last_layer = False # In our experiments, we typically set this paramater to False with vit_small and True with vit_base.

# 3) Training
epochs = 100
warmup_teacher_temp = 0.04 # Try decreasing it if the training loss does not decrease.
teacher_temp = 0.04        # For most experiments, anything above 0.07 is unstable. 
                           # We recommend starting with the default value of 0.04 and increase this slightly if needed.
warmup_teacher_temp_epochs = 30 # Number of warmup epochs for the teacher temperature (Default: 30).
freeze_last_layer = 1 # Number of epochs
                      # during which we keep the output layer fixed. Typically doing so during
                      # the first epoch helps training. Try increasing this value if the loss does not decrease.
lr = 0.0005
min_lr = 1e-5 # 1e-6
warmup_epochs = 10 
weight_decay = 0.04 
weight_decay_end = 0.4 
momentum_teacher = 0.996 #0.9995 # We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.

output_dir = "." #"./pretrained_vit-s-16/" # change this to the .pth folder if you want to use pre-trained weights
saveckp_freq = 20
clip_grad = 3.0