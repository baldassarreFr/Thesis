# DINO v1

Note: This is a fork from https://github.com/facebookresearch/dino


## Train DINO on Nvidia Jetson Orin

- sudo docker build -t dino .

- sudo docker run -it --rm --runtime nvidia --shm-size=15g -v /home/nvidia/fssl-foundation/ZODPretraining:/dino -v /mnt/ZODversions/ZODCropped:/dino/ZOD dino 

- cd src

- python3 dino_wrapper.py

## Train DINO on 2 A100 GPUs

# DINO with ViT

python -m torch.distributed.run --nproc_per_node=2 main_dino.py --epochs 401 --arch vit_small --data_path ../data1/ZODCropped --output_dir . --patch_size 16 --global_crops_scale 0.14 1.0 --local_crops_number 8 --local_crops_scale 0.05 0.4 --batch_size 64 --out_dim 65536 --drop_path_rate 0.1 --norm_last_layer False --warmup_teacher_temp 0.04 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --freeze_last_layer 1 --lr 0.0005 --min_lr 1e-5 --warmup_epochs 10 --weight_decay 0.04 --weight_decay_end 0.4 --momentum_teacher 0.996 --saveckp_freq 20 --clip_grad 3.0

# DINO with ResNet

python -m torch.distributed.run --nproc_per_node=2 main_dino.py --epochs 401 --arch resnet50 --data_path ../data1/ZODCropped --output_dir . --patch_size 16 --global_crops_scale 0.14 1.0 --local_crops_number 8 --local_crops_scale 0.05 0.4 --batch_size 64 --out_dim 65536 --drop_path_rate 0.1 --norm_last_layer False --warmup_teacher_temp 0.04 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --freeze_last_layer 1 --lr 0.0005 --min_lr 1e-5 --warmup_epochs 10 --weight_decay 0.04 --weight_decay_end 0.4 --momentum_teacher 0.996 --saveckp_freq 20 --clip_grad 3.0

## Visualize DINO attention on ZOD

- python3 visualization_wrapper.py
