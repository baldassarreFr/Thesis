#!/bin/bash
# Train DINO ResNet50 on ZOD with 4 GPUs

python -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --data_path /mnt/tier2/project/p201222/data/ZOD_clone_2018_scaleout_zenseact \
    --epochs 10 \
    --arch resnet50 \
    --output_dir ./output \
    --patch_size 16 \
    --global_crops_scale 0.14 1.0 \
    --local_crops_number 8 \
    --local_crops_scale 0.05 0.4 \
    --batch_size_per_gpu 128 \
    --out_dim 65536 \
    --drop_path_rate 0.1 \
    --norm_last_layer False \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 2 \
    --freeze_last_layer 1 \
    --lr 0.0005 \
    --min_lr 1e-5 \
    --warmup_epochs 2 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --momentum_teacher 0.996 \
    --saveckp_freq 20 \
    --clip_grad 3.0