#!/usr/bin/env bash

# Plain-DETR config for ZOD dataset with ResNet-50 backbone
# Backbone: DINO-pretrained on ZOD (converted weights)
# Training: Full training run (50 epochs, eval every 5)

set -x

FILE_NAME=$(basename $0)
EXP_DIR=./exps/${FILE_NAME%.*}
PY_ARGS=${@:1}

# Path to converted DINO-pretrained ResNet checkpoint
PRETRAINED_BACKBONE=/root/projects/Plain-DETR/converted_backbones/dino_resnet50.pth

# Training hyperparameters
GPUS=8  # 8x GTX 1080Ti (11GB each)
EPOCHS=50
EVAL_EVERY=5
BATCH_SIZE=24  # Per GPU - 11GB memory, conservative
LR_TRANSFORMER=1e-4
LR_BACKBONE=1e-5
WEIGHT_DECAY=1e-4
NUM_WORKERS=4  # 32 CPUs / 8 GPUs = 4 workers per GPU
K_ONE2MANY=0

bash ./tools/run_dist_launch.sh ${GPUS} /opt/conda/bin/python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file zod \
    --zod_path /root/zod-dataset/ \
    --zod_crop none \
    --zod_image_size 800 \
    --zod_image_height 448 \
    --zod_train_split val_finetune \
    --backbone resnet50 \
    --pretrained_backbone_path ${PRETRAINED_BACKBONE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --num_feature_levels 1 \
    --decoder_type global_ape \
    --epochs ${EPOCHS} \
    --lr ${LR_TRANSFORMER} \
    --lr_backbone ${LR_BACKBONE} \
    --weight_decay ${WEIGHT_DECAY} \
    --eval_every ${EVAL_EVERY} \
    --k_one2many ${K_ONE2MANY} \
    --not_auto_resume \
    ${PY_ARGS}
