#!/usr/bin/env bash

set -x

FILE_NAME=$(basename "$0")
EXP_DIR=./exps/${FILE_NAME%.*}
PY_ARGS=("${@:1}")

python -u -m plain_detr.main \
    --args.output_dir "${EXP_DIR}" \
    --args.with_box_refine \        # Instead of predicting boxes once, the model refines them step by step through each decoder layer
    --args.two_stage \              # First stage generates potential object regions (proposals), second stage refines them
    --args.mixed_selection \        
    --args.look_forward_twice \
    --args.num_queries_one2one 300 \    # The model can detect up to 300 objects per image
    --args.num_queries_one2many 1500 \  # More queries (1500) used during TRAINING only to help learn better
    --args.k_one2many 6 \               # For each real object, pretend to find it 6 different ways
    --args.lambda_one2many 1.0 \        # How much to care about these extra pretend matches during training
    --args.dropout 0.0 \                # Dropout rate - randomly "turns off" neurons during training
    --args.norm_type pre_norm \
    --args.backbone dinov3_vit_small \
    --args.drop_path_rate 0.1 \         # Randomly skip entire transformer blocks during training
    --args.num_feature_levels 1 \       # Backbone outputs one scale of features
    --args.decoder_type global_rpe_decomp \   # Global attention with relative position encoding handles spatial relationships
    --args.decoder_rpe_type linear \
    --args.proposal_feature_levels 4 \      # Use features from 4 different scales to generate object proposals
    --args.proposal_in_stride 16 \          # Check object proposals every 16 pixel
    --args.pretrained_backbone_path ./pt_models/dinov3_vit_small.safetensors \
    --args.epochs 12 \
    --args.lr_drop 11 \
    --args.warmup 1000 \        # Gradually increase learning rate for the first 1000 steps to help training stability
    --args.lr 2e-4 \
    --args.use_layerwise_decay \  # Different layers get different learning rates
    --args.lr_decay_rate 0.9 \     # Each layer's learning rate is 0.9 times the previous layer's rate
    --args.weight_decay 0.05 \     # Regularization to prevent overfitting
    --args.wd_norm_mult 0.0 \
    --args.batch_size 8 \
    --args.use_wandb true \
    --args.wandb_name dinov3_vit_small_run1 \
    "${PY_ARGS[@]}"
