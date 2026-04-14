#!/bin/bash
cd /root/Plain-DETR-v2

uv run torchrun --standalone --nproc-per-node=4 -m plain_detr.main \
    --args.resume true \
    --args.output_dir ./exps/dinov3_vit_small_boxrpe \
    --args.backbone dinov3_vit_small \
    --args.pretrained_backbone_path ./pt_models/dinov3_vit_small.safetensors \
    --args.batch_size 8 \
    --args.num_workers 8 \
    --args.with_box_refine \
    --args.two_stage \
    --args.mixed_selection \
    --args.look_forward_twice \
    --args.num_queries_one2one 300 \
    --args.num_queries_one2many 1500 \
    --args.epochs 12 \
    --args.lr 2e-4 \
    --args.lr_drop 11 \
    --args.use_layerwise_decay \
    --args.lr_decay_rate 0.9 \
    --args.weight_decay 0.05 \
    --args.norm_type pre_norm \
    --args.num_feature_levels 1 \
    --args.decoder_type global_rpe_decomp \
    --args.decoder_rpe_type linear \
    --args.proposal_feature_levels 4 \
    --args.proposal_in_stride 16 \
    --args.drop_path_rate 0.1 \
    --args.k_one2many 6 \
    --args.lambda_one2many 1.0 \
    --args.dropout 0.0 \
    --args.wd_norm_mult 0.0 \
    --args.warmup 1000 \
    --args.use_wandb \
    --args.wandb_name dinov3_vit_small_run3