#!/bin/bash
set -e
source /venv/main/bin/activate
cd /workspace/TaxoGen

LOG=logs/large14_bt_ls01_drop05_clean.log

python train_clip_pairs_only.py \
    --train_csv  splits/train.csv \
    --val_csv    splits/val.csv \
    --test_csv   splits/test.csv \
    --images_root /workspace/TaxoGen_sampled \
    --model_name openai/clip-vit-large-patch14 \
    --loss bt \
    --label_smoothing 0.1 \
    --dropout 0.5 \
    --lr 3e-5 \
    --batch_size 16 \
    --epochs 15 \
    --early_stopping_patience 5 \
    --freeze_backbone \
    --unfreeze_top_vision_layers 2 \
    --unfreeze_top_text_layers 2 \
    --position_swap_aug \
    --output_dir clip_ckpt_clean \
    --run_tag large14_bt_ls01_drop05_clean \
    2>&1 | tee "$LOG"
