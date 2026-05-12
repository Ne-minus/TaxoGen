#!/bin/bash
set -e
source /venv/main/bin/activate
cd /workspace/TaxoGen

LOG=logs/ir_finetune_clean.log

python train_imagereward_pairs_only.py \
    --train_csv  splits/train.csv \
    --val_csv    splits/val.csv \
    --test_csv   splits/test.csv \
    --images_root /workspace/TaxoGen_sampled \
    --dropout 0.2 \
    --label_smoothing 0.1 \
    --lr 1e-5 \
    --batch_size 8 \
    --epochs 15 \
    --early_stopping_patience 5 \
    --freeze_backbone \
    --unfreeze_top_layers 2 \
    --position_swap_aug \
    --output_dir ir_ckpt_clean \
    2>&1 | tee "$LOG"
