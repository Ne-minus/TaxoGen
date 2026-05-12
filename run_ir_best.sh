#!/bin/bash
set -e
source /venv/main/bin/activate
cd /workspace/TaxoGen

LOG=logs/ir_best_final.log

python train_imagereward_pairs_only.py \
    --train_csv  splits/train.csv \
    --val_csv    splits/val.csv \
    --test_csv   splits/test.csv \
    --images_root /workspace/TaxoGen_sampled \
    --lr 3e-5 \
    --dropout 0.2 \
    --label_smoothing 0.1 \
    --unfreeze_top_layers 5 \
    --batch_size 8 \
    --epochs 15 \
    --early_stopping_patience 5 \
    --freeze_backbone \
    --position_swap_aug \
    --weight_decay 1e-2 \
    --output_dir ir_ckpt_best \
    2>&1 | tee "$LOG"
