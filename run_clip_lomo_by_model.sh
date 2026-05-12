#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/TaxoGen

FOLDS_DIR="${FOLDS_DIR:-splits/lomo_by_model}"
LOG_DIR="${LOG_DIR:-logs/lomo_by_model}"
CKPT_DIR="${CKPT_DIR:-clip_lomo_by_model_ckpt}"

python make_lomo_splits.py --splits_dir splits --out_dir "$FOLDS_DIR"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

tail -n +2 "$FOLDS_DIR/manifest.csv" | while IFS=, read -r model fold train_rows val_rows test_rows train_wids val_wids test_wids removed_train removed_val; do
    tag="lomo_${fold}_large14_bt_ls01_drop05"
    log="$LOG_DIR/${tag}.log"

    echo "================================================================"
    echo "Held-out model: $model"
    echo "Fold: $fold | train=$train_rows val=$val_rows test=$test_rows"
    echo "Log: $log"
    echo "================================================================"

    python train_clip_pairs_only.py \
        --train_csv  "$FOLDS_DIR/$fold/train.csv" \
        --val_csv    "$FOLDS_DIR/$fold/val.csv" \
        --test_csv   "$FOLDS_DIR/$fold/test.csv" \
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
        --output_dir "$CKPT_DIR" \
        --run_tag "$tag" \
        2>&1 | tee "$log"
done
