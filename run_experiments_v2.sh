#!/bin/bash
# Round 2: dial in the margin loss winner.
# Axes: margin value, unfreeze depth, weight decay, more epochs.

set -e
cd /workspace/TaxoGen
source /workspace/benv/bin/activate

OUT=ir_loss_runs_v2
mkdir -p $OUT

COMMON="--train_csv splits/train.csv --val_csv splits/val.csv --test_csv splits/test.csv \
        --images_root /workspace/data --output_dir $OUT \
        --freeze_backbone --position_swap_aug \
        --batch_size 8 --epochs 6 --early_stopping_patience 3 \
        --loss margin --lr 3e-5"

declare -a EXPS=(
  # margin sweep at uf2
  "v2_m03_uf2_wd1e2  --margin 0.3 --unfreeze_top_layers 2 --weight_decay 1e-2"
  "v2_m05_uf2_wd1e2  --margin 0.5 --unfreeze_top_layers 2 --weight_decay 1e-2"
  "v2_m07_uf2_wd1e2  --margin 0.7 --unfreeze_top_layers 2 --weight_decay 1e-2"
  "v2_m10_uf2_wd1e2  --margin 1.0 --unfreeze_top_layers 2 --weight_decay 1e-2"

  # uf4 with stronger regularization (winner on val last round, mild overfit)
  "v2_m05_uf4_wd1e2  --margin 0.5 --unfreeze_top_layers 4 --weight_decay 1e-2"
  "v2_m05_uf4_wd1e1  --margin 0.5 --unfreeze_top_layers 4 --weight_decay 1e-1"
  "v2_m07_uf4_wd1e1  --margin 0.7 --unfreeze_top_layers 4 --weight_decay 1e-1"
  "v2_m05_uf4_drop3  --margin 0.5 --unfreeze_top_layers 4 --weight_decay 1e-2 --dropout 0.3"

  # uf6 deeper
  "v2_m05_uf6_wd1e1  --margin 0.5 --unfreeze_top_layers 6 --weight_decay 1e-1"

  # smaller lr at uf4 to stabilize
  "v2_m05_uf4_lr1e5  --margin 0.5 --unfreeze_top_layers 4 --weight_decay 1e-2 --lr 1e-5"
)

# All except v2_m05_uf4_lr1e5 inherit lr=3e-5 from COMMON.
# For the lr override row we just append --lr 1e-5 (last one wins).

for entry in "${EXPS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  flags=$(echo "$entry" | cut -d' ' -f2-)
  log=$OUT/${tag}.log
  echo ""
  echo "############ $tag ############"
  python train_imagereward_losses.py $COMMON --run_tag "$tag" $flags 2>&1 | tee "$log"
done

echo ""
echo "=========================================="
echo "Round 2 summary:"
for entry in "${EXPS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  log=$OUT/${tag}.log
  best_val=$(grep "Saved best" "$log" 2>/dev/null | tail -1 | grep -oE "val_acc=[0-9.]+")
  test_line=$(grep -A1 "TEST ===" "$log" 2>/dev/null | tail -1)
  test_acc=$(echo "$test_line" | grep -oE "'binary_acc': [0-9.]+" | head -1)
  test_std=$(echo "$test_line" | grep -oE "'reward_std': [0-9.]+" | head -1)
  printf "%-22s  %-18s  %-25s  %s\n" "$tag" "$best_val" "$test_acc" "$test_std"
done
