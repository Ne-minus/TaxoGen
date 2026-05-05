#!/bin/bash
# Run a sweep of loss/hyperparam combos. Each writes its own log + checkpoint.
# Tweak EPOCHS/BATCH if needed. Logs land in ir_loss_runs/.

set -e
cd /workspace/TaxoGen
source /workspace/benv/bin/activate

OUT=ir_loss_runs
mkdir -p $OUT

COMMON="--train_csv splits/train.csv --val_csv splits/val.csv --test_csv splits/test.csv \
        --images_root /workspace/data --output_dir $OUT \
        --freeze_backbone --position_swap_aug \
        --batch_size 8 --epochs 8 --early_stopping_patience 4"

# ─── Experiment grid ────────────────────────────────────────────────────────
# Each row: TAG + flags. Logs: $OUT/<tag>.log, ckpts: $OUT/best_<tag>.pt
declare -a EXPS=(
  # baseline reference
  "bt_lr1e5_uf2     --loss bt          --lr 1e-5 --unfreeze_top_layers 2"

  # margin loss: the main fix for diff-collapse
  "margin_m05_lr1e5 --loss margin      --lr 1e-5 --margin 0.5 --unfreeze_top_layers 2"
  "margin_m10_lr1e5 --loss margin      --lr 1e-5 --margin 1.0 --unfreeze_top_layers 2"
  "margin_m05_lr3e5 --loss margin      --lr 3e-5 --margin 0.5 --unfreeze_top_layers 2"
  "margin_m05_uf4   --loss margin      --lr 1e-5 --margin 0.5 --unfreeze_top_layers 4"

  # margin + decorrelation (prevent both rewards drifting up together)
  "mdec_m05_l01     --loss margin_dec  --lr 1e-5 --margin 0.5 --lambda_dec 0.1 --unfreeze_top_layers 2"
  "mdec_m05_l05     --loss margin_dec  --lr 1e-5 --margin 0.5 --lambda_dec 0.5 --unfreeze_top_layers 2"

  # margin + variance penalty (force batch-level reward spread)
  "mvar_m05_l05     --loss margin_var  --lr 1e-5 --margin 0.5 --lambda_var 0.5 --unfreeze_top_layers 2"

  # in-batch contrastive
  "infonce_t10      --loss infonce     --lr 1e-5 --infonce_temp 1.0 --unfreeze_top_layers 2"
  "infonce_t05      --loss infonce     --lr 1e-5 --infonce_temp 0.5 --unfreeze_top_layers 2"

  # hybrid bt+margin
  "hybrid_b05_m05   --loss hybrid      --lr 1e-5 --margin 0.5 --lambda_bt 0.5 --unfreeze_top_layers 2"
)

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
echo "All runs done. Summary:"
for entry in "${EXPS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  log=$OUT/${tag}.log
  test_line=$(grep -A1 "TEST ===" "$log" 2>/dev/null | tail -1)
  best_val=$(grep "Saved best" "$log" 2>/dev/null | tail -1 | grep -oE "val_acc=[0-9.]+")
  printf "%-22s  %s  |  %s\n" "$tag" "$best_val" "$test_line"
done
