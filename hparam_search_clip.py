"""Grid search for CLIP reward model hyperparameters (parallel, 2 workers)."""

import subprocess, itertools, csv, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIGS = list(itertools.product(
    [1e-5, 3e-5, 1e-4],      # lr
    [0.3, 0.5],               # dropout
    [0.0, 0.1, 0.2],          # label_smoothing
    [1, 2, 3],                # unfreeze_top layers
))

BASE = [
    "python", "train_clip_pairs_only.py",
    "--train_csv",  "splits/train.csv",
    "--val_csv",    "splits/val.csv",
    "--test_csv",   "splits/test.csv",
    "--images_root", "/workspace/TaxoGen_sampled",
    "--model_name", "openai/clip-vit-large-patch14",
    "--loss", "bt",
    "--batch_size", "16",
    "--epochs", "12",
    "--early_stopping_patience", "3",
    "--freeze_backbone",
    "--position_swap_aug",
    "--output_dir", "clip_hparam_ckpts",
]

os.makedirs("clip_hparam_ckpts", exist_ok=True)
results_file = "logs/hparam_clip_results.csv"
os.makedirs("logs", exist_ok=True)

with open(results_file, "w", newline="") as f:
    csv.writer(f).writerow(["lr", "dropout", "label_smoothing", "unfreeze", "val_acc", "test_acc", "run_tag"])

print(f"Total configs: {len(CONFIGS)}")
write_lock = threading.Lock()
counter = {"n": 0}


def run_config(args_tuple):
    i, (lr, dropout, ls, unfreeze) = args_tuple
    tag = f"lr{lr:.0e}_drop{dropout}_ls{ls}_unf{unfreeze}"
    out_dir = f"clip_hparam_ckpts/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = BASE + [
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--label_smoothing", str(ls),
        "--unfreeze_top_vision_layers", str(unfreeze),
        "--unfreeze_top_text_layers", str(unfreeze),
        "--run_tag", tag,
        "--output_dir", out_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/TaxoGen")

    val_acc = None
    test_acc = None
    in_test = False
    for line in result.stdout.splitlines():
        if "Saved best" in line and "val binary_acc" in line:
            try:
                val_acc = float(line.split("val binary_acc=")[1].strip().rstrip(")"))
            except Exception:
                pass
        if "=== " in line and "TEST ===" in line:
            in_test = True
        if in_test and "'binary_acc':" in line:
            try:
                test_acc = float(line.split("'binary_acc':")[1].split(",")[0].strip())
            except Exception:
                pass
            in_test = False

    with write_lock:
        counter["n"] += 1
        print(f"[{counter['n']}/{len(CONFIGS)}] {tag}  val={val_acc}  test={test_acc}", flush=True)
        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([lr, dropout, ls, unfreeze, val_acc, test_acc, tag])

    return val_acc, test_acc, tag


with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(run_config, (i, cfg)) for i, cfg in enumerate(CONFIGS)]
    for fut in as_completed(futures):
        fut.result()

import pandas as pd
df = pd.read_csv(results_file).dropna()
df = df.sort_values("val_acc", ascending=False)
print("\n=== Top 10 configs ===")
print(df.head(10).to_string(index=False))
