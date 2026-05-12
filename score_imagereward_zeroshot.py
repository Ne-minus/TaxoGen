"""
Zero-shot ImageReward scoring for test set images.
Produces model, wordnet_id, reward CSV (same format as scores_collected.csv).
"""
import os
import pandas as pd
import numpy as np
import ImageReward as RM
from PIL import Image

IMAGES_ROOT = "/workspace/collected_wordnet_images"
META_CSV = "/workspace/TaxoGen/splits/wordnet_meta.csv"
TEST_CSV = "/workspace/TaxoGen/splits/test.csv"
OUT_CSV = "/workspace/TaxoGen/scores_imagereward_zeroshot.csv"
NO_DEF = True   # set False for with-definition prompts

if NO_DEF:
    OUT_CSV = "/workspace/TaxoGen/scores_imagereward_zeroshot_nodef.csv"

# Load model
print("Loading ImageReward model...")
model = RM.load("ImageReward-v1.0")
model.eval()

# Build prompts
meta = pd.read_csv(META_CSV)
meta["wordnet_id"] = meta["wordnet_id"].astype(str)
if NO_DEF:
    wid_to_prompt = {
        r["wordnet_id"]: f"An image of {r['core_synset']}"
        for _, r in meta.iterrows()
    }
else:
    wid_to_prompt = {
        r["wordnet_id"]: f"An image of {r['core_synset']} ({r['definition']})"
        for _, r in meta.iterrows()
    }

# Restrict to test wids
test = pd.read_csv(TEST_CSV)
test_wids = set(test["wordnet_id"].astype(str).tolist())
wid_to_prompt = {w: p for w, p in wid_to_prompt.items() if w in test_wids}
print(f"Test wids: {len(wid_to_prompt)}")

# Model dirs -> normalized model name (lowercase, same as scores_collected.csv)
model_dirs = sorted(
    d for d in os.listdir(IMAGES_ROOT)
    if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and not d.startswith(".")
    and d != "collection_summary.txt"
)
print(f"Model dirs: {model_dirs}")

rows = []
total = len(model_dirs) * len(wid_to_prompt)
done = 0

for d in model_dirs:
    model_name = d.lower()
    dir_path = os.path.join(IMAGES_ROOT, d)
    for wid, prompt in wid_to_prompt.items():
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(dir_path, wid + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        try:
            score = model.score(prompt, img_path)
            rows.append({"model": model_name, "wordnet_id": wid, "reward": float(score)})
        except Exception as e:
            print(f"  Error {d}/{wid}: {e}")
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{total} ({100*done/total:.1f}%)")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows -> {OUT_CSV}")

stats = df.groupby("model")["reward"].agg(["count", "median", "mean"])
print("\nPer-model stats:")
print(stats.sort_values("median").to_string())
