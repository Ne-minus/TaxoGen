"""
Resplit train/val/test so that all rows with the same wordnet_id
land in the same split, while preserving per-subset proportions.

Target ratios: train=0.75, val=0.10, test=0.15  (same as original).
"""

import pandas as pd
import numpy as np

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
# TEST_RATIO = 0.15  (remainder)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
val = pd.read_csv("val.csv")

all_df = pd.concat(
    [train.assign(split="train"), test.assign(split="test"), val.assign(split="val")],
    ignore_index=True,
)

# One row per wordnet_id with its subset label (every WN ID belongs to exactly one subset)
wn_meta = all_df.drop_duplicates("wordnet_id")[["wordnet_id", "subset"]].reset_index(drop=True)

rng = np.random.default_rng(SEED)

new_split_map = {}  # wordnet_id -> split

for subset, group in wn_meta.groupby("subset"):
    ids = group["wordnet_id"].values.copy()
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    # remainder goes to test
    for wid in ids[:n_train]:
        new_split_map[wid] = "train"
    for wid in ids[n_train : n_train + n_val]:
        new_split_map[wid] = "val"
    for wid in ids[n_train + n_val :]:
        new_split_map[wid] = "test"

all_df["new_split"] = all_df["wordnet_id"].map(new_split_map)

# Verify no leakage
for a, b in [("train", "test"), ("train", "val"), ("test", "val")]:
    ids_a = set(all_df.loc[all_df["new_split"] == a, "wordnet_id"])
    ids_b = set(all_df.loc[all_df["new_split"] == b, "wordnet_id"])
    overlap = ids_a & ids_b
    assert len(overlap) == 0, f"Leakage between {a} and {b}: {overlap}"

print("No leakage — OK")

# Save
drop_cols = ["split", "new_split"]
for split_name in ("train", "val", "test"):
    split_df = all_df[all_df["new_split"] == split_name].drop(columns=drop_cols)
    split_df.to_csv(f"{split_name}.csv", index=False)
    print(f"{split_name}: {len(split_df)} rows, {split_df['wordnet_id'].nunique()} unique WN IDs")

# Print new distribution
print("\nNew subset distribution:")
print(all_df.groupby(["subset", "new_split"]).size().unstack(fill_value=0))

print("\nNew overall proportions:")
counts = all_df["new_split"].value_counts()
total = len(all_df)
for s in ("train", "val", "test"):
    print(f"  {s}: {counts[s]} ({counts[s]/total:.3f})")
