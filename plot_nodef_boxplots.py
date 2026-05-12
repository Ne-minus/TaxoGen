"""
Generate two boxplots for nodef CLIP model in the same style as plot_boxplot_clip.py:
  1. Model reward distribution, sorted by median
  2. Reward diff (A−B) by human label category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TEAL = "#4BC8C8"

NAME_MAP = {
    "black-forest-labs_flux.1-dev":                        "FLUX",
    "deepfloyd_if-i-xl-v1.0":                             "DeepFloyd",
    "kandinsky-community_kandinsky-3":                     "Kandinsky3",
    "pixart-alpha_pixart-sigma-xl-2-512-ms":               "PixArt",
    "playgroundai_playground-v2.5-1024px-aesthetic":       "Playground",
    "prompthero_openjourney":                              "Openjourney",
    "retrieval":                                           "Retrieval",
    "runwayml_stable-diffusion-v1-5":                      "SD1.5",
    "stabilityai_sdxl-turbo":                              "SDXL-turbo",
    "stabilityai_stable-diffusion-3-medium-diffusers":     "SD3",
    "stabilityai_stable-diffusion-xl-base-1.0":            "SDXL",
    "tencent-hunyuan_hunyuandit-v1.2-diffusers":           "HDiT",
}

LABEL_NAMES = {0: "B_win", 1: "A_win", 2: "Tie", 3: "BothBad"}

# ── Load scores ───────────────────────────────────────────────────────────────
scores = pd.read_csv("scores_clip_nodef_test_full.csv")
scores["model"] = scores["model"].str.lower().str.strip()
scores["wordnet_id"] = scores["wordnet_id"].astype(str)
scores["short"] = scores["model"].map(NAME_MAP).fillna(scores["model"])

# ── Plot 1: model reward boxplot ──────────────────────────────────────────────
medians = scores.groupby("short")["reward"].median().sort_values(ascending=False)
order = medians.index.tolist()
data = [scores.loc[scores["short"] == m, "reward"].values for m in order]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(
    data,
    tick_labels=order,
    showfliers=False,
    showmeans=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    meanprops=dict(marker="o", markerfacecolor="black",
                   markeredgecolor="black", markersize=5),
    whiskerprops=dict(color="black", linewidth=1),
    capprops=dict(color="black", linewidth=1),
    boxprops=dict(linewidth=1),
)
for patch in bp["boxes"]:
    patch.set_facecolor(TEAL)
    patch.set_alpha(0.85)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_ylabel("Reward", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("plots/nodef_boxplot_models.png", dpi=150)
print("Saved plots/nodef_boxplot_models.png")
print(medians.to_string())

# ── Plot 2: reward diff by label ──────────────────────────────────────────────
score_map = scores.set_index(["model", "wordnet_id"])["reward"].to_dict()
test_df = pd.read_csv("splits/test.csv")

rows = []
for _, row in test_df.iterrows():
    v = row.get("result_human_def")
    if pd.isna(v):
        continue
    label = int(v)
    if label not in LABEL_NAMES:
        continue
    wid = str(row["wordnet_id"])
    ra = score_map.get((str(row["model_a"]).lower(), wid), np.nan)
    rb = score_map.get((str(row["model_b"]).lower(), wid), np.nan)
    if np.isnan(ra) or np.isnan(rb):
        continue
    rows.append({"label_name": LABEL_NAMES[label], "reward": ra})
    rows.append({"label_name": LABEL_NAMES[label], "reward": rb})

pair_df = pd.DataFrame(rows)
print(f"\nReward counts per label:\n{pair_df['label_name'].value_counts().to_string()}")

label_order = ["Tie", "BothBad"]
present = [l for l in label_order if l in pair_df["label_name"].values]
data2 = [pair_df.loc[pair_df["label_name"] == l, "reward"].values for l in present]
tick_labels = [f"{l}\n(n={len(d)//2} pairs)" for l, d in zip(present, data2)]

fig2, ax2 = plt.subplots(figsize=(9, 6))
bp2 = ax2.boxplot(
    data2,
    tick_labels=tick_labels,
    showfliers=False,
    showmeans=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    meanprops=dict(marker="o", markerfacecolor="black",
                   markeredgecolor="black", markersize=5),
    whiskerprops=dict(color="black", linewidth=1),
    capprops=dict(color="black", linewidth=1),
    boxprops=dict(linewidth=1),
)
for patch in bp2["boxes"]:
    patch.set_facecolor(TEAL)
    patch.set_alpha(0.85)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.set_ylabel("Reward", fontsize=12)
ax2.set_xlabel("Human label", fontsize=12)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/nodef_boxplot_tie_bb.png", dpi=150)
print("Saved plots/nodef_boxplot_tie_bb.png")
