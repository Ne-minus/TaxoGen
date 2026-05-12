"""Boxplot of CLIP rewards per model, sorted by median descending."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NAME_MAP = {
    "black-forest-labs_flux.1-dev": "FLUX",
    "deepfloyd_if-i-xl-v1.0": "DeepFloyd",
    "kandinsky-community_kandinsky-3": "Kandinsky3",
    "pixart-alpha_pixart-sigma-xl-2-512-ms": "PixArt",
    "playgroundai_playground-v2.5-1024px-aesthetic": "Playground",
    "prompthero_openjourney": "Openjourney",
    "retrieval": "Retrieval",
    "runwayml_stable-diffusion-v1-5": "SD1.5",
    "stabilityai_sdxl-turbo": "SDXL-turbo",
    "stabilityai_stable-diffusion-3-medium-diffusers": "SD3",
    "stabilityai_stable-diffusion-xl-base-1.0": "SDXL",
    "tencent-hunyuan_hunyuandit-v1.2-diffusers": "HDiT",
}

df = pd.read_csv("scores_clip_test_full.csv")
df["short"] = df["model"].map(NAME_MAP).fillna(df["model"])

medians = df.groupby("short")["reward"].median().sort_values(ascending=False)
order = medians.index.tolist()
data = [df.loc[df["short"] == m, "reward"].values for m in order]

TEAL = "#4BC8C8"

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
plt.savefig("test_scores_boxplot_clip_clean.png", dpi=150)
print("Saved test_scores_boxplot_clip_clean.png")
print(medians.to_string())
