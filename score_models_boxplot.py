"""
Score every (model, wordnet_id) image in test.csv using the trained ImageReward
checkpoint, then plot a boxplot of rewards per generative model, sorted by median.
"""

import os
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import ImageReward as RM
import matplotlib.pyplot as plt


IMG_SIZE  = 224
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_transform():
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class ImageRewardFineTuner(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.ir_model = RM.load("ImageReward-v1.0")
        self.register_buffer("rw_mean", torch.tensor(self.ir_model.mean))
        self.register_buffer("rw_std",  torch.tensor(self.ir_model.std))
        self.head_dropout = nn.Dropout(dropout)
        self.blip = self.ir_model.blip
        self.mlp  = self.ir_model.mlp

    def score_batch(self, images, prompts):
        device = images.device
        ti = self.blip.tokenizer(prompts, padding="max_length", truncation=True,
                                 max_length=35, return_tensors="pt").to(device)
        emb = self.blip.visual_encoder(images)
        atts = torch.ones(emb.size()[:-1], dtype=torch.long, device=device)
        out = self.blip.text_encoder(ti.input_ids, attention_mask=ti.attention_mask,
                                     encoder_hidden_states=emb, encoder_attention_mask=atts,
                                     return_dict=True)
        feats = self.head_dropout(out.last_hidden_state[:, 0, :])
        raw = self.mlp(feats).squeeze(-1)
        return (raw - self.rw_mean) / self.rw_std


class ImageScoringDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str, str, str]], images_root: str):
        self.items = items
        self.images_root = images_root
        self.transform = build_transform()
        self._dir_map: Dict[str, str] = {}
        if os.path.isdir(images_root):
            for d in os.listdir(images_root):
                if os.path.isdir(os.path.join(images_root, d)):
                    self._dir_map[d.lower()] = d

    def __len__(self): return len(self.items)

    def _load_path(self, model: str, wid: str) -> str:
        actual = self._dir_map.get(model.lower(), model)
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.images_root, actual, f"{wid}{ext}")
            if os.path.exists(p): return p
        return None

    def __getitem__(self, idx):
        model, wid, prompt, _ = self.items[idx]
        path = self._load_path(model, wid)
        if path is None:
            return None
        img = self.transform(Image.open(path).convert("RGB"))
        return {"image": img, "prompt": prompt, "model": model, "wid": wid}


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {
        "image":  torch.stack([b["image"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "model":  [b["model"]  for b in batch],
        "wid":    [b["wid"]    for b in batch],
    }


def build_prompt(row):
    if "definition" in row.index and "core_synset" in row.index and pd.notna(row.get("core_synset")) and pd.notna(row.get("definition")):
        return f"An image of {row['core_synset']} ({row['definition']})"
    if "definition" in row.index and pd.notna(row.get("definition")):
        return str(row["definition"])
    return str(row["prompt"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--out_csv",     default="test_scores.csv")
    p.add_argument("--out_plot",    default="test_scores_boxplot.png")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build (model, wid) → prompt deduped list ──
    df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(df)} test rows")

    seen = {}    # (model, wid) → prompt
    for _, r in df.iterrows():
        wid = str(r["wordnet_id"])
        prompt = build_prompt(r)
        for m_col in ("model_a", "model_b"):
            key = (str(r[m_col]), wid)
            if key not in seen:
                seen[key] = prompt
    items = [(m, w, p, "") for (m, w), p in seen.items()]
    print(f"Unique (model, image) pairs to score: {len(items)}")

    # ── Load model ──
    print(f"Loading checkpoint: {args.ckpt}")
    model = ImageRewardFineTuner(dropout=0.0)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # ── Score ──
    ds = ImageScoringDataset(items, args.images_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_skip_none,
                    pin_memory=True)

    rows = []
    skipped = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if batch is None:
                continue
            imgs = batch["image"].to(device, non_blocking=True)
            scores = model.score_batch(imgs, batch["prompt"]).cpu().numpy()
            for m, w, s in zip(batch["model"], batch["wid"], scores):
                rows.append({"model": m, "wordnet_id": w, "reward": float(s)})
            if (i + 1) % 50 == 0:
                print(f"  scored {len(rows)} images")

    expected = len(items)
    skipped = expected - len(rows)
    print(f"Scored {len(rows)} / {expected} images (skipped {skipped} missing files)")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved scores to {args.out_csv}")

    # ── Stats per model ──
    stats = out.groupby("model")["reward"].agg(["count", "median", "mean", "std"])
    stats = stats.sort_values("median")
    print("\nPer-model stats (sorted by median):")
    print(stats.to_string())

    # ── Boxplot ──
    order = stats.index.tolist()
    data  = [out.loc[out["model"] == m, "reward"].values for m in order]

    short_names = [m.split("_", 1)[1] if "_" in m else m for m in order]
    short_names = [n if len(n) < 30 else n[:27] + "..." for n in short_names]

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.0), 6))
    bp = ax.boxplot(data, labels=short_names, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(order)))
    for patch, c in zip(bp["boxes"], cmap):
        patch.set_facecolor(c); patch.set_alpha(0.7)

    ax.set_ylabel("Predicted reward")
    ax.set_xlabel("Generative model (sorted by median, low → high)")
    ax.set_title(f"Per-model reward distribution on test set\n(checkpoint: {os.path.basename(args.ckpt)})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
