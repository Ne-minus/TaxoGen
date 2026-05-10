"""
Score all images in collected_wordnet_images/ using a CLIPRewardModel checkpoint.
Produces model, wordnet_id, reward CSV ready for fair_correlation.py.

Directory layout:
  images_root/
    {model_dir}/
      {wordnet_id}.png   (or .jpg / .jpeg)

Prompts come from splits/wordnet_meta.csv (wordnet_id, core_synset, definition).
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPRewardModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.0):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden = self.clip.config.projection_dim
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, 1),
        )

    def encode(self, input_ids, attention_mask, pixel_values):
        out = self.clip(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, return_dict=True)
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds,  dim=-1)
        fused = torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        return self.reward_head(fused).squeeze(-1)


class ScoringDataset(Dataset):
    def __init__(self, items, processor, max_length=77):
        self.items = items
        self.processor = processor
        self.max_length = max_length

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, model, wid, prompt = self.items[idx]
        img = Image.open(path).convert("RGB")
        enc = self.processor(text=prompt, images=img, return_tensors="pt",
                             padding="max_length", truncation=True,
                             max_length=self.max_length)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "model": model,
            "wid":   wid,
        }


def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
        "model":          [b["model"] for b in batch],
        "wid":            [b["wid"]   for b in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--images_root", default="collected_wordnet_images")
    ap.add_argument("--meta_csv",    default="splits/wordnet_meta.csv")
    ap.add_argument("--wids_filter", default=None,
                    help="File with one wordnet_id per line (to restrict to test split)")
    ap.add_argument("--out_csv",        default="scores_collected.csv")
    ap.add_argument("--no_definition",  action="store_true",
                    help="Use 'An image of {core_synset}' without definition")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint metadata
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "openai/clip-vit-large-patch14")
    print(f"Backbone: {model_name}  |  epoch: {ckpt.get('epoch')}  "
          f"|  val_acc: {ckpt.get('val_binary_acc', '?'):.4f}")

    # Prompts
    meta = pd.read_csv(args.meta_csv)
    meta["wordnet_id"] = meta["wordnet_id"].astype(str)
    if args.no_definition:
        wid_to_prompt = {
            r["wordnet_id"]: f"An image of {r['core_synset']}"
            for _, r in meta.iterrows()
        }
    else:
        wid_to_prompt = {
            r["wordnet_id"]: f"An image of {r['core_synset']} ({r['definition']})"
            for _, r in meta.iterrows()
        }

    if args.wids_filter:
        with open(args.wids_filter) as f:
            allowed = {l.strip() for l in f if l.strip()}
        wid_to_prompt = {w: p for w, p in wid_to_prompt.items() if w in allowed}
        print(f"Filtered to {len(wid_to_prompt)} wids")

    # Discover model dirs
    model_dirs = sorted(
        d for d in os.listdir(args.images_root)
        if os.path.isdir(os.path.join(args.images_root, d)) and not d.startswith(".")
    )
    print(f"Model dirs: {model_dirs}")

    # Build items list
    items, missing = [], []
    for d in model_dirs:
        for wid, prompt in wid_to_prompt.items():
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(args.images_root, d, wid + ext)
                if os.path.exists(p):
                    items.append((p, d.lower(), wid, prompt))
                    break
            else:
                missing.append((d, wid))

    print(f"Images to score: {len(items)}")
    if missing:
        print(f"Missing: {len(missing)} (model, wid) pairs")
        for d, w in missing[:5]:
            print(f"  {d} / {w}")

    # Load model
    scorer = CLIPRewardModel(model_name=model_name, dropout=0.0)
    scorer.load_state_dict(ckpt["model_state_dict"])
    scorer.to(device).eval()

    processor = CLIPProcessor.from_pretrained(model_name)

    ds = ScoringDataset(items, processor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            pix  = batch["pixel_values"].to(device, non_blocking=True)
            sc   = scorer.encode(ids, mask, pix).cpu().numpy()
            for m, w, s in zip(batch["model"], batch["wid"], sc):
                rows.append({"model": m, "wordnet_id": w, "reward": float(s)})
            if (i + 1) % 20 == 0:
                print(f"  {len(rows)} / {len(items)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved {len(df)} rows → {args.out_csv}")

    stats = df.groupby("model")["reward"].agg(["count", "median", "mean"])
    print("\nPer-model stats (sorted by median):")
    print(stats.sort_values("median").to_string())


if __name__ == "__main__":
    main()