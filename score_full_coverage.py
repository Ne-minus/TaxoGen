"""
For each test wordnet_id, score ALL available model images on disk
(not just those that appeared in test.csv pairs).
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import ImageReward as RM


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
    def __init__(self, dropout=0.0):
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


class ScoringDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, model, wid, prompt = self.items[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return {"image": img, "prompt": prompt, "model": model, "wid": wid}


def collate(batch):
    return {
        "image":  torch.stack([b["image"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "model":  [b["model"]  for b in batch],
        "wid":    [b["wid"]    for b in batch],
    }


def build_prompt(row):
    if pd.notna(row.get('core_synset')) and pd.notna(row.get('definition')):
        return f"An image of {row['core_synset']} ({row['definition']})"
    if pd.notna(row.get('definition')):
        return str(row['definition'])
    return str(row.get('prompt', ''))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--out_csv", default="test_scores_full.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test = pd.read_csv(args.test_csv)
    # Build wid -> prompt map (first occurrence)
    wid_to_prompt = {}
    for _, r in test.iterrows():
        wid = str(r['wordnet_id'])
        if wid not in wid_to_prompt:
            wid_to_prompt[wid] = build_prompt(r)
    test_wids = sorted(wid_to_prompt.keys())
    print(f"Test wids: {len(test_wids)}")

    # All model dirs
    dirs = [d for d in os.listdir(args.images_root)
            if os.path.isdir(os.path.join(args.images_root, d)) and not d.startswith('.')]
    print(f"Model dirs on disk: {len(dirs)}")

    items = []
    for d in dirs:
        for wid in test_wids:
            for ext in ('.png', '.jpg', '.jpeg'):
                p_ = os.path.join(args.images_root, d, wid + ext)
                if os.path.exists(p_):
                    items.append((p_, d, wid, wid_to_prompt[wid]))
                    break
    print(f"(model, wid) pairs to score: {len(items)}")

    print(f"Loading checkpoint: {args.ckpt}")
    model = ImageRewardFineTuner(dropout=0.0)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    ds = ScoringDataset(items, build_transform())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            imgs = batch["image"].to(device, non_blocking=True)
            sc = model.score_batch(imgs, batch["prompt"]).cpu().numpy()
            for m, w, s in zip(batch["model"], batch["wid"], sc):
                rows.append({"model": m, "wordnet_id": w, "reward": float(s)})
            if (i + 1) % 20 == 0:
                print(f"  scored {len(rows)} / {len(items)}")

    df = pd.DataFrame(rows)
    # Lowercase model name to match CSV
    df['model'] = df['model'].str.lower()
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")

    # Stats per model
    s = df.groupby('model')['reward'].agg(['count', 'median', 'mean'])
    print("\nPer-model stats:")
    print(s.sort_values('median').to_string())


if __name__ == "__main__":
    main()
