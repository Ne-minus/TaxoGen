"""
Compare untouched ImageReward-v1.0 vs the fine-tuned best checkpoint
on the pairwise test set. Reports binary accuracy and F1 (per-class + weighted).
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

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
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def normalize_label(x):
    if pd.isna(x): return None
    try: x = int(x)
    except: return None
    if x == 0: return 1   # A_win
    if x == 1: return 0   # B_win
    return None


class ImageRewardScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ir_model = RM.load("ImageReward-v1.0")
        self.register_buffer("rw_mean", torch.tensor(self.ir_model.mean))
        self.register_buffer("rw_std",  torch.tensor(self.ir_model.std))
        self.head_dropout = nn.Dropout(0.0)
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


class PairTestDataset(Dataset):
    def __init__(self, csv_path, images_root):
        df = pd.read_csv(csv_path)
        df["label_id"] = df["result_human_def"].apply(normalize_label)
        before = len(df)
        df = df[df["label_id"].notna()].reset_index(drop=True)
        print(f"[{os.path.basename(csv_path)}] kept {len(df)}/{before} pair rows")
        self.df = df
        self.images_root = images_root
        self.transform = build_transform()
        self._dir_map = {}
        if os.path.isdir(images_root):
            for d in os.listdir(images_root):
                if os.path.isdir(os.path.join(images_root, d)):
                    self._dir_map[d.lower()] = d

    def __len__(self): return len(self.df)

    def _load(self, model, wid):
        actual = self._dir_map.get(model.lower(), model)
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.images_root, actual, f"{wid}{ext}")
            if os.path.exists(p): return p
        raise FileNotFoundError(f"{model}/{wid}")

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wid, label = str(r["wordnet_id"]), int(r["label_id"])
        if "core_synset" in self.df.columns and "definition" in self.df.columns \
                and pd.notna(r.get("core_synset")) and pd.notna(r.get("definition")):
            prompt = f"An image of {r['core_synset']} ({r['definition']})"
        elif "definition" in self.df.columns and pd.notna(r.get("definition")):
            prompt = str(r["definition"])
        else:
            prompt = str(r.get("prompt", ""))
        img_a = self.transform(Image.open(self._load(str(r["model_a"]), wid)).convert("RGB"))
        img_b = self.transform(Image.open(self._load(str(r["model_b"]), wid)).convert("RGB"))
        return {"prompt": prompt, "image_a": img_a, "image_b": img_b, "label": label}


def collate(batch):
    return {
        "prompts":  [b["prompt"] for b in batch],
        "images_a": torch.stack([b["image_a"] for b in batch]),
        "images_b": torch.stack([b["image_b"] for b in batch]),
        "labels":   torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ia = batch["images_a"].to(device, non_blocking=True)
            ib = batch["images_b"].to(device, non_blocking=True)
            ra = model.score_batch(ia, batch["prompts"])
            rb = model.score_batch(ib, batch["prompts"])
            pred = (ra > rb).long().cpu().numpy()
            all_preds.append(pred)
            all_labels.append(batch["labels"].numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc  = (preds == labels).mean()
    f1_w = f1_score(labels, preds, average="weighted", labels=[0, 1], zero_division=0)
    f1_each = f1_score(labels, preds, average=None,    labels=[0, 1], zero_division=0)
    return {
        "binary_acc": float(acc),
        "weighted_f1": float(f1_w),
        "f1_B_win": float(f1_each[0]),
        "f1_A_win": float(f1_each[1]),
        "n": int(len(labels)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="fine-tuned checkpoint .pt")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = PairTestDataset(args.test_csv, args.images_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    print("\n=== Baseline: untouched ImageReward-v1.0 ===")
    base = ImageRewardScorer().to(device)
    base_metrics = evaluate(base, dl, device)
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    del base
    torch.cuda.empty_cache()

    print(f"\n=== Fine-tuned: {os.path.basename(args.ckpt)} ===")
    ft = ImageRewardScorer()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ft.load_state_dict(ckpt["model_state_dict"])
    ft.to(device)
    ft_metrics = evaluate(ft, dl, device)
    for k, v in ft_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Diff (fine-tuned − baseline) ===")
    for k in ["binary_acc", "weighted_f1", "f1_B_win", "f1_A_win"]:
        d = ft_metrics[k] - base_metrics[k]
        sign = "+" if d >= 0 else ""
        print(f"  {k}: {sign}{d:.4f}")


if __name__ == "__main__":
    main()
