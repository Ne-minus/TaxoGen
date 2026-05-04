"""
Fine-tune ImageReward on pairs only, with multiple loss choices.

Loss types (via --loss):
  bt          : Bradley-Terry, -log σ(r_winner - r_loser)        [baseline, collapses]
  margin      : ReLU(margin - (r_winner - r_loser))              [non-saturating]
  margin_dec  : margin + λ_dec * (r_a + r_b)^2                   [+ reward-decorrelation]
  margin_var  : margin + λ_var * (1 - var(rewards in batch))     [+ encourage spread]
  infonce     : in-batch contrastive: winner > every loser in batch
  hybrid      : 0.5 * margin + 0.5 * bt                          [smooth + non-saturating]
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import ImageReward as RM
try:
    import wandb
except ImportError:
    wandb = None
from transformers import get_cosine_schedule_with_warmup


def normalize_label(x):
    """
    Note: in the source CSVs the convention is {0: A_win, 1: B_win}
    (verified by reconstructing ELO from pairwise data — matches published
    ELO chart only under this interpretation). We flip to internal
    {1: A_win, 0: B_win} so the training loss treats label==1 as 'A is winner'.
    """
    if pd.isna(x): return None
    try: x = int(x)
    except: return None
    if x == 0: return 1   # A_win  → internal label 1
    if x == 1: return 0   # B_win  → internal label 0
    return None


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


IMG_SIZE  = 224
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_transform(train: bool):
    if train:
        return T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class PairOnlyDataset(Dataset):
    def __init__(self, csv_path, images_root="", augment=False, position_swap=False):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.position_swap = position_swap
        self.transform = build_transform(train=augment)
        self.df["label_id"] = self.df["result_human_def"].apply(normalize_label)
        raw = len(self.df)
        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)
        print(f"[{os.path.basename(csv_path)}] kept {len(self.df)}/{raw} pair rows")
        self._dir_map: Dict[str, str] = {}
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
        if "definition" in self.df.columns and "core_synset" in self.df.columns:
            prompt = f"An image of {r['core_synset']} ({r['definition']})"
        elif "definition" in self.df.columns:
            prompt = str(r["definition"])
        else:
            prompt = str(r["prompt"])
        img_a = self.transform(Image.open(self._load(str(r["model_a"]), wid)).convert("RGB"))
        img_b = self.transform(Image.open(self._load(str(r["model_b"]), wid)).convert("RGB"))
        if self.position_swap and random.random() < 0.5:
            img_a, img_b = img_b, img_a
            label = 1 - label
        return {"prompt": prompt, "image_a": img_a, "image_b": img_b, "label": label}


@dataclass
class Batch:
    prompts: List[str]
    images_a: torch.Tensor
    images_b: torch.Tensor
    labels:   torch.Tensor


class Collator:
    def __call__(self, items):
        return Batch(
            prompts=[x["prompt"] for x in items],
            images_a=torch.stack([x["image_a"] for x in items]),
            images_b=torch.stack([x["image_b"] for x in items]),
            labels=torch.tensor([x["label"] for x in items], dtype=torch.long),
        )


class ImageRewardFineTuner(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.ir_model = RM.load("ImageReward-v1.0")
        self.register_buffer("rw_mean", torch.tensor(self.ir_model.mean))
        self.register_buffer("rw_std",  torch.tensor(self.ir_model.std))
        self.head_dropout = nn.Dropout(dropout)
        self.blip = self.ir_model.blip
        self.mlp  = self.ir_model.mlp

    def freeze_backbone(self):
        for p in self.blip.parameters(): p.requires_grad = False

    def unfreeze_top_layers(self, n=2):
        try:
            for blk in self.blip.visual_encoder.blocks[-n:]:
                for p in blk.parameters(): p.requires_grad = True
        except AttributeError: pass
        try:
            for layer in self.blip.text_encoder.encoder.layer[-n:]:
                for p in layer.parameters(): p.requires_grad = True
        except AttributeError: pass

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

    def forward(self, prompts, images_a, images_b):
        return {"reward_a": self.score_batch(images_a, prompts),
                "reward_b": self.score_batch(images_b, prompts)}


def compute_loss(reward_a, reward_b, labels, loss_type, *,
                 margin=0.5, lambda_dec=0.1, lambda_var=0.1,
                 lambda_bt=0.5, label_smoothing=0.0, infonce_temp=1.0):
    diff = reward_a - reward_b
    # signed_diff > 0 when prediction matches label
    signed_diff = torch.where(labels == 1, diff, -diff)

    if loss_type == "bt":
        s = label_smoothing
        loss = (-(1 - s) * F.logsigmoid(signed_diff) - s * F.logsigmoid(-signed_diff)).mean()

    elif loss_type == "margin":
        loss = F.relu(margin - signed_diff).mean()

    elif loss_type == "margin_dec":
        margin_l = F.relu(margin - signed_diff).mean()
        # decorrelation: pull mean reward toward 0, prevent both rewards drifting together
        decorr = ((reward_a + reward_b) / 2).pow(2).mean()
        loss = margin_l + lambda_dec * decorr

    elif loss_type == "margin_var":
        margin_l = F.relu(margin - signed_diff).mean()
        all_r = torch.cat([reward_a, reward_b], dim=0)
        var = all_r.var(unbiased=False)
        loss = margin_l + lambda_var * F.relu(1.0 - var)

    elif loss_type == "infonce":
        # winner of pair i must be higher than all losers in batch
        winner = torch.where(labels == 1, reward_a, reward_b)   # (B,)
        loser  = torch.where(labels == 1, reward_b, reward_a)   # (B,)
        # logits[i,j]: winner_i vs loser_j; diagonal = matched pair
        # cross-entropy with target = i (own pair)
        logits = (winner.unsqueeze(1) - loser.unsqueeze(0)) / infonce_temp
        targets = torch.arange(len(labels), device=labels.device)
        loss = F.cross_entropy(logits, targets)

    elif loss_type == "hybrid":
        s = label_smoothing
        bt = (-(1 - s) * F.logsigmoid(signed_diff) - s * F.logsigmoid(-signed_diff)).mean()
        m  = F.relu(margin - signed_diff).mean()
        loss = lambda_bt * bt + (1 - lambda_bt) * m

    else:
        raise ValueError(f"unknown loss: {loss_type}")

    with torch.no_grad():
        pred = (diff > 0).long()
        labs_np, pred_np = labels.cpu().numpy(), pred.cpu().numpy()
        acc  = (pred == labels).float().mean().item()
        wf1  = f1_score(labs_np, pred_np, average="weighted", labels=[0,1], zero_division=0)
        f1pc = f1_score(labs_np, pred_np, average=None,       labels=[0,1], zero_division=0)
        diff_corr = diff[pred == labels].abs().mean().item() if (pred == labels).any() else float("nan")
        diff_inc  = diff[pred != labels].abs().mean().item() if (pred != labels).any() else float("nan")
        all_r = torch.cat([reward_a, reward_b])

    return loss, {
        "loss": loss.item(), "binary_acc": acc, "weighted_f1": wf1,
        "f1_B_win": float(f1pc[0]), "f1_A_win": float(f1pc[1]),
        "mean_signed_diff": signed_diff.mean().item(),
        "mean_abs_diff":    diff.abs().mean().item(),
        "reward_std":       all_r.std().item(),
        "reward_mean":      all_r.mean().item(),
        "diff_correct":     diff_corr,
        "diff_incorrect":   diff_inc,
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None, train=True, **lk):
    model.train(train)
    keys = ["loss","binary_acc","weighted_f1","f1_B_win","f1_A_win",
            "mean_signed_diff","mean_abs_diff","reward_std","reward_mean",
            "diff_correct","diff_incorrect"]
    agg = {k: 0.0 for k in keys}; n = 0
    for batch in loader:
        ia = batch.images_a.to(device, non_blocking=True)
        ib = batch.images_b.to(device, non_blocking=True)
        lb = batch.labels.to(device)
        with torch.set_grad_enabled(train):
            out = model(prompts=batch.prompts, images_a=ia, images_b=ib)
            loss, m = compute_loss(out["reward_a"], out["reward_b"], lb, **lk)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler is not None: scheduler.step()
        for k in keys:
            v = m.get(k, float("nan"))
            if not math.isnan(v): agg[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True); p.add_argument("--val_csv", required=True)
    p.add_argument("--test_csv", default=None);   p.add_argument("--images_root", required=True)
    p.add_argument("--output_dir", default="./ir_loss_ckpt")
    p.add_argument("--loss", default="margin",
                   choices=["bt","margin","margin_dec","margin_var","infonce","hybrid"])
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--lambda_dec", type=float, default=0.1)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--lambda_bt",  type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--infonce_temp", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr_mult", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_top_layers", type=int, default=2)
    p.add_argument("--position_swap_aug", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--run_tag", default="run")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {args.run_tag} | loss={args.loss} | lr={args.lr} | margin={args.margin} | "
          f"unfreeze={args.unfreeze_top_layers} ===")
    print(f"Device: {device}")

    coll = Collator()
    def mk(path, aug, sh, swp):
        ds = PairOnlyDataset(path, args.images_root, augment=aug, position_swap=swp)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.num_workers, collate_fn=coll, pin_memory=True)

    tl = mk(args.train_csv, True,  True,  args.position_swap_aug)
    vl = mk(args.val_csv,   False, False, False)
    te = mk(args.test_csv,  False, False, False) if args.test_csv else None

    model = ImageRewardFineTuner(dropout=args.dropout)
    if args.freeze_backbone:
        model.freeze_backbone()
        if args.unfreeze_top_layers > 0:
            model.unfreeze_top_layers(args.unfreeze_top_layers)
    model.to(device)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {nt:,}")

    head_ids = {id(p) for p in model.mlp.parameters()} | {id(p) for p in model.head_dropout.parameters()}
    bp = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    hp = [p for p in model.mlp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": bp, "lr": args.lr},
         {"params": hp, "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay)

    total = len(tl) * args.epochs
    warm  = max(10, int(args.warmup_frac * total))
    sched = get_cosine_schedule_with_warmup(optimizer, warm, total)

    lk = dict(loss_type=args.loss, margin=args.margin,
              lambda_dec=args.lambda_dec, lambda_var=args.lambda_var,
              lambda_bt=args.lambda_bt, label_smoothing=args.label_smoothing,
              infonce_temp=args.infonce_temp)

    best, patience = -1.0, 0
    best_path = os.path.join(args.output_dir, f"best_{args.run_tag}.pt")

    for ep in range(1, args.epochs + 1):
        tm = run_epoch(model, tl, device, optimizer, sched, train=True,  **lk)
        vm = run_epoch(model, vl, device,                     train=False, **lk)
        print(f"\nEpoch {ep}/{args.epochs}")
        print("Train:", {k: round(v, 4) for k, v in tm.items()})
        print("Val:  ", {k: round(v, 4) for k, v in vm.items()})
        if vm["binary_acc"] > best:
            best = vm["binary_acc"]; patience = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": ep, "val_binary_acc": best, "args": vars(args)}, best_path)
            print(f"  Saved best (val_acc={best:.4f})")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.early_stopping_patience})")
            if patience >= args.early_stopping_patience:
                print("Early stopping."); break

    print("\nDone.")
    if te:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        tm = run_epoch(model, te, device, train=False, **lk)
        print(f"=== {args.run_tag} TEST ===")
        print({k: round(v, 4) for k, v in tm.items()})


if __name__ == "__main__":
    main()
