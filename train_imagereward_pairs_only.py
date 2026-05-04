"""
Fine-tune ImageReward only on pairwise preference data (A_win / B_win).

Tie (2) and BothBad (3) rows are dropped — the model learns a single scalar
reward and is trained with a plain Bradley-Terry log-loss on the remaining pairs.

Changes vs train_imagereward.py:
  - Dataset filters to label ∈ {0, 1} only
  - Loss = -log σ(r_winner − r_loser) with optional label smoothing
  - No anchor loss, no tie_margin, no lambda_tie, no tau_neg/tau_pos
  - Prediction is binary: A_win if reward_a > reward_b else B_win
  - Metrics: binary accuracy + binary F1

Usage:
    python train_imagereward_pairs_only.py \
        --train_csv train.csv --val_csv val.csv --test_csv test.csv \
        --images_root ./images \
        --freeze_backbone --unfreeze_top_layers 2 \
        --epochs 10 --batch_size 8 --lr 1e-5
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

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


# ─── Labels ──────────────────────────────────────────────────────────────────

ID_TO_LABEL = {0: "B_win", 1: "A_win"}  # only pairwise labels


def normalize_label(x):
    if pd.isna(x):
        return None
    try:
        x = int(x)
    except Exception:
        return None
    return x if x in [0, 1] else None   # drop Tie (2) and BothBad (3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    tmp = df.copy()
    tmp["label_id"] = tmp["result_human_def"].apply(normalize_label)
    counts = tmp["label_id"].value_counts().to_dict()
    return {v: int(counts.get(k, 0)) for k, v in ID_TO_LABEL.items()}


# ─── Transforms ──────────────────────────────────────────────────────────────

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
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    else:
        return T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PairOnlyDataset(Dataset):
    """
    Loads only rows where result_human_def ∈ {0 (B_win), 1 (A_win)}.
    Position-swap aug: with p=0.5 swaps A↔B and flips label 0↔1.
    """

    def __init__(self, csv_path: str, images_root: str = "",
                 augment: bool = False, position_swap: bool = False):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.augment = augment
        self.position_swap = position_swap
        self.transform = build_transform(train=augment)

        required = {"wordnet_id", "model_a", "model_b", "result_human_def"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        has_prompt = "definition" in self.df.columns or "prompt" in self.df.columns
        if not has_prompt:
            raise ValueError("Need 'definition' or 'prompt' column.")

        self.df["label_id"] = self.df["result_human_def"].apply(normalize_label)

        raw_total = len(self.df)
        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)
        kept = len(self.df)
        print(f"[{os.path.basename(csv_path)}] kept {kept}/{raw_total} rows "
              f"(dropped Tie/BothBad); label distribution: "
              f"{self.df['label_id'].value_counts().to_dict()}")

        if kept == 0:
            raise ValueError("No valid pair rows after filtering.")

        self._dir_map: Dict[str, str] = {}
        if os.path.isdir(images_root):
            for d in os.listdir(images_root):
                if os.path.isdir(os.path.join(images_root, d)):
                    self._dir_map[d.lower()] = d

    def __len__(self):
        return len(self.df)

    def _load_image_path(self, model_name: str, wordnet_id: str) -> str:
        actual_dir = self._dir_map.get(model_name.lower(), model_name)
        model_dir = os.path.join(self.images_root, actual_dir)
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(model_dir, f"{wordnet_id}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image: model={model_name}, id={wordnet_id}")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        wid   = str(row["wordnet_id"])
        label = int(row["label_id"])

        if "definition" in self.df.columns and "core_synset" in self.df.columns:
            prompt = f"An image of {row['core_synset']} ({row['definition']})"
        elif "definition" in self.df.columns:
            prompt = str(row["definition"])
        else:
            prompt = str(row["prompt"])

        img_a = Image.open(self._load_image_path(str(row["model_a"]), wid)).convert("RGB")
        img_b = Image.open(self._load_image_path(str(row["model_b"]), wid)).convert("RGB")

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        if self.position_swap and random.random() < 0.5:
            img_a, img_b = img_b, img_a
            label = 1 - label   # 0↔1

        return {"prompt": prompt, "image_a": img_a, "image_b": img_b, "label": label}


@dataclass
class Batch:
    prompts:  List[str]
    images_a: torch.Tensor
    images_b: torch.Tensor
    labels:   torch.Tensor


class Collator:
    def __call__(self, items: List[Dict]) -> Batch:
        return Batch(
            prompts=[x["prompt"] for x in items],
            images_a=torch.stack([x["image_a"] for x in items]),
            images_b=torch.stack([x["image_b"] for x in items]),
            labels=torch.tensor([x["label"] for x in items], dtype=torch.long),
        )


# ─── Model ───────────────────────────────────────────────────────────────────

class ImageRewardFineTuner(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.ir_model = RM.load("ImageReward-v1.0")
        self.register_buffer("rw_mean", torch.tensor(self.ir_model.mean))
        self.register_buffer("rw_std",  torch.tensor(self.ir_model.std))
        self.head_dropout = nn.Dropout(dropout)
        self.blip = self.ir_model.blip
        self.mlp  = self.ir_model.mlp

    def freeze_backbone(self):
        for p in self.blip.parameters():
            p.requires_grad = False
        print("BLIP backbone frozen.")

    def unfreeze_top_layers(self, num_layers: int = 2):
        try:
            vis_blocks = self.blip.visual_encoder.blocks
            for block in vis_blocks[-num_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            print(f"Unfrozen top {num_layers} visual encoder blocks.")
        except AttributeError:
            print("Warning: could not access visual_encoder.blocks")
        try:
            txt_layers = self.blip.text_encoder.encoder.layer
            for layer in txt_layers[-num_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"Unfrozen top {num_layers} text encoder layers.")
        except AttributeError:
            print("Warning: could not access text_encoder.encoder.layer")

    def score_batch(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        device = images.device
        text_input = self.blip.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        ).to(device)
        image_embeds = self.blip.visual_encoder(images)
        image_atts   = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        text_output  = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :]
        txt_features = self.head_dropout(txt_features)
        raw    = self.mlp(txt_features).squeeze(-1)
        reward = (raw - self.rw_mean) / self.rw_std
        return reward

    def forward(self, prompts, images_a, images_b):
        return {
            "reward_a": self.score_batch(images_a, prompts),
            "reward_b": self.score_batch(images_b, prompts),
        }


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    labels:   torch.Tensor,
    label_smoothing: float = 0.1,
) -> tuple:
    """
    Pure Bradley-Terry loss on pairs only.

    label=1 (A_win): loss = -log σ(r_a - r_b)
    label=0 (B_win): loss = -log σ(r_b - r_a)

    With label smoothing ε:
      loss = -(1-ε) log σ(diff) - ε log σ(-diff)
    """
    diff = reward_a - reward_b   # positive → A preferred
    s    = label_smoothing

    # label=1 → want diff > 0 → log σ(diff)
    # label=0 → want diff < 0 → log σ(-diff)
    signed_diff = torch.where(labels == 1, diff, -diff)
    loss = -(1 - s) * F.logsigmoid(signed_diff) - s * F.logsigmoid(-signed_diff)
    loss = loss.mean()

    with torch.no_grad():
        pred  = (diff > 0).long()          # 1=A_win, 0=B_win
        acc   = (pred == labels).float().mean().item()
        f1_b  = f1_score(labels.cpu().numpy(), pred.cpu().numpy(),
                         average=None, labels=[0, 1], zero_division=0)
        wf1   = f1_score(labels.cpu().numpy(), pred.cpu().numpy(),
                         average="weighted", labels=[0, 1], zero_division=0)
        mean_diff_correct   = diff[(pred == labels)].abs().mean().item() \
                              if (pred == labels).any() else float("nan")
        mean_diff_incorrect = diff[(pred != labels)].abs().mean().item() \
                              if (pred != labels).any() else float("nan")

    return loss, {
        "loss":                 loss.item(),
        "binary_acc":           acc,
        "weighted_f1":          wf1,
        "f1_B_win":             float(f1_b[0]),
        "f1_A_win":             float(f1_b[1]),
        "mean_reward_a":        reward_a.mean().item(),
        "mean_reward_b":        reward_b.mean().item(),
        "mean_diff_correct":    mean_diff_correct,
        "mean_diff_incorrect":  mean_diff_incorrect,
    }


# ─── Train / Eval loop ───────────────────────────────────────────────────────

def run_epoch(
    model, loader, device,
    optimizer=None, scheduler=None, train=True,
    label_smoothing=0.1,
):
    model.train(train)

    agg = {
        "loss": 0.0, "binary_acc": 0.0, "weighted_f1": 0.0,
        "f1_B_win": 0.0, "f1_A_win": 0.0,
        "mean_reward_a": 0.0, "mean_reward_b": 0.0,
        "mean_diff_correct": 0.0, "mean_diff_incorrect": 0.0,
    }
    n = 0

    for batch in loader:
        images_a = batch.images_a.to(device, non_blocking=True)
        images_b = batch.images_b.to(device, non_blocking=True)
        labels   = batch.labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(prompts=batch.prompts, images_a=images_a, images_b=images_b)
            loss, metrics = compute_loss(
                reward_a=outputs["reward_a"],
                reward_b=outputs["reward_b"],
                labels=labels,
                label_smoothing=label_smoothing,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        for k in agg:
            v = metrics.get(k, float("nan"))
            if not math.isnan(v):
                agg[k] += v
        n += 1

    return {k: v / max(n, 1) for k, v in agg.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",   required=True)
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--test_csv",    default=None)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir",  default="./ir_pairs_only_ckpt")
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--freeze_backbone",     action="store_true")
    parser.add_argument("--unfreeze_top_layers", type=int, default=2)
    parser.add_argument("--position_swap_aug",   action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  default="taxonomy-reward-ir-pairs-only")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.use_wandb and wandb is None:
        print("Warning: wandb not installed, disabling wandb logging.")
        args.use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))

    for split, path in [("Train", args.train_csv), ("Val", args.val_csv)]:
        print(f"{split}:", get_label_distribution(pd.read_csv(path)))
    if args.test_csv:
        print("Test:", get_label_distribution(pd.read_csv(args.test_csv)))

    collator = Collator()

    def make_loader(path, augment, shuffle, swap):
        ds = PairOnlyDataset(
            path, images_root=args.images_root,
            augment=augment, position_swap=swap,
        )
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
        )

    train_loader = make_loader(args.train_csv, augment=True,  shuffle=True,
                               swap=args.position_swap_aug)
    val_loader   = make_loader(args.val_csv,   augment=False, shuffle=False, swap=False)
    test_loader  = make_loader(args.test_csv,  augment=False, shuffle=False, swap=False) \
                   if args.test_csv else None

    model = ImageRewardFineTuner(dropout=args.dropout)

    if args.freeze_backbone:
        model.freeze_backbone()
        if args.unfreeze_top_layers > 0:
            model.unfreeze_top_layers(num_layers=args.unfreeze_top_layers)

    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    if args.use_wandb:
        wandb.config.update(
            {"num_trainable_params": n_trainable, "num_total_params": n_total},
            allow_val_change=True,
        )

    head_ids = {id(p) for p in model.mlp.parameters()} | \
               {id(p) for p in model.head_dropout.parameters()}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    head_params     = [p for p in model.mlp.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * 10},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_acc = -1.0
    best_path    = os.path.join(args.output_dir, "best_model.pt")
    patience     = 0

    epoch_kwargs = dict(device=device, label_smoothing=args.label_smoothing)

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader,
                            optimizer=optimizer, scheduler=scheduler, train=True,
                            **epoch_kwargs)
        val_m   = run_epoch(model, val_loader, train=False, **epoch_kwargs)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", {k: round(v, 4) for k, v in train_m.items()})
        print("Val:  ", {k: round(v, 4) for k, v in val_m.items()})

        if args.use_wandb:
            wandb.log({
                "epoch":       epoch,
                "lr_head":     optimizer.param_groups[1]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}":   v for k, v in val_m.items()},
            }, step=epoch)

        monitor = val_m["binary_acc"]
        if monitor > best_val_acc:
            best_val_acc = monitor
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_binary_acc":   best_val_acc,
                "args":             vars(args),
            }, best_path)
            print(f"  Saved best (val binary_acc={best_val_acc:.4f})")
            if args.use_wandb:
                wandb.summary["best_val_binary_acc"] = best_val_acc
                wandb.summary["best_epoch"] = epoch
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.early_stopping_patience})")
            if patience >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nTraining done.")

    if test_loader:
        print("Loading best model for test...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = run_epoch(model, test_loader, train=False, **epoch_kwargs)
        print("Test:", {k: round(v, 4) for k, v in test_m.items()})
        if args.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_m.items()})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
