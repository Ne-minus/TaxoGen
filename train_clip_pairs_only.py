"""
Fine-tune CLIP-based reward model on pairwise preference data (pairs only).

Tie (2) and BothBad (3) rows are dropped — model learns a scalar reward and
is trained with one of several contrastive/ranking losses on the remaining pairs.

Loss types (--loss):
  bt          : Bradley-Terry, -log σ(r_winner - r_loser)
  margin      : ReLU(margin - (r_winner - r_loser))
  margin_dec  : margin + λ_dec * (mean_reward)^2         [pull rewards toward 0]
  margin_var  : margin + λ_var * ReLU(1 - var(rewards))  [encourage reward spread]
  infonce     : in-batch contrastive: winner_i must beat all losers in batch
  hybrid      : λ_bt * bt + (1 - λ_bt) * margin

CLIP versions (--model_name):
  openai/clip-vit-base-patch32      (default, 512-dim projection)
  openai/clip-vit-base-patch16      (512-dim, finer patches)
  openai/clip-vit-large-patch14     (768-dim, stronger visual encoder)
  openai/clip-vit-large-patch14-336 (768-dim, 336px input — larger batch cost)

Usage:
    python train_clip_pairs_only.py \
        --train_csv train.csv --val_csv val.csv --test_csv test.csv \
        --images_root ./images \
        --model_name openai/clip-vit-large-patch14 \
        --loss margin --margin 0.5 \
        --freeze_backbone --unfreeze_top_vision_layers 2 --unfreeze_top_text_layers 2 \
        --epochs 10 --batch_size 16 --lr 3e-5 \
        --run_tag clip_large_margin
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

try:
    import wandb
except ImportError:
    wandb = None
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


# ─── Labels ──────────────────────────────────────────────────────────────────

# CSV convention (arena_elo.py): 0=A_win, 1=B_win, 2=Tie, 3=BothBad
# Loss expects:                  1=A_win, 0=B_win  → flip 0↔1 on load

def normalize_label(x):
    if pd.isna(x):
        return None
    try:
        x = int(x)
    except Exception:
        return None
    if x == 0: return 1  # arena 0=A_win → loss 1=A_win
    if x == 1: return 0  # arena 1=B_win → loss 0=B_win
    return None           # drop Tie (2) and BothBad (3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PairOnlyDataset(Dataset):
    """
    Loads only rows where result_human_def ∈ {0 (B_win), 1 (A_win)}.
    CLIPProcessor runs inside __getitem__ so DataLoader workers do preprocessing
    in parallel — keeps GPU fed and avoids CPU bottleneck in the main loop.
    Position-swap aug: with p=0.5 swaps A↔B and flips label 0↔1.
    """

    def __init__(self, csv_path: str, images_root: str = "",
                 processor: "CLIPProcessor" = None,
                 augment: bool = False, position_swap: bool = False):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.processor = processor
        self.augment = augment
        self.position_swap = position_swap

        required = {"wordnet_id", "model_a", "model_b", "result_human_def"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if "definition" not in self.df.columns and "prompt" not in self.df.columns:
            raise ValueError("Need 'definition' or 'prompt' column.")

        self.df["label_id"] = self.df["result_human_def"].apply(normalize_label)
        raw = len(self.df)
        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)
        kept = len(self.df)
        print(f"[{os.path.basename(csv_path)}] kept {kept}/{raw} pair rows "
              f"(dropped Tie/BothBad); distribution: "
              f"{self.df['label_id'].value_counts().to_dict()}")
        if kept == 0:
            raise ValueError("No valid pair rows after filtering.")

        if self.augment:
            from torchvision import transforms as T
            self._aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.05),
            ])
        else:
            self._aug = None

        self._dir_map: Dict[str, str] = {}
        if os.path.isdir(images_root):
            for d in os.listdir(images_root):
                if os.path.isdir(os.path.join(images_root, d)):
                    self._dir_map[d.lower()] = d

    def __len__(self):
        return len(self.df)

    def _load_path(self, model_name: str, wordnet_id: str) -> str:
        actual = self._dir_map.get(model_name.lower(), model_name)
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.images_root, actual, f"{wordnet_id}{ext}")
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No image: model={model_name}, id={wordnet_id}")

    def _load_image(self, model_name: str, wordnet_id: str) -> Image.Image:
        img = Image.open(self._load_path(model_name, wordnet_id)).convert("RGB")
        if self._aug is not None:
            img = self._aug(img)
        return img

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

        img_a = self._load_image(str(row["model_a"]), wid)
        img_b = self._load_image(str(row["model_b"]), wid)

        if self.position_swap and random.random() < 0.5:
            img_a, img_b = img_b, img_a
            label = 1 - label

        # Run CLIPProcessor here (in DataLoader worker) so GPU stays fed
        enc_a = self.processor(text=prompt, images=img_a, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=77)
        enc_b = self.processor(text=prompt, images=img_b, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=77)

        return {
            "input_ids_a":      enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "pixel_values_a":   enc_a["pixel_values"].squeeze(0),
            "input_ids_b":      enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "pixel_values_b":   enc_b["pixel_values"].squeeze(0),
            "label":            label,
        }


@dataclass
class Batch:
    input_ids_a:      torch.Tensor
    attention_mask_a: torch.Tensor
    pixel_values_a:   torch.Tensor
    input_ids_b:      torch.Tensor
    attention_mask_b: torch.Tensor
    pixel_values_b:   torch.Tensor
    labels:           torch.Tensor


class Collator:
    def __call__(self, items: List[Dict]) -> Batch:
        return Batch(
            input_ids_a      = torch.stack([x["input_ids_a"]      for x in items]),
            attention_mask_a = torch.stack([x["attention_mask_a"] for x in items]),
            pixel_values_a   = torch.stack([x["pixel_values_a"]   for x in items]),
            input_ids_b      = torch.stack([x["input_ids_b"]      for x in items]),
            attention_mask_b = torch.stack([x["attention_mask_b"] for x in items]),
            pixel_values_b   = torch.stack([x["pixel_values_b"]   for x in items]),
            labels           = torch.tensor([x["label"] for x in items], dtype=torch.long),
        )


# ─── Model ───────────────────────────────────────────────────────────────────

class CLIPRewardModel(nn.Module):
    """
    CLIP backbone + scalar reward head.
    Fused feature: [img_emb, txt_emb, img*txt, |img-txt|]  → (projection_dim * 4,)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 dropout: float = 0.3):
        super().__init__()
        self.clip     = CLIPModel.from_pretrained(model_name)
        hidden        = self.clip.config.projection_dim
        fused_dim     = hidden * 4

        self.reward_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, num_text: int = 0, num_vision: int = 0) -> None:
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = True
        for p in self.clip.text_projection.parameters():
            p.requires_grad = True
        if num_text > 0 and hasattr(self.clip.text_model.encoder, "layers"):
            for layer in self.clip.text_model.encoder.layers[-num_text:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if num_vision > 0 and hasattr(self.clip.vision_model.encoder, "layers"):
            for layer in self.clip.vision_model.encoder.layers[-num_vision:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def apply_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                   target_modules: List[str] = None) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Install peft: pip install peft")
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, lora_cfg)
        # projections оставляем обучаемыми
        for p in self.clip.base_model.model.visual_projection.parameters():
            p.requires_grad = True
        for p in self.clip.base_model.model.text_projection.parameters():
            p.requires_grad = True

    def encode(self, input_ids, attention_mask, pixel_values) -> torch.Tensor:
        out = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds,  dim=-1)
        fused = torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        return self.reward_head(fused).squeeze(-1)

    def forward(self, input_ids_a, attention_mask_a, pixel_values_a,
                      input_ids_b, attention_mask_b, pixel_values_b):
        reward_a = self.encode(input_ids_a, attention_mask_a, pixel_values_a)
        reward_b = self.encode(input_ids_b, attention_mask_b, pixel_values_b)
        return {"reward_a": reward_a, "reward_b": reward_b}


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    labels:   torch.Tensor,
    loss_type: str,
    *,
    margin: float       = 0.5,
    lambda_dec: float   = 0.1,
    lambda_var: float   = 0.1,
    lambda_bt: float    = 0.5,
    label_smoothing: float = 0.0,
    infonce_temp: float = 1.0,
):
    """
    labels: 0=B_win, 1=A_win
    diff = reward_a - reward_b: positive → A preferred, negative → B preferred
    signed_diff = diff when A_win, -diff when B_win  →  >0 when prediction correct
    """
    diff        = reward_a - reward_b
    signed_diff = torch.where(labels == 1, diff, -diff)

    if loss_type == "bt":
        s    = label_smoothing
        loss = (-(1 - s) * F.logsigmoid(signed_diff)
                - s       * F.logsigmoid(-signed_diff)).mean()

    elif loss_type == "margin":
        loss = F.relu(margin - signed_diff).mean()

    elif loss_type == "margin_dec":
        margin_l = F.relu(margin - signed_diff).mean()
        decorr   = ((reward_a + reward_b) / 2).pow(2).mean()
        loss     = margin_l + lambda_dec * decorr

    elif loss_type == "margin_var":
        margin_l = F.relu(margin - signed_diff).mean()
        all_r    = torch.cat([reward_a, reward_b])
        var      = all_r.var(unbiased=False)
        loss     = margin_l + lambda_var * F.relu(1.0 - var)

    elif loss_type == "infonce":
        winner = torch.where(labels == 1, reward_a, reward_b)
        loser  = torch.where(labels == 1, reward_b, reward_a)
        # logits[i,j] = winner_i - loser_j; target = diagonal (own pair)
        logits  = (winner.unsqueeze(1) - loser.unsqueeze(0)) / infonce_temp
        targets = torch.arange(len(labels), device=labels.device)
        loss    = F.cross_entropy(logits, targets)

    elif loss_type == "hybrid":
        s      = label_smoothing
        bt_l   = (-(1 - s) * F.logsigmoid(signed_diff)
                  - s       * F.logsigmoid(-signed_diff)).mean()
        margin_l = F.relu(margin - signed_diff).mean()
        loss     = lambda_bt * bt_l + (1 - lambda_bt) * margin_l

    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}")

    with torch.no_grad():
        pred     = (diff > 0).long()
        labs_np  = labels.cpu().numpy()
        pred_np  = pred.cpu().numpy()
        acc      = (pred == labels).float().mean().item()
        wf1      = f1_score(labs_np, pred_np, average="weighted",
                            labels=[0, 1], zero_division=0)
        f1pc     = f1_score(labs_np, pred_np, average=None,
                            labels=[0, 1], zero_division=0)
        all_r    = torch.cat([reward_a, reward_b])
        d_corr   = diff[pred == labels].abs().mean().item() \
                   if (pred == labels).any() else float("nan")
        d_incorr = diff[pred != labels].abs().mean().item() \
                   if (pred != labels).any() else float("nan")

    return loss, {
        "loss":             loss.item(),
        "binary_acc":       acc,
        "weighted_f1":      wf1,
        "f1_B_win":         float(f1pc[0]),
        "f1_A_win":         float(f1pc[1]),
        "mean_signed_diff": signed_diff.mean().item(),
        "mean_abs_diff":    diff.abs().mean().item(),
        "reward_mean":      all_r.mean().item(),
        "reward_std":       all_r.std().item(),
        "diff_correct":     d_corr,
        "diff_incorrect":   d_incorr,
    }


# ─── Train / Eval loop ───────────────────────────────────────────────────────

METRIC_KEYS = [
    "loss", "binary_acc", "weighted_f1", "f1_B_win", "f1_A_win",
    "mean_signed_diff", "mean_abs_diff", "reward_mean", "reward_std",
    "diff_correct", "diff_incorrect",
]


def run_epoch(model, loader, device,
              optimizer=None, scheduler=None, train=True, **loss_kwargs):
    model.train(train)
    agg = {k: 0.0 for k in METRIC_KEYS}
    n   = 0

    for batch in loader:
        # All preprocessing already done in DataLoader workers
        ia = batch.input_ids_a.to(device, non_blocking=True)
        ma = batch.attention_mask_a.to(device, non_blocking=True)
        pa = batch.pixel_values_a.to(device, non_blocking=True)
        ib = batch.input_ids_b.to(device, non_blocking=True)
        mb = batch.attention_mask_b.to(device, non_blocking=True)
        pb = batch.pixel_values_b.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            out = model(
                input_ids_a=ia, attention_mask_a=ma, pixel_values_a=pa,
                input_ids_b=ib, attention_mask_b=mb, pixel_values_b=pb,
            )
            loss, metrics = compute_loss(
                out["reward_a"], out["reward_b"], labels, **loss_kwargs,
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

        for k in METRIC_KEYS:
            v = metrics.get(k, float("nan"))
            if not math.isnan(v):
                agg[k] += v
        n += 1

    return {k: v / max(n, 1) for k, v in agg.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP reward model on pairs only, multiple loss choices."
    )
    # Data
    p.add_argument("--train_csv",   required=True)
    p.add_argument("--val_csv",     required=True)
    p.add_argument("--test_csv",    default=None)
    p.add_argument("--images_root", required=True)
    # Model
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32",
                   choices=[
                       "openai/clip-vit-base-patch32",
                       "openai/clip-vit-base-patch16",
                       "openai/clip-vit-large-patch14",
                       "openai/clip-vit-large-patch14-336",
                   ])
    p.add_argument("--dropout",     type=float, default=0.3)
    # Loss
    p.add_argument("--loss", default="margin",
                   choices=["bt", "margin", "margin_dec", "margin_var", "infonce", "hybrid"])
    p.add_argument("--margin",          type=float, default=0.5)
    p.add_argument("--lambda_dec",      type=float, default=0.1)
    p.add_argument("--lambda_var",      type=float, default=0.1)
    p.add_argument("--lambda_bt",       type=float, default=0.5,
                   help="Weight of BT term in hybrid loss (1-lambda_bt goes to margin)")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--infonce_temp",    type=float, default=1.0)
    # Optimization
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--head_lr_mult",  type=float, default=10.0,
                   help="Head LR = lr * head_lr_mult")
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--warmup_frac",   type=float, default=0.1)
    # Backbone freezing
    p.add_argument("--freeze_backbone",              action="store_true")
    p.add_argument("--unfreeze_top_vision_layers",   type=int, default=2)
    p.add_argument("--unfreeze_top_text_layers",     type=int, default=2)
    # LoRA (альтернатива unfreeze_top_layers)
    p.add_argument("--use_lora",            action="store_true",
                   help="Использовать LoRA вместо разморозки слоёв")
    p.add_argument("--lora_r",              type=int,   default=8)
    p.add_argument("--lora_alpha",          type=int,   default=16)
    p.add_argument("--lora_dropout",        type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,   default="q_proj,v_proj",
                   help="Через запятую: q_proj,v_proj,out_proj,fc1,fc2")
    # Misc
    p.add_argument("--seed",                     type=int, default=42)
    p.add_argument("--num_workers",              type=int, default=4)
    p.add_argument("--position_swap_aug",        action="store_true")
    p.add_argument("--early_stopping_patience",  type=int, default=5)
    p.add_argument("--output_dir",  default="./clip_pairs_only_ckpt")
    p.add_argument("--run_tag",     default="run",
                   help="Tag added to checkpoint filename and wandb run name")
    # W&B
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  default="taxonomy-clip-pairs-only")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_entity",   default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {args.run_tag} | model={args.model_name} | loss={args.loss} "
          f"| margin={args.margin} | lr={args.lr} "
          f"| unfreeze_vis={args.unfreeze_top_vision_layers} "
          f"| unfreeze_txt={args.unfreeze_top_text_layers} ===")
    print(f"Device: {device}")

    if args.use_wandb:
        if wandb is None:
            print("Warning: wandb not installed, disabling.")
            args.use_wandb = False
        else:
            run_name = args.wandb_run_name or args.run_tag
            wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=run_name, config=vars(args),
            )

    processor = CLIPProcessor.from_pretrained(args.model_name)
    collator  = Collator()

    def make_loader(path, augment, shuffle, swap):
        ds = PairOnlyDataset(path, images_root=args.images_root,
                             processor=processor,
                             augment=augment, position_swap=swap)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=collator,
                          pin_memory=True, persistent_workers=(args.num_workers > 0))

    train_loader = make_loader(args.train_csv, augment=True,  shuffle=True,
                               swap=args.position_swap_aug)
    val_loader   = make_loader(args.val_csv,   augment=False, shuffle=False, swap=False)
    test_loader  = make_loader(args.test_csv,  augment=False, shuffle=False, swap=False) \
                   if args.test_csv else None

    model = CLIPRewardModel(model_name=args.model_name, dropout=args.dropout)

    if args.use_lora:
        model.freeze_backbone()
        model.apply_lora(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
        )
    elif args.freeze_backbone:
        model.freeze_backbone()
        model.unfreeze_top_layers(
            num_text=args.unfreeze_top_text_layers,
            num_vision=args.unfreeze_top_vision_layers,
        )

    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,}")

    if args.use_wandb:
        wandb.config.update(
            {"num_trainable_params": n_trainable, "num_total_params": n_total},
            allow_val_change=True,
        )

    head_ids        = {id(p) for p in model.reward_head.parameters()}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    head_params     = [p for p in model.reward_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(10, int(args.warmup_frac * total_steps))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_kwargs = dict(
        loss_type       = args.loss,
        margin          = args.margin,
        lambda_dec      = args.lambda_dec,
        lambda_var      = args.lambda_var,
        lambda_bt       = args.lambda_bt,
        label_smoothing = args.label_smoothing,
        infonce_temp    = args.infonce_temp,
    )

    best_val_acc = -1.0
    patience     = 0
    best_path    = os.path.join(args.output_dir, f"best_{args.run_tag}.pt")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, device,
                            optimizer=optimizer, scheduler=scheduler, train=True,
                            **loss_kwargs)
        val_m   = run_epoch(model, val_loader, device, train=False,
                            **loss_kwargs)

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
                "model_name":       args.model_name,
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
        print("Loading best model for test evaluation...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = run_epoch(model, test_loader, device, train=False,
                           **loss_kwargs)
        print(f"=== {args.run_tag} TEST ===")
        print({k: round(v, 4) for k, v in test_m.items()})
        if args.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_m.items()})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
