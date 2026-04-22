"""
Fine-tune CLIP-based reward model on pairwise preference data.

Changes vs previous version:
  - BothBad отдельный бинарный классификатор на [fused_a, fused_b]
  - Tie loss: margin loss вместо MSE
  - Метрики: weighted_f1, macro_f1 по всем 4 классам
  - predict_class: diff + both_bad_logit → {0,1,2,3}

Usage:
    python train_reward.py \
        --train_csv train.csv \
        --val_csv val.csv \
        --test_csv test.csv \
        --images_root ./images \
        --freeze_backbone \
        --unfreeze_top_vision_layers 2 \
        --unfreeze_top_text_layers 2 \
        --epochs 10 --batch_size 16 --lr 3e-5
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import wandb
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


# ─── Labels ──────────────────────────────────────────────────────────────────

# 0 = B_win
# 1 = A_win
# 2 = Tie
# 3 = BothBad
ID_TO_LABEL = {0: "B_win", 1: "A_win", 2: "Tie", 3: "BothBad"}


def normalize_label(x):
    if pd.isna(x):
        return None
    try:
        x = int(x)
    except Exception:
        return None
    return x if x in [0, 1, 2, 3] else None


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    tmp = df.copy()
    tmp["label_id"] = tmp["result_human_def"].apply(normalize_label)
    counts = tmp["label_id"].value_counts().to_dict()
    return {v: int(counts.get(k, 0)) for k, v in ID_TO_LABEL.items()}


# ─── Augmentation ────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.05),
])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PairPreferenceDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str = "", augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.augment = augment

        required = {"wordnet_id", "model_a", "model_b", "result_human_def"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if "definition" not in self.df.columns and "prompt" not in self.df.columns:
            raise ValueError("Need 'definition' or 'prompt' column.")

        self.df["label_id"] = self.df["result_human_def"].apply(normalize_label)

        print(f"[{os.path.basename(csv_path)}] raw labels:",
              self.df["result_human_def"].dropna().unique()[:20])
        print(f"[{os.path.basename(csv_path)}] normalized labels:",
              self.df["label_id"].dropna().unique())

        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid rows after filtering.")

    def __len__(self):
        return len(self.df)

    def _load_image(self, model_name: str, wordnet_id: str) -> Image.Image:
        model_dir = os.path.join(self.images_root, str(model_name))
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(model_dir, f"{wordnet_id}{ext}")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                if self.augment:
                    img = TRAIN_TRANSFORM(img)
                return img
        raise FileNotFoundError(f"No image: model={model_name}, id={wordnet_id}")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        wid = str(row["wordnet_id"])
        prompt = f"An image of {row['core_synset']} ({row['definition']})"
        return {
            "prompt":    prompt,
            "image_a":  self._load_image(str(row["model_a"]), wid),
            "image_b":  self._load_image(str(row["model_b"]), wid),
            "label":    int(row["label_id"]),
            "wordnet_id": wid,
            "model_a":  str(row["model_a"]),
            "model_b":  str(row["model_b"]),
        }


@dataclass
class Batch:
    prompts:  List[str]
    images_a: List[Image.Image]
    images_b: List[Image.Image]
    labels:   torch.Tensor


class Collator:
    def __call__(self, items: List[Dict]) -> Batch:
        return Batch(
            prompts=[x["prompt"] for x in items],
            images_a=[x["image_a"] for x in items],
            images_b=[x["image_b"] for x in items],
            labels=torch.tensor([x["label"] for x in items], dtype=torch.long),
        )


# ─── Model ───────────────────────────────────────────────────────────────────

class TaxonomyRewardModel(nn.Module):
    """
    Backbone: CLIP
    Выходы:
      - reward_a, reward_b : скалярные реварды для каждого изображения
      - both_bad_logit     : бинарный классификатор "оба плохие"
                             на concat(fused_a, fused_b)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", dropout: float = 0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden    = self.clip.config.projection_dim
        fused_dim = hidden * 4  # [img, txt, img*txt, |img-txt|]

        # Упрощённая reward голова
        self.reward_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

        # BothBad классификатор: видит обе картинки одновременно
        self.both_bad_head = nn.Sequential(
            nn.LayerNorm(fused_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, num_text_layers: int = 0, num_vision_layers: int = 0) -> None:
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = True
        for p in self.clip.text_projection.parameters():
            p.requires_grad = True
        if hasattr(self.clip.text_model.encoder, "layers") and num_text_layers > 0:
            for layer in self.clip.text_model.encoder.layers[-num_text_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.clip.vision_model.encoder, "layers") and num_vision_layers > 0:
            for layer in self.clip.vision_model.encoder.layers[-num_vision_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def encode_text_image(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        text_emb  = F.normalize(outputs.text_embeds,  dim=-1)
        image_emb = F.normalize(outputs.image_embeds, dim=-1)
        fused = torch.cat(
            [image_emb, text_emb, image_emb * text_emb, torch.abs(image_emb - text_emb)],
            dim=-1,
        )
        reward = self.reward_head(fused).squeeze(-1)
        return fused, reward

    def forward(
        self,
        input_ids_a, attention_mask_a, pixel_values_a,
        input_ids_b, attention_mask_b, pixel_values_b,
    ):
        fused_a, reward_a = self.encode_text_image(input_ids_a, attention_mask_a, pixel_values_a)
        fused_b, reward_b = self.encode_text_image(input_ids_b, attention_mask_b, pixel_values_b)

        both_bad_logit = self.both_bad_head(
            torch.cat([fused_a, fused_b], dim=-1)
        ).squeeze(-1)

        return {
            "reward_a":       reward_a,
            "reward_b":       reward_b,
            "both_bad_logit": both_bad_logit,
        }


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_class(
    diff: torch.Tensor,
    both_bad_logit: Optional[torch.Tensor] = None,
    tie_margin: float = 0.3,
) -> torch.Tensor:
    """
    Переводит скалярные выходы в предсказанный класс:

      |diff| <= tie_margin  → Tie (2)
      diff >  tie_margin    → A_win (1)
      diff < -tie_margin    → B_win (0)
      both_bad_logit > 0    → BothBad (3)  [перебивает всё остальное]
    """
    device = diff.device
    pred = torch.where(
        diff > tie_margin,
        torch.tensor(1, device=device),
        torch.where(
            diff < -tie_margin,
            torch.tensor(0, device=device),
            torch.tensor(2, device=device),
        ),
    )
    if both_bad_logit is not None:
        pred = torch.where(both_bad_logit > 0, torch.tensor(3, device=device), pred)
    return pred


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    labels: torch.Tensor,
    both_bad_logit: Optional[torch.Tensor] = None,
    lambda_tie: float = 0.5,
    lambda_bb: float = 1.0,
    tie_margin: float = 0.3,
    label_smoothing: float = 0.1,
) -> tuple:
    """
    labels:
        0 = B_win   → diff < 0
        1 = A_win   → diff > 0
        2 = Tie     → |diff| → 0  (margin loss)
        3 = BothBad → |diff| → 0  + both_bad_logit → 1
    """
    diff = reward_a - reward_b
    s    = label_smoothing

    # Win losses с label smoothing
    win_a_loss = -(1 - s) * F.logsigmoid(diff)  - s * F.logsigmoid(-diff)
    win_b_loss = -(1 - s) * F.logsigmoid(-diff) - s * F.logsigmoid(diff)

    # Margin loss для Tie/BothBad
    tie_loss = F.relu(diff.abs() - tie_margin).pow(2)

    pair_loss = torch.zeros_like(diff)
    pair_loss = torch.where(labels == 1, win_a_loss,            pair_loss)
    pair_loss = torch.where(labels == 0, win_b_loss,            pair_loss)
    pair_loss = torch.where(labels == 2, lambda_tie * tie_loss, pair_loss)
    pair_loss = torch.where(labels == 3, lambda_tie * tie_loss, pair_loss)

    loss = pair_loss.mean()

    # BothBad бинарный классификатор
    if both_bad_logit is not None:
        bb_target = (labels == 3).float()
        bb_loss   = F.binary_cross_entropy_with_logits(both_bad_logit, bb_target)
        loss      = loss + lambda_bb * bb_loss

    # ── Метрики ──
    with torch.no_grad():
        pred_classes = predict_class(diff, both_bad_logit, tie_margin=tie_margin)
        labels_np    = labels.cpu().numpy()
        pred_np      = pred_classes.cpu().numpy()

        weighted_f1 = f1_score(
            labels_np, pred_np, average="weighted",
            labels=[0, 1, 2, 3], zero_division=0,
        )
        macro_f1 = f1_score(
            labels_np, pred_np, average="macro",
            labels=[0, 1, 2, 3], zero_division=0,
        )
        per_class = f1_score(
            labels_np, pred_np, average=None,
            labels=[0, 1, 2, 3], zero_division=0,
        )

        # Старые метрики для совместимости
        pref_pred   = (diff > 0).long()
        mask_binary = (labels == 0) | (labels == 1)
        binary_acc  = (
            (pref_pred[mask_binary] == labels[mask_binary]).float().mean().item()
            if mask_binary.any() else float("nan")
        )
        tie_abs_diff = diff[labels == 2].abs().mean().item() if (labels == 2).any() else float("nan")
        bb_abs_diff  = diff[labels == 3].abs().mean().item() if (labels == 3).any() else float("nan")

        if both_bad_logit is not None:
            bb_pred = (both_bad_logit > 0).float()
            bb_acc  = (bb_pred == (labels == 3).float()).float().mean().item()
        else:
            bb_acc = float("nan")

    return loss, {
        "loss":         loss.item(),
        "weighted_f1":  weighted_f1,          # главная метрика
        "macro_f1":     macro_f1,
        "f1_B_win":     float(per_class[0]),
        "f1_A_win":     float(per_class[1]),
        "f1_Tie":       float(per_class[2]),
        "f1_BothBad":   float(per_class[3]),
        "binary_acc":   binary_acc,
        "tie_abs_diff": tie_abs_diff,
        "bb_abs_diff":  bb_abs_diff,
        "both_bad_acc": bb_acc,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def move_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def prepare_pair_inputs(processor, prompts, images, max_length=77):
    encoded = processor(
        text=prompts, images=images,
        return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "pixel_values":   encoded["pixel_values"],
    }


# ─── Train / Eval loop ───────────────────────────────────────────────────────

def run_epoch(
    model, loader, processor, device,
    optimizer=None, scheduler=None, train=True,
    tie_margin: float = 0.3,
):
    model.train(train)

    agg = {
        "loss": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0,
        "f1_B_win": 0.0, "f1_A_win": 0.0, "f1_Tie": 0.0, "f1_BothBad": 0.0,
        "binary_acc": 0.0, "tie_abs_diff": 0.0, "bb_abs_diff": 0.0, "both_bad_acc": 0.0,
    }
    n = 0

    for batch in loader:
        labels = batch.labels.to(device)

        enc_a = move_to_device(prepare_pair_inputs(processor, batch.prompts, batch.images_a), device)
        enc_b = move_to_device(prepare_pair_inputs(processor, batch.prompts, batch.images_b), device)

        with torch.set_grad_enabled(train):
            outputs = model(
                input_ids_a=enc_a["input_ids"],
                attention_mask_a=enc_a["attention_mask"],
                pixel_values_a=enc_a["pixel_values"],
                input_ids_b=enc_b["input_ids"],
                attention_mask_b=enc_b["attention_mask"],
                pixel_values_b=enc_b["pixel_values"],
            )
            loss, metrics = compute_loss(
                reward_a=outputs["reward_a"],
                reward_b=outputs["reward_b"],
                both_bad_logit=outputs["both_bad_logit"],
                labels=labels,
                tie_margin=tie_margin,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        for k in agg:
            v = metrics.get(k, float("nan"))
            if not math.isnan(v):
                agg[k] += v
        n += 1

    return {k: v / max(n, 1) for k, v in agg.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",    required=True)
    parser.add_argument("--val_csv",      required=True)
    parser.add_argument("--test_csv",     default=None)
    parser.add_argument("--images_root",  required=True)
    parser.add_argument("--model_name",   default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir",   default="./reward_ckpt")
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tie_margin",   type=float, default=0.3,
                        help="Margin для tie loss и predict_class; попробуй 0.1, 0.2, 0.3")
    parser.add_argument("--lambda_tie",   type=float, default=0.5)
    parser.add_argument("--lambda_bb",    type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_top_vision_layers", type=int, default=0)
    parser.add_argument("--unfreeze_top_text_layers",   type=int, default=0)
    parser.add_argument("--early_stopping_patience",    type=int, default=3)
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  default="taxonomy-reward")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name, config=vars(args),
        )

    for split, path in [("Train", args.train_csv), ("Val", args.val_csv)]:
        print(f"{split}:", get_label_distribution(pd.read_csv(path)))
    if args.test_csv:
        print("Test:", get_label_distribution(pd.read_csv(args.test_csv)))

    # ── Datasets ──
    processor = CLIPProcessor.from_pretrained(args.model_name)
    collator  = Collator()

    def make_loader(path, augment, shuffle):
        ds = PairPreferenceDataset(path, images_root=args.images_root, augment=augment)
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
        )

    train_loader = make_loader(args.train_csv, augment=True,  shuffle=True)
    val_loader   = make_loader(args.val_csv,   augment=False, shuffle=False)
    test_loader  = make_loader(args.test_csv,  augment=False, shuffle=False) if args.test_csv else None

    # ── Model ──
    model = TaxonomyRewardModel(model_name=args.model_name, dropout=args.dropout)

    if args.freeze_backbone:
        model.freeze_backbone()
        model.unfreeze_top_layers(
            num_text_layers=args.unfreeze_top_text_layers,
            num_vision_layers=args.unfreeze_top_vision_layers,
        )

    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

    if args.use_wandb:
        wandb.config.update({"num_trainable_params": n_trainable}, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=100)

    # ── Optimizer: раздельный lr для backbone и голов ──
    head_ids = (
        {id(p) for p in model.reward_head.parameters()} |
        {id(p) for p in model.both_bad_head.parameters()}
    )
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head_params     = (list(model.reward_head.parameters()) +
                       list(model.both_bad_head.parameters()))

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_f1 = -1.0   # сохраняем по weighted_f1
    best_path   = os.path.join(args.output_dir, "best_model.pt")
    patience    = 0

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, processor, device,
            optimizer=optimizer, scheduler=scheduler, train=True,
            tie_margin=args.tie_margin,
        )
        val_m = run_epoch(
            model, val_loader, processor, device, train=False,
            tie_margin=args.tie_margin,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", {k: round(v, 4) for k, v in train_m.items()})
        print("Val:  ", {k: round(v, 4) for k, v in val_m.items()})

        if args.use_wandb:
            wandb.log(
                {
                    "epoch":       epoch,
                    "lr_head":     optimizer.param_groups[1]["lr"],
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    **{f"train/{k}": v for k, v in train_m.items()},
                    **{f"val/{k}":   v for k, v in val_m.items()},
                },
                step=epoch,
            )

        if val_m["weighted_f1"] > best_val_f1:
            best_val_f1 = val_m["weighted_f1"]
            patience = 0
            torch.save(
                {
                    "model_state_dict":  model.state_dict(),
                    "model_name":        args.model_name,
                    "id_to_label":       ID_TO_LABEL,
                    "epoch":             epoch,
                    "val_weighted_f1":   best_val_f1,
                    "tie_margin":        args.tie_margin,
                },
                best_path,
            )
            print(f"  ✓ Saved best (val weighted_f1={best_val_f1:.4f})")
            if args.use_wandb:
                wandb.summary["best_val_weighted_f1"] = best_val_f1
                wandb.summary["best_epoch"] = epoch
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.early_stopping_patience})")
            if patience >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nTraining done.")

    # ── Test ──
    if test_loader is not None:
        print("Loading best model for test evaluation...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_m = run_epoch(
            model, test_loader, processor, device, train=False,
            tie_margin=args.tie_margin,
        )
        print("Test:", {k: round(v, 4) for k, v in test_m.items()})

        if args.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_m.items()})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
