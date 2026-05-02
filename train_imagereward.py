

"""
Fine-tune ImageReward on pairwise preference data.

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ vs предыдущей версии:
  1. Используем self.blip напрямую (visual_encoder + text_encoder через cross-attention),
     а не .score() которая возвращает float и ломает градиент.
  2. Батчуем обработку изображений — один проход на весь батч вместо цикла.
  3. Добавлены: weighted_f1, anchor loss, position-swap аугментация, early stopping по F1.

Архитектура ImageReward:
  - self.blip.visual_encoder         (ViT-L из BLIP)
  - self.blip.text_encoder           (BERT-like с cross-attention на image)
  - self.blip.tokenizer              (BLIP tokenizer)
  - self.mlp                         (MLP голова: 768 → 1)
  - self.mean, self.std              (нормализация выхода)
  - self.preprocess                  (PIL → tensor, для инференса; мы делаем свой pipeline)

Install:
    pip install image-reward torch torchvision transformers wandb pandas scikit-learn

Usage:
    python train_imagereward.py \
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
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import ImageReward as RM
import wandb
from transformers import get_cosine_schedule_with_warmup


# ─── Labels ──────────────────────────────────────────────────────────────────

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

# BLIP / ImageReward использует 224x224 + нормализацию CLIP.
# Строим transform который (а) делает PIL → tensor (224x224 normalized)
# и (б) на трейне дополнительно аугментирует.
IMG_SIZE = 224
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

class PairPreferenceDataset(Dataset):
    """
    Возвращает уже тензоры (C, H, W) а не PIL.
    Position-swap аугментация: с вероятностью 0.5 в трейне меняет A ↔ B
    и соответственно инвертирует лейбл (0↔1, 2 и 3 остаются).
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

        print(f"[{os.path.basename(csv_path)}] raw labels:",
              self.df["result_human_def"].dropna().unique()[:20])
        print(f"[{os.path.basename(csv_path)}] normalized labels:",
              self.df["label_id"].dropna().unique())

        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid rows after filtering.")

        # Case-insensitive lookup: lowercase(dirname) → actual dirname
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

        # Position-swap аугментация: меняем A и B местами случайно
        if self.position_swap and random.random() < 0.5:
            img_a, img_b = img_b, img_a
            if   label == 0: label = 1  # B_win → A_win
            elif label == 1: label = 0  # A_win → B_win
            # Tie (2) и BothBad (3) не меняются

        return {
            "prompt":  prompt,
            "image_a": img_a,     # тензор (3, 224, 224)
            "image_b": img_b,
            "label":   label,
        }


@dataclass
class Batch:
    prompts: List[str]
    images_a: torch.Tensor   # (B, 3, 224, 224)
    images_b: torch.Tensor   # (B, 3, 224, 224)
    labels:   torch.Tensor   # (B,)


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
    """
    Правильная обёртка над ImageReward:
      - Напрямую дёргаем self.blip.visual_encoder, self.blip.text_encoder, self.mlp
      - Работаем с батчами, а не поэлементно
      - Градиенты не теряются

    Reward scalar = (raw_mlp_output - mean) / std
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        # Загружаем претрейн
        self.ir_model = RM.load("ImageReward-v1.0")

        # Нормализационные константы ImageReward (reward ~ N(0, 1))
        self.register_buffer("rw_mean", torch.tensor(self.ir_model.mean))
        self.register_buffer("rw_std",  torch.tensor(self.ir_model.std))

        # Добавим dropout перед MLP головой как регуляризацию
        self.head_dropout = nn.Dropout(dropout)

        # Распаковка для удобства
        self.blip = self.ir_model.blip
        self.mlp  = self.ir_model.mlp

    def freeze_backbone(self):
        """Замораживаем весь BLIP."""
        for p in self.blip.parameters():
            p.requires_grad = False
        print("BLIP backbone frozen.")

    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Размораживаем последние N слоёв visual_encoder (ViT blocks)
        и text_encoder (BertLayer).
        """
        # Visual encoder (ViT)
        try:
            vis_blocks = self.blip.visual_encoder.blocks
            for block in vis_blocks[-num_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            print(f"Unfrozen top {num_layers} visual encoder blocks.")
        except AttributeError:
            print("Warning: could not access visual_encoder.blocks")

        # Text encoder (BERT-like)
        try:
            txt_layers = self.blip.text_encoder.encoder.layer
            for layer in txt_layers[-num_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"Unfrozen top {num_layers} text encoder layers.")
        except AttributeError:
            print("Warning: could not access text_encoder.encoder.layer")

    def score_batch(
        self,
        images: torch.Tensor,          # (B, 3, 224, 224)
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Батчевый скор с градиентом.

        Повторяем логику score_gard() из ImageReward, но на батче:
          1) visual_encoder(image) → image_embeds
          2) text_encoder(prompt_ids, encoder_hidden_states=image_embeds)
             → cross-attention fused features
          3) mlp(text_features[:, 0, :]) → raw scalar
          4) reward = (raw - mean) / std
        """
        device = images.device

        # Токенизация промптов
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # 1) Visual encoder
        image_embeds = self.blip.visual_encoder(images)     # (B, num_patches+1, D)
        image_atts   = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=device,
        )

        # 2) Text encoder с cross-attention на image_embeds
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Берём CLS-токен (первый)
        txt_features = text_output.last_hidden_state[:, 0, :]   # (B, 768)
        txt_features = self.head_dropout(txt_features)

        # 3) MLP голова → raw scalar
        raw = self.mlp(txt_features).squeeze(-1)                # (B,)

        # 4) Нормализация (как в оригинальном ImageReward)
        reward = (raw - self.rw_mean) / self.rw_std             # (B,)
        return reward

    def forward(
        self,
        prompts:  List[str],
        images_a: torch.Tensor,
        images_b: torch.Tensor,
    ):
        reward_a = self.score_batch(images_a, prompts)
        reward_b = self.score_batch(images_b, prompts)
        return {"reward_a": reward_a, "reward_b": reward_b}


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_class(
    diff: torch.Tensor,
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    tie_margin: float = 0.1,
    bad_threshold: float = 0.4,
) -> torch.Tensor:
    """
    Логика:
      diff > tie_margin              → A_win (1)
      diff < -tie_margin             → B_win (0)
      |diff| <= tie_margin + mean_reward < bad_threshold → BothBad (3)
      |diff| <= tie_margin + mean_reward >= bad_threshold → Tie (2)
    BothBad и Tie различаются по среднему reward пары, а не по абсолютному порогу каждого.
    """
    device = diff.device
    mean_reward = (reward_a + reward_b) / 2

    # Сначала по diff: кто лучше
    pred = torch.where(
        diff > tie_margin,
        torch.tensor(1, device=device),   # A_win
        torch.where(
            diff < -tie_margin,
            torch.tensor(0, device=device),  # B_win
            torch.tensor(2, device=device),  # Tie (пока)
        ),
    )
    # Потом override: если оба плохие — BothBad, независимо от diff
    both_bad = mean_reward < bad_threshold
    pred = torch.where(both_bad, torch.tensor(3, device=device), pred)
    return pred


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    labels:   torch.Tensor,
    lambda_tie:     float = 0.5,
    lambda_anchor:  float = 1.0,
    tie_margin:     float = 0.3,
    bad_threshold:  float = -0.5,
    label_smoothing: float = 0.1,
    tau_neg: float = -1.0,
    tau_pos: float =  1.0,
) -> tuple:
    diff = reward_a - reward_b
    s    = label_smoothing

    # Pair loss
    win_a_loss = -(1 - s) * F.logsigmoid(diff)  - s * F.logsigmoid(-diff)
    win_b_loss = -(1 - s) * F.logsigmoid(-diff) - s * F.logsigmoid(diff)
    tie_loss   = F.relu(diff.abs() - tie_margin).pow(2)

    pair_loss = torch.zeros_like(diff)
    pair_loss = torch.where(labels == 1, win_a_loss,            pair_loss)
    pair_loss = torch.where(labels == 0, win_b_loss,            pair_loss)
    pair_loss = torch.where(labels == 2, lambda_tie * tie_loss, pair_loss)
    pair_loss = torch.where(labels == 3, lambda_tie * tie_loss, pair_loss)

    # Anchor loss — разделяем классы по абсолютному уровню reward
    mean_reward = (reward_a + reward_b) / 2

    # A_win/B_win: winner должен быть выше tau_pos
    anchor_a_win = F.relu(tau_pos - reward_a)
    anchor_b_win = F.relu(tau_pos - reward_b)
    # Tie: оба изображения хорошие → среднее выше tau_pos
    anchor_tie = F.relu(tau_pos - mean_reward)
    # BothBad: оба плохие → каждый reward ниже tau_neg (sum = двойной gradient)
    anchor_bb  = F.relu(reward_a - tau_neg) + F.relu(reward_b - tau_neg)

    anchor_loss = torch.zeros_like(diff)
    anchor_loss = torch.where(labels == 1, anchor_a_win, anchor_loss)
    anchor_loss = torch.where(labels == 0, anchor_b_win, anchor_loss)
    anchor_loss = torch.where(labels == 2, anchor_tie,   anchor_loss)
    anchor_loss = torch.where(labels == 3, anchor_bb,    anchor_loss)

    loss = pair_loss.mean() + lambda_anchor * anchor_loss.mean()

    # Метрики
    with torch.no_grad():
        pred_classes = predict_class(
            diff, reward_a, reward_b,
            tie_margin=tie_margin,
            bad_threshold=bad_threshold,
        )
        labels_np = labels.cpu().numpy()
        pred_np   = pred_classes.cpu().numpy()

        weighted_f1 = f1_score(labels_np, pred_np, average="weighted",
                               labels=[0,1,2,3], zero_division=0)
        macro_f1    = f1_score(labels_np, pred_np, average="macro",
                               labels=[0,1,2,3], zero_division=0)
        per_class   = f1_score(labels_np, pred_np, average=None,
                               labels=[0,1,2,3], zero_division=0)

        pref_pred   = (diff > 0).long()
        mask_binary = (labels == 0) | (labels == 1)
        binary_acc  = (
            (pref_pred[mask_binary] == labels[mask_binary]).float().mean().item()
            if mask_binary.any() else float("nan")
        )
        tie_abs_diff = diff[labels == 2].abs().mean().item() if (labels == 2).any() else float("nan")
        bb_abs_diff  = diff[labels == 3].abs().mean().item() if (labels == 3).any() else float("nan")

        mean_r_win  = reward_a[labels == 1].mean().item() if (labels == 1).any() else float("nan")
        mean_r_bb   = (
            ((reward_a[labels == 3].mean() + reward_b[labels == 3].mean()) / 2).item()
            if (labels == 3).any() else float("nan")
        )

    return loss, {
        "loss":          loss.item(),
        "weighted_f1":   weighted_f1,
        "macro_f1":      macro_f1,
        "f1_B_win":      float(per_class[0]),
        "f1_A_win":      float(per_class[1]),
        "f1_Tie":        float(per_class[2]),
        "f1_BothBad":    float(per_class[3]),
        "binary_acc":    binary_acc,
        "tie_abs_diff":  tie_abs_diff,
        "bb_abs_diff":   bb_abs_diff,
        "mean_reward_winner":  mean_r_win,
        "mean_reward_bothbad": mean_r_bb,
    }


# ─── Train / Eval loop ───────────────────────────────────────────────────────

def run_epoch(
    model, loader, device,
    optimizer=None, scheduler=None, train=True,
    tie_margin=0.3, bad_threshold=-0.5,
    tau_neg=-1.0, tau_pos=1.0,
    lambda_anchor=1.0, lambda_tie=0.5,
    label_smoothing=0.1,
):
    model.train(train)

    agg = {
        "loss": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0,
        "f1_B_win": 0.0, "f1_A_win": 0.0, "f1_Tie": 0.0, "f1_BothBad": 0.0,
        "binary_acc": 0.0, "tie_abs_diff": 0.0, "bb_abs_diff": 0.0,
        "mean_reward_winner": 0.0, "mean_reward_bothbad": 0.0,
    }
    n = 0

    for batch in loader:
        images_a = batch.images_a.to(device, non_blocking=True)
        images_b = batch.images_b.to(device, non_blocking=True)
        labels   = batch.labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(
                prompts=batch.prompts,
                images_a=images_a,
                images_b=images_b,
            )
            loss, metrics = compute_loss(
                reward_a=outputs["reward_a"],
                reward_b=outputs["reward_b"],
                labels=labels,
                tie_margin=tie_margin,
                bad_threshold=bad_threshold,
                tau_neg=tau_neg,
                tau_pos=tau_pos,
                lambda_anchor=lambda_anchor,
                lambda_tie=lambda_tie,
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
    parser.add_argument("--output_dir",  default="./ir_reward_ckpt")
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-5)   # paper uses 1e-5
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tie_margin",   type=float, default=0.1)
    parser.add_argument("--lambda_tie",   type=float, default=0.5)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--tau_pos",      type=float, default=1.0)
    parser.add_argument("--tau_neg",      type=float, default=-1.0)
    parser.add_argument("--bad_threshold", type=float, default=0.4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--freeze_backbone",    action="store_true")
    parser.add_argument("--unfreeze_top_layers", type=int, default=2)
    parser.add_argument("--position_swap_aug",  action="store_true",
                        help="На трейне с p=0.5 менять местами A/B (полезно против position bias)")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--wandb_project",  default="taxonomy-reward-ir")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))

    for split, path in [("Train", args.train_csv), ("Val", args.val_csv)]:
        print(f"{split}:", get_label_distribution(pd.read_csv(path)))
    if args.test_csv:
        print("Test:", get_label_distribution(pd.read_csv(args.test_csv)))

    # ── Datasets ──
    collator = Collator()

    def make_loader(path, augment, shuffle, swap):
        ds = PairPreferenceDataset(
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

    # ── Model ──
    model = ImageRewardFineTuner(dropout=args.dropout)

    if args.freeze_backbone:
        model.freeze_backbone()
        if args.unfreeze_top_layers > 0:
            model.unfreeze_top_layers(num_layers=args.unfreeze_top_layers)
    # MLP голова всегда обучается (не заморожена по умолчанию — требует grad)

    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    if args.use_wandb:
        wandb.config.update(
            {"num_trainable_params": n_trainable, "num_total_params": n_total},
            allow_val_change=True,
        )

    # ── Optimizer: раздельный lr для backbone и MLP ──
    head_ids = {id(p) for p in model.mlp.parameters()} | \
               {id(p) for p in model.head_dropout.parameters()}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    head_params     = [p for p in model.mlp.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},        # paper: 1e-5 для всего
            {"params": head_params,     "lr": args.lr * 10},   # голова чуть быстрее
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_path   = os.path.join(args.output_dir, "best_model.pt")
    patience    = 0

    epoch_kwargs = dict(
        device=device,
        tie_margin=args.tie_margin,
        bad_threshold=args.bad_threshold,
        tau_neg=args.tau_neg, tau_pos=args.tau_pos,
        lambda_anchor=args.lambda_anchor,
        lambda_tie=args.lambda_tie,
        label_smoothing=args.label_smoothing,
    )

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

        if val_m["weighted_f1"] > best_val_f1:
            best_val_f1 = val_m["weighted_f1"]
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_weighted_f1":  best_val_f1,
                "args":             vars(args),
            }, best_path)
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
