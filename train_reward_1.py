import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import wandb
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


# 0 = B_win
# 1 = A_win
# 2 = Tie
# 3 = BothBad
ID_TO_LABEL = {
    0: "B_win",
    1: "A_win",
    2: "Tie",
    3: "BothBad",
}


def normalize_label(x):
    if pd.isna(x):
        return None
    try:
        x = int(x)
    except Exception:
        return None
    if x in [0, 1, 2, 3]:
        return x
    return None


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
    return {
        "B_win": int(counts.get(0, 0)),
        "A_win": int(counts.get(1, 0)),
        "Tie": int(counts.get(2, 0)),
        "BothBad": int(counts.get(3, 0)),
    }


# ─── Augmentation ────────────────────────────────────────────────────────────

def build_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
    ])

TRAIN_TRANSFORM = build_train_transform()


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PairPreferenceDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str = "", augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.augment = augment

        required = {"wordnet_id", "model_a", "model_b", "result_human_def"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        if "prompt" not in self.df.columns and "definition" not in self.df.columns:
            raise ValueError("CSV must contain either 'prompt' or 'definition' column.")

        self.df["label_id"] = self.df["result_human_def"].apply(normalize_label)

        print(f"[{os.path.basename(csv_path)}] unique raw labels:",
              self.df["result_human_def"].dropna().unique()[:20])
        print(f"[{os.path.basename(csv_path)}] unique normalized labels:",
              self.df["label_id"].dropna().unique())

        self.df = self.df[self.df["label_id"].notna()].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("No valid rows left after filtering labels.")

    def __len__(self):
        return len(self.df)

    def _build_image_path(self, model_name: str, wordnet_id: str) -> str:
        model_dir = os.path.join(self.images_root, str(model_name))
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(model_dir, f"{wordnet_id}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No image found for wordnet_id={wordnet_id}, model={model_name} in {model_dir}"
        )

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        wordnet_id = str(row["wordnet_id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])

        prompt = f"An image of {row['core_synset']} ({row['definition']})"

        image_a = Image.open(self._build_image_path(model_a, wordnet_id)).convert("RGB")
        image_b = Image.open(self._build_image_path(model_b, wordnet_id)).convert("RGB")

        if self.augment:
            image_a = TRAIN_TRANSFORM(image_a)
            image_b = TRAIN_TRANSFORM(image_b)

        return {
            "prompt": str(prompt),
            "image_a": image_a,
            "image_b": image_b,
            "label": int(row["label_id"]),
            "wordnet_id": wordnet_id,
            "model_a": model_a,
            "model_b": model_b,
        }


@dataclass
class Batch:
    prompts: List[str]
    images_a: List[Image.Image]
    images_b: List[Image.Image]
    labels: torch.Tensor


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
    Упрощённая reward head + увеличенный dropout для борьбы с переобучением.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        dropout: float = 0.3,   # было 0.1 → стало 0.3
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden = self.clip.config.projection_dim
        fused_dim = hidden * 4  # [img, txt, img*txt, |img-txt|]

        # Упрощённая голова: убрали промежуточный слой
        self.reward_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, num_text_layers: int = 0, num_vision_layers: int = 0) -> None:
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = True
        for p in self.clip.text_projection.parameters():
            p.requires_grad = True

        if hasattr(self.clip.text_model.encoder, "layers"):
            for layer in self.clip.text_model.encoder.layers[-num_text_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if hasattr(self.clip.vision_model.encoder, "layers"):
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
        text_emb = F.normalize(outputs.text_embeds, dim=-1)
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
        _, reward_a = self.encode_text_image(input_ids_a, attention_mask_a, pixel_values_a)
        _, reward_b = self.encode_text_image(input_ids_b, attention_mask_b, pixel_values_b)
        return {"reward_a": reward_a, "reward_b": reward_b}


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    labels: torch.Tensor,
    lambda_tie: float = 1.0,
    label_smoothing: float = 0.1,   # новое: сглаживание для бинарных пар
):
    """
    labels:
        0 = B_win  → reward_b > reward_a
        1 = A_win  → reward_a > reward_b
        2 = Tie    → |diff| → 0
        3 = BothBad → |diff| → 0
    """
    diff = reward_a - reward_b

    # Label smoothing для win-пар
    s = label_smoothing
    win_a_loss = -(1 - s) * F.logsigmoid(diff) - s * F.logsigmoid(-diff)
    win_b_loss = -(1 - s) * F.logsigmoid(-diff) - s * F.logsigmoid(diff)
    tie_loss = diff.pow(2)

    pair_loss = torch.zeros_like(diff)
    pair_loss = torch.where(labels == 1, win_a_loss, pair_loss)
    pair_loss = torch.where(labels == 0, win_b_loss, pair_loss)
    pair_loss = torch.where(labels == 2, lambda_tie * tie_loss, pair_loss)
    pair_loss = torch.where(labels == 3, lambda_tie * tie_loss, pair_loss)

    loss = pair_loss.mean()

    with torch.no_grad():
        pref_pred = torch.where(diff > 0, 1, 0)
        mask_binary = (labels == 0) | (labels == 1)
        binary_acc = (
            (pref_pred[mask_binary] == labels[mask_binary]).float().mean().item()
            if mask_binary.any() else float("nan")
        )
        tie_abs_diff = diff[labels == 2].abs().mean().item() if (labels == 2).any() else float("nan")
        both_bad_abs_diff = diff[labels == 3].abs().mean().item() if (labels == 3).any() else float("nan")

    metrics = {
        "loss": loss.item(),
        "pair_loss": loss.item(),
        "binary_acc": binary_acc,
        "tie_abs_diff": tie_abs_diff,
        "both_bad_abs_diff": both_bad_abs_diff,
    }
    return loss, metrics


# ─── Helpers ─────────────────────────────────────────────────────────────────

def move_to_device(batch_inputs, device):
    return {k: v.to(device) for k, v in batch_inputs.items()}


def prepare_pair_inputs(processor, prompts, images, max_length=77):
    encoded = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "pixel_values": encoded["pixel_values"],
    }


# ─── Train / Eval loop ───────────────────────────────────────────────────────

def run_epoch(
    model, loader, processor, device,
    optimizer=None, scheduler=None, train=True,
):
    model.train(train)

    agg = {"loss": 0.0, "pair_loss": 0.0, "binary_acc": 0.0,
           "tie_abs_diff": 0.0, "both_bad_abs_diff": 0.0}
    n_steps = 0

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
                labels=labels,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        for k in agg:
            v = metrics[k]
            if not math.isnan(v):
                agg[k] += v
        n_steps += 1

    for k in agg:
        agg[k] /= max(n_steps, 1)

    return agg


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--images_root", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", type=str, default="./reward_ckpt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)           # было 1e-4
    parser.add_argument("--weight_decay", type=float, default=1e-2)  # было 1e-4
    parser.add_argument("--dropout", type=float, default=0.3)        # было 0.1
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_top_vision_layers", type=int, default=0)
    parser.add_argument("--unfreeze_top_text_layers", type=int, default=0)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="taxonomy-reward")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ── Label distributions ──
    for split, path in [("Train", args.train_csv), ("Val", args.val_csv)]:
        dist = get_label_distribution(pd.read_csv(path))
        print(f"{split} label distribution: {dist}")
    if args.test_csv:
        dist = get_label_distribution(pd.read_csv(args.test_csv))
        print(f"Test label distribution: {dist}")

    # ── Datasets ──
    processor = CLIPProcessor.from_pretrained(args.model_name)

    train_ds = PairPreferenceDataset(args.train_csv, images_root=args.images_root, augment=True)
    val_ds   = PairPreferenceDataset(args.val_csv,   images_root=args.images_root, augment=False)
    test_ds  = (PairPreferenceDataset(args.test_csv, images_root=args.images_root, augment=False)
                if args.test_csv else None)

    collator = Collator()

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False) if test_ds else None

    # ── Model ──
    model = TaxonomyRewardModel(model_name=args.model_name, dropout=args.dropout)

    if args.freeze_backbone:
        model.freeze_backbone()
        model.unfreeze_top_layers(
            num_text_layers=args.unfreeze_top_text_layers,
            num_vision_layers=args.unfreeze_top_vision_layers,
        )

    model.to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_trainable:,}")

    if args.use_wandb:
        wandb.config.update({"num_trainable_params": num_trainable}, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=100)

    # ── Optimizer: разделяем LR для backbone и головы ──
    head_params = list(model.reward_head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1},   # backbone — меньший LR
        {"params": head_params,     "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    patience_counter = 0

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, processor, device,
            optimizer=optimizer, scheduler=scheduler, train=True,
        )
        val_metrics = run_epoch(
            model, val_loader, processor, device, train=False,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", {k: round(v, 4) for k, v in train_metrics.items()})
        print("Val:  ", {k: round(v, 4) for k, v in val_metrics.items()})

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "lr_head": optimizer.param_groups[1]["lr"],
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                },
                step=epoch,
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "id_to_label": ID_TO_LABEL,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                best_model_path,
            )
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

            if args.use_wandb:
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nTraining done.")

    # ── Test ──
    if test_loader is not None:
        print("Loading best model for test evaluation...")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics = run_epoch(model, test_loader, processor, device, train=False)
        print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})

        if args.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()