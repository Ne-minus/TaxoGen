"""
Leave-One-Model-Out (LOMO) experiment.

For each of the 12 models:
  1. Remove all train/val pairs where that model appears
  2. Fine-tune CLIP-L (same hyperparams as best checkpoint: large14_bt_ls01_drop05)
  3. Score the held-out model's images on all test wids
  4. Record mean reward

Then compute Spearman correlation of predicted ranks vs human ELO.
Saves results to lomo_results.csv.
"""

import os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from PIL import Image
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_ROOT = "/workspace/TaxoGen_sampled"
TRAIN_CSV   = "/workspace/TaxoGen_sampled/splits/train.csv"
VAL_CSV     = "/workspace/TaxoGen_sampled/splits/val.csv"
TEST_CSV    = "/workspace/TaxoGen_sampled/splits/test.csv"
META_CSV    = "/workspace/TaxoGen/splits/wordnet_meta.csv"
OUT_CSV     = "/workspace/TaxoGen/lomo_results.csv"

MODEL_NAME   = "openai/clip-vit-large-patch14"
LR           = 3e-5
EPOCHS       = 15
BATCH_SIZE   = 16
DROPOUT      = 0.5
UNFREEZE_VIS = 2
UNFREEZE_TXT = 2
LABEL_SMOOTH = 0.1
PATIENCE     = 4
NUM_WORKERS  = 4
SEED         = 42

HUMAN_ELO = {
    'black-forest-labs_flux.1-dev':                        1085,
    'playgroundai_playground-v2.5-1024px-aesthetic':       1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms':               1043,
    'stabilityai_stable-diffusion-xl-base-1.0':            1027,
    'kandinsky-community_kandinsky-3':                     1017,
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':           1013,
    'stabilityai_sdxl-turbo':                              1011,
    'deepfloyd_if-i-xl-v1.0':                              993,
    'stabilityai_stable-diffusion-3-medium-diffusers':      990,
    'retrieval':                                            950,
    'prompthero_openjourney':                               907,
    'runwayml_stable-diffusion-v1-5':                       901,
}

SHORT = {
    'black-forest-labs_flux.1-dev':                        'FLUX',
    'playgroundai_playground-v2.5-1024px-aesthetic':       'Playground',
    'pixart-alpha_pixart-sigma-xl-2-512-ms':               'PixArt',
    'stabilityai_stable-diffusion-xl-base-1.0':            'SDXL-base',
    'kandinsky-community_kandinsky-3':                     'Kandinsky3',
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':           'HunyuanDiT',
    'stabilityai_sdxl-turbo':                              'SDXL-turbo',
    'deepfloyd_if-i-xl-v1.0':                             'DeepFloyd',
    'stabilityai_stable-diffusion-3-medium-diffusers':     'SD3',
    'retrieval':                                           'Retrieval',
    'prompthero_openjourney':                              'Openjourney',
    'runwayml_stable-diffusion-v1-5':                      'SD1.5',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def normalize_label(x):
    if pd.isna(x): return None
    try: x = int(x)
    except: return None
    if x == 0: return 1   # arena 0=A_win → loss 1=A_win
    if x == 1: return 0   # arena 1=B_win → loss 0=B_win
    return None            # drop Tie / BothBad

def build_dir_map():
    d = {}
    for name in os.listdir(IMAGES_ROOT):
        full = os.path.join(IMAGES_ROOT, name)
        if os.path.isdir(full) and name != "splits":
            d[name.lower()] = name
    return d

def find_img(dir_map, model, wid):
    actual = dir_map.get(model.lower(), model)
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(IMAGES_ROOT, actual, f"{wid}{ext}")
        if os.path.exists(p): return p
    return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, df, processor, position_swap=False):
        df = df.copy()
        df["label_id"] = df["result_human_def"].apply(normalize_label)
        df = df[df["label_id"].notna()].reset_index(drop=True)
        self.processor = processor
        self.position_swap = position_swap
        self._dir_map = build_dir_map()
        # keep only rows where both images exist
        mask = [
            find_img(self._dir_map, str(row["model_a"]), str(row["wordnet_id"])) is not None and
            find_img(self._dir_map, str(row["model_b"]), str(row["wordnet_id"])) is not None
            for _, row in df.iterrows()
        ]
        self.df = df[mask].reset_index(drop=True)
        print(f"    pairs: {len(self.df)}/{len(df)} (images found)")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        wid   = str(row["wordnet_id"])
        label = int(row["label_id"])
        prompt = f"An image of {row['core_synset']} ({row['definition']})"
        img_a = Image.open(find_img(self._dir_map, str(row["model_a"]), wid)).convert("RGB")
        img_b = Image.open(find_img(self._dir_map, str(row["model_b"]), wid)).convert("RGB")
        if self.position_swap and random.random() < 0.5:
            img_a, img_b = img_b, img_a
            label = 1 - label
        enc_a = self.processor(text=prompt, images=img_a, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=77)
        enc_b = self.processor(text=prompt, images=img_b, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=77)
        return {
            "ids_a": enc_a["input_ids"].squeeze(0),
            "msk_a": enc_a["attention_mask"].squeeze(0),
            "pix_a": enc_a["pixel_values"].squeeze(0),
            "ids_b": enc_b["input_ids"].squeeze(0),
            "msk_b": enc_b["attention_mask"].squeeze(0),
            "pix_b": enc_b["pixel_values"].squeeze(0),
            "label": label,
        }

def collate(items):
    keys = ["ids_a","msk_a","pix_a","ids_b","msk_b","pix_b"]
    out = {k: torch.stack([x[k] for x in items]) for k in keys}
    out["labels"] = torch.tensor([x["label"] for x in items], dtype=torch.long)
    return out


# ── Model ─────────────────────────────────────────────────────────────────────

class CLIPReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(MODEL_NAME)
        h = self.clip.config.projection_dim
        self.head = nn.Sequential(nn.LayerNorm(h*4), nn.Dropout(DROPOUT), nn.Linear(h*4, 1))

    def freeze(self):
        for p in self.clip.parameters(): p.requires_grad = False
        for p in self.clip.visual_projection.parameters(): p.requires_grad = True
        for p in self.clip.text_projection.parameters():  p.requires_grad = True
        for layer in self.clip.vision_model.encoder.layers[-UNFREEZE_VIS:]:
            for p in layer.parameters(): p.requires_grad = True
        for layer in self.clip.text_model.encoder.layers[-UNFREEZE_TXT:]:
            for p in layer.parameters(): p.requires_grad = True

    def encode(self, ids, msk, pix):
        out = self.clip(input_ids=ids, attention_mask=msk, pixel_values=pix, return_dict=True)
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds,  dim=-1)
        return self.head(torch.cat([img, txt, img*txt, (img-txt).abs()], dim=-1)).squeeze(-1)


# ── Train one fold ─────────────────────────────────────────────────────────────

def train_fold(train_df, val_df, device):
    set_seed(SEED)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("  Building datasets...")
    tr_ds = PairDataset(train_df, processor, position_swap=True)
    va_ds = PairDataset(val_df,   processor, position_swap=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    model = CLIPReward().to(device)
    model.freeze()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_tr:,}")

    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    steps = len(tr_dl) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=steps//10,
                                            num_training_steps=steps)

    best_acc, best_state, patience_left = 0.0, None, PATIENCE
    s = LABEL_SMOOTH

    for epoch in range(1, EPOCHS+1):
        model.train()
        for batch in tr_dl:
            r_a = model.encode(batch["ids_a"].to(device), batch["msk_a"].to(device),
                               batch["pix_a"].to(device))
            r_b = model.encode(batch["ids_b"].to(device), batch["msk_b"].to(device),
                               batch["pix_b"].to(device))
            labels = batch["labels"].to(device)
            signed = torch.where(labels == 1, r_a - r_b, r_b - r_a)
            loss   = (-(1-s)*F.logsigmoid(signed) - s*F.logsigmoid(-signed)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for batch in va_dl:
                r_a = model.encode(batch["ids_a"].to(device), batch["msk_a"].to(device),
                                   batch["pix_a"].to(device))
                r_b = model.encode(batch["ids_b"].to(device), batch["msk_b"].to(device),
                                   batch["pix_b"].to(device))
                preds  = (r_a > r_b).long()
                labels = batch["labels"].to(device)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        val_acc = correct / total
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
            marker = " ✓"
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  epoch {epoch:2d}  val={val_acc:.4f}  early stop")
                break
        print(f"  epoch {epoch:2d}  val={val_acc:.4f}{marker}")

    model.load_state_dict(best_state)
    return model, processor, best_acc


# ── Score held-out model ───────────────────────────────────────────────────────

class ScoreDataset(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        img_path, prompt = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        enc = self.processor(text=prompt, images=img, return_tensors="pt",
                             padding="max_length", truncation=True, max_length=77)
        return {k: enc[k].squeeze(0) for k in enc}

def score_held_out(reward_model, processor, held_out, test_wids, wid_to_prompt, device):
    dir_map = build_dir_map()
    items = []
    for wid in test_wids:
        p = find_img(dir_map, held_out, wid)
        if p and wid in wid_to_prompt:
            items.append((p, wid_to_prompt[wid]))

    ds = ScoreDataset(items, processor)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    rewards = []
    reward_model.eval()
    with torch.no_grad():
        for batch in dl:
            r = reward_model.encode(batch["input_ids"].to(device),
                                    batch["attention_mask"].to(device),
                                    batch["pixel_values"].to(device))
            rewards.extend(r.cpu().numpy().tolist())
    return np.array(rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    meta     = pd.read_csv(META_CSV)
    meta["wordnet_id"] = meta["wordnet_id"].astype(str)
    wid_to_prompt = {
        r["wordnet_id"]: f"An image of {r['core_synset']} ({r['definition']})"
        for _, r in meta.iterrows()
    }
    test_wids = set(test_df["wordnet_id"].astype(str).tolist())

    models  = sorted(HUMAN_ELO.keys())
    results = []

    for i, held_out in enumerate(models, 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] Held-out: {SHORT[held_out]}")
        print(f"{'='*60}")

        tr = train_df[~((train_df.model_a == held_out) | (train_df.model_b == held_out))]
        va = val_df[  ~((val_df.model_a   == held_out) | (val_df.model_b   == held_out))]
        print(f"  train rows: {len(tr)} (removed {len(train_df)-len(tr)})")
        print(f"  val rows:   {len(va)}  (removed {len(val_df)-len(va)})")

        reward_model, processor, best_val_acc = train_fold(tr, va, device)

        rewards  = score_held_out(reward_model, processor, held_out,
                                  test_wids, wid_to_prompt, device)
        mean_r   = float(np.mean(rewards))
        elapsed  = (time.time() - t0) / 60
        print(f"  → mean_reward={mean_r:+.4f}  n={len(rewards)}  "
              f"best_val={best_val_acc:.4f}  time={elapsed:.1f}min")

        results.append({
            "model":        held_out,
            "short":        SHORT[held_out],
            "human_elo":    HUMAN_ELO[held_out],
            "mean_reward":  mean_r,
            "n_scored":     len(rewards),
            "best_val_acc": best_val_acc,
        })

        del reward_model
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df["elo_rank"]    = df["human_elo"].rank(ascending=False).astype(int)
    df["reward_rank"] = df["mean_reward"].rank(ascending=False).astype(int)
    df["rank_delta"]  = df["reward_rank"] - df["elo_rank"]
    df = df.sort_values("elo_rank").reset_index(drop=True)

    rho, p_val = spearmanr(df["human_elo"], df["mean_reward"])

    print(f"\n{'='*60}")
    print(f"LOMO Spearman ρ = {rho:+.3f}  (p = {p_val:.4f})")
    print(f"{'='*60}")
    print(f"\n  {'Model':12s}  ELO   rk_h  rk_pred  Δ    mean_r  val_acc")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        flag = "  ← big" if abs(row["rank_delta"]) >= 3 else ""
        print(f"  {row['short']:12s}  {int(row['human_elo'])}   "
              f"{int(row['elo_rank']):2d}     {int(row['reward_rank']):2d}    "
              f"{int(row['rank_delta']):+3d}   {row['mean_reward']:+.3f}   "
              f"{row['best_val_acc']:.3f}{flag}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
