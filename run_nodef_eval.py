"""
Post-training pipeline for nodef CLIP model:
  1. Score all (model, wid) from test split using collected_wordnet_images
  2. Bootstrap correlation with human ELO
  3. Plot: model ranking (ELO vs model rank)
  4. Plot: reward distribution on Win / Tie / BothBad pairs
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from scipy.stats import spearmanr

# ── Human ELO ──────────────────────────────────────────────────────────────────
HUMAN_ELO = {
    'black-forest-labs_flux.1-dev':                        1085,
    'playgroundai_playground-v2.5-1024px-aesthetic':       1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms':               1043,
    'stabilityai_stable-diffusion-xl-base-1.0':            1027,
    'kandinsky-community_kandinsky-3':                     1017,
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':           1013,
    'stabilityai_sdxl-turbo':                              1011,
    'deepfloyd_if-i-xl-v1.0':                             993,
    'stabilityai_stable-diffusion-3-medium-diffusers':     990,
    'retrieval':                                           950,
    'prompthero_openjourney':                              907,
    'runwayml_stable-diffusion-v1-5':                      901,
}
SHORT = {
    'black-forest-labs_flux.1-dev':                       'FLUX',
    'playgroundai_playground-v2.5-1024px-aesthetic':      'Playground',
    'pixart-alpha_pixart-sigma-xl-2-512-ms':              'PixArt',
    'stabilityai_stable-diffusion-xl-base-1.0':           'SDXL',
    'kandinsky-community_kandinsky-3':                    'Kandinsky3',
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':          'HDiT',
    'stabilityai_sdxl-turbo':                             'SDXL-turbo',
    'deepfloyd_if-i-xl-v1.0':                            'DeepFloyd',
    'stabilityai_stable-diffusion-3-medium-diffusers':    'SD3',
    'retrieval':                                          'Retrieval',
    'prompthero_openjourney':                             'Openjourney',
    'runwayml_stable-diffusion-v1-5':                     'SD1.5',
}
LABEL_NAMES = {0: 'B_win', 1: 'A_win', 2: 'Tie', 3: 'BothBad'}


# ── Model ──────────────────────────────────────────────────────────────────────
class CLIPRewardModel(nn.Module):
    def __init__(self, model_name, dropout=0.0):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        h = self.clip.config.projection_dim
        self.reward_head = nn.Sequential(
            nn.LayerNorm(h * 4), nn.Dropout(dropout), nn.Linear(h * 4, 1)
        )

    def encode(self, input_ids, attention_mask, pixel_values):
        out = self.clip(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, return_dict=True)
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds, dim=-1)
        return self.reward_head(
            torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        ).squeeze(-1)


# ── Dataset for scoring ────────────────────────────────────────────────────────
class ScoreDS(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.proc = processor

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, model, wid, prompt = self.items[i]
        img = Image.open(path).convert("RGB")
        enc = self.proc(text=prompt, images=img, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=77)
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "pixel_values":   enc["pixel_values"].squeeze(0),
                "model": model, "wid": wid}


def collate(batch):
    return {"input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
            "model":          [b["model"] for b in batch],
            "wid":            [b["wid"]   for b in batch]}


# ── Bootstrap correlation ──────────────────────────────────────────────────────
def compute_stats(df, metric='median_reward'):
    df = df.copy()
    df['rank'] = df.groupby('wordnet_id')['reward'].rank(ascending=False, method='average')
    agg = df.groupby('model').agg(
        mean_reward  =('reward', 'mean'),
        median_reward=('reward', 'median'),
        mean_rank    =('rank',   'mean'),
        win1_frac    =('rank',   lambda x: (x == 1).mean()),
    )
    h = pd.Series(HUMAN_ELO)
    common = [m for m in agg.index if m in h.index]
    rho, _ = spearmanr(h[common], agg.loc[common, metric])
    return rho, agg.loc[common]


def bootstrap(df, metric='median_reward', n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    wids = df['wordnet_id'].unique()
    rhos = []
    for _ in range(n_boot):
        frames = [df[df['wordnet_id'] == w] for w in rng.choice(wids, size=len(wids), replace=True)]
        boot = pd.concat(frames, ignore_index=True)
        try:
            rho, _ = compute_stats(boot, metric)
            rhos.append(rho)
        except Exception:
            pass
    return np.array(rhos)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',        required=True)
    ap.add_argument('--images_root', default='/workspace/collected_wordnet_images')
    ap.add_argument('--test_csv',    default='splits/test.csv')
    ap.add_argument('--out_scores',  default='scores_clip_nodef_test_full.csv')
    ap.add_argument('--out_plot',    default='plots/nodef_eval.png')
    ap.add_argument('--n_boot',      type=int, default=1000)
    ap.add_argument('--batch_size',  type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--skip_scoring', action='store_true',
                    help='Use existing out_scores CSV, skip scoring step')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_plot) or '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df = pd.read_csv(args.test_csv)

    # ── 1. Scoring ─────────────────────────────────────────────────────────────
    if not args.skip_scoring:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model_name = ckpt.get('model_name', 'openai/clip-vit-large-patch14')
        no_def = ckpt.get('args', {}).get('no_definition', False)
        print(f"Backbone: {model_name}  |  no_definition: {no_def}  |  epoch: {ckpt.get('epoch')}")

        processor = CLIPProcessor.from_pretrained(model_name)
        reward_model = CLIPRewardModel(model_name)
        reward_model.load_state_dict(ckpt['model_state_dict'])
        reward_model.eval().to(device)

        dir_map = {d.lower(): d for d in os.listdir(args.images_root)
                   if os.path.isdir(os.path.join(args.images_root, d))}

        def find_img(model, wid):
            actual = dir_map.get(model.lower(), model)
            for ext in ('.png', '.jpg', '.jpeg'):
                p = os.path.join(args.images_root, actual, f"{wid}{ext}")
                if os.path.exists(p): return p
            return None

        # build prompt per wid
        wid_meta = test_df[['wordnet_id', 'core_synset', 'definition']].drop_duplicates('wordnet_id')
        if no_def:
            wid_prompt = {str(r.wordnet_id): f"An image of {r.core_synset}"
                          for r in wid_meta.itertuples()}
        else:
            wid_prompt = {str(r.wordnet_id): f"An image of {r.core_synset} ({r.definition})"
                          for r in wid_meta.itertuples()}

        # all models × all test wids
        all_models = sorted(dir_map.keys())
        all_wids   = sorted(wid_prompt.keys())
        items = {}
        for m in all_models:
            for wid in all_wids:
                key = (m, wid)
                p = find_img(m, wid)
                if p:
                    items[key] = (p, wid_prompt[wid])

        items_list = [(path, m, wid, prompt) for (m, wid), (path, prompt) in items.items()]
        print(f"Items to score: {len(items_list)}")

        loader = DataLoader(ScoreDS(items_list, processor), batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

        rows = []
        with torch.no_grad():
            for batch in loader:
                rewards = reward_model.encode(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['pixel_values'].to(device),
                ).cpu().tolist()
                for m, wid, r in zip(batch['model'], batch['wid'], rewards):
                    rows.append({'model': m, 'wordnet_id': int(wid), 'reward': r})

        scores = pd.DataFrame(rows)
        scores.to_csv(args.out_scores, index=False)
        print(f"Saved {len(scores)} rows → {args.out_scores}")
    else:
        scores = pd.read_csv(args.out_scores)
        print(f"Loaded {len(scores)} rows from {args.out_scores}")

    scores['model'] = scores['model'].str.lower().str.strip()
    scores['wordnet_id'] = scores['wordnet_id'].astype(str)

    # keep only wids where ALL models present
    n_models = scores.groupby('wordnet_id')['model'].nunique()
    n = n_models.max()
    full_wids = n_models[n_models == n].index
    df = scores[scores['wordnet_id'].isin(full_wids)].copy()
    print(f"Full coverage: {len(full_wids)} wids × {n} models = {len(df)} rows")

    # ── 2. Bootstrap correlation ────────────────────────────────────────────────
    metrics = {
        'median_reward': 'Median reward',
        'mean_reward':   'Mean reward',
        'mean_rank':     'Mean rank (1=best)',
        'win1_frac':     'Win@1 fraction',
    }
    print(f"\n{'Metric':22s}  ρ point   95% CI            n_boot={args.n_boot}")
    print('-' * 60)
    results = {}
    for col, label in metrics.items():
        rho_point, _ = compute_stats(df, col)
        rhos = bootstrap(df, col, args.n_boot)
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        results[col] = dict(label=label, rho=rho_point, lo=lo, hi=hi, rhos=rhos)
        print(f"  {label:20s}  {rho_point:+.3f}    [{lo:+.3f}, {hi:+.3f}]")

    _, agg = compute_stats(df, 'median_reward')
    h = pd.Series(HUMAN_ELO)
    agg['elo']    = h
    agg['rk_elo'] = agg['elo'].rank(ascending=False).astype(int)
    agg['rk_mod'] = agg['median_reward'].rank(ascending=False).astype(int)
    agg['delta']  = agg['rk_mod'] - agg['rk_elo']
    agg = agg.sort_values('rk_elo')

    print(f"\n{'Model':14s}  ELO  rk_ELO  rk_model  Δ   median_reward  win@1")
    print('-' * 72)
    for m, r in agg.iterrows():
        flag = ' ←' if abs(r['delta']) >= 3 else ''
        print(f"  {SHORT.get(m,m):12s}  {int(r.elo):4d}     {int(r.rk_elo):2d}        {int(r.rk_mod):2d}   "
              f"{int(r.delta):+2d}   {r.median_reward:+.3f}          {r.win1_frac:.3f}{flag}")

    # ── 3+4. Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot A: bootstrap CI bars ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    labels_   = [results[c]['label'] for c in metrics]
    rhos_pt   = [results[c]['rho']   for c in metrics]
    lo_err    = [results[c]['rho'] - results[c]['lo'] for c in metrics]
    hi_err    = [results[c]['hi']  - results[c]['rho'] for c in metrics]
    colors_a  = ['steelblue' if r > 0 else 'salmon' for r in rhos_pt]
    bars = ax_a.barh(labels_, rhos_pt, xerr=[lo_err, hi_err],
                     color=colors_a, edgecolor='black', alpha=0.8,
                     error_kw=dict(ecolor='black', capsize=5, linewidth=1.5))
    ax_a.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_a.set_xlabel('Spearman ρ')
    ax_a.set_title(f'Correlation with Human ELO\n95% CI (bootstrap n={args.n_boot})')
    ax_a.set_xlim(-1.05, 1.05)
    for bar, rho in zip(bars, rhos_pt):
        ax_a.text(rho + 0.02, bar.get_y() + bar.get_height() / 2,
                  f'{rho:+.3f}', va='center', ha='left', fontsize=10)
    ax_a.grid(axis='x', alpha=0.3)

    # ── Plot B: ranking comparison ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    order = agg.sort_values('rk_elo').index.tolist()
    x = np.arange(len(order))
    ax_b.plot(x, agg.loc[order, 'rk_elo'].values,  'o-', label='Human ELO',
              color='steelblue', linewidth=2, markersize=7)
    ax_b.plot(x, agg.loc[order, 'rk_mod'].values, 's--', label='CLIP nodef',
              color='coral', linewidth=2, markersize=7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SHORT.get(m, m) for m in order], rotation=35, ha='right')
    ax_b.set_ylabel('Rank (1 = best)')
    ax_b.set_title('Human ELO rank vs CLIP nodef rank')
    ax_b.legend()
    ax_b.invert_yaxis()
    ax_b.grid(alpha=0.3)

    # ── Plot C: reward dist by label (Win/Tie/BothBad) ─────────────────────────
    # Join scores back to test pairs to get rewards for both model_a and model_b
    score_map = scores.set_index(['model', 'wordnet_id'])['reward'].to_dict()
    rows_tie = []
    for _, row in test_df.iterrows():
        wid   = str(row['wordnet_id'])
        label = int(row['result_human_def']) if not pd.isna(row.get('result_human_def')) else -1
        if label not in [0, 1, 2, 3]: continue
        ra = score_map.get((str(row['model_a']).lower(), wid), np.nan)
        rb = score_map.get((str(row['model_b']).lower(), wid), np.nan)
        if np.isnan(ra) or np.isnan(rb): continue
        rows_tie.append({'label': label, 'label_name': LABEL_NAMES[label],
                         'reward_a': ra, 'reward_b': rb, 'diff': ra - rb})
    pair_df = pd.DataFrame(rows_tie)

    ax_c = fig.add_subplot(gs[1, 0])
    colors_c = {'A_win': 'steelblue', 'B_win': 'coral', 'Tie': 'gold', 'BothBad': 'gray'}
    bins = np.linspace(pair_df['diff'].quantile(0.01), pair_df['diff'].quantile(0.99), 40)
    for lname, grp in pair_df.groupby('label_name'):
        ax_c.hist(grp['diff'], bins=bins, alpha=0.5, label=f"{lname} (n={len(grp)})",
                  color=colors_c.get(lname, 'purple'), density=True)
    ax_c.axvline(0, color='black', linewidth=1, linestyle='--')
    ax_c.set_xlabel('reward_A − reward_B')
    ax_c.set_ylabel('Density')
    ax_c.set_title('Reward diff distribution by label')
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.3)

    # ── Plot D: mean reward per label ──────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    # show individual rewards (not diff) for winner vs loser vs tie
    rewards_by_cat = []
    for _, row in pair_df.iterrows():
        lname = row['label_name']
        if lname == 'A_win':
            rewards_by_cat.append({'category': 'Winner', 'reward': row['reward_a']})
            rewards_by_cat.append({'category': 'Loser',  'reward': row['reward_b']})
        elif lname == 'B_win':
            rewards_by_cat.append({'category': 'Winner', 'reward': row['reward_b']})
            rewards_by_cat.append({'category': 'Loser',  'reward': row['reward_a']})
        elif lname == 'Tie':
            rewards_by_cat.append({'category': 'Tie (A)', 'reward': row['reward_a']})
            rewards_by_cat.append({'category': 'Tie (B)', 'reward': row['reward_b']})
        elif lname == 'BothBad':
            rewards_by_cat.append({'category': 'BothBad (A)', 'reward': row['reward_a']})
            rewards_by_cat.append({'category': 'BothBad (B)', 'reward': row['reward_b']})
    cat_df = pd.DataFrame(rewards_by_cat)
    order_cat = ['Winner', 'Loser', 'Tie (A)', 'Tie (B)', 'BothBad (A)', 'BothBad (B)']
    cat_colors = {'Winner': 'steelblue', 'Loser': 'coral',
                  'Tie (A)': 'gold', 'Tie (B)': 'gold',
                  'BothBad (A)': 'gray', 'BothBad (B)': 'gray'}
    present = [c for c in order_cat if c in cat_df['category'].values]
    positions = np.arange(len(present))
    for i, cat in enumerate(present):
        data = cat_df[cat_df['category'] == cat]['reward'].values
        ax_d.violinplot(data, positions=[i], showmedians=True, showextrema=False)
        ax_d.scatter([i] * len(data), data, alpha=0.05, s=5,
                     color=cat_colors.get(cat, 'purple'))
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(present, rotation=20, ha='right')
    ax_d.set_ylabel('Reward')
    ax_d.set_title('Reward by outcome category')
    ax_d.grid(axis='y', alpha=0.3)

    plt.suptitle('CLIP Reward Model (w/o def) — Test set evaluation', fontsize=13, y=1.01)
    plt.savefig(args.out_plot, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {args.out_plot}')


if __name__ == '__main__':
    main()
