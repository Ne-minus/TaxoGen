"""
Compute MLE ELO from model reward scores (scores_fixed.csv) and plot
in the same style as the Human Preference chart.

For each wordnet_id, all pairwise comparisons are derived from reward ranking.
Bootstrap resamples wordnet_ids for 95% CI.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import math
from tqdm import tqdm

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

HUMAN_ELO = {
    'black-forest-labs_flux.1-dev': 1085,
    'playgroundai_playground-v2.5-1024px-aesthetic': 1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms': 1043,
    'stabilityai_stable-diffusion-xl-base-1.0': 1027,
    'kandinsky-community_kandinsky-3': 1017,
    'tencent-hunyuan_hunyuandit-v1.2-diffusers': 1013,
    'stabilityai_sdxl-turbo': 1011,
    'deepfloyd_if-i-xl-v1.0': 993,
    'stabilityai_stable-diffusion-3-medium-diffusers': 990,
    'retrieval': 950,
    'prompthero_openjourney': 907,
    'runwayml_stable-diffusion-v1-5': 901,
}


def make_battles(df):
    """Generate all pairwise comparisons from per-image rewards within each wid."""
    rows = []
    for wid, g in df.groupby('wordnet_id'):
        g = g.set_index('model')['reward']
        models = g.index.tolist()
        for i, ma in enumerate(models):
            for mb in models[i+1:]:
                ra, rb = g[ma], g[mb]
                if ra > rb:
                    result = 0   # arena: 0 = A_win
                elif rb > ra:
                    result = 1   # arena: 1 = B_win
                else:
                    result = 2   # tie
                rows.append({'model_a': ma, 'model_b': mb,
                             'result': result, 'wordnet_id': wid})
    return pd.DataFrame(rows)


def compute_mle_elo(battles, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([battles['model_a'], battles['model_b']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    df2 = pd.concat([battles, battles], ignore_index=True)
    p, n = len(models), len(df2)

    X = np.zeros([n, p])
    X[np.arange(n), models[df2['model_a']]] = +math.log(BASE)
    X[np.arange(n), models[df2['model_b']]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df2['result'] == 0] = 1.0          # A wins
    tie_idx = df2['result'] == 2
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, C=1e9, tol=1e-8, max_iter=1000)
    lr.fit(X, Y)
    elo = SCALE * lr.coef_[0] + INIT_RATING
    return pd.Series(elo, index=models.index).sort_values(ascending=False)


def bootstrap_elo(battles, wids, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    # Pre-index by wid to avoid O(n) filtering per iteration
    wid_groups = {w: g for w, g in battles.groupby('wordnet_id')}
    rows = []
    for _ in tqdm(range(n_boot), desc='bootstrap'):
        sampled = rng.choice(wids, size=len(wids), replace=True)
        boot = pd.concat([wid_groups[w] for w in sampled], ignore_index=True)
        try:
            rows.append(compute_mle_elo(boot))
        except Exception:
            pass
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores_csv', default='scores_fixed.csv')
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--out', default='model_elo_plot.png')
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    df['model'] = df['model'].str.lower().str.strip()
    df['wordnet_id'] = df['wordnet_id'].astype(str)

    # Keep only complete wids
    n_models = df.groupby('wordnet_id')['model'].nunique().max()
    full_wids = df.groupby('wordnet_id')['model'].nunique()
    full_wids = full_wids[full_wids == n_models].index
    df = df[df['wordnet_id'].isin(full_wids)].copy()
    print(f"Wids: {len(full_wids)}  Models: {n_models}")

    print("Building pairwise battles...")
    battles = make_battles(df)
    print(f"Battles: {len(battles)}")

    print("Computing point-estimate ELO...")
    elo_point = compute_mle_elo(battles)

    print(f"Bootstrapping (n={args.n_boot})...")
    boot_df = bootstrap_elo(battles, full_wids.tolist(), n_boot=args.n_boot)

    # Sort by point ELO descending
    order = elo_point.sort_values(ascending=False).index.tolist()
    elo_point = elo_point.loc[order]

    lo  = boot_df[order].quantile(0.025)
    hi  = boot_df[order].quantile(0.975)
    med = boot_df[order].median()

    # Spearman vs human ELO
    h = pd.Series(HUMAN_ELO)
    common = [m for m in order if m in h.index]
    rho, _ = spearmanr(h[common], elo_point[common])
    print(f"\nSpearman ρ vs human ELO: {rho:+.3f}")

    print(f"\n{'Model':14s}  ELO    95% CI")
    for m in order:
        name = SHORT.get(m, m)
        print(f"  {name:12s}  {elo_point[m]:6.1f}  [{lo[m]:6.1f}, {hi[m]:6.1f}]")

    # ── Plot ──
    labels = [SHORT.get(m, m) for m in order]
    x = np.arange(len(order))
    scores = elo_point.loc[order].values
    err_lo = scores - lo.loc[order].values
    err_hi = hi.loc[order].values - scores

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(x, scores,
                yerr=[err_lo, err_hi],
                fmt='none',
                ecolor='#4a9e8e',
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=2)

    ax.scatter(x, scores,
               marker='*',
               s=180,
               color='#4a9e8e',
               zorder=3,
               linewidths=0.5,
               edgecolors='#2d6e61')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('ELO Score', fontsize=11)
    ax.set_title('Model Prediction (CLIP reward)', fontsize=12)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax.grid(axis='y', which='minor', alpha=0.15, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    # Annotate rho
    ax.text(0.98, 0.05, f'Spearman ρ = {rho:+.3f} vs Human ELO',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()