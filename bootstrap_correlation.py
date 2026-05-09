"""
Bootstrap confidence intervals for Spearman correlation (model rank vs human ELO).
Resamples wordnet_ids (prompt-level bootstrap) to preserve per-model structure.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

HUMAN_ELO = {
    'black-forest-labs_flux.1-dev':                        1085,
    'playgroundai_playground-v2.5-1024px-aesthetic':       1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms':               1043,
    'stabilityai_stable-diffusion-xl-base-1.0':            1027,
    'kandinsky-community_kandinsky-3':                     1017,
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':           1013,
    'stabilityai_sdxl-turbo':                              1011,
    'deepfloyd_if-i-xl-v1.0':                              993,
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


def compute_stats(df, metric='median_reward'):
    """Compute per-model metric and Spearman rho vs ELO."""
    df = df.copy()
    df['rank'] = df.groupby('wordnet_id')['reward'].rank(ascending=False, method='average')

    agg = df.groupby('model').agg(
        mean_reward  = ('reward', 'mean'),
        median_reward= ('reward', 'median'),
        mean_rank    = ('rank',   'mean'),
        win1_frac    = ('rank',   lambda x: (x == 1).mean()),
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
        sampled_wids = rng.choice(wids, size=len(wids), replace=True)
        boot = df[df['wordnet_id'].isin(sampled_wids)]
        # resample with replacement at prompt level
        frames = [df[df['wordnet_id'] == w] for w in sampled_wids]
        boot = pd.concat(frames, ignore_index=True)
        try:
            rho, _ = compute_stats(boot, metric)
            rhos.append(rho)
        except Exception:
            pass
    return np.array(rhos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores_csv', default='scores_fixed.csv')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--out_plot', default='bootstrap_correlation.png')
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    df['model'] = df['model'].str.lower().str.strip()
    df['wordnet_id'] = df['wordnet_id'].astype(str)

    # Keep only wids where all models present
    n_models = df.groupby('wordnet_id')['model'].nunique()
    n = n_models.max()
    full_wids = n_models[n_models == n].index
    df = df[df['wordnet_id'].isin(full_wids)].copy()
    print(f"Wids: {len(full_wids)}  |  Models: {n}  |  Rows: {len(df)}")

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

    # ── Point-estimate ranking table ──
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

    # ── Plot: bootstrap distributions ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: CI bar chart
    ax = axes[0]
    labels = [results[c]['label'] for c in metrics]
    rhos_pt = [results[c]['rho'] for c in metrics]
    lo_err  = [results[c]['rho'] - results[c]['lo'] for c in metrics]
    hi_err  = [results[c]['hi'] - results[c]['rho'] for c in metrics]
    colors  = ['steelblue' if r > 0 else 'salmon' for r in rhos_pt]
    bars = ax.barh(labels, rhos_pt, xerr=[lo_err, hi_err],
                   color=colors, edgecolor='black', alpha=0.8,
                   error_kw=dict(ecolor='black', capsize=5, linewidth=1.5))
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Spearman ρ')
    ax.set_title(f'Correlation with Human ELO\n95% CI (prompt bootstrap, n={args.n_boot})')
    ax.set_xlim(-1.05, 1.05)
    for bar, rho in zip(bars, rhos_pt):
        ax.text(rho + (0.02 if rho >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                f'{rho:+.3f}', va='center', ha='left' if rho >= 0 else 'right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Right: ranking comparison
    ax = axes[1]
    order = agg.sort_values('rk_elo').index.tolist()
    x = np.arange(len(order))
    ax.plot(x, agg.loc[order, 'rk_elo'].values,  'o-', label='Human ELO rank', color='steelblue', linewidth=2)
    ax.plot(x, agg.loc[order, 'rk_mod'].values, 's--', label='Model rank',      color='coral',    linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(m, m) for m in order], rotation=35, ha='right')
    ax.set_ylabel('Rank (1 = best)')
    ax.set_title('Human ELO rank vs Model rank')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f'\nSaved → {args.out_plot}')


if __name__ == '__main__':
    main()