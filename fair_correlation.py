"""
Fair correlation with human ELO using within-prompt rank across all models.

All models are scored on the same set of wordnet_ids, so raw reward is also
a valid comparison. Within-prompt rank additionally removes inter-prompt
variance (some concepts are inherently harder to generate).

The old approach computed rank within a *pair* (only 2 models), creating bias
from pair composition. Here every wordnet_id has images from ALL models, so
rank is computed across all simultaneously — no pair-selection bias.

Expected input CSV columns: model, wordnet_id, reward
  - one row per (model, wordnet_id)
  - every wordnet_id has the same set of models
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


HUMAN_ELO = {
    'black-forest-labs_flux.1-dev':                         1085,
    'playgroundai_playground-v2.5-1024px-aesthetic':        1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms':                1043,
    'stabilityai_stable-diffusion-xl-base-1.0':             1027,
    'kandinsky-community_kandinsky-3':                      1017,
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':            1013,
    'stabilityai_sdxl-turbo':                               1011,
    'deepfloyd_if-i-xl-v1.0':                               993,
    'stabilityai_stable-diffusion-3-medium-diffusers':       990,
    'retrieval':                                             950,
    'prompthero_openjourney':                                907,
    'runwayml_stable-diffusion-v1-5':                        901,
}

SHORT = {
    'black-forest-labs_flux.1-dev':                        'FLUX',
    'playgroundai_playground-v2.5-1024px-aesthetic':       'Playground',
    'pixart-alpha_pixart-sigma-xl-2-512-ms':               'PixArt',
    'stabilityai_stable-diffusion-xl-base-1.0':            'SDXL',
    'kandinsky-community_kandinsky-3':                     'Kandinsky3',
    'tencent-hunyuan_hunyuandit-v1.2-diffusers':           'HDiT',
    'stabilityai_sdxl-turbo':                              'SDXL-turbo',
    'deepfloyd_if-i-xl-v1.0':                             'DeepFloyd',
    'stabilityai_stable-diffusion-3-medium-diffusers':     'SD3',
    'retrieval':                                           'Retrieval',
    'prompthero_openjourney':                              'Openjourney',
    'runwayml_stable-diffusion-v1-5':                      'SD1.5',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True,
                    help="CSV with columns: model, wordnet_id, reward")
    ap.add_argument("--out_plot", default="fair_correlation.png")
    ap.add_argument("--out_table", default="fair_correlation_table.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    assert {"model", "wordnet_id", "reward"}.issubset(df.columns), \
        "CSV must have columns: model, wordnet_id, reward"

    df["model"] = df["model"].str.lower().str.strip()
    df["wordnet_id"] = df["wordnet_id"].astype(str)

    models = sorted(df["model"].unique())
    wids   = sorted(df["wordnet_id"].unique())
    n_models = df.groupby("wordnet_id")["model"].nunique()
    print(f"Models: {len(models)}")
    print(f"WordNet IDs: {len(wids)}")
    print(f"Models per wid — min: {n_models.min()}, max: {n_models.max()}, median: {n_models.median():.0f}")

    # Keep only wids where ALL models are present (fair comparison)
    full_wids = n_models[n_models == len(models)].index
    dropped = len(wids) - len(full_wids)
    if dropped:
        print(f"Dropping {dropped} wids with incomplete model coverage")
    df = df[df["wordnet_id"].isin(full_wids)].copy()
    print(f"Remaining rows: {len(df)} across {len(full_wids)} wids")

    # ── Fair within-wid rank (all models, not pairs) ──
    # rank 1 = best (highest reward), rank N = worst
    df["rank"] = df.groupby("wordnet_id")["reward"].rank(ascending=False, method="average")
    n = len(models)
    df["rank_pct"] = (df["rank"] - 1) / (n - 1)   # 0 = best, 1 = worst

    # ── Per-model aggregations ──
    agg = df.groupby("model").agg(
        mean_reward  = ("reward",   "mean"),
        median_reward= ("reward",   "median"),
        mean_rank    = ("rank",     "mean"),      # lower = better
        mean_rank_pct= ("rank_pct", "mean"),
        win1_frac    = ("rank",     lambda x: (x == 1).mean()),  # fraction of wids where rank=1
        n_images     = ("reward",   "count"),
    ).reset_index()

    # ── Correlation with human ELO ──
    h = pd.Series(HUMAN_ELO)
    common = [m for m in agg["model"] if m in h.index]
    missing = [m for m in agg["model"] if m not in h.index]
    if missing:
        print(f"\nModels in scores but not in HUMAN_ELO (skipped): {missing}")

    sub = agg[agg["model"].isin(common)].set_index("model")
    elo_vals = h.loc[sub.index]

    metrics = {
        "mean_reward":   ("mean reward",       True),   # True = higher is better
        "median_reward": ("median reward",      True),
        "mean_rank":     ("mean rank (1=best)", False),  # False = lower is better
        "win1_frac":     ("win@1 fraction",     True),
    }

    print(f"\n{'Metric':22s}  Spearman-ρ  p-value  Pearson-r  p-value")
    print("-" * 65)
    results = {}
    for col, (label, higher_better) in metrics.items():
        vals = sub[col] if higher_better else -sub[col]
        rho, p_sp = spearmanr(elo_vals, sub[col])
        r, p_pe   = pearsonr(elo_vals, sub[col])
        results[col] = dict(label=label, rho=rho, p_sp=p_sp, r=r, p_pe=p_pe)
        print(f"  {label:20s}   {rho:+.3f}     {p_sp:.4f}   {r:+.3f}     {p_pe:.4f}")

    # ── Scatter plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (col, (label, _)) in zip(axes, metrics.items()):
        xs = elo_vals.values
        ys = sub[col].values
        rho = results[col]["rho"]
        p   = results[col]["p_sp"]

        ax.scatter(xs, ys, s=120, c="steelblue", edgecolor="black", zorder=3)
        for m in sub.index:
            name = SHORT.get(m, m.split("_", 1)[-1])
            ax.annotate(name, (elo_vals[m], sub.loc[m, col]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)

        z = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min() - 15, xs.max() + 15, 100)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.5)
        ax.set_xlabel("Human ELO")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Human ELO\nSpearman ρ = {rho:+.3f}  (p = {p:.4f})")
        ax.grid(alpha=0.3)

    plt.suptitle("Fair correlation: all models ranked per wordnet_id (no pair bias)", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"\nSaved plot → {args.out_plot}")

    # ── Full ranking table ──
    out = sub.copy()
    out["human_elo"] = elo_vals
    out["rk_human"]  = elo_vals.rank(ascending=False).astype(int)
    out["rk_model"]  = out["mean_rank"].rank(ascending=True).astype(int)   # lower rank = better
    out["rk_delta"]  = out["rk_model"] - out["rk_human"]
    out = out.sort_values("rk_human")

    print(f"\n{'model':52s}  ELO  rk_h  rk_m  Δ  mean_rank  win@1")
    for m, r in out.iterrows():
        name = SHORT.get(m, m)
        flag = "  <-- big" if abs(r["rk_delta"]) >= 3 else ""
        print(f"  {m:50s}  {int(r['human_elo']):>4d}   {int(r['rk_human']):>2d}    {int(r['rk_model']):>2d}  {int(r['rk_delta']):+3d}  "
              f"{r['mean_rank']:7.2f}  {r['win1_frac']:.3f}{flag}")

    out.to_csv(args.out_table, index=True)
    print(f"Saved table  → {args.out_table}")


if __name__ == "__main__":
    main()
