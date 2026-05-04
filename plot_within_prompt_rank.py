"""Plot within-prompt rank per model and a scatter against ELO."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

scores = pd.read_csv('test_scores_fixed.csv')

# Per-row pct rank within each wordnet_id (smaller = won the prompt)
scores['prompt_rank'] = scores.groupby('wordnet_id')['reward'].rank(ascending=False, pct=True)

ELO = {
    'black-forest-labs_flux.1-dev': 1085, 'playgroundai_playground-v2.5-1024px-aesthetic': 1058,
    'pixart-alpha_pixart-sigma-xl-2-512-ms': 1043, 'stabilityai_stable-diffusion-xl-base-1.0': 1027,
    'kandinsky-community_kandinsky-3': 1017, 'tencent-hunyuan_hunyuandit-v1.2-diffusers': 1013,
    'stabilityai_sdxl-turbo': 1011, 'deepfloyd_if-i-xl-v1.0': 993,
    'stabilityai_stable-diffusion-3-medium-diffusers': 990, 'retrieval': 950,
    'prompthero_openjourney': 907, 'runwayml_stable-diffusion-v1-5': 901,
}

short = {
    'black-forest-labs_flux.1-dev': 'FLUX',
    'playgroundai_playground-v2.5-1024px-aesthetic': 'Playground',
    'pixart-alpha_pixart-sigma-xl-2-512-ms': 'PixArt',
    'stabilityai_stable-diffusion-xl-base-1.0': 'SDXL',
    'kandinsky-community_kandinsky-3': 'Kandinsky3',
    'tencent-hunyuan_hunyuandit-v1.2-diffusers': 'HDiT',
    'stabilityai_sdxl-turbo': 'SDXL-turbo',
    'deepfloyd_if-i-xl-v1.0': 'DeepFloyd',
    'stabilityai_stable-diffusion-3-medium-diffusers': 'SD3',
    'retrieval': 'Retrieval',
    'prompthero_openjourney': 'Openjourney',
    'runwayml_stable-diffusion-v1-5': 'SD1.5',
}

# Win-fraction = fraction of prompts where the model has the lower (=better) prompt_rank
# For 2-image prompts: prompt_rank is 0.5 (winner) or 1.0 (loser).
# Win-fraction per model:
def win_score(g):
    # In 2-way prompts: rank 0.5 = won, rank 1.0 = lost
    # We define "win_fraction" = mean over rows of (1.0 if rank==min_rank_in_prompt else 0.0)
    return (g == g.min()).mean() if len(g) > 0 else np.nan

mean_rank = scores.groupby('model')['prompt_rank'].mean()         # raw mean (smaller = better)
win_frac  = 1 - mean_rank                                          # bigger = better, range ~[0, 0.5]
elo_s = pd.Series(ELO)

# Sort models by win_frac descending (best on left)
order = win_frac.sort_values(ascending=False).index.tolist()

# ─── Plot 1: per-model boxplot of within-prompt rank ───
# Each image contributes its rank (0.5 = won the pair, 1.0 = lost)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
data = [scores.loc[scores['model']==m, 'prompt_rank'].values for m in order]
labels = [short[m] for m in order]
bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=6))
cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(order)))
for patch, c in zip(bp['boxes'], cmap):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_ylabel('Within-prompt rank  (0.5 = won, 1.0 = lost)')
ax.set_xlabel('Model (sorted by mean win rate, best → worst)')
ax.set_title('Within-prompt rank distribution per model\n(lower = better)')
ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.5, label='neutral (50/50)')
ax.legend(loc='center right')
plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
ax.grid(axis='y', alpha=0.3)

# ─── Plot 2: scatter ELO vs win_fraction ───
ax = axes[1]
common = list(elo_s.index)
xs = elo_s.loc[common].values
ys = win_frac.loc[common].values
rho, p = spearmanr(xs, ys)
ax.scatter(xs, ys, s=120, c='steelblue', edgecolor='black', zorder=3)
for m in common:
    ax.annotate(short[m], (elo_s[m], win_frac[m]),
                xytext=(6, 4), textcoords='offset points', fontsize=9)
# Trend
z = np.polyfit(xs, ys, 1)
xline = np.linspace(xs.min()-10, xs.max()+10, 100)
ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5, label=f'fit')
ax.set_xlabel('Human ELO score')
ax.set_ylabel('Mean within-prompt win fraction')
ax.set_title(f'Model win-fraction vs human ELO\nSpearman ρ = {rho:.3f}  (p = {p:.4f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('within_prompt_rank.png', dpi=150)
print("Saved within_prompt_rank.png")

# Print table
print(f"\n{'model':52s}  win_frac  mean_rank  ELO  ELO_rk  model_rk")
df = pd.DataFrame({'win_frac': win_frac, 'mean_rank': mean_rank, 'elo': elo_s})
df['rk_h'] = df['elo'].rank(ascending=False).astype(int)
df['rk_m'] = df['win_frac'].rank(ascending=False).astype(int)
df = df.sort_values('rk_h')
for m, r in df.iterrows():
    print(f"{m:52s}   {r['win_frac']:.3f}    {r['mean_rank']:.3f}  {int(r['elo']):>4d}    {int(r['rk_h']):>2d}      {int(r['rk_m']):>2d}")
