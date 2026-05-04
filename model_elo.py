"""
Compute model-side ELO from reward predictions on test pairs,
properly accounting for opponent strength.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

scores = pd.read_csv('test_scores_fixed.csv')
test = pd.read_csv('splits/test.csv')

# (wid, model) -> reward map
sk = {(str(r['wordnet_id']), r['model']): r['reward'] for _, r in scores.iterrows()}

# For each row of test.csv, build a "match" using the model's predicted reward diff
matches = []
for _, r in test.iterrows():
    wid = str(r['wordnet_id'])
    a, b = str(r['model_a']), str(r['model_b'])
    if (wid, a) in sk and (wid, b) in sk:
        ra, rb = sk[(wid, a)], sk[(wid, b)]
        matches.append((a, b, ra, rb))

print(f"Test matches: {len(matches)}")

models = sorted(set([m for ma, mb, *_ in matches for m in (ma, mb)]))

# ─── Method 1: ELO from binary winner (sign of reward diff) ───
def elo_binary(matches, models, K=16, n_passes=50):
    rating = {m: 1000.0 for m in models}
    rng = np.random.default_rng(0)
    for _ in range(n_passes):
        idx = rng.permutation(len(matches))
        for i in idx:
            a, b, ra, rb = matches[i]
            s_a = 1.0 if ra > rb else 0.0
            e_a = 1 / (1 + 10 ** ((rating[b] - rating[a]) / 400))
            rating[a] += K * (s_a - e_a)
            rating[b] -= K * (s_a - e_a)
    return rating

# ─── Method 2: Bradley-Terry MLE on soft predictions ───
# P(a beats b) = σ(reward_a - reward_b)  -- model's confidence
# Fit skill[m] s.t. σ(skill_a - skill_b) ≈ σ(reward_a - reward_b) on all matches
def bt_mle(matches, models, n_iter=300, lr=0.05):
    s = np.zeros(len(models))
    idx = {m: i for i, m in enumerate(models)}
    pred_p = np.array([1 / (1 + np.exp(-(ra - rb))) for _, _, ra, rb in matches])
    a_idx = np.array([idx[a] for a, _, _, _ in matches])
    b_idx = np.array([idx[b] for _, b, _, _ in matches])
    for _ in range(n_iter):
        diff = s[a_idx] - s[b_idx]
        p = 1 / (1 + np.exp(-diff))
        grad = p - pred_p     # gradient of log-loss
        gs = np.zeros(len(models))
        np.add.at(gs, a_idx,  grad)
        np.add.at(gs, b_idx, -grad)
        s -= lr * gs / len(matches)
        s -= s.mean()         # center
    # Convert BT skill to ELO scale: ELO = 1000 + s * 400/ln(10)
    return {m: 1000 + s[idx[m]] * 400 / np.log(10) for m in models}

elo_bin = elo_binary(matches, models)
elo_bt  = bt_mle(matches, models)

# Human ELO from chart
HUMAN_ELO = {
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

med = scores.groupby('model')['reward'].median()
scores['prompt_rank'] = scores.groupby('wordnet_id')['reward'].rank(ascending=False, pct=True)
win_frac = 1 - scores.groupby('model')['prompt_rank'].mean()

common = list(HUMAN_ELO.keys())
h = pd.Series(HUMAN_ELO)

methods = {
    'median':       med,
    'win_frac':     win_frac,
    'ELO (binary)': pd.Series(elo_bin),
    'ELO (BT MLE)': pd.Series(elo_bt),
}
print(f"\n{'method':18s}  Spearman vs human ELO")
print("-"*45)
for name, s in methods.items():
    rho, p = spearmanr(s.loc[common], h.loc[common])
    print(f"  {name:16s}  ρ = {rho:.3f}  (p = {p:.4f})")

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, (name, s) in zip(axes, [('win_frac', win_frac), ('ELO (BT MLE)', pd.Series(elo_bt))]):
    xs = h.loc[common].values
    ys = s.loc[common].values
    rho, p = spearmanr(xs, ys)
    ax.scatter(xs, ys, s=120, c='steelblue', edgecolor='black', zorder=3)
    for m in common:
        ax.annotate(short[m], (h[m], s[m]),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)
    z = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min()-10, xs.max()+10, 100)
    ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5)
    ax.set_xlabel('Human ELO')
    ax.set_ylabel(name)
    ax.set_title(f'{name} vs human ELO\nSpearman ρ = {rho:.3f}')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_elo_comparison.png', dpi=150)
print("\nSaved model_elo_comparison.png")

# Print full table for BT MLE
print(f"\n{'model':52s}  human_ELO  model_BT  rk_h  rk_m")
df = pd.DataFrame({'h': h, 'm': pd.Series(elo_bt)})
df['rk_h'] = df['h'].rank(ascending=False).astype(int)
df['rk_m'] = df['m'].rank(ascending=False).astype(int)
df = df.sort_values('rk_h')
for m, r in df.iterrows():
    diff = int(r['rk_m']) - int(r['rk_h'])
    flag = "  <-- big" if abs(diff) >= 3 else ""
    print(f"{m:52s}     {int(r['h']):>4d}     {r['m']:>6.1f}    {int(r['rk_h']):>2d}    {int(r['rk_m']):>2d}  {diff:+d}{flag}")
