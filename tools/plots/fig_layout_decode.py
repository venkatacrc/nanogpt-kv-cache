"""Plot 11: KV cache layout effect on decode TPOT (FA3 vs SDPA).

Sources:
  step-6-shapes/bench_shapes.log     (cache layout [B, H, S, D])
  step-6b-layout/bench_layout.log    (cache layout [B, S, H, D])

Headline: switching the cache layout from [B, H, S, D] to [B, S, H, D]
closes most of FA3's decode gap (0.87-0.89x → 0.96-0.99x) — without
changing the kernel. Layout matters.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

S_pre   = [128, 512, 2048, 4096]
# values below are SDPA_time / FA3_time (matches log columns).
# >1 means FA3 wins; <1 means FA3 loses.
ratio_bhsd = [0.89, 0.90, 0.87, 0.87]   # [B,H,S,D] cache
ratio_bshd = [0.99, 0.98, 0.96, 0.96]   # [B,S,H,D] cache

x = np.arange(len(S_pre))
w = 0.36

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars_a = ax.bar(x - w/2, ratio_bhsd, w, label='[B, H, S, D] cache',
                color='#d62728', edgecolor='black', linewidth=0.5)
bars_b = ax.bar(x + w/2, ratio_bshd, w, label='[B, S, H, D] cache',
                color='#2ca02c', edgecolor='black', linewidth=0.5)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
           label='break-even (SDPA = FA3)')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in S_pre])
ax.set_xlabel('prefill length S')
ax.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax.set_title('Cache layout closes most of FA3\'s decode gap  '
             '(custom 1.4B, head_dim=128, FP16, N=50)')
ax.set_ylim(0.7, 1.15)
ax.legend(loc='lower right')

for bar, v in zip(bars_a, ratio_bhsd):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, v in zip(bars_b, ratio_bshd):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('layout refactor\nlifts FA3',
            xy=(0 + w/2, 0.99), xytext=(0.5, 1.08),
            arrowprops=dict(arrowstyle='->', color='#2ca02c'),
            fontsize=9, color='#2ca02c', fontweight='bold')

save(fig, 'layout_decode')
