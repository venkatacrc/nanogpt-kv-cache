"""Plot 12: batched decode with [B, S, H, D] cache, FA3 vs SDPA.

Source: step-6b-layout/bench_batch_decode.log

Headline: even with the FA3-friendly cache layout and batching up to B=8,
FA3 still loses decode by 2-6%. The remaining gap is intrinsic kernel
quality at q_len=1, which SDPA's specialized decode path handles better.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

B = [1, 4, 8]

# values are SDPA_time / FA3_time (matches log columns).
# >1 means FA3 wins; <1 means FA3 loses.
ratio_short = [0.98, 0.98, 0.98]   # S_pre=128
ratio_long  = [0.95, 0.95, 0.94]   # S_pre=4096

x = np.arange(len(B))
w = 0.36

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars_s = ax.bar(x - w/2, ratio_short, w, label='S_pre=128',
                color='#1f77b4', edgecolor='black', linewidth=0.5)
bars_l = ax.bar(x + w/2, ratio_long, w, label='S_pre=4096',
                color='#5a9bd4', edgecolor='black', linewidth=0.5)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
           label='break-even')
ax.set_xticks(x)
ax.set_xticklabels([f'B={b}' for b in B])
ax.set_xlabel('batch size')
ax.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax.set_title('Batching does NOT close the residual decode gap  '
             '(custom 1.4B, head_dim=128, [B, S, H, D] cache, N=50)')
ax.set_ylim(0.85, 1.05)
ax.legend(loc='upper right')

for bar, v in zip(bars_s, ratio_short):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f'{v:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, v in zip(bars_l, ratio_long):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f'{v:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.text(0.99, 0.05, 'gap is q_len=1 kernel quality,\nnot launch overhead',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color='#444',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888'))

save(fig, 'batched_decode')
