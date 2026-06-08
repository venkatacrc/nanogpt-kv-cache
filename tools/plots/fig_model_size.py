"""Plot 13: cache speedup vs model size, FP16 B=1 P=1 N=500.

Source: step-7-extras/model_size.log

Headline surprise: speedup is FLAT at ~0.90x across gpt2 (124M) through
gpt2-xl (1.56B). The cache loses by the same margin at every scale.
This overturns the naive "small models hit the launch floor harder"
intuition — both paths scale together with model size, so the cache's
relative position barely moves.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

names    = ['gpt2',  'gpt2-medium', 'gpt2-large', 'gpt2-xl']
params_M = [124.4,    354.8,         774.0,        1557.6]
unc_tpot = [2.55,     4.96,          7.35,         9.93]
cac_tpot = [2.78,     5.51,          8.20,         10.94]
speedup  = [0.92,     0.90,          0.90,         0.91]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_l.plot(params_M, unc_tpot, 'o-', label='uncached',
          color='#d62728', linewidth=2, markersize=9)
ax_l.plot(params_M, cac_tpot, 's-', label='cached',
          color='#1f77b4', linewidth=2, markersize=9)
ax_l.set_xscale('log')
ax_l.set_xlabel('parameters (M)')
ax_l.set_ylabel('TPOT (ms / token)')
ax_l.set_title('Both paths scale linearly with model size')
ax_l.set_xticks(params_M)
ax_l.set_xticklabels([n.replace('gpt2', 'gpt2\n') for n in names], fontsize=9)
ax_l.legend()

colors = ['#d62728' if s < 1 else '#2ca02c' for s in speedup]
bars = ax_r.bar(names, speedup, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xticklabels([n.replace('gpt2', 'gpt2\n') for n in names], fontsize=9)
ax_r.set_ylabel('cache speedup')
ax_r.set_title('Speedup is flat across model size  (~0.90x)')
ax_r.set_ylim(0, 1.15)
for bar, s in zip(bars, speedup):
    ax_r.text(bar.get_x() + bar.get_width() / 2, s + 0.025, f'{s:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper right')

fig.suptitle('Cache speedup is independent of model size  '
             '(FP16, B=1, P=1, N=500, median of 3)', fontsize=13, y=1.02)
save(fig, 'model_size')
