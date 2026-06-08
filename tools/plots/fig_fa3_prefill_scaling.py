"""Plot 10: FA3 vs SDPA prefill scaling with head_dim=128, block_size=16K.

Source: step-6-shapes/bench_shapes.log

Headline: with FA3's sweet spot (head_dim=128) and enough sequence length,
FA3 finally beats SDPA: from 1.11x at S=2K to 1.32x at S=16K. But the
crossover only happens around S>=2048.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

S    = [128, 512, 2048, 4096, 8192, 16384]
sdpa = [6.23, 7.02, 19.15, 39.28, 93.92, 241.70]
fa3  = [7.85, 8.44, 17.29, 35.31, 75.84, 182.68]

ratio = [s / f for s, f in zip(sdpa, fa3)]      # SDPA/FA3, >1 means FA3 wins

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_l.plot(S, sdpa, 's-', label='SDPA', color='#1f77b4',
          linewidth=2, markersize=8)
ax_l.plot(S, fa3,  '^-', label='FA3',  color='#d62728',
          linewidth=2, markersize=8)
ax_l.set_xscale('log', base=2)
ax_l.set_yscale('log', base=10)
ax_l.set_xlabel('prefill length S')
ax_l.set_ylabel('time (ms, log scale)')
ax_l.set_title('Prefill latency (head_dim=128, 1.4B params)')
ax_l.set_xticks(S)
ax_l.set_xticklabels([str(s) for s in S], rotation=30)
ax_l.legend()

colors = ['#d62728' if r < 1 else '#2ca02c' for r in ratio]
bars = ax_r.bar([str(s) for s in S], ratio, color=colors, width=0.6)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('prefill length S')
ax_r.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax_r.set_title('Crossover near S=2048; FA3 grows from there')
ax_r.set_xticklabels([str(s) for s in S], rotation=30)
ax_r.set_ylim(0, max(ratio) * 1.2)
for bar, r in zip(bars, ratio):
    ax_r.text(bar.get_x() + bar.get_width() / 2, r + 0.03, f'{r:.2f}x',
              ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_r.legend(loc='upper left')

fig.suptitle('FA3 needs long sequences AND head_dim=128 to win  '
             '(custom 1.4B model, FP16)', fontsize=13, y=1.02)
save(fig, 'fa3_prefill_scaling')
