"""Plot 14: cache speedup vs generated tokens N, gpt2-xl FP16 B=1 P=1.

Source: step-7-extras/long_generation.log

Headline: cached TPOT is flat at ~10.87 ms across all N. Uncached TPOT
grows from 9.69 ms at N=100 to 14.32 ms at N=1000 (average seq_len
grows → more attention work per step). Cache crosses break-even
around N=700 and reaches 1.31x at N=1000 (the gpt2 block_size limit).
This is the "fourth rescue" — just generate more tokens.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

N        = [100, 300, 500, 800, 1000]
unc_tpot = [9.69, 9.68, 9.90, 11.88, 14.32]
cac_tpot = [10.90, 10.86, 10.86, 10.87, 10.89]
speedup  = [0.89, 0.89, 0.91, 1.09, 1.31]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_l.plot(N, unc_tpot, 'o-', label='uncached (grows with avg seq_len)',
          color='#d62728', linewidth=2, markersize=9)
ax_l.plot(N, cac_tpot, 's-', label='cached (flat ~10.87 ms)',
          color='#1f77b4', linewidth=2, markersize=9)
ax_l.set_xlabel('generated tokens N')
ax_l.set_ylabel('TPOT (ms / token)')
ax_l.set_title('Uncached grows, cached stays flat')
ax_l.set_xticks(N)
ax_l.legend(loc='upper left')
ax_l.annotate('crossover',
              xy=(700, 10.87), xytext=(450, 12.5),
              arrowprops=dict(arrowstyle='->', color='#666'),
              fontsize=10, color='#444', fontweight='bold')

colors = ['#d62728' if s < 1 else '#2ca02c' for s in speedup]
bars = ax_r.bar([str(n) for n in N], speedup, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('generated tokens N')
ax_r.set_ylabel('cache speedup')
ax_r.set_title('Cache crosses break-even at N \u2248 700')
ax_r.set_ylim(0, max(speedup) * 1.2)
for bar, s in zip(bars, speedup):
    ax_r.text(bar.get_x() + bar.get_width() / 2, s + 0.03, f'{s:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper left')

fig.suptitle('Rescue #4: long generation  '
             '(gpt2-xl, FP16, B=1, P=1, median of 3)',
             fontsize=13, y=1.02)
save(fig, 'long_generation')
