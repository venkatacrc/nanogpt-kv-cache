"""Plot 5: cache speedup vs batch size B, gpt2-xl FP16, P=1 N=500.

Source: step-4-rescues/batch.log

Headline: batching amortizes the launch floor across B sequences. Cache
speedup goes from 0.91x (loss) at B=1 to 2.91x at B=8.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

B        = [1, 4, 8]
speedup  = [0.91, 1.70, 2.91]
unc_ms   = [4961, 9040, 15306]
cac_ms   = [5439, 5325, 5252]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

ax_l.plot(B, unc_ms, 'o-', label='uncached', color='#d62728',
          linewidth=2, markersize=9)
ax_l.plot(B, cac_ms, 's-', label='cached',   color='#1f77b4',
          linewidth=2, markersize=9)
ax_l.set_xlabel('batch size B')
ax_l.set_ylabel('total time, N=500 (ms)')
ax_l.set_title('Cached cost barely changes with B')
ax_l.set_xticks(B)
ax_l.legend()
ax_l.annotate('cached almost flat\n→ launch-floor amortized',
              xy=(8, 5252), xytext=(3.2, 8400),
              arrowprops=dict(arrowstyle='->', color='#1f77b4'),
              fontsize=9, color='#1f77b4')

colors = ['#d62728' if s < 1 else '#2ca02c' for s in speedup]
bars = ax_r.bar([str(b) for b in B], speedup, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('batch size B')
ax_r.set_ylabel('cache speedup')
ax_r.set_title('Bigger batch → bigger cache win')
ax_r.set_ylim(0, max(speedup) * 1.2)
for bar, s in zip(bars, speedup):
    ax_r.text(bar.get_x() + bar.get_width() / 2, s + 0.07, f'{s:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper left')

fig.suptitle('Rescue #2: batching  (gpt2-xl, FP16, P=1, N=500, median of 3)',
             fontsize=13, y=1.02)
save(fig, 'batch')
