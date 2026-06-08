"""Plot 1: KV cache speedup in FP32, gpt2-xl, B=1 P=1 N=500.

Source: step-1-kvcache/bench_gpt2xl.log

Headline: in FP32 the cache delivers a clean 3.72x total speedup, with
TPOT dropping from 35.4 ms to 9.5 ms per token.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

labels = ['uncached', 'cached']
tpot   = [35.41, 9.50]
total  = [17678.4, 4749.0]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

bars1 = ax_l.bar(labels, tpot, color=['#888', '#1f77b4'], width=0.5)
ax_l.set_ylabel('TPOT (ms / token)')
ax_l.set_title('Per-token decode latency')
for bar, v in zip(bars1, tpot):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.7, f'{v:.1f} ms',
              ha='center', va='bottom', fontsize=11, fontweight='bold')

bars2 = ax_r.bar(labels, [t / 1000 for t in total],
                 color=['#888', '#1f77b4'], width=0.5)
ax_r.set_ylabel('total time (s)')
ax_r.set_title('Total time for 500 generated tokens')
for bar, v in zip(bars2, total):
    ax_r.text(bar.get_x() + bar.get_width() / 2, v / 1000 + 0.3,
              f'{v / 1000:.1f} s', ha='center', va='bottom',
              fontsize=11, fontweight='bold')
ax_r.text(0.98, 0.95, '3.72x speedup',
          transform=ax_r.transAxes, ha='right', va='top',
          fontsize=13, fontweight='bold', color='#2ca02c',
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ca02c'))

fig.suptitle('FP32 KV cache wins cleanly  (gpt2-xl, B=1, P=1, N=500)',
             fontsize=13, y=1.02)
save(fig, 'cache_speedup_fp32')
