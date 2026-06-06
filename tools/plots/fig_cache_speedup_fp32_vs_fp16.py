"""Plot: KV cache speedup, FP32 vs FP16, gpt2-xl, B=1 P=1 N=500.

Source:
  step-1-kvcache/bench_gpt2xl.log  (FP32)
  step-2-fp16/bench.log            (FP16, single-trial)

Headline finding: FP32 cache delivers 3.72x speedup; FP16 only 1.44x.
Same model, same workload, only precision differs. Sets up section 5's
investigation into where the cache spends its time.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

precisions   = ['FP32', 'FP16']
uncached_tpot = [35.41, 15.78]
cached_tpot   = [9.50,  10.94]
speedups      = [3.72,  1.44]

x = np.arange(len(precisions))
w = 0.35

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

bars_unc = ax_l.bar(x - w/2, uncached_tpot, w, label='uncached', color='#888')
bars_cac = ax_l.bar(x + w/2, cached_tpot,   w, label='cached',   color='#1f77b4')
ax_l.set_xticks(x)
ax_l.set_xticklabels(precisions)
ax_l.set_ylabel('TPOT (ms / token)')
ax_l.set_title('Time per decoded token, gpt2-xl, N=500')
ax_l.legend()
for bar, v in zip(bars_unc, uncached_tpot):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
              ha='center', va='bottom', fontsize=9)
for bar, v in zip(bars_cac, cached_tpot):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
              ha='center', va='bottom', fontsize=9)

colors = ['#2ca02c' if s >= 2 else '#ff7f0e' for s in speedups]
bars_s = ax_r.bar(precisions, speedups, color=colors, width=0.5)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even (1.0x)')
ax_r.set_ylabel('cache speedup (uncached / cached)')
ax_r.set_title('Cache speedup ratio')
ax_r.set_ylim(0, max(speedups) * 1.25)
for bar, s in zip(bars_s, speedups):
    ax_r.text(bar.get_x() + bar.get_width() / 2, s + 0.1, f'{s:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper right')

fig.suptitle('KV cache speedup shrinks at FP16  (gpt2-xl, B=1, P=1, N=500)',
             fontsize=13, y=1.02)
save(fig, 'cache_speedup_fp32_vs_fp16')
