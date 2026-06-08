"""Plot 4: cache speedup vs prompt length P, gpt2-xl FP16, N=500, B=1.

Source: step-4-rescues/long_prompt.log

Headline: longer prompts make the uncached path do more re-work per
decode step, so cache advantage grows. Crossover near P=128.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

P        = [1, 128, 512]
speedup  = [0.91, 0.98, 1.67]
unc_ms   = [4986, 5344, 9165]
cac_ms   = [5461, 5461, 5492]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

ax_l.plot(P, unc_ms, 'o-', label='uncached', color='#d62728',
          linewidth=2, markersize=9)
ax_l.plot(P, cac_ms, 's-', label='cached',   color='#1f77b4',
          linewidth=2, markersize=9)
ax_l.set_xlabel('prompt length P')
ax_l.set_ylabel('total time, N=500 (ms)')
ax_l.set_title('Wall time vs prompt length')
ax_l.set_xticks(P)
ax_l.set_xscale('log')
ax_l.set_xticks(P)
ax_l.set_xticklabels(P)
ax_l.legend()

colors = ['#d62728' if s < 1 else ('#ff7f0e' if s < 1.05 else '#2ca02c')
          for s in speedup]
bars = ax_r.bar([str(p) for p in P], speedup, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('prompt length P')
ax_r.set_ylabel('cache speedup')
ax_r.set_title('Cache wins once P >> 1')
ax_r.set_ylim(0, max(speedup) * 1.2)
for bar, s in zip(bars, speedup):
    ax_r.text(bar.get_x() + bar.get_width() / 2, s + 0.04, f'{s:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper left')

fig.suptitle('Rescue #1: long prompts  (gpt2-xl, FP16, B=1, N=500, median of 3)',
             fontsize=13, y=1.02)
save(fig, 'long_prompt')
