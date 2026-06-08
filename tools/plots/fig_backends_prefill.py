"""Plot 8: backend prefill latency, gpt2-xl FP16.

Source: step-5-backends/bench_backends.log

Headline: SDPA always beats manual (1.5-2.2x). FA3 helps over manual at
S=1024 but LOSES to SDPA across all tested prefill lengths on gpt2-xl
(head_dim=64). Setup for the "why does FA3 lose?" investigation.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

S      = [128, 256, 512, 1024]
manual = [9.71, 9.87, 10.36, 23.09]
sdpa   = [6.30, 6.53, 6.91, 10.55]
fa3    = [8.30, 8.53, 8.89, 11.05]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

ax_l.plot(S, manual, 'o-', label='manual',  color='#888',
          linewidth=2, markersize=8)
ax_l.plot(S, sdpa,   's-', label='SDPA',    color='#1f77b4',
          linewidth=2, markersize=8)
ax_l.plot(S, fa3,    '^-', label='FA3',     color='#d62728',
          linewidth=2, markersize=8)
ax_l.set_xlabel('prefill length S')
ax_l.set_ylabel('time (ms)')
ax_l.set_title('Prefill latency vs S')
ax_l.set_xticks(S)
ax_l.set_xticklabels(S)
ax_l.set_xscale('log', base=2)
ax_l.legend()

ratios = [s / f for s, f in zip(sdpa, fa3)]   # SDPA/FA3: <1 → FA3 loses
colors = ['#d62728' if r < 1 else '#2ca02c' for r in ratios]
bars = ax_r.bar([str(s) for s in S], ratios, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('prefill length S')
ax_r.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax_r.set_title('FA3 loses to SDPA at all tested S')
ax_r.set_ylim(0, 1.2)
for bar, r in zip(bars, ratios):
    ax_r.text(bar.get_x() + bar.get_width() / 2, r + 0.03, f'{r:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper right')

fig.suptitle('Attention backends: prefill  (gpt2-xl, FP16, head_dim=64)',
             fontsize=13, y=1.02)
save(fig, 'backends_prefill')
