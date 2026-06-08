"""Plot 9: backend cached-decode latency, gpt2-xl FP16, N=50.

Source: step-5-backends/bench_backends.log

Headline: same pattern as prefill — SDPA wins, FA3 loses by ~13-15% on
gpt2-xl. Decode is q_len=1 and dominated by overhead; FA3's wgmma
advantage doesn't show, but the cache-layout transpose tax does.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

S_pre  = [128, 256, 512]
manual = [10.60, 10.64, 10.65]
sdpa   = [8.02,  8.23,  8.23]
fa3    = [9.46,  9.49,  9.42]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

ax_l.plot(S_pre, manual, 'o-', label='manual', color='#888',
          linewidth=2, markersize=8)
ax_l.plot(S_pre, sdpa,   's-', label='SDPA',   color='#1f77b4',
          linewidth=2, markersize=8)
ax_l.plot(S_pre, fa3,    '^-', label='FA3',    color='#d62728',
          linewidth=2, markersize=8)
ax_l.set_xlabel('prefill length S')
ax_l.set_ylabel('TPOT (ms / token)')
ax_l.set_title('Cached-decode TPOT vs prefill length')
ax_l.set_xticks(S_pre)
ax_l.set_ylim(7, 12)
ax_l.legend()

ratios = [s / f for s, f in zip(sdpa, fa3)]   # SDPA/FA3: <1 → FA3 loses
colors = ['#d62728' if r < 1 else '#2ca02c' for r in ratios]
bars = ax_r.bar([str(s) for s in S_pre], ratios, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('prefill length S')
ax_r.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax_r.set_title('FA3 loses to SDPA in decode too')
ax_r.set_ylim(0, 1.2)
for bar, r in zip(bars, ratios):
    ax_r.text(bar.get_x() + bar.get_width() / 2, r + 0.03, f'{r:.2f}x',
              ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.legend(loc='upper right')

fig.suptitle('Attention backends: decode  (gpt2-xl, FP16, head_dim=64, N=50)',
             fontsize=13, y=1.02)
save(fig, 'backends_decode')
