"""Plot 16: GQA prefill, SDPA vs FA3, sweep across S and n_kv_head.

Source: step-8-gqa/bench_gqa.log

Headline: FA3's prefill advantage (~20-30% over SDPA) is preserved across
all GQA ratios. Smaller K/V projections do shrink absolute times modestly
(GQA 6:1 prefill is ~5% faster than MHA at S=8192), but the SDPA-vs-FA3
ratio stays roughly constant.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

# S values and per-(n_kv_head) timings from bench_gqa.log
S = [512, 2048, 4096, 8192]

# (n_kv_head, SDPA[S], FA3[S])
data = {
    12: {'sdpa': [7.05, 18.98, 39.23, 91.98], 'fa3': [7.11, 15.58, 32.32, 71.39]},
    4:  {'sdpa': [7.08, 18.07, 36.65, 88.65], 'fa3': [7.22, 14.67, 29.76, 66.83]},
    2:  {'sdpa': [7.06, 17.92, 36.07, 86.93], 'fa3': [7.26, 14.52, 29.24, 66.44]},
}
labels = {12: 'MHA (12)', 4: 'GQA 3:1 (4)', 2: 'GQA 6:1 (2)'}

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

colors_sdpa = {12: '#1f77b4', 4: '#4a90d9', 2: '#7eb6e8'}
colors_fa3  = {12: '#d62728', 4: '#e85c5c', 2: '#f08d8d'}

for n_kv in [12, 4, 2]:
    ax_l.plot(S, data[n_kv]['sdpa'], 's-',
              label=f"SDPA {labels[n_kv]}", color=colors_sdpa[n_kv],
              linewidth=1.8, markersize=7)
    ax_l.plot(S, data[n_kv]['fa3'],  '^-',
              label=f"FA3 {labels[n_kv]}",  color=colors_fa3[n_kv],
              linewidth=1.8, markersize=7)
ax_l.set_xscale('log', base=2)
ax_l.set_yscale('log', base=10)
ax_l.set_xticks(S)
ax_l.set_xticklabels(S)
ax_l.set_xlabel('prefill length S')
ax_l.set_ylabel('time (ms, log scale)')
ax_l.set_title('Prefill latency (FA3 wins at S \u2265 2048)')
ax_l.legend(fontsize=8, loc='upper left', ncol=2)

# Right panel: SDPA / FA3 ratio at each S, grouped by n_kv_head
ratios = {n_kv: [s / f for s, f in zip(data[n_kv]['sdpa'], data[n_kv]['fa3'])]
          for n_kv in [12, 4, 2]}
x = np.arange(len(S))
w = 0.27
bars_a = ax_r.bar(x - w, ratios[12], w, label='MHA',
                  color='#1f77b4', edgecolor='black', linewidth=0.5)
bars_b = ax_r.bar(x,     ratios[4],  w, label='GQA 3:1',
                  color='#ffa64d', edgecolor='black', linewidth=0.5)
bars_c = ax_r.bar(x + w, ratios[2],  w, label='GQA 6:1',
                  color='#2ca02c', edgecolor='black', linewidth=0.5)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xticks(x)
ax_r.set_xticklabels([str(s) for s in S])
ax_r.set_xlabel('prefill length S')
ax_r.set_ylabel('SDPA / FA3  (higher = FA3 wins)')
ax_r.set_title('FA3 prefill advantage is preserved across GQA')
ax_r.set_ylim(0.85, 1.45)
ax_r.legend(loc='upper left', fontsize=9, ncol=2)
for bars in (bars_a, bars_b, bars_c):
    for bar in bars:
        h = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                  ha='center', va='bottom', fontsize=8)

fig.suptitle('GQA preserves the FA3 prefill advantage  '
             '(custom 1.4B, head_dim=128, FP16)',
             fontsize=12, y=1.02)
save(fig, 'gqa_prefill')
