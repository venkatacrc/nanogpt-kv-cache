"""Plot 17: FA3 FP16 vs FP8 prefill latency, custom 1.4B model.

Source: step-9-fp8/bench_fp8.log

Headline negative finding: naively casting Q, K, V to torch.float8_e4m3fn
right before the FA3 call is both SLOWER (0.48x at S=512, asymptoting to
~0.93x at S=16384) AND WRONG (argmax_match=11%, output is essentially
random vs FP16).

Why slower: per-call per-tensor amax computation + 3 divisions + 3 dtype
casts = ~9 extra kernel launches per layer × 48 layers per forward. At
small S that overhead dwarfs the FP8 attention saving. As S grows and
attention becomes more compute-bound, the ratio drifts toward 1.0 but
doesn't cross within tested range.

Why wrong: per-tensor scaling loses too much precision on K (heads have
wildly different magnitudes). Production FP8 uses per-block or static
calibrated scales.

Real lesson for the blog: FP8 attention is a system optimization, not a
1-line drop-in.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

S       = [512, 2048, 4096, 8192, 16384]
fp16_ms = [7.98, 15.63, 32.57, 71.95, 176.76]
fp8_ms  = [16.55, 24.42, 45.56, 88.44, 190.15]
ratio   = [a / b for a, b in zip(fp16_ms, fp8_ms)]   # >1 would mean FP8 wins

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_l.plot(S, fp16_ms, 's-', label='FA3 FP16', color='#1f77b4',
          linewidth=2, markersize=8)
ax_l.plot(S, fp8_ms,  '^-', label='FA3 FP8 (naive per-tensor scaling)',
          color='#d62728', linewidth=2, markersize=8)
ax_l.set_xscale('log', base=2)
ax_l.set_yscale('log', base=10)
ax_l.set_xticks(S)
ax_l.set_xticklabels([str(s) for s in S])
ax_l.set_xlabel('prefill length S')
ax_l.set_ylabel('time (ms, log scale)')
ax_l.set_title('FP8 is slower at every tested S')
ax_l.legend(loc='upper left')

colors = ['#d62728' if r < 1 else '#2ca02c' for r in ratio]
bars = ax_r.bar([str(s) for s in S], ratio, color=colors, width=0.55)
ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.6,
             label='break-even')
ax_r.set_xlabel('prefill length S')
ax_r.set_ylabel('FP16 / FP8  (higher = FP8 wins)')
ax_r.set_title('Ratio drifts toward 1.0 but never crosses')
ax_r.set_ylim(0, 1.15)
for bar, r in zip(bars, ratio):
    ax_r.text(bar.get_x() + bar.get_width() / 2, r + 0.025, f'{r:.2f}x',
              ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_r.legend(loc='upper left')

# Add the accuracy warning as a footer annotation
fig.text(0.5, -0.03,
         'And the output is wrong: argmax_match = 11% vs FP16 '
         '(naive per-tensor scaling loses too much K precision)',
         ha='center', va='top', fontsize=10, color='#d62728',
         fontweight='bold')

fig.suptitle('FP8 attention with naive scaling: slower AND wrong  '
             '(custom 1.4B, FA3 only, FP16 weights elsewhere)',
             fontsize=12, y=1.02)
save(fig, 'fp8')
