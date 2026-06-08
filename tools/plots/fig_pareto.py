"""fig_pareto.py - throughput vs per-user latency (TPOT) on H100.

Data from step-11-pareto/bench_pareto.py on the 1.25B GQA model
(48L, 12 query heads, 2 KV heads, head_dim=128), P=128, N=128, FA3,
median of 3, FP16.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from plot_style import setup, save

setup()

B         = [1,      2,     4,     8,     16,    32,    64,    128,    256]
tpot_ms   = [8.80,   8.13,  8.18,  8.14,  8.23,  8.16,  8.22,  8.07,   8.06]
tps_total = [113.7, 245.9, 489.1, 982.9, 1943.2, 3922.1, 7790.0, 15858.2, 31755.3]
prefill   = [8.8,   9.2,   9.4,   10.2,  14.2,  25.6,  49.9,  101.8,  208.1]

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.2))

ax_left.plot(tps_total, tpot_ms, marker='o', linewidth=2, color='#2E86AB')
for b, x, y in zip(B, tps_total, tpot_ms):
    offset = (8, 6) if b not in (1,) else (8, -14)
    ax_left.annotate(f'B={b}', xy=(x, y), xytext=offset,
                     textcoords='offset points', fontsize=9)
ax_left.set_xlabel('Total throughput (tokens / s, log scale)')
ax_left.set_ylabel('Per-user TPOT (ms / token)')
ax_left.set_title('Pareto: throughput vs per-user latency  (1.25B GQA, FA3, H100)')
ax_left.set_xscale('log')
ax_left.set_ylim(6.5, 10.0)
ax_left.axhspan(8.0, 8.3, alpha=0.10, color='#2E86AB')
ax_left.text(tps_total[-1] * 0.9, 8.4,
             'TPOT band: 8.0–8.3 ms across B=2 to B=256  (±1%)',
             fontsize=9, ha='right', color='#2E86AB')

ax_right.plot(B, prefill,  marker='s', linewidth=2, color='#A23B72', label='Prefill (ms)')
ax_right.plot(B, tpot_ms,  marker='o', linewidth=2, color='#2E86AB', label='TPOT (ms / token)')
ref_anchor_idx = B.index(32)
ref_slope = prefill[ref_anchor_idx] / B[ref_anchor_idx]
ax_right.plot(B, [ref_slope * b for b in B], linestyle='--',
              color='#A23B72', alpha=0.4, label='Prefill linear-in-B reference')
ax_right.set_xlabel('Batch size B (log)')
ax_right.set_ylabel('Latency (ms, log)')
ax_right.set_title('Prefill scales linearly with B; Decode TPOT does not')
ax_right.set_xscale('log', base=2)
ax_right.set_yscale('log')
ax_right.set_xticks(B)
ax_right.set_xticklabels([str(b) for b in B])
ax_right.legend(loc='upper left')

save(fig, 'fig_pareto')
