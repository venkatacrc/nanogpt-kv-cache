"""fig_tp.py - 1-GPU vs TP-2 on a Llama-3 ~6B-class custom model, H100.

Data from step-10-tp/bench_tp.py:
  - 1-GPU baseline: python3 bench_tp.py
  - TP-2:           python3 run_tp2.py   (wraps torchrun --nproc_per_node=2)

Two panels:
  (left)  Prefill latency vs S, 1-GPU vs TP-2 (FA3).
  (right) Decode TPOT vs S_pre, 1-GPU vs TP-2 (FA3).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from plot_style import setup, save

setup()

S_prefill            = [512, 2048, 4096]
prefill_1gpu_fa3_ms  = [13.33, 48.97, 101.59]
prefill_tp2_fa3_ms   = [11.14, 34.15, 69.22]

S_decode             = [128, 1024, 4096]
decode_1gpu_fa3_ms   = [5.94, 6.10, 6.24]
decode_tp2_fa3_ms    = [7.55, 7.67, 7.70]

fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(12, 4.4))

ax_p.plot(S_prefill, prefill_1gpu_fa3_ms, marker='o', linewidth=2,
          color='#2E86AB', label='1-GPU')
ax_p.plot(S_prefill, prefill_tp2_fa3_ms,  marker='s', linewidth=2,
          color='#A23B72', label='TP-2')
for s, t1, t2 in zip(S_prefill, prefill_1gpu_fa3_ms, prefill_tp2_fa3_ms):
    ax_p.annotate(f'{t1/t2:.2f}x', xy=(s, t2), xytext=(0, -16),
                  textcoords='offset points', ha='center',
                  fontsize=9, color='#A23B72')
ax_p.set_xlabel('Prefill length S (tokens)')
ax_p.set_ylabel('Prefill latency (ms)')
ax_p.set_title('Prefill: TP-2 wins, scaling toward ideal 2x as S grows')
ax_p.set_xscale('log', base=2)
ax_p.set_xticks(S_prefill)
ax_p.set_xticklabels([str(s) for s in S_prefill])
ax_p.legend(loc='upper left')

ax_d.plot(S_decode, decode_1gpu_fa3_ms, marker='o', linewidth=2,
          color='#2E86AB', label='1-GPU')
ax_d.plot(S_decode, decode_tp2_fa3_ms,  marker='s', linewidth=2,
          color='#A23B72', label='TP-2')
for s, t1, t2 in zip(S_decode, decode_1gpu_fa3_ms, decode_tp2_fa3_ms):
    ax_d.annotate(f'{t1/t2:.2f}x', xy=(s, t2), xytext=(0, 8),
                  textcoords='offset points', ha='center',
                  fontsize=9, color='#A23B72')
ax_d.set_xlabel('Pre-fill context S_pre (tokens)')
ax_d.set_ylabel('Decode TPOT (ms / token)')
ax_d.set_title('Decode: TP-2 LOSES ~20% — NCCL all-reduce overhead dominates')
ax_d.set_xscale('log', base=2)
ax_d.set_xticks(S_decode)
ax_d.set_xticklabels([str(s) for s in S_decode])
ax_d.set_ylim(0, max(decode_tp2_fa3_ms) * 1.3)
ax_d.legend(loc='lower right')

save(fig, 'fig_tp')
