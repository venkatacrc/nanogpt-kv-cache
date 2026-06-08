"""fig_ttft_vs_batch.py - TTFT vs batch size from Pareto sweep.

Definition used in the blog:
  TTFT = prefill_ms + first_decode_step_ms

We approximate first_decode_step_ms with TPOT from the same run.
Data source: step-11-pareto extended sweep (P=128, N=128, FA3, FP16, H100).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from plot_style import setup, save

setup()

B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
prefill_ms = [8.8, 9.2, 9.4, 10.2, 14.2, 25.6, 49.9, 101.8, 208.1]
tpot_ms = [8.80, 8.13, 8.18, 8.14, 8.23, 8.16, 8.22, 8.07, 8.06]
ttft_ms = [p + d for p, d in zip(prefill_ms, tpot_ms)]

fig, ax = plt.subplots()

ax.plot(B, ttft_ms, marker="o", linewidth=2, color="#A23B72", label="TTFT")
ax.plot(B, prefill_ms, marker="s", linewidth=1.5, color="#2E86AB", alpha=0.8, label="Prefill")

for b, t in zip(B, ttft_ms):
    if b in (1, 32, 128, 256):
        ax.annotate(f"{t:.1f} ms", xy=(b, t), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(B)
ax.set_xticklabels([str(x) for x in B])
ax.set_xlabel("Batch size B (log)")
ax.set_ylabel("Latency (ms, log)")
ax.set_title("TTFT vs batch size (1.25B GQA, FA3, H100)")
ax.legend(loc="upper left")

save(fig, "fig_ttft_vs_batch")
