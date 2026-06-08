"""Plot 15: GQA decode TPOT, SDPA vs FA3, sweep across n_kv_head.

Source: step-8-gqa/bench_gqa.log

Headline (the surprise): cutting n_kv_head from 12 → 4 → 2 (a 6x reduction
in KV cache size) barely moves decode TPOT or the SDPA/FA3 ratio. At
gpt2-xl-class on H100, decode is NOT bandwidth-bound on the K/V cache —
it's bound on per-layer weight loads and projection matmuls.

GQA's real value here is memory capacity, not decode wall-clock.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

# (n_kv_head, [S_pre], SDPA, FA3) -> TPOT (ms)
n_kv_heads = [12, 4, 2]
labels     = ['MHA\n(12 KV)', 'GQA 3:1\n(4 KV)', 'GQA 6:1\n(2 KV)']

# Take S_pre = 4096 (long-context decode, GQA's strongest theoretical case)
sdpa_4096 = [8.06, 8.14, 8.13]
fa3_4096  = [8.17, 8.17, 8.17]

x = np.arange(len(n_kv_heads))
w = 0.36

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

bars_s = ax_l.bar(x - w/2, sdpa_4096, w, label='SDPA',
                  color='#1f77b4', edgecolor='black', linewidth=0.5)
bars_f = ax_l.bar(x + w/2, fa3_4096,  w, label='FA3',
                  color='#d62728', edgecolor='black', linewidth=0.5)
ax_l.set_xticks(x)
ax_l.set_xticklabels(labels)
ax_l.set_ylabel('decode TPOT (ms / token)')
ax_l.set_title('Decode TPOT is flat across GQA ratios  (S_pre=4096)')
ax_l.set_ylim(7.5, 8.5)
ax_l.legend(loc='lower right')
for bar, v in zip(bars_s, sdpa_4096):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.2f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, v in zip(bars_f, fa3_4096):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.2f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

# KV cache size at block_size=16384, head_dim=128, n_layer=48, FP16:
#   bytes = 16384 * n_kv_head * 128 * 48 * 2 (K,V) * 2 bytes
# In GB:
def kv_cache_gb(n_kv):
    return 16384 * n_kv * 128 * 48 * 2 * 2 / 1e9

kv_gb = [kv_cache_gb(n) for n in n_kv_heads]
bars_m = ax_r.bar(x, kv_gb, width=0.55,
                  color=['#888', '#ffa64d', '#2ca02c'],
                  edgecolor='black', linewidth=0.5)
ax_r.set_xticks(x)
ax_r.set_xticklabels(labels)
ax_r.set_ylabel('KV cache memory (GB) at S=16384')
ax_r.set_title('GQA shrinks the KV cache — by 6x at 6:1')
ax_r.set_ylim(0, max(kv_gb) * 1.2)
for bar, v in zip(bars_m, kv_gb):
    ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f'{v:.2f} GB',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.suptitle('GQA: huge memory savings, no decode-latency change  '
             '(custom 1.4B, head_dim=128, FP16, B=1)',
             fontsize=12, y=1.02)
save(fig, 'gqa_decode')
