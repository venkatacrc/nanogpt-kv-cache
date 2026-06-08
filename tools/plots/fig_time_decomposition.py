"""Plot 3: per-step time decomposition, gpt2-xl FP16, median of 5.

Source: step-3-decomp/decomp.log

Headline: cached decode is FLAT at ~11 ms regardless of S — that's the
launch floor. Uncached decode grows with S because it actually has to
recompute attention over the full sequence.
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

S            = [128, 256, 512]
prefill      = [10.9, 11.1, 11.5]
cached_dec   = [10.89, 10.99, 10.75]
uncached_dec = [10.0, 10.3, 12.8]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(S, prefill,      'o-', label='prefill (S tokens, with cache update)',
        color='#1f77b4', linewidth=2, markersize=8)
ax.plot(S, cached_dec,   's-', label='cached decode (1 token + cached S)',
        color='#2ca02c', linewidth=2, markersize=8)
ax.plot(S, uncached_dec, '^-', label='uncached decode (S+1 tokens, no cache)',
        color='#d62728', linewidth=2, markersize=8)

ax.axhspan(9.5, 11.5, alpha=0.08, color='gray',
           label='~10 ms launch floor')

ax.set_xlabel('sequence length S')
ax.set_ylabel('time per call (ms)')
ax.set_title('Where the time goes  (gpt2-xl, FP16, median of 5)')
ax.set_xticks(S)
ax.set_ylim(8, 14)
ax.legend(loc='upper left', fontsize=9)

ax.annotate('uncached starts to grow',
            xy=(512, 12.8), xytext=(300, 13.3),
            arrowprops=dict(arrowstyle='->', color='#d62728'),
            fontsize=9, color='#d62728')

save(fig, 'time_decomposition_fp16')
