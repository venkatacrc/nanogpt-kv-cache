"""Plot 6: torch.compile rescues the FP16 cache. gpt2-xl FP16 P=1 B=1 N=500.

Source: step-4-rescues/compile.log

Headline: torch.compile attacks the launch floor itself. TPOT drops from
10.93 ms (cached eager) to 5.89 ms (compiled default). Cache flips from
loss (0.91x) to win (1.68x).
"""

import matplotlib.pyplot as plt
from plot_style import setup, save


setup()

labels = ['uncached\neager', 'cached\neager',
          'cached\ncompile (default)', 'cached\ncompile (reduce-oh)']
tpot    = [9.92, 10.93, 5.89, 6.02]
speedup = [1.00, 0.91, 1.68, 1.65]

colors = ['#888', '#d62728', '#2ca02c', '#5fa55a']

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.bar(labels, tpot, color=colors, width=0.6)
ax.set_ylabel('TPOT (ms / token)')
ax.set_title('torch.compile attacks the launch floor  '
             '(gpt2-xl, FP16, B=1, P=1, N=500, median of 3)')
ax.set_ylim(0, max(tpot) * 1.3)
ax.axhline(9.92, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(3.4, 10.1, 'eager floor', color='gray', fontsize=9,
        ha='right', va='bottom')

for bar, t, s in zip(bars, tpot, speedup):
    ax.text(bar.get_x() + bar.get_width() / 2, t + 0.25,
            f'{t:.2f} ms', ha='center', va='bottom', fontsize=10,
            fontweight='bold')
    ax.text(bar.get_x() + bar.get_width() / 2, t / 2,
            f'{s:.2f}x', ha='center', va='center', fontsize=12,
            fontweight='bold', color='white')

ax.annotate('cache loses\nat eager',
            xy=(1, 10.93), xytext=(1.2, 13.0),
            arrowprops=dict(arrowstyle='->', color='#d62728'),
            fontsize=9, color='#d62728')
ax.annotate('cache wins\nafter compile',
            xy=(2, 5.89), xytext=(2.3, 3.0),
            arrowprops=dict(arrowstyle='->', color='#2ca02c'),
            fontsize=9, color='#2ca02c')

save(fig, 'compile')
