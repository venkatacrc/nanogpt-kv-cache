"""Plot 7: rescues stack sub-multiplicatively. gpt2-xl FP16 P=1 N=500.

Source: step-4-rescues/stack.log

Headline: batch rescue (3.06x) and compile rescue (1.74x) predict 5.34x
combined under independence, but we measure 4.31x — only 81% efficient.
Both attack the same launch-floor bottleneck, so they don't fully stack.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import setup, save


setup()

labels   = ['batch alone\n(B=8 eager)',
            'compile alone\n(B=1 compile)',
            'predicted\nproduct',
            'measured\nstacked (B=8 compile)']
values   = [3.06, 1.74, 5.34, 4.31]
colors   = ['#1f77b4', '#1f77b4', '#cccccc', '#2ca02c']
hatches  = ['', '', '///', '']

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='black',
              linewidth=0.5)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax.set_ylabel('cache speedup (vs uncached eager)')
ax.set_title('Rescues stack at 81% efficiency  '
             '(gpt2-xl, FP16, P=1, N=500)')
ax.set_ylim(0, max(values) * 1.2)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f'{v:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.annotate('', xy=(3, 4.31), xytext=(2, 5.34),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.text(2.5, 4.85, '−19%\n(shared bottleneck)', color='red',
        fontsize=9, ha='center', fontweight='bold')

save(fig, 'stack')
