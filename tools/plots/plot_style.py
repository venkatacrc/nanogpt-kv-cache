"""Shared style for all blog plots.

Use:
    from plot_style import setup, save
    setup()
    fig, ax = plt.subplots()
    ...
    save(fig, 'fig_name')   # writes docs/plots/fig_name.png at 150 dpi
"""

import os
import matplotlib
matplotlib.use('Agg')          # headless; no display needed
import matplotlib.pyplot as plt


DOCS_PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'docs', 'plots',
)


def setup():
    plt.rcParams.update({
        'figure.figsize':   (7, 4),
        'figure.dpi':       150,
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'legend.fontsize':  10,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'axes.spines.top':  False,
        'axes.spines.right': False,
    })


def save(fig, name):
    os.makedirs(DOCS_PLOTS_DIR, exist_ok=True)
    out = os.path.join(DOCS_PLOTS_DIR, f'{name}.png')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f"wrote {out}")
