"""Plot the completed clean (no inverse / no margin) JEPA screen.

The figure deliberately keeps optimization terms separate from behavioural
speed: a small auxiliary loss does not itself demonstrate useful transfer.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('dumps')
METHODS = [
    ('variance', 'Variance floor', '#2563eb'),
    ('sigreg', 'SIGReg', '#dc2626'),
    ('spherical_mmd', 'Spherical MMD', '#059669'),
]
STAGES = [
    ('source_a3', r'Source $\alpha=3$', 'source'),
    ('frozen_a2', r'Frozen transfer $\alpha=2$', 'frozen'),
    ('frozen_a35', r'Frozen transfer $\alpha=3.5$', 'frozen'),
]


def read_csv(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def path_for(method, stage):
    token = {'variance': 'variance_010', 'sigreg': 'sigreg_005',
             'spherical_mmd': 'spherical_mmd_010'}[method]
    return ROOT / f'sac_noim_{token}_{stage}' / 'metrics' / 'training.csv'


def plot_line(ax, data, key, label, color, uncertainty=False, linestyle='-'):
    x = data['steps'] / 1000
    y = data[key]
    line, = ax.plot(x, y, marker='o', markersize=3, linewidth=1.8,
                    color=color, linestyle=linestyle, label=label)
    if uncertainty:
        std = data['evaluation_std_speed']
        ax.fill_between(x, y - std, y + std, color=color, alpha=.12)
    return line


def style(ax, title, ylabel):
    ax.set(title=title, xlabel='transitions (thousands)', ylabel=ylabel)
    ax.grid(alpha=.22)
    ax.set_xlim(left=0)


def main():
    data = {(method, stage): read_csv(path_for(method, stage))
            for method, _label, _color in METHODS for stage, _title, _kind in STAGES}

    # Behavioural curves, one panel for each training/transfer condition.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.45), constrained_layout=True)
    for ax, (stage, title, _kind) in zip(axes, STAGES):
        for method, label, color in METHODS:
            plot_line(ax, data[method, stage], 'evaluation_mean_speed', label, color, uncertainty=True)
        style(ax, title, 'held-out mean speed')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Clean JEPA regularizer screen: behavior (inverse and margin coefficients are zero)', fontsize=12)
    fig.savefig('dumps/sac_noim_analysis/clean_noim_behavior.png', dpi=180)
    plt.close(fig)

    # Individual loss terms.  Every row is one exact condition; columns retain
    # the SAC actor/critic objectives and the *weighted* auxiliary terms.
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.7), constrained_layout=True, sharex='col')
    for row, (stage, title, _kind) in enumerate(STAGES):
        for method, label, color in METHODS:
            d = data[method, stage]
            plot_line(axes[row, 0], d, 'actor_loss', label + ' SAC', color)
            plot_line(axes[row, 0], d, 'actor_total_loss', label + ' total', color, linestyle='--')
            # These completed clean runs predate per-critic-JEPA logging.  The
            # recorded critic loss is therefore plotted as the exact logged
            # critic objective; do not invent a separate critic auxiliary term.
            plot_line(axes[row, 1], d, 'critic_loss', label + ' critic', color)
            plot_line(axes[row, 2], d, 'jepa_term', label + ' actor JEPA', color)
            plot_line(axes[row, 2], d, 'variance_term', label + ' logged variance', color, linestyle='--')
        style(axes[row, 0], title if row else 'Actor objective', 'loss')
        style(axes[row, 1], title if row else 'Critic terms', 'loss')
        style(axes[row, 2], title if row else 'Weighted auxiliary terms', 'loss')
        if row == 0:
            for ax in axes[row]:
                ax.legend(frameon=False, fontsize=6.4, ncol=2)
    axes[0, 0].set_title('Actor: SAC (solid) and total (same color dashed)')
    axes[0, 1].set_title('Logged critic objective (TD MSE)')
    axes[0, 2].set_title('Actor: weighted JEPA / logged variance')
    for row in range(3):
        for ax in axes[row]:
            ax.axhline(0, color='#64748b', linewidth=.55, zorder=0)
    fig.suptitle('Per-condition optimization terms — all runs have inverse term = margin term = 0', fontsize=12)
    fig.savefig('dumps/sac_noim_analysis/clean_noim_loss_terms.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
