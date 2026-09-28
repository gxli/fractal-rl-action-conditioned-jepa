"""Overlay SAC and SAC+JEPA transfer curves after a pink-noise alpha shift."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    rows = list(csv.DictReader(Path(path).open()))
    step = np.asarray([float(row['steps']) for row in rows])
    mean = np.asarray([float(row['evaluation_mean_speed']) for row in rows])
    std = np.asarray([float(row['evaluation_std_speed']) for row in rows])
    valid = np.isfinite(mean)
    return step[valid], mean[valid], std[valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sac-alpha-low', required=True)
    parser.add_argument('--jepa-alpha-low', required=True)
    parser.add_argument('--sac-alpha-high', required=True)
    parser.add_argument('--jepa-alpha-high', required=True)
    parser.add_argument('--sac-reference', type=float, required=True)
    parser.add_argument('--jepa-reference', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    panels = [(r'$\alpha=2$ (more high-frequency terrain)', args.sac_alpha_low, args.jepa_alpha_low),
              (r'$\alpha=4$ (more low-frequency terrain)', args.sac_alpha_high, args.jepa_alpha_high)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True, constrained_layout=True)
    for axis, (title, sac, jepa) in zip(axes, panels):
        for label, path, color in [('SAC transfer', sac, '#0f766e'), ('SAC+JEPA transfer', jepa, '#7c3aed')]:
            steps, mean, std = read(path)
            axis.plot(steps, mean, marker='o', markersize=3, linewidth=1.8, color=color, label=label)
            axis.fill_between(steps, mean - std, mean + std, color=color, alpha=.13)
        axis.axhline(args.sac_reference, color='#0f766e', linestyle='--', linewidth=.9, alpha=.7,
                     label='SAC α=3 endpoint')
        axis.axhline(args.jepa_reference, color='#7c3aed', linestyle='--', linewidth=.9, alpha=.7,
                     label='SAC+JEPA α=3 endpoint')
        axis.set(title=title, xlabel='transfer transitions')
        axis.grid(alpha=.23)
    axes[0].set_ylabel('held-out mean speed')
    axes[0].legend(frameon=False, fontsize=8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
