"""Plot held-out performance and action-conditioning diagnostics from SAC+JEPA metrics."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def column(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.metrics).open()))
    required = {'steps', 'evaluation_mean_speed', 'evaluation_std_speed', 'jepa_loss',
                'jepa_loss_shuffled', 'jepa_real_to_shuffled_ratio', 'jepa_inverse_loss'}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f'missing V3 diagnostic columns: {sorted(missing)}')
    steps = column(rows, 'steps')
    speed, speed_std = column(rows, 'evaluation_mean_speed'), column(rows, 'evaluation_std_speed')
    real, shuffled = column(rows, 'jepa_loss'), column(rows, 'jepa_loss_shuffled')
    ratio, inverse = column(rows, 'jepa_real_to_shuffled_ratio'), column(rows, 'jepa_inverse_loss')
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True, constrained_layout=True)
    axes[0].plot(steps, speed, 'o-', color='#0f766e', label='held-out speed')
    axes[0].fill_between(steps, speed - speed_std, speed + speed_std, color='#0f766e', alpha=.15)
    axes[0].set(title='V3 K=8 + inverse-dynamics SAC+JEPA', ylabel='held-out mean speed')
    axes[0].legend(frameon=False); axes[0].grid(alpha=.24)
    axes[1].plot(steps, real, 'o-', color='#7c3aed', label='real action sequence')
    axes[1].plot(steps, shuffled, 'o-', color='#ea580c', label='block-shuffled sequence')
    axes[1].set(ylabel='cosine distance', title='K-step predictive loss')
    axes[1].legend(frameon=False); axes[1].grid(alpha=.24)
    axes[2].plot(steps, ratio, 'o-', color='#2563eb', label='real / shuffled')
    axes[2].plot(steps, inverse, 'o-', color='#dc2626', label='inverse-action MSE')
    axes[2].axhline(1, color='#64748b', linestyle='--', linewidth=.9, label='no action separation')
    axes[2].axhspan(.3, .7, color='#2563eb', alpha=.08, label='target ratio band')
    axes[2].set(xlabel='training transitions', ylabel='diagnostic value', title='Action-conditioning diagnostics')
    axes[2].legend(ncol=2, frameon=False); axes[2].grid(alpha=.24)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
