"""Plot rollout and held-out evaluation speed for a PPO or SAC training CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='dumps/metrics/training.csv')
    parser.add_argument('--output', default='dumps/plots/learning_curve.png')
    parser.add_argument('--algorithm', default='PPO', help='display label, e.g. PPO or SAC+JEPA')
    parser.add_argument('--jepa-loss-label', default='JEPA distance',
                        help='y-axis label for the auxiliary diagnostic')
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.metrics).open()))
    if not rows:
        raise ValueError(f'no rows in {args.metrics}')
    steps = np.asarray([float(row['steps']) for row in rows])
    rollout = np.asarray([float(row['rollout_mean_speed']) for row in rows])
    evaluation = np.asarray([float(row['evaluation_mean_speed']) for row in rows])
    eval_std = np.asarray([float(row['evaluation_std_speed']) for row in rows])
    valid = np.isfinite(evaluation)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    has_jepa = 'jepa_loss' in rows[0]
    fig, axes = plt.subplots(2 if has_jepa else 1, 1, figsize=(8, 6.4 if has_jepa else 4.5),
                             constrained_layout=True)
    ax = axes[0] if has_jepa else axes
    ax.plot(steps, rollout, color='#9aa5b1', linewidth=0.9, alpha=0.8, label='rollout mean speed')
    ax.plot(steps[valid], evaluation[valid], 'o-', color='#0f766e', markersize=3,
            linewidth=1.7, label='held-out mean speed')
    ax.fill_between(steps[valid], evaluation[valid] - eval_std[valid],
                    evaluation[valid] + eval_std[valid], color='#0f766e', alpha=0.16,
                    label='held-out standard deviation')
    ax.set(xlabel='training transitions', ylabel='mean speed', title=f'{args.algorithm} learning curve')
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    if has_jepa:
        jepa = np.asarray([float(row['jepa_loss']) for row in rows])
        axes[1].plot(steps, jepa, color='#7c3aed', linewidth=1.5, label='JEPA embedding prediction loss')
        axes[1].set(xlabel='training transitions', ylabel=args.jepa_loss_label,
                    title='JEPA auxiliary objective')
        axes[1].grid(alpha=0.22)
        axes[1].legend(frameon=False)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == '__main__':
    main()
