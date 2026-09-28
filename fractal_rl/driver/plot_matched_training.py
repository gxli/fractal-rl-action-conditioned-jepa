"""Overlay held-out learning curves from two isolated training runs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path):
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError(f'no rows in {path}')
    steps = np.asarray([float(row['steps']) for row in rows])
    mean = np.asarray([float(row['evaluation_mean_speed']) for row in rows])
    std = np.asarray([float(row['evaluation_std_speed']) for row in rows])
    valid = np.isfinite(mean)
    return steps[valid], mean[valid], std[valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True, help='first-run metrics/training.csv')
    parser.add_argument('--auxiliary', required=True, help='second-run metrics/training.csv')
    parser.add_argument('--control-label', required=True, help='legend label for the first run')
    parser.add_argument('--auxiliary-label', required=True, help='legend label for the second run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(8.7, 4.8), constrained_layout=True)
    budgets = []
    for path, label, color in ((args.control, args.control_label, '#0f766e'),
                               (args.auxiliary, args.auxiliary_label, '#7c3aed')):
        steps, mean, std = read_metrics(path)
        budgets.append(steps.max())
        axis.plot(steps, mean, marker='o', markersize=3, linewidth=1.8,
                  color=color, label=label)
        axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14)
    budget = min(budgets)
    matched = np.allclose(budgets, budgets[0])
    title = (f'Matched {budget / 1_000:g}k held-out learning curves' if matched
             else f'Held-out learning curves (common budget {budget / 1_000:g}k)')
    axis.set(title=title, xlabel='training transitions',
             ylabel='held-out mean speed')
    axis.grid(alpha=0.24)
    axis.legend(frameon=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
