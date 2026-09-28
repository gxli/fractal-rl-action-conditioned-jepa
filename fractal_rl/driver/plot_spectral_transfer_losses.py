"""Plot every logged SAC loss/temperature curve for a set of comparison runs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_run(value):
    label, separator, path = value.rpartition('=')
    if not separator:
        raise argparse.ArgumentTypeError('--run must be LABEL=METRICS_CSV')
    return label, Path(path)


def read(path):
    rows = list(csv.DictReader(path.open()))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=parse_run, action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    colors = ['#0f766e', '#7c3aed', '#0891b2', '#a855f7', '#0369a1', '#c026d3']
    data = [(label, read(path), color) for (label, path), color in zip(args.run, colors)]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = [('actor_loss', 'SAC actor objective'), ('critic_loss', 'twin-critic loss'),
              ('alpha', 'entropy temperature'), ('jepa_loss', 'JEPA auxiliary distance')]
    for axis, (field, title) in zip(axes.flat, fields):
        for label, values, color in data:
            if field in values:
                axis.plot(values['steps'], values[field], color=color, linewidth=1.35, label=label)
        axis.set(title=title, xlabel='training transitions', ylabel=field)
        axis.grid(alpha=.25)
        axis.legend(fontsize=7, frameon=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
