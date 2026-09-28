"""Plot pairwise UMAP-space latent speeds from a saved encoder UMAP matrix."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def smooth(values: np.ndarray, width: int = 25) -> np.ndarray:
    kernel = np.full(width, 1.0 / width)
    return np.convolve(values, kernel, mode='same')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix', help='pairwise_encoder_umap_*.npz')
    parser.add_argument('--plot-size', type=int, default=5)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    data = np.load(args.matrix)
    embeddings, steps = data['embeddings'], data['checkpoint_steps']
    count, _, total_points, _ = embeddings.shape
    points = total_points // 2
    selected = np.unique(np.linspace(0, count - 1, min(args.plot_size, count), dtype=int))
    figure, axes = plt.subplots(len(selected), len(selected), figsize=(3.0 * len(selected), 2.15 * len(selected)),
                                squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    for row, first in enumerate(selected):
        for column, second in enumerate(selected):
            axis = axes[row, column]
            pair = embeddings[first, second]
            first_speed = smooth(np.r_[0.0, np.linalg.norm(np.diff(pair[:points], axis=0), axis=1)])
            second_speed = smooth(np.r_[0.0, np.linalg.norm(np.diff(pair[points:], axis=0), axis=1)])
            axis.plot(first_speed, color='#2563eb', linewidth=.9)
            if first != second:
                axis.plot(second_speed, color='#f97316', linewidth=.9)
            axis.grid(alpha=.2)
            if row == 0:
                axis.set_title(f'counterpart\n{steps[second] / 1_000_000:g}M', fontsize=8)
            if column == 0:
                axis.set_ylabel(f'row\n{steps[first] / 1_000_000:g}M', fontsize=8)
            if row == len(selected) - 1:
                axis.set_xlabel('inference step', fontsize=7)
    figure.suptitle('Pairwise latent-speed matrix in joint UMAP space', fontsize=14)
    figure.text(.5, .008, 'blue = row encoder UMAP speed; orange = counterpart UMAP speed; both use the same inference states',
                ha='center', fontsize=9, color='#374151')
    source = Path(args.matrix)
    destination = Path(args.output) if args.output else source.parent.parent / 'plots' / 'pairwise_latent_speed_5x5.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    print(destination)


if __name__ == '__main__':
    main()
