"""Four-panel source/transfer comparison for the SIGReg coefficient sweep."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with open(path, newline='') as f: rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in ('steps', 'rollout_mean_speed', 'evaluation_mean_speed', 'evaluation_std_speed')}


def add(ax, path, label, metric, shade=False, screen_steps=20_000):
    d = read(path); x = [v / 1000 for v in d['steps']]; y = d[metric]
    # The historical vanilla controls logged at 2,048 transitions and the
    # SIGReg screen logs at 2,000.  Draw all methods on the same 2k, 20k
    # screening grid; interpolation is visual alignment only, never used for
    # endpoint statistics.
    grid = np.arange(2_000, min(screen_steps, int(d['steps'][-1])) + 1, 2_000)
    if len(grid):
        x0 = np.asarray(d['steps']); y0 = np.asarray(y)
        x = grid / 1000; y = np.interp(grid, x0, y0)
    line, = ax.plot(x, y, marker='o', ms=3, lw=1.8, label=label)
    if shade and metric == 'evaluation_mean_speed':
        s = np.interp(grid, np.asarray(d['steps']), np.asarray(d['evaluation_std_speed']))
        ax.fill_between(x, y-s, y+s, color=line.get_color(), alpha=.12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='dumps'); ap.add_argument('--output', default='output/pdf/sigreg_strength_ablation.png')
    args = ap.parse_args(); root = Path(args.root)
    # Existing matched controls provide the no-SIGReg reference.  The sweep uses
    # 20k source/transfer screens with the same actor-and-critic JEPA settings.
    arms = [('Vanilla SAC', root/'sac_50k_control_v2/metrics/training.csv', root/'sac_frozen_curve50k_alpha2/metrics/training.csv')]
    for token, coeff in [('005', '.005'), ('020', '.02'), ('050', '.05')]:
        arms.append((f'SIGReg $\\lambda={coeff}$', root/f'sac_jepa_sigreg_{token}_source/metrics/training.csv', root/f'sac_jepa_sigreg_{token}_frozen_alpha2/metrics/training.csv'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True, sharex='col')
    panels = [(axes[0,0], 1, 'rollout_mean_speed', 'Source alpha=3: training speed'),
              (axes[0,1], 1, 'evaluation_mean_speed', 'Source alpha=3: held-out speed'),
              (axes[1,0], 2, 'rollout_mean_speed', 'Frozen transfer alpha=2: training speed'),
              (axes[1,1], 2, 'evaluation_mean_speed', 'Frozen transfer alpha=2: held-out speed')]
    for ax, index, metric, title in panels:
        for label, source, transfer in arms:
            path = (source, transfer)[index - 1]
            if path.exists(): add(ax, path, label, metric, shade=metric.startswith('evaluation'))
        ax.set(title=title, xlabel='transitions (thousands)', ylabel='mean speed', xlim=(1.5, 20.5)); ax.grid(alpha=.25)
    axes[0,1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('SIGReg strength sweep: source training and frozen-encoder transfer (common 2k grid)', weight='bold')
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=220); print(out)


if __name__ == '__main__': main()
