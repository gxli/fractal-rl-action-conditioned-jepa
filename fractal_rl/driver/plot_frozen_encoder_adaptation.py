"""Plot the matched frozen-encoder SAC transfer comparison."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_curve(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return ([int(row['steps']) for row in rows],
            [float(row['evaluation_mean_speed']) for row in rows],
            [float(row['evaluation_std_speed']) for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/pdf/frozen_encoder_adaptation_curve.png')
    parser.add_argument('--vanilla-alpha2', default='dumps/sac_frozen_alpha2_transfer/metrics/training.csv')
    parser.add_argument('--jepa-alpha2', default='dumps/sac_jepa_actorcritic_frozen_alpha2_transfer/metrics/training.csv')
    parser.add_argument('--vanilla-alpha4', default='dumps/sac_frozen_alpha4_transfer/metrics/training.csv')
    parser.add_argument('--jepa-alpha4', default='dumps/sac_jepa_actorcritic_frozen_alpha4_transfer/metrics/training.csv')
    args = parser.parse_args()
    panels = [('Transfer to alpha = 2', args.vanilla_alpha2, args.jepa_alpha2),
              ('Transfer to alpha = 4', args.vanilla_alpha4, args.jepa_alpha4)]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35), sharey=True, constrained_layout=True)
    for axis, (title, vanilla_path, jepa_path) in zip(axes, panels):
        for label, path, color, marker in [('Vanilla SAC', vanilla_path, '#3572a5', 'o'),
                                           ('Actor + critic JEPA', jepa_path, '#b15b35', 's')]:
            steps, mean, std = read_curve(path)
            x = [step / 1000 for step in steps]
            axis.plot(x, mean, color=color, marker=marker, linewidth=2, label=label)
            axis.fill_between(x, [m - s for m, s in zip(mean, std)],
                              [m + s for m, s in zip(mean, std)], color=color, alpha=.16)
        axis.set_title(title, fontsize=11, weight='bold')
        axis.set_xlabel('Transfer transitions (thousands)')
        axis.set_xticks([10, 20])
        axis.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Held-out mean speed')
    axes[0].set_ylim(0, 3.35)
    axes[1].legend(frameon=False, loc='lower right')
    fig.suptitle('Frozen visual encoders: transfer adaptation', fontsize=13, weight='bold')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    print(output)


if __name__ == '__main__':
    main()
