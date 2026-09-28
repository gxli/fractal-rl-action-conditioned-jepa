"""Fresh fair five-arm SAC / JEPA / latent-regularizer learning curves."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('dumps')
ARMS = [('sac', 'SAC', '#2563eb'), ('sac_jepa', 'SAC+JEPA', '#dc2626'),
        ('sac_jepa_reg', 'SAC+JEPA+variance', '#059669'),
        ('sac_jepa_sigreg', 'SAC+JEPA+SIGReg', '#7c3aed'),
        ('sac_jepa_spherical_mmd', 'SAC+JEPA+spherical MMD', '#d97706')]
STAGES = [('source_a3', r'Source $\alpha=3$'), ('frozen_a2', r'Frozen $\alpha=2$'),
          ('frozen_a35', r'Frozen $\alpha=3.5$')]

def load(path):
    with path.open() as f: rows = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}

def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.35), sharex='col')
    for col, (stage, title) in enumerate(STAGES):
        for arm, label, color in ARMS:
            candidates = [ROOT / f'minstart_v2_{arm}_{stage}' / 'metrics/training.csv']
            if arm in ('sac_jepa_sigreg', 'sac_jepa_spherical_mmd') and stage != 'source_a3':
                candidates.insert(0, ROOT / f'minstart_v2c_{arm}_{stage}' / 'metrics/training.csv')
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            if not path.exists():
                continue
            d = load(path)
            x = d['steps'] / 1000
            for row, key in enumerate(('rollout_mean_speed', 'evaluation_mean_speed')):
                axes[row, col].plot(x, d[key], marker='o', ms=3, lw=1.8, color=color, label=label)
                if row == 1:
                    axes[row, col].fill_between(x, d[key]-d['evaluation_std_speed'], d[key]+d['evaluation_std_speed'], color=color, alpha=.11)
        axes[0,col].set_title(title)
    for col in range(3):
        axes[0,col].set(ylabel='training mean speed'); axes[1,col].set(xlabel='transitions (thousands)', ylabel='held-out mean speed')
        for ax in axes[:,col]: ax.grid(alpha=.22); ax.set_xlim(0,20)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .01), fontsize=8.5)
    fig.suptitle('Fresh global-minimum starts: fair five-arm comparison', fontsize=13, y=.98)
    fig.subplots_adjust(left=.07, right=.985, top=.89, bottom=.18, wspace=.22, hspace=.27)
    out = ROOT / 'minstart_v2_analysis'; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'three_arm_curves.png', dpi=180)

if __name__ == '__main__': main()
