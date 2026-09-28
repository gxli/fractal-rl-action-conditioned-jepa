"""Show how alpha changes spatial structure, not the standardized marginal."""
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.terrain.pink_noise import generate
from src.utils.config import load_config


def main():
    cfg = load_config('config/sac_jepa_v3.yaml')
    alphas = (2.0, 3.0, 3.5)
    seeds = (100001, 100002, 100003, 100004)
    colors = ('#2563eb', '#7c3aed', '#dc2626')
    bins = np.linspace(0, 10, 151)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.45), constrained_layout=True,
                             gridspec_kw={'width_ratios': (1, 1, 1, 1.25)})
    ax_hist = axes[3]
    for alpha, color in zip(alphas, colors):
        values = []
        terrain_for_map = None
        for seed in seeds:
            terrain_cfg = copy.deepcopy(cfg['terrain'])
            terrain_cfg['spectral_exponent'] = alpha
            terrain = generate(terrain_cfg, seed)
            values.append(terrain.potential.ravel())
            if seed == seeds[0]:
                terrain_for_map = terrain
        values = np.concatenate(values)
        # A common colour scale makes visible how alpha changes structure.
        image = axes[alphas.index(alpha)].imshow(terrain_for_map.potential, origin='lower',
                                                  cmap='viridis_r', vmin=0, vmax=8)
        axes[alphas.index(alpha)].set(title=fr'potential, $\alpha={alpha:g}$', xticks=[], yticks=[])
        centers = .5 * (bins[1:] + bins[:-1])
        pdf, _ = np.histogram(values, bins=bins, density=True)
        keep = (centers > 0) & (pdf > 0)
        ax_hist.plot(centers[keep], pdf[keep], color=color, linewidth=1.8,
                     label=fr'$\alpha={alpha:g}$')
    fig.colorbar(image, ax=axes[:3], shrink=.78, label=r'potential $\phi$')
    ax_hist.set(xscale='log', yscale='log', xlim=(.01, 10), ylim=(1e-3, 10),
                xlabel=r'potential value $\phi=I^2$', ylabel='PDF density',
                title='Whole-terrain value PDF (log--log)')
    ax_hist.grid(alpha=.22, which='both'); ax_hist.legend(frameon=False)
    fig.suptitle('Spatial terrain and marginal potential distribution, same held-out seed', fontsize=11)
    out = Path('dumps/sac_noim_analysis/terrain_value_histograms.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)


if __name__ == '__main__':
    main()
