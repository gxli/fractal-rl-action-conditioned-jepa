"""Plot the centered Gaussian potential and longitudinal power throttle."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.terrain.pink_noise import generate
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay', action='append', default=[])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--power-limit', type=float, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    cfg = load_config(overlays=args.overlay)
    terrain = generate(cfg['terrain'], args.seed)
    power = args.power_limit if args.power_limit is not None else cfg['physics']['power_limit']
    if power is None:
        parser.error('set physics.power_limit or pass --power-limit')
    speeds = np.linspace(0, 8, 400)
    acceleration = np.minimum(float(cfg['physics']['max_forward_acceleration']),
                              float(power) / (float(cfg['physics']['mass']) * np.maximum(
                                  speeds, float(cfg['physics']['power_speed_epsilon']))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    image = axes[0].imshow(terrain.potential, cmap='magma', origin='lower')
    axes[0].set(title=r'Periodic potential $\phi=(I-\bar I)^2$', xlabel='tile x', ylabel='tile y')
    fig.colorbar(image, ax=axes[0], label='potential')
    axes[1].plot(speeds, acceleration, color='#0f766e', linewidth=2,
                 label=r'$|a_\parallel|\leq\min(a_{max},P/(m\max(v,\epsilon)))$')
    axes[1].axvline(float(power) / (float(cfg['physics']['mass']) * float(cfg['physics']['max_forward_acceleration'])),
                    color='#64748b', linestyle='--', linewidth=1, label='throttle onset')
    axes[1].set(xlabel='speed', ylabel='maximum |parallel acceleration|', ylim=(0, None),
                title=f'Power-limited longitudinal control (P={power:g})')
    axes[1].legend(fontsize=8)
    destination = Path(args.output) if args.output else Path(cfg['output']['dump_dir']) / 'plots' / 'power_limited_mode.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    print(destination)


if __name__ == '__main__':
    main()
