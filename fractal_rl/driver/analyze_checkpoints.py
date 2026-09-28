"""Compare deterministic inference trajectories from up to ten PPO checkpoints.

Example:
  python -m driver.analyze_checkpoints 'dumps/strong_field/checkpoints/ppo_*.pt' --device cuda
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.fractal_env import FractalEnv
from src.physics.dynamics import acceleration_components
from src.rl.policy import tensor_obs
from src.rl.trainer import load_policy
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample


def checkpoint_step(path: Path) -> int:
    """Read the numerical PPO step from a standard checkpoint filename."""
    return int(path.stem.rsplit('_', 1)[-1])


def choose_checkpoints(pattern: str, limit: int) -> list[Path]:
    paths = sorted(Path().glob(pattern), key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f'No checkpoints match {pattern!r}')
    if len(paths) <= limit:
        return paths
    indices = np.unique(np.linspace(0, len(paths) - 1, limit, dtype=int))
    return [paths[index] for index in indices]


def infer(checkpoint: Path, device: str, seed: int, steps: int) -> tuple[dict, list[dict]]:
    policy, cfg = load_policy(checkpoint, device)
    policy.eval()
    terrain = generate(cfg['terrain'], seed=seed)
    env = FractalEnv(cfg, terrain=terrain, seed=seed)
    observation, _ = env.reset(seed=seed)
    rows = []
    for step in range(steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], device)
            action, _, _, _ = policy.act(terrain_t, state_t, deterministic=True)
        action_np = action[0].cpu().numpy()
        control, field, damping, _ = acceleration_components(env.body, action_np, terrain, cfg['physics'])
        position = env.body.position.copy()
        velocity = env.body.velocity.copy()
        speed = float(np.linalg.norm(velocity))
        potential = float(sample(terrain.potential, *position, terrain.pixel_size))
        observation, _, terminated, truncated, info = env.step(action_np)
        rows.append({
            'step': step,
            # Every quantity in this row describes the pre-step observation
            # acted on by action_np, including the acceleration components.
            'x': float(position[0]), 'y': float(position[1]),
            'vx': float(velocity[0]), 'vy': float(velocity[1]),
            'speed': speed, 'potential': potential,
            'throttle': float(action_np[0]), 'steering': float(action_np[1]),
            'control_acceleration': float(np.linalg.norm(control)),
            'potential_acceleration': float(np.linalg.norm(field)),
            'damping_acceleration': float(np.linalg.norm(damping)),
            'net_acceleration': float(np.linalg.norm(control + field + damping)),
        })
        if terminated or truncated:
            break
    return {'cfg': cfg, 'terrain': terrain}, rows


def write_dump(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_plot(results: list[tuple[Path, list[dict]]], destination: Path) -> None:
    colors = plt.get_cmap('viridis')(np.linspace(0.08, 0.92, len(results)))
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for color, (checkpoint, rows) in zip(colors, results):
        values = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        label = f'{checkpoint_step(checkpoint):,}'
        axes[0].plot(values['step'], values['speed'], color=color, label=label)
        axes[1].plot(values['step'], values['net_acceleration'], color=color, label=label)
        axes[2].plot(values['step'], values['potential'], color=color, label=label)
    axes[0].set(title='Velocity during deterministic inference', ylabel='speed')
    axes[1].set(title='Net acceleration during deterministic inference', ylabel='acceleration magnitude')
    axes[2].set(title='Potential encountered during deterministic inference', xlabel='environment step', ylabel='potential')
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(title='checkpoint step', ncol=2, fontsize=8, frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.text(0.01, 0.005, f'Inference CSV dumps: {destination.parent.parent / "inference_dumps"}',
                fontsize=8, color='#374151')
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def trajectory_plot(results: list[tuple[Path, list[dict]]], terrain, destination: Path) -> None:
    count = len(results)
    columns = min(5, count)
    rows_count = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(3.2 * columns, 3.2 * rows_count), squeeze=False,
                                constrained_layout=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, (checkpoint, rows) in zip(axes.flat, results):
        axis.set_visible(True)
        positions = np.asarray([[row['x'], row['y']] for row in rows])
        pad = 12.0
        x_min, x_max = positions[:, 0].min() - pad, positions[:, 0].max() + pad
        y_min, y_max = positions[:, 1].min() - pad, positions[:, 1].max() + pad
        grid_x = np.linspace(x_min, x_max, 150)
        grid_y = np.linspace(y_min, y_max, 150)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        potential = sample(terrain.potential, mesh_x, mesh_y, terrain.pixel_size)
        axis.contourf(mesh_x, mesh_y, potential, levels=24, cmap='viridis_r')
        axis.plot(positions[:, 0], positions[:, 1], color='#ef4444', linewidth=1.25)
        axis.scatter(*positions[0], s=18, color='white', edgecolor='#111827', zorder=3)
        axis.set(title=f'{checkpoint_step(checkpoint):,} steps', aspect='equal', xticks=[], yticks=[])
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.text(0.01, 0.005, f'Inference CSV dumps: {destination.parent.parent / "inference_dumps"}',
                fontsize=8, color='#374151')
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', help='Glob such as dumps/strong_field/checkpoints/ppo_*.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    if args.limit < 1 or args.steps < 1:
        parser.error('--limit and --steps must be positive')
    checkpoints = choose_checkpoints(args.checkpoints, args.limit)
    results, terrain = [], None
    for checkpoint in checkpoints:
        context, rows = infer(checkpoint, args.device, args.seed, args.steps)
        root = Path(args.output_dir or context['cfg']['output']['dump_dir'])
        write_dump(rows, root / 'inference_dumps' / f'inference_{checkpoint.stem}_seed{args.seed}.csv')
        results.append((checkpoint, rows))
        terrain = context['terrain']
    history_plot(results, root / 'plots' / f'inference_histories_seed{args.seed}.png')
    trajectory_plot(results, terrain, root / 'plots' / f'inference_trajectories_seed{args.seed}.png')
    print(f'Wrote {len(results)} inference dumps to {root / "inference_dumps"}')
    print(root / 'plots' / f'inference_histories_seed{args.seed}.png')
    print(root / 'plots' / f'inference_trajectories_seed{args.seed}.png')


if __name__ == '__main__':
    main()
