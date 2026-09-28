"""Evaluate deterministic policies at shared, controlled start locations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.fractal_env import FractalEnv
from src.rl.policy import tensor_obs
from src.rl.trainer import load_policy
from src.terrain.periodic_field import sample
from src.terrain.pink_noise import generate


def parse_run(value):
    label, separator, checkpoint = value.rpartition('=')
    if not separator or not label or not checkpoint:
        raise argparse.ArgumentTypeError('--run must be LABEL=CHECKPOINT')
    return label, Path(checkpoint)


def rollout(policy, cfg, terrain, terrain_seed, start):
    """Hold heading/speed fixed per terrain; vary only the supplied location."""
    env = FractalEnv(cfg, terrain=terrain, seed=terrain_seed)
    observation, _ = env.reset(seed=terrain_seed, start_position=start)
    positions = [env.body.position.copy()]
    done = False
    while not done:
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], next(policy.parameters()).device)
            action, _, _, _ = policy.act(terrain_t, state_t, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        positions.append(env.body.position.copy())
        done = terminated or truncated
    return float(info['mean_speed']), np.asarray(positions)


def trajectory_panel(samples, destination):
    columns = 4
    figure, axes = plt.subplots(len(samples), columns, figsize=(3.2 * columns, 3.0 * len(samples)),
                                squeeze=False, constrained_layout=True)
    for row, (label, terrain, paths, speeds) in enumerate(samples):
        width = terrain.size * terrain.pixel_size
        points = np.linspace(0, width, 180, endpoint=False)
        xx, yy = np.meshgrid(points, points)
        potential = sample(terrain.potential, xx, yy, terrain.pixel_size)
        for column, axis in enumerate(axes[row]):
            path = paths[column]
            image = axis.contourf(xx, yy, potential, levels=24, cmap='viridis_r')
            axis.plot(path[:, 0] % width, path[:, 1] % width, color='#ef4444', linewidth=.85)
            axis.scatter(*path[0] % width, color='white', edgecolor='#111827', s=20, zorder=3)
            axis.set(xticks=[], yticks=[], aspect='equal',
                     title=f'{label}: start {column + 1}\nmean speed {speeds[column]:.3f}')
        figure.colorbar(image, ax=axes[row].tolist(), shrink=.82, label=r'potential $\phi$')
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=parse_run, action='append', required=True)
    parser.add_argument('--starts-per-terrain', type=int, default=16)
    parser.add_argument('--location-seed', type=int, default=20260922)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.starts_per_terrain < 4:
        parser.error('--starts-per-terrain must be at least 4 for trajectory panels')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records, panels = [], []
    for label, checkpoint in args.run:
        policy, cfg = load_policy(checkpoint, args.device)
        policy.eval()
        first_panel = None
        for terrain_seed in cfg['environment']['evaluation_terrain_seeds']:
            terrain_seed = int(terrain_seed)
            terrain = generate(cfg['terrain'], seed=terrain_seed)
            width = terrain.size * terrain.pixel_size
            rng = np.random.default_rng(args.location_seed + terrain_seed)
            starts = rng.uniform(0, width, size=(args.starts_per_terrain, 2))
            paths, speeds = [], []
            for start_index, start in enumerate(starts):
                mean_speed, path = rollout(policy, cfg, terrain, terrain_seed, start)
                records.append({'label': label, 'checkpoint': str(checkpoint), 'terrain_seed': terrain_seed,
                                'start_index': start_index, 'start_x': float(start[0]), 'start_y': float(start[1]),
                                'mean_speed': mean_speed})
                if first_panel is None:
                    paths.append(path)
                    speeds.append(mean_speed)
            if first_panel is None:
                first_panel = (label, terrain, paths[:4], speeds[:4])
        panels.append(first_panel)
    with (root / 'random_start_scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for label, _ in args.run:
        values = np.asarray([row['mean_speed'] for row in records if row['label'] == label])
        summary[label] = {'episodes': len(values), 'mean_speed': float(values.mean()),
                          'std_speed': float(values.std()), 'median_speed': float(np.median(values)),
                          'min_speed': float(values.min()), 'max_speed': float(values.max())}
    (root / 'random_start_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    trajectory_panel(panels, root / 'random_start_trajectory_panels.png')
    labels = list(summary)
    values = [[row['mean_speed'] for row in records if row['label'] == label] for label in labels]
    figure, axis = plt.subplots(figsize=(max(8, 1.55 * len(labels)), 5.0), constrained_layout=True)
    box = axis.boxplot(values, labels=labels, showmeans=True, patch_artist=True)
    colors = ['#0f766e', '#7c3aed'] * (len(labels) // 2 + 1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(.35)
    axis.set(title='Deterministic speed across shared start locations', ylabel='episode mean speed')
    axis.grid(axis='y', alpha=.25)
    axis.tick_params(axis='x', rotation=20)
    figure.savefig(root / 'random_start_speed_distribution.png', dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
