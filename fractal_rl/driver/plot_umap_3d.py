"""Joint 3-D UMAP and RGB physical-trajectory diagnostics across policies.

Example:
  python -m driver.plot_umap_3d RUN_A/checkpoints/ppo_*.pt RUN_B/checkpoints/ppo_*.pt
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
from src.rl.policy import tensor_obs
from src.rl.trainer import load_policy
from src.terrain.pink_noise import generate


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        candidate = Path(pattern)
        paths.extend(sorted(candidate.parent.glob(candidate.name)) if any(c in pattern for c in '*?[') else [candidate])
    paths = sorted(set(paths))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError('every checkpoint argument must resolve to a checkpoint file')
    return paths


def infer(checkpoint, device, seed, steps):
    policy, cfg = load_policy(checkpoint, device)
    policy.eval()
    terrain = generate(cfg['terrain'], seed=seed)
    env = FractalEnv(cfg, terrain=terrain, seed=seed)
    observation, _ = env.reset(seed=seed)
    rows, latents = [], []
    for step in range(steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], device)
            latents.append(policy.encode(terrain_t, state_t)[0].cpu().numpy())
            action, _, _, _ = policy.act(terrain_t, state_t, deterministic=True)
        rows.append((step, *env.body.position, float(np.linalg.norm(env.body.velocity))))
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        if terminated or truncated:
            break
    return cfg, np.asarray(rows, dtype=np.float32), np.asarray(latents, dtype=np.float32)


def rgb(embedding):
    lo, hi = embedding.min(axis=0), embedding.max(axis=0)
    return (embedding - lo) / np.maximum(hi - lo, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+', help='checkpoint paths or quoted globs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error('--steps must be at least 3')
    paths = expand_paths(args.checkpoints)
    cases, cfg = [], None
    for path in paths:
        cfg, rows, latents = infer(path, args.device, args.seed, args.steps)
        cases.append((path.stem, rows, latents))
    try:
        import umap
    except ImportError as error:
        raise RuntimeError('UMAP requires umap-learn.') from error
    joined = np.concatenate([values[2] for values in cases])
    normalized = (joined - joined.mean(axis=0)) / np.maximum(joined.std(axis=0), 1e-6)
    embedding = umap.UMAP(n_components=3, n_neighbors=min(30, len(joined) - 1), min_dist=0.1,
                          metric='euclidean', random_state=cfg['training']['seed']).fit_transform(normalized)
    colors = rgb(embedding)
    root = Path(args.output_dir)
    plot_dir = root / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(values[1]) for values in cases])

    figure = plt.figure(figsize=(5.1 * len(cases), 4.8), constrained_layout=True)
    for index, (label, rows, _) in enumerate(cases):
        axis = figure.add_subplot(1, len(cases), index + 1, projection='3d')
        point_colors = colors[offsets[index]:offsets[index + 1]]
        points = embedding[offsets[index]:offsets[index + 1]]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=6, depthshade=False)
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color='#334155', linewidth=.45, alpha=.45)
        axis.set(title=label, xlabel='UMAP 1', ylabel='UMAP 2', zlabel='UMAP 3')
    figure.suptitle('Joint 3-D UMAP: each policy on the same held-out terrain', fontsize=14)
    figure.savefig(plot_dir / 'joint_umap_3d_all_cases.png', dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(5.1 * len(cases), 4.8), constrained_layout=True)
    for index, (label, rows, _) in enumerate(cases):
        axis = figure.add_subplot(1, len(cases), index + 1, projection='3d')
        point_colors = colors[offsets[index]:offsets[index + 1]]
        axis.scatter(rows[:, 1], rows[:, 2], rows[:, 3], c=point_colors, s=7, depthshade=False)
        axis.plot(rows[:, 1], rows[:, 2], rows[:, 3], color='#334155', linewidth=.45, alpha=.4)
        axis.set(title=label, xlabel='world x', ylabel='world y', zlabel='speed')
    figure.suptitle('Physical trajectories in 3-D, colored by joint UMAP RGB', fontsize=14)
    figure.savefig(plot_dir / 'trajectory_umap_rgb_3d_all_cases.png', dpi=180)
    plt.close(figure)

    with (plot_dir / 'joint_umap_3d_all_cases.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case', 'step', 'x', 'y', 'speed', 'umap_1', 'umap_2', 'umap_3', 'red', 'green', 'blue'])
        for index, (label, rows, _) in enumerate(cases):
            for row, point, color in zip(rows, embedding[offsets[index]:offsets[index + 1]],
                                         colors[offsets[index]:offsets[index + 1]]):
                writer.writerow([label, *row, *point, *color])
    print(plot_dir / 'joint_umap_3d_all_cases.png')
    print(plot_dir / 'trajectory_umap_rgb_3d_all_cases.png')


if __name__ == '__main__':
    main()
