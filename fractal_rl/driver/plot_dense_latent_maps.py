"""Dense spatial latent maps with a shared RGB embedding across checkpoints.

The probe is a fixed 256-by-256 world-unit tile centred at the deterministic
held-out start.  Every spatial location is observed with the same initial
proprioceptive state, so RGB variation shows what each encoder maps from the
local potential patch rather than changes in velocity or previous action.
All checkpoints in one invocation share a PCA-to-RGB transform and colour
limits, making changes over training directly comparable.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.fractal_env import FractalEnv
from src.env.observation import observe
from src.rl.policy import tensor_obs
from driver.analyze_sac_actor_critics import load as load_sac
from src.terrain.periodic_field import sample
from src.terrain.pink_noise import generate


def checkpoint_step(path):
    return int(path.stem.rsplit('_', 1)[-1])


def checkpoints_from_glob(pattern, limit):
    candidate = Path(pattern)
    paths = sorted(candidate.parent.glob(candidate.name), key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f'No checkpoints match {pattern!r}')
    if len(paths) <= limit:
        return paths
    indices = np.unique(np.linspace(0, len(paths) - 1, limit, dtype=int))
    return [paths[index] for index in indices]


def dense_observations(terrain, cfg, centre, initial_body, resolution, batch_size):
    """Yield fixed-state local-potential observations over one physical square."""
    width = resolution * terrain.pixel_size
    offset = (np.arange(resolution, dtype=np.float64) + 0.5 - resolution / 2) * terrain.pixel_size
    coordinate_x, coordinate_y = centre[0] + offset, centre[1] + offset
    xx, yy = np.meshgrid(coordinate_x, coordinate_y)
    locations = np.column_stack((xx.ravel(), yy.ravel()))
    reference = observe(initial_body, terrain, cfg['agent'], np.zeros(2, dtype=np.float32))
    fixed_state = reference['state']
    size = int(cfg['agent']['fov_size'])
    pixel = float(cfg['agent']['fov_pixel_size'])
    offsets = (np.arange(size, dtype=np.float64) - (size - 1) / 2) * pixel
    horizontal, vertical = np.meshgrid(offsets, -offsets)
    frame = cfg['agent'].get('observation_frame', 'velocity')
    if frame == 'world':
        forward = np.array([0.0, 1.0])
    else:
        speed = float(np.linalg.norm(initial_body.velocity))
        epsilon = float(cfg['physics']['heading_epsilon'])
        forward = (initial_body.velocity / speed if speed > epsilon else
                   np.array([np.cos(initial_body.heading), np.sin(initial_body.heading)]))
    right = np.array([forward[1], -forward[0]])
    if cfg['agent'].get('steering_convention', 'left') == 'left':
        right = -right
    for start in range(0, len(locations), batch_size):
        position = locations[start:start + batch_size]
        x = position[:, 0, None, None] + horizontal * right[0] + vertical * forward[0]
        y = position[:, 1, None, None] + horizontal * right[1] + vertical * forward[1]
        patches = sample(terrain.potential, x, y, terrain.pixel_size).astype(np.float32)[:, None]
        yield locations[start:start + batch_size], patches, np.repeat(fixed_state[None], len(position), axis=0)


def encode_grid(policy, terrain, cfg, centre, initial_body, resolution, batch_size, device):
    values = []
    with torch.no_grad():
        for _, patches, states in dense_observations(terrain, cfg, centre, initial_body, resolution, batch_size):
            values.append(policy.encode(torch.as_tensor(patches, device=device),
                                        torch.as_tensor(states, device=device)).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def infer_trajectory(actor, encoder, terrain, cfg, seed, steps, device):
    env = FractalEnv(cfg, terrain=terrain, seed=seed)
    observation, _ = env.reset(seed=seed)
    initial_body = type(env.body)(env.body.position.copy(), env.body.velocity.copy(), env.body.heading, env.body.mass)
    positions, latents = [], []
    for _ in range(steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], device)
            latents.append(encoder.encode(terrain_t, state_t)[0].cpu().numpy())
            action, _, _, _ = actor.act(terrain_t, state_t, deterministic=True)
        positions.append(env.body.position.copy())
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        if terminated or truncated:
            break
    return initial_body, np.asarray(positions, dtype=np.float32), np.asarray(latents, dtype=np.float32)


def initial_body_for_probe(cfg, terrain, seed):
    """Recreate the deterministic held-out reset without taking an environment step."""
    env = FractalEnv(cfg, terrain=terrain, seed=seed)
    env.reset(seed=seed)
    return type(env.body)(env.body.position.copy(), env.body.velocity.copy(), env.body.heading, env.body.mass)


def fit_rgb(samples):
    """PCA via SVD; percentile scaling makes one RGB legend valid for every epoch."""
    mean = samples.mean(axis=0)
    _, _, vectors = np.linalg.svd(samples - mean, full_matrices=False)
    components = vectors[:3]
    projected = (samples - mean) @ components.T
    low, high = np.percentile(projected, (1, 99), axis=0)
    return mean.astype(np.float32), components.astype(np.float32), low.astype(np.float32), high.astype(np.float32)


def to_rgb(values, mean, components, low, high):
    projected = (values - mean) @ components.T
    rgb = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    return projected.astype(np.float32), rgb.astype(np.float32)


def fit_umap(samples, seed):
    """One shared 3-D UMAP coordinate system for all dense epoch probes."""
    try:
        from cuml.manifold import UMAP
    except ImportError as error:
        raise RuntimeError('Dense latent UMAP requires cuML in the analysis environment.') from error
    normalized_mean = samples.mean(axis=0)
    normalized_scale = np.maximum(samples.std(axis=0), 1e-6)
    mapper = UMAP(n_components=3, n_neighbors=min(30, len(samples) - 1),
                  min_dist=0.1, metric='euclidean', random_state=seed,
                  output_type='numpy')
    mapper.fit((samples - normalized_mean) / normalized_scale)
    return mapper, normalized_mean.astype(np.float32), normalized_scale.astype(np.float32)


def plot_checkpoint(destination, potential, rgb_grid, positions, visited_rgb, dense_umap, visited_umap, step, width):
    figure = plt.figure(figsize=(14.2, 4.4), constrained_layout=True)
    map_axis = figure.add_subplot(1, 3, 1)
    rgb_axis = figure.add_subplot(1, 3, 2)
    scatter_axis = figure.add_subplot(1, 3, 3, projection='3d')
    extent = (-width / 2, width / 2, -width / 2, width / 2)
    map_axis.imshow(potential, cmap='viridis_r', origin='lower', extent=extent)
    map_axis.plot(positions[:, 0], positions[:, 1], color='white', linewidth=1.1, alpha=.9)
    map_axis.scatter(positions[0, 0], positions[0, 1], s=24, color='#ef4444', edgecolor='black', zorder=3)
    map_axis.set(title='Potential with visited trajectory', xlabel='relative x', ylabel='relative y', aspect='equal')
    rgb_axis.imshow(rgb_grid, origin='lower', extent=extent)
    rgb_axis.plot(positions[:, 0], positions[:, 1], color='white', linewidth=.9, alpha=.85)
    rgb_axis.scatter(positions[:, 0], positions[:, 1], s=4, c=visited_rgb, edgecolor='white', linewidth=.08, zorder=3)
    rgb_axis.set(title='Dense encoder map (shared PCA-to-RGB)', xlabel='relative x', ylabel='relative y', aspect='equal')
    # All 65,536 fixed-state observations form the UMAP background.  The
    # trajectory samples use colour by time and a dark edge, so they remain
    # legible rather than being mistaken for a PCA/RGB plot.
    scatter_axis.scatter(dense_umap[:, 0], dense_umap[:, 1], dense_umap[:, 2],
                         s=.35, color='#94a3b8', alpha=.075, depthshade=False, rasterized=True)
    time = np.arange(len(visited_umap))
    visited_plot = scatter_axis.scatter(visited_umap[:, 0], visited_umap[:, 1], visited_umap[:, 2],
                                        c=time, cmap='plasma', s=11, edgecolor='#111827', linewidth=.15,
                                        depthshade=False, zorder=3)
    scatter_axis.plot(visited_umap[:, 0], visited_umap[:, 1], visited_umap[:, 2],
                      color='#111827', linewidth=.45, alpha=.65, zorder=2)
    figure.colorbar(visited_plot, ax=scatter_axis, shrink=.62, pad=.08, label='visited step')
    scatter_axis.set(title='3-D UMAP: dense background + visited states', xlabel='UMAP 1', ylabel='UMAP 2', zlabel='UMAP 3')
    figure.suptitle(f'Dense 256×256 encoder probe, checkpoint {step:,}', fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_evolution(destination, rows, columns=5):
    columns = min(columns, len(rows))
    height = int(np.ceil(len(rows) / columns))
    figure, axes = plt.subplots(height, columns, figsize=(3.2 * columns, 3.4 * height), squeeze=False, constrained_layout=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, (step, rgb, positions, width) in zip(axes.flat, rows):
        axis.set_visible(True)
        extent = (-width / 2, width / 2, -width / 2, width / 2)
        axis.imshow(rgb, origin='lower', extent=extent)
        axis.plot(positions[:, 0], positions[:, 1], color='white', linewidth=.75)
        axis.set(title=f'{step / 1000:g}k', xticks=[], yticks=[], aspect='equal')
    figure.suptitle('Dense latent-RGB map evolution (one shared PCA color basis)', fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', help='glob such as dumps/run/checkpoints/sac_*.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--encoder', choices=('actor', 'q1', 'q2'), default='actor',
                        help='representation to map; actions always come from the actor')
    args = parser.parse_args()
    if min(args.steps, args.resolution, args.batch_size, args.limit) < 1:
        parser.error('steps, resolution, batch-size, and limit must be positive')
    checkpoints = checkpoints_from_glob(args.checkpoints, args.limit)
    actor, critics, cfg = load_sac(checkpoints[-1], args.device)
    terrain = generate(cfg['terrain'], seed=args.seed)
    initial_body = initial_body_for_probe(cfg, terrain, args.seed)
    centre = initial_body.position.copy()
    width = args.resolution * terrain.pixel_size
    dense = []
    trajectories = []
    for checkpoint in checkpoints:
        actor, critics, checkpoint_cfg = load_sac(checkpoint, args.device)
        if checkpoint_cfg['terrain'] != cfg['terrain'] or checkpoint_cfg['agent'] != cfg['agent']:
            raise ValueError('all checkpoints must use the same terrain and agent configuration')
        encoder = actor if args.encoder == 'actor' else getattr(critics, args.encoder)
        body, positions, latents = infer_trajectory(actor, encoder, terrain, cfg, args.seed, args.steps, args.device)
        dense.append(encode_grid(encoder, terrain, cfg, centre, body, args.resolution, args.batch_size, args.device))
        trajectories.append((positions, latents))
        print(f'encoded checkpoint {checkpoint_step(checkpoint):,}', flush=True)
    rng = np.random.default_rng(cfg['training']['seed'])
    sample_size = min(8192, args.resolution * args.resolution)
    samples = np.concatenate([values[rng.choice(len(values), sample_size, replace=False)] for values in dense])
    mean, components, low, high = fit_rgb(samples)
    mapper, umap_mean, umap_scale = fit_umap(samples, cfg['training']['seed'])
    root = Path(args.output_dir)
    dense_dir, plot_dir = root / 'dense_latents', root / 'plots'
    dense_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    evolution = []
    grid_offset = (np.arange(args.resolution, dtype=np.float64) + .5 - args.resolution / 2) * terrain.pixel_size
    grid_x, grid_y = np.meshgrid(centre[0] + grid_offset, centre[1] + grid_offset)
    potential = sample(terrain.potential, grid_x, grid_y, terrain.pixel_size)
    for checkpoint, grid, (positions, visited) in zip(checkpoints, dense, trajectories):
        projected, rgb = to_rgb(grid, mean, components, low, high)
        visited_projected, visited_rgb = to_rgb(visited, mean, components, low, high)
        dense_umap = mapper.transform((grid - umap_mean) / umap_scale).astype(np.float32)
        visited_umap = mapper.transform((visited - umap_mean) / umap_scale).astype(np.float32)
        relative = positions - centre
        stem = f'{checkpoint.stem}_{args.encoder}_seed{args.seed}'
        np.savez_compressed(dense_dir / f'dense_latent_{stem}.npz', latent=grid, pca=projected,
                            rgb=rgb, visited_position=relative, visited_pca=visited_projected,
                            visited_rgb=visited_rgb, dense_umap=dense_umap, visited_umap=visited_umap,
                            centre=centre, pca_mean=mean,
                            pca_components=components, pca_low=low, pca_high=high)
        plot_checkpoint(plot_dir / f'dense_latent_umap_{stem}.png', potential,
                        rgb.reshape(args.resolution, args.resolution, 3), relative, visited_rgb,
                        dense_umap, visited_umap, checkpoint_step(checkpoint), width)
        evolution.append((checkpoint_step(checkpoint), rgb.reshape(args.resolution, args.resolution, 3), relative, width))
    plot_evolution(plot_dir / f'dense_latent_rgb_evolution_{args.encoder}_{len(checkpoints)}x_seed{args.seed}.png', evolution)
    print(plot_dir / f'dense_latent_rgb_evolution_{args.encoder}_{len(checkpoints)}x_seed{args.seed}.png')


if __name__ == '__main__':
    main()
