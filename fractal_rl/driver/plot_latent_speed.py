"""Relate a checkpoint's encoder trajectory to physical speed during inference."""
import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    try:
        import umap
    except ImportError as error:
        raise RuntimeError('UMAP requires `umap-learn`.') from error
    policy, cfg = load_policy(args.checkpoint, args.device)
    policy.eval()
    terrain = generate(cfg['terrain'], seed=args.seed)
    env = FractalEnv(cfg, terrain=terrain, seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    latents, speeds = [], []
    for _ in range(args.steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], args.device)
            latents.append(policy.encode(terrain_t, state_t)[0].cpu().numpy())
            action, _, _, _ = policy.act(terrain_t, state_t, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        speeds.append(info['speed'])
        if terminated or truncated:
            break
    latents, speeds = np.asarray(latents, dtype=np.float32), np.asarray(speeds, dtype=np.float32)
    normalized = (latents - latents.mean(axis=0)) / np.maximum(latents.std(axis=0), 1e-6)
    embedding = umap.UMAP(n_neighbors=min(30, len(latents) - 1), min_dist=0.1, metric='euclidean',
                          random_state=cfg['training']['seed']).fit_transform(normalized)
    latent_velocity = np.r_[0.0, np.linalg.norm(np.diff(latents, axis=0), axis=1)]
    nearest_one = np.argsort(np.abs(speeds - 1.0))[:max(1, len(speeds) // 40)]
    nearest_two = np.argsort(np.abs(speeds - 2.0))[:max(1, len(speeds) // 40)]
    root = Path(args.output_dir or cfg['output']['dump_dir'])
    dump_dir, plot_dir = root / 'inference_dumps', root / 'plots'
    dump_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dump_dir / f'latent_speed_{Path(args.checkpoint).stem}_seed{args.seed}.npz',
                        latent=latents, umap=embedding, speed=speeds, latent_velocity=latent_velocity)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(embedding[:, 0], embedding[:, 1], color='#64748b', alpha=.45, linewidth=.65)
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=speeds, cmap='viridis', s=10, alpha=.8)
    axes[0].scatter(embedding[nearest_one, 0], embedding[nearest_one, 1], color='#2563eb', s=26,
                    label='speed ≈ 1', edgecolor='white', linewidth=.35)
    axes[0].scatter(embedding[nearest_two, 0], embedding[nearest_two, 1], color='#f97316', s=26,
                    label='speed ≈ 2', edgecolor='white', linewidth=.35)
    axes[0].set(title='Encoder trajectory in UMAP space', xlabel='UMAP 1', ylabel='UMAP 2')
    axes[0].legend(frameon=False)
    figure.colorbar(scatter, ax=axes[0], label='physical speed')
    time = np.arange(len(speeds))
    axes[1].plot(time, speeds, color='#0f766e', label='physical speed')
    twin = axes[1].twinx()
    twin.plot(time, latent_velocity, color='#7c3aed', alpha=.8, label='latent speed')
    axes[1].scatter(nearest_one, speeds[nearest_one], color='#2563eb', s=18, zorder=3)
    axes[1].scatter(nearest_two, speeds[nearest_two], color='#f97316', s=18, zorder=3)
    axes[1].set(title='Physical and latent speed along inference', xlabel='environment step', ylabel='physical speed')
    twin.set_ylabel(r'latent speed ($||z_t-z_{t-1}||$)')
    axes[1].grid(alpha=.25)
    axes[1].legend(loc='upper left', frameon=False)
    twin.legend(loc='upper right', frameon=False)
    destination = plot_dir / f'latent_speed_{Path(args.checkpoint).stem}_seed{args.seed}.png'
    figure.savefig(destination, dpi=180)
    print(destination)


if __name__ == '__main__':
    main()
