"""Build a pairwise UMAP matrix for deterministic inference encoder states.

Each ordered pair of checkpoints is embedded jointly, preserving the question
"where does this checkpoint's inference occupy latent space relative to that
counterpart?"  The full N x N embeddings are saved in an NPZ; an evenly
spaced 5 x 5 subset is rendered for readable inspection.
"""
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


def checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit('_', 1)[-1])


def checkpoints_from_glob(pattern: str, limit: int) -> list[Path]:
    paths = sorted(Path().glob(pattern), key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f'No checkpoints match {pattern!r}')
    if len(paths) <= limit:
        return paths
    indices = np.unique(np.linspace(0, len(paths) - 1, limit, dtype=int))
    return [paths[index] for index in indices]


def probe_observations(checkpoint: Path, device: str, seed: int, steps: int) -> tuple[dict, list[dict]]:
    """Collect one held-out deterministic trajectory to use as a shared probe."""
    policy, cfg = load_policy(checkpoint, device)
    policy.eval()
    terrain = generate(cfg['terrain'], seed=seed)
    env = FractalEnv(cfg, terrain=terrain, seed=seed)
    observation, _ = env.reset(seed=seed)
    observations = []
    for _ in range(steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([observation], device)
            action, _, _, _ = policy.act(terrain_t, state_t, deterministic=True)
        observations.append(observation)
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        if terminated or truncated:
            break
    return cfg, observations


def encode_observations(checkpoint: Path, device: str, observations: list[dict]) -> np.ndarray:
    """Encode the same probe observations with one checkpoint's encoder."""
    policy, _ = load_policy(checkpoint, device)
    policy.eval()
    values = []
    with torch.no_grad():
        for observation in observations:
            terrain_t, state_t = tensor_obs([observation], device)
            values.append(policy.encode(terrain_t, state_t)[0].cpu().numpy())
    return np.asarray(values, dtype=np.float32)


def joint_umap(first: np.ndarray, second: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as error:
        raise RuntimeError('UMAP requires `umap-learn`; install project requirements first.') from error
    joined = np.concatenate((first, second), axis=0)
    joined = (joined - joined.mean(axis=0)) / np.maximum(joined.std(axis=0), 1e-6)
    neighbors = min(30, max(2, len(joined) - 1))
    return umap.UMAP(n_neighbors=neighbors, min_dist=0.1, metric='euclidean',
                     random_state=seed).fit_transform(joined).astype(np.float32)


def plot_subset(matrix: np.ndarray, checkpoints: list[Path], subset: np.ndarray, points: int, destination: Path) -> None:
    count = len(subset)
    figure, axes = plt.subplots(count, count, figsize=(3.0 * count, 3.0 * count), squeeze=False,
                                constrained_layout=True)
    for row, first_index in enumerate(subset):
        for column, second_index in enumerate(subset):
            axis = axes[row, column]
            embedding = matrix[first_index, second_index, :2 * points]
            if first_index == second_index:
                axis.scatter(embedding[:points, 0], embedding[:points, 1], s=3, color='#2563eb', alpha=0.68)
            else:
                axis.scatter(embedding[:points, 0], embedding[:points, 1], s=3, color='#2563eb', alpha=0.55)
                axis.scatter(embedding[points:, 0], embedding[points:, 1], s=3, color='#f97316', alpha=0.55)
            axis.set(xticks=[], yticks=[])
            if row == 0:
                axis.set_title(f'counterpart\n{checkpoint_step(checkpoints[second_index]):,}', fontsize=8)
            if column == 0:
                axis.set_ylabel(f'row\n{checkpoint_step(checkpoints[first_index]):,}', fontsize=8)
    figure.suptitle('Pairwise joint UMAPs on the same held-out inference trajectory', fontsize=14)
    figure.text(0.5, 0.008, 'blue = row encoder, orange = counterpart encoder; diagonal = self comparison',
                ha='center', fontsize=9, color='#374151')
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Centered linear CKA: 1 means equivalent representational geometry."""
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(first.T @ second, ord='fro') ** 2
    scale = np.linalg.norm(first.T @ first, ord='fro') * np.linalg.norm(second.T @ second, ord='fro')
    return float(cross / max(scale, 1e-12))


def cka_plot(matrix: np.ndarray, checkpoints: list[Path], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap='magma')
    labels = [f'{checkpoint_step(path) / 1_000_000:g}M' for path in checkpoints]
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
             xlabel='counterpart encoder checkpoint', ylabel='row encoder checkpoint',
             title='Encoder similarity on shared held-out inference (linear CKA)')
    axis.tick_params(axis='x', rotation=45)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f'{matrix[row, column]:.2f}', ha='center', va='center', fontsize=6,
                      color='white' if matrix[row, column] < 0.55 else '#111827')
    figure.colorbar(image, ax=axis, label='linear CKA (1 = same geometry)')
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', help='Glob such as dumps/strong_field/checkpoints/ppo_*.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--plot-size', type=int, default=5)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    if min(args.steps, args.limit, args.plot_size) < 1:
        parser.error('--steps, --limit, and --plot-size must be positive')
    checkpoints = checkpoints_from_glob(args.checkpoints, args.limit)
    # The latest policy supplies the held-out inference trajectory.  All
    # encoders then see precisely the same observations, avoiding a policy
    # behavior confound in the representation comparison.
    cfg, observations = probe_observations(checkpoints[-1], args.device, args.seed, args.steps)
    latents = np.asarray([encode_observations(checkpoint, args.device, observations)
                          for checkpoint in checkpoints], dtype=np.float32)
    points = len(observations)
    root = Path(args.output_dir or cfg['output']['dump_dir'])
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.empty((len(checkpoints), len(checkpoints), 2 * points, 2), dtype=np.float32)
    for first in range(len(checkpoints)):
        for second in range(len(checkpoints)):
            matrix[first, second] = joint_umap(latents[first], latents[second], cfg['training']['seed'])
            print(f'UMAP pair {first + 1}/{len(checkpoints)} x {second + 1}/{len(checkpoints)}', flush=True)
    matrix_dir = root / 'umap_matrix'
    matrix_dir.mkdir(parents=True, exist_ok=True)
    cka = np.asarray([[linear_cka(latents[first], latents[second]) for second in range(len(checkpoints))]
                      for first in range(len(checkpoints))], dtype=np.float32)
    np.savez_compressed(matrix_dir / f'pairwise_encoder_umap_seed{args.seed}.npz',
                        embeddings=matrix, latents=latents,
                        checkpoint_steps=np.asarray([checkpoint_step(path) for path in checkpoints]),
                        cka=cka, seed=args.seed)
    subset_count = min(args.plot_size, len(checkpoints))
    subset = np.unique(np.linspace(0, len(checkpoints) - 1, subset_count, dtype=int))
    destination = root / 'plots' / f'pairwise_encoder_umap_{len(subset)}x{len(subset)}_seed{args.seed}.png'
    plot_subset(matrix, checkpoints, subset, points, destination)
    cka_destination = root / 'plots' / f'encoder_cka_matrix_seed{args.seed}.png'
    cka_plot(cka, checkpoints, cka_destination)
    print(matrix_dir / f'pairwise_encoder_umap_seed{args.seed}.npz')
    print(destination)
    print(cka_destination)


if __name__ == '__main__':
    main()
