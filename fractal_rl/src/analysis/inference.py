"""Canonical artifact locations for deterministic policy inference."""
from pathlib import Path


def standard_inference_paths(output_dir, checkpoint, seed):
    """Return all artifacts emitted by the standard deterministic analysis.

    The standard set comprises raw latent/inference dumps, a map-and-UMAP view,
    acceleration history, and a self-contained episode diagnostic panel.
    """
    root = Path(output_dir)
    stem = f'{Path(checkpoint).stem}_seed{seed}'
    return {
        'latent_npz': root / 'latents' / f'latent_trajectory_{stem}.npz',
        'latent_csv': root / 'latents' / f'latent_trajectory_{stem}.csv',
        'umap_csv': root / 'plots' / f'latent_umap_{stem}.csv',
        'umap_plot': root / 'plots' / f'latent_umap_{stem}.png',
        'acceleration_plot': root / 'plots' / f'acceleration_history_{stem}.png',
        'dynamics_plot': root / 'plots' / f'dynamics_history_{stem}.png',
        'episode_plot': root / 'plots' / f'episode_diagnostics_{stem}.png',
    }
