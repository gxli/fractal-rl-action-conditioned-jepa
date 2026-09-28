"""Run the complete default visualization pipeline for one training run.

This is the standard post-training flush: curve (including JEPA loss),
deterministic inference, UMAP, episode diagnostics, pairwise UMAP, CKA, dense
spatial latent RGB maps, and a machine-readable manifest. Run from the project
root.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.rl.trainer import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='training dump directory containing checkpoints/ and metrics/')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(args.run_dir)
    checkpoints = sorted([*(root / 'checkpoints').glob('ppo_*.pt'), *(root / 'checkpoints').glob('sac_*.pt')])
    if not checkpoints:
        raise FileNotFoundError(f'no checkpoints in {root / "checkpoints"}')
    _, cfg = load_policy(checkpoints[-1], args.device)
    ac = cfg['analysis']
    latest = checkpoints[-1]
    prefix = checkpoints[-1].stem.split('_', 1)[0]
    jepa_enabled = float(cfg['training'].get('jepa_coef', 0.0)) > 0
    algorithm_label = prefix.upper() + ('+JEPA' if jepa_enabled else '')
    checkpoint_glob = str(root / 'checkpoints' / f'{prefix}_*.pt')
    commands = [
        ['-m', 'driver.plot_training', '--metrics', str(root / 'metrics' / 'training.csv'),
         '--output', str(root / 'plots' / 'learning_curve.png'), '--algorithm', algorithm_label],
        ['-m', 'driver.analyze_latent', str(latest), '--device', args.device,
         '--seed', str(ac['inference_seed']), '--steps', str(ac['inference_steps']), '--output-dir', str(root)],
        ['-m', 'driver.analyze_checkpoints', checkpoint_glob, '--device', args.device,
         '--seed', str(ac['inference_seed']), '--steps', str(ac['inference_steps']),
         '--limit', str(ac['pairwise_limit']), '--output-dir', str(root)],
        ['-m', 'driver.compare_checkpoint_umaps', checkpoint_glob, '--device', args.device,
         '--seed', str(ac['inference_seed']), '--steps', str(ac['pairwise_steps']),
         '--limit', str(ac['pairwise_limit']), '--plot-size', str(ac['pairwise_plot_size']), '--output-dir', str(root)],
        ['-m', 'driver.plot_dense_latent_maps', checkpoint_glob, '--device', args.device,
         '--seed', str(ac['inference_seed']), '--steps', str(ac['inference_steps']),
         '--resolution', str(ac.get('dense_map_resolution', 256)),
         '--batch-size', str(ac.get('dense_map_batch_size', 256)),
         '--limit', str(ac['pairwise_limit']), '--output-dir', str(root)],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], check=True)
    manifest = {
        'checkpoint_count': len(checkpoints), 'latest_checkpoint': str(latest),
        'analysis_config': ac,
        'artifacts': [str(path.relative_to(root)) for path in sorted((root / 'plots').glob('*.png'))],
    }
    (root / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(root / 'analysis_manifest.json')


if __name__ == '__main__':
    main()
