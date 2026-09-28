"""Assemble the finalized K=128 dense-map comparison for the report."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path('dumps')
OUT = ROOT / 'ar4k128_analysis'
ARMS = (
    ('ar4k128_sac_source_a3', 'SAC separate: actor encoder', 'actor'),
    ('ar4k128_sac_source_a3', 'SAC separate: Q1 encoder', 'q1'),
    ('ar4k128_sac_shared_source_a3', 'SAC shared critic encoder', 'actor'),
    ('ar4k128_jepa_shared_source_a3', 'SAC+JEPA K=128 shared encoder', 'actor'),
    ('ar4k128_jepa_sigreg_shared_source_a3', 'SAC+JEPA K=128 + SIGReg shared encoder', 'actor'),
)


def source(run, pattern):
    path = ROOT / run / 'plots' / pattern
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_rgb():
    figure, axes = plt.subplots(len(ARMS), 1, figsize=(12.8, 13.8), constrained_layout=True)
    for axis, (run, label, encoder) in zip(axes, ARMS):
        image = mpimg.imread(source(run, f'dense_latent_rgb_evolution_{encoder}_5x_seed100001.png'))
        axis.imshow(image)
        axis.set(title=label, xticks=[], yticks=[])
    figure.suptitle('K=128 dense latent-RGB evolution on one fixed held-out alpha=3 terrain', fontsize=15)
    figure.savefig(OUT / 'k128_dense_rgb_all_encoders.png', dpi=180, facecolor='white')
    plt.close(figure)


def build_umap():
    figure, axes = plt.subplots(len(ARMS), 1, figsize=(12.8, 19.2), constrained_layout=True)
    for axis, (run, label, encoder) in zip(axes, ARMS):
        image = mpimg.imread(source(run, f'dense_latent_umap_sac_000020000_{encoder}_seed100001.png'))
        axis.imshow(image)
        axis.set(title=label, xticks=[], yticks=[])
    figure.suptitle('Final K=128 dense landscape UMAPs: 20k-decision checkpoint', fontsize=15)
    figure.savefig(OUT / 'k128_dense_umap_all_encoders_20k.png', dpi=180, facecolor='white')
    plt.close(figure)


def build_umap_evolution():
    """Show early, middle, and final held-out UMAP trajectories per encoder."""
    stages = (2000, 10000, 20000)
    for run, label, encoder in ARMS:
        figure, axes = plt.subplots(
            len(stages), 1, figsize=(12.8, 12.2), constrained_layout=True
        )
        for axis, step in zip(axes, stages):
            image = mpimg.imread(
                source(run, f'dense_latent_umap_sac_{step:09d}_{encoder}_seed100001.png')
            )
            axis.imshow(image)
            axis.set(title=f'{label} - {step // 1000}k decisions', xticks=[], yticks=[])
        figure.suptitle(
            'Held-out UMAP trajectory evolution: 2k, 10k, and 20k checkpoints',
            fontsize=15,
        )
        stem = run.removeprefix('ar4k128_') + '_' + encoder
        destination = OUT / f'k128_umap_evolution_{stem}.png'
        figure.savefig(
            destination,
            dpi=180,
            facecolor='white',
            transparent=False,
        )
        # The source diagnostics include alpha; flatten the report artifact to
        # white so TeX/PDF renderers never display transparent regions as black.
        with Image.open(destination) as image:
            background = Image.new('RGB', image.size, 'white')
            background.paste(image, mask=image.getchannel('A'))
            background.save(destination)
        plt.close(figure)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    build_rgb()
    build_umap()
    build_umap_evolution()
