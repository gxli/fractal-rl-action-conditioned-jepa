"""GPU PCA and 3-D UMAP landscape dashboards for every saved checkpoint.

Consumes dense latent dumps made by ``plot_dense_latent_maps`` and rebuilds a
joint cuML PCA and UMAP over *all* 256x256 map observations from one run.
Each HTML file has potential plus path, PCA-RGB landscape, 3-D PCA trajectory,
and a rotatable 3-D UMAP.  The UMAP background is every dense observation;
only visited states are highlighted, with no misleading UMAP trajectory chords.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rl.trainer import load_policy
from src.terrain.periodic_field import sample
from src.terrain.pink_noise import generate
from driver.plot_dense_latent_maps import infer_trajectory


def checkpoint_step(path):
    return int(path.stem.rsplit('_', 1)[-1])


def as_numpy(value):
    """Materialise NumPy, CuPy, and cuDF outputs without hard-coding a type."""
    if hasattr(value, 'to_numpy'):
        return value.to_numpy()
    try:
        import cupy as cp
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


def gpu_umap(values, seed):
    try:
        from cuml.manifold import UMAP
    except ImportError as error:
        raise RuntimeError('This dashboard requires cuML; run it in the GPU environment.') from error
    mapper = UMAP(n_components=3, n_neighbors=30, min_dist=.1, metric='euclidean',
                  random_state=seed)
    embedded = as_numpy(mapper.fit_transform(values)).astype(np.float32)
    return mapper, embedded


def gpu_pca(values):
    try:
        from cuml.decomposition import PCA
    except ImportError as error:
        raise RuntimeError('This dashboard requires cuML; run it in the GPU environment.') from error
    mapper = PCA(n_components=3)
    embedded = as_numpy(mapper.fit_transform(values)).astype(np.float32)
    return mapper, embedded


def rgb(embedding, low, high):
    return np.clip((embedding - low) / np.maximum(high - low, 1e-6), 0, 1)


def potential_grid(terrain, centre, resolution):
    offsets = (np.arange(resolution, dtype=np.float64) + .5 - resolution / 2) * terrain.pixel_size
    x, y = np.meshgrid(centre[0] + offsets, centre[1] + offsets)
    return sample(terrain.potential, x, y, terrain.pixel_size), resolution * terrain.pixel_size


def dashboard(destination, potential, width, pca_rgb_map, dense_pca, visited_pca,
              dense_umap, visited_umap, visited_position, step):
    extent = (-width / 2, width / 2)
    figure = make_subplots(rows=1, cols=4,
                           specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'scene'}, {'type': 'scene'}]],
                           column_widths=[.24, .24, .26, .26],
                           subplot_titles=('Potential and visited path',
                                           'PCA-to-RGB landscape',
                                           '3-D PCA: dense background + trajectory',
                                           '3-D UMAP: dense background + visited states'))
    figure.add_trace(go.Heatmap(z=potential, colorscale='Viridis', reversescale=True,
                                 showscale=False, x=np.linspace(*extent, potential.shape[1]),
                                 y=np.linspace(*extent, potential.shape[0]), hoverinfo='skip'), row=1, col=1)
    figure.add_trace(go.Scatter(x=visited_position[:, 0], y=visited_position[:, 1], mode='lines',
                                line=dict(color='white', width=2), hoverinfo='skip', showlegend=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=[visited_position[0, 0]], y=[visited_position[0, 1]], mode='markers',
                                marker=dict(size=8, color='#ef4444', line=dict(color='#111827', width=1)),
                                name='episode start'), row=1, col=1)
    figure.add_trace(go.Image(z=np.uint8(np.clip(pca_rgb_map, 0, 1) * 255), hoverinfo='skip'), row=1, col=2)
    figure.add_trace(go.Scatter(x=visited_position[:, 0] + width / 2, y=width / 2 - visited_position[:, 1],
                                mode='lines', line=dict(color='white', width=2), hoverinfo='skip', showlegend=False), row=1, col=2)
    # PCA is linear, so joining temporally consecutive visited points is
    # meaningful. The dense background gives its global geometry.
    figure.add_trace(go.Scatter3d(x=dense_pca[:, 0], y=dense_pca[:, 1], z=dense_pca[:, 2], mode='markers',
                                  marker=dict(size=1.6, color='#64748b'), opacity=.38,
                                  hoverinfo='skip', name='all map states (PCA)'), row=1, col=3)
    time = np.arange(len(visited_pca))
    figure.add_trace(go.Scatter3d(x=visited_pca[:, 0], y=visited_pca[:, 1], z=visited_pca[:, 2], mode='lines',
                                  line=dict(color='#111827', width=3), hoverinfo='skip', showlegend=False), row=1, col=3)
    figure.add_trace(go.Scatter3d(x=visited_pca[:, 0], y=visited_pca[:, 1], z=visited_pca[:, 2], mode='markers',
                                  marker=dict(size=4.5, color=time, colorscale='Plasma', showscale=False),
                                  hovertemplate='visited step %{marker.color}<extra>PCA trajectory</extra>',
                                  name='visited trajectory (PCA)'), row=1, col=3)
    # All 65,536 grid observations are rendered as a subdued cloud.  No path
    # line is drawn in UMAP: it creates false long-range chords when the
    # manifold folds.  Visited states alone are colour-coded and hoverable.
    figure.add_trace(go.Scatter3d(x=dense_umap[:, 0], y=dense_umap[:, 1], z=dense_umap[:, 2], mode='markers',
                                  # Scatter3d does not reliably honour alpha in
                                  # an rgba marker colour.  Trace opacity makes
                                  # the full 65,536-point cloud visibly gray.
                                  marker=dict(size=1.7, color='#64748b'), opacity=.42,
                                  hoverinfo='skip', name='all 65,536 map states (UMAP)'), row=1, col=4)
    time = np.arange(len(visited_umap))
    figure.add_trace(go.Scatter3d(x=visited_umap[:, 0], y=visited_umap[:, 1], z=visited_umap[:, 2], mode='markers',
                                  marker=dict(size=5.2, color=time, colorscale='Plasma', showscale=True,
                                              colorbar=dict(title='visited step', len=.72)),
                                  customdata=np.column_stack((time, visited_position)),
                                  hovertemplate='step %{customdata[0]}<br>x=%{customdata[1]:.2f}<br>'
                                                'y=%{customdata[2]:.2f}<extra>visited</extra>',
                                  name='visited states'), row=1, col=4)
    figure.update_xaxes(range=extent, scaleanchor='y', scaleratio=1, title='relative x', row=1, col=1)
    figure.update_yaxes(range=extent, title='relative y', row=1, col=1)
    figure.update_xaxes(visible=False, row=1, col=2)
    figure.update_yaxes(visible=False, row=1, col=2)
    figure.update_layout(title=f'Dense 256×256 latent landscape — checkpoint {step:,}', height=700, width=2200,
                         template='plotly_white', margin=dict(l=20, r=20, t=70, b=20),
                         legend=dict(orientation='h', y=-.04),
                         scene=dict(xaxis=dict(title='PC 1', showgrid=False, zeroline=False),
                                    yaxis=dict(title='PC 2', showgrid=False, zeroline=False),
                                    zaxis=dict(title='PC 3', showgrid=False, zeroline=False),
                                    bgcolor='white', aspectmode='data'),
                         scene2=dict(xaxis=dict(title='UMAP 1', showgrid=False, zeroline=False),
                                     yaxis=dict(title='UMAP 2', showgrid=False, zeroline=False),
                                     zaxis=dict(title='UMAP 3', showgrid=False, zeroline=False),
                                     bgcolor='white', aspectmode='data'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination, include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='dump directory with checkpoints/ and dense_latents/')
    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(args.run_dir)
    checkpoints = sorted([*(root / 'checkpoints').glob('sac_*.pt'), *(root / 'checkpoints').glob('ppo_*.pt')],
                         key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints in {root / "checkpoints"}')
    if len(checkpoints) > args.limit:
        indices = np.unique(np.linspace(0, len(checkpoints) - 1, args.limit, dtype=int))
        checkpoints = [checkpoints[index] for index in indices]
    policy, cfg = load_policy(checkpoints[-1], args.device)
    del policy
    dense_dir = root / 'dense_latents'
    cases = []
    for checkpoint in checkpoints:
        path = dense_dir / f'dense_latent_{checkpoint.stem}_seed{args.seed}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing dense latent dump {path}; run plot_dense_latent_maps first.')
        data = np.load(path)
        cases.append((checkpoint, checkpoint_step(checkpoint), data['latent'].astype(np.float32),
                      data['visited_position'].astype(np.float32), data['centre'].astype(np.float64)))
    all_latent = np.concatenate([case[2] for case in cases])
    pca_mapper, all_pca = gpu_pca(all_latent)
    umap_mapper, all_umap = gpu_umap(all_latent, cfg['training']['seed'])
    pca_low, pca_high = np.percentile(all_pca, (1, 99), axis=0)
    terrain = generate(cfg['terrain'], seed=args.seed)
    dashboard_dir = root / 'dashboard'
    offset = 0
    entries = []
    for checkpoint, step, latent, visited_position, centre in cases:
        count = len(latent)
        dense_pca, dense_umap = all_pca[offset:offset + count], all_umap[offset:offset + count]
        offset += count
        # Recompute the actual deterministic visited states rather than using
        # an incompatible projection saved by a previous UMAP fit.
        policy, _ = load_policy(checkpoint, args.device)
        _, current_position, visited_latent = infer_trajectory(policy, terrain, cfg, args.seed, 2000, args.device)
        visited_position = current_position - centre
        visited_pca = as_numpy(pca_mapper.transform(visited_latent)).astype(np.float32)
        visited_umap = as_numpy(umap_mapper.transform(visited_latent)).astype(np.float32)
        pca_rgb_map = rgb(dense_pca, pca_low, pca_high).reshape(256, 256, 3)
        potential, width = potential_grid(terrain, centre, 256)
        filename = f'dense_landscape_umap_{step:09d}_seed{args.seed}.html'
        dashboard(dashboard_dir / filename, potential, width, pca_rgb_map, dense_pca, visited_pca,
                  dense_umap, visited_umap, visited_position, step)
        entries.append((step, filename))
        print(dashboard_dir / filename, flush=True)
    links = '\n'.join(f'<li><a href="{name}">{step:,} transitions</a></li>' for step, name in entries)
    (dashboard_dir / 'index.html').write_text('<!doctype html><title>Dense latent landscapes</title>'
        '<h1>Dense latent landscapes</h1><p>Both 3-D panels are rotatable. Gray = all map states; '
        'plasma = actually visited trajectory states. PCA connects the visited trajectory; UMAP does not.</p><ul>' + links + '</ul>')
    (dashboard_dir / 'manifest.json').write_text(json.dumps({'seed': args.seed, 'checkpoints': entries,
        'backend': 'cuml.PCA + cuml.UMAP', 'dense_background_points_per_checkpoint': 256 ** 2}, indent=2) + '\n')
    print(dashboard_dir / 'index.html')


if __name__ == '__main__':
    main()
