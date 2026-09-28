"""Dump a deterministic policy trajectory and visualize its encoder latents with UMAP."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.fractal_env import FractalEnv
from src.physics.dynamics import acceleration_components, velocity_frame
from src.rl.policy import tensor_obs
from src.rl.trainer import load_policy
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample
from src.analysis.inference import standard_inference_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=100001, help='held-out terrain seed')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    policy, cfg = load_policy(args.checkpoint, args.device)
    policy.eval()
    root = Path(args.output_dir or cfg['output']['dump_dir'])
    latent_dir, plot_dir = root / 'latents', root / 'plots'
    latent_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    terrain = generate(cfg['terrain'], seed=args.seed)
    env = FractalEnv(cfg, terrain=terrain, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    rows, latent_rows = [], []
    for step in range(args.steps):
        with torch.no_grad():
            terrain_t, state_t = tensor_obs([obs], args.device)
            latent = policy.encode(terrain_t, state_t)[0].cpu().numpy()
            action, _, _, value = policy.act(terrain_t, state_t, deterministic=True)
        control_acc, potential_acc, damping_acc, potential = acceleration_components(
            env.body, action[0].cpu().numpy(), terrain, cfg['physics'])
        forward, lateral = velocity_frame(env.body, cfg['physics'])
        position = env.body.position.copy()
        speed = float(np.linalg.norm(env.body.velocity))
        obs, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        # The latent, action, acceleration, and physical state are all sampled
        # immediately before the same integration step.
        rows.append({'step': step, 'x': position[0], 'y': position[1],
                     'speed': speed, 'potential': potential,
                     'action_forward': action[0, 0].item(), 'action_right': action[0, 1].item(),
                     # SAC has no state-value head; retain a stable CSV schema.
                     'value': float('nan') if value is None else value.item(),
                     'control_acceleration': np.linalg.norm(control_acc),
                     'potential_acceleration': np.linalg.norm(potential_acc),
                     'damping_acceleration': np.linalg.norm(damping_acc),
                     'net_acceleration': np.linalg.norm(control_acc + potential_acc + damping_acc),
                     'input_parallel_acceleration': np.dot(control_acc, forward),
                     'input_perpendicular_acceleration': np.dot(control_acc, lateral),
                     'potential_parallel_acceleration': np.dot(potential_acc, forward),
                     'potential_perpendicular_acceleration': np.dot(potential_acc, lateral)})
        latent_rows.append(latent)
        if terminated or truncated:
            break
    latents = np.asarray(latent_rows, dtype=np.float32)
    latent_velocity = np.concatenate(([0.0], np.linalg.norm(np.diff(latents, axis=0), axis=1)))
    for row, value in zip(rows, latent_velocity):
        row['latent_velocity'] = float(value)
    metadata_columns = ('step', 'x', 'y', 'speed', 'potential', 'action_forward', 'action_right', 'value',
                        'latent_velocity',
                        'control_acceleration', 'potential_acceleration', 'damping_acceleration', 'net_acceleration',
                        'input_parallel_acceleration', 'input_perpendicular_acceleration',
                        'potential_parallel_acceleration', 'potential_perpendicular_acceleration')
    metadata = np.asarray([[r[k] for k in metadata_columns]
                           for r in rows], dtype=np.float32)
    paths = standard_inference_paths(root, args.checkpoint, args.seed)
    stem = f'{Path(args.checkpoint).stem}_seed{args.seed}'
    np.savez_compressed(paths['latent_npz'],
                        latent=latents, metadata=metadata,
                        metadata_columns=np.asarray(metadata_columns))
    with paths['latent_csv'].open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + [f'latent_{i:03d}' for i in range(latents.shape[1])])
        writer.writeheader()
        for row, latent in zip(rows, latents):
            writer.writerow(row | {f'latent_{i:03d}': value for i, value in enumerate(latent)})
    try:
        import umap
    except ImportError as error:
        raise RuntimeError('UMAP requires `umap-learn`; install project requirements first.') from error
    normalized = (latents - latents.mean(axis=0)) / np.maximum(latents.std(axis=0), 1e-6)
    embedding = umap.UMAP(n_neighbors=min(30, max(2, len(latents) - 1)), min_dist=0.1,
                          metric='euclidean', random_state=cfg['training']['seed']).fit_transform(normalized)
    umap_csv = paths['umap_csv']
    np.savetxt(umap_csv, np.column_stack([metadata, embedding]), delimiter=',',
               header=','.join(metadata_columns + ('umap_1', 'umap_2')), comments='')
    positions = metadata[:, 1:3]
    speed = metadata[:, 3]
    time = metadata[:, 0]
    values = {name: metadata[:, index] for index, name in enumerate(metadata_columns)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    pad = 8.0
    x_min, x_max = positions[:, 0].min() - pad, positions[:, 0].max() + pad
    y_min, y_max = positions[:, 1].min() - pad, positions[:, 1].max() + pad
    grid_x = np.linspace(x_min, x_max, 180)
    grid_y = np.linspace(y_min, y_max, 180)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    potential_grid = sample(terrain.potential, mesh_x, mesh_y, terrain.pixel_size)
    contours = axes[0].contourf(mesh_x, mesh_y, potential_grid, levels=24, cmap='viridis_r')
    axes[0].contour(mesh_x, mesh_y, potential_grid, levels=10, colors='white', linewidths=0.35, alpha=0.55)
    axes[0].plot(positions[:, 0], positions[:, 1], color='#ef4444', linewidth=1.4,
                 label='deterministic policy trajectory')
    axes[0].scatter(positions[0, 0], positions[0, 1], color='white', edgecolor='#111827', s=32, zorder=3, label='start')
    axes[0].set(title='Potential $\\phi=|I|^2$ and trajectory', xlabel='x', ylabel='y', aspect='equal')
    axes[0].legend(loc='best', fontsize=8, frameon=True)
    fig.colorbar(contours, ax=axes[0], label='potential $\\phi$')
    # UMAP coordinate 1 and 2 are deliberately encoded as red and blue on the
    # physical trajectory, making correspondence with the latent view visible.
    umap_x = (embedding[:, 0] - embedding[:, 0].min()) / max(np.ptp(embedding[:, 0]), 1e-6)
    umap_y = (embedding[:, 1] - embedding[:, 1].min()) / max(np.ptp(embedding[:, 1]), 1e-6)
    axes[0].scatter(positions[:, 0], positions[:, 1], c=np.c_[umap_x, np.zeros_like(umap_x), umap_y],
                    s=8, zorder=3, label='UMAP 1 = red, UMAP 2 = blue')
    axes[0].legend(loc='best', fontsize=8, frameon=True)
    latent_plot = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=speed, s=12, cmap='magma')
    axes[1].plot(embedding[:, 0], embedding[:, 1], color='#374151', linewidth=0.4, alpha=0.35)
    axes[1].set(title='Encoder latent trajectory (UMAP)', xlabel='UMAP 1', ylabel='UMAP 2')
    fig.colorbar(latent_plot, ax=axes[1], label='speed')
    figure = paths['umap_plot']
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True, constrained_layout=True)
    axes[0].plot(time, values['action_forward'], label='forward command', color='#0f766e')
    axes[0].plot(time, values['action_right'], label='right command', color='#ea580c')
    axes[0].set(ylabel='normalized action', title='Deterministic steering commands')
    axes[0].legend(ncol=2, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, values['input_parallel_acceleration'], label='input', color='#0f766e')
    axes[1].plot(time, values['potential_parallel_acceleration'], label='potential', color='#7c3aed')
    axes[1].axhline(0, color='#94a3b8', linewidth=0.7)
    axes[1].set(ylabel='signed acceleration', title='Parallel acceleration (along velocity)')
    axes[1].legend(ncol=2, frameon=False)
    axes[1].grid(alpha=0.25)
    axes[2].plot(time, values['input_perpendicular_acceleration'], label='input', color='#0f766e')
    axes[2].plot(time, values['potential_perpendicular_acceleration'], label='potential', color='#7c3aed')
    axes[2].axhline(0, color='#94a3b8', linewidth=0.7)
    axes[2].set(xlabel='environment step', ylabel='signed acceleration',
                title='Perpendicular acceleration (right of velocity)')
    axes[2].legend(ncol=2, frameon=False)
    axes[2].grid(alpha=0.25)
    history = paths['acceleration_plot']
    fig.savefig(history, dpi=180)
    plt.close(fig)

    # Complete dynamics view: physical/latent velocity, controls, all
    # acceleration magnitudes, and velocity-frame field/control components.
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    axes[0].plot(time, speed, color='#0f766e', label='physical speed')
    twin = axes[0].twinx()
    twin.plot(time, values['latent_velocity'], color='#7c3aed', alpha=.85, label='latent velocity')
    axes[0].set(title='Physical and encoder velocity', ylabel='speed')
    twin.set_ylabel('latent velocity')
    axes[0].legend(loc='upper left', frameon=False); twin.legend(loc='upper right', frameon=False)
    axes[1].plot(time, values['action_forward'], label='forward command', color='#0f766e')
    axes[1].plot(time, values['action_right'], label='right command', color='#ea580c')
    axes[1].set(title='Deterministic controls', ylabel='normalized action'); axes[1].legend(ncol=2, frameon=False)
    axes[2].plot(time, values['control_acceleration'], label='control', color='#0f766e')
    axes[2].plot(time, values['potential_acceleration'], label='potential field', color='#7c3aed')
    axes[2].plot(time, values['damping_acceleration'], label='damping', color='#64748b')
    axes[2].plot(time, values['net_acceleration'], label='net', color='#ef4444')
    axes[2].set(title='Acceleration magnitudes', ylabel='acceleration'); axes[2].legend(ncol=4, frameon=False)
    axes[3].plot(time, values['input_parallel_acceleration'], label='input', color='#0f766e')
    axes[3].plot(time, values['potential_parallel_acceleration'], label='potential', color='#7c3aed')
    axes[3].axhline(0, color='#94a3b8', linewidth=.7); axes[3].set(title='Parallel acceleration', ylabel='signed acceleration'); axes[3].legend(ncol=2, frameon=False)
    axes[4].plot(time, values['input_perpendicular_acceleration'], label='input', color='#0f766e')
    axes[4].plot(time, values['potential_perpendicular_acceleration'], label='potential', color='#7c3aed')
    axes[4].axhline(0, color='#94a3b8', linewidth=.7); axes[4].set(title='Perpendicular acceleration', xlabel='environment step', ylabel='signed acceleration'); axes[4].legend(ncol=2, frameon=False)
    for axis in axes: axis.grid(alpha=.25)
    dynamics = paths['dynamics_plot']
    fig.savefig(dynamics, dpi=180)
    plt.close(fig)

    # One complete episode diagnostic per dumped checkpoint.  Keep the smaller
    # legacy views above, but make this self-contained panel the default report
    # figure: map/UMAP correspondence, latent velocity, steering, and both
    # signed acceleration projections all share one rollout time axis.
    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, height_ratios=(1.25, 0.72, 0.72, 0.72, 0.72))
    map_axis = fig.add_subplot(grid[0, 0])
    umap_axis = fig.add_subplot(grid[0, 1])
    latent_axis = fig.add_subplot(grid[1, :])
    steering_axis = fig.add_subplot(grid[2, :], sharex=latent_axis)
    parallel_axis = fig.add_subplot(grid[3, :], sharex=latent_axis)
    perpendicular_axis = fig.add_subplot(grid[4, :], sharex=latent_axis)
    contours = map_axis.contourf(mesh_x, mesh_y, potential_grid, levels=24, cmap='viridis_r')
    map_axis.contour(mesh_x, mesh_y, potential_grid, levels=10, colors='white', linewidths=0.35, alpha=0.55)
    map_axis.plot(positions[:, 0], positions[:, 1], color='#374151', linewidth=0.65, alpha=0.65)
    map_axis.scatter(positions[:, 0], positions[:, 1], c=np.c_[umap_x, np.zeros_like(umap_x), umap_y],
                     s=8, zorder=3, label='UMAP 1 = red, UMAP 2 = blue')
    map_axis.scatter(positions[0, 0], positions[0, 1], color='white', edgecolor='#111827', s=28, zorder=4, label='start')
    map_axis.set(title='Potential map and physical trajectory', xlabel='x', ylabel='y', aspect='equal')
    map_axis.legend(loc='best', fontsize=7, frameon=True)
    fig.colorbar(contours, ax=map_axis, label='potential $\\phi$')
    latent_plot = umap_axis.scatter(embedding[:, 0], embedding[:, 1], c=speed, s=12, cmap='magma')
    umap_axis.plot(embedding[:, 0], embedding[:, 1], color='#374151', linewidth=0.4, alpha=0.35)
    umap_axis.set(title='Encoder trajectory in UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    fig.colorbar(latent_plot, ax=umap_axis, label='physical speed')
    latent_axis.plot(time, values['latent_velocity'], color='#7c3aed')
    latent_axis.set(title='Encoder latent velocity', ylabel=r'$||z_t-z_{t-1}||$')
    steering_axis.plot(time, values['action_forward'], label='forward command', color='#0f766e')
    steering_axis.plot(time, values['action_right'], label='right command', color='#ea580c')
    steering_axis.set(title='Deterministic steering commands', ylabel='normalized action')
    steering_axis.legend(ncol=2, frameon=False, loc='upper right')
    parallel_axis.plot(time, values['input_parallel_acceleration'], label='input', color='#0f766e')
    parallel_axis.plot(time, values['potential_parallel_acceleration'], label='potential', color='#7c3aed')
    parallel_axis.axhline(0, color='#94a3b8', linewidth=0.7)
    parallel_axis.set(title='Parallel acceleration (along velocity)', ylabel='signed acceleration')
    parallel_axis.legend(ncol=2, frameon=False, loc='upper right')
    perpendicular_axis.plot(time, values['input_perpendicular_acceleration'], label='input', color='#0f766e')
    perpendicular_axis.plot(time, values['potential_perpendicular_acceleration'], label='potential', color='#7c3aed')
    perpendicular_axis.axhline(0, color='#94a3b8', linewidth=0.7)
    perpendicular_axis.set(title='Perpendicular acceleration (right of velocity)', xlabel='environment step',
                           ylabel='signed acceleration')
    perpendicular_axis.legend(ncol=2, frameon=False, loc='upper right')
    for axis in (latent_axis, steering_axis, parallel_axis, perpendicular_axis):
        axis.grid(alpha=0.25)
    episode = paths['episode_plot']
    fig.savefig(episode, dpi=180)
    plt.close(fig)
    print(figure)
    print(history)
    print(dynamics)
    print(episode)


if __name__ == '__main__':
    main()
