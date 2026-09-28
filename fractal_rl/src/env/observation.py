"""Local potential and directional proprioception in a configurable frame."""
import numpy as np
from src.terrain.periodic_field import sample


def observe(body, terrain, cfg, previous_action):
    size = int(cfg['fov_size'])
    if size < 2:
        raise ValueError('fov_size must be >= 2')
    pixel = float(cfg['fov_pixel_size'])
    if pixel <= 0:
        raise ValueError('fov_pixel_size must be positive')
    speed = float(np.linalg.norm(body.velocity))
    frame = cfg.get('observation_frame', 'velocity')
    if frame == 'world':
        # Keep the map and velocity in one inertial frame.  Unlike the legacy
        # velocity-aligned view, this retains both velocity components instead
        # of reducing them to [speed, 0], and avoids rotating the whole patch
        # after each small lateral acceleration.
        forward = np.array([0.0, 1.0])
    elif frame == 'velocity':
        # Compatibility mode for existing checkpoints/configurations.
        speed_epsilon = float(cfg.get('heading_epsilon', 1e-6))
        if speed > speed_epsilon:
            forward = body.velocity / speed
        else:
            forward = np.array([np.cos(body.heading), np.sin(body.heading)])
    else:
        raise ValueError("observation_frame must be 'world' or 'velocity'")
    convention = cfg.get('steering_convention', 'left')
    if convention not in {'left', 'right'}:
        raise ValueError("steering_convention must be 'left' or 'right'")
    # Preserve old left-positive checkpoints, while new runs use a consistent
    # positive-right image/state/action convention.
    right = np.array([forward[1], -forward[0]])
    if convention == 'left':
        right = -right
    offsets = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * pixel
    horizontal, vertical = np.meshgrid(offsets, -offsets)
    x = body.position[0] + horizontal * right[0] + vertical * forward[0]
    y = body.position[1] + horizontal * right[1] + vertical * forward[1]
    source = terrain.cdd if terrain.cdd is not None else terrain.potential
    patch = sample(source, x, y, terrain.pixel_size).astype(np.float32)
    if patch.ndim == 2:
        patch = patch[None, :, :]
    local_v = np.array([np.dot(body.velocity, forward), np.dot(body.velocity, right)])
    velocity_scale = float(cfg['velocity_scale'])
    state = np.array([local_v[0] / velocity_scale, local_v[1] / velocity_scale,
                      speed / velocity_scale, body.mass / float(cfg['mass_scale']),
                      *previous_action], dtype=np.float32)
    return {'terrain': patch, 'state': state}
