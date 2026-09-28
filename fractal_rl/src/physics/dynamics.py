"""Continuous 2D dynamics with control, potential acceleration, and linear damping."""
from dataclasses import dataclass
import numpy as np
from src.terrain.periodic_field import sample, sample_gradient


@dataclass
class Body:
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    mass: float


def velocity_frame(body, cfg):
    """Return unit vectors parallel and right-perpendicular to the current heading.

    At non-zero speed the frame follows velocity.  At rest it falls back to
    the stored heading, matching the control semantics used by the integrator.
    """
    speed = float(np.linalg.norm(body.velocity))
    if speed > float(cfg['heading_epsilon']):
        forward = body.velocity / speed
    else:
        forward = np.array([np.cos(body.heading), np.sin(body.heading)])
    return forward, np.array([forward[1], -forward[0]])


def acceleration_components(body, action, terrain, cfg):
    """Return control, conservative-field, damping acceleration, and local phi."""
    velocity = body.velocity
    forward, right = velocity_frame(body, cfg)
    convention = cfg.get('steering_convention', 'left')
    if convention not in {'left', 'right'}:
        raise ValueError("steering_convention must be 'left' or 'right'")
    lateral = right if convention == 'right' else -right
    throttle, steer = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    forward_limit = float(cfg['max_forward_acceleration'])
    power_limit = cfg.get('power_limit')
    if power_limit is not None:
        power_limit = float(power_limit)
        speed_epsilon = float(cfg.get('power_speed_epsilon', 0.1))
        if power_limit <= 0 or speed_epsilon <= 0:
            raise ValueError('power_limit and power_speed_epsilon must be positive')
        # Both thrust and braking are limited by the magnitude of longitudinal
        # mechanical power m |a_parallel| |v|.  The zero-speed guard retains
        # the configured acceleration cap for launch.
        forward_limit = min(forward_limit, power_limit / (body.mass * max(float(np.linalg.norm(velocity)), speed_epsilon)))
    control = (forward_limit * throttle * forward
               + float(cfg['max_perpendicular_acceleration']) * steer * lateral)
    x, y = body.position
    potential_value = float(sample(terrain.potential, x, y, terrain.pixel_size))
    # Use the analytic derivative of exactly the same bilinear potential that
    # supplies potential_value.  The field is therefore conservative within
    # each interpolation cell, rather than a separately interpolated gradient.
    grad = np.asarray(sample_gradient(terrain.potential, x, y, terrain.pixel_size), dtype=np.float64)
    potential_acceleration = -float(cfg['potential_scale']) * grad / float(terrain.gradient_reference)
    damping_acceleration = -float(cfg['linear_damping']) * velocity
    return control, potential_acceleration, damping_acceleration, potential_value


def acceleration(body, action, terrain, cfg):
    control, potential_acceleration, damping_acceleration, potential_value = acceleration_components(
        body, action, terrain, cfg)
    return control + potential_acceleration + damping_acceleration, potential_value


def step(body, action, terrain, cfg):
    """Semi-implicit Euler; return mean substep speed for distance/time reward."""
    dt = float(cfg['dt']) / int(cfg['substeps'])
    if dt <= 0 or int(cfg['substeps']) < 1 or body.mass <= 0:
        raise ValueError('dt, substeps and mass must be positive')
    integral_speed = 0.0
    potential = 0.0
    for _ in range(int(cfg['substeps'])):
        acc, potential = acceleration(body, action, terrain, cfg)
        old_speed = np.linalg.norm(body.velocity)
        body.velocity = body.velocity + dt * acc
        # Trapezoidal speed quadrature approximates distance traveled.
        integral_speed += 0.5 * (old_speed + np.linalg.norm(body.velocity)) * dt
        body.position = body.position + dt * body.velocity
        if np.linalg.norm(body.velocity) > float(cfg['heading_epsilon']):
            body.heading = float(np.arctan2(body.velocity[1], body.velocity[0]))
    return integral_speed, potential
