import numpy as np
from src.physics.dynamics import Body, acceleration, step
from src.terrain.pink_noise import Terrain
from src.terrain.periodic_field import sample, sample_gradient
from src.utils.config import load_config


def uniform(value=0.5):
    constant = np.full((16, 16), value, dtype=np.float32)
    zero = np.zeros_like(constant)
    return Terrain(constant, constant.copy() ** 2, zero.copy(), zero.copy(), 1.0)


def test_linear_damping_dissipates_independently_of_mass():
    cfg = load_config()['physics']
    a = Body(np.zeros(2), np.array([3.0, 4.0]), 0.0, 1.0)
    b = Body(np.zeros(2), np.array([3.0, 4.0]), 0.0, 2.0)
    aa, _ = acceleration(a, [0, 0], uniform(), cfg)
    ab, _ = acceleration(b, [0, 0], uniform(), cfg)
    assert np.dot(a.velocity, aa) < 0
    np.testing.assert_allclose(ab, aa)


def test_perpendicular_control_and_zero_speed():
    cfg = load_config()['physics']
    cfg.update(linear_damping=0.0, potential_scale=0.0)
    b = Body(np.zeros(2), np.array([2.0, 0.0]), 0.0, 1.0)
    acc, _ = acceleration(b, [0, 1], uniform(), cfg)
    np.testing.assert_allclose(acc, [0, -cfg['max_perpendicular_acceleration']])
    stopped = Body(np.zeros(2), np.zeros(2), np.pi / 2, 1.0)
    acc, _ = acceleration(stopped, [1, 0], uniform(), cfg)
    np.testing.assert_allclose(acc, [0, cfg['max_forward_acceleration']], atol=1e-12)


def test_zero_force_constant_velocity_and_distance():
    cfg = load_config()['physics']
    cfg.update(linear_damping=0.0, potential_scale=0.0)
    b = Body(np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 1.0)
    distance, _ = step(b, [0, 0], uniform(), cfg)
    np.testing.assert_allclose(distance, 2.0 * cfg['dt'])
    np.testing.assert_allclose(b.position, [2.0 * cfg['dt'], 0.0])


def test_potential_force_points_down_gradient():
    cfg = load_config()['physics']
    cfg.update(linear_damping=0.0, potential_scale=3.0)
    base = uniform()
    base.potential = np.tile(np.arange(16, dtype=np.float64) * 2.0, (16, 1))
    base.gradient_reference = 2.0
    b = Body(np.zeros(2), np.zeros(2), 0.0, 2.0)
    acc, _ = acceleration(b, [0, 0], base, cfg)
    np.testing.assert_allclose(acc, [-3.0, 0.0])


def test_bilinear_gradient_matches_finite_difference():
    field = np.arange(25, dtype=np.float64).reshape(5, 5)
    x, y, h = 1.3, 2.4, 1e-5
    gradient = sample_gradient(field, x, y)
    finite_difference = np.array([(sample(field, x + h, y) - sample(field, x - h, y)) / (2 * h),
                                  (sample(field, x, y + h) - sample(field, x, y - h)) / (2 * h)])
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-8)


def test_power_limit_throttles_parallel_acceleration_at_speed():
    cfg = load_config()['physics']
    cfg.update(linear_damping=0.0, potential_scale=0.0, power_limit=2.0,
               power_speed_epsilon=0.1)
    body = Body(np.zeros(2), np.array([4.0, 0.0]), 0.0, 1.0)
    acc, _ = acceleration(body, [1, 0], uniform(), cfg)
    # P = m a_parallel v = 1 * 0.5 * 4 = 2.
    np.testing.assert_allclose(acc, [0.5, 0.0])


def test_power_limit_keeps_launch_acceleration_cap():
    cfg = load_config()['physics']
    cfg.update(linear_damping=0.0, potential_scale=0.0, power_limit=2.0,
               power_speed_epsilon=0.1)
    body = Body(np.zeros(2), np.zeros(2), 0.0, 1.0)
    acc, _ = acceleration(body, [1, 0], uniform(), cfg)
    np.testing.assert_allclose(acc, [cfg['max_forward_acceleration'], 0.0])
