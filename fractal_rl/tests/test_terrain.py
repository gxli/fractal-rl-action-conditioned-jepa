import numpy as np
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample
from src.utils.config import load_config


def test_periodic_and_deterministic():
    cfg = load_config()['terrain']
    cfg['resolution'] = 32
    a, b = generate(cfg, 13), generate(cfg, 13)
    assert a.potential.shape == (32, 32)
    np.testing.assert_array_equal(a.intensity, b.intensity)
    np.testing.assert_array_equal(a.potential, b.potential)
    for field in (a.intensity, a.potential, a.gradient_x, a.gradient_y):
        np.testing.assert_allclose(sample(field, -0.35, 4.7), sample(field, 31.65, 4.7), atol=1e-6)
        np.testing.assert_allclose(sample(field, 4.2, -1.4), sample(field, 4.2, 30.6), atol=1e-6)


def test_gradients_and_mean_centered_squared_potential():
    cfg = load_config()['terrain']
    cfg['resolution'] = 64
    t = generate(cfg, 99)
    np.testing.assert_allclose(t.intensity.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(t.intensity.std(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(t.potential, t.intensity ** 2, rtol=1e-6)
    assert np.isfinite(t.gradient_x).all()
    assert np.isfinite(t.gradient_y).all()
    assert t.gradient_reference > 0
