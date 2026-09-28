"""Periodic bilinear interpolation in physical world units, including negative positions."""
import numpy as np


def sample(field, x, y, pixel_size=1.0):
    # Spatial axes are always the trailing axes, allowing a cached CDD cube
    # with leading scale channels (S,H,W) to use the same periodic sampler.
    n_y, n_x = field.shape[-2:]
    xx = np.asarray(x, dtype=np.float64) / pixel_size
    yy = np.asarray(y, dtype=np.float64) / pixel_size
    x0 = np.floor(xx).astype(np.int64)
    y0 = np.floor(yy).astype(np.int64)
    fx = xx - x0
    fy = yy - y0
    a = field[..., y0 % n_y, x0 % n_x]
    b = field[..., y0 % n_y, (x0 + 1) % n_x]
    c = field[..., (y0 + 1) % n_y, x0 % n_x]
    d = field[..., (y0 + 1) % n_y, (x0 + 1) % n_x]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def sample_gradient(field, x, y, pixel_size=1.0):
    """Exact gradient of the periodic bilinear interpolant returned by ``sample``."""
    n_y, n_x = field.shape
    xx = np.asarray(x, dtype=np.float64) / pixel_size
    yy = np.asarray(y, dtype=np.float64) / pixel_size
    x0 = np.floor(xx).astype(np.int64)
    y0 = np.floor(yy).astype(np.int64)
    fx, fy = xx - x0, yy - y0
    a = field[y0 % n_y, x0 % n_x]
    b = field[y0 % n_y, (x0 + 1) % n_x]
    c = field[(y0 + 1) % n_y, x0 % n_x]
    d = field[(y0 + 1) % n_y, (x0 + 1) % n_x]
    gradient_x = ((b - a) * (1 - fy) + (d - c) * fy) / pixel_size
    gradient_y = ((c - a) * (1 - fx) + (d - b) * fx) / pixel_size
    return np.stack((gradient_x, gradient_y), axis=-1)
