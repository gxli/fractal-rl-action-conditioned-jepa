"""Periodic mean-centered Gaussian field, squared potential, and derivatives."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Terrain:
    intensity: np.ndarray
    potential: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_reference: float
    pixel_size: float = 1.0

    @property
    def size(self):
        return self.potential.shape[0]


def generate(cfg, seed=None):
    n = int(cfg['resolution'])
    if n < 4:
        raise ValueError('resolution must be >= 4')
    dx = float(cfg['pixel_size'])
    if dx <= 0:
        raise ValueError('pixel_size must be positive')
    rng = np.random.default_rng(cfg['seed'] if seed is None else seed)
    white = rng.standard_normal((n, n))
    freq_y = np.fft.fftfreq(n, d=dx)[:, None]
    freq_x = np.fft.rfftfreq(n, d=dx)[None, :]
    radius = np.sqrt(freq_x**2 + freq_y**2)
    # Soft infrared cutoff suppresses the singular DC component.
    cutoff = 1.0 / (n * dx)
    weight = (radius**2 + cutoff**2)**(-float(cfg['spectral_exponent']) / 4.0)
    weight[0, 0] = 0.0
    field = np.fft.irfft2(np.fft.rfft2(white) * weight, s=(n, n)).real
    # The filtered field is Gaussian. Remove its finite-sample DC offset and
    # standardize it before squaring, so phi is centered-Gaussian energy with
    # a stable observation scale across resolutions and terrain seeds.
    intensity = field - field.mean()
    standard_deviation = float(np.std(intensity))
    if not np.isfinite(intensity).all() or standard_deviation <= 1e-12:
        raise ValueError('noise has zero dynamic range')
    intensity /= standard_deviation
    potential = intensity ** 2
    # Derivatives at bilinear-cell centres.  This is the same interpolant used
    # by physics, so its percentile provides a consistent field normalization.
    east = np.roll(potential, -1, axis=1)
    south = np.roll(potential, -1, axis=0)
    southeast = np.roll(south, -1, axis=1)
    gx = 0.5 * ((east - potential) + (southeast - south)) / dx
    gy = 0.5 * ((south - potential) + (southeast - east)) / dx
    gradient_norm = np.sqrt(gx**2 + gy**2)
    percentile = float(cfg.get('gradient_normalization_percentile', 95.0))
    if not 0 < percentile <= 100:
        raise ValueError('gradient_normalization_percentile must be in (0, 100]')
    gradient_reference = float(np.percentile(gradient_norm, percentile))
    if gradient_reference <= 1e-12:
        raise ValueError('potential gradient has zero scale')
    gx /= gradient_reference
    gy /= gradient_reference
    return Terrain(intensity.astype(np.float32), potential.astype(np.float32),
                   gx.astype(np.float32), gy.astype(np.float32), gradient_reference, dx)
