"""Initial condition generators, shared across the heat and wave experiments.

Each function takes host-side NumPy coordinate arrays (X, Y) over [0, lx] x
[0, ly] and returns a host-side NumPy field. Converting to a particular
backend/dtype happens afterwards via ``Backend.array``.
"""

from __future__ import annotations

import numpy as np


def point_source(X: np.ndarray, Y: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    u = np.zeros_like(X)
    nx, ny = X.shape
    u[nx // 2, ny // 2] = amplitude
    return u


def gaussian(X: np.ndarray, Y: np.ndarray, amplitude: float = 1.0, sigma: float = 0.05,
             cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    lx, ly = X.max(), Y.max()
    return amplitude * np.exp(-(((X - cx * lx) ** 2 + (Y - cy * ly) ** 2) / (2 * sigma ** 2)))


def multiple_sources(X: np.ndarray, Y: np.ndarray, amplitude: float = 1.0, sigma: float = 0.04,
                      n: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lx, ly = X.max(), Y.max()
    u = np.zeros_like(X)
    for _ in range(n):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        sign = rng.choice([-1.0, 1.0])
        u += sign * amplitude * np.exp(-(((X - cx * lx) ** 2 + (Y - cy * ly) ** 2) / (2 * sigma ** 2)))
    return u


def hot_cold_regions(X: np.ndarray, Y: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    lx = X.max()
    u = np.where(X < lx / 2, amplitude, -amplitude)
    return u.astype(X.dtype)


def zeros(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.zeros_like(X)


IC_REGISTRY = {
    "point_source": point_source,
    "gaussian": gaussian,
    "multiple_sources": multiple_sources,
    "hot_cold_regions": hot_cold_regions,
    "zeros": zeros,
}


def inject_gaussian_blob(u: np.ndarray, X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
                          amplitude: float = 1.0, sigma: float = 0.03) -> np.ndarray:
    """Add a localized Gaussian bump centered at physical coords (cx, cy).

    Used for mouse-driven heat injection / wave disturbance: the UI maps a
    click's pixel position to physical (cx, cy) and calls this to perturb
    the live field in place.
    """
    return u + amplitude * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)))
