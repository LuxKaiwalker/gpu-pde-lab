"""Lightweight velocity-field quiver overlay, built from plain pyqtgraph
primitives (no new dependency): a down-sampled grid of 2-point line segments
drawn via ``pg.PlotDataItem(x, y, connect="pairs")``, one pair per arrow
shaft. Deliberately a pure function -- no Qt import here -- so it's testable
headless and the caller decides how/whether to plot it.
"""

from __future__ import annotations

import numpy as np


def build_quiver_segments(u: np.ndarray, v: np.ndarray, stride: int = 8,
                           scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample the (u, v) velocity field by `stride` and return (x, y)
    coordinate arrays for `pg.PlotDataItem(x, y, connect="pairs")`: each
    consecutive pair of points is one arrow shaft from a sample's grid
    position to that position plus its (scaled) velocity vector.
    """
    if u.shape != v.shape:
        raise ValueError(f"u/v shape mismatch: {u.shape} vs {v.shape}")
    nx, ny = u.shape
    ii = np.arange(0, nx, stride)
    jj = np.arange(0, ny, stride)
    gi, gj = np.meshgrid(ii, jj, indexing="ij")

    x0 = gi.astype(np.float64).ravel()
    y0 = gj.astype(np.float64).ravel()
    x1 = x0 + scale * u[gi, gj].astype(np.float64).ravel()
    y1 = y0 + scale * v[gi, gj].astype(np.float64).ravel()

    x = np.empty(2 * x0.size, dtype=np.float64)
    y = np.empty(2 * y0.size, dtype=np.float64)
    x[0::2], x[1::2] = x0, x1
    y[0::2], y[1::2] = y0, y1
    return x, y
