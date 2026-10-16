r"""3D heat equation: u_t = alpha * nabla^2 u, on a Grid3D.

Direct 3D generalization of ``pdes/heat.py``: same explicit FTCS scheme,
same backend abstraction, same reset/step/dt_max shape -- the only real
differences are the 7-point (not 5-point) stencil and a 3-term (not 2-term)
stability bound.

Discretization (explicit FTCS):

    u[i,j,k]^{n+1} = u[i,j,k]^n + dt * alpha * Laplacian_3D(u^n)[i,j,k]

Stability (von Neumann analysis of the 7-point stencil):

    dt <= 1 / ( 2*alpha*(1/dx^2 + 1/dy^2 + 1/dz^2) )

For an isotropic grid (dx=dy=dz=h) this is h^2/(6*alpha) -- the direct 3D
generalization of the 2D bound h^2/(4*alpha): each additional dimension adds
one more pair of neighbors contributing to the same von Neumann amplification
analysis, tightening the bound by exactly that dimension's curvature term.
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend
from ..core.boundary import BoundaryKind, apply_boundary_3d
from ..core.grid import Grid3D
from ..core.stencil import laplacian_3d
from .base import PDEModel


def _gaussian_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, amplitude: float = 1.0,
                  sigma: float = 0.1, cx: float = 0.5, cy: float = 0.5, cz: float = 0.5) -> np.ndarray:
    lx, ly, lz = X.max(), Y.max(), Z.max()
    return amplitude * np.exp(
        -(((X - cx * lx) ** 2 + (Y - cy * ly) ** 2 + (Z - cz * lz) ** 2) / (2 * sigma ** 2))
    )


def _zeros_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.zeros_like(X)


# Deliberately minimal relative to the 2D `IC_REGISTRY` in initial_conditions.py --
# just what this one 3D experiment needs, not a speculative full 3D generator
# library.
IC_REGISTRY_3D = {
    "gaussian": _gaussian_3d,
    "zeros": _zeros_3d,
}


class HeatEquation3D(PDEModel):
    name = "heat_3d"

    def __init__(self, grid: Grid3D, backend: Backend, boundary: BoundaryKind, alpha: float = 0.1):
        super().__init__(grid, backend, boundary)
        self.alpha = alpha
        self.u = backend.zeros(grid.shape)
        self.reset()

    def reset(self, ic: str = "gaussian", **ic_kwargs) -> None:
        X, Y, Z = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            np.linspace(0.0, self.grid.lz, self.grid.nz),
            indexing="ij",
        )
        gen = IC_REGISTRY_3D[ic]
        u0 = gen(X, Y, Z, **ic_kwargs)
        self.u = self.backend.array(u0)
        apply_boundary_3d(self.u, self.boundary)
        self.state.reset()

    def dt_max(self) -> float:
        inv_h2_sum = 1.0 / self.grid.dx ** 2 + 1.0 / self.grid.dy ** 2 + 1.0 / self.grid.dz ** 2
        return 1.0 / (2.0 * self.alpha * inv_h2_sum)

    def step(self, dt: float) -> None:
        lap = laplacian_3d(self.u, self.grid.dx, self.grid.dy, self.grid.dz, self.boundary)
        self.u = self.u + dt * self.alpha * lap
        apply_boundary_3d(self.u, self.boundary)
        self.state.time += dt
        self.state.step += 1

    def primary_field_numpy(self) -> np.ndarray:
        """The full (nx, ny, nz) host array -- genuinely 3D data, not a 2D
        slice. ``Heat3DExperiment.field_numpy()`` takes a mid-Z slice of this
        for the legacy 2D-only display contract; the real display path
        (``ui/viz/slice_viewer_3d.py``) consumes this full volume directly."""
        return self.backend.to_numpy(self.u)
