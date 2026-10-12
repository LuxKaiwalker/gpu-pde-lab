r"""2D heat equation: u_t = alpha * nabla^2 u.

Discretization (explicit FTCS -- Forward-Time, Centered-Space):

    u[i,j]^{n+1} = u[i,j]^n + dt * alpha * Laplacian(u^n)[i,j]

Stability (von Neumann analysis of the 5-point stencil on a square grid,
dx == dy == h):

    dt <= h^2 / (4 * alpha)

This bound is exposed via :meth:`HeatEquation.dt_max` and is the basis of
the stability experiment: pushing dt past this value makes the amplification
factor of some Fourier mode exceed 1 in magnitude, so high-frequency error
grows without bound every step (visible as checkerboard oscillation that
blows up).
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend
from ..core.boundary import BoundaryKind, apply_boundary
from ..core.grid import Grid2D
from ..core.stencil import laplacian
from ..initial_conditions import IC_REGISTRY
from .base import PDEModel


class HeatEquation(PDEModel):
    name = "heat"

    def __init__(self, grid: Grid2D, backend: Backend, boundary: BoundaryKind, alpha: float = 0.1):
        super().__init__(grid, backend, boundary)
        self.alpha = alpha
        self.u = backend.zeros(grid.shape)
        self.reset()

    def reset(self, ic: str = "gaussian", **ic_kwargs) -> None:
        X, Y = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            indexing="ij",
        )
        gen = IC_REGISTRY[ic]
        u0 = gen(X, Y, **ic_kwargs)
        self.u = self.backend.array(u0)
        apply_boundary(self.u, self.boundary)
        self.state.reset()

    def dt_max(self) -> float:
        h2 = min(self.grid.dx, self.grid.dy) ** 2
        return h2 / (4.0 * self.alpha)

    def step(self, dt: float) -> None:
        lap = laplacian(self.u, self.grid.dx, self.grid.dy, self.boundary)
        self.u = self.u + dt * self.alpha * lap
        apply_boundary(self.u, self.boundary)
        self.state.time += dt
        self.state.step += 1

    def inject_gaussian(self, cx: float, cy: float, amplitude: float = 1.0, sigma: float = 0.03) -> None:
        """Mouse-driven heat injection: add a localized bump at physical (cx, cy)."""
        X, Y = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            indexing="ij",
        )
        bump = amplitude * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)))
        self.u = self.u + self.backend.array(bump)

    def primary_field_numpy(self) -> np.ndarray:
        return self.backend.to_numpy(self.u)

    def total_heat(self) -> float:
        """Sum of u over the independent degrees of freedom -- should be
        (approximately) conserved under Neumann or periodic boundaries with
        no sources, since no heat flux leaves the domain.

        Under periodic boundaries every grid point is an independent cell,
        so the sum is taken over the whole array. Under Dirichlet/Neumann,
        the boundary ring is a *ghost* row/column (mirrored or fixed for
        the stencil's benefit, not an independently-evolving control
        volume), so it is excluded -- including it would count the
        boundary's mirrored copy of interior values as extra, spurious
        "heat". Dirichlet boundaries do not conserve this quantity regardless
        (heat can flow out through fixed-value walls).
        """
        if self.boundary == BoundaryKind.PERIODIC:
            return self.backend.sum(self.u)
        return self.backend.sum(self.u[1:-1, 1:-1])
