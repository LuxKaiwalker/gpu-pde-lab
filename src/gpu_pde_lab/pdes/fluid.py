r"""2D incompressible fluid solver ("Stable Fluids", Stam 1999).

Governing equations (incompressible Navier-Stokes, velocity-pressure form):

    du/dt = -(u . grad)u + nu * nabla^2 u - grad(p) + f        (momentum)
    div(u) = 0                                                  (incompressibility)

Rather than solving this coupled system directly, Stam's method splits one
timestep into independent, each individually-simple stages (operator
splitting):

1. **Add forces** -- external input (mouse drag) added directly to velocity.
2. **Diffuse** -- apply viscosity implicitly via a few Jacobi iterations of
   $(I - \nu\Delta t\nabla^2)u = u_0$ (the same Jacobi relaxation used in
   the Poisson experiment, just solving a slightly different linear system).
3. **Project** -- solve a Poisson equation for pressure,
   $\nabla^2 p = \nabla \cdot u$, and subtract $\nabla p$ from the velocity.
   This is a Helmholtz decomposition step that removes the divergent part of
   the field, enforcing (approximate) incompressibility. It is the *same*
   Poisson-solve building block as the standalone Poisson experiment.
4. **Advect** -- move the (now divergence-free) velocity field along itself
   using semi-Lagrangian backtracing: for each cell, trace its position
   backward along the current velocity by one timestep and bilinearly
   sample the field there. This is unconditionally stable for any dt
   (no CFL restriction), which is the whole point of the method -- at the
   cost of numerical diffusion from repeated interpolation.
5. **Project again** -- advection can reintroduce divergence; one more
   pressure solve cleans it up.
6. **Advect dye** using the final divergence-free velocity field, for
   visualization.

Every stage reuses a primitive already built for the other experiments:
the 5-point Laplacian, Jacobi relaxation, and boundary handling. The fluid
solver is "new" only in how it *composes* those primitives.
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..core.interp import bilinear_sample


class StableFluids2D:
    def __init__(self, grid: Grid2D, backend: Backend, boundary: BoundaryKind = BoundaryKind.NEUMANN,
                 viscosity: float = 0.0001, diffusion_iters: int = 20, pressure_iters: int = 40):
        self.grid = grid
        self.backend = backend
        self.boundary = boundary
        self.viscosity = viscosity
        self.diffusion_iters = diffusion_iters
        self.pressure_iters = pressure_iters

        self.u = backend.zeros(grid.shape)
        self.v = backend.zeros(grid.shape)
        self.dye = backend.zeros(grid.shape)

        ii = backend.linspace(0.0, grid.nx - 1, grid.nx)
        jj = backend.linspace(0.0, grid.ny - 1, grid.ny)
        self._II, self._JJ = backend.meshgrid(ii, jj)

        self.time = 0.0
        self.step_count = 0

    def reset(self) -> None:
        self.u = self.backend.zeros(self.grid.shape)
        self.v = self.backend.zeros(self.grid.shape)
        self.dye = self.backend.zeros(self.grid.shape)
        self.time = 0.0
        self.step_count = 0

    def set_fields(self, u: np.ndarray | None = None, v: np.ndarray | None = None,
                    dye: np.ndarray | None = None) -> None:
        """Seed one or more fields from user-supplied host arrays (e.g. hand-drawn
        in the field editor), leaving any field not provided untouched. Never
        merges u/v/dye into a single buffer -- each stays an independent slot."""
        if u is not None:
            self.u = self.backend.array(u)
        if v is not None:
            self.v = self.backend.array(v)
        if dye is not None:
            self.dye = self.backend.array(dye)
        if u is not None or v is not None:
            self._apply_velocity_boundary()
        if dye is not None:
            self._apply_scalar_boundary(self.dye)
        self.time = 0.0
        self.step_count = 0

    # -- boundary handling ("solid box" vs periodic) ------------------------

    def _apply_velocity_boundary(self) -> None:
        if self.boundary == BoundaryKind.PERIODIC:
            return
        u, v = self.u, self.v
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

    def _apply_scalar_boundary(self, field) -> None:
        if self.boundary == BoundaryKind.PERIODIC:
            return
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

    # -- stages ---------------------------------------------------------

    def _diffuse(self, field, rate: float, dt: float):
        dx, dy = self.grid.dx, self.grid.dy
        a = rate * dt / (dx * dy)
        x = field
        x0 = field
        for _ in range(self.diffusion_iters):
            new = x * 0.0
            new[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (
                x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2]
            )) / (1.0 + 4.0 * a)
            self._apply_scalar_boundary(new)
            x = new
        return x

    def _divergence(self):
        u, v = self.u, self.v
        dx, dy = self.grid.dx, self.grid.dy
        div = u * 0.0
        div[1:-1, 1:-1] = (
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx) + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)
        )
        return div

    def _project(self) -> None:
        div = self._divergence()
        h2 = self.grid.dx * self.grid.dy
        p = self.u * 0.0
        for _ in range(self.pressure_iters):
            new_p = p * 0.0
            new_p[1:-1, 1:-1] = (
                p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - h2 * div[1:-1, 1:-1]
            ) / 4.0
            self._apply_scalar_boundary(new_p)
            p = new_p

        dx, dy = self.grid.dx, self.grid.dy
        self.u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
        self.v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy)
        self._apply_velocity_boundary()

    def _advect(self, field, dt: float):
        dx, dy = self.grid.dx, self.grid.dy
        gx = self._II - self.u * (dt / dx)
        gy = self._JJ - self.v * (dt / dy)
        if self.boundary == BoundaryKind.PERIODIC:
            nx, ny = self.grid.shape
            gx = gx % nx
            gy = gy % ny
        sampled = bilinear_sample(field, gx, gy, self.backend)
        self._apply_scalar_boundary(sampled)
        return sampled

    def step(self, dt: float) -> None:
        self._apply_velocity_boundary()

        if self.viscosity > 0:
            self.u = self._diffuse(self.u, self.viscosity, dt)
            self.v = self._diffuse(self.v, self.viscosity, dt)
            self._apply_velocity_boundary()

        self._project()

        self.u = self._advect(self.u, dt)
        self.v = self._advect(self.v, dt)
        self._apply_velocity_boundary()

        self._project()

        self.dye = self._advect(self.dye, dt)
        self._apply_scalar_boundary(self.dye)

        self.time += dt
        self.step_count += 1

    def add_source(self, cx: float, cy: float, fx: float, fy: float, dye_amount: float,
                    radius: float = 0.03) -> None:
        """Inject velocity + dye at physical coords (cx, cy), e.g. from a mouse drag."""
        X, Y = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            indexing="ij",
        )
        falloff = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * radius ** 2)))
        self.u = self.u + self.backend.array(fx * falloff)
        self.v = self.v + self.backend.array(fy * falloff)
        self.dye = self.dye + self.backend.array(dye_amount * falloff)

    def velocity_magnitude_numpy(self) -> np.ndarray:
        u = self.backend.to_numpy(self.u)
        v = self.backend.to_numpy(self.v)
        return np.sqrt(u * u + v * v)

    def dye_numpy(self) -> np.ndarray:
        return self.backend.to_numpy(self.dye)

    def divergence_norm(self) -> float:
        return self.backend.norm(self._divergence())
