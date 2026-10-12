r"""2D wave equation: u_tt = c^2 * nabla^2 u - gamma * u_t  (gamma = damping).

Discretized with central differences in time (leapfrog / Stormer-Verlet),
which is the standard explicit scheme for a second-order-in-time PDE:

    u^{n+1} = 2u^n - u^{n-1} + dt^2 * (c^2 * Laplacian(u^n) - gamma * (u^n - u^{n-1})/dt)

Rearranged to avoid dividing by dt in a way that would complicate damping=0:

    u^{n+1} = ( 2u^n - u^{n-1} + dt^2 * c^2 * Laplacian(u^n)
                + gamma*dt/2 * u^{n-1} ) / (1 + gamma*dt/2)

This is the standard "damped leapfrog" discretization, averaging the damping
term implicitly in time for stability.

Stability (CFL condition for the 2D wave equation on a square grid):

    c * dt / h <= 1 / sqrt(2)   i.e.   dt <= h / (c * sqrt(2))

Exceeding this allows a Fourier mode's amplification factor to exceed 1 in
magnitude, and the simulation blows up -- same qualitative story as the heat
equation's instability, but governed by a different (hyperbolic, not
parabolic) CFL number.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.backend import Backend
from ..core.boundary import BoundaryKind, apply_boundary
from ..core.grid import Grid2D
from ..core.stencil import laplacian
from ..initial_conditions import IC_REGISTRY
from .base import PDEModel


class WaveEquation(PDEModel):
    name = "wave"

    def __init__(self, grid: Grid2D, backend: Backend, boundary: BoundaryKind,
                 c: float = 1.0, damping: float = 0.0):
        super().__init__(grid, backend, boundary)
        self.c = c
        self.damping = damping
        self.u = backend.zeros(grid.shape)
        self.u_prev = backend.zeros(grid.shape)
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
        self.u_prev = self.backend.clone(self.u)  # zero initial velocity
        apply_boundary(self.u, self.boundary)
        apply_boundary(self.u_prev, self.boundary)
        self.state.reset()

    def dt_max(self) -> float:
        h = min(self.grid.dx, self.grid.dy)
        return h / (self.c * math.sqrt(2.0))

    def step(self, dt: float) -> None:
        lap = laplacian(self.u, self.grid.dx, self.grid.dy, self.boundary)
        g = self.damping
        u_new = (2 * self.u - self.u_prev + (dt * dt) * (self.c ** 2) * lap
                 + (g * dt / 2) * self.u_prev) / (1.0 + g * dt / 2)
        apply_boundary(u_new, self.boundary)
        self.u_prev = self.u
        self.u = u_new
        self.state.time += dt
        self.state.step += 1

    def disturb(self, cx: float, cy: float, amplitude: float = 1.0, sigma: float = 0.03) -> None:
        """Mouse-driven membrane disturbance at physical coords (cx, cy)."""
        X, Y = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            indexing="ij",
        )
        bump = amplitude * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)))
        self.u = self.u + self.backend.array(bump)

    def primary_field_numpy(self) -> np.ndarray:
        return self.backend.to_numpy(self.u)

    def energy(self, dt: float) -> float:
        """Approximate total energy (kinetic + potential) given the last-used dt.

        Used to sanity-check damping=0 runs: energy should stay roughly
        constant (up to discretization error) and should monotonically
        decrease once damping > 0."""
        velocity = (self.u - self.u_prev) / dt
        kinetic = 0.5 * self.backend.sum(velocity * velocity)
        lap = laplacian(self.u, self.grid.dx, self.grid.dy, self.boundary)
        potential = -0.5 * (self.c ** 2) * self.backend.sum(self.u * lap)
        return kinetic + potential

    def max_amplitude(self) -> float:
        return self.backend.max(self.backend.abs(self.u))
