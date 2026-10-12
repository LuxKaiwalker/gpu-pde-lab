r"""Convergence / resolution experiment: discretization error vs. GPU speed.

We use a heat-equation initial condition with a known closed-form solution
so that numerical error can be measured directly rather than only inferred.

On the square [0, L] x [0, L] with homogeneous Dirichlet boundaries and
initial condition

    u0(x, y) = sin(pi x / L) * sin(pi y / L)

the exact solution of u_t = alpha * nabla^2 u is the single decaying Fourier
mode

    u(x, y, t) = exp(-alpha * pi^2 * (1/L^2 + 1/L^2) * t) * sin(pi x / L) * sin(pi y / L)

because this initial condition is already an eigenfunction of the Laplacian
with eigenvalue -pi^2 (1/L^2 + 1/L^2) under Dirichlet boundaries. This lets
us isolate two independent sources of numerical error:

* Spatial (discretization) error: fix dt very small (negligible time error),
  vary grid spacing h, and measure max error vs. the analytical solution at
  a fixed physical time. The 5-point stencil is second-order accurate in
  space, so error should shrink like O(h^2) -- halving h should cut the
  error by ~4x.

* Temporal error: fix a fine grid (negligible spatial error), vary dt
  (staying under the explicit-Euler stability bound), and measure error at
  a fixed physical time. Forward Euler is first-order accurate in time, so
  error should shrink like O(dt) -- halving dt should cut the error by ~2x.

The central lesson (section 14 of the spec): a GPU can run an inaccurate
discretization very fast, but it cannot fix the discretization's order of
accuracy. Running the exact same explicit-Euler/5-point-stencil combination
on a GPU does not change these convergence rates at all -- only the wall
time needed to produce them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.backend import Backend, BackendSpec
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.heat import HeatEquation


def analytical_heat_solution(X: np.ndarray, Y: np.ndarray, t: float, alpha: float, L: float) -> np.ndarray:
    decay = math.exp(-alpha * (math.pi ** 2) * (1.0 / L ** 2 + 1.0 / L ** 2) * t)
    return decay * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)


def _run_to_time(n: int, L: float, alpha: float, dt: float, t_final: float, backend: Backend):
    grid = Grid2D(nx=n, ny=n, lx=L, ly=L)
    heat = HeatEquation(grid, backend, BoundaryKind.DIRICHLET, alpha=alpha)
    X, Y = np.meshgrid(np.linspace(0, L, n), np.linspace(0, L, n), indexing="ij")
    u0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    heat.u = backend.array(u0)
    n_steps = max(1, round(t_final / dt))
    actual_dt = t_final / n_steps
    for _ in range(n_steps):
        heat.step(actual_dt)
    numeric = heat.primary_field_numpy()
    exact = analytical_heat_solution(X, Y, heat.state.time, alpha, L)
    return numeric, exact


@dataclass
class ConvergencePoint:
    h_or_dt: float
    error_linf: float


@dataclass
class ConvergenceResult:
    points: list[ConvergencePoint]
    observed_orders: list[float]  # one per consecutive pair, log2(e1/e2)

    @staticmethod
    def from_points(points: list[ConvergencePoint]) -> "ConvergenceResult":
        orders = []
        for a, b in zip(points, points[1:]):
            if b.error_linf > 0 and a.h_or_dt > 0 and b.h_or_dt > 0:
                ratio_h = a.h_or_dt / b.h_or_dt
                ratio_e = a.error_linf / b.error_linf
                orders.append(math.log(ratio_e) / math.log(ratio_h))
            else:
                orders.append(float("nan"))
        return ConvergenceResult(points=points, observed_orders=orders)


def spatial_convergence(sizes: list[int], alpha: float = 0.1, L: float = 1.0,
                         t_final: float = 0.01) -> ConvergenceResult:
    """Fix dt tiny (relative to the finest grid's stability bound) and vary h."""
    backend = Backend(BackendSpec("numpy", "cpu", "fp64"))
    finest_dt_max = (L / (max(sizes) - 1)) ** 2 / (4 * alpha)
    dt = finest_dt_max * 0.1  # comfortably stable and small time-error for all sizes

    points = []
    for n in sizes:
        h = L / (n - 1)
        numeric, exact = _run_to_time(n, L, alpha, dt, t_final, backend)
        err = float(np.max(np.abs(numeric - exact)))
        points.append(ConvergencePoint(h_or_dt=h, error_linf=err))
    return ConvergenceResult.from_points(points)


def temporal_convergence(dts: list[float], n: int = 128, alpha: float = 0.1, L: float = 1.0,
                          t_final: float = 0.01) -> ConvergenceResult:
    """Fix a fine grid and vary dt (all values must stay under the stability bound)."""
    backend = Backend(BackendSpec("numpy", "cpu", "fp64"))
    grid = Grid2D(nx=n, ny=n, lx=L, ly=L)
    dt_max = (min(grid.dx, grid.dy)) ** 2 / (4 * alpha)

    points = []
    for dt in dts:
        if dt >= dt_max:
            continue
        numeric, exact = _run_to_time(n, L, alpha, dt, t_final, backend)
        err = float(np.max(np.abs(numeric - exact)))
        points.append(ConvergencePoint(h_or_dt=dt, error_linf=err))
    return ConvergenceResult.from_points(points)
