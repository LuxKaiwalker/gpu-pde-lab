r"""Poisson equation: -nabla^2 u = f, discretized as a linear system A u = b.

We never form the matrix A explicitly. Instead every solver below is
"matrix-free": applying A to a vector u means computing
``-Laplacian(u)`` with the 5-point stencil, which is exactly the same
stencil operation used by the heat and wave equations. This is the key
conceptual link the Poisson experiment is meant to teach: an elliptic PDE
turns into *iterative linear algebra*, but the linear algebra itself still
bottoms out in the same stencil kernel, so it parallelizes the same way.

Three iterative solvers are implemented:

Jacobi
    u_{i,j}^{(k+1)} = 1/4 * ( u_{i+1,j}^{(k)} + u_{i-1,j}^{(k)}
                              + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k)} - h^2 f_{i,j} )

    Every new value only reads *old* (iteration k) neighbor values, so the
    whole grid can be updated in one embarrassingly-parallel sweep: there's
    no dependency between output cells within an iteration. This is what
    makes Jacobi map directly onto one GPU thread per cell.

Gauss-Seidel
    Same update, but cells are updated in place, in order, so later cells in
    the sweep read already-updated (iteration k+1) neighbors from earlier in
    the sweep. This converges roughly twice as fast per iteration as Jacobi,
    but the in-place, order-dependent update is inherently *sequential*
    along the sweep direction -- naively it does not parallelize across the
    whole grid the way Jacobi does. We implement it here as true sequential
    Gauss-Seidel (for CPU/teaching) and as a red-black Gauss-Seidel variant
    (for GPU), which restores parallelism by splitting the grid into two
    independent checkerboard color classes (like squares on a chessboard):
    all "red" cells depend only on "black" neighbors and vice versa, so each
    color can be updated in parallel, alternating colors each half-step.

Conjugate Gradient (CG)
    A Krylov-subspace method for symmetric positive-definite systems. Here
    A = -Laplacian (with the stencil's boundary handling) is SPD for
    Dirichlet boundaries. CG needs only matrix-vector products (i.e. stencil
    applications) and vector dot products/axpy updates, all of which are
    GPU-friendly, and it converges much faster than Jacobi/Gauss-Seidel for
    well-conditioned problems.
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend
from ..core.boundary import BoundaryKind, apply_boundary
from ..core.grid import Grid2D
from ..core.stencil import laplacian


def apply_A(u, dx: float, dy: float, boundary: BoundaryKind):
    """Matrix-free application of A = -Laplacian (restricted to interior points)."""
    return -laplacian(u, dx, dy, boundary)


class PoissonProblem:
    """Holds the grid/boundary/RHS for -nabla^2 u = f and iterative solver state."""

    def __init__(self, grid: Grid2D, backend: Backend, boundary: BoundaryKind = BoundaryKind.DIRICHLET):
        self.grid = grid
        self.backend = backend
        self.boundary = boundary
        self.f = backend.zeros(grid.shape)
        self.u = backend.zeros(grid.shape)
        self.residual_history: list[float] = []
        self._cg_r = None
        self._cg_p = None
        self._cg_rs_old = 0.0
        self._cg_converged = True

    def set_rhs_gaussian_sources(self, n: int = 2, amplitude: float = 50.0, sigma: float = 0.06,
                                  seed: int = 0) -> None:
        X, Y = np.meshgrid(
            np.linspace(0.0, self.grid.lx, self.grid.nx),
            np.linspace(0.0, self.grid.ly, self.grid.ny),
            indexing="ij",
        )
        rng = np.random.default_rng(seed)
        f = np.zeros_like(X)
        for k in range(n):
            cx, cy = rng.uniform(0.2, 0.8, size=2)
            sign = 1.0 if k % 2 == 0 else -1.0
            f += sign * amplitude * np.exp(-(((X - cx * self.grid.lx) ** 2 + (Y - cy * self.grid.ly) ** 2) / (2 * sigma ** 2)))
        self.f = self.backend.array(f)

    def reset_solution(self) -> None:
        self.u = self.backend.zeros(self.grid.shape)
        self.residual_history = []
        self._cg_r = None
        self._cg_p = None
        self._cg_converged = True

    def residual(self, u=None):
        u = self.u if u is None else u
        r = self.f - apply_A(u, self.grid.dx, self.grid.dy, self.boundary)
        # Only interior points are meaningful for Dirichlet/Neumann; zero the
        # boundary ring so it doesn't pollute the norm.
        if self.boundary != BoundaryKind.PERIODIC:
            r = self.backend.clone(r)
            r[0, :] = 0
            r[-1, :] = 0
            r[:, 0] = 0
            r[:, -1] = 0
        return r

    def residual_norm(self, u=None) -> float:
        return self.backend.norm(self.residual(u))

    # -- Jacobi -------------------------------------------------------------

    def jacobi_step(self) -> None:
        h2 = self.grid.dx * self.grid.dy
        b = self.backend
        center = self.u
        up = center * 0.0
        up[1:-1, 1:-1] = (
            center[:-2, 1:-1] + center[2:, 1:-1] + center[1:-1, :-2] + center[1:-1, 2:]
            + h2 * self.f[1:-1, 1:-1]
        ) / 4.0
        apply_boundary(up, self.boundary)
        self.u = up

    # -- Gauss-Seidel (sequential, reference / CPU teaching version) -------

    def gauss_seidel_step_sequential(self) -> None:
        """True in-place Gauss-Seidel. Only meaningful on NumPy (CPU); this
        is O(N^2) Python-level iteration and is intentionally slow -- it
        exists to let the user *see* why naive Gauss-Seidel does not
        parallelize, by contrast with the red-black version below."""
        u = self.u
        f = self.u * 0.0 + self.f  # ensure same backend object type
        h2 = self.grid.dx * self.grid.dy
        u_host = self.backend.to_numpy(u)
        f_host = self.backend.to_numpy(f)
        nx, ny = u_host.shape
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_host[i, j] = 0.25 * (
                    u_host[i - 1, j] + u_host[i + 1, j] + u_host[i, j - 1] + u_host[i, j + 1]
                    + h2 * f_host[i, j]
                )
        self.u = self.backend.array(u_host)
        apply_boundary(self.u, self.boundary)

    # -- Red-black Gauss-Seidel (parallel, GPU-friendly) --------------------

    def gauss_seidel_step_redblack(self) -> None:
        """Parallel Gauss-Seidel via red/black checkerboard ordering.

        A cell (i,j) is "red" if (i+j) is even, "black" if odd. Every red
        cell's four neighbors are all black and vice versa, so updating all
        red cells using only black values (then all black cells using the
        just-updated red values) reproduces Gauss-Seidel's faster
        convergence while keeping each half-sweep fully data-parallel --
        this is the standard GPU-friendly Gauss-Seidel variant.
        """
        h2 = self.grid.dx * self.grid.dy
        nx, ny = self.grid.shape
        ii = np.arange(nx)
        jj = np.arange(ny)
        parity = (ii[:, None] + jj[None, :]) % 2  # 0 = red, 1 = black
        mask_red = self.backend.array((parity == 0).astype(np.float64))
        mask_black = self.backend.array((parity == 1).astype(np.float64))

        for mask in (mask_red, mask_black):
            u = self.u
            new_interior = u * 0.0
            new_interior[1:-1, 1:-1] = (
                u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] + h2 * self.f[1:-1, 1:-1]
            ) / 4.0
            self.u = mask * new_interior + (1.0 - mask) * u
            apply_boundary(self.u, self.boundary)

    # -- Conjugate Gradient ---------------------------------------------------

    def cg_init(self) -> None:
        """(Re)start CG from the current `self.u`, storing the (r, p, rs_old)
        state needed to take single steps -- used by the live stepping UI so
        CG can be visualized one iteration at a time, just like Jacobi/GS."""
        b = self.backend
        r = self.residual(self.u)
        self._cg_r = r
        self._cg_p = b.clone(r)
        self._cg_rs_old = b.sum(r * r)
        self._cg_converged = False
        self.residual_history = [float(np.sqrt(max(self._cg_rs_old, 0.0)))]

    def cg_step(self, tol: float = 1e-8) -> float:
        """Perform one CG iteration (assumes cg_init() was already called).
        Returns the new residual norm. A no-op (returns last residual) once
        converged or once the search direction degenerates."""
        if getattr(self, "_cg_converged", True) and getattr(self, "_cg_r", None) is None:
            self.cg_init()
        if self._cg_converged:
            return self.residual_history[-1]

        b = self.backend
        p, r, rs_old = self._cg_p, self._cg_r, self._cg_rs_old
        Ap = apply_A(p, self.grid.dx, self.grid.dy, self.boundary)
        if self.boundary != BoundaryKind.PERIODIC:
            Ap = b.clone(Ap)
            Ap[0, :] = 0
            Ap[-1, :] = 0
            Ap[:, 0] = 0
            Ap[:, -1] = 0
        denom = b.sum(p * Ap)
        if abs(denom) < 1e-30:
            self._cg_converged = True
            return self.residual_history[-1]

        alpha = rs_old / denom
        self.u = self.u + alpha * p
        r = r - alpha * Ap
        rs_new = b.sum(r * r)
        res_norm = float(np.sqrt(max(rs_new, 0.0)))
        self.residual_history.append(res_norm)
        apply_boundary(self.u, self.boundary)

        if res_norm < tol:
            self._cg_converged = True
        else:
            self._cg_p = r + (rs_new / rs_old) * p
            self._cg_r = r
            self._cg_rs_old = rs_new
        return res_norm

    def solve_cg(self, max_iter: int = 500, tol: float = 1e-6) -> list[float]:
        """Run CG to (approximate) convergence in one call, for headless/batch use."""
        self.cg_init()
        for _ in range(max_iter):
            self.cg_step(tol=tol)
            if self._cg_converged:
                break
        return self.residual_history
