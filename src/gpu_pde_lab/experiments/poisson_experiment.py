"""Poisson equation experiment: -nabla^2 u = f as iterative linear algebra.

Lets the user watch Jacobi, (sequential or red-black) Gauss-Seidel, and
Conjugate Gradient converge to the same solution at very different rates,
and compare how naturally each parallelizes.
"""

from __future__ import annotations

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.poisson import PoissonProblem
from ..ui.controls import ControlPanel
from .base import EditSpec, LiveExperiment

EXPLANATION = r"""
### What PDE are we solving?

The Poisson equation, an *elliptic* (steady-state) PDE:

$$ -\nabla^2 u = f $$

Unlike the heat/wave equations there is no time axis -- we solve for the
equilibrium field directly. The field editor below lets you hand-draw the
*source* $f(x,y)$ (e.g. where heat/charge is injected or removed) -- it does
**not** set an initial guess for $u$, which is always re-solved from zero.
Boundary values of $u$ are fixed separately by the chosen boundary condition,
independent of whatever $f$ you draw.

### How is it discretized?

The same 5-point Laplacian turns this into a linear system $Au = b$, where
$A$ represents "negative discrete Laplacian" and is never built as an
explicit matrix -- applying $A$ to a vector just means running the stencil.

### What does one "iteration" compute?

**Jacobi**: $u_{i,j}^{(k+1)} = \frac14\left(u_{i+1,j}^{(k)}+u_{i-1,j}^{(k)}+u_{i,j+1}^{(k)}+u_{i,j-1}^{(k)}-h^2 f_{i,j}\right)$,
reading only *old* (iteration $k$) neighbor values.

**Gauss-Seidel**: the same formula, but updated in place -- cells later in
the sweep read neighbors that were *already updated this iteration*. This
converges roughly 2x faster per iteration than Jacobi.

**Conjugate Gradient**: builds a Krylov subspace from repeated $A$-stencil
applications and takes an optimal step within it each iteration -- far
fewer iterations than Jacobi/GS for the same accuracy.

### Why can it run in parallel?

Jacobi: trivially -- every cell reads only last iteration's values, so the
whole grid updates in one parallel pass (exactly like one heat-equation
step). Gauss-Seidel: the naive sequential sweep is *not* parallel (cell
$(i,j)$ depends on a cell updated earlier in the same pass) -- select
"Gauss-Seidel (red-black)" to see the GPU-friendly fix: splitting the grid
into a checkerboard of two independent colors restores full parallelism
within each color, at the cost of two half-sweeps instead of one.
Conjugate Gradient: every iteration is a stencil application plus vector
dot-products/axpy, all embarrassingly parallel -- but CG has a true serial
dependency *between* iterations (each iteration needs the previous one's
result), so its speed comes from fewer iterations, not from more
parallelism per iteration.

### What can go wrong?

With zero Dirichlet boundaries, $A$ is symmetric positive-definite, so CG
is guaranteed to converge. Jacobi/Gauss-Seidel always converge for this
discrete Laplacian too, but can take thousands of iterations on large
grids -- the residual-vs-iteration plot below makes the difference in
convergence *rate* directly visible.
"""


class PoissonExperiment(LiveExperiment):
    title = "Poisson Equation (Iterative Solvers)"

    def __init__(self):
        self.n = 128
        self.method = "conjugate_gradient"
        self.backend_name = "GPU" if cuda_is_available() else "CPU"
        self.n_sources = 2
        self._build_problem()

    def _make_backend(self) -> Backend:
        if self.backend_name == "GPU" and cuda_is_available():
            return Backend(BackendSpec("torch", "cuda", "fp32"))
        return Backend(BackendSpec("torch", "cpu", "fp32"))

    def _build_problem(self) -> None:
        backend = self._make_backend()
        grid = Grid2D(nx=self.n, ny=self.n)
        self.problem = PoissonProblem(grid, backend, BoundaryKind.DIRICHLET)
        self.problem.set_rhs_gaussian_sources(n=self.n_sources, amplitude=50.0)
        self.problem.reset_solution()
        self._iteration = 0

    def reset(self) -> None:
        self._build_problem()

    def step(self, dt: float) -> None:
        if self.method == "jacobi":
            self.problem.jacobi_step()
            self.problem.residual_history.append(self.problem.residual_norm())
        elif self.method == "gauss_seidel_redblack":
            self.problem.gauss_seidel_step_redblack()
            self.problem.residual_history.append(self.problem.residual_norm())
        elif self.method == "gauss_seidel_sequential":
            self.problem.gauss_seidel_step_sequential()
            self.problem.residual_history.append(self.problem.residual_norm())
        else:  # conjugate_gradient
            self.problem.cg_step()
        self._iteration += 1

    def field_numpy(self):
        return self.problem.backend.to_numpy(self.problem.u)

    # -- field editor hooks ----------------------------------------------

    def editable_fields(self) -> dict[str, EditSpec]:
        return {
            "f": EditSpec(
                key="f",
                label="Source term f(x,y) in  -∇²u = f",
                clip_to_interior=False,
            )
        }

    def edit_field_numpy(self, key: str) -> np.ndarray:
        if key != "f":
            raise NotImplementedError(f"PoissonExperiment has no editable field {key!r}")
        return self.problem.backend.to_numpy(self.problem.f)

    def apply_edited_fields(self, fields: dict[str, np.ndarray]) -> None:
        self.problem.set_rhs_from_array(fields["f"])
        self._iteration = 0

    def editor_grid(self):
        return self.problem.grid

    def extra_plot_data(self):
        return {"residual_history": list(self.problem.residual_history)}

    def stats(self) -> dict:
        hist = self.problem.residual_history
        res = hist[-1] if hist else self.problem.residual_norm()
        return {
            "Method": self.method,
            "Iteration": str(self._iteration),
            "Residual norm": f"{res:.3e}",
            "Grid": f"{self.n} x {self.n}",
            "Backend": self.problem.backend.spec.label(),
            "GPU memory": (f"{m/1e6:.2f} MB" if (m := self.problem.backend.memory_allocated_bytes()) is not None else "n/a"),
        }

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_method(text):
            self.method = text
            self._iteration = 0
            self.problem.reset_solution()

        panel.add_combo("Method",
                         ["jacobi", "gauss_seidel_redblack", "gauss_seidel_sequential", "conjugate_gradient"],
                         current=self.method, on_change=set_method)

        def set_grid(text):
            self.n = int(text)
            on_change()

        panel.add_combo("Grid resolution", ["32", "64", "128", "256", "512"],
                         current=str(self.n), on_change=set_grid)

        def set_sources(text):
            self.n_sources = int(text)
            on_change()

        panel.add_combo("Number of sources", ["1", "2", "3", "4"], current=str(self.n_sources),
                         on_change=set_sources)

        backend_opts = ["GPU", "CPU"] if cuda_is_available() else ["CPU"]

        def set_backend(text):
            self.backend_name = text
            on_change()

        panel.add_combo("Backend", backend_opts, current=self.backend_name, on_change=set_backend)

        return panel
