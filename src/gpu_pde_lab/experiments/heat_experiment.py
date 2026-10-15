"""Heat equation experiment, combining sections 4 and 5 of the spec: the
live 2D heat simulation plus an explicit stability/instability explorer.

The dt slider is deliberately allowed to go past the stable bound -- with an
"allow unstable dt" checkbox gating it, per the "explicit explore instability
mode" requirement. The stats panel always reports STABLE/UNSTABLE based on
dt vs dt_max, which is the actual mathematical criterion, not a crash
detector: the sim will run (and visibly blow up) even when "unstable".
"""

from __future__ import annotations

import time

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.heat import HeatEquation
from ..ui.controls import ControlPanel
from .base import EditSpec, LiveExperiment

EXPLANATION = r"""
### What PDE are we solving?

The 2D heat (diffusion) equation:

$$ \frac{\partial u}{\partial t} = \alpha \nabla^2 u $$

### How is it discretized?

Space: the standard 5-point finite-difference Laplacian,

$$ \nabla^2 u_{i,j} \approx \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2} $$

Time: explicit (forward) Euler,

$$ u^{n+1} = u^n + \Delta t\,\alpha \nabla^2 u^n $$

### What does one grid cell compute?

Cell $(i,j)$ reads its own value and its four direct neighbors, and writes
one new value. That's it -- no other cell's data is needed.

### Why can it run in parallel?

Because every output cell only depends on the *previous* timestep's
neighbors, not on other cells being updated *this* step. There is no
ordering requirement between cells within a single update -- the update for
every cell could happen simultaneously, in any order, and give the same
answer.

### What does the GPU actually do?

PyTorch dispatches the shifted-slice addition as one CUDA kernel launch.
The GPU scheduler spreads the ~N² independent cell updates across thousands
of CUDA cores in parallel, each handling a small tile of cells. The kernel
is memory-bound: it spends most of its time moving the 5 neighbor reads +
1 write per cell to/from GPU memory, not doing the handful of arithmetic
operations.

### What can go wrong? -- Stability

The explicit scheme is only *conditionally stable*. Von Neumann analysis of
the 5-point stencil gives:

$$ \Delta t \le \frac{h^2}{4\alpha} $$

Push $\Delta t$ past this bound (toggle "allow unstable dt" below) and the
amplification factor for high-frequency error exceeds 1 in magnitude: small
rounding/discretization error grows geometrically every step. You'll see
checkerboard oscillation, then the field diverging to ±∞ within a handful of
steps. This is **not a bug** -- it's the same floating-point equations,
computed correctly, revealing a genuine property of the discretization.
Running this on a faster GPU does not fix it; it just reaches ∞ sooner.
"""


class HeatExperiment(LiveExperiment):
    title = "Heat Equation (+ Stability)"

    def __init__(self):
        self.n = 256
        self.alpha = 0.12
        self.dt_fraction = 0.5  # fraction of dt_max, can exceed 1.0 if unlocked
        self.allow_unstable = False
        self.boundary = BoundaryKind.NEUMANN
        self.ic = "gaussian"
        self.backend_name = "GPU" if cuda_is_available() else "CPU"
        self.precision = "fp32"
        self._last_dt_used = 0.0
        self._fps_timestamps: list[float] = []
        self._build_pde()

    def _make_backend(self) -> Backend:
        if self.backend_name == "GPU" and cuda_is_available():
            return Backend(BackendSpec("torch", "cuda", self.precision))
        return Backend(BackendSpec("torch", "cpu", self.precision if self.precision != "fp16" else "fp32"))

    def _build_pde(self) -> None:
        backend = self._make_backend()
        grid = Grid2D(nx=self.n, ny=self.n)
        self.pde = HeatEquation(grid, backend, self.boundary, alpha=self.alpha)
        self.pde.reset(ic=self.ic)

    # -- LiveExperiment interface --------------------------------------

    def reset(self) -> None:
        self._build_pde()

    def dt_max(self) -> float:
        return self.pde.dt_max()

    def _current_dt(self) -> float:
        return self.dt_fraction * self.pde.dt_max()

    def step(self, dt: float) -> None:
        use_dt = self._current_dt()
        if not self.allow_unstable:
            use_dt = min(use_dt, self.pde.dt_max())
        self._last_dt_used = use_dt
        self.pde.step(use_dt)
        now = time.perf_counter()
        self._fps_timestamps.append(now)
        if len(self._fps_timestamps) > 120:
            self._fps_timestamps.pop(0)

    def field_numpy(self):
        return self.pde.primary_field_numpy()

    # -- field editor hooks ----------------------------------------------

    def editable_fields(self) -> dict[str, EditSpec]:
        return {
            "u0": EditSpec(
                key="u0",
                label="Initial temperature u(x,y,0)",
                clip_to_interior=self.boundary != BoundaryKind.PERIODIC,
            )
        }

    def edit_field_numpy(self, key: str) -> np.ndarray:
        if key != "u0":
            raise NotImplementedError(f"HeatExperiment has no editable field {key!r}")
        return self.pde.primary_field_numpy()

    def apply_edited_fields(self, fields: dict[str, np.ndarray]) -> None:
        self.pde.reset_from_array(fields["u0"])

    def editor_grid(self):
        return self.pde.grid

    def supports_click_injection(self) -> bool:
        return True

    def on_click(self, x_frac: float, y_frac: float) -> None:
        cx, cy = x_frac * self.pde.grid.lx, y_frac * self.pde.grid.ly
        self.pde.inject_gaussian(cx, cy, amplitude=1.0, sigma=0.03)

    def stats(self) -> dict:
        dt_max = self.pde.dt_max()
        dt_used = self._last_dt_used or self._current_dt()
        ratio = dt_used / dt_max if dt_max > 0 else float("inf")
        stability = "STABLE" if ratio <= 1.0 + 1e-9 else "UNSTABLE"
        field = self.field_numpy()
        finite = bool(np.all(np.isfinite(field)))
        if not finite:
            stability = "DIVERGED (inf/nan)"

        fps = 0.0
        if len(self._fps_timestamps) >= 2:
            span = self._fps_timestamps[-1] - self._fps_timestamps[0]
            fps = (len(self._fps_timestamps) - 1) / span if span > 0 else 0.0

        mem_mb = self.pde.backend.memory_allocated_bytes()
        return {
            "Sim time": f"{self.pde.state.time:.5f}",
            "Step": str(self.pde.state.step),
            "Grid": f"{self.n} x {self.n} ({self.n*self.n:,} cells)",
            "Backend": self.pde.backend.spec.label(),
            "Device": self.pde.backend.device_name(),
            "dt": f"{dt_used:.3e}",
            "dt_max": f"{dt_max:.3e}",
            "dt / dt_max": f"{ratio:.2f}x",
            "Stability": stability,
            "FPS": f"{fps:.1f}",
            "Steps/s": f"{fps:.1f}",
            "GPU memory": f"{mem_mb/1e6:.2f} MB" if mem_mb is not None else "n/a",
        }

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_alpha(v):
            self.alpha = v
            self.pde.alpha = v

        panel.add_slider("alpha (diffusivity)", 0.001, 1.0, self.alpha, step=0.001, decimals=4,
                          log_scale=True, on_change=set_alpha)

        def set_dt_fraction(v):
            self.dt_fraction = v

        panel.add_slider("dt (fraction of dt_max)", 0.01, 3.0, self.dt_fraction, step=0.01,
                          decimals=3, on_change=set_dt_fraction)

        def set_unstable(checked):
            self.allow_unstable = checked

        panel.add_checkbox("Allow unstable dt (explore instability)", self.allow_unstable,
                            on_change=set_unstable)

        def set_grid(text):
            self.n = int(text)
            on_change()

        panel.add_combo("Grid resolution", ["64", "128", "256", "512", "1024", "2048"],
                         current=str(self.n), on_change=set_grid)

        def set_boundary(text):
            self.boundary = BoundaryKind(text.lower())
            on_change()

        panel.add_combo("Boundary condition", ["dirichlet", "neumann", "periodic"],
                         current=self.boundary.value, on_change=set_boundary)

        def set_ic(text):
            self.ic = text
            on_change()

        panel.add_combo("Initial condition",
                         ["point_source", "gaussian", "multiple_sources", "hot_cold_regions", "zeros"],
                         current=self.ic, on_change=set_ic)

        backend_opts = ["GPU", "CPU"] if cuda_is_available() else ["CPU"]

        def set_backend(text):
            self.backend_name = text
            on_change()

        panel.add_combo("Backend", backend_opts, current=self.backend_name, on_change=set_backend)

        def set_precision(text):
            self.precision = text
            on_change()

        panel.add_combo("Precision", ["fp16", "fp32", "fp64"], current=self.precision,
                         on_change=set_precision)

        return panel
