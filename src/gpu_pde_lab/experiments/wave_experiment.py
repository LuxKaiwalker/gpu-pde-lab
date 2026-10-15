"""2D wave equation experiment: propagation, damping, click disturbances,
and a CFL-based stability explorer analogous to the heat experiment."""

from __future__ import annotations

import time

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.wave import WaveEquation
from ..ui.controls import ControlPanel
from .base import EditSpec, LiveExperiment

EXPLANATION = r"""
### What PDE are we solving?

The 2D (damped) wave equation, e.g. a vibrating membrane:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t} $$

### How is it discretized?

Space: the same 5-point Laplacian as the heat equation. Time: central
differences (leapfrog), since this is second-order in time:

$$ u^{n+1} = \frac{2u^n - u^{n-1} + \Delta t^2\left(c^2\nabla^2 u^n\right) + \frac{\gamma\Delta t}{2}u^{n-1}}{1+\frac{\gamma \Delta t}{2}} $$

### What does one grid cell compute?

The same 5-neighbor read as the heat equation, plus the cell's own value
from *two* previous time levels ($u^n$ and $u^{n-1}$) -- the wave equation
needs one extra array of "memory" because it is second-order in time.

### Why can it run in parallel?

Identical reasoning to the heat equation: every output cell only reads
already-computed values (two previous time levels of neighbors), so all
cells update independently within one timestep.

### What does the GPU actually do?

One kernel launch computes the Laplacian; a second (fused by PyTorch into
very few actual GPU kernels) combines it with the two previous time levels.
Still fundamentally a memory-bound stencil operation, like the heat
equation.

### What can go wrong? -- the CFL condition

For the wave equation, the relevant stability bound is the **CFL condition**:

$$ \Delta t \le \frac{h}{c\sqrt{2}} $$

Note this is linear in $h$ (not quadratic, as for the heat equation) --
doubling resolution only *halves* the maximum stable timestep for waves,
whereas it *quarters* it for the heat equation. Exceed this bound and the
membrane does not just wobble: the amplitude grows without bound within a
few dozen steps, visible as it departing from the display's color range.
"""


class WaveExperiment(LiveExperiment):
    title = "Wave Equation"

    def __init__(self):
        self.n = 256
        self.c = 1.0
        self.damping = 0.0
        self.dt_fraction = 0.9
        self.allow_unstable = False
        self.boundary = BoundaryKind.DIRICHLET
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
        self.pde = WaveEquation(grid, backend, self.boundary, c=self.c, damping=self.damping)
        self.pde.reset(ic=self.ic, sigma=0.04)

    def reset(self) -> None:
        self._build_pde()

    def dt_max(self) -> float:
        return self.pde.dt_max()

    def _current_dt(self) -> float:
        return self.dt_fraction * self.pde.dt_max()

    def step(self, dt: float) -> None:
        self.pde.c = self.c
        self.pde.damping = self.damping
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
        clip = self.boundary != BoundaryKind.PERIODIC
        return {
            "u0": EditSpec(key="u0", label="Initial displacement u(x,y,0)", clip_to_interior=clip),
            "v0": EditSpec(key="v0", label="Initial velocity du/dt(x,y,0)", clip_to_interior=clip),
        }

    def edit_field_numpy(self, key: str) -> np.ndarray:
        if key == "u0":
            return self.pde.primary_field_numpy()
        if key == "v0":
            # No live velocity buffer to seed from (leapfrog has no explicit
            # velocity state) -- zero is the correct "nothing drawn yet" default.
            return np.zeros((self.pde.grid.nx, self.pde.grid.ny), dtype=np.float64)
        raise NotImplementedError(f"WaveExperiment has no editable field {key!r}")

    def apply_edited_fields(self, fields: dict[str, np.ndarray]) -> None:
        v0 = fields.get("v0")
        self.pde.reset_from_arrays(fields["u0"], v0 if v0 is not None and np.any(v0) else None)

    def editor_grid(self):
        return self.pde.grid

    def supports_click_injection(self) -> bool:
        return True

    def on_click(self, x_frac: float, y_frac: float) -> None:
        cx, cy = x_frac * self.pde.grid.lx, y_frac * self.pde.grid.ly
        self.pde.disturb(cx, cy, amplitude=1.0, sigma=0.03)

    def stats(self) -> dict:
        dt_max = self.pde.dt_max()
        dt_used = self._last_dt_used or self._current_dt()
        ratio = dt_used / dt_max if dt_max > 0 else float("inf")
        stability = "STABLE" if ratio <= 1.0 + 1e-9 else "UNSTABLE"
        field = self.field_numpy()
        if not bool(np.all(np.isfinite(field))):
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
            "dt_max (CFL)": f"{dt_max:.3e}",
            "dt / dt_max": f"{ratio:.2f}x",
            "Stability": stability,
            "Max amplitude": f"{self.pde.max_amplitude():.3e}",
            "FPS": f"{fps:.1f}",
            "GPU memory": f"{mem_mb/1e6:.2f} MB" if mem_mb is not None else "n/a",
        }

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_c(v):
            self.c = v

        panel.add_slider("c (wave speed)", 0.05, 3.0, self.c, step=0.01, decimals=3, on_change=set_c)

        def set_damping(v):
            self.damping = v

        panel.add_slider("damping (gamma)", 0.0, 10.0, self.damping, step=0.01, decimals=3,
                          on_change=set_damping)

        def set_dt_fraction(v):
            self.dt_fraction = v

        panel.add_slider("dt (fraction of CFL dt_max)", 0.01, 3.0, self.dt_fraction, step=0.01,
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

        panel.add_combo("Initial condition", ["gaussian", "point_source", "multiple_sources", "zeros"],
                         current=self.ic, on_change=set_ic)

        backend_opts = ["GPU", "CPU"] if cuda_is_available() else ["CPU"]

        def set_backend(text):
            self.backend_name = text
            on_change()

        panel.add_combo("Backend", backend_opts, current=self.backend_name, on_change=set_backend)

        return panel
