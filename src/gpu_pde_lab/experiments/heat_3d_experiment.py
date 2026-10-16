"""3D heat equation experiment: the smallest valid 3D vertical slice --
a real `Grid3D`, a real 7-point-stencil solver (`pdes/heat_3d.py`), and a
genuine 3-orthogonal-slice viewer (`ui/viz/slice_viewer_3d.py`), not a 2D
field extruded or broadcast along a third axis.

Deliberately smaller in scope than the 2D Heat experiment: no field editor,
no click injection, no precision picker -- just enough controls to run a
real, inspectable 3D simulation. Grid sizes are kept modest (16-48 per axis,
default 32 = 32,768 cells) since 3D grids grow as n^3, not n^2, and GPU
memory on the dev machine this was built on is already under pressure from
an unrelated process.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid3D
from ..pdes.heat_3d import HeatEquation3D
from ..ui.controls import ControlPanel
from .base import LiveExperiment

EXPLANATION = r"""
### What PDE are we solving?

The exact same 3D heat (diffusion) equation as the 2D experiment, just with a
third spatial dimension:

$$ \frac{\partial u}{\partial t} = \alpha \nabla^2 u, \qquad
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} $$

### How is it discretized? -- 7-point vs. 5-point

Space: the 3D finite-difference Laplacian reads **6 face neighbors** instead
of the 2D version's 4 edge neighbors (7 points total, including the center,
instead of 5):

$$ \nabla^2 u_{i,j,k} \approx \frac{u_{i+1,j,k}+u_{i-1,j,k}+u_{i,j+1,k}+u_{i,j-1,k}+u_{i,j,k+1}+u_{i,j,k-1}-6u_{i,j,k}}{h^2} $$

Time: the same explicit (forward) Euler as 2D:

$$ u^{n+1} = u^n + \Delta t\,\alpha \nabla^2 u^n $$

### What does one grid cell compute?

Cell $(i,j,k)$ reads its own value and its **6** direct face-neighbors (not
4) and writes one new value -- 50% more memory traffic per cell than the 2D
stencil, for the same handful of arithmetic operations. This is the direct
reason 3D stencil solvers are even more memory-bandwidth-bound than their 2D
counterparts.

### Why can it run in parallel?

Identical reasoning to every other explicit stencil in this lab: every
output cell depends only on the *previous* timestep's neighbors, so the
whole volume updates in one embarrassingly-parallel pass.

### What can go wrong? -- a tighter stability bound

$$ \Delta t \le \frac{1}{2\alpha\left(\frac{1}{dx^2}+\frac{1}{dy^2}+\frac{1}{dz^2}\right)} $$

For an isotropic grid ($dx=dy=dz=h$) this is $h^2/(6\alpha)$ -- tighter than
2D's $h^2/(4\alpha)$: each extra dimension adds another curvature term to the
same von Neumann amplification analysis, shrinking the largest stable
timestep for the same grid spacing.

### Memory and performance

A 3D field of resolution $n$ has $n^3$ cells, not $n^2$ -- doubling
resolution multiplies memory (and work per step) by 8x in 3D, versus 4x in
2D. That's why this experiment's grid-resolution choices top out at 48
(48³ ≈ 110k cells) rather than the 2D experiment's 2048 (2048² ≈ 4.2M cells).
"""


class Heat3DExperiment(LiveExperiment):
    title = "3D Heat Equation"

    def __init__(self):
        self.n = 32
        self.alpha = 0.12
        self.dt_fraction = 0.5
        self.allow_unstable = False
        self.boundary = BoundaryKind.NEUMANN
        self.ic = "gaussian"
        self.backend_name = "GPU" if cuda_is_available() else "CPU"
        self._last_dt_used = 0.0
        self._fps_timestamps: list[float] = []
        self._build_pde()

    def _make_backend(self) -> Backend:
        if self.backend_name == "GPU" and cuda_is_available():
            return Backend(BackendSpec("torch", "cuda", "fp32"))
        return Backend(BackendSpec("torch", "cpu", "fp32"))

    def _build_pde(self) -> None:
        backend = self._make_backend()
        grid = Grid3D(nx=self.n, ny=self.n, nz=self.n)
        self.pde = HeatEquation3D(grid, backend, self.boundary, alpha=self.alpha)
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
        """Legacy 2D-only display contract: a mid-Z slice, as a defensive
        fallback only. The real display path for this experiment is
        `volume_numpy()` via the 3D slice viewer."""
        volume = self.pde.primary_field_numpy()
        return volume[:, :, volume.shape[2] // 2]

    # -- 3D volume hooks --------------------------------------------------

    def is_3d(self) -> bool:
        return True

    def volume_numpy(self) -> np.ndarray:
        return self.pde.primary_field_numpy()

    def volume_grid(self):
        return self.pde.grid

    def stats(self) -> dict:
        dt_max = self.pde.dt_max()
        dt_used = self._last_dt_used or self._current_dt()
        ratio = dt_used / dt_max if dt_max > 0 else float("inf")
        stability = "STABLE" if ratio <= 1.0 + 1e-9 else "UNSTABLE"
        volume = self.volume_numpy()
        finite = bool(np.all(np.isfinite(volume)))
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
            "Grid": f"{self.n}^3 ({self.n**3:,} cells)",
            "Backend": self.pde.backend.spec.label(),
            "Device": self.pde.backend.device_name(),
            "dt": f"{dt_used:.3e}",
            "dt_max": f"{dt_max:.3e}",
            "dt / dt_max": f"{ratio:.2f}x",
            "Stability": stability,
            "FPS": f"{fps:.1f}",
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
                          decimals=3, on_change=set_dt_fraction,
                          tooltip="Timestep as a fraction of the stability bound dt_max "
                                  "(tighter in 3D than 2D -- see the explanation panel).")

        def set_unstable(checked):
            self.allow_unstable = checked

        panel.add_checkbox("Allow unstable dt (explore instability)", self.allow_unstable,
                            on_change=set_unstable,
                            tooltip="When off, dt is clamped to dt_max. When on, you can push "
                                    "past the stability bound and watch it diverge.")

        def set_grid(text):
            self.n = int(text)
            on_change()

        panel.add_combo("Grid resolution (n^3 cells)", ["16", "24", "32", "48"],
                         current=str(self.n), on_change=set_grid)

        def set_boundary(text):
            self.boundary = BoundaryKind(text.lower())
            on_change()

        panel.add_combo("Boundary condition", ["dirichlet", "neumann", "periodic"],
                         current=self.boundary.value, on_change=set_boundary)

        def set_ic(text):
            self.ic = text
            on_change()

        panel.add_combo("Initial condition", ["gaussian", "zeros"], current=self.ic,
                         on_change=set_ic)

        backend_opts = ["GPU", "CPU"] if cuda_is_available() else ["CPU"]

        def set_backend(text):
            self.backend_name = text
            on_change()

        panel.add_combo("Backend", backend_opts, current=self.backend_name, on_change=set_backend)

        return panel
