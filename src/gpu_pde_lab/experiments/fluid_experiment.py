"""2D incompressible fluid experiment: dye/velocity advection driven by
mouse clicks, built entirely from primitives used elsewhere in the lab
(the 5-point Laplacian, Jacobi relaxation, bilinear sampling)."""

from __future__ import annotations

import math
import random

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.fluid import StableFluids2D
from ..ui.controls import ControlPanel
from .base import EditSpec, LiveExperiment

EXPLANATION = r"""
### What are we solving?

Incompressible Navier-Stokes in 2D:

$$ \frac{\partial \mathbf u}{\partial t} = -(\mathbf u\cdot\nabla)\mathbf u + \nu\nabla^2\mathbf u - \nabla p, \qquad \nabla\cdot\mathbf u = 0 $$

### How is it discretized? (operator splitting)

Each timestep is split into independent stages, each reusing a building
block from earlier experiments:

1. **Diffuse** -- implicit viscosity via Jacobi relaxation (same iteration
   as the Poisson experiment, solving $(I-\nu\Delta t\nabla^2)u=u_0$).
2. **Project** -- solve $\nabla^2 p = \nabla\cdot\mathbf u$ by Jacobi
   relaxation, then subtract $\nabla p$ from velocity. This is exactly a
   Poisson solve with a different right-hand side.
3. **Advect** -- semi-Lagrangian backtracing: each cell's new value is
   found by bilinearly sampling the *old* field at the position one
   timestep upstream along the velocity. Unconditionally stable for any
   $\Delta t$ (no CFL restriction), unlike the explicit heat/wave schemes.
4. **Project again** (advection reintroduces divergence), then advect dye.

### Why can it run in parallel?

Diffusion and projection are both the same 5-point-stencil Jacobi
relaxation used in the Poisson experiment -- every cell update in one
Jacobi sweep only reads the *previous* sweep's neighbor values, so each
sweep is embarrassingly parallel. Semi-Lagrangian advection is also
trivially parallel: every output cell independently traces its own
backward path and samples -- no cell depends on another cell's *new*
value.

### What can go wrong?

The pressure projection only runs a fixed number of Jacobi iterations per
step (for interactivity) rather than to full convergence, so the velocity
field is only *approximately* divergence-free -- watch the "divergence
norm" stat: it drops sharply after each projection but is never exactly
zero. This is the same accuracy-vs-speed tradeoff seen in the Poisson
experiment, now hidden inside a real-time loop.
"""


class FluidExperiment(LiveExperiment):
    title = "2D Fluid Simulation"

    def __init__(self):
        self.n = 192
        self.viscosity = 0.0001
        self.dt = 0.016
        self.boundary = BoundaryKind.NEUMANN
        self.backend_name = "GPU" if cuda_is_available() else "CPU"
        self.display_mode = "dye"
        self._build()

    def _make_backend(self) -> Backend:
        if self.backend_name == "GPU" and cuda_is_available():
            return Backend(BackendSpec("torch", "cuda", "fp32"))
        return Backend(BackendSpec("torch", "cpu", "fp32"))

    def _build(self) -> None:
        backend = self._make_backend()
        grid = Grid2D(nx=self.n, ny=self.n)
        self.fluid = StableFluids2D(grid, backend, self.boundary, viscosity=self.viscosity,
                                     diffusion_iters=15, pressure_iters=25)
        self._add_swirl_sources()

    def _add_swirl_sources(self) -> None:
        self.fluid.add_source(0.3, 0.5, fx=1.5, fy=0.3, dye_amount=1.0, radius=0.06)
        self.fluid.add_source(0.7, 0.5, fx=-1.5, fy=-0.3, dye_amount=1.0, radius=0.06)

    def reset(self) -> None:
        self._build()

    def step(self, dt: float) -> None:
        self.fluid.viscosity = self.viscosity
        self.fluid.step(self.dt)

    def field_numpy(self):
        if self.display_mode == "velocity":
            return self.fluid.velocity_magnitude_numpy()
        return self.fluid.dye_numpy()

    # -- field editor hooks ----------------------------------------------

    def editable_fields(self) -> dict[str, EditSpec]:
        clip = self.boundary != BoundaryKind.PERIODIC
        return {
            "u0": EditSpec(key="u0", label="Initial velocity u-component", clip_to_interior=clip),
            "v0": EditSpec(key="v0", label="Initial velocity v-component", clip_to_interior=clip),
            "dye0": EditSpec(key="dye0", label="Initial dye concentration", clip_to_interior=clip),
        }

    def edit_field_numpy(self, key: str) -> np.ndarray:
        if key == "u0":
            return self.fluid.backend.to_numpy(self.fluid.u)
        if key == "v0":
            return self.fluid.backend.to_numpy(self.fluid.v)
        if key == "dye0":
            return self.fluid.backend.to_numpy(self.fluid.dye)
        raise NotImplementedError(f"FluidExperiment has no editable field {key!r}")

    def apply_edited_fields(self, fields: dict[str, np.ndarray]) -> None:
        self.fluid.set_fields(u=fields.get("u0"), v=fields.get("v0"), dye=fields.get("dye0"))

    def editor_grid(self):
        return self.fluid.grid

    def supports_click_injection(self) -> bool:
        return True

    def on_click(self, x_frac: float, y_frac: float) -> None:
        cx, cy = x_frac * self.fluid.grid.lx, y_frac * self.fluid.grid.ly
        angle = random.uniform(0, 2 * math.pi)
        fx, fy = 2.5 * math.cos(angle), 2.5 * math.sin(angle)
        self.fluid.add_source(cx, cy, fx=fx, fy=fy, dye_amount=1.0, radius=0.05)

    def stats(self) -> dict:
        mem = self.fluid.backend.memory_allocated_bytes()
        return {
            "Sim time": f"{self.fluid.time:.3f}",
            "Step": str(self.fluid.step_count),
            "Grid": f"{self.n} x {self.n} ({self.n*self.n:,} cells)",
            "Backend": self.fluid.backend.spec.label(),
            "dt": f"{self.dt:.4f}",
            "Viscosity": f"{self.viscosity:.2e}",
            "Divergence norm": f"{self.fluid.divergence_norm():.3e}",
            "Max |velocity|": f"{self.fluid.velocity_magnitude_numpy().max():.3f}",
            "GPU memory": f"{mem/1e6:.2f} MB" if mem is not None else "n/a",
        }

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_viscosity(v):
            self.viscosity = v

        panel.add_slider("viscosity", 0.0, 0.01, self.viscosity, step=0.0001, decimals=5,
                          on_change=set_viscosity)

        def set_dt(v):
            self.dt = v

        panel.add_slider("dt", 0.001, 0.05, self.dt, step=0.001, decimals=4, on_change=set_dt)

        def set_display(text):
            self.display_mode = text

        panel.add_combo("Display", ["dye", "velocity"], current=self.display_mode,
                         on_change=set_display)

        def set_grid(text):
            self.n = int(text)
            on_change()

        panel.add_combo("Grid resolution", ["96", "128", "192", "256", "384"], current=str(self.n),
                         on_change=set_grid)

        def set_boundary(text):
            self.boundary = BoundaryKind(text.lower())
            on_change()

        panel.add_combo("Boundary", ["neumann", "periodic"], current=self.boundary.value,
                         on_change=set_boundary)

        backend_opts = ["GPU", "CPU"] if cuda_is_available() else ["CPU"]

        def set_backend(text):
            self.backend_name = text
            on_change()

        panel.add_combo("Backend", backend_opts, current=self.backend_name, on_change=set_backend)

        return panel
