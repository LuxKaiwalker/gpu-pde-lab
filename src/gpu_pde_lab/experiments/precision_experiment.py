"""Precision experiment: fp16 vs fp32 vs fp64 -- runtime, memory, error, stability."""

from __future__ import annotations

import numpy as np

from ..benchmarking.timer import timed
from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from ..pdes.heat import HeatEquation
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### What's being compared

The same heat-equation run at fp16, fp32, and fp64, all on the GPU. GPU
hardware genuinely has different throughput for different precisions (this
RTX 2080 Ti's Tensor Cores accelerate fp16 matmuls heavily, though a plain
stencil update is mostly bandwidth-bound rather than compute-bound, so the
speedup from fp16 here mostly comes from moving half as many bytes per
array, not from specialized math units).

### The catch

**Lower precision is not automatically "fine" for PDE solvers.** fp16 has
only ~3 decimal digits of precision and a much narrower representable
range. For an explicit diffusion scheme running many timesteps, fp16
rounding error accumulates, and the *same* dt that is safely stable in
fp64 can show visibly larger error -- or, at extreme parameter choices,
numerical artifacts -- in fp16. This experiment reports runtime, memory,
and actual error *against an fp64 reference run with identical dt/steps*,
so you can see the real tradeoff rather than assuming "lower precision =
proportionally lower quality" or "free speed with no downside".
"""


class PrecisionExperiment(BenchmarkExperiment):
    title = "Precision Comparison (fp16 / fp32 / fp64)"

    def __init__(self):
        self.n = 512
        self.n_steps = 200
        self.alpha = 0.1

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def _run_one(self, precision: str, n_steps: int, dt: float):
        backend = Backend(BackendSpec("torch", "cuda" if cuda_is_available() else "cpu", precision))
        grid = Grid2D(nx=self.n, ny=self.n)
        heat = HeatEquation(grid, backend, BoundaryKind.NEUMANN, alpha=self.alpha)
        heat.reset(ic="gaussian")

        def run_steps():
            for _ in range(n_steps):
                heat.step(dt)

        t = timed(backend, run_steps, n_calls=1)
        mem = backend.max_memory_allocated_bytes()
        finite = bool(np.all(np.isfinite(heat.primary_field_numpy())))
        return heat.primary_field_numpy(), t.elapsed_seconds, mem, finite

    def run(self) -> dict:
        grid = Grid2D(nx=self.n, ny=self.n)
        ref_backend = Backend(BackendSpec("torch", "cpu", "fp64"))
        dt = HeatEquation(grid, ref_backend, BoundaryKind.NEUMANN, alpha=self.alpha).dt_max() * 0.5

        precisions = ["fp16", "fp32", "fp64"]
        fields, times, mems, finites = {}, {}, {}, {}
        for p in precisions:
            try:
                field, elapsed, mem, finite = self._run_one(p, self.n_steps, dt)
                fields[p] = field
                times[p] = elapsed
                mems[p] = mem
                finites[p] = finite
            except RuntimeError as e:
                fields[p] = None
                times[p] = float("nan")
                mems[p] = None
                finites[p] = False

        ref = fields.get("fp64")
        errors = {}
        for p in precisions:
            if fields[p] is not None and ref is not None:
                errors[p] = float(np.max(np.abs(fields[p] - ref)))
            else:
                errors[p] = float("nan")

        rows = ["| Precision | Time (ms) | Peak GPU mem (MB) | Max error vs fp64 | Finite? |",
                "|---|---|---|---|---|"]
        for p in precisions:
            mem_mb = mems[p] / 1e6 if mems[p] is not None else float("nan")
            rows.append(f"| {p} | {times[p]*1e3:.2f} | {mem_mb:.2f} | {errors[p]:.3e} | "
                       f"{'yes' if finites[p] else 'NO (diverged)'} |")

        return {
            "summary": f"Grid {self.n}x{self.n}, {self.n_steps} steps, dt={dt:.3e}\n\n" + "\n".join(rows),
            "bar_charts": [
                {"title": "Runtime by precision", "y_label": "seconds",
                 "categories": precisions, "values": [times[p] for p in precisions]},
                {"title": "Error vs fp64 reference (log scale)", "y_label": "max abs error",
                 "categories": [p for p in precisions if p != "fp64"],
                 "values": [errors[p] for p in precisions if p != "fp64"], "log_y": True},
            ],
        }

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_n(text):
            self.n = int(text)

        panel.add_combo("Grid resolution", ["128", "256", "512", "1024"], current=str(self.n),
                         on_change=set_n)

        def set_steps(v):
            self.n_steps = int(v)

        panel.add_slider("Number of steps", 10, 1000, self.n_steps, step=10, decimals=0,
                          on_change=set_steps)
        return panel
