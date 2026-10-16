"""Resolution/convergence experiment: discretization error vs h and dt,
against a known analytical heat-equation solution."""

from __future__ import annotations

from ..benchmarking.convergence import spatial_convergence, temporal_convergence
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### The point of this experiment

$$ \boxed{\text{GPU acceleration does not make an inaccurate numerical method accurate.}} $$

A GPU can run a bad discretization extremely fast -- it will still be
wrong. This experiment measures the *actual numerical error* of the heat
solver against a known closed-form solution, independent of which backend
it runs on.

### The test problem

On $[0,L]^2$ with Dirichlet boundaries and $u_0 = \sin(\pi x/L)\sin(\pi y/L)$,
the exact solution is a single decaying Fourier mode:

$$ u(x,y,t) = e^{-\alpha\pi^2(2/L^2) t}\sin(\pi x/L)\sin(\pi y/L) $$

**Spatial convergence**: dt is fixed very small (negligible time error); h is
varied. The 5-point stencil is 2nd-order accurate in space, so error should
shrink by ~4x each time h is halved (slope 2 on a log-log plot).

**Temporal convergence**: a fine, fixed grid is used; dt is varied (staying
under the stability bound). Forward Euler is 1st-order accurate in time, so
error should shrink by ~2x each time dt is halved (slope 1) -- *until* it
reaches the fixed spatial-discretization floor set by that grid's h, below
which further shrinking dt no longer improves total error. That floor is
itself an important lesson: **total error is limited by whichever source of
error (space or time) is currently dominant.**
"""


class ConvergenceExperiment(BenchmarkExperiment):
    title = "Resolution / Convergence Study"

    def __init__(self):
        self.mode = "spatial"
        self.alpha = 0.1

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def run(self) -> dict:
        if self.mode == "spatial":
            sizes = [16, 32, 64, 128, 256]
            result = spatial_convergence(sizes, alpha=self.alpha, t_final=0.01)
            x_label = "h (grid spacing)"
        else:
            dt_max_at_512 = (1.0 / 511) ** 2 / (4 * self.alpha)
            dts = [dt_max_at_512 * f for f in (0.8, 0.4, 0.2, 0.1)]
            result = temporal_convergence(dts, n=512, alpha=self.alpha, t_final=0.02)
            x_label = "dt"

        xs = [p.h_or_dt for p in result.points]
        errs = [p.error_linf for p in result.points]
        orders_str = ", ".join(f"{o:.2f}" for o in result.observed_orders)

        expected = "2 (space)" if self.mode == "spatial" else "1 (time, until floor)"
        rows = [f"| {x_label} | L-inf error |", "|---|---|"]
        for x, e in zip(xs, errs):
            rows.append(f"| {x:.3e} | {e:.3e} |")

        summary = (
            "\n".join(rows)
            + f"\n\nObserved convergence orders between consecutive points: [{orders_str}]"
            + f"\n\nExpected theoretical order: **{expected}**"
        )

        return {
            "summary": summary,
            "line_plots": [
                {
                    "title": f"{self.mode.title()} convergence: error vs {x_label}",
                    "x_label": x_label, "y_label": "L-inf error",
                    "log_x": True, "log_y": True,
                    "series": {"error": (xs, errs)},
                },
            ],
        }

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_mode(text):
            self.mode = text

        panel.add_combo("Mode", ["spatial", "temporal"], current=self.mode, on_change=set_mode)
        return panel
