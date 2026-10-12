"""CPU vs GPU benchmark: the same heat-equation step, same precision, run on
both backends across a sweep of grid sizes."""

from __future__ import annotations

from ..benchmarking.suite import run_cpu_gpu_sweep
from ..core.backend import cuda_is_available
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### What are we comparing?

The *identical* explicit heat-equation step -- same discretization, same
code path through `HeatEquation.step()` -- run once with a NumPy/Torch-CPU
backend and once with a Torch-CUDA backend, across increasing grid sizes.

### Why does the GPU sometimes lose at small sizes?

Every GPU operation has fixed overhead: launching a CUDA kernel, scheduling
it, synchronizing. At $128^2 = 16{,}384$ cells that overhead can exceed the
actual compute time, so the "massively parallel" GPU can be *slower* than a
single CPU core. This crossover is real and important: **parallelism has a
minimum problem size below which it doesn't pay off.**

### Why does the GPU win by 10-40x at large sizes?

At $2048^2$ or $4096^2$ there are millions of independent cells; the GPU's
thousands of concurrently-active threads amortize the fixed launch overhead
across far more work than a handful of CPU cores can match, even with
vectorized CPU code.

### Caution: CUDA timing correctness

CUDA kernel launches are asynchronous. Every measurement here brackets the
timed region with `torch.cuda.synchronize()` so the stopwatch reflects
actual GPU completion time, not just "time to enqueue work" -- skipping that
synchronization is the most common beginner mistake in GPU benchmarking.
"""


class CpuGpuExperiment(BenchmarkExperiment):
    title = "CPU vs GPU Benchmark"

    def __init__(self):
        self.sizes = [128, 256, 512, 1024, 2048]
        self.precision = "fp32"
        self.n_steps = 30

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def run(self, progress_cb=None) -> dict:
        if not cuda_is_available():
            return {"summary": "No CUDA device available -- cannot run CPU vs GPU comparison.",
                     "line_plots": []}

        result = run_cpu_gpu_sweep(self.sizes, precision=self.precision, n_steps=self.n_steps)
        cpu_times = [r.seconds_per_step * 1e3 for r in result.cpu]
        gpu_times = [r.seconds_per_step * 1e3 for r in result.gpu]
        speedups = result.speedups()

        lines = ["| N | CPU ms/step | GPU ms/step | Speedup | GPU BW (GB/s) | GPU mem (MB) |",
                 "|---|---|---|---|---|---|"]
        for n, c, g, s in zip(result.sizes, result.cpu, result.gpu, speedups):
            lines.append(f"| {n} | {c.seconds_per_step*1e3:.3f} | {g.seconds_per_step*1e3:.3f} | "
                         f"{s:.1f}x | {g.bandwidth_gb_s:.1f} | {g.gpu_memory_mb:.2f} |")

        return {
            "summary": "\n".join(lines),
            "line_plots": [
                {
                    "title": "Runtime per step vs grid size",
                    "x_label": "N (grid is N x N)", "y_label": "ms / step",
                    "log_x": True, "log_y": True,
                    "series": {"CPU": (result.sizes, cpu_times), "GPU": (result.sizes, gpu_times)},
                },
                {
                    "title": "GPU speedup vs grid size",
                    "x_label": "N (grid is N x N)", "y_label": "T_cpu / T_gpu",
                    "log_x": True, "log_y": False,
                    "series": {"Speedup": (result.sizes, speedups)},
                },
            ],
        }

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_precision(text):
            self.precision = text

        panel.add_combo("Precision", ["fp32", "fp64"], current=self.precision, on_change=set_precision)

        def set_max_size(text):
            all_sizes = [128, 256, 512, 1024, 2048, 4096]
            self.sizes = [s for s in all_sizes if s <= int(text)]

        panel.add_combo("Max grid size", ["512", "1024", "2048", "4096"], current="2048",
                         on_change=set_max_size)
        return panel
