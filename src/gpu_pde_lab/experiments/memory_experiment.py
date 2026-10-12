"""GPU memory experiment: how field size and buffer count determine memory use."""

from __future__ import annotations

from ..benchmarking.memory import cuda_memory_snapshot, estimate_memory, format_bytes
from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.grid import Grid2D
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### Why memory adds up quickly

A single scalar field at $N \times N$ resolution in fp32 needs
$4N^2$ bytes. That sounds tiny -- but a real simulation never keeps only
one array alive. A minimal explicit stencil solver typically needs:

* `u` -- the current field
* `u_next` (or a Laplacian temporary) -- the next/intermediate field
* a visualization buffer (often a downsampled or color-mapped copy)
* for the wave equation, an additional `u_prev`
* for CG/Poisson, several more vectors ($r$, $p$, $Ap$)

Each of these is a *full-size* array. At $4096^2$ in fp32, **one** such
buffer is 64 MB; five of them is 320 MB -- and that's still only 2D. The
same accounting in 3D ($N^3$ cells) grows the per-buffer cost by another
factor of $N$, which is why naive 3D stencil codes exhaust GPU memory far
sooner than people expect.

### What this experiment shows

The table below estimates memory for a configurable number of buffers at
several grid sizes and precisions, and -- when running on CUDA -- also
reports the *actual* memory PyTorch has allocated/reserved after allocating
that many real buffers, so you can compare the theoretical estimate against
ground truth (PyTorch's caching allocator typically reserves somewhat more
than the raw byte count due to allocation granularity).
"""


class MemoryExperiment(BenchmarkExperiment):
    title = "GPU Memory"

    def __init__(self):
        self.n = 2048
        self.num_buffers = 4
        self.precision = "fp32"

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def run(self, progress_cb=None) -> dict:
        sizes = [256, 512, 1024, 2048, 4096]
        backend_cpu = Backend(BackendSpec("torch", "cpu", self.precision if self.precision != "fp16" else "fp32"))

        rows = ["| N | cells | bytes/buffer | total (N buffers) |", "|---|---|---|---|"]
        for n in sizes:
            grid = Grid2D(nx=n, ny=n)
            est = estimate_memory(grid, backend_cpu, self.num_buffers)
            rows.append(f"| {n} | {grid.num_cells:,} | {format_bytes(est.bytes_per_buffer)} | "
                       f"{format_bytes(est.total_bytes)} |")

        summary = "\n".join(rows)

        if cuda_is_available():
            gpu_backend = Backend(BackendSpec("torch", "cuda", self.precision))
            grid = Grid2D(nx=self.n, ny=self.n)
            gpu_backend.reset_peak_memory_stats()
            buffers = [gpu_backend.zeros(grid.shape) for _ in range(self.num_buffers)]
            snap = cuda_memory_snapshot(gpu_backend)
            est = estimate_memory(grid, gpu_backend, self.num_buffers)
            summary += (
                f"\n\n**Live CUDA allocation test** at N={self.n}, {self.num_buffers} buffers, "
                f"{self.precision}:\n\n"
                f"- Theoretical estimate: {format_bytes(est.total_bytes)}\n"
                f"- Actual `torch.cuda.memory_allocated()`: {snap.allocated_mb:.2f} MB\n"
                f"- Actual `torch.cuda.memory_reserved()`: {snap.reserved_mb:.2f} MB "
                f"(the caching allocator reserves more than strictly needed)\n"
            )
            del buffers

        return {"summary": summary, "line_plots": [
            {
                "title": "Estimated memory vs grid size",
                "x_label": "N (grid is N x N)", "y_label": "MB",
                "log_x": True, "log_y": True,
                "series": {
                    f"{self.num_buffers} buffers": (
                        sizes,
                        [estimate_memory(Grid2D(nx=n, ny=n), backend_cpu, self.num_buffers).total_mb for n in sizes],
                    ),
                },
            },
        ]}

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_n(text):
            self.n = int(text)

        panel.add_combo("Grid resolution (live CUDA test)", ["512", "1024", "2048", "4096"],
                         current=str(self.n), on_change=set_n)

        def set_buffers(v):
            self.num_buffers = int(v)

        panel.add_slider("Number of buffers", 1, 10, self.num_buffers, step=1, decimals=0,
                          on_change=set_buffers)

        def set_precision(text):
            self.precision = text

        panel.add_combo("Precision", ["fp16", "fp32", "fp64"], current=self.precision,
                         on_change=set_precision)
        return panel
