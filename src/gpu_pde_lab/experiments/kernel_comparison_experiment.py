"""Python loop vs NumPy vs Torch-CPU vs Torch-CUDA: the same Laplacian,
five ways, to make "why GPUs are fast" concrete rather than assumed."""

from __future__ import annotations

from ..benchmarking.kernel_comparison import run_kernel_comparison
from ..core.backend import cuda_is_available
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### Same math, five implementations

All four (or five, with CUDA) implementations below compute the *exact
same* 5-point Laplacian on the *exact same* random input. Only the
execution strategy changes:

1. **Python loop**: a literal translation of the math into nested `for`
   loops -- one Python bytecode interpretation per scalar operation, per
   cell. This is what "an algorithm" looks like before any thought is given
   to hardware.
2. **NumPy vectorized**: the same math expressed as whole-array slice
   arithmetic. NumPy dispatches this to compiled C loops operating on
   contiguous memory -- no Python-level iteration at all.
3. **PyTorch CPU**: the same vectorized expression, routed through
   PyTorch's own (multi-threaded) CPU tensor backend.
4. **PyTorch CUDA**: the same vectorized expression, but the underlying
   arrays live in GPU memory, so PyTorch dispatches it as a single CUDA
   kernel launch executing thousands of threads concurrently.

### What to expect

The Python loop is typically **100-1000x slower** than any vectorized
version even at modest grid sizes -- not because Python is "bad at math",
but because each of its scalar operations carries interpreter overhead that
vectorized/compiled code pays once per *array*, not once per *element*.
GPU vs vectorized-CPU only pulls ahead once the grid is large enough to
keep thousands of GPU threads busy; at small grids, kernel-launch overhead
can make the GPU implementation the *slowest* of the bunch.
"""


class KernelComparisonExperiment(BenchmarkExperiment):
    title = "Kernel / Vectorization Comparison"

    def __init__(self):
        self.n = 64
        self.include_loop = True

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def run(self, progress_cb=None) -> dict:
        results = run_kernel_comparison(self.n, include_python_loop=self.include_loop, n_calls=5)
        rows = ["| Method | ms/call | Mcells/s | max diff vs reference |", "|---|---|---|---|"]
        for r in results:
            rows.append(f"| {r.method} | {r.seconds_per_call*1e3:.4f} | "
                       f"{r.cells_per_second/1e6:.2f} | {r.max_abs_diff_from_reference:.1e} |")
        return {
            "summary": f"Grid: {self.n} x {self.n}\n\n" + "\n".join(rows),
            "bar_charts": [
                {
                    "title": "Throughput by implementation (log scale)",
                    "y_label": "Mcells/s",
                    "categories": [r.method for r in results],
                    "values": [r.cells_per_second / 1e6 for r in results],
                    "log_y": True,
                },
            ],
        }

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_n(text):
            n = int(text)
            self.n = n
            self.include_loop = n <= 128

        panel.add_combo("Grid resolution", ["32", "64", "128", "256", "512", "1024"],
                         current=str(self.n), on_change=set_n)
        return panel
