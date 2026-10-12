"""Host <-> device transfer experiment: shows why shuttling data across PCIe
every step is costly compared to keeping the simulation GPU-resident."""

from __future__ import annotations

from ..benchmarking.suite import benchmark_transfer
from ..core.backend import cuda_is_available
from ..ui.controls import ControlPanel
from .base import BenchmarkExperiment

EXPLANATION = r"""
### The pipeline being measured

$$ \text{CPU memory} \to \text{PCIe transfer} \to \text{GPU memory} \to \text{GPU compute} \to \text{GPU} \to \text{CPU} $$

This experiment runs a number of heat-equation steps two ways:

1. **Round-trip per step**: copy the field to the GPU, take one step, copy
   the result back to the CPU, repeat -- as if something on the CPU needed
   the field every single step (e.g. naive logging, or mixing GPU compute
   with CPU-side post-processing without thinking about it).
2. **GPU-resident**: the field never leaves the GPU across all steps; data
   only needs to come back at the very end (or not at all, if you're only
   visualizing a downsampled or periodic snapshot).

### Why this matters

PCIe bandwidth (tens of GB/s) is far below GPU memory bandwidth (hundreds of
GB/s to >1 TB/s on modern cards), and each transfer call also carries fixed
latency. For small-to-medium stencil problems, the *compute itself* can take
microseconds while a round-trip transfer takes milliseconds -- so the
transfer, not the math, dominates wall time. The bar chart below usually
shows host<->device transfer time *dwarfing* compute time at moderate grid
sizes, which is exactly why real GPU simulation codes keep state resident on
the device and only transfer when a human actually needs to look at it
(e.g. once per rendered frame, not once per physics step).
"""


class TransferExperiment(BenchmarkExperiment):
    title = "Host <-> Device Transfer"

    def __init__(self):
        self.n = 1024
        self.n_steps = 20

    def explanation_markdown(self) -> str:
        return EXPLANATION

    def run(self, progress_cb=None) -> dict:
        if not cuda_is_available():
            return {"summary": "No CUDA device available -- cannot run transfer experiment.",
                     "line_plots": [], "bar_charts": []}

        result = benchmark_transfer(self.n, n_steps=self.n_steps)
        summary = (
            f"Grid: {self.n} x {self.n}, {self.n_steps} steps\n\n"
            f"| Stage | Total time (ms) |\n|---|---|\n"
            f"| Host -> Device (H2D) | {result.h2d_seconds*1e3:.3f} |\n"
            f"| GPU compute | {result.compute_seconds*1e3:.3f} |\n"
            f"| Device -> Host (D2H) | {result.d2h_seconds*1e3:.3f} |\n"
            f"| **Total (per-step round trip)** | **{result.total_roundtrip_seconds*1e3:.3f}** |\n"
            f"| GPU-resident (no per-step transfer) | {result.gpu_resident_seconds*1e3:.3f} |\n\n"
            f"Transfer overhead fraction of round-trip time: "
            f"**{result.overhead_fraction*100:.1f}%**\n\n"
            f"Round-trip is **{result.total_roundtrip_seconds/result.gpu_resident_seconds:.1f}x "
            f"slower** than keeping the simulation GPU-resident."
        )
        return {
            "summary": summary,
            "bar_charts": [
                {
                    "title": "Time breakdown: per-step round trip vs GPU-resident",
                    "y_label": "seconds (total over all steps)",
                    "categories": ["H2D", "Compute", "D2H", "GPU-resident (compute only)"],
                    "values": [result.h2d_seconds, result.compute_seconds, result.d2h_seconds,
                               result.gpu_resident_seconds],
                },
            ],
        }

    def build_controls(self, on_change):
        panel = ControlPanel()

        def set_n(text):
            self.n = int(text)

        panel.add_combo("Grid resolution", ["256", "512", "1024", "2048"], current=str(self.n),
                         on_change=set_n)
        return panel
