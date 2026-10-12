"""Effective memory bandwidth for stencil workloads.

A 5-point Laplacian update touches 5 input values and writes 1 output value
per cell. Stencil PDE solvers are typically *memory-bandwidth-bound*, not
FLOP-bound: the GPU spends most of its time waiting on reads/writes to
memory rather than on the handful of adds/multiplies per cell. Effective
bandwidth is a standard way to quantify this:

    BW_effective = bytes_moved / runtime

For a single heat-equation step with a naive (non-fused) implementation, we
count: read u (1 array), read/write the laplacian temporary (1 array round
trip counted as read+write), and read/write the updated u (1 array round
trip). This is intentionally a simple, documented accounting -- real GPU
profilers (nsight, nvprof) measure actual DRAM traffic, which this estimate
approximates but does not replace.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BandwidthEstimate:
    bytes_moved: int
    seconds: float

    @property
    def gb_per_second(self) -> float:
        if self.seconds <= 0:
            return float("inf")
        return self.bytes_moved / self.seconds / 1e9


def heat_step_bytes_moved(num_cells: int, bytes_per_element: int) -> int:
    """Bytes moved per explicit heat-equation step: u^n -> laplacian -> u^{n+1}.

    Accounting (per cell): read u^n once for the laplacian (1x), write the
    laplacian temporary (1x) and read it back (1x) for the update, write the
    new u (1x). Total: 4 array-sized transfers.
    """
    arrays_touched = 4
    return num_cells * bytes_per_element * arrays_touched


def estimate_bandwidth(num_cells: int, bytes_per_element: int, seconds: float) -> BandwidthEstimate:
    moved = heat_step_bytes_moved(num_cells, bytes_per_element)
    return BandwidthEstimate(bytes_moved=moved, seconds=seconds)
