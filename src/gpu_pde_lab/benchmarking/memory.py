"""Memory accounting: theoretical estimates and actual CUDA usage.

A single PDE field looks tiny ("just a 2D array of floats"), but a running
simulation typically keeps several full-size buffers alive at once (current
state, next state, temporaries for stencil results, visualization buffers),
and that multiplier is where "a seemingly small mathematical field" turns
into real memory pressure -- especially once a user pushes grid resolution
up or moves to 3D.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.backend import Backend
from ..core.grid import Grid2D


@dataclass
class MemoryEstimate:
    num_cells: int
    bytes_per_element: int
    num_buffers: int

    @property
    def bytes_per_buffer(self) -> int:
        return self.num_cells * self.bytes_per_element

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_buffer * self.num_buffers

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)


def estimate_memory(grid: Grid2D, backend: Backend, num_buffers: int) -> MemoryEstimate:
    return MemoryEstimate(
        num_cells=grid.num_cells,
        bytes_per_element=backend.bytes_per_element(),
        num_buffers=num_buffers,
    )


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


@dataclass
class CudaMemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 ** 2)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 ** 2)


def cuda_memory_snapshot(backend: Backend) -> CudaMemorySnapshot | None:
    alloc = backend.memory_allocated_bytes()
    reserved = backend.memory_reserved_bytes()
    if alloc is None or reserved is None:
        return None
    return CudaMemorySnapshot(allocated_bytes=alloc, reserved_bytes=reserved)
