"""Benchmark sweeps: CPU vs GPU runtime across grid sizes, and the
host<->device transfer breakdown.

These are plain functions returning plain dataclasses/dicts so they can be
driven identically from a headless script (``scripts/run_benchmark.py``) or
from the interactive UI's performance dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.grid import Grid2D
from .bandwidth import estimate_bandwidth
from .timer import Stopwatch, timed

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class GridBenchmarkResult:
    n: int
    backend_label: str
    seconds_per_step: float
    steps_per_second: float
    cells_per_second: float
    bandwidth_gb_s: float
    gpu_memory_mb: float | None


def benchmark_heat_step(grid: Grid2D, backend: Backend, boundary: BoundaryKind, alpha: float,
                         n_steps: int = 50, warmup: int = 5) -> GridBenchmarkResult:
    from ..pdes.heat import HeatEquation

    heat = HeatEquation(grid, backend, boundary, alpha=alpha)
    dt = heat.dt_max() * 0.5

    for _ in range(warmup):
        heat.step(dt)

    backend.reset_peak_memory_stats()
    result = timed(backend, lambda: heat.step(dt), n_calls=n_steps)

    spc = result.seconds_per_call
    bw = estimate_bandwidth(grid.num_cells, backend.bytes_per_element(), spc)
    mem = backend.max_memory_allocated_bytes()
    return GridBenchmarkResult(
        n=grid.nx,
        backend_label=backend.spec.label(),
        seconds_per_step=spc,
        steps_per_second=1.0 / spc if spc > 0 else float("inf"),
        cells_per_second=grid.num_cells / spc if spc > 0 else float("inf"),
        bandwidth_gb_s=bw.gb_per_second,
        gpu_memory_mb=(mem / (1024 ** 2)) if mem is not None else None,
    )


@dataclass
class CpuGpuSweepResult:
    sizes: list[int]
    cpu: list[GridBenchmarkResult]
    gpu: list[GridBenchmarkResult] | None

    def speedups(self) -> list[float]:
        if not self.gpu:
            return []
        return [c.seconds_per_step / g.seconds_per_step for c, g in zip(self.cpu, self.gpu)]


def run_cpu_gpu_sweep(sizes: list[int], precision: str = "fp32", alpha: float = 0.1,
                       n_steps: int = 50) -> CpuGpuSweepResult:
    cpu_results = []
    gpu_results = []
    has_cuda = cuda_is_available()

    for n in sizes:
        grid = Grid2D(nx=n, ny=n)
        cpu_backend = Backend(BackendSpec("torch", "cpu", precision if precision != "fp16" else "fp32"))
        cpu_results.append(benchmark_heat_step(grid, cpu_backend, BoundaryKind.NEUMANN, alpha, n_steps=n_steps))

        if has_cuda:
            gpu_backend = Backend(BackendSpec("torch", "cuda", precision))
            gpu_results.append(benchmark_heat_step(grid, gpu_backend, BoundaryKind.NEUMANN, alpha, n_steps=n_steps))

    return CpuGpuSweepResult(sizes=sizes, cpu=cpu_results, gpu=gpu_results if has_cuda else None)


@dataclass
class TransferBenchmarkResult:
    n: int
    h2d_seconds: float
    compute_seconds: float
    d2h_seconds: float
    total_roundtrip_seconds: float
    gpu_resident_seconds: float  # same compute, but staying on GPU across all steps

    @property
    def overhead_fraction(self) -> float:
        total = self.total_roundtrip_seconds
        return (self.h2d_seconds + self.d2h_seconds) / total if total > 0 else 0.0


def benchmark_transfer(n: int, n_steps: int = 20, alpha: float = 0.1) -> TransferBenchmarkResult:
    """Simulate a (deliberately bad) pattern of transferring the field back and
    forth across PCIe every step, versus keeping it resident on the GPU."""
    from ..pdes.heat import HeatEquation

    grid = Grid2D(nx=n, ny=n)
    cpu_backend = Backend(BackendSpec("torch", "cpu", "fp32"))
    gpu_backend = Backend(BackendSpec("torch", "cuda", "fp32"))

    host_field = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)

    sw = Stopwatch(gpu_backend)
    sw.start()
    gpu_array = gpu_backend.array(host_field)
    sw.lap("h2d")

    heat_resident = HeatEquation(grid, gpu_backend, BoundaryKind.NEUMANN, alpha=alpha)
    heat_resident.u = gpu_array
    dt = heat_resident.dt_max() * 0.5

    h2d_total = sw.laps[-1].seconds
    compute_total = 0.0
    d2h_total = 0.0

    for _step in range(n_steps):
        t_sw = Stopwatch(gpu_backend)
        t_sw.start()
        heat_resident.step(dt)
        t_sw.lap("compute")
        compute_total += t_sw.laps[-1].seconds

        t_sw.start()
        host_copy = gpu_backend.to_numpy(heat_resident.u)
        t_sw.lap("d2h")
        d2h_total += t_sw.laps[-1].seconds

        t_sw.start()
        heat_resident.u = gpu_backend.array(host_copy)
        t_sw.lap("h2d_again")
        h2d_total += t_sw.laps[-1].seconds

    # Compare against the same number of steps fully resident on GPU (no
    # per-step transfer at all) -- only the compute-only time matters.
    heat_gpu_only = HeatEquation(grid, gpu_backend, BoundaryKind.NEUMANN, alpha=alpha)
    resident_timing = timed(gpu_backend, lambda: heat_gpu_only.step(dt), n_calls=n_steps)

    total_roundtrip = h2d_total + compute_total + d2h_total
    return TransferBenchmarkResult(
        n=n,
        h2d_seconds=h2d_total,
        compute_seconds=compute_total,
        d2h_seconds=d2h_total,
        total_roundtrip_seconds=total_roundtrip,
        gpu_resident_seconds=resident_timing.elapsed_seconds,
    )
