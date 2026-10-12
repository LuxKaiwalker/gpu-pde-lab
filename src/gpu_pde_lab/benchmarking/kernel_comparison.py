"""Five implementations of the same 5-point Laplacian, compared head to head.

This is the clearest illustration in the whole project of why "GPU vs CPU"
is really "massively parallel vs serial", not a property of hardware alone:

1. Pure Python double loop      -- one Python-level scalar op per cell, serial.
2. NumPy vectorized slicing     -- same math, but dispatched as bulk C loops
                                    over contiguous memory (SIMD-friendly).
3. PyTorch CPU tensor ops       -- same vectorized slicing, through PyTorch's
                                    CPU tensor backend (multi-threaded internally).
4. PyTorch CUDA tensor ops      -- same vectorized slicing, dispatched as a
                                    CUDA kernel launch running thousands of
                                    GPU threads concurrently.
5. (optional) custom CUDA kernel via torch.compile, included only if it
   demonstrates something the built-in vectorized op doesn't.

The Python-loop implementation is capped to small grids by the caller --
at anything beyond ~128x128 it would take minutes per step, which is itself
the point: interpreted, scalar, serial execution does not scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.backend import Backend, BackendSpec
from .timer import timed

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def laplacian_python_loop(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The naive, textbook implementation: one Python-level iteration per cell."""
    nx, ny = u.shape
    out = np.zeros_like(u)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            out[i, j] = (
                (u[i + 1, j] + u[i - 1, j] - 2.0 * u[i, j]) * inv_dx2
                + (u[i, j + 1] + u[i, j - 1] - 2.0 * u[i, j]) * inv_dy2
            )
    return out


def laplacian_numpy_vectorized(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    c = u[1:-1, 1:-1]
    out[1:-1, 1:-1] = (
        (u[2:, 1:-1] + u[:-2, 1:-1] - 2 * c) / (dx * dx)
        + (u[1:-1, 2:] + u[1:-1, :-2] - 2 * c) / (dy * dy)
    )
    return out


def laplacian_torch(u, dx: float, dy: float):
    """Same vectorized-slicing algorithm; works for both CPU and CUDA tensors
    depending on which device `u` lives on."""
    out = torch.zeros_like(u)
    c = u[1:-1, 1:-1]
    out[1:-1, 1:-1] = (
        (u[2:, 1:-1] + u[:-2, 1:-1] - 2 * c) / (dx * dx)
        + (u[1:-1, 2:] + u[1:-1, :-2] - 2 * c) / (dy * dy)
    )
    return out


@dataclass
class KernelComparisonResult:
    method: str
    n: int
    seconds_per_call: float
    cells_per_second: float
    max_abs_diff_from_reference: float


def run_kernel_comparison(n: int, include_python_loop: bool = True,
                           n_calls: int = 5) -> list[KernelComparisonResult]:
    """Benchmark all applicable implementations at grid size n x n.

    ``include_python_loop`` should be False for n beyond ~128, since the
    pure-Python loop becomes impractically slow -- that slowness is itself
    the demonstration, but we don't want it to block a UI event loop.
    """
    rng = np.random.default_rng(0)
    u_host = rng.standard_normal((n, n)).astype(np.float64)
    dx = dy = 1.0 / (n - 1)

    reference = laplacian_numpy_vectorized(u_host, dx, dy)
    results: list[KernelComparisonResult] = []

    if include_python_loop:
        t0 = timed(Backend(BackendSpec("numpy", "cpu", "fp64")),
                   lambda: laplacian_python_loop(u_host, dx, dy), n_calls=1)
        out = laplacian_python_loop(u_host, dx, dy)
        results.append(KernelComparisonResult(
            method="python_loop", n=n, seconds_per_call=t0.seconds_per_call,
            cells_per_second=n * n / t0.seconds_per_call,
            max_abs_diff_from_reference=float(np.max(np.abs(out - reference))),
        ))

    np_backend = Backend(BackendSpec("numpy", "cpu", "fp64"))
    t1 = timed(np_backend, lambda: laplacian_numpy_vectorized(u_host, dx, dy), n_calls=n_calls)
    results.append(KernelComparisonResult(
        method="numpy_vectorized", n=n, seconds_per_call=t1.seconds_per_call,
        cells_per_second=n * n / t1.seconds_per_call,
        max_abs_diff_from_reference=0.0,
    ))

    if torch is not None:
        u_cpu = torch.as_tensor(u_host)
        cpu_backend = Backend(BackendSpec("torch", "cpu", "fp64"))
        t2 = timed(cpu_backend, lambda: laplacian_torch(u_cpu, dx, dy), n_calls=n_calls)
        out_cpu = laplacian_torch(u_cpu, dx, dy).numpy()
        results.append(KernelComparisonResult(
            method="torch_cpu", n=n, seconds_per_call=t2.seconds_per_call,
            cells_per_second=n * n / t2.seconds_per_call,
            max_abs_diff_from_reference=float(np.max(np.abs(out_cpu - reference))),
        ))

        if torch.cuda.is_available():
            u_cuda = torch.as_tensor(u_host, device="cuda")
            gpu_backend = Backend(BackendSpec("torch", "cuda", "fp64"))
            # warmup to exclude one-time CUDA context / kernel compilation cost
            laplacian_torch(u_cuda, dx, dy)
            gpu_backend.synchronize()
            t3 = timed(gpu_backend, lambda: laplacian_torch(u_cuda, dx, dy), n_calls=n_calls)
            out_cuda = laplacian_torch(u_cuda, dx, dy).cpu().numpy()
            results.append(KernelComparisonResult(
                method="torch_cuda", n=n, seconds_per_call=t3.seconds_per_call,
                cells_per_second=n * n / t3.seconds_per_call,
                max_abs_diff_from_reference=float(np.max(np.abs(out_cuda - reference))),
            ))

    return results
