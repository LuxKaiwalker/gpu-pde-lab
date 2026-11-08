"""Backend abstraction: one array API, multiple execution engines.

The rest of the codebase (PDEs, solvers, experiments) is written against a
small array-API-like surface (``Backend``) instead of importing numpy/torch
directly. This is what lets the *identical* finite-difference code run on
CPU (NumPy or Torch-CPU) or GPU (Torch-CUDA) -- the numerical algorithm never
changes, only the execution engine underneath it.

This is deliberately a thin layer, not a full array-API shim: we only expose
the handful of operations the PDE solvers actually need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
    torch = None  # type: ignore[assignment]

DeviceKind = Literal["cpu", "cuda"]
EngineKind = Literal["numpy", "torch"]
PrecisionKind = Literal["fp16", "fp32", "fp64"]

_PRECISION_TO_NP = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}


def cuda_is_available() -> bool:
    """True if a CUDA device is present and the driver answers.

    Presence is not the same as usability -- see :func:`cuda_is_usable`. This
    is the right check for "should a CUDA code path exist at all"; it is the
    wrong check for "can I allocate on it right now".
    """
    return torch is not None and torch.cuda.is_available()


#: Floor for calling CUDA usable. A CUDA context costs a few hundred MB
#: before a single grid cell is allocated, and the benchmark sweep reaches
#: 4096x4096 (64 MB per fp32 buffer, several buffers live at once). Below
#: this we would start, then die partway through a sweep -- which is a worse
#: failure than declining up front.
MIN_CUDA_FREE_BYTES = 512 * 1024 * 1024

_cuda_usable_cache: tuple[int, bool] | None = None


def cuda_free_memory_bytes() -> int | None:
    """Free device memory in bytes, or None if that cannot be determined.

    Note this reports what the *driver* says is free, which accounts for
    every process on the card -- not just this one. A machine running
    something else GPU-heavy (an inference server, another training job)
    can leave a perfectly healthy device with nothing left to allocate.
    """
    if not cuda_is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        # mem_get_info has to initialize a CUDA context to answer, and that
        # initialization is itself an allocation that can fail when the card
        # is full. An unanswerable query means "assume nothing is free".
        return None


def cuda_is_usable(min_free_bytes: int = MIN_CUDA_FREE_BYTES) -> bool:
    """True if CUDA is present *and* has room to actually run something.

    Probes with a real allocation rather than trusting the free-memory
    figure, because a context that cannot be created at all reports no
    figure to trust. The result is cached per threshold: the probe costs a
    context initialization, and callers ask this question in loops.
    """
    global _cuda_usable_cache
    if not cuda_is_available():
        return False
    if _cuda_usable_cache is not None and _cuda_usable_cache[0] == min_free_bytes:
        return _cuda_usable_cache[1]

    usable = False
    try:
        free = cuda_free_memory_bytes()
        if free is not None and free >= min_free_bytes:
            # Allocate for real. mem_get_info can report memory that
            # fragmentation or another process's reservation makes
            # unobtainable in practice.
            probe = torch.zeros(1024 * 1024, dtype=torch.float32, device="cuda")
            del probe
            torch.cuda.empty_cache()
            usable = True
    except Exception:
        usable = False

    _cuda_usable_cache = (min_free_bytes, usable)
    return usable


def reset_cuda_usable_cache() -> None:
    """Forget the cached :func:`cuda_is_usable` probe.

    Device pressure comes and goes with other processes, so a long-running
    session may want to re-ask. Tests use this to assert both branches.
    """
    global _cuda_usable_cache
    _cuda_usable_cache = None


@dataclass(frozen=True)
class BackendSpec:
    """Identifies a concrete (engine, device, precision) combination."""

    engine: EngineKind
    device: DeviceKind
    precision: PrecisionKind = "fp32"

    def label(self) -> str:
        return f"{self.engine}/{self.device}/{self.precision}"


class Backend:
    """Executes array operations for a given :class:`BackendSpec`.

    NumPy backends always run on "cpu". Torch backends can run on "cpu" or
    "cuda". The same method names are used regardless of engine so that
    solver code is backend-agnostic.
    """

    def __init__(self, spec: BackendSpec):
        if spec.engine == "numpy" and spec.device != "cpu":
            raise ValueError("NumPy backend only supports the 'cpu' device")
        if spec.engine == "torch" and torch is None:
            raise RuntimeError("PyTorch is not installed")
        if spec.device == "cuda" and not cuda_is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        if spec.engine == "numpy" and spec.precision == "fp16":
            raise ValueError("NumPy backend does not support fp16 well on CPU; use torch")
        self.spec = spec

    # -- construction -----------------------------------------------------

    @property
    def np_dtype(self):
        return _PRECISION_TO_NP[self.spec.precision]

    @property
    def torch_dtype(self):
        return {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp64": torch.float64,
        }[self.spec.precision]

    def zeros(self, shape: tuple[int, ...]) -> Any:
        if self.spec.engine == "numpy":
            return np.zeros(shape, dtype=self.np_dtype)
        return torch.zeros(shape, dtype=self.torch_dtype, device=self.spec.device)

    def full(self, shape: tuple[int, ...], value: float) -> Any:
        if self.spec.engine == "numpy":
            return np.full(shape, value, dtype=self.np_dtype)
        return torch.full(shape, value, dtype=self.torch_dtype, device=self.spec.device)

    def array(self, data) -> Any:
        if self.spec.engine == "numpy":
            return np.asarray(data, dtype=self.np_dtype)
        return torch.as_tensor(np.asarray(data), dtype=self.torch_dtype, device=self.spec.device)

    def to_numpy(self, arr) -> np.ndarray:
        """Bring an array back to host memory as an independent float64 NumPy
        copy (for plotting/IO). Always copies -- `np.asarray`/`.numpy()` can
        silently return a view sharing memory with live solver state when
        the source is already float64 on CPU, which would let a caller that
        mutates the "snapshot" (e.g. buggy plotting code) corrupt the
        running simulation."""
        if self.spec.engine == "numpy":
            return np.array(arr, dtype=np.float64, copy=True)
        return arr.detach().to("cpu", dtype=torch.float64).numpy().copy()

    def clone(self, arr) -> Any:
        if self.spec.engine == "numpy":
            return arr.copy()
        return arr.clone()

    def meshgrid(self, x, y):
        if self.spec.engine == "numpy":
            return np.meshgrid(x, y, indexing="ij")
        return torch.meshgrid(x, y, indexing="ij")

    def linspace(self, start: float, stop: float, num: int):
        if self.spec.engine == "numpy":
            return np.linspace(start, stop, num, dtype=self.np_dtype)
        return torch.linspace(start, stop, num, dtype=self.torch_dtype, device=self.spec.device)

    def sum(self, arr) -> float:
        if self.spec.engine == "numpy":
            return float(np.sum(arr))
        return float(torch.sum(arr).item())

    def norm(self, arr) -> float:
        if self.spec.engine == "numpy":
            return float(np.linalg.norm(arr))
        return float(torch.linalg.vector_norm(arr).item())

    def max(self, arr) -> float:
        if self.spec.engine == "numpy":
            return float(np.max(arr))
        return float(torch.max(arr).item())

    def min(self, arr) -> float:
        if self.spec.engine == "numpy":
            return float(np.min(arr))
        return float(torch.min(arr).item())

    def abs(self, arr):
        if self.spec.engine == "numpy":
            return np.abs(arr)
        return torch.abs(arr)

    def isfinite_all(self, arr) -> bool:
        if self.spec.engine == "numpy":
            return bool(np.all(np.isfinite(arr)))
        return bool(torch.all(torch.isfinite(arr)).item())

    # -- synchronization & memory ------------------------------------------

    def synchronize(self) -> None:
        """Block until all queued device work has completed.

        Required before taking wall-clock timings: CUDA kernel launches are
        asynchronous from the host's perspective, so without this a timer
        would measure "time to enqueue work" rather than "time to do work".
        """
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            torch.cuda.synchronize()

    def memory_allocated_bytes(self) -> int | None:
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            return int(torch.cuda.memory_allocated())
        return None

    def memory_reserved_bytes(self) -> int | None:
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            return int(torch.cuda.memory_reserved())
        return None

    def reset_peak_memory_stats(self) -> None:
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def max_memory_allocated_bytes(self) -> int | None:
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return None

    def device_name(self) -> str:
        if self.spec.engine == "torch" and self.spec.device == "cuda":
            return torch.cuda.get_device_name(0)
        return "CPU"

    def bytes_per_element(self) -> int:
        return {"fp16": 2, "fp32": 4, "fp64": 8}[self.spec.precision]

    def to_device(self, arr, other_backend: Backend):
        """Move/convert an array produced by this backend to another backend."""
        host = self.to_numpy(arr)
        return other_backend.array(host)


def cpu_equivalent(spec: BackendSpec) -> BackendSpec:
    """The nearest CPU spec to ``spec``.

    fp16 is dropped to fp32: NumPy has no usable CPU fp16 and Torch's CPU
    fp16 is emulated slowly enough that timing it would mislead rather than
    inform.
    """
    precision = "fp32" if spec.precision == "fp16" else spec.precision
    return BackendSpec(spec.engine, "cpu", precision)


def make_backend(spec: BackendSpec, allow_cpu_fallback: bool = True) -> tuple[Backend, str | None]:
    """Build a backend, degrading to CPU rather than raising when asked for an
    unusable CUDA device.

    Returns ``(backend, warning)``. ``warning`` is None on success and a
    human-readable explanation when the request was downgraded, so the caller
    can surface *why* it is suddenly running on the CPU instead of leaving
    the user to guess from the frame rate.

    Raising was the old behaviour and it is the wrong one for an interactive
    lab: a card that is merely busy with someone else's process is not a
    reason to refuse to run the mathematics.
    """
    if spec.device == "cuda" and not cuda_is_usable():
        if not allow_cpu_fallback:
            raise RuntimeError("CUDA requested but not usable (absent, or out of free memory)")
        free = cuda_free_memory_bytes()
        if not cuda_is_available():
            why = "no CUDA device is present"
        elif free is None:
            why = "the CUDA context could not be created (device likely full)"
        else:
            why = (
                f"only {free / 1024 ** 2:.0f} MB of device memory is free, "
                f"below the {MIN_CUDA_FREE_BYTES / 1024 ** 2:.0f} MB floor"
            )
        fallback = cpu_equivalent(spec)
        return Backend(fallback), (
            f"Requested {spec.label()} but {why}; running on {fallback.label()} instead."
        )
    return Backend(spec), None


def available_backend_specs() -> list[BackendSpec]:
    """Specs that can actually be constructed right now.

    Gated on :func:`cuda_is_usable` rather than mere availability so the UI
    never offers a backend that would throw the moment it is selected.
    """
    specs = [
        BackendSpec("numpy", "cpu", "fp32"),
        BackendSpec("numpy", "cpu", "fp64"),
        BackendSpec("torch", "cpu", "fp32"),
        BackendSpec("torch", "cpu", "fp64"),
    ]
    if cuda_is_usable():
        specs += [
            BackendSpec("torch", "cuda", "fp16"),
            BackendSpec("torch", "cuda", "fp32"),
            BackendSpec("torch", "cuda", "fp64"),
        ]
    return specs


class WallClock:
    """A small helper that pairs a backend's synchronize() with time.perf_counter()."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self._t0 = 0.0

    def __enter__(self) -> WallClock:
        self.backend.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.backend.synchronize()
        self.elapsed = time.perf_counter() - self._t0
