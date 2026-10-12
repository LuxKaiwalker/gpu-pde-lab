"""CUDA-safe wall-clock timing.

CUDA kernel launches are asynchronous: the host issues a launch and moves on
immediately, while the GPU executes in the background. If you call
``time.perf_counter()`` before and after a GPU op without synchronizing, you
measure "how long it took to *enqueue* the work" (microseconds) rather than
"how long the GPU took to *do* the work" (which could be milliseconds). Every
timing helper here calls ``backend.synchronize()`` (a no-op on CPU backends,
``torch.cuda.synchronize()`` on CUDA) immediately before starting the clock
and immediately before stopping it, so queued work is flushed on both ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..core.backend import Backend


@dataclass
class TimingResult:
    elapsed_seconds: float
    n_calls: int = 1

    @property
    def seconds_per_call(self) -> float:
        return self.elapsed_seconds / max(self.n_calls, 1)


def timed(backend: Backend, fn, n_calls: int = 1) -> TimingResult:
    """Run ``fn()`` ``n_calls`` times, returning total synchronized wall time."""
    backend.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    backend.synchronize()
    elapsed = time.perf_counter() - t0
    return TimingResult(elapsed_seconds=elapsed, n_calls=n_calls)


@dataclass
class Lap:
    label: str
    seconds: float


class Stopwatch:
    """Records a sequence of named, backend-synchronized laps.

    Used by the host<->device transfer experiment to break a pipeline into
    stages (H2D copy, compute, D2H copy) and report each stage's cost
    individually rather than only the total.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.laps: list[Lap] = []
        self._t_last = 0.0

    def start(self) -> None:
        self.backend.synchronize()
        self._t_last = time.perf_counter()
        self.laps = []

    def lap(self, label: str) -> None:
        self.backend.synchronize()
        now = time.perf_counter()
        self.laps.append(Lap(label=label, seconds=now - self._t_last))
        self._t_last = now

    def total_seconds(self) -> float:
        return sum(l.seconds for l in self.laps)
