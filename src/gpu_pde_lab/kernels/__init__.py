"""Four implementations of one heat-equation step, for measuring against the roofline.

The rest of the project answers "is the GPU faster than the CPU". This package
answers the question after it: *given* a GPU, how much of it is this code
actually using, and what has to change to use more.

All four compute the identical update

    u^{n+1} = u^n + dt*alpha*laplacian(u^n),   boundary condition reapplied

and are validated against each other by
``tests/test_kernels.py``. They differ only in how the work reaches the device:

``baseline``
    What :mod:`gpu_pde_lab.pdes.heat` does: shifted-slice arithmetic through
    torch's eager dispatcher. Every intermediate is a full-sized temporary, so
    one step makes roughly four array-sized trips to memory and launches on the
    order of a dozen kernels.

``compiled``
    The same Python, handed to ``torch.compile``. The inductor backend fuses
    the elementwise chain into (close to) one kernel without anybody writing a
    kernel. This is the "what do I get for free" data point.

``triton``
    A hand-written fused kernel: one read of ``u``, arithmetic in registers,
    one write of ``u_new``. Boundary handling is folded in rather than left to
    a second launch. This is the floor the other three are measured against --
    a 5-point stencil cannot do better than read-once-write-once.

``cuda_graph``
    The ``baseline`` op sequence captured into a CUDA graph and replayed. It
    moves exactly as much memory as ``baseline``; all it removes is per-launch
    host overhead. Keeping it in the comparison is the point: it separates
    *launch-bound* from *bandwidth-bound*, which look identical from a
    wall-clock number alone and have completely different fixes.

Everything here is CUDA-only by nature and degrades by simply not being
offered: :func:`available_implementations` reports what can actually run.
"""

from __future__ import annotations

from .baseline import heat_step_baseline
from .compiled import compiled_is_available, make_compiled_heat_step
from .graphs import cuda_graph_is_available, make_graphed_heat_step
from .registry import (
    HeatStepImpl,
    available_implementations,
    build_implementation,
    implementation_names,
)
from .triton_stencil import make_triton_heat_step, triton_is_available

__all__ = [
    "HeatStepImpl",
    "available_implementations",
    "build_implementation",
    "compiled_is_available",
    "cuda_graph_is_available",
    "heat_step_baseline",
    "implementation_names",
    "make_compiled_heat_step",
    "make_graphed_heat_step",
    "make_triton_heat_step",
    "triton_is_available",
]
