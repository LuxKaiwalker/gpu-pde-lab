"""``torch.compile`` over the baseline: fusion without writing a kernel.

This is the "what do I get for free" data point. The source is the same
shifted-slice arithmetic as :mod:`.baseline`; inductor traces it, sees a chain
of elementwise ops over identically-shaped tensors, and emits (close to) a
single kernel -- recovering much of what the hand-written Triton kernel gets
explicitly, with no new code.

Keeping it in the comparison is what makes the Triton number honest. "Hand
kernel beats eager" is unsurprising and proves nothing about the kernel;
"hand kernel beats the compiler" is the claim worth measuring, and it is the
one a reviewer would otherwise rightly ask about.

Two practical notes the timings depend on:

- Compilation happens on the first call and costs seconds. Every caller here
  must warm up outside the timed region, or the first measurement is dominated
  by compile time.
- ``dynamic=False`` specializes on the grid shape. The roofline harness sweeps
  sizes, and without this each new size silently triggers a recompile that
  lands inside whatever is being measured next.
"""

from __future__ import annotations

from ..core.boundary import BoundaryKind
from .baseline import heat_step_baseline


def compiled_is_available() -> bool:
    """True if ``torch.compile`` exists and has a usable CUDA device to target.

    Inductor can target CPU too, but this package exists to measure GPU
    behaviour, so a CPU-only machine reports the implementation as absent
    rather than quietly measuring something else.
    """
    try:
        import torch
    except ImportError:
        return False
    if not hasattr(torch, "compile"):
        return False
    from ..core.backend import cuda_is_usable

    return cuda_is_usable()


def make_compiled_heat_step(boundary: BoundaryKind, mode: str = "max-autotune"):
    """Return a ``(u, dt, alpha, dx, dy) -> u_new`` closure compiled by inductor.

    ``mode="max-autotune"`` lets inductor benchmark candidate schedules and
    keep the best. It makes the first call substantially slower -- tens of
    seconds is normal -- which is exactly why warmup is the caller's job.
    """
    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError("this torch build has no torch.compile")

    def _step(u, dt: float, alpha: float, dx: float, dy: float, dirichlet_value: float = 0.0):
        return heat_step_baseline(u, dt, alpha, dx, dy, boundary, dirichlet_value)

    compiled = torch.compile(_step, mode=mode, dynamic=False)

    def step(u, dt: float, alpha: float, dx: float, dy: float, dirichlet_value: float = 0.0):
        return compiled(u, dt, alpha, dx, dy, dirichlet_value)

    step.__doc__ = f"torch.compile({mode}) heat step, {boundary.value} boundary."
    return step
