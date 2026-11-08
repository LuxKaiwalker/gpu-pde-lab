"""One place that knows which heat-step implementations exist and can run.

Callers (the benchmark harness, the UI experiment, the tests) ask here rather
than each repeating the same availability checks. An implementation that
cannot run on this machine is reported with the reason, not omitted silently:
a roofline plot missing its Triton bar should say *why* it is missing, or a
reader will assume the kernel failed rather than that the GPU was busy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.boundary import BoundaryKind

#: Display order, slowest-expected to fastest-expected. The roofline plot
#: reads left to right as a story, so the order is part of the output.
IMPLEMENTATION_ORDER = ("baseline", "cuda_graph", "compiled", "triton")

_DESCRIPTIONS = {
    "baseline": "eager shifted-slice arithmetic (17 kernels, ~33 array passes)",
    "cuda_graph": "the same ops captured in a CUDA graph (same traffic, one submission)",
    "compiled": "torch.compile max-autotune (inductor fuses to 4 kernels)",
    "triton": "hand-written fused kernel (1 kernel, one read, one write)",
}

#: Array-sized passes over memory per step -- reading or writing the whole
#: field counts as one. **Measured**, not hand-counted, by
#: ``scripts/count_kernels.py`` (a ``TorchDispatchMode`` totalling the elements
#: every non-view op reads and writes) at N=2048, fp32, Neumann.
#:
#: The eager figure is worth pausing on. ``benchmarking/bandwidth.py``
#: documents a 4-pass model -- read u, write the Laplacian, read it back,
#: write the new u -- which is the right conceptual account and about 8x
#: optimistic, because in eager mode every intermediate in
#: ``(north + south - 2*center)/dx2 + (east + west - 2*center)/dy2`` is its
#: own full-sized tensor with its own round trip. The real figure is ~33.
#:
#: ``None`` means not separately countable: inductor fuses the op sequence
#: away, so the dispatch layer no longer sees individual passes and the
#: honest thing is to report its kernel count and runtime instead of
#: inventing a traffic number for it.
ARRAY_PASSES_PER_STEP: dict[str, float | None] = {
    "baseline": 32.9,
    "cuda_graph": 32.9,  # identical ops, captured rather than relaunched
    "compiled": None,
    "triton": 2.0,  # by construction: one read, one write, boundary folded in
}

#: CUDA kernel launches per step, **measured** by ``scripts/count_kernels.py``
#: via ``torch.profiler`` at N=2048, fp32, Neumann, on the reference RTX
#: 2080 Ti. This is the structural number behind the small-grid end of the
#: roofline: every launch costs the host microseconds whether or not the
#: kernel does any work.
#:
#: ``cuda_graph`` launches the same 17 kernels plus 3 device-to-device copies
#: to and from its fixed capture buffers, but submits the whole sequence in
#: one go -- which is the entire point, and why its win shows up only where
#: launch overhead dominates.
KERNEL_LAUNCHES_PER_STEP = {
    "baseline": 17,
    "cuda_graph": 18,
    "compiled": 4,
    "triton": 1,
}


@dataclass(frozen=True)
class HeatStepImpl:
    """A named heat-step implementation plus why it is or isn't runnable."""

    name: str
    description: str
    available: bool
    unavailable_reason: str | None = None
    array_passes: float | None = None
    kernel_launches: int | None = None

    def require(self) -> None:
        if not self.available:
            raise RuntimeError(f"implementation {self.name!r} is unavailable: {self.unavailable_reason}")


def implementation_names() -> tuple[str, ...]:
    return IMPLEMENTATION_ORDER


def _cuda_reason() -> str | None:
    from ..core.backend import cuda_free_memory_bytes, cuda_is_available, cuda_is_usable

    if cuda_is_usable():
        return None
    if not cuda_is_available():
        return "no CUDA device is present"
    free = cuda_free_memory_bytes()
    if free is None:
        return "the CUDA context could not be created (device likely full)"
    return f"only {free / 1024 ** 2:.0f} MB of device memory is free"


def available_implementations() -> list[HeatStepImpl]:
    """Describe every implementation, runnable or not, in display order."""
    from .compiled import compiled_is_available
    from .graphs import cuda_graph_is_available
    from .triton_stencil import triton_is_available, triton_unavailable_reason

    cuda_why = _cuda_reason()
    out: list[HeatStepImpl] = []
    for name in IMPLEMENTATION_ORDER:
        if name == "baseline":
            # The baseline is the only one that also runs on CPU, so it is
            # always available -- it is the reference, and a comparison with
            # nothing in it is worse than one with a single bar.
            available, why = True, None
        elif name == "cuda_graph":
            available, why = cuda_graph_is_available(), cuda_why
        elif name == "compiled":
            available, why = compiled_is_available(), cuda_why
        else:
            available = triton_is_available()
            why = triton_unavailable_reason()
        out.append(
            HeatStepImpl(
                name=name,
                description=_DESCRIPTIONS[name],
                available=available,
                unavailable_reason=None if available else why,
                array_passes=ARRAY_PASSES_PER_STEP[name],
                kernel_launches=KERNEL_LAUNCHES_PER_STEP[name],
            )
        )
    return out


def build_implementation(
    name: str,
    boundary: BoundaryKind,
    *,
    shape: tuple[int, int] | None = None,
    dt: float | None = None,
    alpha: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dtype_str: str = "fp32",
    dirichlet_value: float = 0.0,
) -> Callable:
    """Build a uniform ``(u, dt, alpha, dx, dy) -> u_new`` callable for ``name``.

    The CUDA-graph implementation needs its shape and parameters at capture
    time, so those arguments are required for it and ignored by the others.
    The returned callable still takes ``dt``/``alpha``/``dx``/``dy`` for a
    uniform call signature; for the graph they must match what was captured,
    and a mismatch raises rather than silently replaying the old parameters.
    """
    from .baseline import heat_step_baseline
    from .compiled import make_compiled_heat_step
    from .graphs import make_graphed_heat_step
    from .triton_stencil import make_triton_heat_step

    if name == "baseline":
        def baseline_step(u, dt_, alpha_, dx_, dy_, dirichlet_value_=dirichlet_value):
            return heat_step_baseline(u, dt_, alpha_, dx_, dy_, boundary, dirichlet_value_)

        return baseline_step

    if name == "compiled":
        return make_compiled_heat_step(boundary)

    if name == "triton":
        return make_triton_heat_step(boundary)

    if name == "cuda_graph":
        missing = [k for k, v in
                   {"shape": shape, "dt": dt, "alpha": alpha, "dx": dx, "dy": dy}.items()
                   if v is None]
        if missing:
            raise ValueError(
                "the cuda_graph implementation bakes its parameters in at capture "
                f"time, so these are required: {', '.join(missing)}"
            )
        # Rebind narrowed: the check above proves these are set, but that is a
        # runtime invariant a type checker cannot follow through a dict.
        assert shape is not None and dt is not None and alpha is not None
        assert dx is not None and dy is not None
        graphed = make_graphed_heat_step(
            shape, dt, alpha, dx, dy, boundary, dirichlet_value, dtype_str=dtype_str
        )

        def graph_step(u, dt_, alpha_, dx_, dy_, dirichlet_value_=dirichlet_value):
            if (dt_, alpha_, dx_, dy_) != (dt, alpha, dx, dy):
                raise ValueError(
                    "this graph was captured with "
                    f"dt={dt}, alpha={alpha}, dx={dx}, dy={dy}; replaying it with "
                    f"dt={dt_}, alpha={alpha_}, dx={dx_}, dy={dy_} would silently use "
                    "the captured values. Recapture instead."
                )
            return graphed.step(u)

        graph_step.graphed = graphed  # type: ignore[attr-defined]
        return graph_step

    raise ValueError(f"unknown implementation {name!r}; known: {IMPLEMENTATION_ORDER}")
