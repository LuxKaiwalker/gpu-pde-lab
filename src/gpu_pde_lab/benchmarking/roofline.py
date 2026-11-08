"""How much of the memory system is the stencil actually using?

A 5-point Laplacian does about 6 floating-point operations per cell and moves
at least 8 bytes per cell (one fp32 read, one fp32 write). That is an
arithmetic intensity under 1 FLOP/byte, which on any modern GPU puts it deep
in the memory-bound regime -- hundreds of times below the ridge point where
compute throughput would start to matter. So the only performance question
worth asking is what fraction of available bandwidth the implementation
achieves, and the answer is a number, not an opinion.

Two ceilings, and why both are reported
---------------------------------------
``peak``
    Derived from the device's own reported memory clock and bus width:
    ``clock * 2 (DDR) * bus_width / 8``. For an RTX 2080 Ti that is
    7 GHz x 2 x 44 B = 616 GB/s. It is the number on the spec sheet and no
    real kernel reaches it.

``achievable``
    Measured here, by timing a plain device-to-device copy of the same array
    size. A copy is the simplest possible read-once-write-once kernel, so it
    is the honest ceiling for a fused stencil: whatever fraction of peak the
    copy gets is the fraction a perfect stencil could hope for.

Reporting the stencil against *both* is what keeps the claim defensible.
"Triton reaches 78% of peak" would be a stronger-sounding and less meaningful
statement than "Triton reaches 94% of what a memcpy achieves on this card",
because the second isolates the implementation from the hardware's own
inefficiency.

What "achieved bandwidth" means here
------------------------------------
Bandwidth is reported as **useful** bytes per second: the traffic the
computation irreducibly requires -- one read of ``u`` and one write of
``u_new``, so ``2 * N^2 * sizeof(dtype)`` -- divided by the measured time.
Every implementation is scored against the same numerator.

The alternative, crediting each implementation for the traffic it actually
generates, produces a genuinely misleading table: the eager baseline makes
about **33** array-sized passes where one read and one write would do, so
scoring it on its own traffic rewards it for the waste. Measured that way the
CUDA-graph implementation reports *higher* bandwidth than the fused kernel
while taking *longer* per step, which is true and useless.

Scoring useful bytes instead means the number answers the question actually
being asked -- of the bandwidth this computation needs, what rate is being
delivered -- and an implementation that wastes traffic shows up as a lower
figure, which is the correct direction. The per-implementation pass count is
still reported in its own column, because it is the *explanation* for the
shortfall even though it must not be folded into the metric.

Two further caveats, stated rather than buried.

Pass counts come from ``scripts/count_kernels.py``, which totals the elements
every non-view op reads and writes at the dispatch layer. That is a count of
*requested* traffic, not of DRAM transactions: a real profiler (nsight)
measures the latter and would report less, because the L2 absorbs some of it.
So the pass counts are an upper bound, tight only once the field alone exceeds
L2 (5.5 MB on the reference card, so from N=2048 at fp32).

For the same reason, at small grid sizes the entire working set lives in L2
and the "bandwidth" figure describes cache rather than DRAM -- which is why
the measured memcpy ceiling, taken at the same size, is the ceiling worth
comparing against. The sweep deliberately spans that crossover so it shows up
instead of hiding.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

from ..core.backend import PrecisionKind, cuda_is_usable
from ..core.boundary import BoundaryKind
from ..kernels.registry import available_implementations, build_implementation
from .timer import timed

try:
    import torch
except ImportError:  # pragma: no cover
    # Typed as Any so the None fallback is assignable; the module is optional
    # at import time and every use is guarded.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DeviceInfo:
    """Everything needed to make a published number reproducible."""

    name: str
    total_memory_bytes: int
    memory_clock_khz: int
    memory_bus_bits: int
    multiprocessors: int
    l2_cache_bytes: int
    compute_capability: str
    torch_version: str
    platform: str

    @property
    def peak_bandwidth_bytes_per_s(self) -> float:
        """Theoretical peak: clock x 2 (double data rate) x bus width."""
        return self.memory_clock_khz * 1e3 * 2.0 * (self.memory_bus_bits / 8.0)

    def summary(self) -> str:
        return (
            f"{self.name} (sm_{self.compute_capability}, {self.multiprocessors} SMs, "
            f"{self.total_memory_bytes / 1024 ** 3:.1f} GiB, "
            f"{self.memory_bus_bits}-bit bus, "
            f"peak {self.peak_bandwidth_bytes_per_s / 1e9:.0f} GB/s, "
            f"L2 {self.l2_cache_bytes / 1024 ** 2:.1f} MiB) | "
            f"torch {self.torch_version} | {self.platform}"
        )


def collect_device_info() -> DeviceInfo | None:
    """Describe the active CUDA device, or None if there isn't a usable one."""
    if torch is None or not cuda_is_usable():
        return None
    p = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        name=p.name,
        total_memory_bytes=int(p.total_memory),
        memory_clock_khz=int(getattr(p, "memory_clock_rate", 0)),
        memory_bus_bits=int(getattr(p, "memory_bus_width", 0)),
        multiprocessors=int(p.multi_processor_count),
        l2_cache_bytes=int(getattr(p, "L2_cache_size", 0)),
        compute_capability=f"{p.major}{p.minor}",
        torch_version=torch.__version__,
        platform=f"{platform.system()} {platform.release()}",
    )


def measure_copy_bandwidth(n: int, dtype_str: PrecisionKind = "fp32", n_calls: int = 50) -> float:
    """Device-to-device copy bandwidth for an ``n x n`` array, in bytes/second.

    The empirical read-once-write-once ceiling. Uses ``copy_`` into a
    preallocated destination so the measurement contains no allocation, and
    counts 2 array passes (one read, one write).
    """
    from ..core.backend import Backend, BackendSpec

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[dtype_str]
    src = torch.randn((n, n), dtype=torch_dtype, device="cuda")
    dst = torch.empty_like(src)
    backend = Backend(BackendSpec("torch", "cuda", dtype_str))

    for _ in range(5):  # warm up: first touch faults pages in
        dst.copy_(src)
    result = timed(backend, lambda: dst.copy_(src), n_calls=n_calls)

    bytes_moved = 2 * src.numel() * src.element_size()
    return bytes_moved / result.seconds_per_call


#: The irreducible traffic of a stencil update: read the field once, write it
#: once. Used as the numerator for every implementation -- see the module
#: docstring on why scoring each one on its own wasted traffic would mislead.
USEFUL_PASSES_PER_STEP = 2


@dataclass
class ImplMeasurement:
    """One implementation at one grid size."""

    impl: str
    n: int
    seconds_per_step: float
    kernel_launches: int | None
    array_passes: float | None
    useful_bytes_per_step: int
    estimated_actual_bytes_per_step: int | None
    achieved_bandwidth_bytes_per_s: float
    copy_ceiling_bytes_per_s: float
    peak_bandwidth_bytes_per_s: float
    max_abs_diff_from_baseline: float

    @property
    def fraction_of_peak(self) -> float:
        return self.achieved_bandwidth_bytes_per_s / self.peak_bandwidth_bytes_per_s

    @property
    def fraction_of_achievable(self) -> float:
        if self.copy_ceiling_bytes_per_s <= 0:
            return float("nan")
        return self.achieved_bandwidth_bytes_per_s / self.copy_ceiling_bytes_per_s

    @property
    def cells_per_second(self) -> float:
        if self.seconds_per_step <= 0:
            return float("inf")
        return (self.n * self.n) / self.seconds_per_step

    @property
    def wasted_traffic_factor(self) -> float | None:
        """How many times more memory this moves than it needs to.

        None where the op sequence is fused away and no honest count exists --
        see ``kernels/registry.ARRAY_PASSES_PER_STEP``.
        """
        if self.estimated_actual_bytes_per_step is None:
            return None
        return self.estimated_actual_bytes_per_step / self.useful_bytes_per_step


@dataclass
class RooflineSweep:
    device: DeviceInfo | None
    sizes: list[int]
    precision: str
    boundary: str
    steps_per_measurement: int
    measurements: list[ImplMeasurement] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def by_impl(self, impl: str) -> list[ImplMeasurement]:
        return [m for m in self.measurements if m.impl == impl]

    def impls_present(self) -> list[str]:
        seen: list[str] = []
        for m in self.measurements:
            if m.impl not in seen:
                seen.append(m.impl)
        return seen

    def speedup_table(self) -> dict[str, dict[int, float]]:
        """``impl -> {n: speedup over baseline}``, for the summary table."""
        base = {m.n: m.seconds_per_step for m in self.by_impl("baseline")}
        out: dict[str, dict[int, float]] = {}
        for impl in self.impls_present():
            if impl == "baseline":
                continue
            out[impl] = {
                m.n: base[m.n] / m.seconds_per_step
                for m in self.by_impl(impl)
                if m.n in base and m.seconds_per_step > 0
            }
        return out


def run_roofline_sweep(
    sizes: list[int],
    precision: PrecisionKind = "fp32",
    boundary: BoundaryKind = BoundaryKind.NEUMANN,
    alpha: float = 0.1,
    n_steps: int = 50,
    warmup: int = 10,
    impls: list[str] | None = None,
) -> RooflineSweep:
    """Time every available implementation across ``sizes``.

    Correctness is checked alongside the timing rather than in a separate
    pass: each implementation's output is compared against the baseline's at
    every size, and the worst difference is recorded in the result. A
    performance number from an implementation that computes the wrong thing
    is worse than no number, so the two travel together.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")

    from ..core.backend import Backend, BackendSpec
    from ..core.grid import Grid2D

    device = collect_device_info()
    sweep = RooflineSweep(
        device=device,
        sizes=list(sizes),
        precision=precision,
        boundary=boundary.value,
        steps_per_measurement=n_steps,
    )

    descriptors = {d.name: d for d in available_implementations()}
    wanted = impls or list(descriptors)
    runnable: list[str] = []
    for name in wanted:
        d = descriptors[name]
        # The baseline is listed as always-available because it runs on CPU,
        # but this sweep is GPU-only, so gate it on the device too.
        if name == "baseline" and not cuda_is_usable():
            sweep.skipped[name] = "no usable CUDA device"
            continue
        if d.available:
            runnable.append(name)
        else:
            sweep.skipped[name] = d.unavailable_reason or "unavailable"

    if not runnable:
        return sweep

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[precision]
    backend = Backend(BackendSpec("torch", "cuda", precision))
    peak = device.peak_bandwidth_bytes_per_s if device else float("nan")

    for n in sizes:
        grid = Grid2D(nx=n, ny=n)
        dx, dy = grid.dx, grid.dy
        dt = 0.5 * min(dx, dy) ** 2 / (4.0 * alpha)

        # One fixed initial field per size, so every implementation is timed
        # and compared on identical input.
        generator = torch.Generator(device="cuda").manual_seed(0)
        u0 = torch.rand((n, n), generator=generator, dtype=torch_dtype, device="cuda")
        copy_ceiling = measure_copy_bandwidth(n, precision)

        reference = None
        for name in runnable:
            step = build_implementation(
                name, boundary,
                shape=(n, n), dt=dt, alpha=alpha, dx=dx, dy=dy, dtype_str=precision,
            )

            # Warm up outside the timed region. torch.compile spends seconds
            # on its first call and Triton compiles its kernel there; timing
            # that would measure the compiler, not the code.
            #
            # At least one step always runs: the result is what the
            # correctness check compares, so `warmup=0` must still produce an
            # output rather than leaving it unset.
            out = step(u0, dt, alpha, dx, dy)
            for _ in range(max(warmup - 1, 0)):
                out = step(u0, dt, alpha, dx, dy)
            backend.synchronize()

            if name == "baseline":
                reference = out.clone()
                diff = 0.0
            elif reference is None:
                # Only reachable if the caller excluded the baseline from
                # `impls`; there is then nothing to compare against and
                # claiming agreement would be a lie.
                diff = float("nan")
            else:
                diff = float((out - reference).abs().max().item())

            result = timed(backend, lambda s=step: s(u0, dt, alpha, dx, dy), n_calls=n_steps)
            spc = result.seconds_per_call

            passes = descriptors[name].array_passes
            cell_bytes = n * n * u0.element_size()
            useful_bytes = USEFUL_PASSES_PER_STEP * cell_bytes
            sweep.measurements.append(
                ImplMeasurement(
                    impl=name,
                    n=n,
                    seconds_per_step=spc,
                    kernel_launches=descriptors[name].kernel_launches,
                    array_passes=passes,
                    useful_bytes_per_step=useful_bytes,
                    estimated_actual_bytes_per_step=(
                        int(passes * cell_bytes) if passes is not None else None
                    ),
                    achieved_bandwidth_bytes_per_s=useful_bytes / spc if spc > 0 else float("inf"),
                    copy_ceiling_bytes_per_s=copy_ceiling,
                    peak_bandwidth_bytes_per_s=peak,
                    max_abs_diff_from_baseline=diff,
                )
            )

        # Each size holds several n*n buffers plus whatever inductor cached;
        # release between sizes so a long sweep does not accumulate its way
        # into an out-of-memory error at the largest grid.
        # Rebind rather than `del`: the names are reused on the next
        # iteration, and deleting them leaves the loop body referencing a
        # statically-unbound name. Dropping the references is what matters --
        # empty_cache() cannot reclaim memory something still points at.
        u0 = None  # type: ignore[assignment]
        reference = None
        torch.cuda.empty_cache()

    return sweep
