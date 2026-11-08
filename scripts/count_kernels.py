#!/usr/bin/env python3
"""Count what each heat-step implementation actually asks the machine to do.

Two structural numbers per implementation, both measured rather than asserted:

**CUDA kernel launches per step** -- from ``torch.profiler``. This is the
launch-overhead story: each launch costs the host a few microseconds whether
the kernel does any work or not, which is why the small-grid end of the
roofline sweep is dominated by it.

**Array passes per step** -- from a ``TorchDispatchMode`` that totals the
elements read and written by every non-view op, in units of the whole field.
This is the bandwidth story. It is counted on CPU tensors because the dispatch
layer sees the same op sequence either way and it needs no GPU.

Both are reported per boundary condition, because the boundary handling is
exactly where the implementations differ structurally.

Why measure instead of hand-count: the eager path's traffic is easy to
underestimate badly. ``benchmarking/bandwidth.py`` documents a 4-pass model
(read u, write the Laplacian, read it back, write the new u) which is the
right *conceptual* account but roughly 8x off the real op sequence, because
every intermediate in `(north + south - 2*center)/dx2 + ...` is its own
full-sized tensor with its own round trip to memory.

Caveat worth keeping in mind when reading the pass counts: they are an upper
bound on DRAM traffic. Consecutive elementwise ops on a field small enough to
sit in L2 (5.5 MB on the reference card, so N <= 1024 at fp32) will hit cache
rather than DRAM for some of those passes. At N >= 2048 the field alone
exceeds L2 and the bound is tight.

Usage:
    python scripts/count_kernels.py                  # both, needs a GPU for launches
    python scripts/count_kernels.py --passes-only    # CPU-only, no GPU needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu_pde_lab.core.backend import cuda_is_usable
from gpu_pde_lab.core.boundary import BoundaryKind

# Ops that only reshape or alias memory. They dispatch, but they move nothing,
# so counting their "reads" would inflate the traffic figure with slices that
# are really just pointer arithmetic.
VIEW_OPS = frozenset({
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.lift_fresh.default",
    "aten.detach.default",
    "aten.view.default",
    "aten.expand.default",
    "aten.alias.default",
    "aten.as_strided.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
})


def count_array_passes(n: int, boundary: BoundaryKind) -> tuple[float, int]:
    """Return ``(array_passes, non_view_ops)`` for one baseline step.

    One "pass" is reading or writing the whole field once, so an unfused
    ``a + b -> c`` over full-sized tensors counts as 3.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from gpu_pde_lab.kernels.baseline import heat_step_baseline

    cells = float(n * n)

    class TrafficCounter(TorchDispatchMode):
        def __init__(self) -> None:
            self.passes = 0.0
            self.ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if str(func) not in VIEW_OPS:
                read = sum(a.numel() for a in args if isinstance(a, torch.Tensor))
                written = out.numel() if isinstance(out, torch.Tensor) else 0
                self.passes += (read + written) / cells
                self.ops += 1
            return out

    u = torch.rand((n, n), dtype=torch.float32)
    dx = dy = 1.0 / (n - 1)
    dt = 0.5 * min(dx, dy) ** 2 / (4.0 * 0.1)

    heat_step_baseline(u, dt, 0.1, dx, dy, boundary)  # warm up allocator
    counter = TrafficCounter()
    with counter:
        heat_step_baseline(u, dt, 0.1, dx, dy, boundary)
    return counter.passes, counter.ops


def count_kernel_launches(n: int, boundary: BoundaryKind, impl: str,
                           reps: int = 5, warmup: int = 12) -> tuple[float, list[tuple[float, float, str]]]:
    """Return ``(launches_per_step, [(count, us, name), ...])`` for ``impl``."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from gpu_pde_lab.kernels.registry import build_implementation

    dx = dy = 1.0 / (n - 1)
    alpha = 0.1
    dt = 0.5 * min(dx, dy) ** 2 / (4.0 * alpha)
    u = torch.rand((n, n), dtype=torch.float32, device="cuda")

    step = build_implementation(impl, boundary, shape=(n, n), dt=dt, alpha=alpha, dx=dx, dy=dy)
    # Warm up past compilation and autotuning; profiling those would report
    # the compiler's kernels rather than the steady-state ones.
    for _ in range(warmup):
        step(u, dt, alpha, dx, dy)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            step(u, dt, alpha, dx, dy)
        torch.cuda.synchronize()

    events = [
        e for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]
    total = sum(e.count for e in events) / reps
    rows = sorted(
        ((e.count / reps, e.self_device_time_total / reps, e.key) for e in events),
        key=lambda r: -r[1],
    )
    return total, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2048,
                        help="grid size; default 2048, above the reference card's L2")
    parser.add_argument("--passes-only", action="store_true",
                        help="skip the GPU profiling half")
    parser.add_argument("--detail", action="store_true",
                        help="list individual kernels, not just the count")
    args = parser.parse_args()

    print(f"Grid {args.n}x{args.n}, fp32\n")

    print("Array passes per step (read + write, in whole fields) -- eager baseline")
    print(f"{'boundary':<12} {'passes':>8} {'non-view ops':>13}")
    print("-" * 35)
    for bnd in BoundaryKind:
        passes, ops = count_array_passes(args.n, bnd)
        print(f"{bnd.name.lower():<12} {passes:>8.1f} {ops:>13}")
    print()
    print("A fused stencil needs 2 (one read, one write). The gap is the headroom.")
    print()

    if args.passes_only:
        return 0
    if not cuda_is_usable():
        print("No usable CUDA device; skipping the kernel-launch count.", file=sys.stderr)
        print("Re-run with --passes-only to silence this.", file=sys.stderr)
        return 1

    from gpu_pde_lab.kernels.registry import available_implementations

    print("CUDA kernel launches per step")
    print(f"{'impl':<12} {'launches':>9} {'device us':>11}")
    print("-" * 34)
    for d in available_implementations():
        if not d.available:
            print(f"{d.name:<12} {'skip':>9}   {d.unavailable_reason}")
            continue
        total, rows = count_kernel_launches(args.n, BoundaryKind.NEUMANN, d.name)
        device_us = sum(r[1] for r in rows)
        print(f"{d.name:<12} {total:>9.1f} {device_us:>11.1f}")
        if args.detail:
            for count, us, name in rows[:8]:
                print(f"             {count:>5.1f}x {us:>8.1f}us  {name[:58]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
