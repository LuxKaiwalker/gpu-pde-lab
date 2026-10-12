#!/usr/bin/env python3
"""Headless CPU vs GPU benchmark sweep -- the same numerics the interactive
"CPU vs GPU Benchmark" experiment runs, usable from the command line (e.g.
for recording results outside the GUI, or on a machine without a display).

Usage:
    python scripts/run_benchmark.py --sizes 128 256 512 1024 2048 4096 --out benchmark.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu_pde_lab.benchmarking.suite import run_cpu_gpu_sweep  # noqa: E402
from gpu_pde_lab.core.backend import cuda_is_available  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=str, default=None, help="optional PNG path for T(N)/speedup plots")
    args = parser.parse_args()

    if not cuda_is_available():
        print("No CUDA device detected -- this script compares CPU vs GPU, so it has nothing to do.")
        return 1

    print(f"Running CPU vs GPU heat-equation benchmark: sizes={args.sizes}, "
          f"precision={args.precision}, steps={args.steps}\n")

    result = run_cpu_gpu_sweep(args.sizes, precision=args.precision, n_steps=args.steps)

    header = f"{'N':>6} {'CPU ms/step':>12} {'GPU ms/step':>12} {'Speedup':>9} {'GPU GB/s':>10} {'GPU MB':>8}"
    print(header)
    print("-" * len(header))
    for n, c, g, s in zip(result.sizes, result.cpu, result.gpu, result.speedups()):
        print(f"{n:>6} {c.seconds_per_step*1e3:>12.3f} {g.seconds_per_step*1e3:>12.3f} "
              f"{s:>8.1f}x {g.bandwidth_gb_s:>10.1f} {g.gpu_memory_mb:>8.2f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        cpu_ms = [r.seconds_per_step * 1e3 for r in result.cpu]
        gpu_ms = [r.seconds_per_step * 1e3 for r in result.gpu]
        axes[0].loglog(result.sizes, cpu_ms, "o-", label="CPU")
        axes[0].loglog(result.sizes, gpu_ms, "o-", label="GPU")
        axes[0].set_xlabel("N (grid is N x N)")
        axes[0].set_ylabel("ms / step")
        axes[0].set_title("Runtime per step")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].semilogx(result.sizes, result.speedups(), "o-", color="tab:green")
        axes[1].set_xlabel("N (grid is N x N)")
        axes[1].set_ylabel("T_cpu / T_gpu")
        axes[1].set_title("GPU speedup")
        axes[1].grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved plot to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
