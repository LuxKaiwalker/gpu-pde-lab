#!/usr/bin/env python3
"""Render the project's figures by running the solvers, not by screenshotting.

Everything under ``docs/figures/`` is produced here. That is a deliberate
choice over screen captures: a screenshot is a claim about what the code did
once on one machine, while this script is the claim plus the means to check
it. Re-running it regenerates every image from the same seeded initial
conditions, so a figure that stops matching the code is a diff rather than a
discrepancy nobody notices.

It needs no GPU and no Qt. The solvers run on the NumPy backend and the
figures come out of matplotlib, which means this works in CI and on a laptop.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --out-dir /tmp/figs --only heat_stability
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gpu_pde_lab.core.backend import Backend, BackendSpec
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D, Grid3D
from gpu_pde_lab.pdes.fluid import StableFluids2D
from gpu_pde_lab.pdes.heat import HeatEquation
from gpu_pde_lab.pdes.heat_3d import HeatEquation3D
from gpu_pde_lab.pdes.poisson import PoissonProblem
from gpu_pde_lab.pdes.wave import WaveEquation

CPU = BackendSpec("numpy", "cpu", "fp64")
FIELD_CMAP = "magma"
DIVERGING_CMAP = "RdBu_r"


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel(fig, ax, data, cmap=FIELD_CMAP, symmetric=False):
    if symmetric:
        lim = max(abs(float(data.min())), abs(float(data.max())), 1e-12)
        im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(data.T, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# -- individual figures -----------------------------------------------------


#: Broadband perturbation added to the unstable run's initial condition.
#:
#: Von Neumann analysis says the FTCS amplification factor is
#: ``g = 1 - 4r(sin^2(kx h/2) + sin^2(ky h/2))`` with ``r = alpha dt/h^2``, so
#: the mode that first exceeds ``|g| = 1`` is the highest frequency the grid
#: can represent -- the cell-to-cell checkerboard at ``k = pi/h``.
#:
#: A smooth initial condition contains essentially none of that mode, so an
#: over-the-bound run *looks* stable for a long time: there is nothing at the
#: unstable frequency to amplify until floating-point round-off seeds it. That
#: is a real and slightly counterintuitive property, not an artifact, and it is
#: why this figure seeds the perturbation explicitly instead of waiting for
#: round-off. The amplitude is tiny; what matters is that it is broadband.
INSTABILITY_SEED_AMPLITUDE = 1e-6


def _unstable_initial_condition(grid: Grid2D, base: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return base + INSTABILITY_SEED_AMPLITUDE * rng.standard_normal(base.shape)


def fig_heat_stability(out: Path) -> Path:
    """Heat diffusion, and the same configuration pushed past its stability bound.

    The point of the figure is that the blow-up is a property of the
    discretization, not a bug: running the same unstable configuration on a
    faster GPU only reaches `inf` sooner.
    """
    import matplotlib.pyplot as plt

    grid = Grid2D(nx=128, ny=128)
    backend = Backend(CPU)

    stable = HeatEquation(grid, backend, BoundaryKind.NEUMANN, alpha=0.1)
    stable.reset(ic="multiple_sources")
    dt_max = stable.dt_max()
    initial = stable.primary_field_numpy()

    # 1.10x the von Neumann bound gives |g| = 1.2 for the checkerboard mode:
    # a 20% gain per step, so a 1e-6 seed reaches O(1) in well under 100 steps.
    over = 1.10
    unstable = HeatEquation(grid, backend, BoundaryKind.NEUMANN, alpha=0.1)
    unstable.reset_from_array(_unstable_initial_condition(grid, initial))

    for _ in range(400):
        stable.step(0.4 * dt_max)
    for _ in range(140):
        unstable.step(over * dt_max)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    _panel(fig, axes[0], initial)
    _style(axes[0], "t = 0")
    _panel(fig, axes[1], stable.primary_field_numpy())
    _style(axes[1], "400 steps at 0.40 x dt_max\n(stable: diffuses and decays)")

    u_bad = unstable.primary_field_numpy()
    _panel(fig, axes[2], np.nan_to_num(u_bad), cmap=DIVERGING_CMAP, symmetric=True)
    _style(axes[2], f"140 steps at {over:.2f} x dt_max\n(checkerboard mode, grid-scale)")

    steps = 220
    for label, dt in ((f"{over:.2f} x dt_max", over * dt_max), ("0.40 x dt_max", 0.40 * dt_max)):
        axes[3].semilogy(_amplitude_history(grid, backend, dt, steps, seed=dt > dt_max),
                         label=label)
    axes[3].axhline(float(np.max(np.abs(initial))), color="k", linestyle=":", linewidth=1,
                    label="initial amplitude")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("max |u|")
    axes[3].set_title("Amplitude vs step", fontsize=9)
    axes[3].legend(fontsize=7)
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Heat equation: explicit FTCS is conditionally stable, dt <= h^2/(4 alpha). "
        "Past the bound the grid-scale mode grows ~20% per step.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = out / "heat_stability.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _amplitude_history(grid, backend, dt, steps, seed: bool = False):
    """max|u| per step. ``seed`` adds the broadband perturbation the unstable
    mode needs in order to have anything to amplify."""
    pde = HeatEquation(grid, backend, BoundaryKind.NEUMANN, alpha=0.1)
    pde.reset(ic="multiple_sources")
    if seed:
        pde.reset_from_array(_unstable_initial_condition(grid, pde.primary_field_numpy()))
    history = []
    for _ in range(steps):
        pde.step(dt)
        u = pde.primary_field_numpy()
        history.append(float(np.max(np.abs(u))) if np.all(np.isfinite(u)) else np.nan)
    return history


def fig_wave(out: Path) -> Path:
    """Leapfrog wave propagation and reflection off the domain walls."""
    import matplotlib.pyplot as plt

    grid = Grid2D(nx=192, ny=192)
    backend = Backend(CPU)
    wave = WaveEquation(grid, backend, BoundaryKind.DIRICHLET, c=1.0, damping=0.0)
    wave.reset(ic="gaussian")
    dt = 0.5 * wave.dt_max()

    snapshots = []
    marks = (0, 90, 180, 320)
    for step in range(max(marks) + 1):
        if step in marks:
            snapshots.append((step, wave.primary_field_numpy().copy()))
        wave.step(dt)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.7 * len(snapshots), 3.6))
    for ax, (step, data) in zip(axes, snapshots, strict=True):
        _panel(fig, ax, data, cmap=DIVERGING_CMAP, symmetric=True)
        _style(ax, f"step {step}")
    fig.suptitle(
        "Wave equation: leapfrog integration, Dirichlet walls (CFL dt <= h/(c sqrt2))",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "wave.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_poisson_solvers(out: Path) -> Path:
    """Three iterative solvers on the same system, and their convergence rates.

    The residual plot is the figure's reason to exist: Jacobi and red-black
    Gauss-Seidel both parallelize, CG converges in orders of magnitude fewer
    iterations, and all three are the same stencil underneath.
    """
    import matplotlib.pyplot as plt

    grid = Grid2D(nx=128, ny=128)
    backend = Backend(CPU)

    def solve(method: str, iterations: int):
        problem = PoissonProblem(grid, backend, BoundaryKind.DIRICHLET)
        problem.set_rhs_gaussian_sources(n=4, amplitude=50.0, seed=1)
        residuals = [problem.residual_norm()]
        if method == "cg":
            problem.cg_init()
        for _ in range(iterations):
            if method == "jacobi":
                problem.jacobi_step()
            elif method == "redblack":
                problem.gauss_seidel_step_redblack()
            else:
                problem.cg_step()
            residuals.append(problem.residual_norm())
        return problem, residuals

    iterations = 300
    results = {
        "Jacobi": solve("jacobi", iterations),
        "Gauss-Seidel (red-black)": solve("redblack", iterations),
        "Conjugate Gradient": solve("cg", iterations),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))
    for ax, (name, (problem, _)) in zip(axes[:3], results.items(), strict=True):
        _panel(fig, ax, problem.backend.to_numpy(problem.u), cmap=DIVERGING_CMAP, symmetric=True)
        _style(ax, f"{name}\nafter {iterations} iterations")

    for name, (_, residuals) in results.items():
        axes[3].semilogy(residuals, label=name)
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("residual norm")
    axes[3].set_title("Convergence", fontsize=9)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Poisson equation as Au = b: same matrix-free stencil, three iterative solvers",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "poisson_solvers.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_fluid(out: Path) -> Path:
    """Stable Fluids: dye advected by an incompressible velocity field."""
    import matplotlib.pyplot as plt

    grid = Grid2D(nx=160, ny=160)
    backend = Backend(CPU)
    fluid = StableFluids2D(grid, backend, BoundaryKind.NEUMANN, viscosity=1e-5)
    fluid.reset()

    # Two opposed puffs, injected only at the start rather than every step.
    # A continuous source makes the dye scale grow without bound (the colour
    # bar then reports the injection rate, not the flow) and drives the
    # velocity high enough that semi-Lagrangian backtracing skips cells,
    # which shows up as speckle instead of structure.
    dt = 0.02
    forcing_steps = 25
    snapshots = []
    marks = (25, 120, 280, 500)
    divergence = []
    for step in range(max(marks) + 1):
        if step < forcing_steps:
            fluid.add_source(0.30, 0.30, 3.0, 2.2, 1.5, radius=0.09)
            fluid.add_source(0.70, 0.68, -3.0, -2.2, 1.5, radius=0.09)
        fluid.step(dt)
        divergence.append(fluid.divergence_norm())
        if step in marks:
            snapshots.append((step, fluid.dye_numpy().copy()))

    fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(3.6 * (len(snapshots) + 1), 3.6))
    for ax, (step, data) in zip(axes[:-1], snapshots, strict=True):
        _panel(fig, ax, data)
        _style(ax, f"step {step}")

    axes[-1].semilogy(divergence)
    axes[-1].set_xlabel("step")
    axes[-1].set_ylabel("||div u||")
    axes[-1].set_title("Divergence after projection\n(fixed iteration budget)", fontsize=9)
    axes[-1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "2D Stable Fluids: semi-Lagrangian advection + Jacobi pressure projection",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "fluid.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_heat_3d(out: Path) -> Path:
    """The 3D vertical slice, shown the way the app shows it: three orthogonal planes."""
    import matplotlib.pyplot as plt

    grid = Grid3D(nx=64, ny=64, nz=64)
    backend = Backend(CPU)
    pde = HeatEquation3D(grid, backend, BoundaryKind.DIRICHLET, alpha=0.1)
    pde.reset()
    dt = 0.4 * pde.dt_max()
    for _ in range(120):
        pde.step(dt)

    volume = pde.primary_field_numpy()
    mid = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
    planes = [
        (volume[mid[0], :, :], f"x = {mid[0]}"),
        (volume[:, mid[1], :], f"y = {mid[1]}"),
        (volume[:, :, mid[2]], f"z = {mid[2]}"),
    ]

    # One shared colour scale across the three planes. Autoscaling each
    # independently would make slices through different parts of the volume
    # look equally hot, which is exactly the misreading the app's own viewer
    # avoids.
    vmax = float(volume.max())
    vmin = float(volume.min())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (data, label) in zip(axes, planes, strict=True):
        im = ax.imshow(data.T, origin="lower", cmap=FIELD_CMAP, vmin=vmin, vmax=vmax)
        _style(ax, f"slice {label}")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(
        "3D heat equation: 7-point stencil on a 64^3 grid, three orthogonal slices, shared colour scale",
        fontsize=10,
    )
    path = out / "heat_3d.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_convergence(out: Path) -> Path:
    """Measured discretization error against an exact analytical solution."""
    import matplotlib.pyplot as plt

    from gpu_pde_lab.benchmarking.convergence import spatial_convergence

    result = spatial_convergence([17, 33, 65, 129, 257])
    h = np.array([p.h_or_dt for p in result.points], dtype=float)
    errors = np.array([p.error_linf for p in result.points], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.loglog(h, errors, "o-", label="measured Linf error")
    # Reference slope anchored at the finest point, so the comparison is a
    # slope comparison rather than a fit.
    ref = errors[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "k--", label="order 2 reference")
    ax.set_xlabel("h = 1/(N-1)")
    ax.set_ylabel("Linf error vs analytical solution")
    avg = sum(result.observed_orders) / len(result.observed_orders)
    ax.set_title(f"Spatial convergence of the 5-point stencil\nobserved order {avg:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = out / "convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_stencil(out: Path) -> Path:
    """The 5-point stencil as one GPU thread's view of its neighbourhood."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    n = 7
    for i in range(n + 1):
        ax.axhline(i, color="#cccccc", linewidth=0.8)
        ax.axvline(i, color="#cccccc", linewidth=0.8)

    cx = cy = 3
    neighbours = {(cx, cy + 1): "north", (cx, cy - 1): "south",
                  (cx - 1, cy): "west", (cx + 1, cy): "east"}
    for (x, y), label in neighbours.items():
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#6fa8dc", alpha=0.75))
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((cx, cy), 1, 1, color="#e06666", alpha=0.85))
    ax.text(cx + 0.5, cy + 0.5, "u[i,j]", ha="center", va="center", fontsize=9, weight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "One output cell reads only its four neighbours.\n"
        "No cell depends on another cell's new value, so every\n"
        "cell can be computed at once -- one GPU thread each.",
        fontsize=9,
    )
    fig.tight_layout()
    path = out / "stencil.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


FIGURES = {
    "heat_stability": fig_heat_stability,
    "wave": fig_wave,
    "poisson_solvers": fig_poisson_solvers,
    "fluid": fig_fluid,
    "heat_3d": fig_heat_3d,
    "convergence": fig_convergence,
    "stencil": fig_stencil,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                        help="render a subset (default: all)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wanted = args.only or list(FIGURES)
    for name in wanted:
        path = FIGURES[name](out)
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"\n{len(wanted)} figure(s) written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
