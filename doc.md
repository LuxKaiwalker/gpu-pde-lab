# GPU PDE Lab

An interactive laboratory for learning how GPUs solve partial differential
equations — by implementing the numerical methods yourself and watching them
run, rather than calling into a black-box CFD/PDE package.

The project makes this pipeline visible end-to-end, for every experiment:

```
PDE  →  discretization  →  numerical algorithm  →  parallel computation  →  GPU implementation  →  visual result
```

Every stencil, timestepper, and iterative linear solver in this repository
is implemented from scratch in plain NumPy/PyTorch array operations. PyTorch
is used only as a GPU execution engine (tensors + CUDA kernels for array
arithmetic) and PySide6/pyqtgraph only for windowing/plotting — the actual
numerics (finite differences, explicit time integration, Jacobi/Gauss-Seidel/
Conjugate Gradient, semi-Lagrangian advection) contain no calls into a
pre-built PDE/CFD solver.

## 1. What's in here

Twelve interactive experiments, grouped into three categories in the
experiment dropdown (PDE Simulations / GPU Experiments / Numerical Analysis)
in a single desktop app. A header bar at the top always shows which GPU (or
"CPU only") is active, regardless of which experiment is selected:

| Experiment | Kind | What it teaches |
|---|---|---|
| Heat Equation (+ Stability) | live | explicit FTCS diffusion; CFL-style stability bound; deliberate instability |
| **3D Heat Equation** | live | the same scheme generalized to a 7-point stencil on a `Grid3D`, inspected via three movable orthogonal slices |
| Wave Equation | live | leapfrog time integration; damping; a different (linear, not quadratic) CFL bound |
| Poisson Equation (Iterative Solvers) | live | elliptic PDEs as `Au=b`; Jacobi vs Gauss-Seidel vs Conjugate Gradient |
| 2D Fluid Simulation | live | operator-split incompressible Navier-Stokes (Stam's Stable Fluids) |
| Stencil Visualization | live | the 5-point stencil, cell-by-cell, mapped onto "one GPU thread per cell" |
| CPU vs GPU Benchmark | benchmark | where GPU parallelism pays off (and where it doesn't) |
| Host ↔ Device Transfer | benchmark | why PCIe round-trips dominate wall time at small/medium problem sizes |
| GPU Memory | benchmark | why "just one small array" becomes real memory pressure with multiple buffers |
| Kernel / Vectorization Comparison | benchmark | Python loop vs NumPy vs Torch-CPU vs Torch-CUDA, same math |
| Resolution / Convergence Study | benchmark | discretization error vs `h`/`dt` against a known analytical solution |
| Precision Comparison (fp16/fp32/fp64) | benchmark | runtime/memory/error/stability tradeoffs across precisions |

Every experiment ships with an explanation panel answering: *what PDE is
this, how is it discretized, what does one grid cell compute, why can it run
in parallel, what does the GPU actually do, and what can go wrong.*

**3D is Heat-only, by design.** `3D Heat Equation` is a complete, real
vertical slice (`core/grid.py`'s `Grid3D`, `core/stencil.py`'s `laplacian_3d`,
`pdes/heat_3d.py`, `ui/viz/slice_viewer_3d.py`) — not a placeholder. Wave,
Poisson, and Fluid have no 3D solver yet; extending each would need its own
numerical work (a 3D CFL bound for Wave, a 3D Poisson operator, 3D
semi-Lagrangian advection + trilinear interpolation for Fluid) rather than
reusing this one. Find it at: **Experiment dropdown → PDE Simulations → 3D
Heat Equation**.

## 2. Installation

This project targets **NixOS** (flakes-based dev shell) with an **NVIDIA
GPU + CUDA**, but the Python package itself is ordinary NumPy/PyTorch/
PySide6 code and should work anywhere those install cleanly.

### NixOS (the supported path)

```bash
cd gpu_pde_lab
nix develop
```

The flake's `devShell`:

- provides `python3.11` (with the nixpkgs-built `PySide6` wired up with its
  compiled Qt bindings, which is the fiddly part to get right on NixOS),
  CUDA toolkit/cuDNN libraries, and `gcc.cc.lib` (`libstdc++`) on
  `LD_LIBRARY_PATH` so pip-installed PyTorch's CUDA wheels can find the
  shared libraries they expect;
- creates (once) a `.venv` with `--system-site-packages` so it can see the
  nix-provided `PySide6`, and activates it automatically;
- forces `QT_QPA_PLATFORM=xcb`. On this machine (Hyprland/Wayland), the
  nixpkgs PySide6/Qt build reliably aborts (`SIGABRT`, no error message) on
  the native Wayland backend; XWayland (`xcb`) works correctly. If your
  Wayland setup doesn't have this problem, unset the variable after
  entering the shell.
- sets `allowUnfree = true` in-flake (scoped to this project only, not your
  system config) because PyTorch's CUDA wheels carry an unfree/redistributable
  license tag.

First entry into the shell installs the remaining Python dependencies into
the venv (PySide6 itself comes from the nix-provided system site-packages
above, not pip, so it's intentionally left out of this line):

```bash
pip install torch numpy pyqtgraph pytest matplotlib
pip install -e .
```

(Only needs to be done once — the venv persists across `nix develop` entries.)

### Other platforms

Any Python 3.11+ environment works. The project installs in three pieces, so
the solver core comes without Qt:

```bash
pip install -e .            # core: core/, pdes/, kernels/, benchmarking/
pip install -e ".[lab]"     # + the desktop app (PySide6, pyqtgraph)
pip install -e ".[bench]"   # + figures and benchmark plots (matplotlib)
pip install -e ".[all]"     # everything
```

The CUDA build of PyTorch is optional. Without a GPU the app runs CPU-only; it
just has no CUDA backend to compare against, and the device-specific
experiments say so rather than failing.

### Docker

```bash
docker build -t gpu-pde-lab .
docker run --rm --gpus all gpu-pde-lab pytest -q
```

Needs `nvidia-container-toolkit` for the GPU. The image is verified rather
than merely written: the suite runs inside it unaided at 131 passed / 31
skipped, the skips being the CUDA tests correctly declining a GPU-less
container.

Rootless podman works too and needs no daemon, which makes it the easier
option on NixOS. It is stricter than docker in two ways that are not
properties of the Dockerfile — it refuses unqualified image names, and it
requires a signature policy — so it needs two extra flags. The Dockerfile's
header has the exact invocation.

## 3. Running it

```bash
python -m gpu_pde_lab.app
```

For a headless CPU-vs-GPU benchmark sweep without the GUI (useful for
recording results, or running over SSH without a display):

```bash
python scripts/run_benchmark.py --sizes 128 256 512 1024 2048 4096 --out benchmark.png
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

### Editing initial conditions

Heat, Wave, Poisson, and the 2D Fluid simulation each have an **"Edit IC"**
toggle in the live-page toolbar. Turning it on swaps the heatmap for a
brush/shape editor over the field(s) that PDE exposes (`u(x,y,0)` for Heat;
displacement *and* velocity for Wave; the source term `f` for Poisson, not an
initial guess for `u`; velocity components and dye independently for Fluid —
never merged into one buffer). Paint with the brush, stamp circles/rects/
lines, or apply a preset (Gaussian, multiple sources, hot/cold, impulse,
gradient); `Undo`/`Redo` and `Clear` work per-field. Edits only exist in a
staged host-side buffer — nothing reaches the live simulation (or the GPU)
until you click **"Apply edits & Reset"**, so moving the mouse around never
triggers a device transfer or a reset. **"Save Config..."**/**"Load
Config..."** persist the edited fields plus grid/boundary metadata to a
single `.npz` file (plain NumPy arrays + a JSON blob, no pickle) via
`core/config_io.py`.

## 4. CUDA requirements

- An NVIDIA GPU with a working driver (checked here against an RTX 2080 Ti,
  CUDA 13.2 driver, PyTorch `2.14.1+cu130`).
- No specific CUDA Toolkit version is required for *running* the app — the
  pip-installed PyTorch wheel bundles its own CUDA runtime. The flake's
  `cudaPackages.cudatoolkit`/`cudnn` are only on `LD_LIBRARY_PATH` to satisfy
  dynamic-linking expectations of manylinux wheels on NixOS, not to provide
  the actual CUDA runtime PyTorch uses.
- If no CUDA device is detected, every experiment still runs CPU-only;
  GPU-specific comparisons (CPU vs GPU, transfer, memory) report that no
  CUDA device is available instead of crashing.

## 5. Architecture

```
src/gpu_pde_lab/
  core/            grid (Grid2D + Grid3D), field, backend (NumPy / Torch-CPU / Torch-CUDA),
                   boundary conditions (2D and 3D), the shared 5-point and 7-point stencils,
                   bilinear interpolation (for advection),
                   field_edit.py (brush/stamp primitives + undo/redo for the IC editor),
                   config_io.py (save/load for edited fields + PDE params, no pickle)
  pdes/            HeatEquation, HeatEquation3D, WaveEquation, PoissonProblem, StableFluids2D --
                   each owns its own state and a step()/iteration method
  initial_conditions.py   point/Gaussian/multi-source/hot-cold/gradient generators, shared across PDEs
  benchmarking/    CUDA-safe timing, memory estimation/accounting, bandwidth estimation,
                   CPU-vs-GPU sweep, transfer breakdown, kernel comparison, convergence study
  experiments/     thin UI-facing wrappers: own parameter state + controls + explanation text,
                   delegate all actual numerics to core/ and pdes/
  ui/              PySide6 main window, reusable control widgets, pyqtgraph-based
                   heatmap/line-plot/bar-chart views, the interactive field_editor_widget,
                   slice_viewer_3d (three orthogonal slices + shared color scale) --
                   no numerics live here
  app.py           wires experiments into the main window and starts the Qt event loop
scripts/run_benchmark.py   headless CLI benchmark entry point
tests/             numerical correctness, backend agreement, and solver/viz-separation regression tests
```

**Backend abstraction.** `core/backend.py` defines a `Backend` class wrapping
either NumPy or PyTorch (CPU or CUDA) behind one small method set (`zeros`,
`array`, `to_numpy`, `sum`, `synchronize`, memory queries, …). Every PDE
model is written once against this interface and runs unmodified on any
backend — `HeatEquation(grid, Backend(BackendSpec("torch","cuda","fp32")), ...)`
and the identical call with `"numpy","cpu","fp64"` execute the exact same
Python code path, differing only in which array library actually does the
arithmetic. This is what makes the CPU vs GPU and precision experiments
meaningful comparisons rather than two different implementations.

**Solver/visualization separation.** PDE classes (`pdes/*.py`) have no
import of PySide6, pyqtgraph, or Qt anywhere. Experiments (`experiments/*.py`)
hold UI-facing parameter state and translate user input into solver calls,
but still perform no array math themselves beyond calling into `core`/`pdes`.
The UI layer (`ui/*.py`) only reads already-computed NumPy arrays for
display. `tests/test_regression_separation.py` enforces the boundary
explicitly: it asserts that viewing a field (`HeatmapView.set_field`,
`experiment.field_numpy()`) never mutates live solver state, and that
re-reading a field for display never advances the simulation.

## 6. Mathematical background

### Heat equation

$$ \frac{\partial u}{\partial t} = \alpha \nabla^2 u, \qquad
\nabla^2 u_{i,j} \approx \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2} $$

Explicit (forward Euler) time integration: $u^{n+1} = u^n + \Delta t\,\alpha\nabla^2 u^n$.
Stability (von Neumann analysis): $\Delta t \le h^2/(4\alpha)$.

### Wave equation

$$ \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u - \gamma\frac{\partial u}{\partial t} $$

Leapfrog (central-difference-in-time) integration. CFL stability bound:
$\Delta t \le h/(c\sqrt2)$ — linear in $h$, unlike the heat equation's
quadratic bound.

### Poisson equation

$$ -\nabla^2 u = f \;\Longrightarrow\; Au = b \text{ (matrix-free: applying } A \text{ means applying the stencil)} $$

Solved iteratively via Jacobi, (sequential and red-black) Gauss-Seidel, and
Conjugate Gradient — see each method's docstring in `pdes/poisson.py` for
the update formula and a discussion of why Jacobi parallelizes trivially
while naive Gauss-Seidel does not (red-black reordering restores parallelism
at the cost of two half-sweeps).

### 2D incompressible fluid (Stable Fluids, Stam 1999)

$$ \frac{\partial \mathbf u}{\partial t} = -(\mathbf u\cdot\nabla)\mathbf u + \nu\nabla^2\mathbf u - \nabla p, \qquad \nabla\cdot\mathbf u = 0 $$

Solved by operator splitting into stages that each reuse a primitive from
the other experiments: implicit diffusion via Jacobi relaxation, pressure
projection via a Poisson solve (also Jacobi), and semi-Lagrangian advection
via backward-tracing + bilinear interpolation (unconditionally stable, no
CFL restriction — see `core/interp.py` and `pdes/fluid.py`).

### Convergence

The convergence experiment uses an initial condition that is an eigenfunction
of the discrete Laplacian, so an exact closed-form solution exists:

$$ u_0 = \sin(\pi x/L)\sin(\pi y/L) \;\Rightarrow\; u(x,y,t) = e^{-2\alpha\pi^2 t/L^2}\sin(\pi x/L)\sin(\pi y/L) $$

allowing direct measurement of spatial error (expected order 2) and temporal
error (expected order 1, until it saturates at the fixed spatial-discretization
floor of whatever grid is used for the test — see `benchmarking/convergence.py`).

## 7. Numerical stability

Explicit schemes used here (heat FTCS, wave leapfrog) are only *conditionally*
stable: there is a maximum stable $\Delta t$ for a given grid spacing and
PDE parameters, derived via von Neumann/CFL analysis and exposed as
`dt_max()` on each PDE class. The Heat and Wave experiments let you
deliberately exceed this bound ("Allow unstable dt") to watch oscillation
build into divergence — this is a real property of the discretization, not
an implementation bug, and running the same unstable configuration on a
faster GPU does not fix it (it just reaches `inf` sooner).

## 8. Performance benchmarking

`benchmarking/timer.py` always calls `Backend.synchronize()` (a no-op on
CPU, `torch.cuda.synchronize()` on CUDA) immediately before starting and
stopping a timed region — CUDA kernel launches are asynchronous, so without
this a timer measures "time to enqueue work," not "time for the GPU to
finish it."

`benchmarking/bandwidth.py` estimates effective memory bandwidth
($\text{bytes moved} / \text{runtime}$) for the heat-equation stencil,
illustrating that stencil PDE solvers are typically memory-bandwidth-bound,
not FLOP-bound.

Its accounting is a conceptual model — read `u`, write the Laplacian, read it
back, write the new `u`, so 4 array-sized transfers — and that model turns out
to be about **8× optimistic**. In eager mode every intermediate in
`(north + south - 2*center)/dx² + (east + west - 2*center)/dy²` is its own
full-sized tensor with its own round trip to memory. The measured figure is
**~33 array passes per step**, counted at the dispatch layer by
`scripts/count_kernels.py` rather than by hand.

### 8.1 The roofline

`benchmarking/roofline.py` answers the question the above raises: given the GPU,
how much of it is this code actually using? It measures four implementations of
the identical heat step (`src/gpu_pde_lab/kernels/`) against two ceilings — the
card's spec-sheet peak derived from its reported memory clock and bus width, and
an empirically measured device-to-device `memcpy` at the same array size, which
is the honest ceiling for a read-once-write-once kernel.

On this RTX 2080 Ti (peak 616 GB/s), at fp32 with Neumann boundaries:

| Implementation | kernels/step | array passes | best speedup | peak % of memcpy ceiling |
|---|---|---|---|---|
| `baseline` (eager slices) | 17 | ~33 | 1.00× | 7% |
| `cuda_graph` (captured, replayed) | 18 | ~33 | 5.03× | 13% |
| `compiled` (`torch.compile`) | 4 | fused | 3.25× | 23% |
| `triton` (hand-written fused) | **1** | **2** | **12.91×** | **93%** |

The two optimized implementations fix different bottlenecks, which is why both
are kept. `cuda_graph` removes per-launch host overhead and moves exactly as much
memory as the baseline — it wins 5× at N=128 and *loses* (0.92×) at N≥1024.
`triton` removes the memory traffic and wins everywhere, most at large N where
the field no longer fits in L2. A single wall-clock number cannot distinguish
launch-bound from bandwidth-bound, and the fixes are unrelated.

Bandwidth is reported as **useful** bytes — the one read and one write the
computation irreducibly requires — divided by measured time, so every
implementation is scored against the same numerator and waste shows up as a
lower figure rather than being credited.

Full numbers, including the raw CSV and the device they were taken on:
[`docs/results/roofline.md`](docs/results/roofline.md). Regenerate with
`python scripts/run_roofline.py`.

## 9. How to add a new PDE

1. Add a class to `pdes/your_pde.py` with:
   - construction from `(grid, backend, boundary, ...pde params)`,
   - `reset(ic=..., **kwargs)`,
   - `step(dt)` (or an iteration method, for elliptic problems),
   - `primary_field_numpy()` returning the field to visualize,
   - optionally `dt_max()` if an explicit stability bound exists.
2. Add an `experiments/your_pde_experiment.py` subclassing `LiveExperiment`
   (or `BenchmarkExperiment` for an analysis-style experiment) that owns UI
   parameter state, builds a `ControlPanel` via `build_controls()`, and
   writes an `explanation_markdown()` covering the six standard questions
   (PDE, discretization, per-cell computation, parallelism, GPU mapping,
   failure modes).
3. Register a factory in `app.py`'s `build_live_factories()` /
   `build_benchmark_factories()`.
4. Add numerical tests in `tests/` mirroring the existing PDE test files:
   at minimum, a known-analytical-case or conservation/stability check, and
   a CPU/GPU agreement test guarded by `@pytest.mark.skipif(not cuda_is_available(), ...)`.

No changes to `ui/main_window.py` are needed — it only depends on the
`LiveExperiment`/`BenchmarkExperiment` interface in `experiments/base.py`.

## 10. How to add a new backend

`core/backend.py`'s `Backend` class is the only place that needs to know
about a new array library. Add a new `EngineKind` literal, implement the
methods it defines (`zeros`, `array`, `to_numpy`, `clone`, `meshgrid`,
`linspace`, `sum`, `norm`, `max`/`min`, `abs`, `isfinite_all`, `synchronize`,
memory queries, `device_name`, `bytes_per_element`), and add the relevant
`BackendSpec` combinations to `available_backend_specs()`. Every PDE and
experiment is written against this interface already, so no other file
should need to change.

## 11. Known limitations (by design)

- The Poisson/fluid pressure solves use a fixed iteration budget for
  interactivity rather than solving to full convergence every frame — the
  residual/divergence plots make this approximation visible rather than
  hiding it.
- `Grid2D.dx`/`dy` assume `lx/(nx-1)` spacing (i.e. the last grid point
  coincides with the domain boundary) uniformly, including for periodic
  domains, where a spectral treatment would more conventionally use
  `lx/nx`. This is a deliberate simplification; it does not affect any of
  the qualitative or convergence-order results above.
- The field editor's velocity-arrow overlay (`ui/viz/vector_overlay.py`) has
  a tested downsampling helper but isn't wired into the Fluid experiment's
  view yet -- velocity is still only inspectable via the scalar "velocity
  magnitude" display mode.
- "Save Config"/"Load Config" round-trip the edited fields plus grid
  dimensions and boundary condition, but not PDE-specific scalar parameters
  (`alpha`, `c`, `damping`, `viscosity`, `dt`, backend/precision) -- loading
  a config applies it on top of whatever those are currently set to in the
  UI, not whatever they were when the config was saved.
- 3D support covers Heat only (see §1) -- Wave/Poisson/Fluid remain 2D.
- Explanation panels render their Markdown via `QTextBrowser`, which has no
  LaTeX support: `$$ \nabla^2 u = ... $$` is shown as literal text rather
  than typeset math. A real fix (e.g. rendering via matplotlib's mathtext to
  inline images) is a worthwhile follow-up, not done here.
- Benchmark progress is binary ("Running..." / "Done.") with no intermediate
  progress indicator -- acceptable since these benchmarks run in seconds,
  but worth revisiting if a slower one is added later.

## 12. License

MIT -- see [`LICENSE`](LICENSE).
