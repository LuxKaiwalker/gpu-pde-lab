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

Ten interactive experiments, selectable from a dropdown in a single desktop
app:

| Experiment | Kind | What it teaches |
|---|---|---|
| Heat Equation (+ Stability) | live | explicit FTCS diffusion; CFL-style stability bound; deliberate instability |
| Wave Equation | live | leapfrog time integration; damping; a different (linear, not quadratic) CFL bound |
| Poisson Equation (Iterative Solvers) | live | elliptic PDEs as `Au=b`; Jacobi vs Gauss-Seidel vs Conjugate Gradient |
| Stencil Visualization | live | the 5-point stencil, cell-by-cell, mapped onto "one GPU thread per cell" |
| 2D Fluid Simulation | live | operator-split incompressible Navier-Stokes (Stam's Stable Fluids) |
| CPU vs GPU Benchmark | benchmark | where GPU parallelism pays off (and where it doesn't) |
| Host ↔ Device Transfer | benchmark | why PCIe round-trips dominate wall time at small/medium problem sizes |
| GPU Memory | benchmark | why "just one small array" becomes real memory pressure with multiple buffers |
| Kernel / Vectorization Comparison | benchmark | Python loop vs NumPy vs Torch-CPU vs Torch-CUDA, same math |
| Resolution / Convergence Study | benchmark | discretization error vs `h`/`dt` against a known analytical solution |
| Precision Comparison (fp16/fp32/fp64) | benchmark | runtime/memory/error/stability tradeoffs across precisions |

Every experiment ships with an explanation panel answering: *what PDE is
this, how is it discretized, what does one grid cell compute, why can it run
in parallel, what does the GPU actually do, and what can go wrong.*

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
the venv:

```bash
pip install torch numpy scipy pyqtgraph pytest matplotlib pyopengl
pip install -e .
```

(Only needs to be done once — the venv persists across `nix develop` entries.)

### Other platforms

Any Python 3.11+ environment with PyTorch (CUDA build optional — the app
runs on CPU-only if no GPU is present, it just won't have a CUDA backend to
compare against), NumPy, SciPy, PySide6, and pyqtgraph installed will work:

```bash
pip install -e .
```

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
  core/            grid, field, backend (NumPy / Torch-CPU / Torch-CUDA), boundary conditions,
                   the shared 5-point stencil, bilinear interpolation (for advection)
  pdes/            HeatEquation, WaveEquation, PoissonProblem, StableFluids2D --
                   each owns its own state and a step()/iteration method
  initial_conditions.py   point/Gaussian/multi-source/hot-cold generators, shared across PDEs
  benchmarking/    CUDA-safe timing, memory estimation/accounting, bandwidth estimation,
                   CPU-vs-GPU sweep, transfer breakdown, kernel comparison, convergence study
  experiments/     thin UI-facing wrappers: own parameter state + controls + explanation text,
                   delegate all actual numerics to core/ and pdes/
  ui/              PySide6 main window, reusable control widgets, pyqtgraph-based
                   heatmap/line-plot/bar-chart views -- no numerics live here
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
not FLOP-bound — on this RTX 2080 Ti, the observed effective bandwidth for
the (non-fused, multi-pass) implementation here saturates around ~60 GB/s at
large grid sizes, well under the card's peak HBM bandwidth, which is itself
a teaching point about the cost of unfused elementwise kernels.

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
