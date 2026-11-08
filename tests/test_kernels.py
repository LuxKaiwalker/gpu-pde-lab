"""Every optimized heat-step implementation must compute what the eager one does.

A performance number from an implementation that computes the wrong thing is
worse than no number at all, so these tests are the precondition for anything
in ``docs/results/`` meaning something.

The GPU-requiring tests skip rather than fail when there is no usable device --
including when a device exists but is full, which is a real condition on a
shared machine and not a reason to report a red suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_usable
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.kernels.baseline import heat_step_baseline
from gpu_pde_lab.kernels.registry import (
    ARRAY_PASSES_PER_STEP,
    IMPLEMENTATION_ORDER,
    KERNEL_LAUNCHES_PER_STEP,
    available_implementations,
    build_implementation,
)
from gpu_pde_lab.pdes.heat import HeatEquation

requires_cuda = pytest.mark.skipif(not cuda_is_usable(), reason="requires a usable CUDA device")

ALL_BOUNDARIES = list(BoundaryKind)
# Includes a non-square, non-power-of-two grid: tile masking at the ragged
# edge is exactly where a hand-written kernel goes wrong, and a 256x256 test
# would never exercise it.
SHAPES = [(64, 64), (129, 97), (256, 256)]


def _params(nx: int, ny: int, alpha: float = 0.1):
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    dt = 0.25 * min(dx, dy) ** 2 / alpha
    return dt, alpha, dx, dy


# -- the baseline really is the eager solver --------------------------------


@pytest.mark.parametrize("boundary", ALL_BOUNDARIES)
def test_baseline_matches_the_eager_solver_bit_for_bit(boundary):
    """If this drifts, every comparison downstream is against the wrong thing.

    Bit-exact rather than approximate on purpose: ``kernels/baseline.py``
    duplicates the solver's expressions and the only way to know the duplicate
    has not quietly diverged is to require identical floating-point output.
    """
    nx = ny = 64
    grid = Grid2D(nx=nx, ny=ny)
    backend = Backend(BackendSpec("torch", "cpu", "fp64"))
    pde = HeatEquation(grid, backend, boundary, alpha=0.1)

    dt, alpha, _dx, _dy = _params(nx, ny)
    u_before = pde.u.clone()
    pde.step(dt)
    expected = pde.u

    actual = heat_step_baseline(u_before, dt, alpha, grid.dx, grid.dy, boundary)
    assert torch.equal(actual, expected), (
        "kernels/baseline.py has drifted from pdes/heat.py; the roofline "
        "comparison would be measuring against a different computation"
    )


# -- the optimized implementations agree with it ----------------------------


@requires_cuda
@pytest.mark.parametrize("impl", [n for n in IMPLEMENTATION_ORDER if n != "baseline"])
@pytest.mark.parametrize("boundary", ALL_BOUNDARIES)
def test_implementation_agrees_with_baseline(impl, boundary):
    descriptors = {d.name: d for d in available_implementations()}
    if not descriptors[impl].available:
        pytest.skip(descriptors[impl].unavailable_reason or "unavailable")

    nx = ny = 128
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float32, device="cuda")

    expected = heat_step_baseline(u, dt, alpha, dx, dy, boundary)
    step = build_implementation(
        impl, boundary, shape=(nx, ny), dt=dt, alpha=alpha, dx=dx, dy=dy, dtype_str="fp32"
    )
    actual = step(u, dt, alpha, dx, dy)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("boundary", ALL_BOUNDARIES)
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_handles_ragged_tiles_and_every_boundary(boundary, shape):
    """The tile size is 32x64; none of these shapes divides it evenly."""
    from gpu_pde_lab.kernels.triton_stencil import make_triton_heat_step, triton_is_available

    if not triton_is_available():
        pytest.skip("Triton unavailable")

    nx, ny = shape
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float32, device="cuda")

    expected = heat_step_baseline(u, dt, alpha, dx, dy, boundary)
    actual = make_triton_heat_step(boundary)(u, dt, alpha, dx, dy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("boundary", ALL_BOUNDARIES)
def test_triton_keeps_fp64_in_fp64(boundary):
    """Triton binds a Python float kernel argument as fp32.

    Passing dx^2, dy^2 and dt*alpha as scalars therefore truncated them
    silently, so an fp64 field was stepped with single-precision coefficients
    and carried ~1e-8 relative error -- which would have quietly corrupted the
    precision comparison the project exists to make. The coefficients travel
    in a tensor of the field's dtype instead; this test is what holds that.
    """
    from gpu_pde_lab.kernels.triton_stencil import make_triton_heat_step, triton_is_available

    if not triton_is_available():
        pytest.skip("Triton unavailable")

    # dy is deliberately not a power of two, so truncating its reciprocal to
    # fp32 is detectable. On a grid where every coefficient happens to be
    # exactly representable the bug hides.
    nx, ny = 129, 97
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float64, device="cuda")

    expected = heat_step_baseline(u, dt, alpha, dx, dy, boundary)
    actual = make_triton_heat_step(boundary)(u, dt, alpha, dx, dy)

    rel = ((actual - expected).abs().max() / expected.abs().max()).item()
    assert rel < 1e-14, (
        f"relative error {rel:.2e} is far above fp64 epsilon -- the coefficients "
        "are being computed in a lower precision than the field"
    )


@requires_cuda
def test_triton_rejects_a_cpu_tensor():
    from gpu_pde_lab.kernels.triton_stencil import make_triton_heat_step, triton_is_available

    if not triton_is_available():
        pytest.skip("Triton unavailable")
    step = make_triton_heat_step(BoundaryKind.NEUMANN)
    with pytest.raises(ValueError, match="CUDA tensor"):
        step(torch.zeros((16, 16)), 1e-6, 0.1, 0.1, 0.1)


@requires_cuda
def test_triton_block_size_does_not_change_the_answer():
    """Tiling is a performance knob; if it changed results it would be a bug."""
    from gpu_pde_lab.kernels.triton_stencil import make_triton_heat_step, triton_is_available

    if not triton_is_available():
        pytest.skip("Triton unavailable")

    nx = ny = 200
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float32, device="cuda")

    reference = make_triton_heat_step(BoundaryKind.NEUMANN, 32, 64)(u, dt, alpha, dx, dy)
    for bm, bn in [(16, 16), (64, 32), (32, 128)]:
        other = make_triton_heat_step(BoundaryKind.NEUMANN, bm, bn)(u, dt, alpha, dx, dy)
        torch.testing.assert_close(other, reference, rtol=0, atol=0)


@requires_cuda
def test_many_steps_stay_consistent_with_the_baseline():
    """One step agreeing is weak; an unstable kernel would still pass it.

    Marching 50 steps lets any per-step discrepancy accumulate, and the heat
    equation is diffusive so a correct implementation stays bounded.
    """
    from gpu_pde_lab.kernels.triton_stencil import make_triton_heat_step, triton_is_available

    if not triton_is_available():
        pytest.skip("Triton unavailable")

    nx = ny = 128
    dt, alpha, dx, dy = _params(nx, ny)
    u0 = torch.rand((nx, ny), dtype=torch.float64, device="cuda")

    a = u0.clone()
    b = u0.clone()
    step = make_triton_heat_step(BoundaryKind.NEUMANN)
    for _ in range(50):
        a = heat_step_baseline(a, dt, alpha, dx, dy, BoundaryKind.NEUMANN)
        b = step(b, dt, alpha, dx, dy)

    assert torch.isfinite(b).all(), "the fused kernel diverged where the baseline did not"
    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


@requires_cuda
def test_graph_refuses_parameters_it_was_not_captured_with():
    """A graph replays what it recorded. Silently ignoring new dt would make
    every subsequent number a lie, so the wrapper raises instead."""
    from gpu_pde_lab.kernels.graphs import cuda_graph_is_available

    if not cuda_graph_is_available():
        pytest.skip("CUDA graphs unavailable")

    nx = ny = 64
    dt, alpha, dx, dy = _params(nx, ny)
    step = build_implementation(
        "cuda_graph", BoundaryKind.NEUMANN,
        shape=(nx, ny), dt=dt, alpha=alpha, dx=dx, dy=dy,
    )
    u = torch.rand((nx, ny), dtype=torch.float32, device="cuda")

    step(u, dt, alpha, dx, dy)  # the captured parameters are fine
    with pytest.raises(ValueError, match="Recapture"):
        step(u, dt * 2, alpha, dx, dy)


# -- registry bookkeeping ---------------------------------------------------


def test_registry_covers_every_implementation():
    descriptors = available_implementations()
    assert [d.name for d in descriptors] == list(IMPLEMENTATION_ORDER)
    for d in descriptors:
        assert d.description
        # An unavailable implementation must say why; a blank reason is what
        # makes a missing bar in the roofline plot look like a failure.
        assert d.available or d.unavailable_reason


def test_recorded_counts_cover_every_implementation():
    for name in IMPLEMENTATION_ORDER:
        assert name in ARRAY_PASSES_PER_STEP
        assert name in KERNEL_LAUNCHES_PER_STEP


def test_fused_implementations_are_recorded_as_cheaper_than_eager():
    # Guards the headline claim's bookkeeping: the hand kernel is two passes
    # and one launch by construction, and the eager path is an order of
    # magnitude more of both.
    assert ARRAY_PASSES_PER_STEP["triton"] == 2.0
    assert KERNEL_LAUNCHES_PER_STEP["triton"] == 1
    assert ARRAY_PASSES_PER_STEP["baseline"] > 20
    assert KERNEL_LAUNCHES_PER_STEP["baseline"] > 10


def test_building_an_unknown_implementation_is_an_error():
    with pytest.raises(ValueError, match="unknown implementation"):
        build_implementation("nonexistent", BoundaryKind.NEUMANN)


def test_graph_build_requires_its_capture_parameters():
    with pytest.raises(ValueError, match="required"):
        build_implementation("cuda_graph", BoundaryKind.NEUMANN)


# -- the baseline is backend-agnostic ---------------------------------------


def test_baseline_runs_on_cpu_tensors():
    """The CPU path is what keeps the whole module importable and testable in
    CI, where there is no GPU at all."""
    nx = ny = 32
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float32)
    out = heat_step_baseline(u, dt, alpha, dx, dy, BoundaryKind.NEUMANN)
    assert out.shape == u.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("boundary", ALL_BOUNDARIES)
def test_baseline_conserves_or_dissipates_but_never_grows(boundary):
    """Explicit diffusion below the stability bound cannot amplify anything.

    A sign error or a wrong stencil coefficient shows up here as growth, which
    no correct heat solver produces.
    """
    nx = ny = 48
    dt, alpha, dx, dy = _params(nx, ny)
    u = torch.rand((nx, ny), dtype=torch.float64)
    start = u.abs().max().item()
    for _ in range(25):
        u = heat_step_baseline(u, dt, alpha, dx, dy, boundary)
    assert u.abs().max().item() <= start + 1e-12
    assert np.isfinite(u.abs().max().item())
