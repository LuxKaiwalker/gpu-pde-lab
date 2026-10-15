import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.pdes.heat import HeatEquation


def _make(n=48, alpha=0.1, boundary=BoundaryKind.NEUMANN, backend_spec=("numpy", "cpu", "fp64")):
    grid = Grid2D(nx=n, ny=n)
    backend = Backend(BackendSpec(*backend_spec))
    heat = HeatEquation(grid, backend, boundary, alpha=alpha)
    heat.reset(ic="gaussian")
    return heat


def test_dt_max_matches_formula():
    heat = _make(n=64, alpha=0.2)
    h2 = min(heat.grid.dx, heat.grid.dy) ** 2
    assert heat.dt_max() == pytest.approx(h2 / (4 * 0.2))


def test_stable_run_stays_bounded():
    heat = _make()
    dt = heat.dt_max() * 0.5
    u0_max = heat.primary_field_numpy().max()
    for _ in range(300):
        heat.step(dt)
    field = heat.primary_field_numpy()
    assert np.all(np.isfinite(field))
    # diffusion can only reduce the peak, never create new maxima under Neumann BCs
    assert field.max() <= u0_max + 1e-8


def test_unstable_run_diverges():
    heat = _make()
    dt = heat.dt_max() * 1.8  # well past the stability bound
    for _ in range(200):
        heat.step(dt)
        if not np.all(np.isfinite(heat.primary_field_numpy())):
            break
    field = heat.primary_field_numpy()
    assert not np.all(np.isfinite(field)) or np.abs(field).max() > 1e3


def test_neumann_conserves_total_heat():
    heat = _make(boundary=BoundaryKind.NEUMANN)
    dt = heat.dt_max() * 0.5
    total0 = heat.total_heat()
    for _ in range(100):
        heat.step(dt)
    total1 = heat.total_heat()
    assert total1 == pytest.approx(total0, rel=1e-6)


def test_dirichlet_zero_boundary_stays_zero():
    heat = _make(boundary=BoundaryKind.DIRICHLET)
    dt = heat.dt_max() * 0.5
    for _ in range(50):
        heat.step(dt)
    field = heat.primary_field_numpy()
    assert np.allclose(field[0, :], 0.0)
    assert np.allclose(field[-1, :], 0.0)
    assert np.allclose(field[:, 0], 0.0)
    assert np.allclose(field[:, -1], 0.0)


def test_reset_from_array_seeds_exact_field_and_respects_boundary():
    heat = _make(boundary=BoundaryKind.DIRICHLET)
    custom = np.full((heat.grid.nx, heat.grid.ny), 3.5, dtype=np.float64)
    heat.reset_from_array(custom)

    field = heat.primary_field_numpy()
    assert np.allclose(field[1:-1, 1:-1], 3.5)
    # Dirichlet boundary is immediately re-applied, regardless of what was
    # painted onto the edited field's ring.
    assert np.allclose(field[0, :], 0.0)
    assert np.allclose(field[-1, :], 0.0)
    assert heat.state.time == 0.0
    assert heat.state.step == 0


def test_heat_experiment_editable_fields_apply_round_trip():
    from gpu_pde_lab.experiments.heat_experiment import HeatExperiment

    exp = HeatExperiment()
    exp.backend_name = "CPU"
    exp.reset()

    specs = exp.editable_fields()
    assert "u0" in specs

    drawn = np.zeros((exp.n, exp.n), dtype=np.float64)
    drawn[exp.n // 2, exp.n // 2] = 42.0
    exp.apply_edited_fields({"u0": drawn})

    field = exp.field_numpy()
    assert field[exp.n // 2, exp.n // 2] == pytest.approx(42.0)
    assert exp.pde.state.step == 0


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_cpu_gpu_agreement():
    cpu = _make(n=48, backend_spec=("torch", "cpu", "fp64"))
    gpu = _make(n=48, backend_spec=("torch", "cuda", "fp64"))
    dt = cpu.dt_max() * 0.5
    for _ in range(100):
        cpu.step(dt)
        gpu.step(dt)
    diff = np.max(np.abs(cpu.primary_field_numpy() - gpu.primary_field_numpy()))
    assert diff < 1e-8
