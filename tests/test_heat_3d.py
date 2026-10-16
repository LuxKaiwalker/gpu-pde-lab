import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid3D
from gpu_pde_lab.pdes.heat_3d import HeatEquation3D


def _make(n=16, alpha=0.1, boundary=BoundaryKind.NEUMANN, backend_spec=("numpy", "cpu", "fp64")):
    grid = Grid3D(nx=n, ny=n, nz=n)
    backend = Backend(BackendSpec(*backend_spec))
    heat = HeatEquation3D(grid, backend, boundary, alpha=alpha)
    heat.reset(ic="gaussian")
    return heat


def test_dt_max_matches_closed_form_bound():
    heat = _make(n=20, alpha=0.2)
    inv_h2_sum = 1.0 / heat.grid.dx ** 2 + 1.0 / heat.grid.dy ** 2 + 1.0 / heat.grid.dz ** 2
    assert heat.dt_max() == pytest.approx(1.0 / (2.0 * 0.2 * inv_h2_sum))


def test_dt_max_reduces_to_h2_over_6alpha_for_isotropic_grid():
    heat = _make(n=20, alpha=0.2)
    h = heat.grid.dx  # isotropic (lx=ly=lz=1, nx=ny=nz)
    assert heat.dt_max() == pytest.approx(h ** 2 / (6.0 * 0.2))


def test_primary_field_is_genuinely_three_dimensional():
    """Guards against "faking" a 3D experiment by extruding/broadcasting a 2D
    field: the data must actually vary along z, not just x and y."""
    heat = _make(n=16)
    field = heat.primary_field_numpy()
    assert field.shape == (16, 16, 16)
    # a centered 3D Gaussian varies along every axis -- a slice near the
    # center (z=7, close to the peak) must differ from one far from it
    # (z=1). (z=4 and z=11 would be a bad choice here: they're symmetric
    # around the center at z=7.5 and so are legitimately equal.)
    assert not np.allclose(field[:, :, 1], field[:, :, 7])


def test_stable_run_stays_bounded():
    heat = _make()
    dt = heat.dt_max() * 0.5
    u0_max = heat.primary_field_numpy().max()
    for _ in range(100):
        heat.step(dt)
    field = heat.primary_field_numpy()
    assert np.all(np.isfinite(field))
    assert field.max() <= u0_max + 1e-8


def test_unstable_run_diverges():
    heat = _make()
    dt = heat.dt_max() * 1.8
    for _ in range(200):
        heat.step(dt)
        if not np.all(np.isfinite(heat.primary_field_numpy())):
            break
    field = heat.primary_field_numpy()
    assert not np.all(np.isfinite(field)) or np.abs(field).max() > 1e3


def test_dirichlet_zero_boundary_stays_zero():
    heat = _make(boundary=BoundaryKind.DIRICHLET)
    dt = heat.dt_max() * 0.5
    for _ in range(30):
        heat.step(dt)
    field = heat.primary_field_numpy()
    assert np.allclose(field[0, :, :], 0.0)
    assert np.allclose(field[-1, :, :], 0.0)
    assert np.allclose(field[:, :, 0], 0.0)
    assert np.allclose(field[:, :, -1], 0.0)


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_cpu_gpu_agreement():
    cpu = _make(n=16, backend_spec=("torch", "cpu", "fp64"))
    gpu = _make(n=16, backend_spec=("torch", "cuda", "fp64"))
    dt = cpu.dt_max() * 0.5
    for _ in range(50):
        cpu.step(dt)
        gpu.step(dt)
    diff = np.max(np.abs(cpu.primary_field_numpy() - gpu.primary_field_numpy()))
    assert diff < 1e-8
