import numpy as np

from gpu_pde_lab.core.backend import Backend, BackendSpec
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.pdes.fluid import StableFluids2D


def _make(n=48, boundary=BoundaryKind.NEUMANN):
    grid = Grid2D(nx=n, ny=n)
    backend = Backend(BackendSpec("torch", "cpu", "fp64"))
    fluid = StableFluids2D(grid, backend, boundary, viscosity=0.0001,
                            diffusion_iters=10, pressure_iters=20)
    return fluid


def test_fluid_stays_finite_under_forcing():
    fluid = _make()
    fluid.add_source(0.3, 0.5, fx=2.0, fy=0.5, dye_amount=1.0, radius=0.05)
    for _ in range(60):
        fluid.step(0.01)
    assert np.all(np.isfinite(fluid.dye_numpy()))
    assert np.all(np.isfinite(fluid.velocity_magnitude_numpy()))


def test_projection_reduces_divergence():
    fluid = _make()
    fluid.add_source(0.5, 0.5, fx=3.0, fy=1.0, dye_amount=1.0, radius=0.05)
    div_before = fluid.divergence_norm()
    fluid._project()
    div_after = fluid.divergence_norm()
    assert div_after < div_before


def test_dye_mass_roughly_conserved_without_sources():
    fluid = _make()
    fluid.add_source(0.4, 0.5, fx=1.0, fy=0.0, dye_amount=1.0, radius=0.05)
    for _ in range(5):
        fluid.step(0.01)
    mass0 = fluid.dye_numpy().sum()
    for _ in range(50):
        fluid.step(0.01)  # no further sources added
    mass1 = fluid.dye_numpy().sum()
    # semi-Lagrangian advection is not perfectly mass-conserving (some
    # numerical diffusion from interpolation), but should not grow or decay
    # by an order of magnitude over 50 steps.
    assert 0.5 * mass0 < mass1 < 1.5 * mass0


def test_periodic_boundary_does_not_crash():
    fluid = _make(boundary=BoundaryKind.PERIODIC)
    fluid.add_source(0.5, 0.5, fx=2.0, fy=2.0, dye_amount=1.0, radius=0.05)
    for _ in range(30):
        fluid.step(0.01)
    assert np.all(np.isfinite(fluid.dye_numpy()))


def test_set_fields_only_touches_provided_slots():
    fluid = _make()
    fluid.add_source(0.3, 0.5, fx=1.0, fy=0.5, dye_amount=1.0, radius=0.05)
    u_before = fluid.backend.to_numpy(fluid.u).copy()
    v_before = fluid.backend.to_numpy(fluid.v).copy()

    n = fluid.grid.nx
    new_dye = np.full((n, n), 7.0, dtype=np.float64)
    fluid.set_fields(dye=new_dye)

    assert np.array_equal(fluid.backend.to_numpy(fluid.u), u_before)
    assert np.array_equal(fluid.backend.to_numpy(fluid.v), v_before)
    dye = fluid.dye_numpy()
    assert np.allclose(dye[1:-1, 1:-1], 7.0)
    assert fluid.step_count == 0
    assert fluid.time == 0.0


def test_fluid_experiment_editable_fields_apply_round_trip():
    from gpu_pde_lab.experiments.fluid_experiment import FluidExperiment

    exp = FluidExperiment()
    exp.backend_name = "CPU"
    exp.reset()

    specs = exp.editable_fields()
    assert set(specs) == {"u0", "v0", "dye0"}

    n = exp.n
    u0 = np.full((n, n), 2.0, dtype=np.float64)
    v0 = np.full((n, n), -1.0, dtype=np.float64)
    dye0 = np.full((n, n), 3.0, dtype=np.float64)
    exp.apply_edited_fields({"u0": u0, "v0": v0, "dye0": dye0})

    assert exp.fluid.step_count == 0
    dye = exp.fluid.dye_numpy()
    assert np.allclose(dye[1:-1, 1:-1], 3.0)
