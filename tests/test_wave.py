import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.pdes.wave import WaveEquation


def _make(n=48, c=1.0, damping=0.0, backend_spec=("numpy", "cpu", "fp64")):
    grid = Grid2D(nx=n, ny=n)
    backend = Backend(BackendSpec(*backend_spec))
    wave = WaveEquation(grid, backend, BoundaryKind.DIRICHLET, c=c, damping=damping)
    wave.reset(ic="gaussian", sigma=0.08)
    return wave


def test_cfl_matches_formula():
    wave = _make(n=64, c=2.0)
    h = min(wave.grid.dx, wave.grid.dy)
    assert wave.dt_max() == pytest.approx(h / (2.0 * np.sqrt(2.0)))


def test_undamped_energy_roughly_conserved():
    wave = _make(damping=0.0)
    dt = wave.dt_max() * 0.8
    e0 = wave.energy(dt)
    for _ in range(300):
        wave.step(dt)
    e1 = wave.energy(dt)
    assert abs(e1 - e0) / abs(e0) < 0.1


def test_damping_monotonically_reduces_amplitude():
    wave = _make(damping=5.0)
    dt = wave.dt_max() * 0.5
    amps = []
    for _ in range(200):
        wave.step(dt)
        amps.append(wave.max_amplitude())
    # amplitude should trend down under strong damping (allow local wiggle)
    assert amps[-1] < amps[10]


def test_unstable_run_diverges():
    wave = _make()
    dt = wave.dt_max() * 2.0
    for _ in range(100):
        wave.step(dt)
        if not np.all(np.isfinite(wave.primary_field_numpy())):
            break
    field = wave.primary_field_numpy()
    assert not np.all(np.isfinite(field)) or np.abs(field).max() > 1e3


def test_reset_from_arrays_encodes_initial_velocity():
    wave = _make(n=32)
    nx, ny = wave.grid.nx, wave.grid.ny
    u0 = np.zeros((nx, ny), dtype=np.float64)
    v0 = np.full((nx, ny), 2.5, dtype=np.float64)
    dt = 1e-3

    wave.reset_from_arrays(u0, v0, dt_for_velocity=dt)

    velocity = (wave.u - wave.u_prev) / dt
    assert np.allclose(velocity[1:-1, 1:-1], 2.5, atol=1e-9)
    assert wave.state.time == 0.0
    assert wave.state.step == 0


def test_reset_from_arrays_with_no_velocity_matches_zero_velocity_default():
    wave = _make(n=16)
    u0 = np.full((16, 16), 1.25, dtype=np.float64)
    wave.reset_from_arrays(u0, v0=None)
    assert np.allclose(wave.u, wave.u_prev)


def test_wave_experiment_editable_fields_apply_round_trip():
    from gpu_pde_lab.experiments.wave_experiment import WaveExperiment

    exp = WaveExperiment()
    exp.backend_name = "CPU"
    exp.reset()

    specs = exp.editable_fields()
    assert set(specs) == {"u0", "v0"}

    n = exp.n
    drawn_u0 = np.zeros((n, n), dtype=np.float64)
    drawn_u0[n // 2, n // 2] = 7.0
    drawn_v0 = np.full((n, n), 3.0, dtype=np.float64)
    exp.apply_edited_fields({"u0": drawn_u0, "v0": drawn_v0})

    field = exp.field_numpy()
    assert field[n // 2, n // 2] == pytest.approx(7.0)
    assert exp.pde.state.step == 0

    # dt_for_velocity defaults to dt_max() inside reset_from_arrays, so
    # reconstruct velocity using that same dt rather than an arbitrary one.
    velocity = (exp.pde.u - exp.pde.u_prev) / exp.pde.dt_max()
    assert np.allclose(velocity[1:-1, 1:-1], 3.0, atol=1e-6)


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
