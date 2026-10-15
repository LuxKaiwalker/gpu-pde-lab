import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.pdes.poisson import PoissonProblem


def _make(n=48, backend_spec=("numpy", "cpu", "fp64")):
    grid = Grid2D(nx=n, ny=n)
    backend = Backend(BackendSpec(*backend_spec))
    p = PoissonProblem(grid, backend, BoundaryKind.DIRICHLET)
    p.set_rhs_gaussian_sources(n=2, amplitude=50.0, seed=0)
    p.reset_solution()
    return p


def test_jacobi_reduces_residual():
    p = _make()
    r0 = p.residual_norm()
    for _ in range(500):
        p.jacobi_step()
    r1 = p.residual_norm()
    assert r1 < r0


def test_redblack_gs_converges_faster_than_jacobi():
    p_j = _make()
    p_gs = _make()
    n_iters = 500
    for _ in range(n_iters):
        p_j.jacobi_step()
        p_gs.gauss_seidel_step_redblack()
    assert p_gs.residual_norm() < p_j.residual_norm()


def test_sequential_and_redblack_gs_agree():
    n = 20  # small grid -- sequential GS is a Python-level loop
    p_seq = _make(n=n)
    p_rb = _make(n=n)
    for _ in range(300):
        p_seq.gauss_seidel_step_sequential()
        p_rb.gauss_seidel_step_redblack()
    diff = np.max(np.abs(p_seq.backend.to_numpy(p_seq.u) - p_rb.backend.to_numpy(p_rb.u)))
    assert diff < 1e-3


def test_cg_converges_to_tight_tolerance():
    p = _make()
    history = p.solve_cg(max_iter=1000, tol=1e-10)
    assert history[-1] < 1e-8


def test_cg_matches_jacobi_solution_at_convergence():
    p_cg = _make()
    p_cg.solve_cg(max_iter=1000, tol=1e-10)

    p_j = _make()
    for _ in range(20000):
        p_j.jacobi_step()

    diff = np.max(np.abs(p_cg.backend.to_numpy(p_cg.u) - p_j.backend.to_numpy(p_j.u)))
    assert diff < 1e-3


def test_set_rhs_from_array_matches_exactly_and_resets_solution():
    p = _make()
    for _ in range(50):
        p.jacobi_step()
    p.residual_history.append(p.residual_norm())

    custom_f = np.random.default_rng(0).standard_normal((p.grid.nx, p.grid.ny))
    p.set_rhs_from_array(custom_f)

    assert np.allclose(p.backend.to_numpy(p.f), custom_f)
    assert p.residual_history == []
    assert np.all(p.backend.to_numpy(p.u) == 0.0)


def test_poisson_experiment_editable_fields_apply_round_trip():
    from gpu_pde_lab.experiments.poisson_experiment import PoissonExperiment

    exp = PoissonExperiment()
    exp.backend_name = "CPU"
    exp.reset()
    for _ in range(10):
        exp.step(0.0)

    specs = exp.editable_fields()
    assert "f" in specs

    drawn_f = np.zeros((exp.n, exp.n), dtype=np.float64)
    drawn_f[exp.n // 2, exp.n // 2] = 7.0
    exp.apply_edited_fields({"f": drawn_f})

    f_now = exp.problem.backend.to_numpy(exp.problem.f)
    assert f_now[exp.n // 2, exp.n // 2] == pytest.approx(7.0)
    assert exp._iteration == 0
    assert np.all(exp.problem.backend.to_numpy(exp.problem.u) == 0.0)


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_cg_cpu_gpu_agreement():
    cpu = _make(backend_spec=("torch", "cpu", "fp64"))
    gpu = _make(backend_spec=("torch", "cuda", "fp64"))
    cpu.solve_cg(max_iter=500, tol=1e-9)
    gpu.solve_cg(max_iter=500, tol=1e-9)
    diff = np.max(np.abs(cpu.backend.to_numpy(cpu.u) - gpu.backend.to_numpy(gpu.u)))
    assert diff < 1e-5
