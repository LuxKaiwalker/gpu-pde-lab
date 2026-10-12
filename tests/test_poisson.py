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


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_cg_cpu_gpu_agreement():
    cpu = _make(backend_spec=("torch", "cpu", "fp64"))
    gpu = _make(backend_spec=("torch", "cuda", "fp64"))
    cpu.solve_cg(max_iter=500, tol=1e-9)
    gpu.solve_cg(max_iter=500, tol=1e-9)
    diff = np.max(np.abs(cpu.backend.to_numpy(cpu.u) - gpu.backend.to_numpy(gpu.u)))
    assert diff < 1e-5
