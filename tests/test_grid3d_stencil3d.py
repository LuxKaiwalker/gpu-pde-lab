"""3D analogs of test_grid_and_backend.py / test_stencil.py / test_boundary.py:
Grid3D geometry, the 7-point Laplacian, and apply_boundary_3d."""

import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.boundary import BoundaryKind, apply_boundary_3d
from gpu_pde_lab.core.grid import Grid3D
from gpu_pde_lab.core.stencil import laplacian_3d


def test_grid3d_spacing_and_shape():
    grid = Grid3D(nx=11, ny=21, nz=6, lx=1.0, ly=2.0, lz=0.5)
    assert grid.dx == pytest.approx(1.0 / 10)
    assert grid.dy == pytest.approx(2.0 / 20)
    assert grid.dz == pytest.approx(0.5 / 5)
    assert grid.shape == (11, 21, 6)
    assert grid.num_cells == 11 * 21 * 6


def test_grid3d_rejects_too_small():
    with pytest.raises(ValueError):
        Grid3D(nx=2, ny=5, nz=5)


def test_laplacian_3d_of_quadratic_is_exact():
    # u = x^2 + y^2 + z^2 -> analytical Laplacian = 2 + 2 + 2 = 6 everywhere,
    # exactly reproduced by a second-order finite-difference stencil (same
    # "exact for quadratics" property the 2D test relies on).
    n = 10
    x = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = X ** 2 + Y ** 2 + Z ** 2
    h = x[1] - x[0]

    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    lap = laplacian_3d(b.array(u), h, h, h, BoundaryKind.DIRICHLET)

    interior = lap[1:-1, 1:-1, 1:-1]
    assert np.allclose(interior, 6.0, atol=1e-10)


def test_laplacian_3d_periodic_matches_nonperiodic_in_interior():
    n = 12
    rng = np.random.default_rng(0)
    u = rng.standard_normal((n, n, n))
    h = 1.0 / (n - 1)
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))

    lap_dirichlet = laplacian_3d(b.array(u), h, h, h, BoundaryKind.DIRICHLET)
    lap_periodic = laplacian_3d(b.array(u), h, h, h, BoundaryKind.PERIODIC)

    assert np.allclose(lap_dirichlet[2:-2, 2:-2, 2:-2], lap_periodic[2:-2, 2:-2, 2:-2])


def test_laplacian_3d_torch_matches_numpy():
    import torch

    n = 10
    rng = np.random.default_rng(1)
    u_host = rng.standard_normal((n, n, n))

    b_np = Backend(BackendSpec("numpy", "cpu", "fp64"))
    b_torch = Backend(BackendSpec("torch", "cpu", "fp64"))

    lap_np = laplacian_3d(b_np.array(u_host), 0.1, 0.1, 0.1, BoundaryKind.NEUMANN)
    lap_torch = laplacian_3d(b_torch.array(u_host), 0.1, 0.1, 0.1, BoundaryKind.NEUMANN)

    assert np.allclose(lap_np, lap_torch.numpy(), atol=1e-10)


def test_apply_boundary_3d_dirichlet_sets_all_six_faces():
    u = np.ones((6, 6, 6))
    apply_boundary_3d(u, BoundaryKind.DIRICHLET, dirichlet_value=0.0)
    assert np.allclose(u[0, :, :], 0.0)
    assert np.allclose(u[-1, :, :], 0.0)
    assert np.allclose(u[:, 0, :], 0.0)
    assert np.allclose(u[:, -1, :], 0.0)
    assert np.allclose(u[:, :, 0], 0.0)
    assert np.allclose(u[:, :, -1], 0.0)
    # interior untouched
    assert np.allclose(u[1:-1, 1:-1, 1:-1], 1.0)


def test_apply_boundary_3d_neumann_mirrors_first_interior_neighbor():
    rng = np.random.default_rng(2)
    u = rng.standard_normal((6, 6, 6))
    apply_boundary_3d(u, BoundaryKind.NEUMANN)
    assert np.allclose(u[0, :, :], u[1, :, :])
    assert np.allclose(u[-1, :, :], u[-2, :, :])
    assert np.allclose(u[:, 0, :], u[:, 1, :])
    assert np.allclose(u[:, -1, :], u[:, -2, :])
    assert np.allclose(u[:, :, 0], u[:, :, 1])
    assert np.allclose(u[:, :, -1], u[:, :, -2])


def test_apply_boundary_3d_periodic_is_a_no_op():
    rng = np.random.default_rng(3)
    u = rng.standard_normal((6, 6, 6))
    original = u.copy()
    result = apply_boundary_3d(u, BoundaryKind.PERIODIC)
    assert result is u
    assert np.array_equal(u, original)


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_laplacian_3d_cuda_matches_cpu():
    n = 10
    rng = np.random.default_rng(4)
    u_host = rng.standard_normal((n, n, n)).astype(np.float32)

    b_cpu = Backend(BackendSpec("torch", "cpu", "fp32"))
    b_gpu = Backend(BackendSpec("torch", "cuda", "fp32"))

    lap_cpu = laplacian_3d(b_cpu.array(u_host), 0.1, 0.1, 0.1, BoundaryKind.NEUMANN)
    lap_gpu = laplacian_3d(b_gpu.array(u_host), 0.1, 0.1, 0.1, BoundaryKind.NEUMANN)

    assert np.allclose(lap_cpu, lap_gpu.cpu().numpy(), atol=1e-4)
