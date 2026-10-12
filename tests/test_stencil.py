"""The 5-point Laplacian should reproduce the analytical Laplacian of a
quadratic exactly (up to floating point), since quadratics have zero
discretization error on a uniform grid."""

import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec
from gpu_pde_lab.core.boundary import BoundaryKind
from gpu_pde_lab.core.stencil import laplacian, stencil_at


def test_laplacian_of_quadratic_is_exact():
    n = 20
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = X ** 2 + Y ** 2  # analytical Laplacian = 2 + 2 = 4 everywhere

    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    lap = laplacian(b.array(u), x[1] - x[0], x[1] - x[0], BoundaryKind.DIRICHLET)

    interior = lap[1:-1, 1:-1]
    assert np.allclose(interior, 4.0, atol=1e-10)


def test_laplacian_periodic_matches_nonperiodic_in_interior():
    n = 16
    rng = np.random.default_rng(0)
    u = rng.standard_normal((n, n))
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))

    lap_dirichlet = laplacian(b.array(u), 1.0 / (n - 1), 1.0 / (n - 1), BoundaryKind.DIRICHLET)
    lap_periodic = laplacian(b.array(u), 1.0 / (n - 1), 1.0 / (n - 1), BoundaryKind.PERIODIC)

    # Away from the boundary ring, periodic wrap-around shouldn't matter --
    # both should compute the identical stencil.
    assert np.allclose(lap_dirichlet[2:-2, 2:-2], lap_periodic[2:-2, 2:-2])


def test_laplacian_torch_matches_numpy():
    import torch

    n = 24
    rng = np.random.default_rng(1)
    u_host = rng.standard_normal((n, n))

    b_np = Backend(BackendSpec("numpy", "cpu", "fp64"))
    b_torch = Backend(BackendSpec("torch", "cpu", "fp64"))

    lap_np = laplacian(b_np.array(u_host), 0.1, 0.1, BoundaryKind.NEUMANN)
    lap_torch = laplacian(b_torch.array(u_host), 0.1, 0.1, BoundaryKind.NEUMANN)

    assert np.allclose(lap_np, lap_torch.numpy(), atol=1e-10)


def test_stencil_at_matches_bulk_laplacian():
    n = 12
    rng = np.random.default_rng(2)
    u = rng.standard_normal((n, n))
    dx = dy = 1.0 / (n - 1)

    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    lap = laplacian(b.array(u), dx, dy, BoundaryKind.PERIODIC)

    for (i, j) in [(0, 0), (5, 7), (11, 11)]:
        info = stencil_at(u, i, j, dx, dy, BoundaryKind.PERIODIC)
        assert info["laplacian"] == pytest.approx(lap[i, j], abs=1e-10)
