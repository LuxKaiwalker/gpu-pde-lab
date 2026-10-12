import numpy as np

from gpu_pde_lab.core.backend import Backend, BackendSpec
from gpu_pde_lab.core.boundary import BoundaryKind, apply_boundary


def test_dirichlet_sets_edges_to_fixed_value():
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    u = b.array(np.ones((6, 6)) * 5.0)
    apply_boundary(u, BoundaryKind.DIRICHLET, dirichlet_value=2.0)
    assert np.all(u[0, :] == 2.0)
    assert np.all(u[-1, :] == 2.0)
    assert np.all(u[:, 0] == 2.0)
    assert np.all(u[:, -1] == 2.0)
    assert np.all(u[1:-1, 1:-1] == 5.0)  # interior untouched


def test_neumann_mirrors_first_interior_neighbor():
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    rng = np.random.default_rng(0)
    u = b.array(rng.standard_normal((8, 8)))
    apply_boundary(u, BoundaryKind.NEUMANN)
    assert np.allclose(u[0, :], u[1, :])
    assert np.allclose(u[-1, :], u[-2, :])
    assert np.allclose(u[:, 0], u[:, 1])
    assert np.allclose(u[:, -1], u[:, -2])


def test_periodic_is_a_no_op():
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    rng = np.random.default_rng(0)
    original = rng.standard_normal((5, 5))
    u = b.array(original.copy())
    result = apply_boundary(u, BoundaryKind.PERIODIC)
    assert np.array_equal(np.asarray(result), original)
