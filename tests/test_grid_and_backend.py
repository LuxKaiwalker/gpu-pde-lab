import numpy as np
import pytest

from gpu_pde_lab.core.backend import Backend, BackendSpec, cuda_is_available
from gpu_pde_lab.core.grid import Grid2D


def test_grid_spacing():
    g = Grid2D(nx=11, ny=21, lx=1.0, ly=2.0)
    assert g.dx == pytest.approx(0.1)
    assert g.dy == pytest.approx(0.1)
    assert g.num_cells == 11 * 21


def test_grid_rejects_too_small():
    with pytest.raises(ValueError):
        Grid2D(nx=2, ny=10)


def test_backend_numpy_roundtrip():
    b = Backend(BackendSpec("numpy", "cpu", "fp64"))
    arr = b.array(np.ones((4, 4)))
    assert b.to_numpy(arr).shape == (4, 4)
    assert b.sum(arr) == pytest.approx(16.0)


def test_backend_torch_cpu_roundtrip():
    b = Backend(BackendSpec("torch", "cpu", "fp32"))
    arr = b.array(np.ones((4, 4)))
    assert b.to_numpy(arr).shape == (4, 4)
    assert b.sum(arr) == pytest.approx(16.0, rel=1e-5)


@pytest.mark.skipif(not cuda_is_available(), reason="requires CUDA device")
def test_backend_cuda_roundtrip():
    b = Backend(BackendSpec("torch", "cuda", "fp32"))
    arr = b.array(np.ones((8, 8)))
    assert b.sum(arr) == pytest.approx(64.0, rel=1e-5)
    assert b.memory_allocated_bytes() is not None and b.memory_allocated_bytes() > 0
