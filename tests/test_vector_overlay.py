import numpy as np

from gpu_pde_lab.ui.viz.vector_overlay import build_quiver_segments


def test_build_quiver_segments_shape_and_stride():
    n = 32
    u = np.ones((n, n))
    v = np.zeros((n, n))
    x, y = build_quiver_segments(u, v, stride=8)

    n_samples = len(np.arange(0, n, 8)) ** 2
    assert x.shape == (2 * n_samples,)
    assert y.shape == (2 * n_samples,)
    # each pair is (start, end) of one shaft; with v=0, u=1, dx=1, dy=0
    assert np.allclose(x[1::2] - x[0::2], 1.0)
    assert np.allclose(y[1::2] - y[0::2], 0.0)


def test_build_quiver_segments_rejects_shape_mismatch():
    u = np.zeros((4, 4))
    v = np.zeros((5, 5))
    try:
        build_quiver_segments(u, v)
        assert False, "expected ValueError"
    except ValueError:
        pass
