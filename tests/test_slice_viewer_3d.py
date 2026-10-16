"""Tests for the 3-orthogonal-slice 3D viewer. Qt runs offscreen via
tests/conftest.py's QT_QPA_PLATFORM=offscreen."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gpu_pde_lab.ui.viz.slice_viewer_3d import SliceViewer3D


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_extract_slices_matches_plain_numpy_indexing():
    volume = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(np.float64)
    xy, xz, yz = SliceViewer3D._extract_slices(volume, ix=1, iy=2, iz=3)

    assert np.array_equal(xy, volume[:, :, 3])
    assert np.array_equal(xz, volume[:, 2, :])
    assert np.array_equal(yz, volume[1, :, :])
    assert xy.shape == (4, 5)
    assert xz.shape == (4, 6)
    assert yz.shape == (5, 6)


def test_set_volume_initializes_slider_ranges_to_center(qapp):
    viewer = SliceViewer3D()
    volume = np.zeros((8, 10, 12))
    viewer.set_volume(volume)

    assert viewer.x_slider.maximum() == 7
    assert viewer.y_slider.maximum() == 9
    assert viewer.z_slider.maximum() == 11
    assert viewer.x_slider.value() == 4
    assert viewer.y_slider.value() == 5
    assert viewer.z_slider.value() == 6


def test_set_volume_does_not_reset_slider_position_on_same_shape(qapp):
    viewer = SliceViewer3D()
    volume = np.zeros((8, 8, 8))
    viewer.set_volume(volume)
    viewer.z_slider.setValue(2)

    viewer.set_volume(np.ones((8, 8, 8)))  # same shape, different data
    assert viewer.z_slider.value() == 2  # user's chosen slice position preserved


def test_moving_slider_changes_which_slice_is_displayed(qapp):
    viewer = SliceViewer3D()
    volume = np.zeros((6, 6, 6))
    volume[:, :, 0] = 1.0
    volume[:, :, 5] = 2.0
    viewer.set_volume(volume)

    viewer.z_slider.setValue(0)
    assert np.allclose(viewer.xy_view.image_item.image, 1.0)

    viewer.z_slider.setValue(5)
    assert np.allclose(viewer.xy_view.image_item.image, 2.0)


def test_color_scale_is_shared_not_independently_autoscaled(qapp):
    """Two slices with very different value ranges must still be displayed
    against the SAME color levels (derived from the whole volume), not each
    autoscaled to its own local min/max."""
    viewer = SliceViewer3D()
    volume = np.zeros((6, 6, 6))
    volume[:, :, 0] = 0.0
    volume[:, :, 5] = 100.0
    viewer.set_volume(volume)

    viewer.z_slider.setValue(0)
    levels_at_low_slice = viewer.xy_view.colorbar.levels()
    viewer.z_slider.setValue(5)
    levels_at_high_slice = viewer.xy_view.colorbar.levels()

    assert levels_at_low_slice == levels_at_high_slice
    assert levels_at_low_slice == (0.0, 100.0)
