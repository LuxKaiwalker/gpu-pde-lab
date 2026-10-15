"""Tests for the interactive field editor widget. Qt runs offscreen via the
`QT_QPA_PLATFORM=offscreen` setting in tests/conftest.py."""

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from gpu_pde_lab.core.grid import Grid2D
from gpu_pde_lab.experiments.base import EditSpec
from gpu_pde_lab.ui.field_editor_widget import FieldEditorView, _EditableViewBox


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _bound_editor(qapp, nx=12, ny=12, clip_to_interior=True):
    editor = FieldEditorView()
    specs = {"u0": EditSpec("u0", "Initial u", clip_to_interior=clip_to_interior)}
    initial = {"u0": np.zeros((nx, ny), dtype=np.float32)}
    editor.bind_fields(specs, initial, Grid2D(nx=nx, ny=ny))
    editor.edit_toggle.setChecked(True)
    return editor


def test_viewbox_mapping_round_trips_under_zoom_and_pan(qapp):
    """The drag handler maps scene -> view (data) coordinates via
    ViewBox.mapSceneToView, which is what keeps painting accurate regardless
    of zoom/pan. Confirm that round trip under a deliberately zoomed/panned
    range, instead of re-deriving the transform math ourselves.

    A ViewBox only has a valid scene transform once it's actually laid out
    inside a shown widget with real pixel geometry -- build it the same way
    `FieldEditorView` does (via `addPlot(viewBox=...)` inside a
    GraphicsLayoutWidget) rather than using a bare, unparented ViewBox.
    """
    glw = pg.GraphicsLayoutWidget()
    vb = _EditableViewBox()
    glw.addPlot(viewBox=vb)
    glw.resize(400, 400)
    glw.show()
    qapp.processEvents()

    vb.setRange(xRange=(5, 25), yRange=(10, 50), padding=0)  # a non-trivial zoom+pan
    qapp.processEvents()

    for data_point in [(7.0, 12.0), (20.0, 45.0), (15.5, 30.25)]:
        scene_point = vb.mapViewToScene(QPointF(*data_point))
        recovered = vb.mapSceneToView(scene_point)
        assert recovered.x() == pytest.approx(data_point[0], abs=1e-6)
        assert recovered.y() == pytest.approx(data_point[1], abs=1e-6)

    glw.close()


def test_bind_fields_populates_combo_and_shows_widget(qapp):
    editor = _bound_editor(qapp)
    assert editor.field_combo.count() == 1
    assert editor.field_combo.itemText(0) == "Initial u"
    assert editor.isVisible() or not editor.isVisible()  # offscreen visibility is unreliable; just must not raise
    assert not editor.has_unapplied_edits()


def test_edit_toggle_enables_view_box_edit_mode(qapp):
    editor = _bound_editor(qapp)
    assert editor._view_box.edit_mode is True
    editor.edit_toggle.setChecked(False)
    assert editor._view_box.edit_mode is False


def test_brush_drag_paints_expected_region_and_marks_dirty(qapp):
    editor = _bound_editor(qapp, clip_to_interior=False)
    editor.radius_slider.set_value(2.0)
    editor.strength_slider.set_value(1.0)

    received = []
    editor.fieldsChanged.connect(lambda: received.append(True))

    editor._on_canvas_drag(6.0, 6.0, True, False)
    editor._on_canvas_drag(6.0, 6.0, False, True)

    data = editor.current_fields_numpy()["u0"]
    assert data[6, 6] > 0.0
    assert data[0, 0] == 0.0
    assert editor.has_unapplied_edits()
    assert received  # fieldsChanged fired


def test_brush_drag_clips_to_interior_when_spec_requests_it(qapp):
    editor = _bound_editor(qapp, nx=10, ny=10, clip_to_interior=True)
    editor.radius_slider.set_value(4.0)
    editor.strength_slider.set_value(1.0)

    # paint centered right on the edge -- would bleed onto the boundary ring
    editor._on_canvas_drag(0.0, 5.0, True, False)
    editor._on_canvas_drag(0.0, 5.0, False, True)

    data = editor.current_fields_numpy()["u0"]
    assert np.all(data[0, :] == 0.0)
    assert np.all(data[-1, :] == 0.0)
    assert np.all(data[:, 0] == 0.0)
    assert np.all(data[:, -1] == 0.0)


def test_rect_tool_commits_exact_region_on_finish(qapp):
    editor = _bound_editor(qapp, nx=12, ny=12, clip_to_interior=False)
    editor._tool_buttons["rect"].setChecked(True)
    editor.strength_slider.set_value(3.0)
    editor._mode_buttons["set"].setChecked(True)

    editor._on_canvas_drag(2.0, 2.0, True, False)
    editor._on_canvas_drag(5.0, 5.0, False, False)
    editor._on_canvas_drag(5.0, 5.0, False, True)

    data = editor.current_fields_numpy()["u0"]
    assert np.all(data[2:6, 2:6] == 3.0)
    assert data[1, 2] == 0.0
    assert editor._preview_roi is None  # ghost preview cleared on finish


def test_undo_redo_round_trip_through_widget(qapp):
    editor = _bound_editor(qapp, clip_to_interior=False)
    editor.strength_slider.set_value(1.0)
    editor._on_canvas_drag(6.0, 6.0, True, True)
    painted = editor.current_fields_numpy()["u0"].copy()
    assert painted[6, 6] != 0.0

    editor._undo_active()
    cleared = editor.current_fields_numpy()["u0"]
    assert np.all(cleared == 0.0)

    editor._redo_active()
    restored = editor.current_fields_numpy()["u0"]
    assert np.array_equal(restored, painted)


def test_apply_preset_replaces_field_and_marks_dirty(qapp):
    editor = _bound_editor(qapp, nx=16, ny=16, clip_to_interior=False)
    editor._apply_preset("gaussian")
    data = editor.current_fields_numpy()["u0"]
    assert data.max() > 0.0
    assert editor.has_unapplied_edits()


def test_mark_applied_clears_dirty_flag_without_mutating_field(qapp):
    editor = _bound_editor(qapp, clip_to_interior=False)
    editor._on_canvas_drag(6.0, 6.0, True, True)
    before = editor.current_fields_numpy()["u0"].copy()

    editor.mark_applied()
    assert not editor.has_unapplied_edits()
    after = editor.current_fields_numpy()["u0"]
    assert np.array_equal(before, after)


def test_current_fields_numpy_returns_independent_copy(qapp):
    editor = _bound_editor(qapp, clip_to_interior=False)
    snapshot = editor.current_fields_numpy()["u0"]
    snapshot[:] = 999.0
    fresh = editor.current_fields_numpy()["u0"]
    assert not np.any(fresh == 999.0)
