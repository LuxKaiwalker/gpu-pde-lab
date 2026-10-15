"""Headless tests for the backend-independent field-editing primitives in
``core/field_edit.py``. No Qt involved -- these only need plain NumPy arrays."""

import numpy as np

from gpu_pde_lab.core.field_edit import (
    EditableField,
    paint_brush,
    stamp_circle,
    stamp_line,
    stamp_rect,
)
from gpu_pde_lab.initial_conditions import gradient


def test_paint_brush_constant_falloff_only_touches_disc():
    field = np.zeros((20, 20), dtype=np.float32)
    paint_brush(field, cx=10, cy=10, radius=3, strength=2.0, mode="add", falloff="constant")

    assert field[10, 10] == 2.0  # center fully painted
    assert field[10, 13] == 2.0  # exactly on the radius boundary
    assert field[0, 0] == 0.0  # far corner untouched
    assert field[19, 19] == 0.0


def test_paint_brush_gaussian_falloff_decays_with_distance():
    field = np.zeros((40, 40), dtype=np.float32)
    paint_brush(field, cx=20, cy=20, radius=8, strength=1.0, mode="add", falloff="gaussian")

    center = field[20, 20]
    near = field[22, 20]
    far = field[0, 0]
    assert center > near > 0.0
    assert far == 0.0  # outside the brush's bounding box entirely


def test_paint_brush_erase_subtracts():
    field = np.full((10, 10), 5.0, dtype=np.float32)
    paint_brush(field, cx=5, cy=5, radius=2, strength=1.0, mode="erase", falloff="constant")
    assert field[5, 5] == 4.0
    assert field[0, 0] == 5.0


def test_stamp_circle_hard_edge():
    field = np.zeros((20, 20), dtype=np.float32)
    stamp_circle(field, cx=10, cy=10, radius=3, value=7.0, mode="set")
    assert field[10, 10] == 7.0
    assert field[0, 0] == 0.0


def test_stamp_rect_exact_region_and_clipping():
    field = np.zeros((10, 10), dtype=np.float32)
    stamp_rect(field, x0=2, y0=2, x1=4, y1=4, value=3.0, mode="set")
    assert np.all(field[2:5, 2:5] == 3.0)
    assert field[1, 2] == 0.0
    assert field[5, 2] == 0.0

    # corners given in reverse order should behave identically
    field2 = np.zeros((10, 10), dtype=np.float32)
    stamp_rect(field2, x0=4, y0=4, x1=2, y1=2, value=3.0, mode="set")
    assert np.array_equal(field, field2)

    # out-of-bounds rect clips instead of raising
    field3 = np.zeros((10, 10), dtype=np.float32)
    stamp_rect(field3, x0=-5, y0=-5, x1=2, y1=2, value=1.0, mode="set")
    assert field3[0, 0] == 1.0


def test_stamp_line_touches_segment_not_far_points():
    field = np.zeros((20, 20), dtype=np.float32)
    stamp_line(field, x0=2, y0=2, x1=17, y1=2, thickness=1.0, value=9.0, mode="set")
    assert field[10, 2] == 9.0  # midpoint of the segment
    assert field[10, 15] == 0.0  # far from the segment


def test_stamp_set_mode_blends_by_weight():
    field = np.full((10, 10), 1.0, dtype=np.float32)
    stamp_circle(field, cx=5, cy=5, radius=2, value=0.0, mode="set")
    assert field[5, 5] == 0.0  # inside disc: fully replaced
    assert field[0, 0] == 1.0  # outside: untouched


def test_editable_field_undo_redo_round_trip():
    ef = EditableField("u0", shape=(5, 5))
    ef.snapshot_for_undo()
    ef.data[:] = 1.0
    ef.snapshot_for_undo()
    ef.data[:] = 2.0

    assert np.all(ef.numpy() == 2.0)
    assert ef.undo()
    assert np.all(ef.numpy() == 1.0)
    assert ef.undo()
    assert np.all(ef.numpy() == 0.0)
    assert not ef.undo()  # stack exhausted

    assert ef.redo()
    assert np.all(ef.numpy() == 1.0)
    assert ef.redo()
    assert np.all(ef.numpy() == 2.0)
    assert not ef.redo()


def test_editable_field_new_edit_clears_redo_stack():
    ef = EditableField("u0", shape=(3, 3))
    ef.snapshot_for_undo()
    ef.data[:] = 1.0
    ef.undo()
    assert ef.can_redo()

    ef.snapshot_for_undo()
    ef.data[:] = 5.0
    assert not ef.can_redo()


def test_editable_field_undo_stack_is_bounded():
    ef = EditableField("u0", shape=(3, 3), undo_depth=4)
    for i in range(20):
        ef.snapshot_for_undo()
        ef.data[:] = float(i)
    assert len(ef._undo) <= 4


def test_editable_field_numpy_is_a_copy_not_a_view():
    ef = EditableField("u0", shape=(4, 4))
    snapshot = ef.numpy()
    snapshot[:] = 999.0
    assert not np.any(ef.numpy() == 999.0)


def test_gradient_generator_spans_lo_to_hi_along_axis():
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11), indexing="ij")
    field = gradient(X, Y, angle_deg=0.0, lo=-1.0, hi=1.0)
    assert np.isclose(field.min(), -1.0, atol=1e-6)
    assert np.isclose(field.max(), 1.0, atol=1e-6)
    # along +x (axis 0 of the "ij"-indexed meshgrid), values should increase
    assert np.all(np.diff(field[:, 0]) >= 0.0)
