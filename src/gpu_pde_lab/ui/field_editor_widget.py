"""Interactive 2D field editor: brush painting, geometric stamps, presets,
undo/redo, bound to whatever editable fields the current experiment exposes
via `experiments.base.EditSpec`.

This widget owns no numerics beyond what's in `core/field_edit.py` -- it only
turns mouse events into continuous grid-index coordinates and calls into that
module's pure-NumPy ops, then redraws. It never touches a `Backend`/CUDA
tensor; everything it edits is a host-side `EditableField` buffer, pushed into
the live PDE only when the host page calls `current_fields_numpy()` and the
experiment's `apply_edited_fields()` explicitly (never from the 60 Hz render
tick).

One widget class serves every PDE: which fields are editable, and whether
each one's boundary ring is paintable, comes entirely from the `EditSpec`
dict passed to `bind_fields()` -- there is no per-PDE subclass.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.field_edit import EditableField, paint_brush, stamp_circle, stamp_line, stamp_rect
from ..core.grid import Grid2D
from ..experiments.base import EditSpec
from ..initial_conditions import IC_REGISTRY
from .controls import FloatSlider
from .viz.heatmap import HeatmapView

_RING_RGBA_ON = (120, 120, 120, 150)
_PRESET_NAMES = ["gaussian", "multiple_sources", "hot_cold_regions", "point_source", "gradient"]
_PRESET_LABELS = {
    "gaussian": "Gaussian",
    "multiple_sources": "Multiple sources",
    "hot_cold_regions": "Hot / cold",
    "point_source": "Impulse",
    "gradient": "Gradient",
}


class _EditableViewBox(pg.ViewBox):
    """A ViewBox that paints (via a callback) instead of panning while
    `edit_mode` is on, and behaves exactly like a stock ViewBox (pan + wheel
    zoom) otherwise. Wheel-zoom is never touched, so zoom always works."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_mode = False
        self.on_drag: Callable[[float, float, bool, bool], None] | None = None

    def mouseDragEvent(self, ev, axis=None):
        if not self.edit_mode or self.on_drag is None or ev.button() != Qt.LeftButton:
            super().mouseDragEvent(ev, axis=axis)
            return
        ev.accept()
        pos = self.mapSceneToView(ev.scenePos())
        self.on_drag(pos.x(), pos.y(), ev.isStart(), ev.isFinish())

    def mouseClickEvent(self, ev):
        if not self.edit_mode or self.on_drag is None or ev.button() != Qt.LeftButton:
            super().mouseClickEvent(ev)
            return
        ev.accept()
        pos = self.mapSceneToView(ev.scenePos())
        self.on_drag(pos.x(), pos.y(), True, True)


def _ring_mask_rgba(nx: int, ny: int) -> np.ndarray:
    rgba = np.zeros((nx, ny, 4), dtype=np.ubyte)
    rgba[0, :] = _RING_RGBA_ON
    rgba[-1, :] = _RING_RGBA_ON
    rgba[:, 0] = _RING_RGBA_ON
    rgba[:, -1] = _RING_RGBA_ON
    return rgba


def _ghost_roi(roi: pg.ROI) -> pg.ROI:
    """Strip interactivity from an ROI so it's a pure visual preview."""
    for h in list(roi.getHandles()):
        roi.removeHandle(h)
    roi.translatable = False
    roi.setAcceptedMouseButtons(Qt.NoButton)
    roi.setPen(pg.mkPen("y", width=1, style=Qt.DashLine))
    return roi


class FieldEditorView(QWidget):
    """Brush/shape/preset editor over one or more named host fields.

    Usage: call `bind_fields(specs, initial, grid)` whenever the bound
    experiment (or its grid resolution/boundary) changes, then read back
    `current_fields_numpy()` to feed `apply_edited_fields()`.
    """

    fieldsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._specs: dict[str, EditSpec] = {}
        self._fields: dict[str, EditableField] = {}
        self._grid: Grid2D | None = None
        self._active_key: str | None = None
        self._dirty = False
        self._drag_start: tuple[float, float] | None = None
        self._preview_roi: pg.ROI | None = None

        self._view_box = _EditableViewBox()
        self._view_box.on_drag = self._on_canvas_drag
        self.heatmap = HeatmapView(view_box=self._view_box)

        self._ring_item = pg.ImageItem()
        self._ring_item.setZValue(5)
        self.heatmap.plot.addItem(self._ring_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.edit_toggle = QPushButton("Edit: Off")
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.toggled.connect(self._on_edit_toggled)
        toolbar.addWidget(self.edit_toggle)

        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self._on_field_selected)
        toolbar.addWidget(self.field_combo)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for i, name in enumerate(("brush", "circle", "rect", "line")):
            btn = QPushButton(name.capitalize())
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            self.tool_group.addButton(btn)
            toolbar.addWidget(btn)
        self._tool_buttons = {b.text().lower(): b for b in self.tool_group.buttons()}

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for i, name in enumerate(("add", "erase", "set")):
            btn = QPushButton(name.capitalize())
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            self.mode_group.addButton(btn)
            toolbar.addWidget(btn)
        self._mode_buttons = {b.text().lower(): b for b in self.mode_group.buttons()}

        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Radius / thickness"))
        self.radius_slider = FloatSlider(0.5, 60.0, 6.0, step=0.5, decimals=1)
        params_row.addWidget(self.radius_slider, stretch=1)
        params_row.addWidget(QLabel("Strength / value"))
        self.strength_slider = FloatSlider(-5.0, 5.0, 1.0, step=0.05, decimals=3)
        params_row.addWidget(self.strength_slider, stretch=1)
        layout.addLayout(params_row)

        presets_row = QHBoxLayout()
        for name in _PRESET_NAMES:
            btn = QPushButton(_PRESET_LABELS[name])
            btn.clicked.connect(lambda _checked=False, n=name: self._apply_preset(n))
            presets_row.addWidget(btn)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_active)
        presets_row.addWidget(self.clear_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo_active)
        presets_row.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo_active)
        presets_row.addWidget(self.redo_button)
        presets_row.addStretch(1)
        layout.addLayout(presets_row)

        layout.addWidget(self.heatmap, stretch=1)

        self.status_label = QLabel("Matches simulation")
        layout.addWidget(self.status_label)

        self._set_controls_visible(False)

    # -- public API ---------------------------------------------------------

    def bind_fields(self, specs: dict[str, EditSpec], initial: dict[str, np.ndarray],
                     grid: Grid2D) -> None:
        """(Re)initialize the editable buffers for the current experiment.
        Call on experiment switch and on "Reset"."""
        self._specs = dict(specs)
        self._grid = grid
        self._fields = {
            key: EditableField(key, grid.shape, initial=initial[key])
            for key, spec in specs.items()
        }
        self._active_key = next(iter(specs), None)

        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems([specs[k].label for k in specs])
        self.field_combo.blockSignals(False)

        has_fields = bool(specs)
        self.setVisible(has_fields)
        self._mark_clean()
        self._refresh_ring_overlay()
        self._refresh_display()

    def current_fields_numpy(self) -> dict[str, np.ndarray]:
        return {key: field.numpy() for key, field in self._fields.items()}

    def has_unapplied_edits(self) -> bool:
        return self._dirty

    def mark_applied(self) -> None:
        self._mark_clean()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_toggle.setChecked(enabled)

    # -- internals: selection / mode -----------------------------------------

    def _active_spec(self) -> EditSpec | None:
        if self._active_key is None:
            return None
        return self._specs.get(self._active_key)

    def _active_field(self) -> EditableField | None:
        if self._active_key is None:
            return None
        return self._fields.get(self._active_key)

    def _on_field_selected(self, label: str) -> None:
        for key, spec in self._specs.items():
            if spec.label == label:
                self._active_key = key
                break
        self._refresh_ring_overlay()
        self._refresh_display()

    def _on_edit_toggled(self, enabled: bool) -> None:
        self._view_box.edit_mode = enabled
        self.edit_toggle.setText("Edit: On" if enabled else "Edit: Off")
        self._set_controls_visible(enabled)

    def _set_controls_visible(self, visible: bool) -> None:
        for b in list(self._tool_buttons.values()) + list(self._mode_buttons.values()):
            b.setVisible(visible)
        self.radius_slider.setVisible(visible)
        self.strength_slider.setVisible(visible)

    def _current_tool(self) -> str:
        for name, btn in self._tool_buttons.items():
            if btn.isChecked():
                return name
        return "brush"

    def _current_mode(self) -> str:
        for name, btn in self._mode_buttons.items():
            if btn.isChecked():
                return name
        return "add"

    # -- internals: painting -------------------------------------------------

    def _clip_ring(self, data: np.ndarray) -> None:
        data[0, :] = 0.0
        data[-1, :] = 0.0
        data[:, 0] = 0.0
        data[:, -1] = 0.0

    def _on_canvas_drag(self, x: float, y: float, is_start: bool, is_finish: bool) -> None:
        field = self._active_field()
        spec = self._active_spec()
        if field is None or spec is None:
            return

        tool = self._current_tool()
        if is_start:
            field.snapshot_for_undo()
            self._drag_start = (x, y)

        if tool == "brush":
            paint_brush(field.data, x, y, self.radius_slider.value(), self.strength_slider.value(),
                        mode=self._current_mode(), falloff="gaussian")
            if spec.clip_to_interior:
                self._clip_ring(field.data)
            self._mark_dirty()
            self._refresh_display()
        else:
            self._update_shape_preview(tool, x, y)
            if is_finish and self._drag_start is not None:
                self._commit_shape(tool, self._drag_start, (x, y), field, spec)

        if is_finish:
            self._clear_shape_preview()
            self._drag_start = None

    def _commit_shape(self, tool: str, start: tuple[float, float], end: tuple[float, float],
                       field: EditableField, spec: EditSpec) -> None:
        x0, y0 = start
        x1, y1 = end
        value = self.strength_slider.value()
        mode = self._current_mode()
        radius = self.radius_slider.value()
        if tool == "circle":
            r = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6)
            stamp_circle(field.data, x0, y0, r, value, mode=mode)
        elif tool == "rect":
            stamp_rect(field.data, x0, y0, x1, y1, value, mode=mode)
        elif tool == "line":
            stamp_line(field.data, x0, y0, x1, y1, radius, value, mode=mode)
        if spec.clip_to_interior:
            self._clip_ring(field.data)
        self._mark_dirty()
        self._refresh_display()

    def _update_shape_preview(self, tool: str, x: float, y: float) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._clear_shape_preview()
        if tool == "circle":
            r = max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5, 1e-6)
            roi = pg.CircleROI([x0 - r, y0 - r], [2 * r, 2 * r])
        elif tool == "rect":
            roi = pg.RectROI([min(x0, x), min(y0, y)], [abs(x - x0) + 1e-6, abs(y - y0) + 1e-6])
        elif tool == "line":
            roi = pg.LineSegmentROI([[x0, y0], [x, y]])
        else:
            return
        self._preview_roi = _ghost_roi(roi)
        self.heatmap.plot.addItem(self._preview_roi)

    def _clear_shape_preview(self) -> None:
        if self._preview_roi is not None:
            self.heatmap.plot.removeItem(self._preview_roi)
            self._preview_roi = None

    # -- internals: presets / clear / undo / redo ----------------------------

    def _apply_preset(self, name: str) -> None:
        field = self._active_field()
        if field is None or self._grid is None:
            return
        grid = self._grid
        X, Y = np.meshgrid(np.linspace(0.0, grid.lx, grid.nx), np.linspace(0.0, grid.ly, grid.ny),
                            indexing="ij")
        gen = IC_REGISTRY[name]
        array = gen(X, Y).astype(field.dtype)
        spec = self._active_spec()
        if spec is not None and spec.clip_to_interior:
            self._clip_ring(array)
        field.set_data(array)
        self._mark_dirty()
        self._refresh_display()

    def _clear_active(self) -> None:
        field = self._active_field()
        if field is None:
            return
        field.clear(0.0)
        self._mark_dirty()
        self._refresh_display()

    def _undo_active(self) -> None:
        field = self._active_field()
        if field is not None and field.undo():
            self._mark_dirty()
            self._refresh_display()

    def _redo_active(self) -> None:
        field = self._active_field()
        if field is not None and field.redo():
            self._mark_dirty()
            self._refresh_display()

    # -- internals: display ---------------------------------------------------

    def _refresh_ring_overlay(self) -> None:
        spec = self._active_spec()
        if spec is None or self._grid is None:
            self._ring_item.setVisible(False)
            return
        if spec.clip_to_interior:
            self._ring_item.setImage(_ring_mask_rgba(self._grid.nx, self._grid.ny))
            self._ring_item.setVisible(True)
        else:
            self._ring_item.setVisible(False)

    def _refresh_display(self) -> None:
        field = self._active_field()
        if field is None:
            return
        self.heatmap.set_field(field.numpy())

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.status_label.setText("Unapplied edits")
        self.fieldsChanged.emit()

    def _mark_clean(self) -> None:
        self._dirty = False
        self.status_label.setText("Matches simulation")
