"""Three orthogonal 2D slices through a 3D scalar field, each with a movable
slice position and a *shared* color scale (computed once from the whole
volume, not independently per slice -- otherwise comparing slices by color
would be misleading).

Deliberately reuses :class:`~gpu_pde_lab.ui.viz.heatmap.HeatmapView` three
times rather than inventing a new rendering primitive or a heavier
volume-rendering dependency: this is the smallest widget that satisfies
"inspect a 3D field" without pretending to be more than it is. Slice
extraction is a pure staticmethod so it's testable without constructing any
Qt objects.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .heatmap import HeatmapView


class SliceViewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        views_row = QHBoxLayout()
        self.xy_view = HeatmapView()
        self.xz_view = HeatmapView()
        self.yz_view = HeatmapView()

        self.x_slider = QSlider(Qt.Horizontal)
        self.y_slider = QSlider(Qt.Horizontal)
        self.z_slider = QSlider(Qt.Horizontal)
        self.x_label = QLabel("x = 0")
        self.y_label = QLabel("y = 0")
        self.z_label = QLabel("z = 0")

        for title, view, label, slider in (
            ("YZ slice (fixed X)", self.yz_view, self.x_label, self.x_slider),
            ("XZ slice (fixed Y)", self.xz_view, self.y_label, self.y_slider),
            ("XY slice (fixed Z)", self.xy_view, self.z_label, self.z_slider),
        ):
            col = QVBoxLayout()
            col.addWidget(QLabel(f"<b>{title}</b>"))
            col.addWidget(view, stretch=1)
            col.addWidget(label)
            col.addWidget(slider)
            views_row.addLayout(col)

        layout.addLayout(views_row)

        self.x_slider.valueChanged.connect(self._refresh)
        self.y_slider.valueChanged.connect(self._refresh)
        self.z_slider.valueChanged.connect(self._refresh)

    def set_volume(self, volume: np.ndarray) -> None:
        """Display a new (nx, ny, nz) host array. Slider ranges/positions
        are only reset when the shape actually changes (e.g. grid resolution
        changed), so repeated calls during a running simulation don't reset
        the user's chosen slice position every frame."""
        nx, ny, nz = volume.shape
        for slider, n in ((self.x_slider, nx), (self.y_slider, ny), (self.z_slider, nz)):
            if slider.maximum() != n - 1:
                slider.blockSignals(True)
                slider.setRange(0, n - 1)
                slider.setValue(n // 2)
                slider.blockSignals(False)
        self._volume = volume
        self._refresh()

    @staticmethod
    def _extract_slices(volume: np.ndarray, ix: int, iy: int, iz: int):
        """Pure slicing logic, unit-testable without any Qt object."""
        yz = volume[ix, :, :]
        xz = volume[:, iy, :]
        xy = volume[:, :, iz]
        return xy, xz, yz

    def _refresh(self) -> None:
        if self._volume is None:
            return
        ix, iy, iz = self.x_slider.value(), self.y_slider.value(), self.z_slider.value()
        xy, xz, yz = self._extract_slices(self._volume, ix, iy, iz)

        lo, hi = float(self._volume.min()), float(self._volume.max())
        if hi <= lo:
            hi = lo + 1e-9
        for view, data in ((self.yz_view, yz), (self.xz_view, xz), (self.xy_view, xy)):
            view.set_field(data, autoscale=False)
            view.colorbar.setLevels((lo, hi))

        self.x_label.setText(f"x = {ix}")
        self.y_label.setText(f"y = {iy}")
        self.z_label.setText(f"z = {iz}")
