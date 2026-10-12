"""2D scalar field heatmap with optional contour overlay and click injection."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal


class HeatmapView(pg.GraphicsLayoutWidget):
    """Displays a 2D NumPy array as a false-color heatmap using pyqtgraph's
    ImageItem, with optional contour (isocurve) overlay and click-to-inject
    support for mouse-driven initial conditions.
    """

    clicked_fraction = Signal(float, float)  # (x_frac, y_frac) in [0, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.colorbar = pg.ColorBarItem(colorMap="viridis")
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot)

        self._contour_items: list[pg.IsocurveItem] = []
        self.show_contours = False
        self.n_contours = 6

        self.image_item.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._last_shape = (1, 1)

    def set_colormap(self, name: str) -> None:
        self.colorbar.setColorMap(pg.colormap.get(name))

    def set_field(self, data: np.ndarray, autoscale: bool = True) -> None:
        self._last_shape = data.shape
        self.image_item.setImage(data, autoLevels=False)
        if autoscale:
            lo, hi = float(np.min(data)), float(np.max(data))
            if hi <= lo:
                hi = lo + 1e-9
            self.colorbar.setLevels((lo, hi))
        self._update_contours(data)

    def set_contours_enabled(self, enabled: bool) -> None:
        self.show_contours = enabled
        if not enabled:
            for item in self._contour_items:
                self.plot.removeItem(item)
            self._contour_items = []

    def _update_contours(self, data: np.ndarray) -> None:
        for item in self._contour_items:
            self.plot.removeItem(item)
        self._contour_items = []
        if not self.show_contours:
            return
        lo, hi = float(np.min(data)), float(np.max(data))
        if hi <= lo:
            return
        levels = np.linspace(lo, hi, self.n_contours + 2)[1:-1]
        for level in levels:
            curve = pg.IsocurveItem(data=data, level=float(level), pen=pg.mkPen("w", width=1))
            self.plot.addItem(curve)
            self._contour_items.append(curve)

    def _on_mouse_clicked(self, event) -> None:
        pos = event.scenePos()
        view_pos = self.image_item.mapFromScene(pos)
        nx, ny = self._last_shape
        x_frac = view_pos.x() / max(nx, 1)
        y_frac = view_pos.y() / max(ny, 1)
        if 0.0 <= x_frac <= 1.0 and 0.0 <= y_frac <= 1.0:
            self.clicked_fraction.emit(float(x_frac), float(y_frac))
