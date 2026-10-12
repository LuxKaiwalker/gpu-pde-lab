"""A thin pyqtgraph PlotWidget wrapper for line/scatter plots: convergence
curves, residual-vs-iteration, runtime-vs-grid-size, speedup-vs-grid-size."""

from __future__ import annotations

import pyqtgraph as pg


class LinePlotView(pg.PlotWidget):
    def __init__(self, title: str = "", x_label: str = "", y_label: str = "",
                 log_x: bool = False, log_y: bool = False, parent=None):
        super().__init__(parent)
        self.setBackground("k")
        self.setTitle(title)
        self.setLabel("bottom", x_label)
        self.setLabel("left", y_label)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLogMode(x=log_x, y=log_y)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._colors = ["#58a6ff", "#f78166", "#3fb950", "#d29922", "#bc8cff", "#ff7b72"]

    def clear_curves(self) -> None:
        for curve in self._curves.values():
            self.removeItem(curve)
        self._curves = {}

    def plot_series(self, name: str, x, y, symbol: str = "o") -> None:
        color = self._colors[len(self._curves) % len(self._colors)]
        if name in self._curves:
            self.removeItem(self._curves[name])
        curve = self.plot(x, y, pen=pg.mkPen(color, width=2), symbol=symbol,
                           symbolBrush=color, symbolSize=6, name=name)
        self._curves[name] = curve

    def set_log_mode(self, log_x: bool, log_y: bool) -> None:
        self.setLogMode(x=log_x, y=log_y)
