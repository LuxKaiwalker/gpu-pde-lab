"""Simple grouped bar chart for comparing a few scalar values by category
(e.g. runtime by implementation method, or memory by buffer)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


class BarChartView(pg.PlotWidget):
    def __init__(self, title: str = "", y_label: str = "", log_y: bool = False, parent=None):
        super().__init__(parent)
        self.setBackground("k")
        self.setTitle(title)
        self.setLabel("left", y_label)
        self.showGrid(y=True, alpha=0.3)
        self.setLogMode(y=log_y)
        self._colors = ["#58a6ff", "#f78166", "#3fb950", "#d29922", "#bc8cff", "#ff7b72"]
        self._items: list = []

    def set_bars(self, categories: list[str], values: list[float]) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items = []

        x = np.arange(len(categories))
        colors = [self._colors[i % len(self._colors)] for i in range(len(categories))]
        bar = pg.BarGraphItem(x=x, height=values, width=0.6, brushes=colors)
        self.addItem(bar)
        self._items.append(bar)

        axis = self.getAxis("bottom")
        axis.setTicks([[(i, c) for i, c in enumerate(categories)]])
