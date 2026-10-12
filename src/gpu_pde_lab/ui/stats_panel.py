"""Live numeric readout: sim time, step, grid, backend, FPS, steps/s, GPU memory, etc."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QWidget


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._labels: dict[str, QLabel] = {}

    def update_stats(self, values: dict[str, object]) -> None:
        for key, value in values.items():
            text = value if isinstance(value, str) else f"{value}"
            if key not in self._labels:
                label = QLabel(text)
                label.setStyleSheet("font-family: monospace;")
                self._form.addRow(QLabel(key), label)
                self._labels[key] = label
            else:
                self._labels[key].setText(text)
