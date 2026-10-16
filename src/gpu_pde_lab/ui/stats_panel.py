"""Live numeric readout: sim time, step, grid, backend, FPS, steps/s, GPU memory, etc."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QWidget


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._labels: dict[str, QLabel] = {}

    def update_stats(self, values: dict[str, object]) -> None:
        # Remove rows for keys the *previous* experiment had but this one
        # doesn't -- without this, switching from e.g. Fluid (which reports
        # "Viscosity"/"Divergence norm") to Heat left those rows frozen at
        # their last value, displayed as if they were still part of the
        # live readout. Keys shared across experiments (Grid/Backend/Sim
        # time/...) are cheaply reused in place, as before -- this only
        # touches rows on an actual key-set change, not on every tick.
        stale_keys = [key for key in self._labels if key not in values]
        for key in stale_keys:
            self._form.removeRow(self._labels[key])
            del self._labels[key]

        for key, value in values.items():
            text = value if isinstance(value, str) else f"{value}"
            if key not in self._labels:
                label = QLabel(text)
                label.setStyleSheet("font-family: monospace;")
                self._form.addRow(QLabel(key), label)
                self._labels[key] = label
            else:
                self._labels[key].setText(text)
