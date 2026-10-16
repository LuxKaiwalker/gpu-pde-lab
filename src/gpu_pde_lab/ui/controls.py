"""Reusable parameter-control widgets: a labeled slider+spinbox pair, and a
labeled combo box. Used to build every experiment's control panel from a
short declarative list rather than hand-wiring Qt layouts each time.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FloatSlider(QWidget):
    """A slider (log or linear) paired with a spin box showing the exact value."""

    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, value: float, step: float = 0.001,
                 decimals: int = 4, log_scale: bool = False, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._log = log_scale
        self._resolution = 1000

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._resolution)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)

        layout.addWidget(self.slider, stretch=3)
        layout.addWidget(self.spin, stretch=1)

        self.slider.valueChanged.connect(self._slider_to_spin)
        self.spin.valueChanged.connect(self._spin_to_slider)
        self.set_value(value)

    def _to_slider_pos(self, value: float) -> int:
        value = min(max(value, self._min), self._max)
        if self._log:
            import math
            lo, hi = math.log(max(self._min, 1e-12)), math.log(self._max)
            frac = (math.log(max(value, 1e-12)) - lo) / (hi - lo)
        else:
            frac = (value - self._min) / (self._max - self._min)
        return int(round(frac * self._resolution))

    def _from_slider_pos(self, pos: int) -> float:
        frac = pos / self._resolution
        if self._log:
            import math
            lo, hi = math.log(max(self._min, 1e-12)), math.log(self._max)
            return math.exp(lo + frac * (hi - lo))
        return self._min + frac * (self._max - self._min)

    def _slider_to_spin(self, pos: int) -> None:
        value = self._from_slider_pos(pos)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _spin_to_slider(self, value: float) -> None:
        pos = self._to_slider_pos(value)
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        self.spin.setValue(value)


class ControlPanel(QWidget):
    """A vertical form of labeled controls, built incrementally with helper methods.

    Each `add_*` method returns the created widget so the caller can read its
    current value later (e.g. in a QTimer tick) without keeping a separate
    registry.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(4, 4, 4, 4)
        self._form = QFormLayout()
        self._outer.addLayout(self._form)
        self._outer.addStretch(1)

    def add_slider(self, label: str, minimum: float, maximum: float, value: float,
                    step: float = 0.001, decimals: int = 4, log_scale: bool = False,
                    on_change: Callable[[float], None] | None = None,
                    tooltip: str | None = None) -> FloatSlider:
        w = FloatSlider(minimum, maximum, value, step, decimals, log_scale)
        if on_change:
            w.valueChanged.connect(on_change)
        label_widget = QLabel(label)
        if tooltip:
            label_widget.setToolTip(tooltip)
            w.setToolTip(tooltip)
        self._form.addRow(label_widget, w)
        return w

    def add_combo(self, label: str, options: list[str], current: str | None = None,
                   on_change: Callable[[str], None] | None = None,
                   tooltip: str | None = None) -> QComboBox:
        w = QComboBox()
        w.addItems(options)
        if current is not None and current in options:
            w.setCurrentText(current)
        if on_change:
            w.currentTextChanged.connect(on_change)
        label_widget = QLabel(label)
        if tooltip:
            label_widget.setToolTip(tooltip)
            w.setToolTip(tooltip)
        self._form.addRow(label_widget, w)
        return w

    def add_checkbox(self, label: str, checked: bool = False,
                      on_change: Callable[[bool], None] | None = None,
                      tooltip: str | None = None) -> QCheckBox:
        w = QCheckBox()
        w.setChecked(checked)
        if on_change:
            w.toggled.connect(on_change)
        label_widget = QLabel(label)
        if tooltip:
            label_widget.setToolTip(tooltip)
            w.setToolTip(tooltip)
        self._form.addRow(label_widget, w)
        return w

    def add_radio_group(self, label: str, options: list[str], current: str,
                         on_change: Callable[[str], None] | None = None) -> dict[str, QRadioButton]:
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        buttons: dict[str, QRadioButton] = {}
        for opt in options:
            rb = QRadioButton(opt)
            rb.setChecked(opt == current)
            if on_change:
                rb.toggled.connect(lambda checked, o=opt: checked and on_change(o))
            box.addWidget(rb)
            buttons[opt] = rb
        self._form.addRow(QLabel(label), container)
        return buttons

    def add_label_row(self, label: str, value_text: str) -> QLabel:
        value_label = QLabel(value_text)
        self._form.addRow(QLabel(label), value_label)
        return value_label
