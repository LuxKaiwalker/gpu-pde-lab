"""Common interface for everything selectable in the lab's experiment dropdown.

Two flavors exist, distinguished by :attr:`Experiment.kind`:

``"live"``
    A real-time stepped simulation (heat, wave, Poisson-iterative, stencil
    inspector, fluid). The UI drives it with a QTimer calling ``step()``
    repeatedly and redraws ``field_numpy()`` every tick.

``"benchmark"``
    An on-demand analysis (CPU/GPU sweep, transfer breakdown, kernel
    comparison, convergence study, precision comparison). The UI shows a
    "Run" button that calls ``run()`` once (on a worker thread) and then
    displays whatever plot/table data it returns.

Both flavors share: a title, Markdown explanation text, and a method to
build their parameter-control widgets. Keeping this interface in one place
means the main window does not need to special-case each experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class Experiment(ABC):
    kind: Literal["live", "benchmark"] = "live"
    title: str = "Experiment"

    @abstractmethod
    def explanation_markdown(self) -> str:
        """Educational text: PDE, discretization, parallelism, GPU mapping, pitfalls."""

    def build_controls(self, on_change) -> Any:
        """Return a QWidget of parameter controls. ``on_change`` is a callback
        the widget should invoke (with no arguments) whenever a parameter
        changes in a way that requires a reset(). Default: no controls."""
        return None

    def stats(self) -> dict[str, Any]:
        """Return a flat dict of label -> value for the live numeric readout."""
        return {}


class LiveExperiment(Experiment):
    kind = "live"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, dt: float) -> None: ...

    @abstractmethod
    def field_numpy(self):
        """Host-side 2D NumPy array to display as a heatmap."""

    def dt_max(self) -> float | None:
        return None

    def on_click(self, x_frac: float, y_frac: float) -> None:
        """Optional mouse-injection hook; x_frac/y_frac in [0, 1] over the field."""

    def supports_click_injection(self) -> bool:
        return False

    def extra_plot_data(self) -> dict[str, Any] | None:
        """Optional secondary plot data (e.g. residual-vs-iteration curve)."""
        return None


class BenchmarkExperiment(Experiment):
    kind = "benchmark"

    @abstractmethod
    def run(self, progress_cb=None) -> dict[str, Any]:
        """Execute the benchmark/analysis and return plot-ready result data.
        ``progress_cb``, if given, may be called with a 0..1 float."""
