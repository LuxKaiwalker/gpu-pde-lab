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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class EditSpec:
    """Describes one user-editable host field slot on a :class:`LiveExperiment`.

    ``key`` is the identifier passed to ``edit_field_numpy``/``apply_edited_fields``
    (e.g. ``"u0"`` for an initial condition, ``"f"`` for a Poisson source term,
    ``"v0"`` for an initial velocity). ``clip_to_interior`` tells the editor
    widget to restrict brush/stamp operations to ``[1:-1, 1:-1]`` because the
    boundary ring is overwritten by ``apply_boundary()`` on the next
    reset/step anyway (periodic boundaries have no such ring, so pass
    ``False`` there).
    """

    key: str
    label: str
    clip_to_interior: bool = True
    value_hint: tuple[float, float] | None = None


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

    # -- 3D volume hooks (optional) ----------------------------------------
    #
    # Default implementations mean a 2D experiment is completely unaffected:
    # `main_window.py` only swaps in the 3-orthogonal-slice viewer
    # (`ui/viz/slice_viewer_3d.py`) when `is_3d()` is True.

    def is_3d(self) -> bool:
        return False

    def volume_numpy(self) -> np.ndarray | None:
        """The full (nx, ny, nz) host array to display via the 3D slice
        viewer. Only needs overriding when `is_3d()` is True."""
        return None

    def volume_grid(self):
        """The `Grid3D` behind `volume_numpy()`. Only needs overriding when
        `is_3d()` is True."""
        return None

    # -- initial-condition / source-field editor hooks (optional) ---------
    #
    # Default implementations mean an experiment that doesn't override
    # `editable_fields()` is completely unaffected: `main_window.py` only
    # shows the editor when this returns a non-empty dict.

    def editable_fields(self) -> dict[str, EditSpec]:
        """Which host fields this experiment exposes to the field editor,
        keyed by the identifier used in `edit_field_numpy`/`apply_edited_fields`."""
        return {}

    def editor_grid(self):
        """The `Grid2D` the field editor should use to size its buffers and
        evaluate preset generators. Only needs overriding if
        `editable_fields()` is non-empty."""
        return None

    def edit_field_numpy(self, key: str) -> np.ndarray:
        """The current host-side array for editable field `key`, used to seed
        the editor buffer. Must not be a view of live solver state (same
        convention as `field_numpy()`)."""
        raise NotImplementedError(f"{type(self).__name__} has no editable field {key!r}")

    def apply_edited_fields(self, fields: dict[str, np.ndarray]) -> None:
        """Push edited host arrays (keyed as in `editable_fields()`) into the
        underlying PDE, replacing its current state. Called only on an
        explicit user action -- never from the per-tick render loop."""
        raise NotImplementedError(f"{type(self).__name__} does not support applying edited fields")


class BenchmarkExperiment(Experiment):
    kind = "benchmark"

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the benchmark/analysis and return plot-ready result data."""
