"""Regression tests enforcing the "visualization never touches solver state"
boundary. If one of these fails, something in the UI/viz layer has started
mutating numerical data in place, which would silently corrupt a running
simulation while looking fine on screen.
"""

import numpy as np


def test_heatmap_set_field_does_not_mutate_input():
    from PySide6.QtWidgets import QApplication

    from gpu_pde_lab.ui.viz.heatmap import HeatmapView

    app = QApplication.instance() or QApplication([])
    view = HeatmapView()

    data = np.random.default_rng(0).standard_normal((16, 16))
    original = data.copy()
    view.set_field(data)
    assert np.array_equal(data, original)


def test_field_numpy_is_a_copy_not_a_view_of_solver_state():
    from gpu_pde_lab.core.backend import Backend, BackendSpec
    from gpu_pde_lab.core.boundary import BoundaryKind
    from gpu_pde_lab.core.grid import Grid2D
    from gpu_pde_lab.pdes.heat import HeatEquation

    grid = Grid2D(nx=16, ny=16)
    backend = Backend(BackendSpec("numpy", "cpu", "fp64"))
    heat = HeatEquation(grid, backend, BoundaryKind.NEUMANN, alpha=0.1)

    snapshot = heat.primary_field_numpy()
    snapshot[:] = 999.0  # mutate the returned array
    fresh = heat.primary_field_numpy()
    assert not np.any(fresh == 999.0), "mutating a plotted snapshot corrupted live solver state"


def test_experiment_step_is_independent_of_display_calls():
    """Reading field_numpy() repeatedly (as the UI does every frame) must not
    itself advance or alter the simulation."""
    from gpu_pde_lab.experiments.heat_experiment import HeatExperiment

    exp = HeatExperiment()
    exp.step(0.0)
    field_a = exp.field_numpy()
    field_b = exp.field_numpy()  # simulate a second redundant UI redraw
    field_c = exp.field_numpy()
    assert np.array_equal(field_a, field_b)
    assert np.array_equal(field_b, field_c)
