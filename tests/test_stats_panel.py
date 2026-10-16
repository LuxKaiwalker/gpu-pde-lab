"""Regression test for a bug found via screenshot inspection: StatsPanel
accumulated stat rows across experiment switches forever, since
`update_stats()` only ever added/overwrote keys and never removed ones the
*previous* experiment had but the current one doesn't. Switching from Fluid
(reports "Viscosity", "Divergence norm") to Heat left those rows frozen at
their last value, displayed as if still part of the live readout."""

from PySide6.QtWidgets import QApplication

from gpu_pde_lab.ui.stats_panel import StatsPanel


def test_update_stats_removes_keys_not_present_in_the_new_values():
    app = QApplication.instance() or QApplication([])
    panel = StatsPanel()

    panel.update_stats({"Sim time": "0.1", "Viscosity": "1e-4", "Divergence norm": "2.9"})
    assert set(panel._labels) == {"Sim time", "Viscosity", "Divergence norm"}

    panel.update_stats({"Sim time": "0.2", "dt_max": "3.2e-5"})
    assert set(panel._labels) == {"Sim time", "dt_max"}, (
        "stale keys from the previous experiment's stats() must be removed, "
        "not left displaying frozen values"
    )
    assert panel._labels["Sim time"].text() == "0.2"


def test_update_stats_reuses_shared_keys_in_place():
    app = QApplication.instance() or QApplication([])
    panel = StatsPanel()

    panel.update_stats({"Grid": "128 x 128", "Backend": "torch/cpu/fp32"})
    grid_label = panel._labels["Grid"]

    panel.update_stats({"Grid": "256 x 256", "Backend": "torch/cuda/fp32"})
    assert panel._labels["Grid"] is grid_label  # same widget reused, not recreated
    assert grid_label.text() == "256 x 256"


def test_switching_live_experiments_does_not_leak_stats_keys():
    """End-to-end version of the bug: drive it through MainWindow exactly as
    the UI does, switching between experiments with disjoint stat keys."""
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())
    win.show()
    app.processEvents()

    live_names = list(build_live_factories())
    fluid_index = next(
        i for i, e in enumerate(win._combo_entries) if e == ("live", "2D Fluid Simulation")
    )
    heat_index = next(
        i for i, e in enumerate(win._combo_entries) if e == ("live", "Heat Equation (+ Stability)")
    )

    win.select_experiment(fluid_index)
    win._on_tick()
    assert "Viscosity" in win.stats_panel._labels

    win.select_experiment(heat_index)
    win._on_tick()
    assert "Viscosity" not in win.stats_panel._labels, (
        "Fluid-specific stat rows must not survive a switch to Heat"
    )


def test_switching_to_a_benchmark_experiment_clears_live_stats():
    """Live Stats is only ever populated from the live-tick path, which
    never runs for a BenchmarkExperiment -- switching to one must clear the
    panel, not leave the previous live experiment's stats frozen on screen."""
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())
    win.show()
    app.processEvents()

    fluid_index = next(
        i for i, e in enumerate(win._combo_entries) if e == ("live", "2D Fluid Simulation")
    )
    benchmark_index = next(
        i for i, e in enumerate(win._combo_entries) if e == ("benchmark", "CPU vs GPU Benchmark")
    )

    win.select_experiment(fluid_index)
    win._on_tick()
    assert win.stats_panel._labels  # sanity: something is actually shown

    win.select_experiment(benchmark_index)
    assert not win.stats_panel._labels, (
        "Live Stats must be cleared when switching to a benchmark experiment"
    )
