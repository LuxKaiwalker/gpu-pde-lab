"""Regression test for a visually-confirmed bug: switching experiments left
the previous experiment's control-panel widget visibly stacked underneath the
new one, because `_rebuild_controls()` called `removeWidget()` + `deleteLater()`
without ever hiding the outgoing widget. `removeWidget()` only detaches a
widget from layout management -- it does not hide it -- and `deleteLater()`'s
actual destruction is deferred to a later event-loop pass, so the stale
widget stays visible (at its last position) in between."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def _first_two_selectable_indices(win) -> tuple[int, int]:
    """The combo now has disabled category-header rows interspersed
    (`_combo_entries[i] is None`), so index 0/1 aren't reliably two
    different real experiments -- find the first two selectable ones."""
    selectable = [i for i, entry in enumerate(win._combo_entries) if entry is not None]
    assert len(selectable) >= 2, "need at least two real experiments to test switching"
    return selectable[0], selectable[1]


def test_switching_experiments_hides_the_previous_controls_widget():
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())
    win.show()
    app.processEvents()
    first_index, second_index = _first_two_selectable_indices(win)

    win.select_experiment(first_index)
    app.processEvents()
    first_widget = win.current_controls_widget
    assert first_widget is not None
    assert first_widget.isVisible()

    win.select_experiment(second_index)
    app.processEvents()

    assert not first_widget.isVisible(), (
        "the outgoing experiment's control panel must be hidden, not left "
        "visible and overlapping the new one"
    )
    second_widget = win.current_controls_widget
    assert second_widget is not first_widget
    assert second_widget.isVisible()


def test_reselecting_the_same_experiment_does_not_leave_a_stale_duplicate():
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())
    win.show()
    app.processEvents()
    first_index, _ = _first_two_selectable_indices(win)

    win.select_experiment(first_index)
    first_widget = win.current_controls_widget

    win.select_experiment(first_index)  # re-select the same experiment
    app.processEvents()

    assert not first_widget.isVisible()
    assert win.current_controls_widget.isVisible()


def test_category_header_rows_are_disabled_and_unselectable():
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())
    win.show()
    app.processEvents()

    header_indices = [i for i, entry in enumerate(win._combo_entries) if entry is None]
    assert header_indices, "expected at least one category header row"
    for i in header_indices:
        item = win.experiment_combo.model().item(i)
        assert not (item.flags() & Qt.ItemIsEnabled)

    before = win.current_experiment
    win.select_experiment(header_indices[0])  # selecting a header row must no-op
    assert win.current_experiment is before


def test_every_registered_experiment_is_reachable_through_the_combo():
    """Every entry in build_live_factories()/build_benchmark_factories() must
    land in exactly one selectable combo row -- the category grouping must
    not silently drop an experiment that isn't named in _CATEGORY_ORDER."""
    from gpu_pde_lab.app import build_benchmark_factories, build_live_factories
    from gpu_pde_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(build_live_factories(), build_benchmark_factories())

    live_names = {("live", n) for n in build_live_factories()}
    benchmark_names = {("benchmark", n) for n in build_benchmark_factories()}
    selectable_entries = {e for e in win._combo_entries if e is not None}

    assert selectable_entries == live_names | benchmark_names
