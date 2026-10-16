"""The GPU PDE Lab main window: experiment selector, live/benchmark center
view, parameter controls, stats, and explanation -- the four-quadrant layout
from the project spec."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core.backend import Backend, BackendSpec, cuda_is_available
from ..core.boundary import BoundaryKind
from ..core.config_io import ConfigValidationError, ExperimentConfig, load_config, save_config
from ..experiments.base import BenchmarkExperiment, LiveExperiment
from .benchmark_worker import BenchmarkWorker
from .explanation_panel import ExplanationPanel
from .field_editor_widget import FieldEditorView
from .stats_panel import StatsPanel
from .viz.barchart import BarChartView
from .viz.heatmap import HeatmapView
from .viz.lineplot import LinePlotView
from .viz.slice_viewer_3d import SliceViewer3D


class LivePage(QWidget):
    """Center view + toolbar for a LiveExperiment: heatmap, play controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Two visually-separated rows instead of one 8-button strip: the top
        # row (simulation controls) is relevant to every live experiment; the
        # bottom row (field-editor controls) only applies to the ones with
        # editable_fields(), and is shown/hidden as one unit via
        # `self.editor_toolbar` rather than toggling 4 buttons individually.
        sim_row = QHBoxLayout()
        self.play_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        self.contour_checkbox = QPushButton("Contours: Off")
        self.contour_checkbox.setCheckable(True)
        self.speed_label = QLabel("Speed (steps/frame):")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(1)
        self.speed_value_label = QLabel("1")

        for w in (self.play_button, self.step_button, self.reset_button, self.contour_checkbox):
            sim_row.addWidget(w)
        sim_row.addWidget(self.speed_label)
        sim_row.addWidget(self.speed_slider)
        sim_row.addWidget(self.speed_value_label)
        sim_row.addStretch(1)
        layout.addLayout(sim_row)

        self.editor_toolbar = QWidget()
        editor_row = QHBoxLayout(self.editor_toolbar)
        editor_row.setContentsMargins(0, 0, 0, 0)
        self.edit_ic_button = QPushButton("Edit IC")
        self.edit_ic_button.setCheckable(True)
        self.apply_edits_button = QPushButton("Apply edits && Reset")
        self.save_config_button = QPushButton("Save Config...")
        self.load_config_button = QPushButton("Load Config...")
        for w in (self.edit_ic_button, self.apply_edits_button, self.save_config_button,
                   self.load_config_button):
            editor_row.addWidget(w)
        editor_row.addStretch(1)
        layout.addWidget(self.editor_toolbar)

        splitter = QSplitter(Qt.Vertical)
        self.canvas_stack = QStackedWidget()
        self.heatmap = HeatmapView()
        self.field_editor = FieldEditorView()
        self.slice_viewer_3d = SliceViewer3D()
        self.canvas_stack.addWidget(self.heatmap)
        self.canvas_stack.addWidget(self.field_editor)
        self.canvas_stack.addWidget(self.slice_viewer_3d)
        splitter.addWidget(self.canvas_stack)
        self.extra_plot = LinePlotView(title="Residual vs iteration", x_label="iteration",
                                        y_label="residual norm", log_y=True)
        splitter.addWidget(self.extra_plot)
        splitter.setSizes([700, 200])
        layout.addWidget(splitter)

        self.speed_slider.valueChanged.connect(lambda v: self.speed_value_label.setText(str(v)))
        self.editor_toolbar.setVisible(False)


class BenchmarkPage(QWidget):
    """Center view for a BenchmarkExperiment: a Run button plus dynamically
    generated plots and a Markdown summary/table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.run_button = QPushButton("Run benchmark")
        self.status_label = QLabel("")
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.status_label)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        self.summary_browser = QTextBrowser()
        self.summary_browser.setMaximumHeight(220)
        self.content_layout.addWidget(self.summary_browser)

        self._plot_widgets: list[QWidget] = []

    def clear_plots(self) -> None:
        for w in self._plot_widgets:
            self.content_layout.removeWidget(w)
            w.deleteLater()
        self._plot_widgets = []

    def show_result(self, result: dict) -> None:
        self.clear_plots()
        self.summary_browser.setMarkdown(result.get("summary", ""))

        for spec in result.get("line_plots", []):
            plot = LinePlotView(title=spec.get("title", ""), x_label=spec.get("x_label", ""),
                                 y_label=spec.get("y_label", ""), log_x=spec.get("log_x", False),
                                 log_y=spec.get("log_y", False))
            for name, (x, y) in spec.get("series", {}).items():
                plot.plot_series(name, x, y)
            plot.setMinimumHeight(260)
            self.content_layout.addWidget(plot)
            self._plot_widgets.append(plot)

        for spec in result.get("bar_charts", []):
            bar = BarChartView(title=spec.get("title", ""), y_label=spec.get("y_label", ""),
                                log_y=spec.get("log_y", False))
            bar.set_bars(spec.get("categories", []), spec.get("values", []))
            bar.setMinimumHeight(260)
            self.content_layout.addWidget(bar)
            self._plot_widgets.append(bar)


class MainWindow(QMainWindow):
    def __init__(self, live_factories: dict[str, callable], benchmark_factories: dict[str, callable]):
        super().__init__()
        self.setWindowTitle("GPU PDE Lab")
        self.resize(1400, 900)
        # Without an explicit floor, the sidebar/toolbar have no minimum
        # width and the window can shrink until every label and toolbar
        # button is truncated to a handful of letters.
        self.setMinimumSize(1100, 700)

        self.live_factories = live_factories
        self.benchmark_factories = benchmark_factories
        self.current_experiment = None
        self.current_controls_widget: QWidget | None = None
        self._worker: BenchmarkWorker | None = None

        self._build_ui()
        self._populate_experiment_list()

        self.timer = QTimer(self)
        self.timer.setInterval(16)  # ~60 Hz UI tick
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()
        self.running = True

        self.select_experiment(self._first_selectable_index)

    # -- UI construction -----------------------------------------------

    @staticmethod
    def _gpu_status_text() -> str:
        # Computed once at startup (not per-frame): CUDA availability and the
        # active device are otherwise only visible by opening a specific
        # experiment's stats panel, with no at-a-glance indicator anywhere
        # else in the window.
        if cuda_is_available():
            name = Backend(BackendSpec("torch", "cuda", "fp32")).device_name()
            return f"GPU: {name}"
        return "GPU: not available (CPU only)"

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>GPU PDE Lab</b>"))
        header.addStretch(1)
        self.gpu_status_label = QLabel(self._gpu_status_text())
        self.gpu_status_label.setStyleSheet("font-family: monospace;")
        header.addWidget(self.gpu_status_label)
        outer.addLayout(header)
        outer.addWidget(self._hline())

        root = QHBoxLayout()
        outer.addLayout(root)

        # Left sidebar: experiment selector + controls + stats
        from PySide6.QtWidgets import QComboBox

        left = QWidget()
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Experiment</b>"))
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self.select_experiment)
        left_layout.addWidget(self.experiment_combo)

        self.controls_container = QVBoxLayout()
        left_layout.addLayout(self.controls_container)

        left_layout.addWidget(self._hline())
        left_layout.addWidget(QLabel("<b>Live stats</b>"))
        self.stats_panel = StatsPanel()
        left_layout.addWidget(self.stats_panel)
        left_layout.addStretch(1)
        root.addWidget(left)

        # Center: stacked live/benchmark pages
        self.stack = QStackedWidget()
        self.live_page = LivePage()
        self.benchmark_page = BenchmarkPage()
        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.benchmark_page)
        root.addWidget(self.stack, stretch=3)

        # Right: explanation panel
        right = QWidget()
        right.setMaximumWidth(420)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Mathematics &amp; GPU computation</b>"))
        self.explanation_panel = ExplanationPanel()
        right_layout.addWidget(self.explanation_panel)
        root.addWidget(right)

        # wire live-page toolbar
        self.live_page.play_button.clicked.connect(self._toggle_play)
        self.live_page.step_button.clicked.connect(self._single_step)
        self.live_page.reset_button.clicked.connect(self._reset_experiment)
        self.live_page.contour_checkbox.toggled.connect(self._toggle_contours)
        self.live_page.heatmap.clicked_fraction.connect(self._on_heatmap_click)
        self.live_page.edit_ic_button.toggled.connect(self._toggle_edit_mode)
        self.live_page.apply_edits_button.clicked.connect(self._apply_edited_fields)
        self.live_page.save_config_button.clicked.connect(self._save_config)
        self.live_page.load_config_button.clicked.connect(self._load_config)

        # wire benchmark-page toolbar
        self.benchmark_page.run_button.clicked.connect(self._run_benchmark)

    def _hline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        return line

    # Groups experiments into the categories the UI/UX audit recommended
    # (PDE simulations vs. GPU experiments vs. numerical analysis) instead of
    # one flat list distinguished only by a "[Live]"/"[Benchmark]" text
    # prefix. Named explicitly rather than derived from `kind` so related
    # benchmark experiments (e.g. "GPU Memory" vs. "Convergence Study") land
    # in different, more meaningful groups. Any experiment not named here
    # (e.g. one added later without updating this list) still appears, under
    # an "Other" group, rather than silently vanishing from the menu.
    _CATEGORY_ORDER = [
        ("PDE Simulations", [
            "Heat Equation (+ Stability)", "3D Heat Equation", "Wave Equation",
            "Poisson Equation (Iterative Solvers)", "2D Fluid Simulation",
            "Stencil Visualization",
        ]),
        ("GPU Experiments", [
            "CPU vs GPU Benchmark", "Host <-> Device Transfer", "GPU Memory",
            "Kernel / Vectorization Comparison",
        ]),
        ("Numerical Analysis", [
            "Resolution / Convergence Study", "Precision Comparison",
        ]),
    ]

    def _add_category_header(self, label: str) -> None:
        self.experiment_combo.addItem(f"— {label} —")
        item = self.experiment_combo.model().item(self.experiment_combo.count() - 1)
        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self._combo_entries.append(None)

    def _add_experiment_row(self, kind: str, name: str) -> None:
        prefix = "[Live]" if kind == "live" else "[Benchmark]"
        self.experiment_combo.addItem(f"{prefix} {name}")
        self._combo_entries.append((kind, name))

    def _populate_experiment_list(self) -> None:
        # `_combo_entries[i]` is `(kind, name)` for a selectable row or
        # `None` for a (disabled) category-header row, kept in lockstep with
        # the combo box's own indices. Signals are blocked while populating
        # so `select_experiment` is never invoked against a partially-built
        # list -- the initial selection is made once, explicitly, afterward.
        self._combo_entries: list[tuple[str, str] | None] = []
        self.experiment_combo.blockSignals(True)

        placed: set[str] = set()
        for category, names in self._CATEGORY_ORDER:
            rows = [n for n in names if n in self.live_factories or n in self.benchmark_factories]
            if not rows:
                continue
            self._add_category_header(category)
            for name in rows:
                kind = "live" if name in self.live_factories else "benchmark"
                self._add_experiment_row(kind, name)
                placed.add(name)

        leftover_live = [n for n in self.live_factories if n not in placed]
        leftover_benchmark = [n for n in self.benchmark_factories if n not in placed]
        if leftover_live or leftover_benchmark:
            self._add_category_header("Other")
            for name in leftover_live:
                self._add_experiment_row("live", name)
            for name in leftover_benchmark:
                self._add_experiment_row("benchmark", name)

        self._first_selectable_index = next(
            i for i, entry in enumerate(self._combo_entries) if entry is not None
        )
        self.experiment_combo.setCurrentIndex(self._first_selectable_index)
        self.experiment_combo.blockSignals(False)

    # -- experiment switching --------------------------------------------

    def select_experiment(self, index: int) -> None:
        if index < 0 or index >= len(self._combo_entries):
            return
        entry = self._combo_entries[index]
        if entry is None:
            return  # a category-header row; disabled in the UI, but guard anyway
        kind, name = entry

        if kind == "live":
            self.current_experiment = self.live_factories[name]()
            self.stack.setCurrentWidget(self.live_page)
            self.running = True
            self.live_page.play_button.setText("Pause")
        else:
            self.current_experiment = self.benchmark_factories[name]()
            self.stack.setCurrentWidget(self.benchmark_page)
            self.benchmark_page.summary_browser.setMarkdown("Click **Run benchmark** to execute.")
            self.benchmark_page.clear_plots()
            # Live Stats is only ever populated from the live-tick path
            # (`_refresh_live_view`), which never runs for a benchmark
            # experiment -- without this, the previous live experiment's
            # stats stay frozen on screen, looking like they still apply.
            self.stats_panel.update_stats({})

        self.explanation_panel.set_markdown(self.current_experiment.explanation_markdown())
        self._rebuild_controls()

        if isinstance(self.current_experiment, LiveExperiment):
            self.live_page.heatmap.set_contours_enabled(False)
            self.live_page.contour_checkbox.setChecked(False)
            self.live_page.contour_checkbox.setText("Contours: Off")
            self.live_page.edit_ic_button.setChecked(False)

            is_3d = self.current_experiment.is_3d()
            if is_3d:
                # A volumetric field has no 2D contour overlay, no field
                # editor, and no save/load in this first 3D pass -- hide
                # controls the solver/viewer can't actually support rather
                # than showing ones that would silently no-op.
                self.live_page.canvas_stack.setCurrentWidget(self.live_page.slice_viewer_3d)
                self.live_page.contour_checkbox.setVisible(False)
                self.live_page.editor_toolbar.setVisible(False)
            else:
                self.live_page.canvas_stack.setCurrentWidget(self.live_page.heatmap)
                self.live_page.contour_checkbox.setVisible(True)
                has_editor = bool(self.current_experiment.editable_fields())
                self.live_page.editor_toolbar.setVisible(has_editor)
        else:
            self.live_page.contour_checkbox.setVisible(True)
            self.live_page.editor_toolbar.setVisible(False)

    def _rebuild_controls(self) -> None:
        if self.current_controls_widget is not None:
            self.controls_container.removeWidget(self.current_controls_widget)
            # removeWidget() only detaches the widget from layout management --
            # it does NOT hide it, and deleteLater()'s actual destruction is
            # deferred to a later event-loop pass. Without this .hide(), the
            # outgoing widget stays visible at its last position, overlapping
            # the newly-added controls underneath it.
            self.current_controls_widget.hide()
            self.current_controls_widget.deleteLater()
            self.current_controls_widget = None

        def on_change():
            self._reset_experiment()

        widget = self.current_experiment.build_controls(on_change)
        if widget is not None:
            self.controls_container.addWidget(widget)
            self.current_controls_widget = widget

    # -- live experiment controls -----------------------------------------

    def _toggle_play(self) -> None:
        self.running = not self.running
        self.live_page.play_button.setText("Pause" if self.running else "Resume")

    def _single_step(self) -> None:
        if isinstance(self.current_experiment, LiveExperiment):
            self.current_experiment.step(0.0)
            self._refresh_live_view()

    def _reset_experiment(self) -> None:
        if isinstance(self.current_experiment, LiveExperiment):
            self.current_experiment.reset()
            self._refresh_live_view()

    def _toggle_contours(self, checked: bool) -> None:
        self.live_page.heatmap.set_contours_enabled(checked)
        self.live_page.contour_checkbox.setText("Contours: On" if checked else "Contours: Off")

    def _on_heatmap_click(self, x_frac: float, y_frac: float) -> None:
        exp = self.current_experiment
        if isinstance(exp, LiveExperiment) and exp.supports_click_injection():
            exp.on_click(x_frac, y_frac)

    def _toggle_edit_mode(self, checked: bool) -> None:
        exp = self.current_experiment
        if not isinstance(exp, LiveExperiment):
            return
        if checked:
            specs = exp.editable_fields()
            grid = exp.editor_grid()
            if not specs or grid is None:
                self.live_page.edit_ic_button.setChecked(False)
                return
            initial = {key: exp.edit_field_numpy(key) for key in specs}
            self.live_page.field_editor.bind_fields(specs, initial, grid)
            self.live_page.canvas_stack.setCurrentWidget(self.live_page.field_editor)
        else:
            self.live_page.canvas_stack.setCurrentWidget(self.live_page.heatmap)

    def _apply_edited_fields(self) -> None:
        exp = self.current_experiment
        if not isinstance(exp, LiveExperiment):
            return
        fields = self.live_page.field_editor.current_fields_numpy()
        exp.apply_edited_fields(fields)
        self.live_page.field_editor.mark_applied()
        self._refresh_live_view()

    def _save_config(self) -> None:
        exp = self.current_experiment
        if not isinstance(exp, LiveExperiment):
            return
        specs = exp.editable_fields()
        grid = exp.editor_grid()
        if not specs or grid is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "GPU PDE Lab Config (*.npz)")
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"

        # Pulled via the current edit buffers if the editor is open (so an
        # in-progress, not-yet-applied drawing can be saved as-is), otherwise
        # straight from the live simulation.
        if self.live_page.canvas_stack.currentWidget() is self.live_page.field_editor:
            fields = self.live_page.field_editor.current_fields_numpy()
        else:
            fields = {key: exp.edit_field_numpy(key) for key in specs}

        boundary = getattr(exp, "boundary", None)
        config = ExperimentConfig(
            pde_name=exp.title, nx=grid.nx, ny=grid.ny, lx=grid.lx, ly=grid.ly,
            boundary=boundary.value if boundary is not None else BoundaryKind.NEUMANN.value,
            params={}, fields=fields,
        )
        try:
            save_config(path, config)
        except OSError as exc:
            QMessageBox.warning(self, "Save Config Failed", str(exc))

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "GPU PDE Lab Config (*.npz)")
        if not path:
            return
        try:
            config = load_config(path)
        except ConfigValidationError as exc:
            QMessageBox.warning(self, "Load Config Failed", str(exc))
            return

        if config.pde_name not in self.live_factories:
            QMessageBox.warning(self, "Load Config Failed",
                                 f"Unknown experiment type: {config.pde_name!r}")
            return
        if config.nx != config.ny:
            QMessageBox.warning(self, "Load Config Failed", "Only square grids are supported.")
            return

        # Combo indices no longer map 1:1 onto `live_factories` positions --
        # category-header rows are interspersed -- so look the row up via
        # `_combo_entries` instead of recomputing index arithmetic here.
        index = next(
            i for i, entry in enumerate(self._combo_entries)
            if entry == ("live", config.pde_name)
        )
        self.experiment_combo.setCurrentIndex(index)  # rebuilds self.current_experiment

        exp = self.current_experiment
        specs = exp.editable_fields()
        missing = set(specs) - set(config.fields)
        if missing:
            QMessageBox.warning(self, "Load Config Failed",
                                 f"Config is missing fields: {sorted(missing)}")
            return

        exp.n = config.nx
        if hasattr(exp, "boundary"):
            try:
                exp.boundary = BoundaryKind(config.boundary)
            except ValueError:
                pass
        exp.reset()

        try:
            exp.apply_edited_fields(config.fields)
        except Exception as exc:  # surfaced to the user rather than crashing the app
            QMessageBox.warning(self, "Load Config Failed", str(exc))
            return
        self._refresh_live_view()

    def _on_tick(self) -> None:
        exp = self.current_experiment
        if not isinstance(exp, LiveExperiment):
            return
        if self.running:
            steps_per_frame = self.live_page.speed_slider.value()
            for _ in range(steps_per_frame):
                exp.step(0.0)
        self._refresh_live_view()

    def _refresh_live_view(self) -> None:
        exp = self.current_experiment
        if not isinstance(exp, LiveExperiment):
            return
        if exp.is_3d():
            self.live_page.slice_viewer_3d.set_volume(exp.volume_numpy())
        else:
            self.live_page.heatmap.set_field(exp.field_numpy())
        self.stats_panel.update_stats(exp.stats())

        extra = exp.extra_plot_data()
        if extra and "residual_history" in extra and len(extra["residual_history"]) > 0:
            hist = extra["residual_history"]
            self.live_page.extra_plot.plot_series("residual", list(range(len(hist))), hist)
            self.live_page.extra_plot.show()
        else:
            self.live_page.extra_plot.hide()

    # -- benchmark experiment controls -------------------------------------

    def _run_benchmark(self) -> None:
        exp = self.current_experiment
        if not isinstance(exp, BenchmarkExperiment):
            return
        self.benchmark_page.run_button.setEnabled(False)
        self.benchmark_page.status_label.setText("Running...")

        self._worker = BenchmarkWorker(exp)
        self._worker.finished_ok.connect(self._on_benchmark_done)
        self._worker.failed.connect(self._on_benchmark_failed)
        self._worker.start()

    def _on_benchmark_done(self, result: dict) -> None:
        self.benchmark_page.run_button.setEnabled(True)
        self.benchmark_page.status_label.setText("Done.")
        self.benchmark_page.show_result(result)

    def _on_benchmark_failed(self, message: str) -> None:
        self.benchmark_page.run_button.setEnabled(True)
        self.benchmark_page.status_label.setText(f"Error: {message}")
