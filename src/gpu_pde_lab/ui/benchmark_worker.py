"""Runs a BenchmarkExperiment.run() off the UI thread so the GUI doesn't freeze."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class BenchmarkWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self.experiment = experiment

    def run(self) -> None:
        try:
            result = self.experiment.run()
            self.finished_ok.emit(result)
        except Exception as exc:  # surfaced to the user rather than crashing the app
            self.failed.emit(f"{type(exc).__name__}: {exc}")
