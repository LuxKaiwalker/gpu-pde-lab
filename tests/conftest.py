"""Shared pytest fixtures. Forces an offscreen Qt platform so UI-touching
tests (regression tests that import PySide6 widgets) can run in CI/headless
environments without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
