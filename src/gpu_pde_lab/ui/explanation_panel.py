"""Renders each experiment's educational Markdown explanation."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser


class ExplanationPanel(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setStyleSheet("font-size: 12px;")

    def set_markdown(self, text: str) -> None:
        self.setMarkdown(text)
