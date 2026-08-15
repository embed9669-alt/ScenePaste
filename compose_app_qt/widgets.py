"""Shared lightweight Qt widgets for ScenePaste dialogs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


class PathRow:
    """Path line-edit + browse button used across generate / data-loop / tools."""

    def __init__(
        self,
        parent,
        value="",
        *,
        directory: bool = True,
        filter_text: str = "All files (*)",
        placeholder: str = "",
    ):
        self.edit = QLineEdit(str(value or ""))
        if placeholder:
            self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton("选择…")
        self.widget = QWidget()
        row = QHBoxLayout(self.widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)

        def choose():
            start = self.edit.text() or str(Path.cwd())
            if directory:
                picked = QFileDialog.getExistingDirectory(parent, "选择目录", start)
            else:
                picked, _ = QFileDialog.getOpenFileName(
                    parent, "选择文件", start, filter_text,
                )
            if picked:
                self.edit.setText(picked)

        self.button.clicked.connect(choose)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value) -> None:
        self.edit.setText(str(value or ""))


class CollapsibleSection(QWidget):
    """Compact disclosure section used by dense configuration dialogs."""

    def __init__(self, title: str, *, expanded: bool = False, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.clicked.connect(self._on_toggle)

        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("mutedLabel")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle) and expanded)

        self.content = QWidget()
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.content)

    def set_content_layout(self, layout) -> None:
        self.content.setLayout(layout)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self._apply_state(expanded)

    def _on_toggle(self, checked: bool) -> None:
        self._apply_state(checked)

    def _apply_state(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        self.subtitle.setVisible(bool(self.subtitle.text()) and expanded)


class MetricCard(QFrame):
    """Small status/metric card for generation dashboards."""

    def __init__(self, title: str, value: str = "—", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("metricSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)

    def set_value(self, value, subtitle: str | None = None) -> None:
        self.value_label.setText(str(value))
        if subtitle is not None:
            self.subtitle_label.setText(str(subtitle))
            self.subtitle_label.setVisible(bool(subtitle))
