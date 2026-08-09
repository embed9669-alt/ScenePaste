"""ScenePaste Dataset Explorer (PySide6).

A lightweight visual QA window for generated datasets. It uses the headless
``scenepaste.explorer`` parser/renderer so format interpretation stays testable
without Qt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from scenepaste.explorer import DatasetImage, index_dataset, render_dataset_image


class DatasetExplorerWindow(QMainWindow):
    """Browse generated images with Detect/Seg/OBB/COCO/Semantic overlays."""

    def __init__(self, dataset_root: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — Dataset Explorer")
        self.resize(1200, 780)
        self.dataset_root: Optional[Path] = None
        self.items: list[DatasetImage] = []
        self.current_index = -1
        self._pixmap: Optional[QPixmap] = None
        self._build_ui()
        self._build_toolbar()
        self._bind_shortcuts()
        if dataset_root is not None:
            self.load_dataset(Path(dataset_root))

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(245)
        self.file_list.currentRowChanged.connect(self._select_row)
        splitter.addWidget(self.file_list)

        center = QWidget()
        layout = QVBoxLayout(center)
        self.image_label = QLabel("选择一个 ScenePaste 数据集以开始浏览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background:#202225; color:#cfd3d8; padding:12px;")
        layout.addWidget(self.image_label, 1)
        self.info_label = QLabel("Detect / Seg / OBB / COCO / Semantic 将自动识别")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 920])
        self.setCentralWidget(splitter)

        status = QStatusBar(self)
        self.setStatusBar(status)
        self._status = QLabel("未加载数据集")
        status.addWidget(self._status, 1)

    def _build_toolbar(self) -> None:
        tb = QToolBar("dataset", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        open_action = QAction("📂 打开数据集", self)
        open_action.triggered.connect(self.pick_dataset)
        tb.addAction(open_action)
        tb.addSeparator()
        prev_action = QAction("◀ 上一张", self)
        prev_action.triggered.connect(lambda: self.step(-1))
        tb.addAction(prev_action)
        next_action = QAction("下一张 ▶", self)
        next_action.triggered.connect(lambda: self.step(1))
        tb.addAction(next_action)
        tb.addSeparator()
        refresh_action = QAction("↻ 刷新", self)
        refresh_action.triggered.connect(self.reload)
        tb.addAction(refresh_action)
        tb.addSeparator()
        qa_action = QAction("📊 QA Dashboard", self)
        qa_action.triggered.connect(self.open_qa_dashboard)
        tb.addAction(qa_action)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.step(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.step(1))
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.pick_dataset)
        QShortcut(QKeySequence("R"), self, activated=self.reload)

    def open_qa_dashboard(self) -> None:
        if self.dataset_root is None:
            self.pick_dataset()
            if self.dataset_root is None:
                return
        try:
            from scenepaste.tools.qa import write_qa_dashboard
            report = write_qa_dashboard(self.dataset_root)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(report["html_path"]).resolve())))
            self._status.setText(f"QA {report['health']} · {report['html_path']}")
        except Exception as exc:
            QMessageBox.warning(self, "QA Dashboard", str(exc))

    def pick_dataset(self) -> None:
        start = str(self.dataset_root or Path.cwd())
        picked = QFileDialog.getExistingDirectory(self, "选择 ScenePaste 数据集", start)
        if picked:
            self.load_dataset(Path(picked))

    def load_dataset(self, root: Path) -> None:
        root = Path(root)
        items = index_dataset(root)
        if not items:
            QMessageBox.warning(self, "无法浏览", f"未在 {root}/images/train|val|test 中找到图片。")
            return
        self.dataset_root = root
        self.items = items
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in items:
            self.file_list.addItem(f"[{item.split}] {item.path.name}")
        self.file_list.blockSignals(False)
        self.current_index = 0
        self.file_list.setCurrentRow(0)
        self._render_current()
        self._status.setText(f"{root} · {len(items)} images")

    def reload(self) -> None:
        if self.dataset_root is not None:
            old_stem = self.items[self.current_index].stem if self.items and self.current_index >= 0 else None
            root = self.dataset_root
            self.load_dataset(root)
            if old_stem:
                for i, item in enumerate(self.items):
                    if item.stem == old_stem:
                        self.file_list.setCurrentRow(i)
                        break

    def _select_row(self, row: int) -> None:
        if 0 <= row < len(self.items):
            self.current_index = row
            self._render_current()

    def step(self, delta: int) -> None:
        if not self.items:
            return
        idx = max(0, min(len(self.items) - 1, self.current_index + int(delta)))
        self.file_list.setCurrentRow(idx)

    def _render_current(self) -> None:
        if self.dataset_root is None or not (0 <= self.current_index < len(self.items)):
            return
        item = self.items[self.current_index]
        try:
            result = render_dataset_image(self.dataset_root, item)
        except Exception as exc:
            self.image_label.setText(f"预览失败：{exc}")
            return
        qimage = ImageQt(result.image.convert("RGBA"))
        self._pixmap = QPixmap.fromImage(qimage)
        self._fit_pixmap()
        note = " · ".join(result.notes)
        self.info_label.setText(
            f"{item.split}/{item.path.name}  |  {result.format_name}  |  objects: {result.object_count}"
            + (f"  |  {note}" if note else "")
        )
        self._status.setText(f"{self.current_index + 1}/{len(self.items)} · {self.dataset_root}")

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._pixmap is None:
            return
        size = self.image_label.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        self.image_label.setPixmap(self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def launch_dataset_explorer(dataset_root: Optional[Path] = None) -> int:
    """Standalone console entry used by ``scenepaste explore``."""
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = DatasetExplorerWindow(dataset_root)
    window.show()
    return int(app.exec())
