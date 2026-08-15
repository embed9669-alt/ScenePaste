"""Object Cutout Studio — LabelMe-style manual + rembg auto cutout.

Save results as image + sibling LabelMe JSON under the objects directory so
``load_object_assets`` can load them like any other annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QImageReader, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scenepaste.core.auto_cutout import _label_for_path
from scenepaste.core.cutout_models import (
    ensure_model,
    get_model_spec,
    is_model_ready,
    list_cutout_models,
    predict_cutout,
)
from scenepaste.core.io import imread_with_exif
from scenepaste.core.labelme import save_cutout_as_labelme
from scenepaste.core.models import IMAGE_SUFFIXES

from .theme import apply_theme, style_danger, style_primary
from .widgets import PathRow


def _labels_from_class_map(class_map_text: str) -> List[str]:
    labels: List[str] = []
    for part in (class_map_text or "").replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split("=", 1)[0].strip()
        if name and name not in labels:
            labels.append(name)
    return labels or ["object"]


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _make_file_thumb(path: Path, size: int = 72) -> QPixmap:
    """Fast scaled thumbnail for the folder browser (EXIF-aware via QImageReader)."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    src = reader.size()
    if src.isValid() and src.width() > 0 and src.height() > 0:
        w, h = int(src.width()), int(src.height())
        scale = min(float(size) / w, float(size) / h, 1.0)
        reader.setScaledSize(QSize(max(1, int(w * scale)), max(1, int(h * scale))))
    img = reader.read()
    if img.isNull():
        return QPixmap()
    pm = QPixmap.fromImage(img)
    if pm.width() > size or pm.height() > size:
        pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


class _ThumbWorker(QThread):
    """Load folder thumbnails off the UI thread."""

    ready = Signal(int, object)  # index, QPixmap

    def __init__(self, paths: Sequence[Path], size: int = 72, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._size = int(size)

    def run(self) -> None:
        for i, path in enumerate(self._paths):
            if self.isInterruptionRequested():
                break
            try:
                pm = _make_file_thumb(Path(path), self._size)
            except Exception:
                pm = QPixmap()
            if self.isInterruptionRequested():
                break
            self.ready.emit(i, pm)


class _EnsureModelWorker(QThread):
    # msg, percent (float 0..100) or None for indeterminate
    progress = Signal(str, object)
    succeeded = Signal(str)  # model_id
    failed = Signal(str)

    def __init__(self, model_id: str, parent=None, *, use_hf_mirror: bool = True):
        super().__init__(parent)
        self._model_id = model_id
        self._use_hf_mirror = use_hf_mirror
        self._discard = False

    def discard(self) -> None:
        """Soft-cancel: keep downloading in background but ignore the result."""
        self._discard = True

    def run(self) -> None:
        try:
            def _cb(msg, pct=None):
                if not self._discard:
                    self.progress.emit(str(msg), pct)

            ensure_model(
                self._model_id,
                progress=_cb,
                use_hf_mirror=self._use_hf_mirror,
            )
            if self._discard:
                return
            self.succeeded.emit(self._model_id)
        except Exception as exc:
            if not self._discard:
                self.failed.emit(str(exc))


class _CutoutWorker(QThread):
    succeeded = Signal(object, object)  # polygon, alpha
    failed = Signal(str)

    def __init__(self, bgr: np.ndarray, model_id: str, text_prompt: str, parent=None):
        super().__init__(parent)
        self._bgr = bgr
        self._model_id = model_id
        self._text_prompt = text_prompt

    def run(self) -> None:
        try:
            if not is_model_ready(self._model_id):
                ensure_model(self._model_id)
            poly, alpha = predict_cutout(
                self._bgr, self._model_id, text_prompt=self._text_prompt,
            )
            self.succeeded.emit(poly, alpha)
        except Exception as exc:
            self.failed.emit(str(exc))


class _BatchCutoutWorker(QThread):
    progress = Signal(int, int, str)  # done, total, message
    finished_ok = Signal(int, int)  # ok, failed
    failed = Signal(str)

    def __init__(
        self,
        image_paths: Sequence[Path],
        objects_dir: Path,
        out_dir: Path,
        *,
        model_id: str,
        default_label: str,
        label_from_subdir: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(image_paths)
        self._objects_dir = Path(objects_dir)
        self._out_dir = Path(out_dir)
        self._model_id = model_id
        self._default_label = default_label
        self._label_from_subdir = label_from_subdir
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            if not is_model_ready(self._model_id):
                ensure_model(self._model_id)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        ok = failed = 0
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if self._abort:
                break
            self.progress.emit(i, total, path.name)
            try:
                bgr = imread_with_exif(path)
                if bgr is None:
                    failed += 1
                    continue
                label = _label_for_path(
                    path,
                    self._objects_dir,
                    default_label=self._default_label,
                    label_from_subdir=self._label_from_subdir,
                )
                poly, _ = predict_cutout(bgr, self._model_id, text_prompt=label)
                if poly is None or len(poly) < 3:
                    failed += 1
                    continue
                rel_parent = Path(".")
                if self._label_from_subdir:
                    try:
                        rel = path.resolve().relative_to(self._objects_dir.resolve())
                        if len(rel.parts) >= 2:
                            rel_parent = Path(rel.parts[0])
                    except ValueError:
                        pass
                dest = self._out_dir / rel_parent
                save_cutout_as_labelme(path, dest, label=label, polygon=poly, bgr=bgr)
                ok += 1
            except Exception:
                failed += 1
        self.progress.emit(total, total, "完成")
        self.finished_ok.emit(ok, failed)


class _PointSegWorker(QThread):
    succeeded = Signal(object, object)  # polygon, alpha
    failed = Signal(str)

    def __init__(
        self,
        bgr: np.ndarray,
        x: float,
        y: float,
        *,
        model_id: str = "sam2_click",
        use_hf_mirror: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._bgr = bgr
        self._x = x
        self._y = y
        self._model_id = model_id
        self._use_hf_mirror = use_hf_mirror

    def run(self) -> None:
        try:
            from scenepaste.core.cutout_models import ensure_model, predict_from_point

            if not is_model_ready(self._model_id):
                ensure_model(self._model_id, use_hf_mirror=self._use_hf_mirror)
            poly, alpha = predict_from_point(
                self._bgr, self._x, self._y, model_id=self._model_id,
            )
            self.succeeded.emit(poly, alpha)
        except Exception as exc:
            self.failed.emit(str(exc))


class AnnotateView(QGraphicsView):
    """Polygon draw and/or click-to-segment on top of an image."""

    polygon_changed = Signal()
    point_prompt = Signal(float, float)  # image x, y for SAM click mode

    MODE_DRAW = "draw"
    MODE_CLICK = "click"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self._pix: Optional[QGraphicsPixmapItem] = None
        self._poly_item: Optional[QGraphicsPolygonItem] = None
        self._preview_item: Optional[QGraphicsPolygonItem] = None
        self._point_items: List[QGraphicsEllipseItem] = []
        self._click_marker: Optional[QGraphicsEllipseItem] = None
        self._points: List[QPointF] = []
        self._closed = False
        self._draw_enabled = True
        self._mode = self.MODE_DRAW
        self._img_w = 0
        self._img_h = 0
        self._cursor = QPointF()

        pen = QPen(QColor(0, 200, 120), 2)
        pen.setCosmetic(True)
        self._pen = pen
        preview = QPen(QColor(255, 180, 40), 1, Qt.DashLine)
        preview.setCosmetic(True)
        self._preview_pen = preview
        click_pen = QPen(QColor(255, 80, 80), 2)
        click_pen.setCosmetic(True)
        self._click_pen = click_pen

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in (self.MODE_DRAW, self.MODE_CLICK) else self.MODE_DRAW
        if self._mode == self.MODE_CLICK and self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def mode(self) -> str:
        return self._mode

    def clear_image(self) -> None:
        self._scene.clear()
        self._pix = None
        self._poly_item = None
        self._preview_item = None
        self._point_items = []
        self._click_marker = None
        self._points = []
        self._closed = False
        self._img_w = self._img_h = 0

    def set_image(self, bgr: np.ndarray) -> None:
        self.clear_image()
        pm = _bgr_to_pixmap(bgr)
        self._img_w, self._img_h = pm.width(), pm.height()
        self._pix = self._scene.addPixmap(pm)
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_draw_enabled(self, enabled: bool) -> None:
        self._draw_enabled = enabled

    def set_polygon(self, points: Sequence[Sequence[float]], *, closed: bool = True) -> None:
        self._clear_drawing_items()
        self._points = [QPointF(float(x), float(y)) for x, y in points]
        self._closed = closed and len(self._points) >= 3
        self._rebuild_items()
        self.polygon_changed.emit()

    def clear_polygon(self) -> None:
        self._clear_drawing_items()
        self._points = []
        self._closed = False
        self.polygon_changed.emit()

    def polygon_array(self) -> Optional[np.ndarray]:
        if len(self._points) < 3:
            return None
        return np.array([[p.x(), p.y()] for p in self._points], dtype=np.float32)

    def close_polygon(self) -> bool:
        if len(self._points) < 3:
            return False
        self._closed = True
        self._rebuild_items()
        self.polygon_changed.emit()
        return True

    def undo_point(self) -> None:
        if self._closed:
            self._closed = False
        if self._points:
            self._points.pop()
            self._rebuild_items()
            self.polygon_changed.emit()

    def _clear_drawing_items(self) -> None:
        for it in self._point_items:
            self._scene.removeItem(it)
        self._point_items = []
        if self._poly_item is not None:
            self._scene.removeItem(self._poly_item)
            self._poly_item = None
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def _rebuild_items(self) -> None:
        self._clear_drawing_items()
        if not self._points:
            return
        for p in self._points:
            ell = self._scene.addEllipse(p.x() - 3, p.y() - 3, 6, 6, self._pen, QColor(0, 200, 120, 180))
            ell.setZValue(2)
            self._point_items.append(ell)
        if len(self._points) >= 2:
            poly = QPolygonF(self._points)
            if self._closed:
                self._poly_item = self._scene.addPolygon(
                    poly, self._pen, QColor(0, 200, 120, 60),
                )
            else:
                self._poly_item = self._scene.addPolygon(poly, self._pen)
            self._poly_item.setZValue(1)
        self._update_preview()

    def _update_preview(self) -> None:
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None
        if self._closed or len(self._points) < 1 or not self._draw_enabled:
            return
        pts = list(self._points) + [self._cursor]
        self._preview_item = self._scene.addPolygon(QPolygonF(pts), self._preview_pen)
        self._preview_item.setZValue(1)

    def _clamp(self, p: QPointF) -> QPointF:
        x = min(max(p.x(), 0.0), float(max(self._img_w - 1, 0)))
        y = min(max(p.y(), 0.0), float(max(self._img_h - 1, 0)))
        return QPointF(x, y)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if self._pix is None or not self._draw_enabled:
            return super().mousePressEvent(event)
        scene_pos = self._clamp(self.mapToScene(event.position().toPoint()))
        if self._mode == self.MODE_CLICK:
            if event.button() == Qt.LeftButton:
                self._show_click_marker(scene_pos)
                self.point_prompt.emit(scene_pos.x(), scene_pos.y())
                return
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if self._closed:
                self.clear_polygon()
            self._points.append(scene_pos)
            self._closed = False
            self._rebuild_items()
            self.polygon_changed.emit()
            return
        if event.button() == Qt.RightButton:
            self.undo_point()
            return
        super().mousePressEvent(event)

    def _show_click_marker(self, p: QPointF) -> None:
        if self._click_marker is not None:
            self._scene.removeItem(self._click_marker)
        self._click_marker = self._scene.addEllipse(
            p.x() - 5, p.y() - 5, 10, 10, self._click_pen, QColor(255, 80, 80, 160),
        )
        self._click_marker.setZValue(3)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._mode == self.MODE_CLICK:
            return super().mouseDoubleClickEvent(event)
        if event.button() == Qt.LeftButton and self._draw_enabled:
            self.close_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._cursor = self._clamp(self.mapToScene(event.position().toPoint()))
        if self._mode == self.MODE_DRAW and not self._closed:
            self._update_preview()
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._mode == self.MODE_CLICK:
            if event.key() == Qt.Key_Escape:
                self.clear_polygon()
            return super().keyPressEvent(event)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.close_polygon()
            return
        if event.key() == Qt.Key_Escape:
            self.clear_polygon()
            return
        if event.key() == Qt.Key_Backspace:
            self.undo_point()
            return
        super().keyPressEvent(event)


class ObjectCutoutStudioDialog(QDialog):
    """Interactive cutout studio: set label, manual polygon, or rembg auto."""

    assets_saved = Signal()

    def __init__(
        self,
        *,
        objects_dir: Optional[Path] = None,
        class_map_text: str = "object=0",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 抠图工作室")
        self.resize(1280, 780)
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        theme_mode = "dark"
        if parent is not None and hasattr(parent, "_theme_mode"):
            theme_mode = str(getattr(parent, "_theme_mode") or "dark")
        apply_theme(self, theme_mode)

        self._objects_dir = Path(objects_dir) if objects_dir else Path.cwd() / "objects"
        self._image_path: Optional[Path] = None
        self._folder_root: Optional[Path] = None
        self._folder_images: List[Path] = []
        self._folder_index = -1
        self._bgr: Optional[np.ndarray] = None
        self._worker: Optional[_CutoutWorker] = None
        self._batch_worker: Optional[_BatchCutoutWorker] = None
        self._ensure_worker: Optional[_EnsureModelWorker] = None
        self._point_worker: Optional[_PointSegWorker] = None
        self._thumb_worker: Optional[_ThumbWorker] = None
        self._syncing_list = False
        self._pending_auto_after_load = False
        self._pending_batch_after_load = False
        self._loading = False
        self._ensure_ok_id = None
        self._busy_cutout = False
        self._busy_batch = False
        self._saved_once = False
        self._thumb_size = 72

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ---- left: folder browser + thumbnails ----
        browser = QWidget()
        browser.setMinimumWidth(200)
        browser.setMaximumWidth(320)
        br_l = QVBoxLayout(browser)
        br_l.setContentsMargins(0, 0, 4, 0)
        br_l.setSpacing(6)
        br_title = QLabel("图片目录")
        br_title.setObjectName("mutedLabel")
        br_l.addWidget(br_title)
        self.folder_label = QLabel("尚未打开文件夹")
        self.folder_label.setObjectName("mutedLabel")
        self.folder_label.setWordWrap(True)
        br_l.addWidget(self.folder_label)
        self.file_count = QLabel("")
        self.file_count.setObjectName("mutedLabel")
        br_l.addWidget(self.file_count)
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(self._thumb_size, self._thumb_size))
        self.file_list.setSpacing(2)
        self.file_list.setUniformItemSizes(True)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.file_list.setWordWrap(True)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        br_l.addWidget(self.file_list, 1)
        self.file_empty = QLabel("点右侧「打开文件夹…」\n后在此显示目录与缩略图")
        self.file_empty.setObjectName("mutedLabel")
        self.file_empty.setAlignment(Qt.AlignCenter)
        self.file_empty.setWordWrap(True)
        br_l.addWidget(self.file_empty)
        split.addWidget(browser)

        # ---- canvas ----
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 4, 0)
        left_l.setSpacing(6)
        self.view = AnnotateView()
        self.view.polygon_changed.connect(self._refresh_status)
        self.view.point_prompt.connect(self._on_point_prompt)
        left_l.addWidget(self.view, 1)
        self.hint = QLabel(
            "模式：手绘 — 左键加点 · 双击/回车闭合 · 右键撤销 · Esc 清空 · 滚轮缩放"
        )
        self.hint.setObjectName("mutedLabel")
        self.hint.setWordWrap(True)
        left_l.addWidget(self.hint)
        split.addWidget(left)

        # ---- right controls (scrollable) ----
        right_host = QWidget()
        right_host.setMinimumWidth(320)
        right_host.setMaximumWidth(420)
        right_outer = QVBoxLayout(right_host)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        panel = QWidget()
        form = QVBoxLayout(panel)
        form.setContentsMargins(2, 2, 8, 2)
        form.setSpacing(6)
        scroll.setWidget(panel)
        right_outer.addWidget(scroll, 1)

        # 1) Image
        io_box = QGroupBox("图像")
        io_f = QVBoxLayout(io_box)
        io_f.setSpacing(6)
        open_row = QHBoxLayout()
        open_btn = QPushButton("打开图片…")
        open_btn.clicked.connect(self._open_image)
        open_row.addWidget(open_btn)
        folder_btn = QPushButton("打开文件夹…")
        folder_btn.clicked.connect(self._open_folder)
        open_row.addWidget(folder_btn)
        io_f.addLayout(open_row)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("‹ 上一张")
        self.prev_btn.clicked.connect(lambda: self._step_folder(-1))
        self.next_btn = QPushButton("下一张 ›")
        self.next_btn.clicked.connect(lambda: self._step_folder(+1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        io_f.addLayout(nav)
        self.file_label = QLabel("未打开图片")
        self.file_label.setObjectName("mutedLabel")
        self.file_label.setWordWrap(True)
        io_f.addWidget(self.file_label)
        form.addWidget(io_box)

        # 2) Label + model
        model_box = QGroupBox("标签与模型")
        model_f = QFormLayout(model_box)
        model_f.setSpacing(8)
        model_f.setContentsMargins(8, 12, 8, 8)
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        for name in _labels_from_class_map(class_map_text):
            self.label_combo.addItem(name)
        model_f.addRow("类别名", self.label_combo)

        self.model_combo = QComboBox()
        for spec in list_cutout_models():
            self.model_combo.addItem(spec.title, spec.id)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_f.addRow("模型", self.model_combo)

        self.model_desc = QLabel("")
        self.model_desc.setObjectName("mutedLabel")
        self.model_desc.setWordWrap(True)
        model_f.addRow(self.model_desc)

        self.model_status = QLabel("状态: 未加载")
        self.model_status.setObjectName("statusLabel")
        self.model_status.setWordWrap(True)
        model_f.addRow(self.model_status)

        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setTextVisible(True)
        self.model_progress.setFormat("%p%")
        self.model_progress.setVisible(False)
        model_f.addRow(self.model_progress)

        self.hf_mirror_cb = QCheckBox("使用 hf-mirror.com（国内推荐）")
        self.hf_mirror_cb.setChecked(True)
        self.hf_mirror_cb.setToolTip(
            "直连 huggingface.co 常被重置；勾选走镜像。"
            "下载时会临时忽略本机 socks:// 代理（避免 Unknown scheme）。"
        )
        model_f.addRow(self.hf_mirror_cb)

        dl_row = QHBoxLayout()
        self.download_btn = QPushButton("下载并加载")
        self.download_btn.setToolTip("后台下载，不阻塞浏览/标注")
        style_primary(self.download_btn)
        self.download_btn.clicked.connect(lambda: self._download_model())
        dl_row.addWidget(self.download_btn, 1)
        self.cancel_download_btn = QPushButton("取消等待")
        self.cancel_download_btn.setToolTip("立刻恢复界面；后台下载可能仍继续")
        self.cancel_download_btn.setEnabled(False)
        self.cancel_download_btn.clicked.connect(self._cancel_download_wait)
        dl_row.addWidget(self.cancel_download_btn)
        model_f.addRow(dl_row)
        form.addWidget(model_box)

        # 3) Cutout + save
        cut_box = QGroupBox("抠图与保存")
        cut_f = QVBoxLayout(cut_box)
        cut_f.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("交互模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("手绘多边形", AnnotateView.MODE_DRAW)
        self.mode_combo.addItem("点选分割（点击物体）", AnnotateView.MODE_CLICK)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        cut_f.addLayout(mode_row)

        self.auto_btn = QPushButton("自动抠图")
        self.auto_btn.setToolTip("整图自动抠图（rembg / GroundingSAM2）")
        style_primary(self.auto_btn)
        self.auto_btn.clicked.connect(self._auto_cutout)
        cut_f.addWidget(self.auto_btn)

        poly_row = QHBoxLayout()
        close_btn = QPushButton("闭合多边形")
        close_btn.clicked.connect(self.view.close_polygon)
        poly_row.addWidget(close_btn)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.view.clear_polygon)
        poly_row.addWidget(clear_btn)
        cut_f.addLayout(poly_row)

        self.out_dir = PathRow(self, str(self._objects_dir), directory=True)
        out_form = QFormLayout()
        out_form.setContentsMargins(0, 4, 0, 0)
        out_form.addRow("输出目录", self.out_dir.widget)
        cut_f.addLayout(out_form)

        save_row = QHBoxLayout()
        self.save_btn = QPushButton("保存当前")
        style_primary(self.save_btn)
        self.save_btn.setToolTip("按当前类别标签保存图像 + LabelMe JSON")
        self.save_btn.clicked.connect(lambda: self._save_current(advance=False))
        save_row.addWidget(self.save_btn, 1)
        self.save_next_btn = QPushButton("保存并下一张")
        self.save_next_btn.clicked.connect(lambda: self._save_current(advance=True))
        save_row.addWidget(self.save_next_btn)
        cut_f.addLayout(save_row)
        form.addWidget(cut_box)

        # 4) Batch (collapsed by default)
        batch_box = QGroupBox("批量自动抠图")
        batch_box.setCheckable(True)
        batch_box.setChecked(False)
        batch_box.setToolTip("展开后可对整个文件夹按类别自动抠图")
        batch_f = QFormLayout(batch_box)
        batch_f.setSpacing(6)
        self.batch_in = PathRow(self, "", directory=True, placeholder="普通原图目录")
        batch_f.addRow("输入目录", self.batch_in.widget)
        self.batch_subdir = QCheckBox("按子文件夹名作为类别")
        self.batch_subdir.setToolTip("input/person/*.jpg → person")
        batch_f.addRow(self.batch_subdir)
        batch_row = QHBoxLayout()
        self.batch_btn = QPushButton("开始批量")
        self.batch_btn.clicked.connect(self._batch_auto)
        batch_row.addWidget(self.batch_btn, 1)
        self.cancel_batch_btn = QPushButton("取消")
        style_danger(self.cancel_batch_btn)
        self.cancel_batch_btn.setEnabled(False)
        self.cancel_batch_btn.clicked.connect(self._cancel_batch)
        batch_row.addWidget(self.cancel_batch_btn)
        batch_f.addRow(batch_row)
        form.addWidget(batch_box)

        form.addStretch(1)

        # Footer (always visible below scroll)
        foot = QVBoxLayout()
        foot.setSpacing(6)
        self.job_progress = QProgressBar()
        self.job_progress.setRange(0, 100)
        self.job_progress.setValue(0)
        self.job_progress.setVisible(False)
        foot.addWidget(self.job_progress)
        self.status = QLabel("就绪 — 下载时仍可浏览图片、手动画多边形")
        self.status.setObjectName("statusLabel")
        self.status.setWordWrap(True)
        foot.addWidget(self.status)
        close = QPushButton("关闭")
        close.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        close.clicked.connect(self.close)
        foot.addWidget(close)
        right_outer.addLayout(foot)

        split.addWidget(right_host)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        split.setSizes([240, 700, 340])

        self._on_model_changed()
        self._on_mode_changed()
        self._refresh_nav()
        self._update_browser_empty()

    def current_model_id(self) -> str:
        mid = self.model_combo.currentData()
        return str(mid or "rembg:u2net")

    def current_label(self) -> str:
        text = self.label_combo.currentText().strip()
        return text or "object"

    def _select_model_id(self, model_id: str) -> None:
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_id:
                self.model_combo.setCurrentIndex(i)
                return

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData() or AnnotateView.MODE_DRAW
        self.view.set_mode(str(mode))
        if mode == AnnotateView.MODE_CLICK:
            self.hint.setText(
                "模式：点选 — 左键点物体自动分割 · 上方「类别名」即标签 · 滚轮缩放"
            )
            self._select_model_id("sam2_click")
            self.status.setText(
                f"点选分割已开。标签「{self.current_label()}」— 点击图中物体即可"
            )
            if not is_model_ready("sam2_click"):
                self.model_status.setText(
                    "状态: 未加载 SAM2 — 点一下图时会自动下载，或先点「下载并加载」"
                )
        else:
            self.hint.setText(
                "模式：手绘 — 左键加点 · 双击/回车闭合 · 右键撤销 · Esc 清空 · 滚轮缩放"
            )
            self.status.setText("手绘模式")

    def _on_point_prompt(self, x: float, y: float) -> None:
        if self._bgr is None:
            return
        if self._point_worker and self._point_worker.isRunning():
            self.status.setText("正在分割，请稍候…")
            return
        model_id = "sam2_click"
        self._select_model_id(model_id)
        label = self.current_label()
        self.job_progress.setVisible(True)
        self.job_progress.setRange(0, 0)
        self.job_progress.setFormat("点选分割中…")
        self.status.setText(f"分割中… ({x:.0f},{y:.0f}) → 标签「{label}」")
        self._point_worker = _PointSegWorker(
            self._bgr.copy(),
            x,
            y,
            model_id=model_id,
            use_hf_mirror=self.hf_mirror_cb.isChecked(),
            parent=self,
        )
        self._point_worker.succeeded.connect(self._on_point_ok)
        self._point_worker.failed.connect(self._on_point_fail)
        self._point_worker.finished.connect(self._on_point_done)
        self._point_worker.start()

    def _on_point_ok(self, poly, _alpha) -> None:
        if poly is None or len(poly) < 3:
            QMessageBox.warning(self, "点选分割", "未得到有效轮廓，请换个位置再点。")
            return
        self.view.set_polygon(poly, closed=True)
        self.status.setText(
            f"已分割 {len(poly)} 点 · 标签「{self.current_label()}」· 可改标签后保存"
        )

    def _on_point_fail(self, msg: str) -> None:
        QMessageBox.critical(self, "点选分割失败", msg)

    def _on_point_done(self) -> None:
        self.job_progress.setVisible(False)
        self.job_progress.setRange(0, 100)

    def _polish_label(self, label: QLabel) -> None:
        sty = label.style()
        sty.unpolish(label)
        sty.polish(label)
        label.update()

    def _show_model_ready(self, title: str, *, announce: bool = True) -> None:
        """Make model-ready state unmistakable."""
        self.model_progress.setObjectName("readyBar")
        self.model_progress.setVisible(True)
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(100)
        self.model_progress.setFormat("已就绪 ✓")
        self.model_progress.style().unpolish(self.model_progress)
        self.model_progress.style().polish(self.model_progress)

        self.model_status.setObjectName("readyLabel")
        self.model_status.setText(
            f"✓ 模型已就绪：{title}\n"
            f"可以点选分割 / 自动抠图，或手绘后保存"
        )
        self._polish_label(self.model_status)

        self.status.setObjectName("readyLabel")
        self.status.setText(f"✓ 下载完成 — {title}")
        self._polish_label(self.status)

        self.download_btn.setText("已加载 ✓")
        QTimer.singleShot(4000, lambda: self.download_btn.setText("下载并加载"))

        self.raise_()
        self.activateWindow()

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            try:
                parent.statusBar().showMessage(f"抠图模型已就绪：{title}", 10000)
            except Exception:
                pass

        if announce and not self._pending_auto_after_load and not self._pending_batch_after_load:
            QMessageBox.information(
                self,
                "模型已就绪",
                f"「{title}」已下载并加载完成。\n\n"
                "接下来可以：\n"
                "• 交互模式选「点选分割」，点击图中物体\n"
                "• 或点「自动抠图」\n"
                "• 设好类别名后点「保存当前」",
            )

    def _on_model_changed(self) -> None:
        try:
            spec = get_model_spec(self.current_model_id())
        except KeyError:
            return
        self.model_desc.setText(spec.description)
        if self._loading:
            return
        if is_model_ready(spec.id):
            self.model_status.setObjectName("readyLabel")
            self.model_status.setText(f"✓ 已就绪：{spec.title}")
            self._polish_label(self.model_status)
            self.model_progress.setObjectName("readyBar")
            self.model_progress.setVisible(True)
            self.model_progress.setRange(0, 100)
            self.model_progress.setValue(100)
            self.model_progress.setFormat("已就绪 ✓")
            self.model_progress.style().unpolish(self.model_progress)
            self.model_progress.style().polish(self.model_progress)
        else:
            self.model_status.setObjectName("statusLabel")
            self._polish_label(self.model_status)
            self.model_progress.setObjectName("")
            self.model_progress.setVisible(False)
            self.model_progress.setFormat("%p%")
            hint = spec.pip_hint
            self.model_status.setText(
                f"状态: 未加载 — 点「下载并加载」\n依赖: {hint}"
            )

    def _set_model_progress(self, msg: str, pct) -> None:
        self.model_status.setObjectName("busyLabel")
        self._polish_label(self.model_status)
        self.model_status.setText(msg)
        self.model_progress.setObjectName("")
        self.model_progress.setVisible(True)
        if pct is None:
            self.model_progress.setRange(0, 0)
            self.model_progress.setFormat("下载中…")
        else:
            self.model_progress.setRange(0, 100)
            value = int(max(0, min(100, float(pct))))
            self.model_progress.setValue(value)
            self.model_progress.setFormat(f"下载中 {value}%")
        self.status.setObjectName("busyLabel")
        self._polish_label(self.status)
        self.status.setText(msg)

    def _download_model(self, *, then_auto: bool = False, then_batch: bool = False) -> None:
        if self._ensure_worker and self._ensure_worker.isRunning():
            self.status.setText("已有模型正在后台下载，请稍候…")
            return
        self._pending_auto_after_load = then_auto
        self._pending_batch_after_load = then_batch
        model_id = self.current_model_id()
        self._loading = True
        self._ensure_ok_id = None
        self.download_btn.setEnabled(False)
        self.download_btn.setText("下载中…")
        self.cancel_download_btn.setEnabled(True)
        self._set_model_progress("开始下载/加载模型…", None)
        self._ensure_worker = _EnsureModelWorker(
            model_id,
            self,
            use_hf_mirror=self.hf_mirror_cb.isChecked(),
        )
        self._ensure_worker.progress.connect(self._on_ensure_progress)
        self._ensure_worker.succeeded.connect(self._on_ensure_ok)
        self._ensure_worker.failed.connect(self._on_ensure_fail)
        self._ensure_worker.finished.connect(self._on_ensure_done)
        self._ensure_worker.start()

    def _cancel_download_wait(self) -> None:
        """Stop waiting for download — UI unlocks; worker may finish ignored."""
        if self._ensure_worker and self._ensure_worker.isRunning():
            self._ensure_worker.discard()
        self._pending_auto_after_load = False
        self._pending_batch_after_load = False
        self._loading = False
        self._ensure_ok_id = None
        self.download_btn.setEnabled(True)
        self.download_btn.setText("下载并加载")
        self.cancel_download_btn.setEnabled(False)
        self.model_progress.setVisible(False)
        self.model_progress.setRange(0, 100)
        self.model_status.setObjectName("statusLabel")
        self._polish_label(self.model_status)
        self.model_status.setText("已取消等待（若后台仍在下载，完成后可再次点加载）")
        self.status.setObjectName("statusLabel")
        self._polish_label(self.status)
        self.status.setText("已取消等待模型，可继续浏览/标注")
        self._on_model_changed()

    def _on_ensure_progress(self, msg: str, pct) -> None:
        if not self._loading:
            return
        self._set_model_progress(msg, pct)

    def _on_ensure_ok(self, model_id: str) -> None:
        self._ensure_ok_id = model_id

    def _on_ensure_fail(self, msg: str) -> None:
        self._pending_auto_after_load = False
        self._pending_batch_after_load = False
        self._ensure_ok_id = None
        self.model_progress.setVisible(False)
        self.model_progress.setRange(0, 100)
        self.model_status.setObjectName("statusLabel")
        self._polish_label(self.model_status)
        self.model_status.setText(f"状态: 加载失败\n{msg}")
        self.status.setObjectName("statusLabel")
        self._polish_label(self.status)
        self.status.setText("模型加载失败")
        self.download_btn.setText("下载并加载")
        QMessageBox.critical(self, "模型加载失败", msg)

    def _on_ensure_done(self) -> None:
        was_loading = self._loading
        self._loading = False
        self.download_btn.setEnabled(True)
        self.cancel_download_btn.setEnabled(False)
        if not was_loading:
            self.download_btn.setText("下载并加载")
            return

        ok_id = self._ensure_ok_id
        if ok_id and is_model_ready(ok_id):
            try:
                title = get_model_spec(ok_id).title
            except KeyError:
                title = str(ok_id)
            do_auto = self._pending_auto_after_load
            do_batch = self._pending_batch_after_load
            self._pending_auto_after_load = False
            self._pending_batch_after_load = False
            self._show_model_ready(title, announce=not (do_auto or do_batch))
            if do_auto:
                self._auto_cutout()
            elif do_batch:
                self._batch_auto()
            return

        self.download_btn.setText("下载并加载")
        self.model_progress.setVisible(False)
        self._on_model_changed()
        self._pending_auto_after_load = False
        self._pending_batch_after_load = False

    def _refresh_status(self) -> None:
        poly = self.view.polygon_array()
        n = 0 if poly is None else len(poly)
        closed = "已闭合" if (poly is not None and self.view._closed) else "绘制中"
        if self.status.objectName() == "readyLabel" and "下载完成" in (self.status.text() or ""):
            return
        self.status.setObjectName("statusLabel")
        self._polish_label(self.status)
        self.status.setText(f"多边形点数: {n}（{closed}）")

    def _refresh_nav(self) -> None:
        has = bool(self._folder_images)
        self.prev_btn.setEnabled(has and self._folder_index > 0)
        self.next_btn.setEnabled(has and self._folder_index < len(self._folder_images) - 1)

    def _update_browser_empty(self) -> None:
        empty = not self._folder_images
        self.file_list.setVisible(not empty)
        self.file_empty.setVisible(empty)
        if empty:
            self.file_count.setText("")
            self.folder_label.setText("尚未打开文件夹")
            self.folder_label.setToolTip("")

    def _rel_name(self, path: Path) -> str:
        if self._folder_root is not None:
            try:
                return str(path.relative_to(self._folder_root))
            except ValueError:
                pass
        return path.name

    def _item_caption(self, path: Path) -> str:
        name = self._rel_name(path)
        if path.with_suffix(".json").is_file():
            return f"✓ {name}"
        return name

    def _stop_thumb_worker(self) -> None:
        w = self._thumb_worker
        if w is None:
            return
        self._thumb_worker = None
        w.ready.disconnect(self._on_thumb_ready)
        w.requestInterruption()
        w.wait(100)
        if w.isRunning():
            w.finished.connect(w.deleteLater)
        else:
            w.deleteLater()

    def _rebuild_file_list(self) -> None:
        self._stop_thumb_worker()
        self._syncing_list = True
        self.file_list.clear()
        placeholder = QPixmap(self._thumb_size, self._thumb_size)
        placeholder.fill(QColor(60, 60, 64))
        for path in self._folder_images:
            item = QListWidgetItem(QIcon(placeholder), self._item_caption(path))
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(str(path))
            item.setSizeHint(QSize(self.file_list.viewport().width() or 200, self._thumb_size + 8))
            self.file_list.addItem(item)
        self._syncing_list = False
        n = len(self._folder_images)
        if self._folder_root is not None:
            self.folder_label.setText(str(self._folder_root))
            self.folder_label.setToolTip(str(self._folder_root))
        self.file_count.setText(f"共 {n} 张" if n else "")
        self._update_browser_empty()
        if n:
            self._thumb_worker = _ThumbWorker(self._folder_images, self._thumb_size, self)
            self._thumb_worker.ready.connect(self._on_thumb_ready)
            self._thumb_worker.start()

    def _on_thumb_ready(self, index: int, pixmap) -> None:
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            return
        item = self.file_list.item(int(index))
        if item is None:
            return
        item.setIcon(QIcon(pixmap))

    def _on_file_selected(self, row: int) -> None:
        if self._syncing_list:
            return
        if 0 <= row < len(self._folder_images):
            self._folder_index = row
            self._load_path(self._folder_images[row], sync_list=False)

    def _set_folder_images(
        self,
        images: List[Path],
        *,
        root: Path,
        select: Optional[Path] = None,
    ) -> None:
        self._folder_root = Path(root)
        self._folder_images = list(images)
        self._rebuild_file_list()
        if not images:
            self._folder_index = -1
            self._refresh_nav()
            return
        idx = 0
        if select is not None:
            try:
                idx = self._folder_images.index(Path(select))
            except ValueError:
                idx = 0
        self._folder_index = idx
        self._syncing_list = True
        self.file_list.setCurrentRow(idx)
        self._syncing_list = False
        self._load_path(self._folder_images[idx], sync_list=False)

    def _open_image(self) -> None:
        start = str(self._folder_root or self._objects_dir)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            start,
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All (*)",
        )
        if not path:
            return
        chosen = Path(path)
        parent = chosen.parent
        siblings = sorted(
            p for p in parent.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not siblings:
            siblings = [chosen]
        self.batch_in.setText(str(parent))
        self._set_folder_images(siblings, root=parent, select=chosen)

    def _open_folder(self) -> None:
        start = str(self._folder_root or self._objects_dir)
        path = QFileDialog.getExistingDirectory(self, "打开图片文件夹", start)
        if not path:
            return
        root = Path(path)
        images = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            QMessageBox.warning(self, "抠图工作室", "该文件夹下没有图片。")
            return
        self.batch_in.setText(str(root))
        self._set_folder_images(images, root=root, select=images[0])

    def _step_folder(self, delta: int) -> None:
        if not self._folder_images:
            return
        idx = self._folder_index + delta
        if 0 <= idx < len(self._folder_images):
            self.file_list.setCurrentRow(idx)

    def _load_path(self, path: Path, *, sync_list: bool = True) -> None:
        bgr = imread_with_exif(path)
        if bgr is None:
            QMessageBox.warning(self, "抠图工作室", f"无法读取：{path}")
            return
        self._image_path = path
        self._bgr = bgr
        self.view.set_image(bgr)
        # Load sibling JSON polygon if present.
        json_path = path.with_suffix(".json")
        if json_path.is_file():
            try:
                import json
                data = json.loads(json_path.read_text(encoding="utf-8"))
                shapes = data.get("shapes") or []
                if shapes:
                    label = str(shapes[0].get("label") or "").strip()
                    if label:
                        idx = self.label_combo.findText(label)
                        if idx < 0:
                            self.label_combo.addItem(label)
                            idx = self.label_combo.findText(label)
                        self.label_combo.setCurrentIndex(idx)
                    pts = shapes[0].get("points") or []
                    if len(pts) >= 3:
                        self.view.set_polygon(pts, closed=True)
            except Exception:
                pass
        else:
            self.view.clear_polygon()
        if self._folder_images:
            try:
                self._folder_index = self._folder_images.index(path)
            except ValueError:
                pass
            if sync_list:
                self._syncing_list = True
                self.file_list.setCurrentRow(self._folder_index)
                self._syncing_list = False
            # Keep caption in sync (✓ after save)
            item = self.file_list.item(self._folder_index)
            if item is not None:
                item.setText(self._item_caption(path))
        pos = ""
        if self._folder_images and self._folder_index >= 0:
            pos = f"（{self._folder_index + 1}/{len(self._folder_images)}）"
        self.file_label.setText(f"{self._rel_name(path)} {pos}".strip())
        self.file_label.setToolTip(str(path))
        self._refresh_nav()
        self._refresh_status()

    def _auto_cutout(self) -> None:
        if self._bgr is None:
            QMessageBox.information(self, "抠图工作室", "请先打开一张图片。")
            return
        if self._worker and self._worker.isRunning():
            return
        if self._loading:
            self.status.setText("模型仍在下载，完成后会自动继续抠图…" if self._pending_auto_after_load
                                else "模型仍在下载，请稍候或点「取消等待」")
            if not self._pending_auto_after_load:
                self._pending_auto_after_load = True
            return
        model_id = self.current_model_id()
        if not is_model_ready(model_id):
            reply = QMessageBox.question(
                self,
                "模型未加载",
                "当前模型尚未下载/加载。现在在后台下载吗？\n（下载期间仍可浏览图片、手动画多边形）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self._download_model(then_auto=True)
            return
        self._busy_cutout = True
        self.auto_btn.setEnabled(False)
        self.job_progress.setVisible(True)
        self.job_progress.setRange(0, 0)  # busy
        self.job_progress.setFormat("抠图中…")
        self.status.setText("正在自动抠图…（可继续浏览其他操作）")
        self._worker = _CutoutWorker(
            self._bgr.copy(), model_id, self.current_label(), self,
        )
        self._worker.succeeded.connect(self._on_auto_ok)
        self._worker.failed.connect(self._on_auto_fail)
        self._worker.finished.connect(self._on_auto_done)
        self._worker.start()

    def _on_auto_ok(self, poly, _alpha) -> None:
        if poly is None or len(poly) < 3:
            QMessageBox.warning(self, "抠图工作室", "自动抠图未得到有效轮廓。")
            return
        self.view.set_polygon(poly, closed=True)
        self.status.setText(f"自动抠图完成：{len(poly)} 点")

    def _on_auto_fail(self, msg: str) -> None:
        QMessageBox.critical(self, "自动抠图失败", msg)

    def _on_auto_done(self) -> None:
        self._busy_cutout = False
        self.auto_btn.setEnabled(True)
        self.job_progress.setVisible(False)
        self.job_progress.setRange(0, 100)

    def _save_current(self, *, advance: bool) -> None:
        if self._image_path is None or self._bgr is None:
            QMessageBox.information(self, "抠图工作室", "请先打开一张图片。")
            return
        poly = self.view.polygon_array()
        if poly is None or len(poly) < 3:
            QMessageBox.warning(self, "抠图工作室", "请先完成多边形（至少 3 个点并闭合）。")
            return
        if not self.view._closed:
            self.view.close_polygon()
        out = Path(self.out_dir.text() or str(self._objects_dir))
        try:
            json_path = save_cutout_as_labelme(
                self._image_path,
                out,
                label=self.current_label(),
                polygon=poly,
                bgr=self._bgr,
            )
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.status.setText(f"已保存：{json_path}")
        self.assets_saved.emit()
        if self._folder_index >= 0:
            item = self.file_list.item(self._folder_index)
            if item is not None and self._image_path is not None:
                item.setText(self._item_caption(self._image_path))
        if advance:
            self._step_folder(+1)

    def closeEvent(self, event) -> None:
        self._stop_thumb_worker()
        super().closeEvent(event)

    def _cancel_batch(self) -> None:
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.abort()
            self.status.setText("正在取消批量…")

    def _batch_auto(self) -> None:
        in_dir = Path(self.batch_in.text() or "")
        if not in_dir.is_dir():
            QMessageBox.warning(self, "批量抠图", "请先选择输入目录。")
            return
        out = Path(self.out_dir.text() or str(self._objects_dir))
        images = sorted(
            p for p in in_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            QMessageBox.warning(self, "批量抠图", "输入目录下没有图片。")
            return
        if self._batch_worker and self._batch_worker.isRunning():
            return
        if self._loading:
            self.status.setText("模型仍在下载，完成后会自动开始批量…")
            self._pending_batch_after_load = True
            return
        model_id = self.current_model_id()
        if not is_model_ready(model_id):
            reply = QMessageBox.question(
                self,
                "模型未加载",
                "当前模型尚未下载/加载。现在在后台下载吗？\n（下载期间仍可浏览图片、手动画多边形）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self._download_model(then_batch=True)
            return
        self._busy_batch = True
        self.batch_btn.setEnabled(False)
        self.cancel_batch_btn.setEnabled(True)
        self.job_progress.setVisible(True)
        self.job_progress.setRange(0, len(images))
        self.job_progress.setValue(0)
        self.job_progress.setFormat("%v / %m")
        self.status.setText(f"批量抠图 0/{len(images)}…")
        worker = _BatchCutoutWorker(
            images,
            in_dir,
            out,
            model_id=model_id,
            default_label=self.current_label(),
            label_from_subdir=self.batch_subdir.isChecked(),
            parent=self,
        )
        self._batch_worker = worker

        def on_prog(done, total, name):
            self.job_progress.setMaximum(total)
            self.job_progress.setValue(done)
            self.status.setText(f"批量抠图 {done}/{total}  {name}")

        def on_done(ok, failed):
            self._busy_batch = False
            self.batch_btn.setEnabled(True)
            self.cancel_batch_btn.setEnabled(False)
            self.job_progress.setVisible(False)
            self.status.setText(f"批量完成：成功 {ok}，失败/跳过 {failed}")
            self.assets_saved.emit()
            QMessageBox.information(
                self, "批量抠图", f"完成。成功 {ok}，失败/跳过 {failed}。\n输出：{out}",
            )

        def on_fail(msg):
            self._busy_batch = False
            self.batch_btn.setEnabled(True)
            self.cancel_batch_btn.setEnabled(False)
            self.job_progress.setVisible(False)
            QMessageBox.critical(self, "批量抠图失败", msg)

        worker.progress.connect(on_prog)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(on_fail)
        worker.start()
