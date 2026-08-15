"""ScenePaste Asset Studio — human-reviewed foreground/background asset editor.

The studio deliberately separates two editable masks:

* Foreground mask: authoritative semantic/instance label and transparent cutout.
* Background-removal mask: region to erase/fill when creating a clean background.

This keeps dataset generation controllable: AI is not asked to invent object
semantics. Users can correct segmentation and decide exactly what background
region should be reconstructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scenepaste.core.asset_studio import (
    binary_mask,
    clean_mask,
    export_asset_bundle,
    fill_mask_holes,
    make_clean_background,
    make_foreground_rgba,
    morph_mask,
)
from scenepaste.core.io import imread_with_exif
from scenepaste.core.labelme import is_paste_zone_label, shape_to_mask
from scenepaste.core.models import IMAGE_SUFFIXES

from .theme import apply_theme, style_primary
from .widgets import PathRow


def _qimage_from_bgr(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def _pixmap_from_bgr(bgr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(_qimage_from_bgr(bgr))


def _make_thumb(path: Path, size: int = 64) -> QPixmap:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    src = reader.size()
    if src.isValid() and src.width() and src.height():
        scale = min(size / src.width(), size / src.height(), 1.0)
        reader.setScaledSize(QSize(max(1, int(src.width() * scale)), max(1, int(src.height() * scale))))
    img = reader.read()
    if img.isNull():
        return QPixmap()
    return QPixmap.fromImage(img).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class MaskPaintView(QGraphicsView):
    """Zoomable image canvas with brush add/erase for two independent masks."""

    masks_changed = Signal()

    MODE_ADD = "add"
    MODE_ERASE = "erase"

    TARGET_FOREGROUND = "foreground"
    TARGET_BACKGROUND = "background"

    VIEW_OVERLAY = "overlay"
    VIEW_MASK = "mask"
    VIEW_FOREGROUND = "foreground"
    VIEW_BACKGROUND_MASK = "background-mask"
    VIEW_CLEAN_BACKGROUND = "clean-background"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._pix: Optional[QGraphicsPixmapItem] = None
        self._bgr: Optional[np.ndarray] = None
        self._auto_mask: Optional[np.ndarray] = None
        self._fg_mask: Optional[np.ndarray] = None
        self._bg_mask: Optional[np.ndarray] = None
        self._clean_background: Optional[np.ndarray] = None
        self._target = self.TARGET_FOREGROUND
        self._brush_mode = self.MODE_ADD
        self._brush_size = 28
        self._view_mode = self.VIEW_OVERLAY
        self._painting = False
        self._last_xy: Optional[tuple[int, int]] = None
        self._history: List[tuple[np.ndarray, np.ndarray]] = []
        self._redo: List[tuple[np.ndarray, np.ndarray]] = []
        self.setMinimumSize(560, 420)

    def set_image_and_masks(
        self,
        bgr: np.ndarray,
        auto_mask: np.ndarray,
        edited_mask: Optional[np.ndarray] = None,
        background_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._bgr = np.asarray(bgr, dtype=np.uint8).copy()
        self._auto_mask = binary_mask(auto_mask)
        self._fg_mask = binary_mask(edited_mask if edited_mask is not None else auto_mask)
        if background_mask is None:
            self._bg_mask = morph_mask(self._fg_mask, 4)
        else:
            self._bg_mask = binary_mask(background_mask)
        self._clean_background = None
        self._history.clear(); self._redo.clear()
        self.resetTransform()
        self._refresh()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def has_image(self) -> bool:
        return self._bgr is not None and self._fg_mask is not None and self._bg_mask is not None

    def foreground_mask(self) -> Optional[np.ndarray]:
        return None if self._fg_mask is None else self._fg_mask.copy()

    def background_mask(self) -> Optional[np.ndarray]:
        return None if self._bg_mask is None else self._bg_mask.copy()

    def auto_mask(self) -> Optional[np.ndarray]:
        return None if self._auto_mask is None else self._auto_mask.copy()

    def set_clean_background(self, bgr: Optional[np.ndarray]) -> None:
        self._clean_background = None if bgr is None else np.asarray(bgr, dtype=np.uint8).copy()
        self._refresh()

    def set_target(self, target: str) -> None:
        self._target = target if target in (self.TARGET_FOREGROUND, self.TARGET_BACKGROUND) else self.TARGET_FOREGROUND
        self._refresh()

    def set_brush_mode(self, mode: str) -> None:
        self._brush_mode = mode if mode in (self.MODE_ADD, self.MODE_ERASE) else self.MODE_ADD
        self.setDragMode(QGraphicsView.NoDrag)

    def set_brush_size(self, pixels: int) -> None:
        self._brush_size = max(1, int(pixels))

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = str(mode)
        self._refresh()

    def _active_mask(self) -> Optional[np.ndarray]:
        return self._fg_mask if self._target == self.TARGET_FOREGROUND else self._bg_mask

    def _set_active_mask(self, mask: np.ndarray, *, push: bool = True) -> None:
        if not self.has_image():
            return
        if push:
            self._push_history()
        m = binary_mask(mask)
        if self._target == self.TARGET_FOREGROUND:
            self._fg_mask = m
        else:
            self._bg_mask = m
        self._clean_background = None
        self._refresh(); self.masks_changed.emit()

    def _push_history(self) -> None:
        if self._fg_mask is None or self._bg_mask is None:
            return
        self._history.append((self._fg_mask.copy(), self._bg_mask.copy()))
        if len(self._history) > 40:
            self._history.pop(0)
        self._redo.clear()

    def undo_mask(self) -> None:
        if not self._history or self._fg_mask is None or self._bg_mask is None:
            return
        self._redo.append((self._fg_mask.copy(), self._bg_mask.copy()))
        self._fg_mask, self._bg_mask = self._history.pop()
        self._clean_background = None
        self._refresh(); self.masks_changed.emit()

    def redo_mask(self) -> None:
        if not self._redo or self._fg_mask is None or self._bg_mask is None:
            return
        self._history.append((self._fg_mask.copy(), self._bg_mask.copy()))
        self._fg_mask, self._bg_mask = self._redo.pop()
        self._clean_background = None
        self._refresh(); self.masks_changed.emit()

    def restore_auto_foreground(self) -> None:
        if self._auto_mask is None:
            return
        self._target = self.TARGET_FOREGROUND
        self._set_active_mask(self._auto_mask)

    def derive_background_from_foreground(self, expand_px: int = 4) -> None:
        if self._fg_mask is None:
            return
        self._target = self.TARGET_BACKGROUND
        self._set_active_mask(morph_mask(self._fg_mask, int(expand_px)))

    def apply_morph(self, pixels: int) -> None:
        m = self._active_mask()
        if m is not None:
            self._set_active_mask(morph_mask(m, int(pixels)))

    def apply_clean(self) -> None:
        m = self._active_mask()
        if m is not None:
            self._set_active_mask(clean_mask(m, radius=1, fill_holes=True))

    def apply_fill_holes(self) -> None:
        m = self._active_mask()
        if m is not None:
            self._set_active_mask(fill_mask_holes(m))

    def _paint_at(self, x: int, y: int, previous: Optional[tuple[int, int]] = None) -> None:
        mask = self._active_mask()
        if mask is None:
            return
        value = 255 if self._brush_mode == self.MODE_ADD else 0
        radius = max(1, int(round(self._brush_size / 2)))
        if previous is None:
            cv2.circle(mask, (x, y), radius, value, -1, lineType=cv2.LINE_AA)
        else:
            cv2.line(mask, previous, (x, y), value, thickness=max(1, radius * 2), lineType=cv2.LINE_AA)
            cv2.circle(mask, (x, y), radius, value, -1, lineType=cv2.LINE_AA)
        # Ensure binary semantics despite anti-aliased paint edges.
        mask[:] = binary_mask(mask)
        self._clean_background = None

    def _scene_xy(self, event) -> Optional[tuple[int, int]]:
        if self._bgr is None:
            return None
        p: QPointF = self.mapToScene(event.position().toPoint())
        h, w = self._bgr.shape[:2]
        x, y = int(round(p.x())), int(round(p.y()))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.has_image() and self._view_mode != self.VIEW_CLEAN_BACKGROUND:
            xy = self._scene_xy(event)
            if xy is not None:
                self._push_history()
                self._painting = True
                self._last_xy = xy
                self._paint_at(*xy)
                self._refresh()
                return
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._painting:
            xy = self._scene_xy(event)
            if xy is not None:
                self._paint_at(*xy, previous=self._last_xy)
                self._last_xy = xy
                self._refresh()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._painting and event.button() == Qt.LeftButton:
            self._painting = False
            self._last_xy = None
            self.masks_changed.emit()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _refresh(self) -> None:
        self._scene.clear(); self._pix = None
        if self._bgr is None or self._fg_mask is None or self._bg_mask is None:
            return
        bgr = self._render_bgr()
        pm = _pixmap_from_bgr(bgr)
        self._pix = self._scene.addPixmap(pm)
        self._scene.setSceneRect(0, 0, pm.width(), pm.height())

    def _render_bgr(self) -> np.ndarray:
        assert self._bgr is not None and self._fg_mask is not None and self._bg_mask is not None
        if self._view_mode == self.VIEW_MASK:
            return cv2.cvtColor(self._fg_mask, cv2.COLOR_GRAY2BGR)
        if self._view_mode == self.VIEW_BACKGROUND_MASK:
            return cv2.cvtColor(self._bg_mask, cv2.COLOR_GRAY2BGR)
        if self._view_mode == self.VIEW_FOREGROUND:
            rgba, _ = make_foreground_rgba(self._bgr, self._fg_mask, feather_px=1.0, crop=False)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            checker = np.full_like(self._bgr, 64)
            checker[::20, ::20] = 92
            out = rgba[..., :3].astype(np.float32) * alpha + checker.astype(np.float32) * (1.0 - alpha)
            return np.clip(out, 0, 255).astype(np.uint8)
        if self._view_mode == self.VIEW_CLEAN_BACKGROUND:
            if self._clean_background is not None:
                return self._clean_background.copy()
            return self._bgr.copy()

        # Overlay mode. Active mask receives stronger tint, the other one a thin contour.
        out = self._bgr.copy()
        active = self._fg_mask if self._target == self.TARGET_FOREGROUND else self._bg_mask
        inactive = self._bg_mask if self._target == self.TARGET_FOREGROUND else self._fg_mask
        color = np.array([70, 220, 70] if self._target == self.TARGET_FOREGROUND else [0, 165, 255], dtype=np.float32)
        hit = active > 0
        if np.any(hit):
            base = out[hit].astype(np.float32)
            out[hit] = np.clip(base * 0.58 + color * 0.42, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask(inactive), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (230, 230, 230), 1, cv2.LINE_AA)
        return out


class AssetStudioDialog(QDialog):
    """Review/edit segmentation masks and export foreground + clean background assets."""

    assets_saved = Signal()

    def __init__(
        self,
        *,
        objects_dir: Optional[Path] = None,
        backgrounds_dir: Optional[Path] = None,
        class_map_text: str = "object=0",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 素材工作室")
        self.resize(1440, 860)
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        theme_mode = str(getattr(parent, "_theme_mode", "dark") or "dark")
        apply_theme(self, theme_mode)

        self._objects_dir = Path(objects_dir) if objects_dir else Path.cwd() / "objects"
        self._backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else Path.cwd() / "backgrounds"
        common_root = self._objects_dir.parent if self._objects_dir.parent == self._backgrounds_dir.parent else Path.cwd()
        self._bundle_root = common_root / ".scenepaste" / "asset_studio"
        self._image_path: Optional[Path] = None
        self._bgr: Optional[np.ndarray] = None
        self._shapes: List[dict] = []
        self._shape_source_indices: List[int] = []
        self._current_shape_row = -1
        self._folder_images: List[Path] = []
        self._folder_root: Optional[Path] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hero = QLabel("素材工作室")
        hero.setObjectName("sectionHeadline")
        root.addWidget(hero)
        desc = QLabel("把真实标注拆成可控数据资产：人工修正前景 Mask → 导出透明前景；独立编辑背景移除 Mask → 生成干净背景。不改写目标语义，只做 Mask 与背景修补。")
        desc.setObjectName("mutedLabel"); desc.setWordWrap(True)
        root.addWidget(desc)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # Left: image and instance browser.
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)
        open_row = QHBoxLayout()
        open_img = QPushButton("打开图片…"); open_img.clicked.connect(self._open_image)
        open_folder = QPushButton("打开文件夹…"); open_folder.clicked.connect(self._open_folder)
        open_row.addWidget(open_img); open_row.addWidget(open_folder)
        ll.addLayout(open_row)
        self.file_list = QListWidget(); self.file_list.currentRowChanged.connect(self._on_file_row)
        ll.addWidget(self.file_list, 1)
        ll.addWidget(QLabel("当前图片中的标注实例"))
        self.shape_list = QListWidget(); self.shape_list.currentRowChanged.connect(self._on_shape_row)
        self.shape_list.setMaximumHeight(220)
        ll.addWidget(self.shape_list)
        split.addWidget(left)

        # Center: mask editor.
        center = QWidget(); cl = QVBoxLayout(center); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(6)
        top = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItem("原图 + Mask 叠加", MaskPaintView.VIEW_OVERLAY)
        self.view_combo.addItem("前景 Mask", MaskPaintView.VIEW_MASK)
        self.view_combo.addItem("透明前景预览", MaskPaintView.VIEW_FOREGROUND)
        self.view_combo.addItem("背景移除 Mask", MaskPaintView.VIEW_BACKGROUND_MASK)
        self.view_combo.addItem("干净背景预览", MaskPaintView.VIEW_CLEAN_BACKGROUND)
        self.view_combo.currentIndexChanged.connect(lambda: self.view.set_view_mode(str(self.view_combo.currentData())))
        top.addWidget(QLabel("查看:")); top.addWidget(self.view_combo, 1)
        self.target_combo = QComboBox()
        self.target_combo.addItem("编辑前景 Mask（训练标签）", MaskPaintView.TARGET_FOREGROUND)
        self.target_combo.addItem("编辑背景移除 Mask（只影响干净背景）", MaskPaintView.TARGET_BACKGROUND)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        top.addWidget(QLabel("编辑:")); top.addWidget(self.target_combo, 2)
        cl.addLayout(top)

        brush = QHBoxLayout()
        self.add_btn = QPushButton("画笔 +")
        self.add_btn.setCheckable(True); self.add_btn.setChecked(True)
        self.erase_btn = QPushButton("橡皮擦 -")
        self.erase_btn.setCheckable(True)
        self.add_btn.clicked.connect(lambda: self._choose_brush(MaskPaintView.MODE_ADD))
        self.erase_btn.clicked.connect(lambda: self._choose_brush(MaskPaintView.MODE_ERASE))
        brush.addWidget(self.add_btn); brush.addWidget(self.erase_btn)
        self.brush_size = QSpinBox(); self.brush_size.setRange(2, 200); self.brush_size.setValue(28); self.brush_size.setSuffix(" px")
        brush.addWidget(QLabel("画笔:")); brush.addWidget(self.brush_size)
        undo = QPushButton("撤销 Mask"); redo = QPushButton("重做 Mask")
        undo.clicked.connect(lambda: self.view.undo_mask()); redo.clicked.connect(lambda: self.view.redo_mask())
        brush.addWidget(undo); brush.addWidget(redo); brush.addStretch(1)
        cl.addLayout(brush)

        self.view = MaskPaintView(self)
        self.brush_size.valueChanged.connect(self.view.set_brush_size)
        cl.addWidget(self.view, 1)

        ops = QHBoxLayout()
        reset = QPushButton("恢复自动分割"); reset.clicked.connect(self._restore_auto)
        derive_bg = QPushButton("背景 Mask ← 前景外扩"); derive_bg.clicked.connect(self._derive_background_mask)
        dil = QPushButton("膨胀 3px"); dil.clicked.connect(lambda: self.view.apply_morph(3))
        ero = QPushButton("腐蚀 3px"); ero.clicked.connect(lambda: self.view.apply_morph(-3))
        holes = QPushButton("填洞"); holes.clicked.connect(self.view.apply_fill_holes)
        smooth = QPushButton("去毛刺"); smooth.clicked.connect(self.view.apply_clean)
        for b in (reset, derive_bg, dil, ero, holes, smooth): ops.addWidget(b)
        ops.addStretch(1)
        cl.addLayout(ops)
        split.addWidget(center)

        # Right: export and background creation.
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(8)
        state_box = QGroupBox("当前资产")
        sf = QFormLayout(state_box)
        self.file_label = QLabel("尚未打开图片"); self.file_label.setWordWrap(True)
        sf.addRow("原图:", self.file_label)
        self.label_edit = QComboBox(); self.label_edit.setEditable(True)
        for part in class_map_text.replace("，", ",").split(","):
            name = part.split("=", 1)[0].strip()
            if name and self.label_edit.findText(name) < 0: self.label_edit.addItem(name)
        sf.addRow("类别:", self.label_edit)
        self.mask_stats = QLabel("前景 0 px · 背景移除 0 px"); self.mask_stats.setObjectName("mutedLabel")
        sf.addRow("Mask:", self.mask_stats)
        rl.addWidget(state_box)

        out_box = QGroupBox("输出位置")
        of = QFormLayout(out_box)
        self.objects_out = PathRow(self, str(self._objects_dir), directory=True)
        self.backgrounds_out = PathRow(self, str(self._backgrounds_dir / "edited_backgrounds"), directory=True)
        self.bundle_out = PathRow(self, str(self._bundle_root), directory=True)
        of.addRow("前景素材库:", self.objects_out.widget)
        of.addRow("背景素材库:", self.backgrounds_out.widget)
        of.addRow("审核记录:", self.bundle_out.widget)
        rl.addWidget(out_box)

        bg_box = QGroupBox("干净背景")
        bf = QFormLayout(bg_box)
        self.bg_expand = QSpinBox(); self.bg_expand.setRange(0, 60); self.bg_expand.setValue(4); self.bg_expand.setSuffix(" px")
        self.bg_radius = QDoubleSpinBox(); self.bg_radius.setRange(1, 30); self.bg_radius.setValue(3.0); self.bg_radius.setDecimals(1)
        self.bg_method = QComboBox(); self.bg_method.addItem("Telea（推荐）", "telea"); self.bg_method.addItem("Navier-Stokes", "ns")
        bf.addRow("额外外扩:", self.bg_expand)
        bf.addRow("修补半径:", self.bg_radius)
        bf.addRow("算法:", self.bg_method)
        preview_bg = QPushButton("预览干净背景"); preview_bg.clicked.connect(self._preview_clean_background)
        bf.addRow(preview_bg)
        rl.addWidget(bg_box)

        self.export_btn = QPushButton("保存审核素材：前景 + Mask + 干净背景")
        self.export_btn.setObjectName("primaryButton"); style_primary(self.export_btn)
        self.export_btn.setMinimumHeight(38); self.export_btn.clicked.connect(self._export_current)
        rl.addWidget(self.export_btn)
        self.status = QLabel("先打开已有分割数据；用画笔修正前景，再单独调整背景移除范围。")
        self.status.setObjectName("statusLabel"); self.status.setWordWrap(True)
        rl.addWidget(self.status)
        rl.addStretch(1)
        close = QPushButton("关闭"); close.clicked.connect(self.close); rl.addWidget(close)
        split.addWidget(right)

        split.setSizes([235, 850, 355])
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setStretchFactor(2, 0)
        self.view.masks_changed.connect(self._refresh_mask_stats)
        self._choose_brush(MaskPaintView.MODE_ADD)

    def view_brush_size_changed(self, value: int) -> None:
        if hasattr(self, "view"):
            self.view.set_brush_size(value)

    def _choose_brush(self, mode: str) -> None:
        add = mode == MaskPaintView.MODE_ADD
        self.add_btn.setChecked(add); self.erase_btn.setChecked(not add)
        if hasattr(self, "view"):
            self.view.set_brush_mode(mode)

    def _on_target_changed(self) -> None:
        target = str(self.target_combo.currentData())
        self.view.set_target(target)
        # Keep overlay visible while editing either mask.
        if self.view_combo.currentData() == MaskPaintView.VIEW_CLEAN_BACKGROUND:
            self.view_combo.setCurrentIndex(0)

    def _open_image(self) -> None:
        start = str(self._folder_root or self._objects_dir.parent)
        path, _ = QFileDialog.getOpenFileName(self, "打开已有分割图片", start, "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not path: return
        p = Path(path); siblings = sorted(x for x in p.parent.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES)
        self._set_folder(siblings or [p], p.parent, select=p)

    def _open_folder(self) -> None:
        start = str(self._folder_root or self._objects_dir.parent)
        path = QFileDialog.getExistingDirectory(self, "选择已有分割图片文件夹", start)
        if not path: return
        root = Path(path)
        images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            QMessageBox.warning(self, "素材工作室", "该目录没有图片。")
            return
        self._set_folder(images, root, select=images[0])

    def _set_folder(self, images: Sequence[Path], root: Path, *, select: Optional[Path] = None) -> None:
        self._folder_images = list(images); self._folder_root = Path(root)
        self.file_list.clear()
        for path in self._folder_images:
            item = QListWidgetItem(path.name); item.setToolTip(str(path))
            pm = _make_thumb(path)
            if not pm.isNull(): item.setIcon(QIcon(pm))
            self.file_list.addItem(item)
        idx = 0
        if select is not None:
            try: idx = self._folder_images.index(Path(select))
            except ValueError: idx = 0
        self.file_list.setCurrentRow(idx)

    def _on_file_row(self, row: int) -> None:
        if 0 <= row < len(self._folder_images):
            self._load_image(self._folder_images[row])

    def _load_image(self, path: Path) -> None:
        bgr = imread_with_exif(path)
        if bgr is None:
            QMessageBox.warning(self, "素材工作室", f"无法读取图片：{path}")
            return
        self._image_path = Path(path); self._bgr = bgr
        self.file_label.setText(path.name); self.file_label.setToolTip(str(path))
        self._shapes = []; self._shape_source_indices = []
        json_path = path.with_suffix(".json")
        if json_path.is_file():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for idx, shape in enumerate(data.get("shapes") or []):
                    label = str(shape.get("label") or "").strip()
                    if not label or is_paste_zone_label(label):
                        continue
                    mask = shape_to_mask(shape, bgr.shape[0], bgr.shape[1])
                    if mask is not None and np.any(mask):
                        self._shapes.append(dict(shape)); self._shape_source_indices.append(idx)
            except Exception as exc:
                self.status.setText(f"标注读取失败：{exc}")
        if not self._shapes:
            # Allow manual creation even without an annotation.
            self._shapes = [{"label": self.label_edit.currentText() or "object", "points": [], "shape_type": "manual"}]
            self._shape_source_indices = [-1]
        self.shape_list.clear()
        for i, shape in enumerate(self._shapes):
            label = str(shape.get("label") or "object")
            st = str(shape.get("shape_type") or "polygon")
            self.shape_list.addItem(f"#{i+1}  {label}  ·  {st}")
        self.shape_list.setCurrentRow(0)

    def _on_shape_row(self, row: int) -> None:
        if self._bgr is None or not (0 <= row < len(self._shapes)):
            return
        self._current_shape_row = row
        shape = self._shapes[row]
        label = str(shape.get("label") or "object")
        idx = self.label_edit.findText(label)
        if idx < 0: self.label_edit.addItem(label); idx = self.label_edit.findText(label)
        self.label_edit.setCurrentIndex(idx)
        mask = shape_to_mask(shape, self._bgr.shape[0], self._bgr.shape[1]) if shape.get("points") else None
        if mask is None:
            mask = np.zeros(self._bgr.shape[:2], dtype=np.uint8)
        bg_mask = morph_mask(mask, 4)
        self.view.set_image_and_masks(self._bgr, mask, background_mask=bg_mask)
        self._refresh_mask_stats()
        self.status.setText("已载入分割。绿色=前景标签；橙色=背景移除范围。画笔可人工修正。")

    def _restore_auto(self) -> None:
        self.target_combo.setCurrentIndex(0)
        self.view.restore_auto_foreground()

    def _derive_background_mask(self) -> None:
        self.target_combo.setCurrentIndex(1)
        self.view.derive_background_from_foreground(4)

    def _refresh_mask_stats(self) -> None:
        fg = self.view.foreground_mask(); bg = self.view.background_mask()
        fa = int(np.count_nonzero(fg)) if fg is not None else 0
        ba = int(np.count_nonzero(bg)) if bg is not None else 0
        self.mask_stats.setText(f"前景 {fa:,} px · 背景移除 {ba:,} px")

    def _preview_clean_background(self) -> None:
        if self._bgr is None:
            return
        bg = self.view.background_mask()
        if bg is None:
            return
        clean = make_clean_background(
            self._bgr, bg,
            expand_px=self.bg_expand.value(),
            radius=self.bg_radius.value(),
            method=str(self.bg_method.currentData()),
        )
        self.view.set_clean_background(clean)
        idx = self.view_combo.findData(MaskPaintView.VIEW_CLEAN_BACKGROUND)
        self.view_combo.setCurrentIndex(max(0, idx))
        self.status.setText("干净背景仅用于预览/素材扩充；如果补洞不理想，可继续编辑“背景移除 Mask”后重新预览。")

    def _export_current(self) -> None:
        if self._image_path is None or self._bgr is None or self._current_shape_row < 0:
            QMessageBox.information(self, "素材工作室", "请先打开一张分割图片并选择实例。")
            return
        fg = self.view.foreground_mask(); bgm = self.view.background_mask(); auto = self.view.auto_mask()
        if fg is None or bgm is None or auto is None or not np.any(fg):
            QMessageBox.warning(self, "素材工作室", "前景 Mask 为空，请先修正分割。")
            return
        try:
            result = export_asset_bundle(
                source_image=self._image_path,
                bgr=self._bgr,
                label=self.label_edit.currentText().strip() or "object",
                instance_index=(self._shape_source_indices[self._current_shape_row] if (self._current_shape_row < len(self._shape_source_indices) and self._shape_source_indices[self._current_shape_row] >= 0) else self._current_shape_row),
                auto_mask=auto,
                edited_mask=fg,
                background_remove_mask=bgm,
                objects_dir=Path(self.objects_out.text() or self._objects_dir),
                backgrounds_dir=Path(self.backgrounds_out.text() or (self._backgrounds_dir / "edited_backgrounds")),
                bundle_root=Path(self.bundle_out.text() or self._bundle_root),
                feather_px=1.0,
                background_expand_px=self.bg_expand.value(),
                background_inpaint_radius=self.bg_radius.value(),
                background_method=str(self.bg_method.currentData()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "保存审核素材失败", str(exc))
            return
        self.assets_saved.emit()
        self.status.setText(f"已保存前景：{result.foreground_path.name}；干净背景：{result.background_path.name}")
        QMessageBox.information(
            self,
            "素材已加入资产库",
            f"透明前景：\n{result.foreground_path}\n\n"
            f"干净背景：\n{result.background_path}\n\n"
            f"自动/人工 Mask 与来源记录：\n{result.bundle_dir}",
        )
