"""Panel builders for the Qt main window.

Three regions:

- Left: thumbnail gallery of loaded cutouts (drag onto canvas to place).
- Right: instance list + transform controls (scale, rotate, flip, delete).
- Top: project buttons (load objects, load backgrounds, save, prev/next bg).
"""

from __future__ import annotations


import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QSize, QMimeData, QByteArray
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QLineEdit, QComboBox, QGroupBox,
    QFormLayout, QDialog, QProgressBar, QMessageBox, QCheckBox, QSlider,
)

from compose_app.models import THUMB_SIZE
from .state import Document


def _pil_rgb_to_qpixmap(rgb: Image.Image) -> QPixmap:
    arr = np.asarray(rgb)
    h, w, _ = arr.shape
    qimg = QImage(arr.astype(np.uint8).tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _CutoutList(QListWidget):
    """Internal list that exposes cutout index via drag mime data."""

    def startDrag(self, supportedActions):
        from PySide6.QtCore import QDrag
        item = self.currentItem()
        if item is None:
            return
        idx = int(item.data(Qt.UserRole))
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-cutout-index", QByteArray(str(idx).encode("ascii")))
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)


class CutoutThumbWidget(QWidget):
    """Left-panel thumbnail gallery with label search. Items drag onto the canvas."""

    canvas_place_requested = Signal(int)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self._doc = doc
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("搜索类别…")
        self._filter.setClearButtonEnabled(True)
        self._filter.setMinimumHeight(28)
        self._filter.textChanged.connect(self._refresh)
        layout.addWidget(self._filter)

        self._list = _CutoutList()
        self._list.setIconSize(QSize(*THUMB_SIZE))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setFlow(QListWidget.LeftToRight)
        self._list.setWrapping(True)
        self._list.setSpacing(4)
        self._list.setDragEnabled(True)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setAcceptDrops(False)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list, 1)
        doc.subscribe(self._refresh)

    def _refresh(self) -> None:
        needle = self._filter.text().strip().lower()
        self._list.clear()
        for i, cutout in enumerate(self._doc.cutouts):
            if needle and needle not in cutout.label.lower():
                continue
            pm = _pil_rgb_to_qpixmap(cutout.thumb) if cutout.thumb else QPixmap()
            item = QListWidgetItem(pm, cutout.label)
            item.setData(Qt.UserRole, i)
            self._list.addItem(item)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        self.canvas_place_requested.emit(int(idx))


class _InstanceRowWidget(QWidget):
    """Per-row widget: visibility toggle, lock toggle, label."""

    def __init__(self, uid: int, label: str, visible: bool, locked: bool,
                 on_visibility, on_lock, on_select=None, parent=None):
        super().__init__(parent)
        self._uid = uid
        self._on_select = on_select
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 8, 4)
        layout.setSpacing(4)

        self._vis_btn = QPushButton("◉" if visible else "◯")
        self._vis_btn.setObjectName("iconToggleButton")
        self._vis_btn.setCheckable(True)
        self._vis_btn.setChecked(visible)
        self._vis_btn.setFixedSize(28, 24)
        self._vis_btn.setToolTip("显示/隐藏（隐藏的实例不参与合成与标注）")
        self._vis_btn.clicked.connect(self._on_vis_clicked)
        self._on_visibility = on_visibility
        layout.addWidget(self._vis_btn)

        self._lock_btn = QPushButton("🔒" if locked else "🔓")
        self._lock_btn.setObjectName("iconToggleButton")
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(locked)
        self._lock_btn.setFixedSize(28, 24)
        self._lock_btn.setToolTip(
            "锁定/解锁（锁定后不可变换：画布拖动/旋转/滚轮、面板与快捷键均拒绝）"
        )
        self._lock_btn.clicked.connect(self._on_lock_clicked)
        self._on_lock = on_lock
        layout.addWidget(self._lock_btn)

        self._label = QLabel(label)
        self._label.setToolTip("点击选中该实例")
        layout.addWidget(self._label, 1)

        self._refresh_style(visible, locked)

    def _select_self(self) -> None:
        if self._on_select is not None:
            self._on_select(self._uid)

    def _on_vis_clicked(self) -> None:
        self._select_self()
        self._on_visibility(self._uid, self._vis_btn.isChecked())

    def _on_lock_clicked(self) -> None:
        self._select_self()
        self._on_lock(self._uid, self._lock_btn.isChecked())

    def mousePressEvent(self, event) -> None:
        # setItemWidget rows swallow QListWidget.itemClicked — select explicitly.
        self._select_self()
        super().mousePressEvent(event)

    def _refresh_style(self, visible: bool, locked: bool) -> None:
        if not visible:
            self._label.setStyleSheet("color: #6e6e72;")
        elif locked:
            self._label.setStyleSheet("color: #ffd60a;")
        else:
            self._label.setStyleSheet("")


class _InstanceList(QListWidget):
    """Internal list with drag-to-reorder; emits uid sequence on drop."""

    reorder_finished = Signal(list)  # list of uids in new visual order

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event):
        super().dropEvent(event)
        uids = []
        for i in range(self.count()):
            uid = self.item(i).data(Qt.UserRole)
            if uid is not None:
                uids.append(int(uid))
        self.reorder_finished.emit(uids)


class InstanceListWidget(QWidget):
    """Right-panel instances list with layer ordering, visibility, and lock.

    The list is displayed front→back (top→bottom), matching Photoshop-style
    layer panels. ``doc.instances`` remains back→front; drag-reorder converts
    visual order back to document order before emitting ``reorder_requested``.

    Signals (consumed by MainWindowController to push undo commands):
        selection_clicked(int): user clicked a row (uid).
        layer_op_requested(str, int): one of bring_front|send_back|forward|backward + uid.
        visibility_toggled(int, bool): uid, new_visible.
        lock_toggled(int, bool): uid, new_locked.
        reorder_requested(list): new uid sequence in document (back→front) order.
    """

    selection_clicked = Signal(int)
    layer_op_requested = Signal(str, int)
    visibility_toggled = Signal(int, bool)
    lock_toggled = Signal(int, bool)
    reorder_requested = Signal(list)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self._doc = doc

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Layer operation buttons (top of panel).
        # List is front→back (top→bottom); doc.instances is back→front.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.setContentsMargins(0, 0, 0, 0)
        for text, op in [("置顶", "bring_front"), ("上移", "forward"),
                         ("下移", "backward"), ("置底", "send_back")]:
            btn = QPushButton(text)
            btn.setObjectName("layerButton")
            btn.setMinimumHeight(26)
            btn.setToolTip({
                "bring_front": "移到画布最上层 / 列表顶部 (Home)",
                "forward": "上移一层 (Ctrl+Shift+])",
                "backward": "下移一层 (Ctrl+Shift+[)",
                "send_back": "移到画布最下层 / 列表底部 (End)",
            }[op])
            btn.clicked.connect(lambda _checked=False, op=op: self._emit_layer_op(op))
            btn_row.addWidget(btn, 1)
        layout.addLayout(btn_row)

        self._list = _InstanceList()
        self._list.setIconSize(QSize(*THUMB_SIZE))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setSpacing(2)
        self._list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.reorder_finished.connect(self._on_reorder_finished)
        layout.addWidget(self._list, 1)

        doc.subscribe(self._refresh)

    # ------------------------------------------------------------- controller
    def _status(self, text: str, ms: int = 2000) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar") and win.statusBar() is not None:
            win.statusBar().showMessage(text, ms)

    def _uid_for_layer_op(self):
        """Prefer the highlighted list row — it is what the user is looking at."""
        item = self._list.currentItem()
        if item is not None:
            uid = item.data(Qt.UserRole)
            if uid is not None:
                return int(uid)
        if self._doc.selected_uid is not None:
            return int(self._doc.selected_uid)
        return None

    def _emit_layer_op(self, op: str) -> None:
        uid = self._uid_for_layer_op()
        if uid is None:
            self._status("请先在列表或画布中选择一个实例")
            return
        if self._doc.selected_uid != uid:
            self.selection_clicked.emit(uid)
        self.layer_op_requested.emit(op, int(uid))

    def _select_uid(self, uid: int) -> None:
        self.selection_clicked.emit(int(uid))

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        uid = item.data(Qt.UserRole)
        if uid is not None:
            self.selection_clicked.emit(int(uid))

    def _on_current_changed(self, current, _previous) -> None:
        if current is None:
            return
        uid = current.data(Qt.UserRole)
        if uid is not None and int(uid) != self._doc.selected_uid:
            self.selection_clicked.emit(int(uid))

    def _on_reorder_finished(self, visual_uids: list) -> None:
        # Visual order is front→back; document stores back→front.
        self.reorder_requested.emit(list(reversed(visual_uids)))

    # ------------------------------------------------------------- rendering
    def _refresh(self) -> None:
        # Block signals so setCurrentRow doesn't loop back into select().
        self._list.blockSignals(True)
        try:
            self._list.clear()
            selected_row = -1
            # Front of canvas (last in doc.instances) at the top of the list.
            for i, inst in enumerate(reversed(self._doc.instances)):
                label = f"#{inst.uid}  {self._doc.cutouts[inst.cutout_index].label}" \
                        if 0 <= inst.cutout_index < len(self._doc.cutouts) else f"#{inst.uid}"
                item = QListWidgetItem()
                item.setData(Qt.UserRole, inst.uid)
                self._list.addItem(item)
                row = _InstanceRowWidget(
                    inst.uid, label,
                    visible=bool(getattr(inst, "visible", True)),
                    locked=bool(getattr(inst, "locked", False)),
                    on_visibility=self.visibility_toggled.emit,
                    on_lock=self.lock_toggled.emit,
                    on_select=self._select_uid,
                )
                item.setSizeHint(row.sizeHint().expandedTo(QSize(0, 32)))
                self._list.setItemWidget(item, row)
                if inst.uid == self._doc.selected_uid:
                    selected_row = i
            if selected_row >= 0:
                self._list.setCurrentRow(selected_row)
        finally:
            self._list.blockSignals(False)


class ControlsPanel(QWidget):
    """Scale / rotation / flip / appearance / delete for the selected instance."""

    scale_changed = Signal(float)
    angle_changed = Signal(float)
    flip_toggled = Signal()
    delete_clicked = Signal()
    nudge = Signal(int, int)
    # Live preview (no undo yet); MainWindow commits after debounce / discrete actions.
    appearance_preview = Signal()
    appearance_committed = Signal()
    appearance_resample = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        tip = QLabel("画布：滚轮缩放 · Shift+拖动旋转 · [ / ] 快捷缩放")
        tip.setObjectName("mutedLabel")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.02, 2.0)
        self.scale_spin.setSingleStep(0.02)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setToolTip("目标高度 / 背景高度。也可在画布上对选中目标滚轮缩放。")
        self.scale_spin.valueChanged.connect(lambda v: self.scale_changed.emit(float(v)))
        form.addRow("缩放(高度比):", self.scale_spin)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 359.9)
        self.angle_spin.setSingleStep(5.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setToolTip("顺时针角度 [0, 360)。也可在画布上 Shift+拖动旋转。")
        self.angle_spin.valueChanged.connect(lambda v: self.angle_changed.emit(float(v)))
        form.addRow("旋转角度:", self.angle_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.flip_btn = QPushButton("水平翻转")
        self.flip_btn.clicked.connect(self.flip_toggled)
        btn_row.addWidget(self.flip_btn, 1)
        self.del_btn = QPushButton("删除")
        self.del_btn.setObjectName("dangerButton")
        self.del_btn.clicked.connect(self.delete_clicked)
        btn_row.addWidget(self.del_btn, 1)
        layout.addLayout(btn_row)

        appearance = QGroupBox("目标外观预览")
        appearance.setToolTip(
            "仅改变贴图 RGB 外观，不移动标签。批量合成可用右侧「批量默认」或数据工厂里的目标外观 Recipe。"
        )
        app_form = QFormLayout(appearance)
        app_form.setContentsMargins(4, 8, 4, 4)
        app_form.setHorizontalSpacing(10)
        app_form.setVerticalSpacing(8)
        app_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.appearance_enable = QCheckBox("启用外观增强")
        self.appearance_enable.setToolTip(
            "关闭时不应用外观。拖动下方滑条会自动勾选并预览。"
        )
        self.appearance_enable.toggled.connect(self._on_appearance_discrete)
        app_form.addRow(self.appearance_enable)

        self.appearance_recipe = QComboBox()
        self.appearance_recipe.setEditable(True)
        self.appearance_recipe.addItems(["mild", "surveillance-object", "legacy", "off"])
        self.appearance_recipe.setToolTip("内置 Recipe 名，或自定义 JSON 路径")
        self.appearance_recipe.currentTextChanged.connect(self._on_appearance_discrete)
        app_form.addRow("Recipe:", self.appearance_recipe)

        self.appearance_resample_btn = QPushButton("换一种随机外观")
        self.appearance_resample_btn.setToolTip(
            "按当前 Recipe 随机采样，并把亮度/对比度/饱和度/色相/色温/模糊/噪声/锐化回填到下方滑条，便于继续微调"
        )
        self.appearance_resample_btn.clicked.connect(self.appearance_resample)
        app_form.addRow(self.appearance_resample_btn)

        self.brightness_slider = self._pct_slider(-20, 20, 0)
        self.brightness_slider.valueChanged.connect(self._on_appearance_live)
        self.brightness_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("亮度:", self.brightness_slider)

        self.contrast_slider = self._pct_slider(70, 130, 100)
        self.contrast_slider.valueChanged.connect(self._on_appearance_live)
        self.contrast_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("对比度:", self.contrast_slider)

        self.saturation_slider = self._pct_slider(70, 130, 100)
        self.saturation_slider.valueChanged.connect(self._on_appearance_live)
        self.saturation_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("饱和度:", self.saturation_slider)

        self.hue_slider = self._pct_slider(-30, 30, 0)
        self.hue_slider.setToolTip("色相偏移（度）")
        self.hue_slider.valueChanged.connect(self._on_appearance_live)
        self.hue_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("色相:", self.hue_slider)

        self.temperature_slider = self._pct_slider(-30, 30, 0)
        self.temperature_slider.setToolTip("色温：正值偏暖，负值偏冷")
        self.temperature_slider.valueChanged.connect(self._on_appearance_live)
        self.temperature_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("色温:", self.temperature_slider)

        self.blur_slider = self._pct_slider(0, 50, 0)  # 0.0 .. 5.0 sigma ×10
        self.blur_slider.setToolTip("高斯模糊强度（sigma）")
        self.blur_slider.valueChanged.connect(self._on_appearance_live)
        self.blur_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("模糊:", self.blur_slider)

        self.noise_slider = self._pct_slider(0, 20, 0)
        self.noise_slider.setToolTip("高斯噪声强度")
        self.noise_slider.valueChanged.connect(self._on_appearance_live)
        self.noise_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("噪声:", self.noise_slider)

        self.sharpness_slider = self._pct_slider(50, 150, 100)
        self.sharpness_slider.setToolTip("锐化：<100 变软，>100 变锐")
        self.sharpness_slider.valueChanged.connect(self._on_appearance_live)
        self.sharpness_slider.sliderReleased.connect(self.appearance_committed.emit)
        app_form.addRow("锐化:", self.sharpness_slider)

        layout.addWidget(appearance)
        self._appearance_widgets = [
            self.appearance_enable, self.appearance_recipe, self.appearance_resample_btn,
            self.brightness_slider, self.contrast_slider, self.saturation_slider,
            self.hue_slider, self.temperature_slider,
            self.blur_slider, self.noise_slider, self.sharpness_slider,
        ]

        layout.addStretch(1)

    @staticmethod
    def _pct_slider(lo: int, hi: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.setSingleStep(1)
        return slider

    def appearance_values(self) -> dict:
        return {
            "enabled": bool(self.appearance_enable.isChecked()),
            "recipe": self.appearance_recipe.currentText().strip() or "mild",
            "brightness": self.brightness_slider.value() / 100.0,
            "contrast": self.contrast_slider.value() / 100.0,
            "saturation": self.saturation_slider.value() / 100.0,
            "hue": float(self.hue_slider.value()),
            "temperature": float(self.temperature_slider.value()),
            "blur": self.blur_slider.value() / 10.0,
            "noise": float(self.noise_slider.value()),
            "sharpness": self.sharpness_slider.value() / 100.0,
        }

    def _on_appearance_live(self, *_args) -> None:
        self.appearance_preview.emit()

    def _on_appearance_discrete(self, *_args) -> None:
        self.appearance_preview.emit()
        self.appearance_committed.emit()

    def reflect(self, doc: Document) -> None:
        """Update spinboxes from the document's current selection."""
        inst = doc.selected()
        if inst is None:
            self.scale_spin.setEnabled(False)
            self.angle_spin.setEnabled(False)
            self.flip_btn.setEnabled(False)
            self.del_btn.setEnabled(False)
            for w in self._appearance_widgets:
                w.setEnabled(False)
            return
        locked = bool(getattr(inst, "locked", False))
        editable = not locked
        self.scale_spin.setEnabled(editable)
        self.angle_spin.setEnabled(editable)
        self.flip_btn.setEnabled(editable)
        self.del_btn.setEnabled(True)  # delete still allowed while locked
        for w in self._appearance_widgets:
            w.setEnabled(editable)
        # Block signals to avoid feedback loops.
        self.scale_spin.blockSignals(True)
        self.angle_spin.blockSignals(True)
        self.scale_spin.setValue(float(inst.h_ratio))
        self.angle_spin.setValue(float(inst.angle) % 360.0)
        self.scale_spin.blockSignals(False)
        self.angle_spin.blockSignals(False)

        for w in self._appearance_widgets:
            w.blockSignals(True)
        self.appearance_enable.setChecked(bool(inst.appearance_enabled))
        recipe = str(inst.appearance_recipe or "mild")
        idx = self.appearance_recipe.findText(recipe)
        if idx >= 0:
            self.appearance_recipe.setCurrentIndex(idx)
        else:
            self.appearance_recipe.setEditText(recipe)
        self.brightness_slider.setValue(int(round(float(inst.appearance_brightness) * 100)))
        self.contrast_slider.setValue(int(round(float(inst.appearance_contrast) * 100)))
        self.saturation_slider.setValue(int(round(float(inst.appearance_saturation) * 100)))
        self.hue_slider.setValue(int(round(float(getattr(inst, "appearance_hue", 0.0) or 0.0))))
        self.temperature_slider.setValue(
            int(round(float(getattr(inst, "appearance_temperature", 0.0) or 0.0)))
        )
        self.blur_slider.setValue(int(round(float(inst.appearance_blur) * 10)))
        self.noise_slider.setValue(int(round(float(getattr(inst, "appearance_noise", 0.0) or 0.0))))
        self.sharpness_slider.setValue(
            int(round(float(getattr(inst, "appearance_sharpness", 1.0) or 1.0) * 100))
        )
        for w in self._appearance_widgets:
            w.blockSignals(False)


class GenerationDefaultsPanel(QWidget):
    """Always-visible main-window controls for batch / large-generation defaults."""

    changed = Signal()
    apply_to_instances = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("批量 / 大规模默认")
        box.setToolTip("这些选项会用于「批量套用」和「批量数据工厂」，无需再进二级对话框才能设置。")
        form = QFormLayout(box)
        form.setContentsMargins(4, 8, 4, 4)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.scene_recipe = QComboBox()
        self.scene_recipe.setEditable(True)
        self.scene_recipe.addItems(["", "camera-mild", "surveillance", "low-light", "clean"])
        self.scene_recipe.setToolTip("整图相机域增强 Recipe；空=关闭")
        self.scene_recipe.currentTextChanged.connect(lambda *_: self.changed.emit())
        form.addRow("场景 Recipe:", self.scene_recipe)

        self.object_recipe = QComboBox()
        self.object_recipe.setEditable(True)
        self.object_recipe.addItems(["", "mild", "surveillance-object", "legacy", "off"])
        self.object_recipe.setToolTip("批量默认目标外观；空=仅用实例自身外观设置")
        self.object_recipe.currentTextChanged.connect(lambda *_: self.changed.emit())
        form.addRow("目标外观:", self.object_recipe)

        self.blend = QComboBox()
        self.blend.addItems(["alpha", "gaussian", "hard"])
        self.blend.currentTextChanged.connect(lambda *_: self.changed.emit())
        form.addRow("边缘 Blend:", self.blend)

        self.empty_scene = QDoubleSpinBox()
        self.empty_scene.setRange(0.0, 1.0)
        self.empty_scene.setDecimals(2)
        self.empty_scene.setSingleStep(0.05)
        self.empty_scene.setToolTip("仅批量数据工厂使用：纯背景负样本比例")
        self.empty_scene.valueChanged.connect(lambda *_: self.changed.emit())
        form.addRow("负样本比例:", self.empty_scene)

        self.apply_btn = QPushButton("应用到全部实例外观")
        self.apply_btn.setToolTip("把上方目标外观 Recipe 写进当前画布全部实例并开启预览")
        self.apply_btn.clicked.connect(self.apply_to_instances)
        form.addRow(self.apply_btn)

        tip = QLabel("批量套用会带上目标外观；「批量数据工厂」还会用场景 Recipe / Blend / 负样本。此处为编辑器默认，打开工厂时会自动带入。")
        tip.setWordWrap(True)
        tip.setObjectName("mutedLabel")
        form.addRow(tip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        layout.addWidget(box)
        layout.addStretch(1)

    def reflect(self, doc: Document) -> None:
        widgets = (self.scene_recipe, self.object_recipe, self.blend, self.empty_scene)
        for w in widgets:
            w.blockSignals(True)
        self._set_combo(self.scene_recipe, getattr(doc, "scene_recipe", "") or "")
        self._set_combo(self.object_recipe, getattr(doc, "object_appearance_recipe", "") or "")
        blend = str(getattr(doc, "blend_mode", "alpha") or "alpha")
        idx = self.blend.findText(blend)
        self.blend.setCurrentIndex(max(0, idx))
        self.empty_scene.setValue(float(getattr(doc, "empty_scene_prob", 0.0) or 0.0))
        for w in widgets:
            w.blockSignals(False)

    @staticmethod
    def _set_combo(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(text)

    def apply_to_document(self, doc: Document) -> None:
        doc.scene_recipe = self.scene_recipe.currentText().strip()
        doc.object_appearance_recipe = self.object_recipe.currentText().strip()
        doc.blend_mode = self.blend.currentText().strip() or "alpha"
        doc.empty_scene_prob = float(self.empty_scene.value())


class BatchProgressDialog(QDialog):
    """Modal progress dialog for ``apply_to_next`` runs.

    Connect the worker's ``progress`` signal to :meth:`on_progress` and the
    ``finished`` / ``failed`` signals to :meth:`on_finished` /
    :meth:`on_failed`. The cancel button calls ``worker.cancel()`` — the
    worker finishes the current image then exits cleanly.
    """

    def __init__(self, total: int, scope_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量套用")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._total = total
        layout = QVBoxLayout(self)
        title = QLabel(scope_text)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        self.status_label = QLabel("准备中…")
        layout.addWidget(self.status_label)
        self.bar = QProgressBar()
        self.bar.setMaximum(total + 1)
        self.bar.setValue(0)
        layout.addWidget(self.bar)
        self.cancel_btn = QPushButton("取消（剩余跳过）")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)
        self._worker = None  # set by caller via set_worker

    def set_worker(self, worker) -> None:
        self._worker = worker

    def on_progress(self, done: int, total: int, name: str) -> None:
        self._total = total
        self.bar.setMaximum(total + 1)
        self.bar.setValue(done + 1)
        self.status_label.setText(f"{done + 1} / {total} — {name}")

    def on_finished(self, summary: dict) -> None:
        gen = summary.get("generated", 0)
        fail = summary.get("failed", 0)
        cancel = summary.get("cancelled", False)
        msg = f"完成：生成 {gen} 张"
        if fail:
            msg += f"，失败 {fail} 张"
        if cancel:
            msg += "（已取消）"
        QMessageBox.information(self, "批量套用", msg)
        self.accept()

    def on_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "批量套用失败", msg)
        self.reject()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("取消中…")


class ProjectSettingsDialog(QDialog):
    """Modal dialog for project-level options.

    Edits the Document's ``output_format`` / ``do_shadow`` / ``do_color_match``
    / ``auto_save`` / ``keep_position`` / ``class_map_text`` in place. The
    caller passes the live Document; changes apply on accept.
    """

    FORMATS = ["detect", "seg", "both", "coco", "semantic", "obb", "all"]

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.setWindowTitle("项目设置")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._doc = doc

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems(self.FORMATS)
        self.format_combo.setCurrentText(doc.output_format or "detect")
        form.addRow("输出格式:", self.format_combo)

        self.class_map_edit = QLineEdit(doc.class_map_text)
        form.addRow("类别映射:", self.class_map_edit)

        self.shadow_check = QCheckBox("合成时加脚底阴影")
        self.shadow_check.setChecked(doc.do_shadow)
        form.addRow("", self.shadow_check)

        self.color_match_check = QCheckBox("前景色调匹配背景")
        self.color_match_check.setChecked(doc.do_color_match)
        form.addRow("", self.color_match_check)

        self.auto_save_check = QCheckBox("切换背景自动保存")
        self.auto_save_check.setChecked(doc.auto_save)
        form.addRow("", self.auto_save_check)

        self.keep_pos_check = QCheckBox("切换保持相对位置")
        self.keep_pos_check.setChecked(doc.keep_position)
        form.addRow("", self.keep_pos_check)

        self.scene_recipe = QComboBox()
        self.scene_recipe.setEditable(True)
        self.scene_recipe.addItems(["", "camera-mild", "surveillance", "low-light", "clean"])
        GenerationDefaultsPanel._set_combo(self.scene_recipe, getattr(doc, "scene_recipe", "") or "")
        form.addRow("场景 Recipe:", self.scene_recipe)

        self.object_recipe = QComboBox()
        self.object_recipe.setEditable(True)
        self.object_recipe.addItems(["", "mild", "surveillance-object", "legacy", "off"])
        GenerationDefaultsPanel._set_combo(
            self.object_recipe, getattr(doc, "object_appearance_recipe", "") or "",
        )
        form.addRow("目标外观 Recipe:", self.object_recipe)

        self.blend = QComboBox()
        self.blend.addItems(["alpha", "gaussian", "hard"])
        blend = str(getattr(doc, "blend_mode", "alpha") or "alpha")
        idx = self.blend.findText(blend)
        self.blend.setCurrentIndex(max(0, idx))
        form.addRow("边缘 Blend:", self.blend)

        self.empty_scene = QDoubleSpinBox()
        self.empty_scene.setRange(0.0, 1.0)
        self.empty_scene.setDecimals(2)
        self.empty_scene.setSingleStep(0.05)
        self.empty_scene.setValue(float(getattr(doc, "empty_scene_prob", 0.0) or 0.0))
        form.addRow("负样本比例:", self.empty_scene)

        tip = QLabel(
            "Recipe / Blend / 负样本与右侧「批量默认」同步到工程 Document；"
            "大规模合成请以「批量数据工厂」为准，编辑器里改完再打开工厂即可带入。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("mutedLabel")
        layout.addLayout(form)
        layout.addWidget(tip)

        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def apply_to_document(self) -> bool:
        """If the user accepted, push edits into the document. Returns True
        if anything changed."""
        if self.exec() != QDialog.Accepted:
            return False
        changed = False
        if self._doc.output_format != self.format_combo.currentText():
            self._doc.output_format = self.format_combo.currentText()
            changed = True
        if self._doc.class_map_text != self.class_map_edit.text():
            self._doc.class_map_text = self.class_map_edit.text()
            changed = True
        for attr, widget in [("do_shadow", self.shadow_check),
                             ("do_color_match", self.color_match_check),
                             ("auto_save", self.auto_save_check),
                             ("keep_position", self.keep_pos_check)]:
            if getattr(self._doc, attr) != widget.isChecked():
                setattr(self._doc, attr, widget.isChecked())
                changed = True
        new_scene = self.scene_recipe.currentText().strip()
        new_object = self.object_recipe.currentText().strip()
        new_blend = self.blend.currentText().strip() or "alpha"
        new_empty = float(self.empty_scene.value())
        if self._doc.scene_recipe != new_scene:
            self._doc.scene_recipe = new_scene
            changed = True
        if self._doc.object_appearance_recipe != new_object:
            self._doc.object_appearance_recipe = new_object
            changed = True
        if self._doc.blend_mode != new_blend:
            self._doc.blend_mode = new_blend
            changed = True
        if abs(float(self._doc.empty_scene_prob) - new_empty) > 1e-9:
            self._doc.empty_scene_prob = new_empty
            changed = True
        if changed:
            self._doc._emit()
        return changed


class TemplateParametersDialog(QDialog):
    """Configure stochastic variation before saving a Scene Template v2."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scene Template 参数化")
        self.setModal(True)
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        hint = QLabel("这些范围只影响批量合成；重新加载模板到编辑器时仍显示名义布局。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.x_jitter = QDoubleSpinBox(); self.x_jitter.setRange(0, .5); self.x_jitter.setDecimals(3); self.x_jitter.setValue(.03)
        self.y_jitter = QDoubleSpinBox(); self.y_jitter.setRange(0, .5); self.y_jitter.setDecimals(3); self.y_jitter.setValue(.02)
        self.scale_jitter = QDoubleSpinBox(); self.scale_jitter.setRange(0, 1.0); self.scale_jitter.setDecimals(2); self.scale_jitter.setValue(.10)
        self.angle_jitter = QDoubleSpinBox(); self.angle_jitter.setRange(0, 180); self.angle_jitter.setSuffix(" °"); self.angle_jitter.setValue(5)
        self.instance_prob = QDoubleSpinBox(); self.instance_prob.setRange(0,1); self.instance_prob.setSingleStep(.05); self.instance_prob.setDecimals(2); self.instance_prob.setValue(1.0)
        self.flip_prob = QDoubleSpinBox(); self.flip_prob.setRange(-1,1); self.flip_prob.setSingleStep(.1); self.flip_prob.setDecimals(2); self.flip_prob.setValue(-1); self.flip_prob.setSpecialValueText("保持原值")
        self.same_class = QCheckBox("同类别随机换素材")
        self.allow_overlap = QCheckBox("保留模板内部遮挡/重叠"); self.allow_overlap.setChecked(True)
        form.addRow("X 位置抖动(比例):", self.x_jitter)
        form.addRow("Y 位置抖动(比例):", self.y_jitter)
        form.addRow("尺度抖动(±比例):", self.scale_jitter)
        form.addRow("角度抖动(±):", self.angle_jitter)
        form.addRow("每实例出现概率:", self.instance_prob)
        form.addRow("随机翻转概率:", self.flip_prob)
        form.addRow("", self.same_class); form.addRow("", self.allow_overlap)
        layout.addLayout(form)
        row = QHBoxLayout(); row.addStretch(1)
        ok = QPushButton("保存参数化模板"); ok.clicked.connect(self.accept)
        cancel = QPushButton("取消"); cancel.clicked.connect(self.reject)
        row.addWidget(ok); row.addWidget(cancel); layout.addLayout(row)

    def parameters(self) -> dict:
        fp = float(self.flip_prob.value())
        return {
            "position_jitter_x": float(self.x_jitter.value()),
            "position_jitter_y": float(self.y_jitter.value()),
            "scale_jitter": float(self.scale_jitter.value()),
            "angle_jitter": float(self.angle_jitter.value()),
            "instance_probability": float(self.instance_prob.value()),
            "flip_probability": None if fp < 0 else fp,
            "same_class_random": self.same_class.isChecked(),
            "allow_overlap": self.allow_overlap.isChecked(),
        }
