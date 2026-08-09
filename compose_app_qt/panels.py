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
    QListWidgetItem, QDoubleSpinBox, QLineEdit,
    QFormLayout, QDialog, QProgressBar, QMessageBox, QCheckBox,
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("搜索类别…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._refresh)
        layout.addWidget(self._filter)

        self._list = _CutoutList()
        self._list.setIconSize(QSize(*THUMB_SIZE))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setFlow(QListWidget.LeftToRight)
        self._list.setWrapping(True)
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


class InstanceListWidget(QListWidget):
    """Right-panel list of placed instances; click to select on canvas."""

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self._doc = doc
        self.itemClicked.connect(self._on_click)
        doc.subscribe(self._refresh)

    def _refresh(self) -> None:
        current = self.currentRow()
        self.clear()
        for i, inst in enumerate(self._doc.instances):
            label = f"#{inst.uid}  {self._doc.cutouts[inst.cutout_index].label}" \
                    if 0 <= inst.cutout_index < len(self._doc.cutouts) else f"#{inst.uid}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, inst.uid)
            self.addItem(item)
            if inst.uid == self._doc.selected_uid:
                self.setCurrentRow(i)
        if current >= 0 and current < self.count() and self._doc.selected_uid is None:
            self.setCurrentRow(current)

    def _on_click(self, item: QListWidgetItem) -> None:
        uid = int(item.data(Qt.UserRole))
        self._doc.select(uid)


class ControlsPanel(QWidget):
    """Scale / rotation / flip / delete controls for the selected instance."""

    scale_changed = Signal(float)
    angle_changed = Signal(float)
    flip_toggled = Signal()
    delete_clicked = Signal()
    nudge = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        tip = QLabel("画布：滚轮缩放 · Shift+拖动旋转")
        tip.setStyleSheet("color: #888; font-size: 11px;")
        tip.setWordWrap(True)
        layout.addWidget(tip)

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
        self.flip_btn = QPushButton("水平翻转")
        self.flip_btn.clicked.connect(self.flip_toggled)
        btn_row.addWidget(self.flip_btn)
        self.del_btn = QPushButton("删除")
        self.del_btn.clicked.connect(self.delete_clicked)
        btn_row.addWidget(self.del_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)

    def reflect(self, doc: Document) -> None:
        """Update spinboxes from the document's current selection."""
        inst = doc.selected()
        if inst is None:
            self.scale_spin.setEnabled(False)
            self.angle_spin.setEnabled(False)
            self.flip_btn.setEnabled(False)
            self.del_btn.setEnabled(False)
            return
        self.scale_spin.setEnabled(True)
        self.angle_spin.setEnabled(True)
        self.flip_btn.setEnabled(True)
        self.del_btn.setEnabled(True)
        # Block signals to avoid feedback loops.
        self.scale_spin.blockSignals(True)
        self.angle_spin.blockSignals(True)
        self.scale_spin.setValue(float(inst.h_ratio))
        self.angle_spin.setValue(float(inst.angle) % 360.0)
        self.scale_spin.blockSignals(False)
        self.angle_spin.blockSignals(False)


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

    FORMATS = ["detect", "seg", "both", "coco", "semantic", "obb"]

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.setWindowTitle("项目设置")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._doc = doc

        layout = QVBoxLayout(self)
        form = QFormLayout()

        from PySide6.QtWidgets import QComboBox
        self.format_combo = QComboBox()
        self.format_combo.addItems(self.FORMATS)
        self.format_combo.setCurrentText(doc.output_format or "detect")
        form.addRow("输出格式:", self.format_combo)

        self.class_map_edit = QLineEdit(doc.class_map_text)
        form.addRow("类别映射:", self.class_map_edit)

        from PySide6.QtWidgets import QCheckBox
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

        layout.addLayout(form)

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
        hint = QLabel("这些范围只影响批量生成；重新加载模板到编辑器时仍显示名义布局。")
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
