"""QGraphicsView-based canvas with draggable, scalable, rotatable instances.

Replaces the hand-written hit-test/transform code in the tkinter
``CanvasController``. Qt's Graphics View framework handles the math
(selection, drag, transform handles, z-order) natively and reliably.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import QPixmap, QBrush, QColor, QImage, QTransform, QPainter
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QGraphicsItem, QApplication,
)

from compose_app.models import Cutout, Instance
from compose_app.rendering import bbox_of_rendered
from .state import Document


def _pil_rgba_to_qpixmap(rgba: Image.Image) -> QPixmap:
    """Convert a PIL RGBA image to a QPixmap safely (no buffer aliasing)."""
    arr = np.ascontiguousarray(np.asarray(rgba.convert("RGBA")), dtype=np.uint8)
    h, w, _ = arr.shape
    # RGBA8888 matches PIL channel order and preserves alpha on all platforms.
    qimg = QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format_RGBA8888)
    # Copy detaches from the NumPy buffer lifetime.
    return QPixmap.fromImage(qimg.copy())


class InstanceItem(QGraphicsPixmapItem):
    """One placed instance on the canvas.

    Stores a back-reference to the :class:`Instance` model object so the
    view can sync geometry → model on every drag/transform.
    """

    def __init__(self, inst: Instance, cutout: Cutout, doc: Document, view: "CanvasView"):
        super().__init__()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setTransformationMode(Qt.SmoothTransformation)
        self.inst = inst
        self.cutout = cutout
        self._doc = doc
        self._view = view
        self._sync_pixmap()
        self._sync_pose()

    # ------------------------------------------------------------ rendering
    def _sync_pixmap(self) -> None:
        """Bake flip/angle into the pixmap (same path as save/composite).

        Do **not** also apply Qt rotation/flip — that would double-transform
        and diverge from exported labels.
        """
        bg_w, bg_h = self._doc.bg_size
        if bg_h <= 0:
            bg_h = self.cutout.rgba.height
        target_h = max(4, int(round(self.inst.h_ratio * bg_h)))
        rendered = self.inst.get_rendered(
            self.cutout.rgba, target_h, class_label=self.cutout.label,
        )
        self._rendered = rendered
        self._bbox_offset = bbox_of_rendered(rendered)
        pm = _pil_rgba_to_qpixmap(rendered)
        self.setPixmap(pm)
        # Item pos is the centre of the full rendered frame (matches paste math).
        self.setOffset(-rendered.width / 2.0, -rendered.height / 2.0)

    def _sync_pose(self) -> None:
        """Apply model cx/cy only — flip/angle already baked into the pixmap."""
        self.setPos(QPointF(self.inst.cx, self.inst.cy))
        self.setTransform(QTransform())
        self.setRotation(0.0)
        # Reflect per-instance layer state.
        self.setVisible(bool(getattr(self.inst, "visible", True)))
        locked = bool(getattr(self.inst, "locked", False))
        self.setFlag(QGraphicsItem.ItemIsMovable, not locked)
        self.setFlag(QGraphicsItem.ItemIsSelectable, not locked)

    def commit_geometry_to_model(self) -> None:
        """Push the current item pos back into the Instance model."""
        pos = self.pos()
        self.inst.cx = float(pos.x())
        self.inst.cy = float(pos.y())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            # Avoid feedback loop: only propagate user-initiated selection.
            if not self._view._syncing:
                selected = bool(value.toInt()[0]) if hasattr(value, "toInt") else bool(value)
                if selected:
                    self._doc.select(self.inst.uid)
        elif change == QGraphicsItem.ItemPositionHasChanged:
            # Ignore programmatic setPos during doc→view sync.
            if self._view._syncing:
                return super().itemChange(change, value)
            # Drag in progress: keep model in sync without triggering redraw.
            pos = self.pos()
            self.inst.cx = float(pos.x())
            self.inst.cy = float(pos.y())
            self._view.instance_moved.emit(self.inst.uid)
        return super().itemChange(change, value)

    # --------------------------------------------------------- transform ops
    def set_scale_ratio(self, h_ratio: float) -> None:
        if bool(getattr(self.inst, "locked", False)):
            return
        self.inst.h_ratio = max(0.02, min(2.0, float(h_ratio)))
        self.inst.invalidate_cache()
        self._sync_pixmap()
        self._sync_pose()
        self.update()

    def set_angle(self, deg: float) -> None:
        if bool(getattr(self.inst, "locked", False)):
            return
        self.inst.angle = float(deg) % 360.0
        self.inst.invalidate_cache()
        self._sync_pixmap()
        self._sync_pose()
        self.update()

    def toggle_flip(self) -> None:
        if bool(getattr(self.inst, "locked", False)):
            return
        self.inst.flip = not self.inst.flip
        self.inst.invalidate_cache()
        self._sync_pixmap()
        self._sync_pose()
        self.update()


class CanvasView(QGraphicsView):
    """The interactive scene view.

    - Drag: move selected instance
    - Wheel: scale selected instance (Ctrl+wheel zooms the view)
    - Shift+drag: rotate selected instance
    """

    instance_moved = Signal(int)       # uid
    instance_placed = Signal(int)      # uid
    instance_removed = Signal(int)     # uid
    background_clicked = Signal(QPointF)
    # Drop / request place: (cutout_index, scene QPointF) — host pushes undo.
    place_at_requested = Signal(int, QPointF)
    # Live scale/rotate while dragging/wheeling — refresh spinboxes only.
    transform_preview = Signal()
    # Emitted after an interactive scale/rotate/move gesture should enter undo.
    # Payload: list of Instance clones captured *before* the gesture.
    transform_committed = Signal(object)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self._doc = doc
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self.setBackgroundBrush(QBrush(QColor(28, 28, 30)))
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._instance_items: List[InstanceItem] = []
        self._zoom = 1.0
        # When True, resize/new-bg will fitInView; Ctrl+wheel clears this.
        self._auto_fit = True
        self._rotate_gesture = None  # dict while Shift+drag rotating
        self._move_gesture = None    # dict while dragging to move
        self._scale_undo_before = None
        self._scale_commit_timer = QTimer(self)
        self._scale_commit_timer.setSingleShot(True)
        self._scale_commit_timer.setInterval(400)
        self._scale_commit_timer.timeout.connect(self._commit_scale_undo)
        # Reentrancy guard: breaks the itemChange -> select -> _emit ->
        # _on_doc_changed -> setSelected -> itemChange loop.
        self._syncing = False
        # When the doc mutates externally (undo/redo, programmatic add),
        # rebuild the scene.
        doc.subscribe(self._on_doc_changed)

    # ---------------------------------------------------------- doc -> view
    def _on_doc_changed(self) -> None:
        """Re-sync the scene from the document model."""
        if self._syncing:
            return
        self._syncing = True
        try:
            # Background
            bg_path = self._doc.current_background()
            if bg_path is not None:
                self._load_background(bg_path)
            # Diff instances by uid: rebuild when sets diverge.
            model_uids = {i.uid for i in self._doc.instances}
            view_uids = {item.inst.uid for item in self._instance_items}
            if model_uids != view_uids:
                self._rebuild_instances()
            else:
                # Undo/redo / layer reorder replaces Instance objects and may
                # change list order without changing the uid set — rebind and
                # refresh z-order so 置顶/上移/下移/置底 actually restack.
                by_uid = {i.uid: i for i in self._doc.instances}
                uid_to_z = {
                    inst.uid: float(z)
                    for z, inst in enumerate(self._doc.instances, start=1)
                }
                for item in self._instance_items:
                    item.inst = by_uid[item.inst.uid]
                    item.setZValue(uid_to_z[item.inst.uid])
                    item._sync_pixmap()
                    item._sync_pose()
                self._instance_items.sort(
                    key=lambda it: uid_to_z.get(it.inst.uid, 0.0)
                )
                # Ensure the scene repaints with the new stacking order.
                self._scene.update()
            # Selection
            sel = self._doc.selected_uid
            for item in self._instance_items:
                item.setSelected(item.inst.uid == sel)
        finally:
            self._syncing = False

    def _load_background(self, path) -> None:
        # Read directly to keep Qt package decoupled from tkinter-era io.
        try:
            import scenepaste.core as core
            arr = core.imread_with_exif(path)
            if arr is None:
                return
            h, w = arr.shape[:2]
            # Update bg_size WITHOUT re-emitting (we're inside _on_doc_changed
            # already, and the recursion would be wasteful).
            self._doc.bg_size = (int(w), int(h))
            # BGR → RGB → QImage
            rgb = arr[..., ::-1].astype(np.uint8).tobytes()
            qimg = QImage(rgb, w, h, 3 * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(qimg.copy())
        except Exception:
            return
        if self._bg_item is None:
            self._bg_item = self._scene.addPixmap(pm)
            self._auto_fit = True
        else:
            self._bg_item.setPixmap(pm)
        self._scene.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))
        if self._auto_fit:
            self.fitInView(self._bg_item, Qt.KeepAspectRatio)
            # Relative zoom factor vs current fit; never wipe fit via setTransform.
            self._zoom = 1.0

    def _rebuild_instances(self) -> None:
        # Remove stale InstanceItems.
        keep = []
        existing = {item.inst.uid: item for item in self._instance_items}
        for inst in self._doc.instances:
            item = existing.get(inst.uid)
            if item is None:
                if 0 <= inst.cutout_index < len(self._doc.cutouts):
                    cutout = self._doc.cutouts[inst.cutout_index]
                    item = InstanceItem(inst, cutout, self._doc, self)
                    self._scene.addItem(item)
            else:
                item.inst = inst
                if 0 <= inst.cutout_index < len(self._doc.cutouts):
                    item.cutout = self._doc.cutouts[inst.cutout_index]
                item._sync_pixmap()
                item._sync_pose()
            if item is not None:
                item.setZValue(len(keep) + 1)
                keep.append(item)
        # Drop items no longer in the model.
        new_uids = {i.uid for i in self._doc.instances}
        for item in self._instance_items:
            if item.inst.uid not in new_uids:
                self._scene.removeItem(item)
        self._instance_items = keep

    def _selected_item(self) -> Optional[InstanceItem]:
        uid = self._doc.selected_uid
        if uid is None:
            return None
        for item in self._instance_items:
            if item.inst.uid == uid:
                return item
        return None

    def _flush_pending_scale_undo(self) -> None:
        self._scale_commit_timer.stop()
        self._commit_scale_undo()

    def reset_gestures(self) -> None:
        """Abort / commit in-flight edit gestures (e.g. on background change)."""
        self._flush_pending_scale_undo()
        self._rotate_gesture = None
        self._move_gesture = None
        self.setDragMode(QGraphicsView.RubberBandDrag)

    # -------------------------------------------------------------- events
    def wheelEvent(self, event) -> None:
        # Ctrl+wheel → zoom the view; plain wheel → scale selected cutout.
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            step = delta / 1200.0 if delta else 0.0
            if step == 0.0:
                event.accept()
                return
            new_zoom = max(0.1, min(8.0, self._zoom * (1.0 + step)))
            factor = new_zoom / max(self._zoom, 1e-9)
            self._zoom = new_zoom
            self._auto_fit = False
            # Relative scale keeps fitInView / pan and honours AnchorUnderMouse.
            self.scale(factor, factor)
            event.accept()
            return

        item = self._selected_item()
        if item is None or delta == 0:
            super().wheelEvent(event)
            return
        if bool(getattr(item.inst, "locked", False)):
            event.accept()
            return
        if self._scale_undo_before is None:
            self._scale_undo_before = self._doc.snapshot()
        factor = 1.10 if delta > 0 else 1.0 / 1.10
        item.set_scale_ratio(item.inst.h_ratio * factor)
        # Lightweight UI sync (avoid full scene rebuild on every wheel tick).
        self.transform_preview.emit()
        self._scale_commit_timer.start()
        event.accept()

    def _commit_scale_undo(self) -> None:
        if self._scale_undo_before is None:
            return
        before = self._scale_undo_before
        self._scale_undo_before = None
        self.transform_committed.emit(before)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            # Finish any in-flight wheel-scale undo before a new gesture.
            self._flush_pending_scale_undo()

        if (
            event.button() == Qt.LeftButton
            and (event.modifiers() & Qt.ShiftModifier)
        ):
            item = self._selected_item()
            # Prefer the item under the cursor when Shift+clicking a cutout.
            under = self.itemAt(event.pos())
            if isinstance(under, InstanceItem):
                item = under
                if not self._syncing and not bool(getattr(item.inst, "locked", False)):
                    self._doc.select(item.inst.uid)
            if item is not None and not bool(getattr(item.inst, "locked", False)):
                scene_pos = self.mapToScene(event.pos())
                origin = item.pos()
                self._rotate_gesture = {
                    "item": item,
                    "before": self._doc.snapshot(),
                    "start_angle": float(item.inst.angle),
                    "origin": origin,
                    "start_mouse_angle": math.degrees(
                        math.atan2(scene_pos.y() - origin.y(),
                                   scene_pos.x() - origin.x())
                    ),
                }
                self.setDragMode(QGraphicsView.NoDrag)
                event.accept()
                return

        if event.button() == Qt.LeftButton and not (event.modifiers() & Qt.ShiftModifier):
            under = self.itemAt(event.pos())
            if isinstance(under, InstanceItem) and not bool(
                getattr(under.inst, "locked", False)
            ):
                self._move_gesture = {
                    "uid": under.inst.uid,
                    "before": self._doc.snapshot(),
                    "start_cx": float(under.inst.cx),
                    "start_cy": float(under.inst.cy),
                }

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        gesture = self._rotate_gesture
        if gesture is not None:
            item = gesture["item"]
            origin = gesture["origin"]
            scene_pos = self.mapToScene(event.pos())
            cur = math.degrees(
                math.atan2(scene_pos.y() - origin.y(), scene_pos.x() - origin.x())
            )
            item.set_angle(gesture["start_angle"] + (cur - gesture["start_mouse_angle"]))
            self.transform_preview.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._rotate_gesture is not None and event.button() == Qt.LeftButton:
            gesture = self._rotate_gesture
            before = gesture["before"]
            start_angle = gesture["start_angle"]
            item = gesture["item"]
            self._rotate_gesture = None
            self.setDragMode(QGraphicsView.RubberBandDrag)
            # Skip no-op Shift+click with no angle change.
            if abs((float(item.inst.angle) - float(start_angle) + 180.0) % 360.0 - 180.0) > 1e-3:
                self.transform_committed.emit(before)
            event.accept()
            return

        super().mouseReleaseEvent(event)

        if self._move_gesture is not None and event.button() == Qt.LeftButton:
            gesture = self._move_gesture
            self._move_gesture = None
            uid = gesture["uid"]
            inst = None
            for candidate in self._doc.instances:
                if candidate.uid == uid:
                    inst = candidate
                    break
            if inst is not None and (
                abs(inst.cx - gesture["start_cx"]) > 0.5
                or abs(inst.cy - gesture["start_cy"]) > 0.5
            ):
                self.transform_committed.emit(gesture["before"])

    def leaveEvent(self, event) -> None:
        self._flush_pending_scale_undo()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            self._flush_pending_scale_undo()
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._bg_item is not None and self._auto_fit:
            self.fitInView(self._bg_item, Qt.KeepAspectRatio)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-cutout-index"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        md = event.mimeData()
        if md.hasFormat("application/x-cutout-index"):
            idx_bytes = bytes(md.data("application/x-cutout-index"))
            try:
                idx = int(idx_bytes.decode("ascii"))
            except ValueError:
                return
            pos = self.mapToScene(event.position().toPoint())
            self.place_at_requested.emit(idx, pos)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
