"""Controllers extracted from MainWindow.

Owns the undo-aware mutation paths: transform / appearance / instance ops /
layer ordering / visibility / lock. The :class:`MainWindow` wires its panels'
signals and shortcut activations to methods here.

Rationale: keep ``app.py`` focused on UI assembly + dialog routing; collect
the snapshot/push logic in one place so layer/visibility features share the
same undo plumbing as transforms.
"""

from __future__ import annotations

import dataclasses
import random
from typing import List, Optional

from compose_app.models import Instance
from .state import Document
from .undo import (
    AddInstanceCommand,
    DeleteSelectedCommand,
    PropertyToggleCommand,
    ReorderCommand,
    TransformCommand,
)


# ----------------------------------------------------------- snapshot helpers
# Fields that must match *exactly* between two snapshots for "no change".
_EXACT_FIELDS = {
    "cutout_index", "flip", "uid",
    "appearance_enabled", "appearance_recipe", "appearance_seed",
    "visible", "locked",
}
# Numeric fields with a per-field epsilon for float comparison.
_NUMERIC_TOL: dict = {
    "cx": 1e-6, "cy": 1e-6, "h_ratio": 1e-9, "angle": 1e-6,
    "appearance_brightness": 1e-6,
    "appearance_contrast": 1e-6,
    "appearance_saturation": 1e-6,
    "appearance_blur": 1e-6,
    "appearance_hue": 1e-6,
    "appearance_temperature": 1e-6,
    "appearance_noise": 1e-6,
    "appearance_sharpness": 1e-6,
}
# Fields excluded from any comparison (caches / private state).
_SKIPPED_FIELDS = {"_cache"}


def _instance_equal(a: Instance, b: Instance) -> bool:
    """Field-driven equality so new appearance fields can't silently skip undo."""
    if a is b:
        return True
    for f in dataclasses.fields(Instance):
        name = f.name
        if name in _SKIPPED_FIELDS:
            continue
        va = getattr(a, name)
        vb = getattr(b, name)
        if name in _EXACT_FIELDS:
            if va != vb:
                return False
        elif name in _NUMERIC_TOL:
            if name == "angle":
                # 370° ≡ 10° — compare in canonical [0, 360) space.
                if abs((float(va) % 360.0) - (float(vb) % 360.0)) > _NUMERIC_TOL[name]:
                    return False
            elif abs(float(va) - float(vb)) > _NUMERIC_TOL[name]:
                return False
        else:
            if va != vb:
                return False
    return True


def _instance_snapshots_equal(a: List[Instance], b: List[Instance]) -> bool:
    """True when two undo snapshots describe the same layout (same order + values).

    Order matters because list position encodes canvas z-order: a reorder that
    only swaps list positions must still produce an undo entry.
    """
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left.uid != right.uid:
            return False
        if not _instance_equal(left, right):
            return False
    return True


# ------------------------------------------------------------------ the controller
class MainWindowController:
    """Routes panel/shortcut signals to undo-aware document mutations.

    Holds a back-reference to :class:`MainWindow` to reach the doc, the
    undo stack, and the controls panel. Methods are public (no leading
    underscore) so they can be bound directly to QAction / QShortcut.
    """

    def __init__(self, main_window):
        self.mw = main_window
        self._appearance_before: Optional[List[Instance]] = None
        self._scale_step = 1.10  # per [ / ] keystroke
        self._last_sample_recipe = "mild"

    # ---------------------------------------------------------- accessors
    @property
    def doc(self) -> Document:
        return self.mw.doc

    @property
    def undo_stack(self):
        return self.mw._undo_stack

    @property
    def controls(self):
        return self.mw.controls

    def _selected_editable(self) -> Optional[Instance]:
        """Return the selection only when it exists and is unlocked.

        Locked instances may still be selected in the layer list (so users can
        unlock / reorder / delete them) but all transform / appearance edits
        are rejected here — matching the canvas lock flags.
        """
        inst = self.doc.selected()
        if inst is None or bool(getattr(inst, "locked", False)):
            return None
        return inst

    # ----------------------------------------------------------- placement
    def add_instance(self, cutout_index: int, cx: float, cy: float) -> None:
        doc = self.doc
        if not (0 <= cutout_index < len(doc.cutouts)):
            return
        cutout = doc.cutouts[cutout_index]
        bg_h = doc.bg_size[1] or cutout.rgba.height
        inst = Instance(
            cutout_index=cutout_index,
            cx=cx, cy=cy,
            h_ratio=max(0.05, min(0.5, cutout.rgba.height / max(1, bg_h) * 0.5)),
            uid=doc.next_uid(),
        )
        self.undo_stack.push(AddInstanceCommand(doc, inst))

    def place_at_centre(self, cutout_index: int) -> None:
        cx, cy = self.doc.bg_size[0] / 2.0, self.doc.bg_size[1] / 2.0
        self.add_instance(cutout_index, cx, cy)

    # ------------------------------------------------------- transformations
    def on_scale_changed(self, value: float) -> None:
        inst = self._selected_editable()
        if inst is None or abs(float(inst.h_ratio) - float(value)) < 1e-6:
            return
        before = self.doc.snapshot()
        inst.h_ratio = float(value)
        inst.invalidate_cache()
        self._push_transform(before)

    def on_angle_changed(self, value: float) -> None:
        inst = self._selected_editable()
        if inst is None:
            return
        angle = float(value) % 360.0
        if abs(float(inst.angle) % 360.0 - angle) < 1e-6:
            return
        before = self.doc.snapshot()
        inst.angle = angle
        inst.invalidate_cache()
        self._push_transform(before)

    def on_canvas_transform_committed(self, before) -> None:
        if before is None:
            return
        # Drop commits that somehow slipped past canvas lock guards.
        after_sel = self.doc.selected()
        if after_sel is not None and bool(getattr(after_sel, "locked", False)):
            return
        self._push_transform(before)

    def on_flip(self) -> None:
        inst = self._selected_editable()
        if inst is None:
            return
        before = self.doc.snapshot()
        inst.flip = not inst.flip
        inst.invalidate_cache()
        self._push_transform(before)

    def scale_selected(self, factor: float) -> None:
        """Scale the selected instance by ``factor`` (used by [ / ] shortcuts)."""
        inst = self._selected_editable()
        if inst is None:
            return
        new_h = max(0.02, min(2.0, float(inst.h_ratio) * float(factor)))
        if abs(new_h - float(inst.h_ratio)) < 1e-9:
            return
        before = self.doc.snapshot()
        inst.h_ratio = new_h
        inst.invalidate_cache()
        self._push_transform(before)
        self.controls.reflect(self.doc)
        self.doc._emit()

    # --------------------------------------------------------------- appearance
    def _apply_appearance_from_controls(self) -> None:
        inst = self._selected_editable()
        if inst is None:
            return
        vals = self.controls.appearance_values()
        enabled = bool(vals["enabled"])
        # Moving a slider implies the user wants a live preview. Without this,
        # values were stored but ``appearance_enabled=False`` skipped rendering
        # until 「换一种随机外观」 forced enable=True.
        has_slider_tweak = (
            abs(float(vals["brightness"])) > 1e-6
            or abs(float(vals["contrast"]) - 1.0) > 1e-6
            or abs(float(vals["saturation"]) - 1.0) > 1e-6
            or abs(float(vals.get("hue", 0.0))) > 1e-6
            or abs(float(vals.get("temperature", 0.0))) > 1e-6
            or float(vals["blur"]) > 1e-6
            or float(vals.get("noise", 0.0)) > 1e-6
            or abs(float(vals.get("sharpness", 1.0)) - 1.0) > 1e-6
        )
        if has_slider_tweak and not enabled:
            enabled = True
            cb = self.controls.appearance_enable
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        inst.appearance_enabled = enabled
        inst.appearance_recipe = str(vals["recipe"] or "mild")
        inst.appearance_brightness = float(vals["brightness"])
        inst.appearance_contrast = float(vals["contrast"])
        inst.appearance_saturation = float(vals["saturation"])
        inst.appearance_blur = float(vals["blur"])
        inst.appearance_hue = float(vals.get("hue", 0.0))
        inst.appearance_temperature = float(vals.get("temperature", 0.0))
        inst.appearance_noise = float(vals.get("noise", 0.0))
        inst.appearance_sharpness = float(vals.get("sharpness", 1.0))
        if enabled and int(inst.appearance_seed or 0) == 0:
            inst.appearance_seed = int(random.randint(1, 2**31 - 1))
        inst.invalidate_cache()

    def on_appearance_preview(self) -> None:
        if self._selected_editable() is None:
            return
        if self._appearance_before is None:
            self._appearance_before = self.doc.snapshot()
        self._apply_appearance_from_controls()
        self.doc._emit()

    def on_appearance_committed(self) -> None:
        if self._selected_editable() is None:
            self._appearance_before = None
            return
        if self._appearance_before is None:
            self._appearance_before = self.doc.snapshot()
            self._apply_appearance_from_controls()
            self.doc._emit()
        self._push_transform(self._appearance_before)
        self._appearance_before = None

    def on_appearance_resample(self) -> None:
        inst = self._selected_editable()
        if inst is None:
            return
        before = self.doc.snapshot()
        # Prefer the recipe currently shown in the combo; if it is "off"
        # (after a previous bake), reuse the last sampled recipe.
        recipe_name = str(self.controls.appearance_recipe.currentText() or "").strip()
        if not recipe_name or recipe_name.lower() == "off":
            recipe_name = str(self._last_sample_recipe or "mild")
        if recipe_name.lower() == "off":
            recipe_name = "mild"
        self._last_sample_recipe = recipe_name

        inst.appearance_enabled = True
        inst.appearance_seed = int(random.randint(1, 2**31 - 1))
        inst.appearance_recipe = recipe_name

        # Sample recipe → bake applied params into sliders (recipe becomes "off").
        from compose_app.rendering import sample_recipe_into_sliders

        cutout = None
        if 0 <= inst.cutout_index < len(self.doc.cutouts):
            cutout = self.doc.cutouts[inst.cutout_index]
        if cutout is not None:
            sample_recipe_into_sliders(
                inst,
                cutout.rgba,
                class_label=cutout.label,
                recipe_name=recipe_name,
            )
        else:
            # No pixels to sample against — still clear stale slider overrides.
            from compose_app.rendering import reset_appearance_sliders
            reset_appearance_sliders(inst)
            inst.appearance_recipe = "off"

        self._push_transform(before)
        self.controls.reflect(self.doc)
        self.doc._emit()

        sb = self.mw.statusBar() if hasattr(self.mw, "statusBar") else None
        if sb is not None:
            sb.showMessage(
                f"已从「{recipe_name}」采样并回填滑条（可继续微调）", 2200
            )

    # ----------------------------------------------------------- instance ops
    def _push_transform(self, before) -> None:
        after = self.doc.snapshot()
        if _instance_snapshots_equal(before, after):
            return
        self.undo_stack.push(TransformCommand(self.doc, before))

    def delete_selected(self) -> None:
        if self.doc.selected_uid is None:
            return
        self.undo_stack.push(DeleteSelectedCommand(self.doc))

    def duplicate_selected(self) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        clone = inst.clone()
        clone.cx += 16
        clone.cy += 16
        clone.uid = self.doc.next_uid()
        self.undo_stack.push(AddInstanceCommand(self.doc, clone))

    def nudge(self, dx: int, dy: int) -> None:
        inst = self._selected_editable()
        if inst is None:
            return
        before = self.doc.snapshot()
        inst.cx += dx
        inst.cy += dy
        self._push_transform(before)

    # -------------------------------------------------------------- layer ops
    def _layer_op(self, op: str, uid: int) -> None:
        """Push one undo entry that reorders ``doc.instances`` for ``op``."""
        doc = self.doc
        i = doc.index_of_uid(uid)
        if i < 0:
            return
        before = doc.snapshot()
        new_index = {
            "bring_front": len(doc.instances) - 1,
            "send_back": 0,
            "forward": min(len(doc.instances) - 1, i + 1),
            "backward": max(0, i - 1),
        }.get(op)
        if new_index is None:
            return
        if new_index == i:
            labels = {
                "bring_front": "已在最上层",
                "send_back": "已在最下层",
                "forward": "无法再上移",
                "backward": "无法再下移",
            }
            sb = self.mw.statusBar() if hasattr(self.mw, "statusBar") else None
            if sb is not None:
                sb.showMessage(labels.get(op, "层级未变化"), 1800)
            return
        after = list(before)
        inst = after.pop(i)
        after.insert(new_index, inst)
        if _instance_snapshots_equal(before, after):
            return
        self.undo_stack.push(ReorderCommand(doc, before, after, text=f"layer {op}"))
        # Keep selection on the moved instance and confirm in the status bar.
        doc.select(uid)
        sb = self.mw.statusBar() if hasattr(self.mw, "statusBar") else None
        if sb is not None:
            sb.showMessage(
                {"bring_front": "已置顶", "send_back": "已置底",
                 "forward": "已上移一层", "backward": "已下移一层"}.get(op, "层级已更新"),
                1800,
            )

    def bring_to_front(self, uid: Optional[int] = None) -> None:
        uid = int(uid if uid is not None else (self.doc.selected_uid or 0))
        if uid:
            self._layer_op("bring_front", uid)

    def send_to_back(self, uid: Optional[int] = None) -> None:
        uid = int(uid if uid is not None else (self.doc.selected_uid or 0))
        if uid:
            self._layer_op("send_back", uid)

    def move_layer_up(self, uid: Optional[int] = None) -> None:
        uid = int(uid if uid is not None else (self.doc.selected_uid or 0))
        if uid:
            self._layer_op("forward", uid)

    def move_layer_down(self, uid: Optional[int] = None) -> None:
        uid = int(uid if uid is not None else (self.doc.selected_uid or 0))
        if uid:
            self._layer_op("backward", uid)

    def reorder_to_uids(self, uids: List[int]) -> None:
        """Apply a drag-drop reorder from the InstanceListWidget."""
        doc = self.doc
        if not uids:
            return
        before = doc.snapshot()
        by_uid = {inst.uid: inst for inst in before}
        after = [by_uid[u] for u in uids if u in by_uid]
        # Keep any instances not in `uids` (defensive) in their original order.
        seen = set(uids)
        for inst in before:
            if inst.uid not in seen:
                after.append(inst)
        if _instance_snapshots_equal(before, after):
            return
        self.undo_stack.push(ReorderCommand(doc, before, after, text="reorder"))

    # --------------------------------------------------- visibility / lock
    def _toggle_property(self, uid: int, field: str, value: bool, label: str) -> None:
        doc = self.doc
        before = doc.snapshot()
        after = [i.clone() for i in before]
        changed = False
        for inst in after:
            if inst.uid == uid and getattr(inst, field) != value:
                setattr(inst, field, value)
                inst.invalidate_cache()
                changed = True
        if not changed or _instance_snapshots_equal(before, after):
            return
        self.undo_stack.push(PropertyToggleCommand(doc, before, after, text=label))

    def set_visibility(self, uid: int, visible: bool) -> None:
        self._toggle_property(uid, "visible", bool(visible), "toggle visibility")

    def set_locked(self, uid: int, locked: bool) -> None:
        self._toggle_property(uid, "locked", bool(locked), "toggle lock")
