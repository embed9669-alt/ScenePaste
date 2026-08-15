"""QUndoCommand subclasses for instance operations.

Each command captures the previous and next document state (full snapshots).
Coarse-grained snapshots keep the code simple; per-instance diffing would be
faster but the dataset is small (typically <30 instances per scene).
"""

from __future__ import annotations

from typing import List

from PySide6.QtGui import QUndoCommand

from compose_app.models import Instance
from .state import Document


class _SnapshotCommand(QUndoCommand):
    """Base class: captures before/after instance snapshots."""

    def __init__(self, doc: Document, before: List[Instance], after: List[Instance],
                 text: str = ""):
        super().__init__(text)
        self._doc = doc
        self._before = before
        self._after = after

    def undo(self) -> None:
        self._doc.replace_instances([i.clone() for i in self._before])

    def redo(self) -> None:
        self._doc.replace_instances([i.clone() for i in self._after])


class AddInstanceCommand(_SnapshotCommand):
    """Add an instance and select it. Undo restores the prior selection."""

    def __init__(self, doc: Document, inst: Instance):
        before = doc.snapshot()
        after = list(before) + [inst]
        super().__init__(doc, before, after, "add instance")
        self._select_uid = inst.uid
        self._prev_select = doc.selected_uid

    def redo(self) -> None:
        super().redo()
        self._doc.select(self._select_uid)

    def undo(self) -> None:
        super().undo()
        self._doc.select(self._prev_select)


class DeleteSelectedCommand(_SnapshotCommand):
    def __init__(self, doc: Document):
        before = doc.snapshot()
        sel = doc.selected_uid
        after = [i.clone() for i in before if i.uid != sel]
        super().__init__(doc, before, after, "delete instance")


class TransformCommand(_SnapshotCommand):
    """Coalesces a sequence of transform nudges into one undo step."""

    def __init__(self, doc: Document, before: List[Instance], text: str = "transform"):
        # ``after`` is read live from the document on each redo; we capture
        # the *initial* after here for the first redo() call.
        after = doc.snapshot()
        super().__init__(doc, before, after, text)


class ReorderCommand(_SnapshotCommand):
    """Reorder instances (changes z-order). Caller pre-computes the after list."""

    def __init__(self, doc: Document, before: List[Instance], after: List[Instance],
                 text: str = "reorder"):
        super().__init__(doc, before, after, text)


class PropertyToggleCommand(_SnapshotCommand):
    """Toggles a per-instance boolean (visible / locked) for one or more uids.

    Used when a visibility / lock change must be undoable as a single step.
    """

    def __init__(self, doc: Document, before: List[Instance], after: List[Instance],
                 text: str = "toggle"):
        super().__init__(doc, before, after, text)
