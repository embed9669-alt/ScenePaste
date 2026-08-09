"""Canvas / panel transforms push undoable TransformCommand entries."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from compose_app.models import Cutout, Instance
from compose_app_qt.app import MainWindow, _instance_snapshots_equal


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _seed_selection(win: MainWindow) -> Instance:
    rgba = Image.fromarray(np.zeros((40, 30, 4), dtype=np.uint8))
    rgba.putalpha(255)
    win.doc.set_cutouts([Cutout(label="t", class_id=0, source="x", rgba=rgba)])
    win.doc.bg_size = (200, 200)
    inst = Instance(cutout_index=0, cx=100, cy=100, h_ratio=0.2, angle=10.0, uid=win.doc.next_uid())
    win.doc.replace_instances([inst])
    win.doc.select(inst.uid)
    return inst


def test_scale_spin_undo_redo(qapp):
    win = MainWindow(theme_mode="dark")
    inst = _seed_selection(win)
    before_ratio = inst.h_ratio
    win._on_scale_changed(0.45)
    assert abs(win.doc.selected().h_ratio - 0.45) < 1e-9
    assert win._undo_stack.canUndo()
    win.undo()
    assert abs(win.doc.selected().h_ratio - before_ratio) < 1e-9
    win.redo()
    assert abs(win.doc.selected().h_ratio - 0.45) < 1e-9
    win.close()


def test_canvas_transform_committed_skips_noop(qapp):
    win = MainWindow(theme_mode="dark")
    _seed_selection(win)
    snap = win.doc.snapshot()
    win._on_canvas_transform_committed(snap)
    assert not win._undo_stack.canUndo()
    win.close()


def test_flip_undo(qapp):
    win = MainWindow(theme_mode="dark")
    inst = _seed_selection(win)
    assert inst.flip is False
    win._on_flip()
    assert win.doc.selected().flip is True
    win.undo()
    assert win.doc.selected().flip is False
    win.close()


def test_instance_snapshots_equal_helper():
    a = [Instance(0, 1.0, 2.0, 0.2, angle=10.0, uid=1)]
    b = [Instance(0, 1.0, 2.0, 0.2, angle=370.0, uid=1)]  # 370 ≡ 10
    assert _instance_snapshots_equal(a, b)
    b[0].h_ratio = 0.3
    assert not _instance_snapshots_equal(a, b)
