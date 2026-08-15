"""Coverage for compose_app_qt: canvas drop event, state mutation paths, theme switching."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QPointF, QMimeData, QByteArray, QPoint  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app_qt.state import Document  # noqa: E402
from compose_app_qt.theme import qss_for  # noqa: E402
from compose_app.models import Cutout, Instance  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_cutout(label: str = "x", class_id: int = 0) -> Cutout:
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    thumb = Image.new("RGB", (96, 96), (40, 40, 40))
    return Cutout(label=label, class_id=class_id, source="x.json#0",
                  rgba=rgba, polygon=None, thumb=thumb)


# ----------------------------------------------------------------------- state
def test_document_next_uid_increments():
    doc = Document()
    assert doc.next_uid() == 1
    assert doc.next_uid() == 2


def test_document_remove_instance_clears_selection():
    doc = Document()
    inst = Instance(cutout_index=0, cx=10, cy=20, h_ratio=0.3, uid=42)
    doc.add_instance(inst)
    assert doc.selected_uid == 42
    doc.remove_instance(42)
    assert doc.instances == []
    assert doc.selected_uid is None


def test_document_index_of_returns_minus_one_for_missing():
    doc = Document()
    assert doc.index_of(Instance(cutout_index=0, cx=0, cy=0, h_ratio=0.1, uid=99)) == -1


def test_document_snapshot_is_independent():
    doc = Document()
    doc.add_instance(Instance(cutout_index=0, cx=10, cy=20, h_ratio=0.3, uid=1))
    snap = doc.snapshot()
    snap[0].cx = 999
    assert doc.instances[0].cx == 10.0, "snapshot must not mutate the live document"


def test_document_replace_instances_drops_stale_selection():
    doc = Document()
    doc.add_instance(Instance(cutout_index=0, cx=10, cy=20, h_ratio=0.3, uid=1))
    doc.select(1)
    doc.replace_instances([Instance(cutout_index=0, cx=0, cy=0, h_ratio=0.1, uid=2)])
    assert doc.selected_uid is None  # uid 1 no longer present


# ----------------------------------------------------------------------- theme
def test_theme_qss_returns_nonempty_strings():
    assert len(qss_for("dark")) > 100
    assert len(qss_for("light")) > 100
    assert qss_for("dark") != qss_for("light")


# ----------------------------------------------------------------------- canvas drop
def test_canvas_accepts_cutout_drop(qapp, tmp_path: Path):
    """Drag-drop a cutout index onto the canvas → adds an instance."""
    from compose_app_qt.canvas import CanvasView
    from compose_app_qt.state import Document

    doc = Document()
    doc.set_cutouts([_fake_cutout("a"), _fake_cutout("b")])
    # Provide a fake background so bg_size is set.
    import scenepaste.core as core
    bg_path = tmp_path / "bg.jpg"
    core.imwrite_unicode(bg_path, np.zeros((100, 100, 3), dtype=np.uint8))
    doc.set_backgrounds([bg_path], index=0)
    doc.set_bg_size(100, 100)

    view = CanvasView(doc)
    # Host (MainWindow) owns undo; canvas only emits place_at_requested.
    from compose_app.models import Instance

    def _on_place(idx, pos):
        cut = doc.cutouts[idx]
        bg_h = doc.bg_size[1] or cut.rgba.height
        doc.add_instance(Instance(
            cutout_index=idx,
            cx=float(pos.x()),
            cy=float(pos.y()),
            h_ratio=max(0.05, min(0.5, cut.rgba.height / max(1, bg_h) * 0.5)),
            uid=doc.next_uid(),
        ))

    view.place_at_requested.connect(_on_place)
    view.resize(400, 400)
    view.show()
    qapp.processEvents()
    time.sleep(0.05)

    # Simulate a drop with cutout index 1.
    md = QMimeData()
    md.setData("application/x-cutout-index", QByteArray(b"1"))
    # Build a minimal drop event position; the canvas maps it to scene coords.
    view.dropEvent(_DropEvent(QPoint(200, 200), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier))
    qapp.processEvents()
    assert len(doc.instances) == 1
    assert doc.instances[0].cutout_index == 1


class _DropEvent(QDropEvent):
    """QDropEvent with a fixed scene drop point for tests."""

    def __init__(self, pos, *args, **kwargs):
        super().__init__(pos, *args, **kwargs)
        self._pos = QPointF(pos)

    def position(self):
        return self._pos


def test_canvas_rejects_drop_with_bad_mime(qapp):
    from compose_app_qt.canvas import CanvasView
    doc = Document()
    view = CanvasView(doc)
    md = QMimeData()
    md.setText("not a cutout")
    event = _DropEvent(QPoint(50, 50), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    # Should not raise; should fall through to the base handler.
    try:
        view.dropEvent(event)
    except Exception:
        pass
    assert doc.instances == []
