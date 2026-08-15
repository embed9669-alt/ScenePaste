"""Qt InstanceItem must bake flip/angle once (no double transform)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication

from compose_app.models import Cutout, Instance
from compose_app.rendering import render_instance
from compose_app_qt.canvas import InstanceItem
from compose_app_qt.state import Document


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_instance_item_pose_is_identity_after_bake(qapp):
    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    cut = Cutout("person", 0, "a.json#0", rgba)
    doc = Document(cutouts=[cut], bg_size=(200, 200))
    inst = Instance(0, 100, 100, 0.4, flip=True, angle=30.0, uid=1)
    doc.instances = [inst]

    # Minimal view stub — InstanceItem only needs _syncing + signals unused here.
    class _View:
        _syncing = False
        instance_moved = type("S", (), {"emit": staticmethod(lambda *_: None)})()

    item = InstanceItem(inst, cut, doc, _View())  # type: ignore[arg-type]
    assert item.rotation() == 0.0
    assert item.transform().isIdentity()
    baked = render_instance(rgba, max(4, int(round(0.4 * 200))), True, 30.0)
    assert item.pixmap().width() == baked.width
    assert item.pixmap().height() == baked.height
