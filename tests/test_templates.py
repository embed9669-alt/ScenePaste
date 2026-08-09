"""Tests for scene-template save/load (Qt + tkinter share the same schema)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app.models import Cutout, Instance  # noqa: E402
from compose_app_qt.state import Document  # noqa: E402
from compose_app_qt import templates  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_cutout(label: str, class_id: int, source: str) -> Cutout:
    return Cutout(
        label=label, class_id=class_id, source=source,
        rgba=Image.new("RGBA", (40, 60), (255, 0, 0, 255)),
        polygon=None,
        thumb=Image.new("RGB", (96, 96), (40, 40, 40)),
    )


def _doc_with_state() -> Document:
    doc = Document()
    doc.set_cutouts([
        _fake_cutout("person", 0, "a.json#0"),
        _fake_cutout("truck", 1, "b.json#1"),
    ])
    doc.set_bg_size(800, 600)
    doc.add_instance(Instance(cutout_index=0, cx=200.0, cy=400.0,
                              h_ratio=0.3, flip=False, angle=15.0, uid=1))
    doc.add_instance(Instance(cutout_index=1, cx=600.0, cy=500.0,
                              h_ratio=0.4, flip=True, angle=-10.0, uid=2))
    return doc


def test_save_template_writes_portable_json(tmp_path: Path):
    doc = _doc_with_state()
    out = tmp_path / "scene.json"
    count = templates.save_template(doc, out)
    assert count == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "scenepaste/scene-template"
    assert payload["version"] == 2
    assert payload["canvas"] == {"width": 800, "height": 600}
    assert len(payload["instances"]) == 2
    # Positions stored as ratios.
    inst0 = payload["instances"][0]
    assert abs(inst0["cx_ratio"] - 200 / 800) < 1e-6
    assert abs(inst0["cy_ratio"] - 400 / 600) < 1e-6
    assert inst0["flip"] is False
    assert inst0["angle"] == 15.0


def test_save_template_requires_background(tmp_path: Path):
    doc = Document()
    doc.set_cutouts([_fake_cutout("x", 0, "x#0")])
    with pytest.raises(RuntimeError, match="背景"):
        templates.save_template(doc, tmp_path / "x.json")


def test_save_template_requires_instances(tmp_path: Path):
    doc = Document()
    doc.set_cutouts([_fake_cutout("x", 0, "x#0")])
    doc.set_bg_size(800, 600)
    with pytest.raises(RuntimeError, match="目标"):
        templates.save_template(doc, tmp_path / "x.json")


def test_load_template_restores_exact_layout(tmp_path: Path):
    doc = _doc_with_state()
    out = tmp_path / "scene.json"
    templates.save_template(doc, out)

    # Fresh doc with same cutouts, different bg size — layout should scale.
    doc2 = Document()
    doc2.set_cutouts([
        _fake_cutout("person", 0, "a.json#0"),
        _fake_cutout("truck", 1, "b.json#1"),
    ])
    doc2.set_bg_size(400, 300)  # half the original
    restored, missing = templates.load_template(doc2, out)
    assert not missing
    assert len(restored) == 2
    # Coordinates should scale with the new bg size.
    assert abs(restored[0].cx - 100.0) < 1e-3  # 200 * (400/800)
    assert abs(restored[0].cy - 200.0) < 1e-3  # 400 * (300/600)
    assert restored[1].flip is True


def test_load_template_handles_missing_cutout(tmp_path: Path):
    """If the template references a source not in the library, the entry
    is reported as missing but the rest still load."""
    doc = _doc_with_state()
    out = tmp_path / "scene.json"
    templates.save_template(doc, out)

    doc2 = Document()
    doc2.set_cutouts([_fake_cutout("person", 0, "a.json#0")])  # only one cutout
    doc2.set_bg_size(800, 600)
    restored, missing = templates.load_template(doc2, out)
    assert len(restored) == 1  # only person matched
    assert len(missing) == 1   # truck (b.json#1) missing


def test_load_template_falls_back_to_label_match(tmp_path: Path):
    """When source differs but label is unique, fall back to label."""
    doc = _doc_with_state()
    out = tmp_path / "scene.json"
    templates.save_template(doc, out)

    doc2 = Document()
    # Same labels, different sources.
    doc2.set_cutouts([
        _fake_cutout("person", 0, "DIFFERENT_person.json#0"),
        _fake_cutout("truck", 1, "DIFFERENT_truck.json#0"),
    ])
    doc2.set_bg_size(800, 600)
    restored, missing = templates.load_template(doc2, out)
    assert not missing
    assert len(restored) == 2


def test_load_template_raises_when_no_match(tmp_path: Path):
    doc = _doc_with_state()
    out = tmp_path / "scene.json"
    templates.save_template(doc, out)

    doc2 = Document()
    doc2.set_cutouts([_fake_cutout("bicycle", 2, "c.json#0")])
    doc2.set_bg_size(800, 600)
    with pytest.raises(RuntimeError, match="无法"):
        templates.load_template(doc2, out)
