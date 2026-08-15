"""LoadCutoutsWorker must auto-extend the class map (no silent drops)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from compose_app_qt.workers import LoadCutoutsWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_worker_auto_extends_class_map_for_samples(qapp):
    samples = Path(__file__).resolve().parents[1] / "samples" / "objects"
    if not samples.is_dir():
        pytest.skip("samples/objects missing")

    worker = LoadCutoutsWorker(samples, "person=0,vehicle=1")
    result = {}

    def _on_ready(cutouts, class_map_text):
        result["cutouts"] = cutouts
        result["class_map_text"] = class_map_text

    def _on_fail(msg):
        result["error"] = msg

    worker.cutouts_ready.connect(_on_ready)
    worker.failed.connect(_on_fail)
    worker.run()  # synchronous in tests

    assert "error" not in result, result.get("error")
    labels = {c.label for c in result["cutouts"]}
    assert "person" in labels
    assert "motorcycle" in labels
    assert "truck" in labels
    assert "motorcycle=" in result["class_map_text"]
    assert "truck=" in result["class_map_text"]
