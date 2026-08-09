"""Headless test for the Qt batch-apply (apply_to_next) flow."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OBJECTS = REPO_ROOT / "samples" / "objects"
SAMPLE_BACKGROUNDS = REPO_ROOT / "samples" / "backgrounds"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _drain(qapp, iterations: int = 60, delay: float = 0.05) -> None:
    for _ in range(iterations):
        qapp.processEvents()
        time.sleep(delay)


def test_batch_apply_runs_to_completion(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow

    win = MainWindow(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    assert win.doc.background_paths, "backgrounds failed to load"

    # Place one instance on the current background so snapshot has content.
    win._place_at_centre(0)
    assert len(win.doc.instances) == 1

    # Drive apply_to_next through the worker directly (skipping the modal
    # dialog, which can't easily be driven headlessly).
    from compose_app_qt.workers import BatchApplyWorker, _InstanceSnap
    bg_w, bg_h = win.doc.bg_size
    snaps = [_InstanceSnap.from_instance(i, bg_w, bg_h) for i in win.doc.instances]
    targets = win.doc.background_paths[win.doc.background_index + 1:]

    results = {"summary": None, "error": None}
    worker = BatchApplyWorker(
        cutouts=win.doc.cutouts,
        instance_snaps=snaps,
        background_paths=targets,
        output_dir=tmp_path,
        class_map_text=win.doc.class_map_text,
        output_format="detect",
        run_id="test_batch",
    )
    worker.finished.connect(lambda s: results.__setitem__("summary", s))
    worker.failed.connect(lambda e: results.__setitem__("error", e))
    worker.start()
    # Wait for completion (or timeout).
    deadline = time.time() + 30
    while time.time() < deadline and results["summary"] is None and results["error"] is None:
        qapp.processEvents()
        time.sleep(0.03)
    # Drain any queued finished/failed signal that arrived after isRunning() went False.
    for _ in range(10):
        qapp.processEvents()
        time.sleep(0.02)

    assert results["error"] is None, f"worker failed: {results['error']}"
    summary = results["summary"]
    assert summary is not None
    assert summary["generated"] == len(targets)
    assert summary["failed"] == 0

    # Verify file outputs.
    jpgs = sorted((tmp_path / "images" / "train").glob("test_batch_*.jpg"))
    labels = sorted((tmp_path / "labels" / "train").glob("test_batch_*.txt"))
    assert len(jpgs) == len(targets)
    assert len(labels) == len(targets)
    # Each label should be a valid YOLO line.
    for label in labels:
        tokens = label.read_text(encoding="utf-8").split()
        assert len(tokens) == 5
        assert 0 <= int(tokens[0]) <= 2
    # data.yaml should exist.
    assert (tmp_path / "data.yaml").is_file()


def test_batch_worker_cancellation(qapp, tmp_path: Path):
    """Calling cancel() mid-run stops the worker after the current image."""
    from compose_app_qt.workers import BatchApplyWorker, _InstanceSnap
    from compose_app.models import Cutout
    from PIL import Image

    # Build a fake doc state directly (skip the heavy MainWindow).
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    cutout = Cutout(label="x", class_id=0, source="x#0", rgba=rgba)
    snaps = [_InstanceSnap(cutout_index=0, cx_ratio=0.5, cy_ratio=0.5,
                           h_ratio=0.2, flip=False, angle=0.0)]

    # Generate fake backgrounds (enough that the worker can't finish
    # before cancel() is called).
    import numpy as np
    bg_paths = []
    for i in range(50):
        p = tmp_path / f"bg_{i}.jpg"
        import cv2
        cv2.imwrite(str(p), np.zeros((100, 100, 3), dtype=np.uint8))
        bg_paths.append(p)

    out_dir = tmp_path / "out"
    worker = BatchApplyWorker(
        cutouts=[cutout], instance_snaps=snaps, background_paths=bg_paths,
        output_dir=out_dir, class_map_text="x=0", output_format="detect",
        run_id="cancel_test",
    )
    summary_box = {"s": None}
    worker.finished.connect(lambda s: summary_box.__setitem__("s", s))
    # Cancel before starting — worker should bail on its first iteration check.
    worker.cancel()
    worker.start()
    deadline = time.time() + 15
    while time.time() < deadline and summary_box["s"] is None:
        qapp.processEvents()
        time.sleep(0.03)
    for _ in range(10):
        qapp.processEvents()
        time.sleep(0.02)
    summary = summary_box["s"]
    assert summary is not None
    assert summary["cancelled"] is True
    # With cancel set before start, generated should be 0 (no iterations completed).
    assert summary["generated"] == 0
