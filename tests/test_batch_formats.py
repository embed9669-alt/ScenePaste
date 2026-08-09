"""Multi-format coverage for BatchApplyWorker.

Each test runs the worker with a different output_format and asserts the
expected files appear in the right directories. Together these exercise
the format-specific branches in BatchApplyWorker._write_labels.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app.models import Cutout  # noqa: E402
from compose_app_qt.workers import BatchApplyWorker, _InstanceSnap  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OBJECTS = REPO_ROOT / "samples" / "objects"
SAMPLE_BACKGROUNDS = REPO_ROOT / "samples" / "backgrounds"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _run_worker(qapp, tmp_path: Path, output_format: str) -> dict:
    """Helper: build a minimal doc, run the worker, return its summary."""
    # Load real cutouts for realistic rendering.
    import scenepaste.core as core
    assets = core.load_object_assets(SAMPLE_OBJECTS,
                                     core.parse_class_map("person=0,motorcycle=1,truck=2"),
                                     feather_sigma=0.8, log=lambda _m: None)
    from compose_app.rendering import pil_from_asset, make_thumbnail
    cutouts = [
        Cutout(label=a.label, class_id=a.class_id,
               source=f"{a.source_json}#{a.source_shape_index}",
               rgba=pil_from_asset(a), polygon=a.polygon,
               thumb=make_thumbnail(pil_from_asset(a)))
        for a in assets
    ]
    snaps = [_InstanceSnap(cutout_index=0, cx_ratio=0.5, cy_ratio=0.6,
                           h_ratio=0.3, flip=False, angle=0.0)]

    bg_paths = sorted(SAMPLE_BACKGROUNDS.glob("*.jpg"))[:2]
    out_dir = tmp_path / "out"

    results = {"summary": None, "error": None}
    worker = BatchApplyWorker(
        cutouts=cutouts, instance_snaps=snaps, background_paths=bg_paths,
        output_dir=out_dir, class_map_text="person=0,motorcycle=1,truck=2",
        output_format=output_format, run_id=f"fmt_{output_format}",
    )
    worker.finished.connect(lambda s: results.__setitem__("summary", s))
    worker.failed.connect(lambda e: results.__setitem__("error", e))
    worker.start()
    deadline = time.time() + 30
    while time.time() < deadline and results["summary"] is None and results["error"] is None:
        qapp.processEvents()
        time.sleep(0.03)
    for _ in range(10):
        qapp.processEvents()
        time.sleep(0.02)
    assert results["error"] is None, f"worker failed: {results['error']}"
    return results["summary"]


@pytest.mark.parametrize("fmt", ["detect", "seg", "coco", "semantic", "obb"])
def test_batch_worker_emits_expected_files_for_each_format(qapp, tmp_path: Path, fmt: str):
    summary = _run_worker(qapp, tmp_path, fmt)
    assert summary["generated"] == 2
    out = tmp_path / "out"

    # Every format emits images + data.yaml.
    assert list((out / "images" / "train").glob("*.jpg"))
    assert (out / "data.yaml").is_file()

    if fmt == "detect":
        labels = list((out / "labels" / "train").glob("*.txt"))
        assert labels
        for label in labels:
            tokens = label.read_text(encoding="utf-8").split()
            assert len(tokens) == 5  # class cx cy w h

    if fmt == "seg":
        labels = list((out / "labels" / "train").glob("*.txt"))
        assert labels
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                assert len(tokens) >= 7
                assert len(tokens) % 2 == 1

    if fmt == "coco":
        coco = json.loads((out / "instances_coco.json").read_text(encoding="utf-8"))
        assert coco["images"]
        assert coco["annotations"]
        for ann in coco["annotations"]:
            assert ann["segmentation"], "coco annotation missing segmentation"
            assert ann["area"] > 0

    if fmt == "semantic":
        masks = list((out / "masks" / "train").glob("*.png"))
        assert masks
        mapping = json.loads((out / "semantic_classes.json").read_text(encoding="utf-8"))
        assert mapping["0"] == "background"

    if fmt == "obb":
        labels = list((out / "labels" / "train").glob("*.txt"))
        assert labels
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                assert len(tokens) == 9  # class + 4 (x,y)


def test_batch_worker_propagates_top_level_error(qapp, tmp_path: Path):
    """Bad class_map surfaces via the failed signal."""
    from compose_app.models import Cutout as _C
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    cutout = _C(label="x", class_id=0, source="x#0", rgba=rgba)
    snaps = [_InstanceSnap(0, 0.5, 0.5, 0.2, False, 0.0)]

    results = {"error": None}
    worker = BatchApplyWorker(
        cutouts=[cutout], instance_snaps=snaps,
        background_paths=[tmp_path / "nope.jpg"],  # missing → fails to read
        output_dir=tmp_path / "out", class_map_text="bad_format_no_equals",
        output_format="detect", run_id="err",
    )
    worker.failed.connect(lambda e: results.__setitem__("error", e))
    worker.start()
    deadline = time.time() + 10
    while time.time() < deadline and results["error"] is None and worker.isRunning():
        qapp.processEvents()
        time.sleep(0.03)
    for _ in range(5):
        qapp.processEvents()
        time.sleep(0.02)
    assert results["error"] is not None
