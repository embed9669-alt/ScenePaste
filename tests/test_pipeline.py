"""End-to-end pipeline test: run generate_dataset on the bundled samples.

These tests use the real ``samples/`` data shipped with the repo. They lock
in the contract that one ``generate_dataset`` call produces a correctly
shaped output tree across every supported output format.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scenepaste import GenerationConfig, generate_dataset, parse_class_map

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OBJECTS = REPO_ROOT / "samples" / "objects"
SAMPLE_BACKGROUNDS = REPO_ROOT / "samples" / "backgrounds"


def _config(tmp_path: Path, fmt: str, count: int = 3, **overrides) -> GenerationConfig:
    return GenerationConfig(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map=parse_class_map("person=0"),
        count=count,
        seed=42,
        preview_ratio=1.0,
        output_format=fmt,
        **overrides,
    )


def _run(fmt: str, tmp_path: Path, **overrides) -> dict:
    summary = generate_dataset(_config(tmp_path, fmt, **overrides))
    assert summary["generated_images"] >= 1, "pipeline produced zero images"
    return summary


@pytest.mark.parametrize("fmt", ["detect", "seg", "coco", "semantic", "obb"])
def test_generate_each_format_produces_expected_files(fmt: str, tmp_path: Path):
    summary = _run(fmt, tmp_path)
    out = tmp_path

    # Common outputs for every format
    assert (out / "images" / "train").is_dir()
    assert (out / "labels" / "train").is_dir()
    assert (out / "previews").is_dir()
    assert (out / "cutouts").is_dir()
    assert (out / "classes.txt").is_file()
    assert (out / "data.yaml").is_file()
    # Every run emits both run-prefixed and latest_* summary/config.
    assert (out / "latest_summary.json").is_file()
    assert (out / "run_config.json").is_file()
    summary_files = list(out.glob("run_*_summary.json"))
    assert len(summary_files) == 1

    image_count = len(list((out / "images" / "train").glob("*.jpg")))
    assert image_count == summary["generated_images"]

    if fmt == "coco":
        coco = json.loads((out / "instances_coco.json").read_text(encoding="utf-8"))
        assert coco["images"], "coco writer produced no images"
        assert coco["annotations"], "coco writer produced no annotations"
        assert {c["name"] for c in coco["categories"]} == {"person"}
    if fmt == "semantic":
        masks = list((out / "masks" / "train").glob("*.png"))
        assert masks, "no semantic masks written"
        mapping = json.loads((out / "semantic_classes.json").read_text(encoding="utf-8"))
        assert mapping["0"] == "background"
        assert mapping["1"] == "person"
    if fmt == "seg":
        # seg format writes single-polygon YOLO-seg lines to labels/train.
        seg_label = next((out / "labels" / "train").glob("*.txt"))
        first_line = seg_label.read_text(encoding="utf-8").splitlines()[0].split()
        assert len(first_line) >= 7, "seg line should have class + >=3 (x,y) pairs"
        assert len(first_line) % 2 == 1, "seg line must have odd token count"
    if fmt == "obb":
        obb_label = next((out / "labels" / "train").glob("*.txt"))
        first_line = obb_label.read_text(encoding="utf-8").splitlines()[0].split()
        # class + 4 corners * 2 coords = 9 tokens
        assert len(first_line) == 9


def test_generation_log_csv_rows_match_summary(tmp_path: Path):
    summary = _run("detect", tmp_path, count=4)
    log_files = list(tmp_path.glob("run_*_log.csv"))
    assert len(log_files) == 1
    with log_files[0].open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # +1 for the header row.
    assert len(rows) == summary["generated_objects"] + 1
    header = rows[0]
    assert header == [
        "generated_stem", "background", "label", "class_id",
        "source_json", "shape_index", "flipped", "box_xyxy",
        "used_paste_zone", "run_id",
    ]


def test_seed_reproducibility(tmp_path: Path):
    """Same seed + same run_id -> identical label bytes."""
    a = _run("detect", tmp_path / "a", count=3, run_id="fixed")
    b = _run("detect", tmp_path / "b", count=3, run_id="fixed")
    a_labels = sorted(p.read_bytes() for p in (tmp_path / "a" / "labels" / "train").glob("*.txt"))
    b_labels = sorted(p.read_bytes() for p in (tmp_path / "b" / "labels" / "train").glob("*.txt"))
    assert a_labels == b_labels
    # output_dir differs (different tmp paths); everything else must match.
    a_cmp = {k: v for k, v in a.items() if k != "output_dir"}
    b_cmp = {k: v for k, v in b.items() if k != "output_dir"}
    assert a_cmp == b_cmp


def test_unknown_class_in_objects_is_skipped(tmp_path: Path):
    """The bundled sample has motorcycle/truck that are not in class_map."""
    # person is the only recognized label; motorcycle and truck must be skipped.
    _run("detect", tmp_path, count=2)
    log_files = list(tmp_path.glob("run_*_log.csv"))
    with log_files[0].open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    labels = {r[2] for r in rows}
    assert labels <= {"person"}, f"unexpected labels leaked: {labels}"


def test_large_run_status_telemetry_is_written(tmp_path: Path):
    summary = _run("detect", tmp_path, count=2, run_id="telemetry")
    status_path = tmp_path / summary["status_file"]
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["completed"] == 2
    assert status["requested"] == 2
    assert status["failed"] == 0
    assert status["images_per_second"] > 0
    assert status["disk_free_bytes"] > 0
    assert status["eta_seconds"] == 0
