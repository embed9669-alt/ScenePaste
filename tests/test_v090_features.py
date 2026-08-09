from __future__ import annotations

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from scenepaste.core.similarity import diversity_summary, select_diverse, visual_embedding
from scenepaste.tools.compare import compare_datasets
from scenepaste.tools.diversity import analyze_diversity, export_selected_dataset
from scenepaste.tools.hardmine import mine_hard_examples, write_hardmine_outputs
from scenepaste.tools.leakage import detect_split_leakage
from scenepaste.tools.qa import build_qa_report
from scenepaste.tools.shard import build_webdataset_shards


def _write_image(path: Path, value: int = 100, marker: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((96, 128, 3), value, dtype=np.uint8)
    if marker:
        cv2.circle(arr, (64, 48), 22, (240, 30, 30), -1)
    assert cv2.imwrite(str(path), arr)


def _make_yolo(root: Path, split: str, stem: str, label: str, value: int = 100, marker: bool = False):
    _write_image(root / "images" / split / f"{stem}.jpg", value, marker)
    lp = root / "labels" / split / f"{stem}.txt"; lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(label, encoding="utf-8")
    (root / "classes.txt").write_text("0: person\n", encoding="utf-8")


def test_cv_lite_embedding_and_diverse_selection(tmp_path):
    paths = []
    for i, (value, marker) in enumerate([(80, False), (80, False), (180, True)]):
        p = tmp_path / f"{i}.jpg"; _write_image(p, value, marker); paths.append(p)
    a = cv2.imread(str(paths[0])); b = cv2.imread(str(paths[0]))
    assert np.allclose(visual_embedding(a), visual_embedding(b))
    summary = diversity_summary(paths, limit=10)
    assert summary["samples"] == 3
    chosen = select_diverse(paths, 2, limit=10)
    assert len(chosen) == 2


def test_cross_split_leakage_detects_exact_duplicate(tmp_path):
    root = tmp_path / "ds"
    _make_yolo(root, "train", "a", "0 0.5 0.5 0.2 0.2\n", 120, True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    data = (root / "images" / "train" / "a.jpg").read_bytes()
    (root / "images" / "val" / "b.jpg").write_bytes(data)
    (root / "labels" / "val" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = detect_split_leakage(root, embedding_limit_per_split=10)
    assert report["exact_cross_split"] >= 1
    assert report["health"] == "warning"


def test_hardmine_turns_model_failures_into_profile(tmp_path):
    root = tmp_path / "ds"
    _make_yolo(root, "val", "missed", "0 0.5 0.5 0.25 0.25\n", 100, True)
    _make_yolo(root, "val", "negative", "", 150, False)
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "missed.txt").write_text("", encoding="utf-8")
    (pred / "negative.txt").write_text("0 0.5 0.5 0.2 0.2 0.91\n", encoding="utf-8")
    report = mine_hard_examples(root, pred, split="val", top=10)
    assert report["total_false_negatives"] == 1
    assert report["total_false_positives"] == 1
    result = write_hardmine_outputs(report, tmp_path / "hard")
    assert Path(result["json_path"]).is_file()
    assert Path(result["html_path"]).is_file()
    assert "Hard Example Mining" in Path(result["html_path"]).read_text(encoding="utf-8")
    assert result["profile_path"] and Path(result["profile_path"]).is_file()
    profile = json.loads(Path(result["profile_path"]).read_text(encoding="utf-8"))
    assert profile["source_type"] == "yolo-hard-subset"


def test_real_synthetic_comparison_and_qa_curation(tmp_path):
    real = tmp_path / "real"; synth = tmp_path / "synth"
    _make_yolo(real, "train", "r1", "0 0.5 0.6 0.2 0.3\n", 100, True)
    _make_yolo(real, "train", "r2", "0 0.6 0.7 0.25 0.35\n", 120, True)
    _make_yolo(synth, "train", "s1", "0 0.5 0.6 0.2 0.3\n", 105, True)
    _make_yolo(synth, "train", "s2", "0 0.6 0.7 0.25 0.35\n", 125, True)
    report = compare_datasets(real, synth, embedding_limit=10)
    assert report["distribution"]["matched_classes"] == 1
    assert report["visual"]["real_samples"] == 2
    qa = build_qa_report(synth, duplicate_limit=10, embedding_limit=10)
    assert "curation" in qa
    assert qa["curation"]["diversity"]["samples"] == 2


def test_webdataset_sharding_preserves_modalities(tmp_path):
    root = tmp_path / "ds"
    _make_yolo(root, "train", "a", "0 0.5 0.5 0.2 0.2\n", 90, True)
    _make_yolo(root, "train", "b", "0 0.4 0.5 0.2 0.2\n", 130, False)
    seg = root / "labels-seg" / "train"; seg.mkdir(parents=True)
    (seg / "a.txt").write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n", encoding="utf-8")
    out = tmp_path / "shards"
    manifest = build_webdataset_shards(root, out, split="train", max_samples=1)
    assert manifest["samples"] == 2 and len(manifest["shards"]) == 2
    assert len(manifest["shards"][0]["sha256"]) == 64
    first = out / manifest["shards"][0]["file"]
    with tarfile.open(first) as tf:
        names = tf.getnames()
    assert any(n.endswith(".jpg") for n in names)
    assert any(n.endswith(".detect.txt") for n in names)
    assert any(n.endswith(".json") for n in names)


def test_diversity_can_export_complete_labeled_subset(tmp_path):
    root = tmp_path / "ds"
    _make_yolo(root, "train", "a", "0 0.5 0.5 0.2 0.2\n", 80, False)
    _make_yolo(root, "train", "b", "0 0.4 0.5 0.2 0.2\n", 180, True)
    report = analyze_diversity(root, limit=10, select=1)
    assert len(report["selected"]) == 1
    selected = [Path(report["selected"][0]["image"])]
    out = tmp_path / "subset"
    result = export_selected_dataset(root, selected, out)
    assert result["images"] == 1
    stem = selected[0].stem
    assert (out / "images" / "train" / selected[0].name).is_file()
    assert (out / "labels" / "train" / f"{stem}.txt").is_file()
    assert (out / "classes.txt").is_file()
