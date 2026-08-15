"""Cross-format consistency: detect / seg / coco / semantic must agree.

The mask-first pipeline guarantees that segmentation, COCO bbox/area, and
semantic pixels all derive from the same final visible pixels. These tests
generate once and check the formats agree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scenepaste import GenerationConfig, generate_dataset, parse_class_map

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gen(fmt: str, tmp_path: Path):
    cfg = GenerationConfig(
        objects_dir=REPO_ROOT / "samples" / "objects",
        backgrounds_dir=REPO_ROOT / "samples" / "backgrounds",
        output_dir=tmp_path,
        class_map=parse_class_map("person=0"),
        count=2,
        seed=42,
        output_format=fmt,
        run_id=f"e2e_{fmt}",
    )
    return generate_dataset(cfg)


def test_coco_bbox_inside_image_and_nonzero_area(tmp_path: Path):
    _gen("coco", tmp_path)
    coco = json.loads((tmp_path / "instances_coco.json").read_text(encoding="utf-8"))
    img_by_id = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        img = img_by_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0, "zero-area coco bbox"
        assert x >= 0 and y >= 0, "bbox origin off-canvas"
        assert x + w <= img["width"] + 1e-6
        assert y + h <= img["height"] + 1e-6
        # Segmentation polygon count must be >= 1, area must be > 0.
        assert ann["segmentation"], "empty segmentation"
        assert ann["area"] > 0


def test_semantic_mask_pixel_values_match_class_map(tmp_path: Path):
    _gen("semantic", tmp_path)
    mapping = json.loads((tmp_path / "semantic_classes.json").read_text(encoding="utf-8"))
    valid_values = {0}  # background
    for k in mapping:
        valid_values.add(int(k))
    for mask_path in (tmp_path / "masks" / "train").glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        present = set(np.unique(mask).tolist())
        extra = present - valid_values
        assert not extra, f"{mask_path.name} has out-of-map values: {extra}"


def test_obbb_lines_normalized_in_unit_range(tmp_path: Path):
    _gen("obb", tmp_path)
    for label in (tmp_path / "labels" / "train").glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            coords = [float(v) for v in tokens[1:]]
            for c in coords:
                assert 0.0 <= c <= 1.0, f"obb coord out of [0,1]: {c} in {label.name}"


def test_detect_and_seg_count_match(tmp_path: Path):
    """`both` mode emits detect labels and a parallel seg dir; counts must match."""
    _gen("both", tmp_path)
    detect = sorted(p.name for p in (tmp_path / "labels" / "train").glob("*.txt"))
    seg = sorted(p.name for p in (tmp_path / "labels-seg" / "train").glob("*.txt"))
    assert detect == seg, "detect and seg stems diverged"
