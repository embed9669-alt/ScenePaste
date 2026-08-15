from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scenepaste.explorer import index_dataset, render_dataset_image


def _base_dataset(tmp_path: Path):
    root = tmp_path / "ds"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image_dir / "a.jpg")
    (root / "classes.txt").write_text("0: person\n1: truck\n", encoding="utf-8")
    return root, label_dir


def test_index_and_detect_overlay(tmp_path: Path):
    root, label_dir = _base_dataset(tmp_path)
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    items = index_dataset(root)
    assert len(items) == 1
    result = render_dataset_image(root, items[0])
    assert result.object_count == 1
    assert "YOLO Detect" in result.format_name
    assert result.image.size == (100, 80)


def test_obb_and_semantic_overlay(tmp_path: Path):
    root, label_dir = _base_dataset(tmp_path)
    (label_dir / "a.txt").write_text("1 0.2 0.2 0.8 0.2 0.8 0.7 0.2 0.7\n", encoding="utf-8")
    mask_dir = root / "masks" / "train"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:60, 20:80] = 2
    Image.fromarray(mask).save(mask_dir / "a.png")
    result = render_dataset_image(root, index_dataset(root)[0])
    assert result.object_count == 1
    assert "YOLO OBB" in result.format_name
    assert "Semantic Mask" in result.format_name


def test_coco_overlay(tmp_path: Path):
    root, _ = _base_dataset(tmp_path)
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [10, 10, 30, 40], "area": 1200,
            "segmentation": [[10, 10, 40, 10, 40, 50, 10, 50]],
        }],
    }
    (root / "instances_coco.json").write_text(json.dumps(coco), encoding="utf-8")
    result = render_dataset_image(root, index_dataset(root)[0])
    assert result.object_count == 1
    assert "COCO Instances" in result.format_name


def test_scene_template_portable_source_identity(tmp_path: Path):
    from types import SimpleNamespace
    from PIL import Image
    from compose_app.models import Cutout, Instance
    from compose_app_qt import templates

    old_cutout = Cutout(
        label="person", class_id=0,
        source="/old/machine/assets/person_001.json#0",
        rgba=Image.new("RGBA", (20, 40), (255, 0, 0, 255)), polygon=None, thumb=None,
    )
    old_doc = SimpleNamespace(
        bg_size=(800, 600), cutouts=[old_cutout],
        instances=[Instance(cutout_index=0, cx=400, cy=300, h_ratio=0.2, uid=1)],
    )
    path = tmp_path / "scene.json"
    assert templates.save_template(old_doc, path) == 1

    new_cutout = Cutout(
        label="person", class_id=0,
        source="/new/location/assets/person_001.json#0",
        rgba=Image.new("RGBA", (20, 40), (255, 0, 0, 255)), polygon=None, thumb=None,
    )
    class Doc:
        bg_size = (1000, 500)
        cutouts = [new_cutout]
        _uid = 10
        def next_uid(self):
            self._uid += 1
            return self._uid
    restored, missing = templates.load_template(Doc(), path)
    assert not missing
    assert len(restored) == 1
    assert restored[0].cutout_index == 0
