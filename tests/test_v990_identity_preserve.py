from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    from scenepaste.core.labelme import load_object_assets
except ModuleNotFoundError as exc:  # package may still be mid-refactor
    pytest.skip(f"scenepaste import unavailable: {exc}", allow_module_level=True)


def test_rectangle_annotation_grabcut_becomes_object_mask(tmp_path: Path):
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    image[:] = (45, 125, 45)
    # Synthetic motorcycle-ish shape on a contrasting scene.
    cv2.rectangle(image, (62, 54), (118, 76), (0, 0, 220), -1)
    cv2.circle(image, (70, 91), 17, (20, 20, 20), -1)
    cv2.circle(image, (118, 91), 17, (20, 20, 20), -1)
    cv2.line(image, (70, 54), (101, 38), (0, 0, 220), 7)
    cv2.imwrite(str(tmp_path / "bike.jpg"), image)
    data = {
        "imagePath": "bike.jpg",
        "shapes": [{
            "label": "motorcycle",
            "shape_type": "rectangle",
            "points": [[45, 28], [140, 115]],
        }],
    }
    (tmp_path / "bike.json").write_text(json.dumps(data), encoding="utf-8")
    logs = []
    assets = load_object_assets(
        tmp_path, {"motorcycle": 0}, 0.0, logs.append,
        rectangle_mask_mode="grabcut",
    )
    assert len(assets) == 1
    asset = assets[0]
    assert asset.mask_source == "rectangle-grabcut"
    fill = float(np.count_nonzero(asset.alpha > 0.5)) / float(asset.alpha.size)
    # Not the full source rectangle any more.
    assert 0.04 < fill < 0.75
    assert asset.polygon is not None
    assert any("矩形标注前景细化" in row for row in logs)
