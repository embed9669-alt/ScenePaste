"""Coverage for labelme.py: shape_to_mask variants, paste zones, asset loading edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scenepaste.core.labelme import (
    load_object_assets,
    load_paste_zone_mask,
    resolve_labelme_image,
    shape_to_mask,
)


def _write_image(path: Path, w: int = 100, h: int = 80) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 1] = 128  # plain green
    cv2.imwrite(str(path), arr)
    return arr


def test_shape_to_mask_polygon_returns_filled_mask():
    shape = {"shape_type": "polygon", "points": [[10, 10], [50, 10], [50, 60], [10, 60]]}
    mask = shape_to_mask(shape, height=100, width=100)
    assert mask is not None
    assert mask[30, 30] == 255      # inside
    assert mask[5, 5] == 0          # outside corner
    assert mask.shape == (100, 100)


def test_shape_to_mask_rectangle():
    shape = {"shape_type": "rectangle", "points": [[20, 20], [70, 70]]}
    mask = shape_to_mask(shape, height=100, width=100)
    assert mask is not None
    assert mask[40, 40] == 255
    assert mask[10, 10] == 0


def test_shape_to_mask_invalid_returns_none():
    assert shape_to_mask({"shape_type": "circle", "points": [[0, 0], [10, 10]]}, 100, 100) is None
    assert shape_to_mask({"shape_type": "polygon", "points": []}, 100, 100) is None
    assert shape_to_mask({"shape_type": "polygon", "points": [[1, 1], [2, 2]]}, 100, 100) is None


def test_load_paste_zone_mask_returns_none_when_no_json(tmp_path: Path):
    bg = tmp_path / "bg.jpg"
    _write_image(bg)
    assert load_paste_zone_mask(bg, 100, 100) is None


def test_load_paste_zone_mask_unions_multiple_zones(tmp_path: Path):
    bg = tmp_path / "bg.jpg"
    _write_image(bg)
    json_path = tmp_path / "bg.json"
    json_path.write_text(json.dumps({
        "imagePath": "bg.jpg",
        "shapes": [
            {"label": "paste_zone", "shape_type": "rectangle", "points": [[0, 0], [40, 40]]},
            {"label": "ground", "shape_type": "rectangle", "points": [[60, 60], [100, 100]]},
            {"label": "ignore_me", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
        ],
    }))
    mask = load_paste_zone_mask(bg, 100, 100)
    assert mask is not None
    assert mask[20, 20] > 0      # first zone
    assert mask[80, 80] > 0      # second zone
    assert mask[50, 50] == 0     # gap between zones


def test_load_paste_zone_mask_skips_when_only_unknown_labels(tmp_path: Path):
    bg = tmp_path / "bg.jpg"
    _write_image(bg)
    (tmp_path / "bg.json").write_text(json.dumps({
        "shapes": [{"label": "sky", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]}],
    }))
    assert load_paste_zone_mask(bg, 100, 100) is None


def test_resolve_labelme_image_with_embedded(tmp_path: Path):
    arr = _write_image(tmp_path / "src.jpg")
    ok, buf = cv2.imencode(".jpg", arr)
    import base64
    data = {
        "imageData": base64.b64encode(buf).decode("ascii"),
        "imagePath": "",
        "shapes": [],
    }
    img, source = resolve_labelme_image(tmp_path / "x.json", data)
    assert img is not None
    assert source == "embedded:imageData"


def test_resolve_labelme_image_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_labelme_image(tmp_path / "nope.json", {"imagePath": "", "shapes": []})


def test_load_object_assets_skips_unknown_labels(tmp_path: Path):
    """Shapes whose label is not in class_map are skipped silently."""
    _write_image(tmp_path / "asset.jpg", w=60, h=60)
    (tmp_path / "asset.json").write_text(json.dumps({
        "imagePath": "asset.jpg",
        "shapes": [
            {"label": "person", "shape_type": "rectangle", "points": [[5, 5], [40, 40]]},
            {"label": "motorcycle", "shape_type": "rectangle", "points": [[5, 5], [40, 40]]},
        ],
    }))
    messages = []
    assets = load_object_assets(
        tmp_path, class_map={"person": 0}, feather_sigma=0.0, log=messages.append,
    )
    assert len(assets) == 1
    assert assets[0].label == "person"
    # The hint about ignored motorcycle label should be logged.
    assert any("motorcycle" in m for m in messages)


def test_load_object_assets_polygon_crop(tmp_path: Path):
    _write_image(tmp_path / "asset.jpg", w=80, h=80)
    (tmp_path / "asset.json").write_text(json.dumps({
        "imagePath": "asset.jpg",
        "shapes": [
            {"label": "truck", "shape_type": "polygon",
             "points": [[10, 10], [60, 10], [60, 60], [10, 60]]},
        ],
    }))
    assets = load_object_assets(
        tmp_path, class_map={"truck": 0}, feather_sigma=0.0, log=lambda _m: None,
    )
    assert len(assets) == 1
    asset = assets[0]
    assert asset.polygon is not None
    assert asset.polygon.shape[1] == 2
    # Polygon coordinates live in the crop's local frame (>= 0).
    assert asset.polygon[:, 0].min() >= 0
    assert asset.polygon[:, 1].min() >= 0


def test_load_object_assets_raises_when_no_json(tmp_path: Path):
    with pytest.raises(RuntimeError):
        load_object_assets(tmp_path, {"person": 0}, 0.0, lambda _m: None)
