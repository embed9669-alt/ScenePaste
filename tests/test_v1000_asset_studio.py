from pathlib import Path
import json

import cv2
import numpy as np

from scenepaste.core.labelme import load_object_assets

from scenepaste.core.asset_studio import (
    binary_mask,
    export_asset_bundle,
    fill_mask_holes,
    make_clean_background,
    make_foreground_rgba,
    mask_to_polygons,
    morph_mask,
)


def test_mask_edit_helpers_are_binary_and_reversible_enough():
    mask = np.zeros((80, 100), np.uint8)
    cv2.rectangle(mask, (25, 20), (70, 60), 255, -1)
    cv2.circle(mask, (47, 40), 8, 0, -1)
    filled = fill_mask_holes(mask)
    assert filled[40, 47] == 255
    expanded = morph_mask(filled, 4)
    assert np.count_nonzero(expanded) > np.count_nonzero(filled)
    eroded = morph_mask(expanded, -4)
    assert eroded.dtype == np.uint8
    assert set(np.unique(binary_mask(eroded))).issubset({0, 255})


def test_foreground_and_background_are_controlled_by_separate_masks():
    img = np.full((96, 128, 3), 150, np.uint8)
    img[30:70, 45:85] = (10, 30, 220)
    fg = np.zeros((96, 128), np.uint8)
    fg[30:70, 45:85] = 255
    bg = morph_mask(fg, 8)

    rgba, offset = make_foreground_rgba(img, fg, feather_px=0, crop=True)
    assert rgba.shape[2] == 4
    assert offset[0] <= 45 and offset[1] <= 30
    assert rgba[..., 3].max() == 255

    clean = make_clean_background(img, bg, expand_px=0, radius=3)
    # Pixels far from the removal region are preserved exactly.
    assert np.array_equal(clean[:15, :15], img[:15, :15])
    # The object centre should be reconstructed rather than copied unchanged.
    assert not np.array_equal(clean[50, 65], img[50, 65])


def test_mask_to_polygons_returns_labelme_ready_shapes():
    mask = np.zeros((80, 120), np.uint8)
    cv2.rectangle(mask, (10, 10), (40, 50), 255, -1)
    cv2.circle(mask, (90, 40), 12, 255, -1)
    polys = mask_to_polygons(mask)
    assert len(polys) == 2
    assert all(poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3 for poly in polys)


def test_export_bundle_is_directly_reusable_and_keeps_provenance(tmp_path: Path):
    image = np.full((120, 160, 3), 175, np.uint8)
    image[35:95, 50:110] = (30, 70, 210)
    src = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(src), image)
    auto = np.zeros((120, 160), np.uint8)
    auto[35:95, 50:110] = 255
    edited = auto.copy()
    edited[:40, :] = 0  # human edit differs from auto mask
    bg_remove = morph_mask(edited, 5)

    out = export_asset_bundle(
        source_image=src,
        bgr=image,
        label="truck",
        instance_index=2,
        auto_mask=auto,
        edited_mask=edited,
        background_remove_mask=bg_remove,
        objects_dir=tmp_path / "objects",
        backgrounds_dir=tmp_path / "backgrounds",
        bundle_root=tmp_path / ".scenepaste" / "asset_studio",
    )
    assert out.foreground_path.is_file()
    assert out.foreground_json.is_file()
    assert out.background_path.is_file()
    assert (out.bundle_dir / "mask_auto.png").is_file()
    assert (out.bundle_dir / "mask_edited.png").is_file()
    assert (out.bundle_dir / "mask_background_remove.png").is_file()
    assert out.edited_annotation.is_file()
    meta = json.loads((out.bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "scenepaste.asset_studio.v1"
    assert meta["label"] == "truck"
    assert meta["instance_index"] == 2
    assert meta["mask_area_px"] == int(np.count_nonzero(edited))

    # The reviewed foreground is written in ScenePaste's normal LabelMe asset
    # format and can immediately be loaded by the production Copy-Paste path.
    assets = load_object_assets(tmp_path / "objects", {"truck": 0}, 0.0, lambda _msg: None)
    assert len(assets) >= 1
    assert assets[0].label == "truck"
    assert float(assets[0].alpha.max()) > 0.9


def test_open_source_ui_promotes_asset_studio_and_copy_paste():
    root = Path(__file__).resolve().parents[1]
    app = (root / "compose_app_qt" / "app.py").read_text(encoding="utf-8")
    factory = (root / "compose_app_qt" / "large_generate.py").read_text(encoding="utf-8")
    gui = (root / "compose_app_qt" / "asset_studio_gui.py").read_text(encoding="utf-8")
    assert "素材工作室" in app
    assert "打开素材工作室" in app
    assert "可控 Copy-Paste" in factory
    assert "困难样本策略" in factory
    assert "生成方案" not in factory
    assert "SD Inpainting" not in factory
    assert "Qwen" not in factory
    assert "编辑前景 Mask（训练标签）" in gui
    assert "编辑背景移除 Mask" in gui
    assert "保存审核素材：前景 + Mask + 干净背景" in gui
