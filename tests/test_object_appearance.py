"""Object appearance recipe unit tests."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from scenepaste.core.augmentation import augment_foreground
from scenepaste.core.object_appearance import (
    BUILTIN_OBJECT_RECIPES,
    apply_object_appearance,
    load_object_appearance_recipe,
    resolve_object_effects,
    save_object_appearance_recipe,
)


def _fg(h=64, w=48, color=(40, 80, 180)):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:] = color
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[8:-8, 6:-6] = 1.0
    return image, alpha


def test_builtin_object_recipes_load():
    for name in ("off", "legacy", "mild", "surveillance-object"):
        recipe = load_object_appearance_recipe(name)
        assert recipe["schema"] == "scenepaste/object-appearance-recipe"
        assert recipe["name"] == name


def test_mild_is_deterministic_and_preserves_shape_alpha():
    image, alpha = _fg()
    recipe = BUILTIN_OBJECT_RECIPES["mild"]
    a, meta_a = apply_object_appearance(image, alpha, recipe, random.Random(11))
    b, meta_b = apply_object_appearance(image, alpha, recipe, random.Random(11))
    assert a.shape == image.shape
    assert np.array_equal(a, b)
    assert meta_a == meta_b
    # Transparent pixels stay black/zero from the source canvas.
    assert np.array_equal(a[alpha <= 0.10], image[alpha <= 0.10])


def test_resolution_degrade_changes_interior():
    image, alpha = _fg()
    # Flat color survives downscale; use high-frequency content so degrade is visible.
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    base = ((xx * 17 + yy * 9) % 256).astype(np.uint8)
    image = np.stack([base, (base.astype(np.uint16) * 3 % 256).astype(np.uint8), 255 - base], axis=-1)
    recipe = {
        "schema": "scenepaste/object-appearance-recipe",
        "version": 1,
        "name": "res-only",
        "effects": {"resolution_degrade": {"p": 1.0, "scale": [0.25, 0.25]}},
    }
    out, meta = apply_object_appearance(image, alpha, recipe, random.Random(3))
    assert any(m["effect"] == "resolution_degrade" for m in meta)
    assert not np.array_equal(out[alpha > 0.10], image[alpha > 0.10])


def test_by_class_override_merges_effects():
    recipe = {
        "schema": "scenepaste/object-appearance-recipe",
        "version": 1,
        "name": "cls",
        "effects": {"hue": {"p": 1.0, "range": [-1, 1]}},
        "by_class": {
            "truck": {"effects": {"saturation": {"p": 1.0, "range": [1.2, 1.2]}}},
        },
    }
    base = resolve_object_effects(recipe, "person")
    truck = resolve_object_effects(recipe, "truck")
    assert "hue" in base and "saturation" not in base
    assert "saturation" in truck and "hue" in truck


def test_augment_foreground_recipe_path_records_color_match():
    image, alpha = _fg()
    bg = np.full((64, 48, 3), 200, dtype=np.uint8)
    out, meta = augment_foreground(
        image, alpha, bg, random.Random(5),
        color_match_strength=0.5,
        blur_prob=0.0,
        object_appearance_recipe="off",
        class_label="person",
    )
    assert out.shape == image.shape
    assert any(m["effect"] == "color_match" for m in meta)


def test_legacy_recipe_respects_blur_prob_null():
    image, alpha = _fg()
    out_a, meta_a = apply_object_appearance(
        image, alpha, BUILTIN_OBJECT_RECIPES["legacy"], random.Random(7),
        blur_prob_fallback=0.0,
    )
    out_b, meta_b = apply_object_appearance(
        image, alpha, BUILTIN_OBJECT_RECIPES["legacy"], random.Random(7),
        blur_prob_fallback=1.0,
    )
    assert any(m["effect"] == "hsv_jitter" for m in meta_a)
    assert not any(m["effect"] == "gaussian_blur" for m in meta_a)
    assert any(m["effect"] == "gaussian_blur" for m in meta_b)
    assert not np.array_equal(out_a, out_b)


def test_export_roundtrip(tmp_path):
    target = tmp_path / "obj.json"
    save_object_appearance_recipe(BUILTIN_OBJECT_RECIPES["mild"], target)
    loaded = load_object_appearance_recipe(target)
    assert loaded["name"] == "mild"
    assert "resolution_degrade" in loaded["effects"]


def test_recipe_cli_object_kind(tmp_path, capsys):
    from scenepaste.cli import recipe_main

    assert recipe_main(["--kind", "object", "list"]) == 0
    listed = capsys.readouterr().out
    assert "mild" in listed
    target = tmp_path / "exported.json"
    assert recipe_main(["--kind", "object", "export", "mild", "-o", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "scenepaste/object-appearance-recipe"


def test_invalid_schema_rejected():
    with pytest.raises(ValueError):
        load_object_appearance_recipe({"schema": "nope", "version": 1, "effects": {}})
