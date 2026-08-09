"""Editor object-appearance preview helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from compose_app.models import Instance
from compose_app.rendering import apply_instance_appearance, build_instance_appearance_recipe


def _rgba(h=48, w=36, color=(30, 90, 200)):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[6:-6, 5:-5, 3] = 255
    return Image.fromarray(arr, "RGBA")


def test_disabled_appearance_is_noop():
    rgba = _rgba()
    inst = Instance(0, 0, 0, 0.2, appearance_enabled=False, appearance_recipe="mild")
    assert build_instance_appearance_recipe(inst) is None
    out = apply_instance_appearance(rgba, inst, class_label="person")
    assert np.array_equal(np.asarray(out), np.asarray(rgba))


def test_slider_only_override_changes_rgb_keeps_alpha():
    rgba = _rgba()
    inst = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="off",
        appearance_brightness=0.15,
        appearance_contrast=1.0,
        appearance_saturation=1.0,
        appearance_blur=0.0,
        appearance_seed=7,
    )
    out = apply_instance_appearance(rgba, inst, class_label="person")
    a0 = np.asarray(rgba)
    a1 = np.asarray(out)
    assert np.array_equal(a0[..., 3], a1[..., 3])
    assert not np.array_equal(a0[..., :3][a0[..., 3] > 0], a1[..., :3][a1[..., 3] > 0])
    assert out.size == rgba.size


def test_recipe_seed_is_deterministic():
    rgba = _rgba()
    inst = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="mild",
        appearance_seed=123,
    )
    a = np.asarray(apply_instance_appearance(rgba, inst, class_label="truck"))
    b = np.asarray(apply_instance_appearance(rgba, inst, class_label="truck"))
    assert np.array_equal(a, b)


def test_get_rendered_includes_appearance_in_cache_key():
    rgba = _rgba()
    inst = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="off",
        appearance_brightness=0.1,
        appearance_seed=1,
    )
    first = inst.get_rendered(rgba, 40, class_label="person")
    inst.appearance_brightness = 0.2
    inst.invalidate_cache()
    second = inst.get_rendered(rgba, 40, class_label="person")
    assert not np.array_equal(np.asarray(first), np.asarray(second))
