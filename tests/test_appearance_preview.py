"""Editor object-appearance preview helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from compose_app.models import Instance
from compose_app.rendering import (
    apply_appearance_meta_to_sliders,
    apply_instance_appearance,
    build_instance_appearance_recipe,
    sample_recipe_into_sliders,
)


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


def test_extra_slider_overrides_are_applied():
    rgba = _rgba()
    base = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="off",
        appearance_seed=3,
    )
    tweaked = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="off",
        appearance_hue=18.0,
        appearance_temperature=20.0,
        appearance_noise=6.0,
        appearance_sharpness=1.35,
        appearance_seed=3,
    )
    recipe = build_instance_appearance_recipe(tweaked)
    assert recipe is not None
    effects = recipe["effects"]
    assert "hue" in effects
    assert "color_temperature" in effects
    assert "gaussian_noise" in effects
    assert "sharpness" in effects
    out0 = np.asarray(apply_instance_appearance(rgba, base, class_label="person"))
    out1 = np.asarray(apply_instance_appearance(rgba, tweaked, class_label="person"))
    mask = out0[..., 3] > 0
    assert not np.array_equal(out0[..., :3][mask], out1[..., :3][mask])


def test_apply_meta_to_sliders_maps_known_effects():
    inst = Instance(0, 0, 0, 0.2)
    apply_appearance_meta_to_sliders(inst, [
        {"effect": "brightness_contrast", "brightness": 0.12, "contrast": 1.08},
        {"effect": "saturation", "factor": 1.15},
        {"effect": "hue", "shift_degrees": -7.5},
        {"effect": "color_temperature", "strength": 11.0},
        {"effect": "gaussian_blur", "sigma": 0.9},
        {"effect": "gaussian_noise", "sigma": 4.0},
        {"effect": "sharpness", "amount": 1.2},
    ])
    assert abs(inst.appearance_brightness - 0.12) < 1e-6
    assert abs(inst.appearance_contrast - 1.08) < 1e-6
    assert abs(inst.appearance_saturation - 1.15) < 1e-6
    assert abs(inst.appearance_hue - (-7.5)) < 1e-6
    assert abs(inst.appearance_temperature - 11.0) < 1e-6
    assert abs(inst.appearance_blur - 0.9) < 1e-6
    assert abs(inst.appearance_noise - 4.0) < 1e-6
    assert abs(inst.appearance_sharpness - 1.2) < 1e-6


def test_sample_recipe_into_sliders_bakes_and_switches_to_off():
    rgba = _rgba()
    inst = Instance(
        0, 0, 0, 0.2,
        appearance_enabled=True,
        appearance_recipe="mild",
        appearance_seed=42,
        # Stale slider values must be replaced by the sampled bake.
        appearance_brightness=0.2,
        appearance_blur=2.0,
    )
    meta = sample_recipe_into_sliders(inst, rgba, class_label="person", recipe_name="mild")
    assert inst.appearance_recipe == "off"
    assert inst.appearance_enabled is True
    assert isinstance(meta, list)
    # Re-rendering via baked sliders must be deterministic.
    a = np.asarray(apply_instance_appearance(rgba, inst, class_label="person"))
    b = np.asarray(apply_instance_appearance(rgba, inst, class_label="person"))
    assert np.array_equal(a, b)


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
