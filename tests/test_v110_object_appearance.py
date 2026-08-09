from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from scenepaste import GenerationConfig, generate_dataset
from scenepaste.cli import build_generate_parser, config_from_args
from scenepaste.core.object_appearance import apply_object_appearance, load_object_appearance_recipe
from scenepaste.tools.qa import write_qa_dashboard


def _recipe(effects: dict) -> dict:
    return {
        "schema": "scenepaste/object-appearance-recipe",
        "version": 1,
        "name": "test",
        "effects": effects,
    }


def test_unknown_effect_and_field_are_rejected():
    with pytest.raises(ValueError, match="未知 object appearance effect"):
        load_object_appearance_recipe(_recipe({"brightnes": {"p": 1.0}}))
    with pytest.raises(ValueError, match="未知字段"):
        load_object_appearance_recipe(_recipe({"hue": {"p": 1.0, "range": [-5, 5], "typo": 1}}))


def test_hue_range_is_expressed_in_degrees():
    # OpenCV H uses 0..179 for 0..358 degrees. A +20 degree recipe shift
    # should therefore move H by ~10 units.
    hsv = np.zeros((20, 20, 3), np.uint8)
    hsv[..., 0] = 40
    hsv[..., 1:] = 200
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = np.ones((20, 20), np.float32)
    out, meta = apply_object_appearance(
        image, alpha,
        _recipe({"hue": {"p": 1.0, "range": [20, 20]}}),
        random.Random(1),
    )
    out_h = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[10, 10, 0]
    assert abs(int(out_h) - 50) <= 1
    assert meta[0]["shift_degrees"] == 20


def test_alpha_aware_blur_does_not_pull_outside_color_into_object_edge():
    image = np.zeros((64, 64, 3), np.uint8)
    image[:] = (0, 255, 0)  # green outside the actual object
    image[16:48, 16:48] = (0, 0, 255)  # red object
    alpha = np.zeros((64, 64), np.float32)
    alpha[16:48, 16:48] = 1.0
    out, _ = apply_object_appearance(
        image, alpha,
        _recipe({"gaussian_blur": {"p": 1.0, "sigma": [2.0, 2.0]}}),
        random.Random(2),
    )
    # Just inside the boundary should remain red rather than inheriting the
    # green pixels that happened to exist outside the segmentation mask.
    b, g, r = [int(v) for v in out[16, 32]]
    assert r > 220
    assert g < 35
    assert b < 35


def test_generation_preloads_custom_recipe_and_qa_reports_coverage(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    recipe_path = tmp_path / "object_recipe.json"
    recipe_path.write_text(json.dumps(_recipe({
        "brightness_contrast": {"p": 1.0, "brightness": [0.05, 0.05], "contrast": [1.0, 1.0]}
    })), encoding="utf-8")

    import scenepaste.core.advanced_pipeline as advanced
    original = advanced.load_object_appearance_recipe
    calls = []

    def counted(source):
        calls.append(str(source))
        return original(source)

    monkeypatch.setattr(advanced, "load_object_appearance_recipe", counted)
    out = tmp_path / "generated"
    summary = generate_dataset(GenerationConfig(
        objects_dir=repo / "samples" / "objects",
        backgrounds_dir=repo / "samples" / "backgrounds",
        output_dir=out,
        class_map={"person": 0, "truck": 1, "motorcycle": 2},
        count=5,
        min_objects=1,
        max_objects=2,
        workers=1,
        save_previews=False,
        object_appearance_recipe=str(recipe_path),
        run_id="appearance",
    ))
    assert summary["generated_images"] == 5
    # One load/validation for the worker context, not once per pasted object.
    assert calls == [str(recipe_path)]
    diagnostics = json.loads((out / "latest_generation_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["object_effect_counts"]["brightness_contrast"] == summary["generated_objects"]

    report = write_qa_dashboard(out, embedding_limit=0)
    html = Path(report["html_path"]).read_text(encoding="utf-8")
    assert "Appearance coverage" in html
    assert "Object-level effects" in html
    assert "brightness_contrast" in html


def test_project_relative_object_recipe_resolves_from_manifest(tmp_path):
    root = tmp_path / "project"
    (root / "objects").mkdir(parents=True)
    (root / "backgrounds").mkdir()
    (root / "configs").mkdir()
    recipe = root / "configs" / "objects.json"
    recipe.write_text(json.dumps(_recipe({"hue": {"p": 0.0, "range": [-5, 5]}})), encoding="utf-8")
    manifest = root / "scenepaste.project.json"
    manifest.write_text(json.dumps({
        "schema": "scenepaste/project",
        "version": 1,
        "name": "portable",
        "paths": {"objects": "objects", "backgrounds": "backgrounds", "output": "generated"},
        "class_map": {"person": 0},
        "defaults": {"object_appearance_recipe": "configs/objects.json"},
    }), encoding="utf-8")

    parser = build_generate_parser()
    args = parser.parse_args(["--project", str(manifest)])
    args._provided_options = {"--project"}
    cfg = config_from_args(args)
    assert Path(cfg.object_appearance_recipe) == recipe.resolve()
