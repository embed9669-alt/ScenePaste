from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from scenepaste import GenerationConfig, generate_dataset, parse_class_map
from scenepaste.core.distribution import learn_distribution_profile, mix_distribution_profiles
from scenepaste.core.recipes import BUILTIN_RECIPES, apply_scene_recipe
from scenepaste.core.templates import sample_template, template_constraints_satisfied
from scenepaste.tools.qa import build_qa_report

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ROOT / "samples" / "objects"
BACKGROUNDS = ROOT / "samples" / "backgrounds"


def test_scene_recipe_is_deterministic_and_shape_preserving():
    image = np.full((80, 120, 3), 127, dtype=np.uint8)
    a, meta_a = apply_scene_recipe(image, BUILTIN_RECIPES["surveillance"], random.Random(9))
    b, meta_b = apply_scene_recipe(image, BUILTIN_RECIPES["surveillance"], random.Random(9))
    assert a.shape == image.shape
    assert np.array_equal(a, b)
    assert meta_a == meta_b


def test_all_output_exports_every_annotation_modality(tmp_path):
    out = tmp_path / "all"
    summary = generate_dataset(GenerationConfig(
        objects_dir=OBJECTS, backgrounds_dir=BACKGROUNDS, output_dir=out,
        class_map=parse_class_map("person=0"), count=2, run_id="allrun",
        output_format="all", save_previews=False,
    ))
    assert summary["generated_images"] == 2
    stems = [p.stem for p in (out / "images" / "train").glob("*.jpg")]
    assert len(stems) == 2
    for stem in stems:
        assert (out / "labels" / "train" / f"{stem}.txt").is_file()
        assert (out / "labels-seg" / "train" / f"{stem}.txt").is_file()
        assert (out / "labels-obb" / "train" / f"{stem}.txt").is_file()
        assert (out / "masks" / "train" / f"{stem}.png").is_file()
    assert (out / "instances_coco.json").is_file()


def test_background_only_negative_samples_are_valid(tmp_path):
    out = tmp_path / "neg"
    summary = generate_dataset(GenerationConfig(
        objects_dir=OBJECTS, backgrounds_dir=BACKGROUNDS, output_dir=out,
        class_map=parse_class_map("person=0"), count=3, run_id="neg",
        output_format="detect", save_previews=False, empty_scene_probability=1.0,
    ))
    assert summary["generated_images"] == 3
    assert summary["generated_objects"] == 0
    for label in (out / "labels" / "train").glob("*.txt"):
        assert label.read_text(encoding="utf-8") == ""


def test_template_relation_constraints_are_enforced():
    tpl = {
        "schema": "scenepaste/scene-template", "version": 2,
        "instances": [
            {"id":"a", "label":"person", "source":"a#0", "cx_ratio":.4, "cy_ratio":.6, "h_ratio":.2,
             "variation":{"cx_range":[.1,.45]}},
            {"id":"b", "label":"truck", "source":"b#0", "cx_ratio":.6, "cy_ratio":.6, "h_ratio":.2,
             "variation":{"cx_range":[.55,.9]}},
        ],
        "constraints": [{"type":"left_of", "a":"a", "b":"b", "margin":.05}],
        "parameters": {"constraint_tries":20},
    }
    placements = sample_template(tpl, random.Random(4), {"person":0,"truck":1})
    assert len(placements) == 2
    assert template_constraints_satisfied(placements, tpl["constraints"])
    by_id = {p.slot_id:p for p in placements}
    assert by_id["a"].cx_ratio + .05 < by_id["b"].cx_ratio


def test_profile_mix_respects_requested_domain_weights(tmp_path):
    p = learn_distribution_profile(OBJECTS, bins=10)
    # Build a second domain where only person remains.
    person_only = json.loads(json.dumps(p.data))
    person_only["classes"] = {"person": person_only["classes"]["person"]}
    person_only["object_count_total"] = person_only["classes"]["person"]["count"]
    q = type(p)(person_only)
    mixed = mix_distribution_profiles([p, q], [0.2, 0.8])
    assert mixed.data["source_type"] == "mixture"
    assert mixed.classes["person"]["count"] > mixed.classes["truck"]["count"]
    out = tmp_path / "mixed.json"
    mixed.save(out)
    assert out.is_file()


def test_class_specific_background_zone_restricts_person(tmp_path):
    bgs = tmp_path / "bgs"; bgs.mkdir()
    image = np.full((240, 320, 3), 180, dtype=np.uint8)
    bg = bgs / "bg.jpg"; cv2.imwrite(str(bg), image)
    payload = {
        "imagePath":"bg.jpg", "imageWidth":320, "imageHeight":240,
        "shapes":[{"label":"paste_zone:person","shape_type":"polygon",
                   "points":[[5,80],[150,80],[150,235],[5,235]]}],
    }
    (bgs / "bg.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "zone"
    generate_dataset(GenerationConfig(
        objects_dir=OBJECTS, backgrounds_dir=bgs, output_dir=out,
        class_map=parse_class_map("person=0"), count=3, min_objects=1, max_objects=1,
        run_id="zone", save_previews=False, seed=1,
    ))
    for label_file in (out / "labels" / "train").glob("*.txt"):
        parts = label_file.read_text().split()
        assert float(parts[1]) < 0.58


def test_qa_detects_perceptual_near_duplicates(tmp_path):
    root = tmp_path / "ds"; images = root / "images" / "train"; images.mkdir(parents=True)
    base = np.zeros((96,128,3), dtype=np.uint8)
    cv2.circle(base, (64,48), 24, (220,220,220), -1)
    cv2.imwrite(str(images / "a.png"), base)
    cv2.imwrite(str(images / "b.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 72])
    report = build_qa_report(root, duplicate_limit=10, near_duplicate_threshold=6)
    assert report["integrity"]["near_duplicate_images"] >= 1
    assert report["integrity"]["perceptual_diversity_ratio"] <= 1.0


def test_recipe_cli_lists_and_exports(tmp_path, capsys):
    from scenepaste.cli import recipe_main
    assert recipe_main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "camera-mild" in listed
    target = tmp_path / "recipe.json"
    assert recipe_main(["export", "camera-mild", "-o", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "scenepaste/augmentation-recipe"


def test_profile_mix_cli_writes_reusable_profile(tmp_path):
    from scenepaste.cli import profile_main
    profile = learn_distribution_profile(OBJECTS, bins=10)
    a = tmp_path / "a.json"; b = tmp_path / "b.json"; out = tmp_path / "mix.json"
    profile.save(a); profile.save(b)
    assert profile_main(["mix", str(a), str(b), "-w", "0.25", "0.75", "-o", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_type"] == "mixture"
    assert payload["weights"] == [0.25, 0.75]


def test_exact_duplicate_not_double_counted_as_near_duplicate(tmp_path):
    root = tmp_path / "ds"; images = root / "images" / "train"; images.mkdir(parents=True)
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(base, (10, 10), (52, 52), (180, 180, 180), -1)
    ok, encoded = cv2.imencode(".png", base)
    assert ok
    (images / "a.png").write_bytes(encoded.tobytes())
    (images / "b.png").write_bytes(encoded.tobytes())
    report = build_qa_report(root, duplicate_limit=10, near_duplicate_threshold=6)
    assert report["integrity"]["duplicate_images"] == 1
    assert report["integrity"]["near_duplicate_images"] == 0


def test_semantic_negative_scene_is_not_a_qa_warning(tmp_path):
    out = tmp_path / "semantic-negative"
    generate_dataset(GenerationConfig(
        objects_dir=OBJECTS, backgrounds_dir=BACKGROUNDS, output_dir=out,
        class_map=parse_class_map("person=0"), count=2, run_id="semneg",
        output_format="semantic", save_previews=False, empty_scene_probability=1.0,
    ))
    report = build_qa_report(out, duplicate_limit=10)
    semantic = report["summary"]["semantic"]
    assert semantic["empty_masks"] == 2
    assert semantic["intentional_empty_masks"] == 2
    assert semantic["suspicious_empty_masks"] == 0
    assert not any("empty semantic" in w for w in report["warnings"])
