from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scenepaste.core.distribution import learn_distribution_profile
from scenepaste.core.similarity import available_embedding_backends, diversity_summary
from scenepaste.project import ScenePasteProject, init_project
from scenepaste.tools.hardmine import mine_hard_examples


def _img(path: Path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((80, 120, 3), value, np.uint8)
    cv2.rectangle(arr, (30, 20), (90, 70), (220, 40, 40), -1)
    assert cv2.imwrite(str(path), arr)


def _classes(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("0: person\n", encoding="utf-8")


def test_project_manifest_is_portable_and_validates(tmp_path):
    root = tmp_path / "project"; root.mkdir()
    objects = root / "objects"; objects.mkdir()
    backgrounds = root / "backgrounds"; backgrounds.mkdir()
    output = root / "generated"
    project = init_project(root, name="Demo", objects=objects, backgrounds=backgrounds,
                           output=output, class_map={"person": 0})
    data = json.loads(project.path.read_text(encoding="utf-8"))
    assert data["schema"] == "scenepaste/project"
    assert data["paths"]["objects"] == "objects"
    loaded = ScenePasteProject.load(project.path)
    assert loaded.objects_dir == objects.resolve()
    assert loaded.class_map == {"person": 0}
    assert loaded.validate(require_generation_paths=True)["ok"]


def test_distribution_learns_overlap_proxy_from_yolo(tmp_path):
    root = tmp_path / "ds"; _classes(root)
    _img(root / "images" / "train" / "a.jpg")
    labels = root / "labels" / "train"; labels.mkdir(parents=True)
    # Two overlapping boxes -> nonzero overlap IoU proxy.
    labels.joinpath("a.txt").write_text(
        "0 0.45 0.55 0.5 0.5\n0 0.58 0.55 0.5 0.5\n", encoding="utf-8")
    profile = learn_distribution_profile(root, bins=10)
    hist = profile.classes["person"]["overlap_iou"]
    assert sum(hist["counts"]) == 2
    assert sum(hist["counts"][1:]) >= 1


def test_seg_hardmine_scores_polygon_failures(tmp_path):
    root = tmp_path / "seg"; _classes(root)
    _img(root / "images" / "val" / "a.jpg")
    gt = root / "labels-seg" / "val"; gt.mkdir(parents=True)
    gt.joinpath("a.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.8 0.2 0.8\n", encoding="utf-8")
    pred = tmp_path / "predseg"; pred.mkdir()
    pred.joinpath("a.txt").write_text("", encoding="utf-8")
    report = mine_hard_examples(root, pred, split="val", task="seg")
    assert report["task"] == "seg"
    assert report["total_false_negatives"] == 1


def test_obb_hardmine_matches_rotated_geometry(tmp_path):
    root = tmp_path / "obb"; _classes(root)
    _img(root / "images" / "val" / "a.jpg")
    gt = root / "labels-obb" / "val"; gt.mkdir(parents=True)
    line = "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"
    gt.joinpath("a.txt").write_text(line + "\n", encoding="utf-8")
    pred = tmp_path / "predobb"; pred.mkdir()
    pred.joinpath("a.txt").write_text(line + " 0.92\n", encoding="utf-8")
    report = mine_hard_examples(root, pred, split="val", task="obb")
    assert report["total_false_negatives"] == 0
    assert report["total_false_positives"] == 0
    assert report["hard_images"] == 0


def test_embedding_backends_keep_cv_lite_default(tmp_path):
    assert {"cv-lite-v1", "clip", "dinov2"}.issubset(set(available_embedding_backends()))
    p1 = tmp_path / "a.jpg"; p2 = tmp_path / "b.jpg"
    _img(p1, 80); _img(p2, 160)
    report = diversity_summary([p1, p2], backend="cv-lite-v1")
    assert report["backend"] == "cv-lite-v1"
    assert report["samples"] == 2


def test_generation_writes_visibility_diagnostics(tmp_path):
    from scenepaste import GenerationConfig, generate_dataset
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "generated"
    summary = generate_dataset(GenerationConfig(
        objects_dir=repo / "samples" / "objects",
        backgrounds_dir=repo / "samples" / "backgrounds",
        output_dir=out,
        class_map={"person": 0, "truck": 1, "motorcycle": 2},
        count=3, min_objects=1, max_objects=2, output_format="all",
        save_previews=False, workers=1, seed=7,
    ))
    path = out / summary["generation_diagnostics"]
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["images"] == summary["generated_images"]
    assert sum(data["visible_ratio"]["counts"]) == summary["generated_objects"]
    assert (out / "latest_generation_diagnostics.json").is_file()


def test_project_validation_checks_path_kind_and_defaults(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    objects_file = root / "objects"
    objects_file.write_text("not a directory", encoding="utf-8")
    backgrounds = root / "backgrounds"
    backgrounds.mkdir()
    project = ScenePasteProject(
        path=root / "scenepaste.project.json",
        objects_dir=objects_file,
        backgrounds_dir=backgrounds,
        output_dir=root / "generated",
        class_map={"person": 0},
        defaults={"workers": -1, "preview_ratio": 1.2},
    )
    report = project.validate(require_generation_paths=True)
    assert not report["ok"]
    assert any("expected directory: objects" in error for error in report["errors"])
    assert any("defaults.workers" in error for error in report["errors"])
    assert any("defaults.preview_ratio" in error for error in report["errors"])


def test_project_rejects_future_manifest_version(tmp_path):
    path = tmp_path / "scenepaste.project.json"
    path.write_text(json.dumps({
        "schema": "scenepaste/project",
        "version": 999,
        "name": "future",
        "paths": {},
        "class_map": {"person": 0},
    }), encoding="utf-8")
    try:
        ScenePasteProject.load(path)
    except ValueError as exc:
        assert "newer than supported" in str(exc)
    else:
        raise AssertionError("future project manifest should be rejected")


def test_optional_embedding_backend_errors_are_not_silenced(tmp_path, monkeypatch):
    import scenepaste.core.similarity as similarity

    image = tmp_path / "a.jpg"
    _img(image, 100)

    def fail(_images, _backend):
        raise RuntimeError("missing model")

    monkeypatch.setattr(similarity, "_model_embeddings", fail)
    try:
        similarity.embed_paths([image], backend="clip")
    except RuntimeError as exc:
        assert "Failed to compute 'clip' embeddings" in str(exc)
    else:
        raise AssertionError("optional embedding failures must be explicit")


def test_model_embedding_path_batches_inference(tmp_path, monkeypatch):
    import scenepaste.core.similarity as similarity

    paths = []
    for idx in range(5):
        path = tmp_path / f"{idx}.jpg"
        _img(path, 60 + idx * 20)
        paths.append(path)
    calls = []

    def fake(images, backend):
        calls.append((len(images), backend))
        rows = []
        for idx, _image in enumerate(images):
            vec = np.array([1.0, float(idx + 1)], dtype=np.float32)
            vec /= np.linalg.norm(vec)
            rows.append(vec)
        return np.stack(rows)

    monkeypatch.setattr(similarity, "_model_embeddings", fake)
    used, emb = similarity.embed_paths(paths, backend="dinov2", batch_size=2)
    assert used == paths
    assert emb.shape == (5, 2)
    assert [size for size, _backend in calls] == [2, 2, 1]


def test_project_generation_update_preserves_data_loop_metadata(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    val = root / "val"
    val.mkdir()
    pred = root / "pred"
    pred.mkdir()
    project = ScenePasteProject(
        path=root / "scenepaste.project.json",
        name="Demo",
        real_dataset=real,
        validation_dataset=val,
        predictions_dir=pred,
        class_map={"person": 0},
    )
    project.update_generation(
        objects_dir=root / "objects",
        backgrounds_dir=root / "backgrounds",
        output_dir=root / "generated",
        defaults={"workers": 8, "output_format": "all"},
    )
    assert project.real_dataset == real
    assert project.validation_dataset == val
    assert project.predictions_dir == pred
    assert project.defaults["workers"] == 8
    assert project.defaults["output_format"] == "all"
