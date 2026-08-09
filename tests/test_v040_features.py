from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import scenepaste.core as core
from compose_app import io_utils, segmentation as seg
from compose_app.models import Cutout, Instance
import split_dataset


def _asset(tmp_path: Path, class_id: int = 0) -> core.ObjectAsset:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = (20, 100, 200)
    alpha = np.ones((20, 20), dtype=np.float32)
    poly = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype=np.float32)
    return core.ObjectAsset("x", class_id, image, alpha, tmp_path / "x.json", 0, poly)


def test_semantic_class_zero_is_not_background(tmp_path: Path):
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:10, 5:12] = 1  # class_id 0 must be stored as value 1
    assert set(np.unique(mask)) == {0, 1}
    path = core.write_semantic_classes({"person": 0, "truck": 1}, tmp_path)
    mapping = json.loads(path.read_text(encoding="utf-8"))
    assert mapping == {"0": "background", "1": "person", "2": "truck"}


def test_ultralytics_obb_has_four_normalized_corners():
    mask = np.zeros((100, 200), dtype=bool)
    cv2.rectangle(mask.view(np.uint8), (20, 30), (80, 60), 1, -1)
    line = core.ultralytics_obb_line(2, mask, 200, 100)
    parts = line.split()
    assert parts[0] == "2"
    assert len(parts) == 9
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_visible_masks_remove_occluded_pixels():
    rgba = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    cutouts = [Cutout("a", 0, "a#0", rgba), Cutout("b", 1, "b#0", rgba)]
    # exact same location: second instance fully occludes first
    instances = [Instance(0, 50, 50, 0.2), Instance(1, 50, 50, 0.2)]
    masks = seg.visible_instance_masks(cutouts, instances, 100, 100)
    assert np.count_nonzero(masks[0]) == 0
    assert np.count_nonzero(masks[1]) > 0


def test_gui_semantic_and_obb_outputs(tmp_path: Path):
    bg = Image.new("RGB", (100, 100), (100, 100, 100))
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    cut = Cutout("person", 0, "p.json#0", rgba,
                 polygon=np.array([[0, 0], [20, 0], [20, 30], [0, 30]], dtype=np.float32))
    inst = Instance(0, 50, 50, 0.3, angle=25)

    out_sem = tmp_path / "sem"
    _, primary, _ = io_utils.save_composite(bg, Path("bg.jpg"), [inst], [cut], out_sem,
                                             "sample", output_format="semantic")
    assert primary and primary.exists()
    arr = cv2.imdecode(np.fromfile(str(primary), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert 1 in np.unique(arr)  # class 0 -> semantic value 1

    out_obb = tmp_path / "obb"
    _, primary, _ = io_utils.save_composite(bg, Path("bg.jpg"), [inst], [cut], out_obb,
                                             "sample", output_format="obb")
    parts = primary.read_text(encoding="utf-8").strip().split()
    assert len(parts) == 9
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_coco_writer_ids_are_o1_counters(tmp_path: Path):
    p = tmp_path / "coco.json"
    w = core.CocoWriter(p, checkpoint_interval=0)
    i1 = w.add_image("a.jpg", 10, 10)
    a1 = w.add_annotation(i1, 0, [[0, 0, 1, 0, 1, 1]], [0, 0, 1, 1], 1)
    i2 = w.add_image("b.jpg", 10, 10)
    a2 = w.add_annotation(i2, 0, [[0, 0, 1, 0, 1, 1]], [0, 0, 1, 1], 1)
    assert (i1, i2) == (1, 2)
    assert (a1, a2) == (1, 2)
    assert not p.exists()  # no per-image rewrite
    w.finalize()
    assert p.exists()


def test_split_resolves_latest_run_log(tmp_path: Path):
    log = tmp_path / "run_20260101_000000_log.csv"
    log.write_text("generated_stem,source_json,shape_index\na,a.json,0\n", encoding="utf-8")
    (tmp_path / "latest_summary.json").write_text(
        json.dumps({"log_file": log.name}), encoding="utf-8")
    assert split_dataset.resolve_log_path(tmp_path) == log


def test_balanced_asset_sampling_ignores_pool_size_bias():
    import random
    import numpy as np
    from pathlib import Path
    import scenepaste.core as core

    def asset(cid, n):
        return core.ObjectAsset(
            label=f"c{cid}", class_id=cid,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            alpha=np.ones((2, 2), dtype=np.uint8) * 255,
            source_json=Path(f"c{cid}_{n}.json"), source_shape_index=n,
        )

    assets = [asset(0, i) for i in range(20)] + [asset(1, 0)]
    groups = core._build_asset_groups(assets)
    rng = random.Random(123)
    counts = {0: 0, 1: 0}
    for _ in range(2000):
        counts[core.sample_asset(rng, assets, groups, "balanced").class_id] += 1
    assert abs(counts[0] - counts[1]) < 150


def test_balanced_background_sampler_cycle_usage():
    import random
    from pathlib import Path
    import scenepaste.core as core

    paths = [Path(f"{i}.jpg") for i in range(5)]
    sampler = core.BackgroundSampler(paths, random.Random(7), "balanced")
    drawn = [sampler.next() for _ in range(12)]
    counts = {p: drawn.count(p) for p in paths}
    assert max(counts.values()) - min(counts.values()) <= 1
