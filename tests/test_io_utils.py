"""IO 与类别映射测试。"""

from __future__ import annotations

import json
from pathlib import Path


from compose_app import io_utils


def test_build_class_map_empty():
    cm = io_utils.build_class_map([], "")
    assert cm == {}


def test_build_class_map_from_scratch():
    # 按输入顺序分配 id，然后 normalize 成 0..N-1 连续
    cm = io_utils.build_class_map(["person", "truck", "motorcycle"], "")
    assert cm == {"person": 0, "truck": 1, "motorcycle": 2}


def test_build_class_map_preserves_user_map():
    cm = io_utils.build_class_map(["person", "truck"], "person=0,vehicle=1")
    # 用户已有的 person=0 保留；truck 自动追加
    assert cm["person"] == 0
    assert "truck" in cm
    # id 连续
    assert sorted(cm.values()) == list(range(len(cm)))


def test_build_class_map_invalid_user_text_falls_back():
    cm = io_utils.build_class_map(["person"], "garbage")
    assert cm == {"person": 0}


def test_format_class_map_roundtrip():
    cm = {"person": 0, "truck": 2, "motorcycle": 1}
    text = io_utils.format_class_map(cm)
    assert "person=0" in text
    assert "motorcycle=1" in text
    assert "truck=2" in text


def test_scan_labels(tmp_path: Path):
    # 准备一个 LabelMe JSON
    img = tmp_path / "001.json"
    img.write_text(json.dumps({
        "shapes": [
            {"label": "person", "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
             "shape_type": "polygon"},
            {"label": "truck", "points": [[0, 0], [10, 10]],
             "shape_type": "rectangle"},
            {"label": "paste_zone", "points": [[0, 0], [1, 1]],
             "shape_type": "polygon"},
        ],
        "imageData": None, "imagePath": "001.jpg",
    }), encoding="utf-8")
    labels = io_utils.scan_labels(tmp_path)
    assert labels == ["person", "truck"]  # 排序 + 过滤 paste_zone


def test_scan_labels_no_json(tmp_path: Path):
    assert io_utils.scan_labels(tmp_path) == []


def test_save_composite_writes_image_and_label(tmp_path: Path):
    from PIL import Image

    bg = Image.new("RGB", (100, 100), (128, 128, 128))
    # 构造一个 instance：用 Cutout
    from compose_app.models import Cutout, Instance
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    cut = Cutout(label="person", class_id=0, source="t#0", rgba=rgba, thumb=None)
    inst = Instance(cutout_index=0, cx=50.0, cy=50.0, h_ratio=0.30, flip=False, angle=0.0)
    out_dir = tmp_path / "out"

    image_path, label_path, boxes = io_utils.save_composite(
        bg_image=bg,
        bg_path=Path("bg.jpg"),
        instances=[inst],
        cutouts=[cut],
        output_dir=out_dir,
        stem="syn_test_0000001",
        apply_shadow=False,
        apply_color_match=False,
    )
    assert image_path.exists()
    assert label_path.exists()
    assert (out_dir / "generation_log.csv").exists()
    # label 内容是 YOLO 格式一行
    text = label_path.read_text(encoding="utf-8").strip()
    parts = text.split()
    assert parts[0] == "0"
    assert len(parts) == 5


def _make_cutout_with_polygon(label="person", class_id=0):
    """带正方形多边形的 cutout（覆盖整个 rgba）。"""
    import numpy as np
    from PIL import Image
    from compose_app.models import Cutout
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    poly = np.array([[0, 0], [20, 0], [20, 30], [0, 30]], dtype=np.float32)
    return Cutout(label=label, class_id=class_id, source="t#0",
                  rgba=rgba, polygon=poly, thumb=None)


def test_save_composite_seg_format(tmp_path: Path):
    """纯 seg 模式写标准 labels/train，可直接给 Ultralytics。"""
    from PIL import Image
    from compose_app.models import Instance
    bg = Image.new("RGB", (100, 100), (128, 128, 128))
    cut = _make_cutout_with_polygon()
    inst = Instance(cutout_index=0, cx=50.0, cy=50.0, h_ratio=0.30)
    out_dir = tmp_path / "out"

    io_utils.save_composite(
        bg_image=bg, bg_path=Path("bg.jpg"),
        instances=[inst], cutouts=[cut],
        output_dir=out_dir, stem="syn_seg_001",
        output_format="seg",
    )
    seg_file = out_dir / "labels" / "train" / "syn_seg_001.txt"
    assert seg_file.exists()
    parts = seg_file.read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0"
    assert len(parts) == 9  # 1 class + 4 points × 2 coords


def test_save_composite_both_format(tmp_path: Path):
    """both 格式：检测 + 分割都写。"""
    from PIL import Image
    from compose_app.models import Instance
    bg = Image.new("RGB", (100, 100), (128, 128, 128))
    cut = _make_cutout_with_polygon()
    inst = Instance(cutout_index=0, cx=50.0, cy=50.0, h_ratio=0.30)
    out_dir = tmp_path / "out"

    io_utils.save_composite(
        bg_image=bg, bg_path=Path("bg.jpg"),
        instances=[inst], cutouts=[cut],
        output_dir=out_dir, stem="syn_both_001",
        output_format="both",
    )
    assert (out_dir / "labels" / "train" / "syn_both_001.txt").exists()
    assert (out_dir / "labels-seg" / "train" / "syn_both_001.txt").exists()


def test_save_composite_coco_format(tmp_path: Path):
    """coco 格式：写 labels/train + instances_coco.json。"""
    import json
    from PIL import Image
    from compose_app.models import Instance
    bg = Image.new("RGB", (100, 100), (128, 128, 128))
    cut = _make_cutout_with_polygon(label="person", class_id=0)
    inst = Instance(cutout_index=0, cx=50.0, cy=50.0, h_ratio=0.30)
    out_dir = tmp_path / "out"

    io_utils.save_composite(
        bg_image=bg, bg_path=Path("bg.jpg"),
        instances=[inst], cutouts=[cut],
        output_dir=out_dir, stem="syn_coco_001",
        output_format="coco",
    )
    coco_path = out_dir / "instances_coco.json"
    assert coco_path.exists()
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["segmentation"] and len(ann["segmentation"][0]) == 8  # 4 points × 2
    assert ann["bbox"][2] > 0 and ann["bbox"][3] > 0


def test_save_composite_seg_falls_back_to_bbox_without_polygon(tmp_path: Path):
    """cutout 没有 polygon 时，seg 输出退化为 bbox 矩形。"""
    from PIL import Image
    from compose_app.models import Cutout, Instance
    bg = Image.new("RGB", (100, 100), (128, 128, 128))
    rgba = Image.new("RGBA", (20, 30), (255, 0, 0, 255))
    cut = Cutout(label="person", class_id=0, source="t#0", rgba=rgba,
                 polygon=None, thumb=None)
    inst = Instance(cutout_index=0, cx=50.0, cy=50.0, h_ratio=0.30)
    out_dir = tmp_path / "out"

    io_utils.save_composite(
        bg_image=bg, bg_path=Path("bg.jpg"),
        instances=[inst], cutouts=[cut],
        output_dir=out_dir, stem="syn_nopoly_001",
        output_format="seg",
    )
    seg_file = out_dir / "labels" / "train" / "syn_nopoly_001.txt"
    assert seg_file.exists()
    assert len(seg_file.read_text(encoding="utf-8").strip().split()) == 9  # 1 + 8
