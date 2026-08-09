"""CocoWriter 测试：避免 O(N²) 全量重写。"""

from __future__ import annotations

import json
from pathlib import Path


import scenepaste.core as core


def test_coco_writer_creates_new(tmp_path: Path):
    p = tmp_path / "coco.json"
    w = core.CocoWriter(p)
    img_id = w.add_image("a.jpg", 100, 200, background="bg.jpg")
    w.add_annotation(img_id, 0, [[0, 0, 10, 0, 10, 10, 0, 10]],
                     [0, 0, 10, 10], 100.0)
    w.finalize()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    assert data["images"][0]["width"] == 100


def test_coco_writer_appends_to_existing(tmp_path: Path):
    p = tmp_path / "coco.json"
    # 第一次 run
    w1 = core.CocoWriter(p)
    img1 = w1.add_image("a.jpg", 100, 200)
    w1.add_annotation(img1, 0, [[0, 0, 1, 0, 1, 1, 0, 1]], [0, 0, 1, 1], 1.0)
    w1.finalize()
    # 第二次 run：重开 writer，应该读到已有内容并追加
    w2 = core.CocoWriter(p)
    assert len(w2.images) == 1
    assert len(w2.annotations) == 1
    img2 = w2.add_image("b.jpg", 50, 60)
    w2.add_annotation(img2, 1, [[0, 0, 1, 0, 1, 1, 0, 1]], [0, 0, 1, 1], 1.0)
    w2.finalize()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    # image id 必须唯一且递增
    ids = [i["id"] for i in data["images"]]
    assert ids == [1, 2]


def test_coco_writer_ensure_category(tmp_path: Path):
    p = tmp_path / "coco.json"
    w = core.CocoWriter(p, categories=[{"id": 0, "name": "person", "supercategory": "object"}])
    cid = w.ensure_category("truck", 1)
    assert cid == 1
    # 已存在的不应重复添加
    cid2 = w.ensure_category("person", 0)
    assert cid2 == 0
    assert len(w.categories) == 2
    w.finalize()


def test_coco_writer_flush_is_idempotent(tmp_path: Path):
    """多次 flush 不应该破坏数据。"""
    p = tmp_path / "coco.json"
    w = core.CocoWriter(p)
    img = w.add_image("a.jpg", 10, 10)
    w.add_annotation(img, 0, [[0, 0, 1, 0, 1, 1, 0, 1]], [0, 0, 1, 1], 1.0)
    w.flush()
    w.flush()
    w.finalize()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["annotations"]) == 1


def test_append_coco_with_writer_param(tmp_path: Path):
    """_append_coco 接受 writer 参数，复用避免全量重写。"""
    from scenepaste.core import _append_coco
    # 用最小 asset 集合模拟
    import numpy as np
    asset = core.ObjectAsset(
        label="x", class_id=0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        alpha=np.ones((10, 10), dtype=np.float32),
        source_json=tmp_path / "fake.json",
        source_shape_index=0,
        polygon=None,
    )
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"")
    annotations = [(asset, (0, 0, 5, 5), False, (1.0, 0, 0, False))]
    writer = core.CocoWriter(tmp_path / "instances_coco.json")
    returned = _append_coco(tmp_path, tmp_path / "bg.jpg", img_path,
                             "stem1", 100, 100, annotations, writer=writer)
    assert returned is writer  # 同一实例复用
    assert len(writer.images) == 1
    _append_coco(tmp_path, tmp_path / "bg.jpg", img_path,
                 "stem2", 100, 100, annotations, writer=returned)
    assert len(returned.images) == 2  # 累积，未重置
