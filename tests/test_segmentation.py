"""分割多边形变换与序列化测试。"""

from __future__ import annotations


import numpy as np
import pytest

from compose_app import segmentation as seg


# ---------------------------------------------------------------------------
# transform_local_polygon
# ---------------------------------------------------------------------------

def test_transform_no_flip_no_rotation_preserves_shape():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    result = seg.transform_local_polygon(poly, source_w=10, source_h=10,
                                         target_h=20, flip=False, angle=0.0)
    assert result is not None
    out, (w, h) = result
    assert w == 20 and h == 20  # 等比放大 2 倍
    # 不翻转不旋转：所有点都放大 2 倍
    expected = poly * 2.0
    assert np.allclose(np.sort(out.flatten()), np.sort(expected.flatten()), atol=0.5)


def test_transform_flip_mirrors_x():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    result = seg.transform_local_polygon(poly, source_w=10, source_h=10,
                                         target_h=10, flip=True, angle=0.0)
    out, (w, h) = result
    # 翻转后 x=0 → x=10，x=10 → x=0
    xs = sorted(out[:, 0].tolist())
    assert xs[0] == pytest.approx(0.0, abs=0.5)
    assert xs[-1] == pytest.approx(10.0, abs=0.5)


def test_transform_rotation_expands_size():
    poly = np.array([[0, 0], [40, 0], [40, 80], [0, 80]], dtype=np.float32)
    result = seg.transform_local_polygon(poly, source_w=40, source_h=80,
                                         target_h=80, flip=False, angle=45.0)
    out, (w, h) = result
    # 45° 旋转后画布应变大（原 40x80 → ~85x113）
    assert w > 40 and h > 80


def test_transform_returns_none_for_short_polygon():
    assert seg.transform_local_polygon(np.array([[0, 0], [1, 1]]),
                                       10, 10, 20, False, 0.0) is None
    assert seg.transform_local_polygon(None, 10, 10, 20, False, 0.0) is None


# ---------------------------------------------------------------------------
# clip_polygon_to_canvas
# ---------------------------------------------------------------------------

def test_clip_polygon_fully_inside():
    poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
    out = seg.clip_polygon_to_canvas(poly, w=100, h=100)
    assert out is not None
    assert len(out) == 4


def test_clip_polygon_partially_outside():
    poly = np.array([[-10, 50], [50, 50], [50, 60], [-10, 60]], dtype=np.float32)
    out = seg.clip_polygon_to_canvas(poly, w=100, h=100)
    assert out is not None
    # 裁剪后所有点应在 [0, 100]
    assert out[:, 0].min() >= -1e-6
    assert out[:, 0].max() <= 100 + 1e-6


def test_clip_polygon_completely_outside():
    poly = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32)
    assert seg.clip_polygon_to_canvas(poly, w=100, h=100) is None


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def test_yolo_seg_line_normalizes():
    poly = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32)
    line = seg.yolo_seg_line(class_id=2, polygon=poly, width=100, height=100)
    parts = line.split()
    assert parts[0] == "2"
    coords = [float(x) for x in parts[1:]]
    assert min(coords) == pytest.approx(0.0, abs=1e-6)
    assert max(coords) == pytest.approx(0.5, abs=1e-6)


def test_yolo_seg_line_empty_for_short_polygon():
    poly = np.array([[0, 0], [1, 1]])
    assert seg.yolo_seg_line(0, poly, 100, 100) == ""


def test_coco_polygon_flattens():
    poly = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = seg.coco_polygon(poly)
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_polygon_area_rectangle():
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    assert seg.polygon_area(poly) == pytest.approx(50.0)


def test_polygon_bbox():
    poly = np.array([[3, 7], [10, 2], [5, 9]], dtype=np.float32)
    x1, y1, x2, y2 = seg.polygon_bbox(poly)
    assert (x1, y1, x2, y2) == (3.0, 2.0, 10.0, 9.0)


# ---------------------------------------------------------------------------
# instance_canvas_polygon：端到端几何一致性
# ---------------------------------------------------------------------------

def test_instance_canvas_polygon_no_transform():
    """无翻转、无旋转时，画布多边形 == 平移后的 cutout 多边形。"""
    from PIL import Image
    from compose_app.models import Cutout, Instance

    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    # cutout 的局部多边形（覆盖整个图）
    poly = np.array([[0, 0], [40, 0], [40, 80], [0, 80]], dtype=np.float32)
    cut = Cutout(label="person", class_id=0, source="t#0", rgba=rgba, polygon=poly)
    inst = Instance(cutout_index=0, cx=100.0, cy=100.0, h_ratio=0.5,
                    flip=False, angle=0.0)
    # bg_h=160 → target_h=80（与 cutout 同高），渲染后图像 40x80
    # cx=100,cy=100 → 左上角 (80, 60)
    out = seg.instance_canvas_polygon(cut, inst, bg_w=200, bg_h=160)
    assert out is not None
    xs = sorted(out[:, 0].tolist())
    ys = sorted(out[:, 1].tolist())
    # 多边形应在 (80..120, 60..140)
    assert xs[0] == pytest.approx(80, abs=1)
    assert xs[-1] == pytest.approx(120, abs=1)
    assert ys[0] == pytest.approx(60, abs=1)
    assert ys[-1] == pytest.approx(140, abs=1)


def test_instance_canvas_polygon_returns_none_when_no_polygon():
    from PIL import Image
    from compose_app.models import Cutout, Instance

    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    cut = Cutout(label="person", class_id=0, source="t#0", rgba=rgba, polygon=None)
    inst = Instance(cutout_index=0, cx=100.0, cy=100.0, h_ratio=0.5)
    assert seg.instance_canvas_polygon(cut, inst, 200, 160) is None
