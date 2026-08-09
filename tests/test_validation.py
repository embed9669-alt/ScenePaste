"""bbox 校验：clip / visible_ratio / is_valid_yolo_box。"""

from __future__ import annotations


import scenepaste.core as core


def test_clip_bbox_inside():
    assert core.clip_bbox((10, 10, 50, 50), 100, 100) == (10, 10, 50, 50)


def test_clip_bbox_negative_origin():
    # 左上角越界 → 拉回 (0, 0)
    assert core.clip_bbox((-20, -10, 50, 60), 100, 100) == (0, 0, 50, 60)


def test_clip_bbox_exceeds_canvas():
    # 右下角越界 → 拉到 (width, height)
    assert core.clip_bbox((80, 80, 200, 250), 100, 100) == (80, 80, 100, 100)


def test_clip_bbox_fully_outside():
    # 完全在画面外 → 被压成边界上的零面积框
    assert core.clip_bbox((200, 200, 300, 400), 100, 100) == (100, 100, 100, 100)


def test_visible_ratio_full():
    box = (10, 10, 50, 50)
    assert core.visible_ratio(box, box) == 1.0


def test_visible_ratio_half():
    # 原框 40x40=1600；裁剪后 20x40=800 → 0.5
    assert core.visible_ratio((10, 10, 50, 50), (30, 10, 50, 50)) == 0.5


def test_visible_ratio_zero_when_original_empty():
    assert core.visible_ratio((10, 10, 10, 10), (10, 10, 10, 10)) == 0.0


def test_is_valid_yolo_box_passes():
    ok, clipped = core.is_valid_yolo_box((10, 10, 50, 50), 100, 100)
    assert ok is True
    assert clipped == (10, 10, 50, 50)


def test_is_valid_yolo_box_rejects_mostly_offscreen():
    # 原框 100x100，只有 5x100 露在画面内 = 5% < 10% 阈值 → 拒绝
    ok, clipped = core.is_valid_yolo_box((-95, 0, 5, 100), 100, 100,
                                          min_visible=0.10)
    assert ok is False


def test_is_valid_yolo_box_rejects_tiny():
    ok, _ = core.is_valid_yolo_box((10, 10, 11, 20), 100, 100)
    # 宽度只有 1 < min_size(4) → 拒绝
    assert ok is False


def test_is_valid_yolo_box_clips_then_passes():
    # 越界但可见部分足够大 → 通过，bbox 已裁剪
    ok, clipped = core.is_valid_yolo_box((-20, 0, 50, 50), 100, 100)
    assert ok is True
    assert clipped == (0, 0, 50, 50)


def test_yolo_line_outputs_in_unit_range_after_clip():
    # 越界 box 经 clip 后输出的归一化坐标必须在 [0, 1]
    line = core.yolo_line(0, core.clip_bbox((-20, -10, 200, 250), 100, 100), 100, 100)
    parts = [float(x) for x in line.split()]
    assert parts[0] == 0.0
    for v in parts[1:]:
        assert 0.0 <= v <= 1.0
