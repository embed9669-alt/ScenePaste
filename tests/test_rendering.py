"""渲染层单元测试（不依赖 tkinter）。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from compose_app import rendering


def test_fit_scale_smaller():
    # 受较短一边约束：min(800/2000, 600/1000) = 0.4
    assert rendering.fit_scale(2000, 1000, 800, 600) == 0.4


def test_fit_scale_larger_box():
    # 目标比 box 小：等比放大填满较短边 → min(8, 6) = 6
    assert rendering.fit_scale(100, 100, 800, 600) == 6.0


def test_fit_scale_zero():
    assert rendering.fit_scale(0, 0, 100, 100) == 1.0


def test_render_instance_basic():
    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    out = rendering.render_instance(rgba, 80, False, 0.0)
    assert out.size == (40, 80)


def test_render_instance_target_h_clamped():
    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    out = rendering.render_instance(rgba, 1, False, 0.0)
    # target_h 被 clamp 到 4
    assert out.height == 4


def test_render_instance_flip():
    # 翻转后非对称像素位置交换
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[:, :5, :] = (255, 0, 0, 255)  # 左半红
    rgba = Image.fromarray(arr, "RGBA")
    out = rendering.render_instance(rgba, 10, True, 0.0)
    a = np.asarray(out)
    # 翻转后右半应该是红
    assert a[:, 5:, 0].mean() > 200
    assert a[:, :5, 0].mean() < 50


def test_render_instance_rotation_changes_size():
    rgba = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    out = rendering.render_instance(rgba, 80, False, 45.0)
    # 旋转 45° 后画布扩大
    assert out.size[0] > 40 and out.size[1] > 80


def test_bbox_of_rendered_full_alpha():
    img = Image.new("RGBA", (20, 30), (0, 0, 0, 255))
    x1, y1, x2, y2 = rendering.bbox_of_rendered(img)
    assert (x1, y1, x2, y2) == (0, 0, 20, 30)


def test_bbox_of_rendered_empty():
    img = Image.new("RGBA", (20, 30), (0, 0, 0, 0))
    assert rendering.bbox_of_rendered(img) == (0, 0, 0, 0)


def test_bbox_of_rendered_partial():
    arr = np.zeros((30, 40, 4), dtype=np.uint8)
    arr[10:20, 15:25, :] = (255, 255, 255, 255)
    img = Image.fromarray(arr, "RGBA")
    x1, y1, x2, y2 = rendering.bbox_of_rendered(img)
    assert (x1, y1, x2, y2) == (15, 10, 25, 20)


def test_make_thumbnail_size():
    rgba = Image.new("RGBA", (300, 200), (0, 128, 255, 255))
    thumb = rendering.make_thumbnail(rgba)
    assert thumb.size == (96, 96)
    assert thumb.mode == "RGB"


def test_draw_shadow_modifies_composite():
    composite = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    before = np.asarray(composite).copy()
    rendering.draw_shadow(composite, 100, 150, 50)
    after = np.asarray(composite)
    # 阴影叠加后像素应有变化
    assert not np.array_equal(before, after)


def test_apply_color_match_no_alpha():
    rendered = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    composite = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    out = rendering.apply_color_match(rendered, composite, 50, 50)
    # 全透明 → 原样返回
    assert out is rendered


def test_apply_color_match_shifts_brightness():
    # 纯红目标贴在纯白底上，应该被往白色方向拉
    rendered = Image.new("RGBA", (50, 50), (50, 0, 0, 255))
    composite = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    out = rendering.apply_color_match(rendered, composite, 50, 50, strength=1.0)
    arr = np.asarray(out)
    # R 通道应该被往 255 拉
    assert arr[..., 0].mean() > 100
