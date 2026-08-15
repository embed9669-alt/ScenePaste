"""CLI segmentation 几何精度测试（捕获 _canvas_polygon 的 scale 反推 bug）。

原 bug：_canvas_polygon 用 alpha 非零区域 bbox 高度反推 scale，
当 cutout 的 alpha 不填满整个 crop 时会系统性偏移。
本测试构造一个 alpha 只占中央一半的 cutout，验证 transform 路径给出的多边形
与真实变换一致，bbox-fallback 路径会偏。
"""

from __future__ import annotations

import numpy as np
import pytest

import scenepaste.core as core


def _make_asset_with_padding(tmp_path):
    """构造一个 cutout：crop 100x100，alpha 只在中央 50..80（高度方向）有内容。"""
    # 100x100 BGR + alpha
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)  # BGR
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[50:80, 20:60] = 1.0  # 30 高 × 40 宽的可见区域
    # polygon 也在 alpha 区域内
    polygon = np.array([[20, 50], [60, 50], [60, 80], [20, 80]], dtype=np.float32)
    asset = core.ObjectAsset(
        label="x", class_id=0, image=image, alpha=alpha,
        source_json=tmp_path / "fake.json", source_shape_index=0,
        polygon=polygon,
    )
    return asset


def test_canvas_polygon_with_transform_is_correct(tmp_path):
    """transform 路径：scale=0.5 → 多边形应在 [10..30]×[25..40]。"""
    asset = _make_asset_with_padding(tmp_path)
    # scale=0.5：原图 100→50；放在 (100, 100)
    transform = (0.5, 100, 100, False)
    # box 是 alpha 非零 bbox 在画布坐标
    box = (100 + int(20 * 0.5), 100 + int(50 * 0.5),    # x1,y1
           100 + int(60 * 0.5), 100 + int(80 * 0.5))    # x2,y2
    poly = core._canvas_polygon(asset, box, flipped=False,
                                 canvas_w=500, canvas_h=500,
                                 transform=transform)
    assert poly is not None
    # 期望：polygon (20,50)→(110,125), (60,80)→(130,140)
    assert poly[:, 0].min() == pytest.approx(110, abs=1)
    assert poly[:, 0].max() == pytest.approx(130, abs=1)
    assert poly[:, 1].min() == pytest.approx(125, abs=1)
    assert poly[:, 1].max() == pytest.approx(140, abs=1)


def test_canvas_polygon_bbox_fallback_differs_from_transform(tmp_path):
    """没有 transform 时用 bbox 反推，会有偏移（捕获原 bug）。"""
    asset = _make_asset_with_padding(tmp_path)
    box = (110, 125, 130, 140)  # 同上的 alpha bbox
    # 不传 transform → 走 bbox fallback
    poly_bbox = core._canvas_polygon(asset, box, flipped=False,
                                      canvas_w=500, canvas_h=500,
                                      transform=None)
    poly_t = core._canvas_polygon(asset, box, flipped=False,
                                   canvas_w=500, canvas_h=500,
                                   transform=(0.5, 100, 100, False))
    # 两者应该不一样（fallback 算错）；fallback 把 scale 算成 box_h/src_h=15/100=0.15
    assert not np.allclose(poly_bbox, poly_t, atol=0.5)


def test_canvas_polygon_flip_mirrors_x(tmp_path):
    asset = _make_asset_with_padding(tmp_path)
    transform = (1.0, 0, 0, True)
    box = (20, 50, 60, 80)
    poly = core._canvas_polygon(asset, box, flipped=True,
                                 canvas_w=200, canvas_h=200,
                                 transform=transform)
    # 原 x: 20..60；镜像后 100-60=40 .. 100-20=80
    assert poly[:, 0].min() == pytest.approx(40, abs=1)
    assert poly[:, 0].max() == pytest.approx(80, abs=1)
