"""自动抠图模块测试（remgb 缺失时跳过实际抠图）。"""

from __future__ import annotations

import numpy as np
import pytest

from compose_app import auto_mask


def test_backends_returns_dict():
    bk = auto_mask.backends()
    assert isinstance(bk, dict)
    assert "rembg" in bk
    # sam 固定 False（除非使用者自己注入已加载的 generator）
    assert bk.get("sam") is False


def test_cutout_from_image_raises_without_rembg():
    if auto_mask.backends()["rembg"]:
        pytest.skip("rembg 已安装，跳过 import 错误路径测试")
    with pytest.raises(ImportError):
        auto_mask.cutout_from_image(np.zeros((10, 10, 3), dtype=np.uint8),
                                    backend="rembg")


def test_polygon_from_alpha_square():
    alpha = np.zeros((50, 50), dtype=np.uint8)
    alpha[10:40, 10:40] = 255
    poly = auto_mask._polygon_from_alpha(alpha, simplify_eps=0.01)
    assert poly is not None
    assert len(poly) >= 4
    # 多边形大致覆盖 10..40 范围
    xs = poly[:, 0]
    assert xs.min() >= 9 and xs.max() <= 41


def test_polygon_from_alpha_empty():
    alpha = np.zeros((50, 50), dtype=np.uint8)
    assert auto_mask._polygon_from_alpha(alpha, 0.01) is None


def test_to_bgr_with_ndarray():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    out = auto_mask._to_bgr(arr)
    assert out is not None
    assert out.shape == (4, 4, 3)


def test_to_bgr_with_missing_path(tmp_path):
    out = auto_mask._to_bgr(tmp_path / "nonexistent.jpg")
    assert out is None
