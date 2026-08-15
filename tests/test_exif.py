"""EXIF 自动旋转测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scenepaste.core as core


def _save_jpeg_with_orientation(path: Path, arr: np.ndarray, orientation: int) -> None:
    """用 PIL 写一张带 EXIF Orientation 的 JPG。"""
    from PIL import Image
    img = Image.fromarray(arr[..., ::-1])  # BGR → RGB
    exif = img.getexif()
    exif[0x0112] = orientation  # Orientation tag
    img.save(path, exif=exif, quality=95)


def test_imread_with_exif_no_orientation(tmp_path: Path):
    """无 EXIF orientation 的图应该原样返回。"""
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    arr[5:15, 10:20] = 255  # 一个明显方块
    p = tmp_path / "plain.jpg"
    core.imwrite_unicode(p, arr)
    out = core.imread_with_exif(p)
    assert out is not None
    assert out.shape == arr.shape


def test_imread_with_exif_orientation_6(tmp_path: Path):
    """Orientation=6（顺时针 90°）应该把图旋转回正。"""
    # 原图：宽 40 高 20，左半红色（BGR: 0,0,255）
    arr = np.zeros((20, 40, 3), dtype=np.uint8)
    arr[:, :20] = (0, 0, 255)
    p = tmp_path / "rotated.jpg"
    _save_jpeg_with_orientation(p, arr, orientation=6)
    out = core.imread_with_exif(p)
    assert out is not None
    # 旋转 90° 后：原来是 (h=20, w=40)，新 (h=40, w=20)
    assert out.shape[0] == 40
    assert out.shape[1] == 20


def test_imread_with_exif_missing_file(tmp_path: Path):
    assert core.imread_with_exif(tmp_path / "nope.jpg") is None


def test_imread_with_exif_png_without_exif(tmp_path: Path):
    """PNG 没带 EXIF 时也不应该报错。"""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    p = tmp_path / "x.png"
    core.imwrite_unicode(p, arr)
    out = core.imread_with_exif(p)
    assert out is not None
    assert out.shape == (10, 10, 3)
