"""数据模型与常量。

这些类型不依赖 tkinter，方便单元测试。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 左侧缩略图尺寸（像素）
THUMB_SIZE = (96, 96)

# 每个实例保留多少份渲染缓存（按 (h, flip, angle) 组合）
# 防止反复旋转/缩放时无限制吃内存
RENDER_CACHE_CAPACITY = 8


@dataclass
class Cutout:
    """抠出的一个透明目标（PIL RGBA + 显示用缩略图 + 可选分割多边形）。"""
    label: str
    class_id: int
    source: str                       # 形如 "20260717-011459.json#3"
    rgba: Image.Image
    # 局部坐标多边形（Nx2，float32），与 rgba 同坐标系；为 None 表示无分割信息
    polygon: Optional[np.ndarray] = None
    # thumb 是 ImageTk.PhotoImage，需要主线程绑定，由调用方注入
    thumb: object = field(default=None, repr=False, compare=False)


@dataclass
class Instance:
    """画布上的一个放置实例。坐标使用背景原生像素。"""
    cutout_index: int
    cx: float            # 中心 x（背景像素）
    cy: float            # 中心 y（背景像素）
    h_ratio: float       # 目标高度 / 背景高度
    flip: bool = False
    angle: float = 0.0   # 顺时针度数
    uid: int = 0
    # 渲染缓存：LRU，按 (target_h, flip, angle) 键控。不参与相等比较。
    _cache: "OrderedDict[Tuple[int, bool, float], Image.Image]" = field(
        default_factory=OrderedDict, repr=False, compare=False)

    def clone(self) -> "Instance":
        """克隆（用于复制 / 撤销快照），不继承缓存。uid 默认保留，
        复制粘贴场景需要新 uid 时由调用方显式赋值。"""
        return Instance(
            cutout_index=self.cutout_index,
            cx=self.cx,
            cy=self.cy,
            h_ratio=self.h_ratio,
            flip=self.flip,
            angle=self.angle,
            uid=self.uid,
        )

    def render_key(self, target_h: int) -> Tuple[int, bool, float]:
        """缓存 key：仅依赖目标高度、翻转、旋转角度（精度 0.1°）。"""
        return (target_h, self.flip, round(self.angle, 1))

    def get_rendered(self, rgba: Image.Image, target_h: int) -> Image.Image:
        """带 LRU 缓存的渲染：当 (h, flip, angle) 没变时直接复用。"""
        from .rendering import render_instance
        key = self.render_key(target_h)
        cache = self._cache
        img = cache.get(key)
        if img is not None:
            cache.move_to_end(key)
            return img
        img = render_instance(rgba, target_h, self.flip, self.angle)
        cache[key] = img
        if len(cache) > RENDER_CACHE_CAPACITY:
            cache.popitem(last=False)
        return img

    def invalidate_cache(self) -> None:
        self._cache.clear()
