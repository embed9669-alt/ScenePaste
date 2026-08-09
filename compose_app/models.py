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
    # Object appearance preview (RGB-only; does not move labels).
    appearance_enabled: bool = False
    appearance_recipe: str = "mild"
    appearance_seed: int = 0
    appearance_brightness: float = 0.0   # additive; 0 = no slider override
    appearance_contrast: float = 1.0     # 1.0 = no slider override
    appearance_saturation: float = 1.0   # 1.0 = no slider override
    appearance_blur: float = 0.0         # sigma; 0 = no blur override
    # 渲染缓存：LRU。不参与相等比较。
    _cache: "OrderedDict[tuple, Image.Image]" = field(
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
            appearance_enabled=bool(self.appearance_enabled),
            appearance_recipe=str(self.appearance_recipe or "mild"),
            appearance_seed=int(self.appearance_seed),
            appearance_brightness=float(self.appearance_brightness),
            appearance_contrast=float(self.appearance_contrast),
            appearance_saturation=float(self.appearance_saturation),
            appearance_blur=float(self.appearance_blur),
        )

    def appearance_key(self) -> tuple:
        return (
            bool(self.appearance_enabled),
            str(self.appearance_recipe or ""),
            int(self.appearance_seed),
            round(float(self.appearance_brightness), 3),
            round(float(self.appearance_contrast), 3),
            round(float(self.appearance_saturation), 3),
            round(float(self.appearance_blur), 3),
        )

    def render_key(self, target_h: int, class_label: str = "") -> tuple:
        """缓存 key：几何 + 外观参数。"""
        return (target_h, self.flip, round(self.angle, 1), class_label, *self.appearance_key())

    def get_rendered(
        self,
        rgba: Image.Image,
        target_h: int,
        class_label: Optional[str] = None,
    ) -> Image.Image:
        """带 LRU 缓存的渲染：几何变换后再做可选外观增强。"""
        from .rendering import render_instance, apply_instance_appearance
        label = str(class_label or "")
        key = self.render_key(target_h, label)
        cache = self._cache
        img = cache.get(key)
        if img is not None:
            cache.move_to_end(key)
            return img
        img = render_instance(rgba, target_h, self.flip, self.angle)
        img = apply_instance_appearance(img, self, class_label=label or None)
        cache[key] = img
        if len(cache) > RENDER_CACHE_CAPACITY:
            cache.popitem(last=False)
        return img

    def invalidate_cache(self) -> None:
        self._cache.clear()
