"""纯渲染函数：缩放/翻转/旋转/阴影/颜色匹配/bbox/外观预览。

全部为无状态函数，输入 PIL/numpy，输出 PIL/numpy，可独立单元测试。
"""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import scenepaste.core as core

from .models import THUMB_SIZE


def pil_from_asset(asset: core.ObjectAsset) -> Image.Image:
    """把 ObjectAsset（BGR + 0~1 alpha）转成 PIL RGBA。"""
    bgr = asset.image
    alpha = (np.clip(asset.alpha, 0, 1) * 255).astype(np.uint8)
    rgb = bgr[..., ::-1].copy()
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def fit_scale(w: int, h: int, box_w: int, box_h: int) -> float:
    """把 (w,h) 等比适配进 (box_w, box_h)，返回缩放比。

    可大于 1：窗口/全屏变大时预览应跟着放大填满画布。
    """
    if w <= 0 or h <= 0:
        return 1.0
    return min(box_w / w, box_h / h)


def make_thumbnail(rgba: Image.Image) -> Image.Image:
    """把 RGBA 目标缩成统一尺寸的 RGB 缩略图（深灰底）。"""
    thumb = rgba.copy()
    thumb.thumbnail(THUMB_SIZE, Image.LANCZOS)
    bg = Image.new("RGBA", THUMB_SIZE, (40, 40, 40, 255))
    offset = ((THUMB_SIZE[0] - thumb.width) // 2,
              (THUMB_SIZE[1] - thumb.height) // 2)
    bg.alpha_composite(thumb, offset)
    return bg.convert("RGB")


def render_instance(rgba: Image.Image, target_h: int, flip: bool, angle: float) -> Image.Image:
    """缩放 + 翻转 + 旋转，返回带 alpha 的 PIL RGBA。"""
    if target_h < 4:
        target_h = 4
    scale = target_h / float(rgba.height)
    target_w = max(2, int(round(rgba.width * scale)))
    img = rgba.resize((target_w, target_h), Image.LANCZOS)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if abs(angle) > 0.05:
        img = img.rotate(-angle, resample=Image.BILINEAR, expand=True)
    return img


def _slider_overrides(inst: Any) -> Dict[str, dict]:
    """Build forced object-appearance effects from editor sliders."""
    effects: Dict[str, dict] = {}
    brightness = float(getattr(inst, "appearance_brightness", 0.0) or 0.0)
    contrast = float(getattr(inst, "appearance_contrast", 1.0) or 1.0)
    saturation = float(getattr(inst, "appearance_saturation", 1.0) or 1.0)
    blur = float(getattr(inst, "appearance_blur", 0.0) or 0.0)
    if abs(brightness) > 1e-6 or abs(contrast - 1.0) > 1e-6:
        effects["brightness_contrast"] = {
            "p": 1.0,
            "brightness": [brightness, brightness],
            "contrast": [contrast, contrast],
        }
    if abs(saturation - 1.0) > 1e-6:
        effects["saturation"] = {"p": 1.0, "range": [saturation, saturation]}
    if blur > 1e-6:
        effects["gaussian_blur"] = {"p": 1.0, "sigma": [blur, blur]}
    return effects


def build_instance_appearance_recipe(inst: Any) -> Optional[dict]:
    """Merge built-in/custom recipe with slider overrides for one Instance."""
    if not bool(getattr(inst, "appearance_enabled", False)):
        return None
    from scenepaste.core.object_appearance import (
        SCHEMA,
        VERSION,
        load_object_appearance_recipe,
        validate_object_appearance_recipe,
    )

    name = str(getattr(inst, "appearance_recipe", "") or "").strip()
    base: Optional[Mapping[str, Any]] = None
    if name and name.lower() not in {"", "off"}:
        try:
            base = load_object_appearance_recipe(name)
        except (OSError, ValueError, FileNotFoundError):
            base = None
    overrides = _slider_overrides(inst)
    if base is None and not overrides:
        return None
    effects = dict((base or {}).get("effects") or {})
    effects.update(overrides)
    by_class = dict((base or {}).get("by_class") or {})
    return validate_object_appearance_recipe({
        "schema": SCHEMA,
        "version": VERSION,
        "name": str((base or {}).get("name") or name or "editor-preview"),
        "effects": effects,
        "by_class": by_class,
    })


def apply_instance_appearance(
    rgba: Image.Image,
    inst: Any,
    *,
    class_label: Optional[str] = None,
) -> Image.Image:
    """Apply RGB-only object appearance; alpha / size stay identical."""
    recipe = build_instance_appearance_recipe(inst)
    if recipe is None:
        return rgba
    from scenepaste.core.object_appearance import apply_object_appearance

    arr = np.asarray(rgba.convert("RGBA"))
    rgb = arr[..., :3]
    alpha = arr[..., 3].astype(np.float32) / 255.0
    bgr = rgb[..., ::-1].copy()
    seed = int(getattr(inst, "appearance_seed", 0) or 0)
    out_bgr, _meta = apply_object_appearance(
        bgr,
        alpha,
        recipe,
        random.Random(seed),
        class_label=class_label,
        blur_prob_fallback=0.0,
    )
    out = np.dstack([out_bgr[..., ::-1], arr[..., 3]])
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def bbox_of_rendered(rendered: Image.Image) -> Tuple[int, int, int, int]:
    """对已渲染好的 RGBA 图算非零 alpha 的 bbox，避免重复 render。"""
    alpha = np.asarray(rendered.split()[-1])
    ys, xs = np.where(alpha > 32)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def transformed_bbox(rgba: Image.Image, target_h: int, flip: bool, angle: float) -> Tuple[int, int, int, int]:
    """遗留接口：直接 render 再算 bbox。新代码应优先用 bbox_of_rendered 复用缓存。"""
    rendered = render_instance(rgba, target_h, flip, angle)
    return bbox_of_rendered(rendered)


def draw_shadow(composite: Image.Image, cx: int, foot_y: int, foot_width: int) -> None:
    """在 composite 上叠加一个半透明椭圆阴影，模拟目标脚底投影。

    在 alpha_composite 目标之前调用，让阴影被目标遮住一部分。
    """
    if foot_width < 4:
        return
    w = int(foot_width * 0.9)
    h = max(4, int(w * 0.18))
    pad = h
    layer = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse(
        [pad, pad, pad + w, pad + h], fill=(0, 0, 0, 95)
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(2.0, h * 0.5)))
    x = int(cx - layer.width / 2)
    y = int(foot_y - layer.height / 2)
    composite.alpha_composite(layer, (x, y))


def apply_color_match(rendered: Image.Image, composite: Image.Image,
                      x1: int, y1: int, strength: float = 0.30) -> Image.Image:
    """把 rendered 的平均亮度/色调往 composite 对应区域对齐。

    strength=0 表示不变；0.30 是经验上的较自然强度。
    """
    cw, ch = composite.size
    px1 = max(0, x1)
    py1 = max(0, y1)
    px2 = min(cw, x1 + rendered.width)
    py2 = min(ch, y1 + rendered.height)
    if px2 - px1 < 4 or py2 - py1 < 4:
        return rendered
    arr = np.asarray(rendered).astype(np.float32)
    alpha_mask = arr[..., 3] > 32
    if not alpha_mask.any():
        return rendered
    fg_mean = arr[..., :3][alpha_mask].mean(axis=0)
    patch = np.asarray(composite.crop((px1, py1, px2, py2)).convert("RGB")).astype(np.float32)
    bg_mean = patch.reshape(-1, 3).mean(axis=0)
    shift = (bg_mean - fg_mean) * strength
    arr[..., :3] = np.clip(arr[..., :3] + shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")
