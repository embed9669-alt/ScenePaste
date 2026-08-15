"""Localized user-facing messages.

The catalog is keyed by a stable message id; each entry has a Chinese and
English string. The active locale is read from ``SCENEPASTE_LANG`` env var
(defaulting to ``"zh"``). Core modules call :func:`t` instead of inlining
strings so that downstream apps can localize without monkeypatching.

To extend: add a (key, zh, en) row to :data:`MESSAGES` and replace the
inline string at its call site with ``t("key")``.
"""

from __future__ import annotations

import os
from typing import Dict

# Default locale is Chinese to preserve historical behaviour.
DEFAULT_LOCALE = "zh"


def _detect_locale() -> str:
    """Pick the active locale from ``SCENEPASTE_LANG`` (``zh`` / ``en``)."""
    raw = (os.environ.get("SCENEPASTE_LANG") or DEFAULT_LOCALE).strip().lower()
    return "en" if raw.startswith("en") else "zh"


# Each entry: id -> {"zh": ..., "en": ...}. Keep keys stable forever — they
# are part of the public contract for downstream tools that grep logs.
_MESSAGES: Dict[str, Dict[str, str]] = {
    # ---- class-map parsing ------------------------------------------------
    "class_map.bad_format": {
        "zh": "类别映射格式错误：{item}，应写成 person=0",
        "en": "Invalid class-map entry: {item} (expected 'name=id')",
    },
    "class_map.empty_label": {
        "zh": "类别名称不能为空",
        "en": "Class name cannot be empty",
    },
    "class_map.need_one": {
        "zh": "至少需要一个类别映射，例如 person=0,vehicle=1",
        "en": "At least one class mapping is required, e.g. person=0,vehicle=1",
    },
    "class_map.dup_or_negative": {
        "zh": "类别编号必须是互不重复的非负整数",
        "en": "Class ids must be unique non-negative integers",
    },
    "class_map.not_contiguous": {
        "zh": "类别编号需要从 0 开始连续，例如 person=0,vehicle=1",
        "en": "Class ids must be contiguous starting from 0, e.g. person=0,vehicle=1",
    },
    # ---- config validation ------------------------------------------------
    "config.objects_dir_missing": {
        "zh": "目标素材目录不存在：{path}",
        "en": "Objects directory does not exist: {path}",
    },
    "config.backgrounds_dir_missing": {
        "zh": "背景图目录不存在：{path}",
        "en": "Backgrounds directory does not exist: {path}",
    },
    "config.count_positive": {
        "zh": "生成数量必须大于 0",
        "en": "count must be greater than 0",
    },
    "config.object_range": {
        "zh": "每图目标数量范围不正确",
        "en": "min/max object range is invalid",
    },
    "config.y_range": {
        "zh": "地面纵向范围必须满足 0 <= y_min < y_max <= 1",
        "en": "Ground band must satisfy 0 <= y_min < y_max <= 1",
    },
    "config.height_range": {
        "zh": "目标高度比例必须满足 0 < 远处 <= 近处 < 1",
        "en": "Height ratio must satisfy 0 < far_height <= near_height < 1",
    },
    "config.iou_range": {
        "zh": "最大目标重叠率必须在 0～1 之间",
        "en": "max_iou must be in [0, 1]",
    },
    "config.preview_ratio_range": {
        "zh": "preview_ratio 必须在 0～1 之间",
        "en": "preview_ratio must be in [0, 1]",
    },
    "config.cache_negative": {
        "zh": "background_cache_size 不能小于 0",
        "en": "background_cache_size cannot be negative",
    },
    "config.checkpoint_negative": {
        "zh": "coco_checkpoint_interval 不能小于 0",
        "en": "coco_checkpoint_interval cannot be negative",
    },
    "config.asset_sampling_invalid": {
        "zh": "asset_sampling 只能是 balanced 或 random",
        "en": "asset_sampling must be 'balanced' or 'random'",
    },
    "config.background_sampling_invalid": {
        "zh": "background_sampling 只能是 balanced 或 random",
        "en": "background_sampling must be 'balanced' or 'random'",
    },
    # ---- asset / background loading --------------------------------------
    "assets.no_json": {
        "zh": "目标素材目录中没有找到 JSON：{path}",
        "en": "No LabelMe JSON found under objects dir: {path}",
    },
    "assets.none_extracted": {
        "zh": "没有提取到可用目标，请检查 JSON 标签与类别映射是否一致",
        "en": "No usable assets extracted — check that JSON labels match the class map",
    },
    "assets.labelme_image_not_found": {
        "zh": "找不到 {name} 对应的原图",
        "en": "Cannot resolve source image for {name}",
    },
    "assets.image_dir_empty": {
        "zh": "目标素材目录里没有图片：{path}",
        "en": "No images found in objects dir: {path}",
    },
    "assets.auto_failed": {
        "zh": "自动抠图失败：{path} 里没有可用目标",
        "en": "Auto-cutout produced no usable assets in: {path}",
    },
    "assets.rembg_missing": {
        "zh": "自动抠图需要 rembg：pip install 'scenepaste[auto]'",
        "en": "Auto-cutout requires rembg: pip install 'scenepaste[auto]'",
    },
    "backgrounds.empty": {
        "zh": "背景图目录中没有图片：{path}",
        "en": "No images found in backgrounds dir: {path}",
    },
    # ---- CLI --------------------------------------------------------------
    "cli.missing_args": {
        "zh": "命令行模式缺少参数：{names}",
        "en": "Missing required CLI arguments: {names}",
    },
    "cli.error": {
        "zh": "错误：{msg}",
        "en": "Error: {msg}",
    },
}


def t(key: str, **fmt) -> str:
    """Look up a localized message and apply ``str.format`` substitutions.

    Falls back to the key itself when unknown, so a missing entry is loud
    rather than silently empty.
    """
    entry = _MESSAGES.get(key)
    if entry is None:
        return key
    text = entry.get(_detect_locale()) or entry[DEFAULT_LOCALE]
    return text.format(**fmt) if fmt else text


# ---- Exception hierarchy ---------------------------------------------------


class ScenePasteError(Exception):
    """Base class for all ScenePaste errors."""


class ConfigError(ScenePasteError, ValueError):
    """Raised when a GenerationConfig field is invalid."""


class AssetError(ScenePasteError, RuntimeError):
    """Raised when no usable object assets can be extracted."""
