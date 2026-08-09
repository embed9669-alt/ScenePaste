"""Deterministic image-only augmentation recipes for rendered scenes.

Recipes intentionally run *after* geometry/annotation generation.  This keeps
boxes, polygons, masks and OBBs aligned while simulating camera/domain effects
such as low light, blur, noise, compression and reduced resolution.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

SCHEMA = "scenepaste/augmentation-recipe"
VERSION = 1

BUILTIN_RECIPES: Dict[str, dict] = {
    "clean": {
        "schema": SCHEMA, "version": VERSION, "name": "clean", "effects": {}
    },
    "camera-mild": {
        "schema": SCHEMA, "version": VERSION, "name": "camera-mild",
        "effects": {
            "brightness_contrast": {"p": 0.45, "brightness": [-0.06, 0.06], "contrast": [0.92, 1.08]},
            "gamma": {"p": 0.20, "range": [0.90, 1.10]},
            "gaussian_noise": {"p": 0.20, "sigma": [1.5, 5.0]},
            "motion_blur": {"p": 0.10, "ksize": [3, 5]},
            "jpeg": {"p": 0.25, "quality": [82, 96]},
            "downscale": {"p": 0.12, "scale": [0.82, 0.97]},
        },
    },
    "surveillance": {
        "schema": SCHEMA, "version": VERSION, "name": "surveillance",
        "effects": {
            "brightness_contrast": {"p": 0.65, "brightness": [-0.12, 0.05], "contrast": [0.85, 1.15]},
            "gamma": {"p": 0.35, "range": [0.78, 1.22]},
            "gaussian_noise": {"p": 0.40, "sigma": [2.0, 9.0]},
            "motion_blur": {"p": 0.25, "ksize": [3, 9]},
            "jpeg": {"p": 0.55, "quality": [55, 90]},
            "downscale": {"p": 0.35, "scale": [0.55, 0.90]},
            "vignette": {"p": 0.18, "strength": [0.05, 0.25]},
        },
    },
    "low-light": {
        "schema": SCHEMA, "version": VERSION, "name": "low-light",
        "effects": {
            "brightness_contrast": {"p": 0.90, "brightness": [-0.28, -0.08], "contrast": [0.82, 1.08]},
            "gamma": {"p": 0.70, "range": [1.05, 1.45]},
            "gaussian_noise": {"p": 0.65, "sigma": [3.0, 12.0]},
            "motion_blur": {"p": 0.18, "ksize": [3, 7]},
            "jpeg": {"p": 0.35, "quality": [65, 92]},
            "vignette": {"p": 0.40, "strength": [0.10, 0.35]},
        },
    },
}


def _clip_probability(value: object, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _float_range(value: object, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lo, hi = float(value[0]), float(value[1])
            return (min(lo, hi), max(lo, hi))
        except (TypeError, ValueError):
            pass
    return default


def _int_range(value: object, default: Tuple[int, int]) -> Tuple[int, int]:
    lo, hi = _float_range(value, (float(default[0]), float(default[1])))
    return int(round(lo)), int(round(hi))


def validate_recipe(payload: Mapping[str, Any]) -> dict:
    data = json.loads(json.dumps(dict(payload)))
    if data.get("schema") != SCHEMA:
        raise ValueError("不是 ScenePaste augmentation recipe")
    version = int(data.get("version", 1))
    if version > VERSION:
        raise ValueError(f"augmentation recipe 版本 {version} 高于当前支持版本 {VERSION}")
    effects = data.get("effects", {})
    if not isinstance(effects, dict):
        raise ValueError("augmentation recipe effects 必须是对象")
    data["version"] = VERSION
    data.setdefault("name", "custom")
    return data


def load_augmentation_recipe(source: Optional[Union[str, Path, Mapping[str, Any]]]) -> Optional[dict]:
    """Load a built-in recipe name, JSON path or mapping.

    ``None`` or an empty string disables post-render scene augmentation.
    """
    if source is None or str(source).strip() == "":
        return None
    if isinstance(source, Mapping):
        return validate_recipe(source)
    text = str(source).strip()
    key = text.lower()
    if key in BUILTIN_RECIPES:
        return validate_recipe(BUILTIN_RECIPES[key])
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"augmentation recipe 不存在：{source}")
    return validate_recipe(json.loads(path.read_text(encoding="utf-8")))


def save_augmentation_recipe(payload: Mapping[str, Any], path: Path) -> Path:
    data = validate_recipe(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _odd_ksize(rng: random.Random, value: object, default=(3, 7)) -> int:
    lo, hi = _int_range(value, default)
    lo, hi = max(1, lo), max(1, hi)
    candidates = [k for k in range(lo, hi + 1) if k % 2 == 1]
    return rng.choice(candidates or [3])


def apply_scene_recipe(image: np.ndarray, recipe: Optional[Mapping[str, Any]], rng: random.Random) -> Tuple[np.ndarray, list]:
    """Apply image-only effects and return ``(image, applied_effect_metadata)``."""
    if recipe is None:
        return image, []
    data = validate_recipe(recipe)
    effects = data.get("effects", {})
    out = image.copy()
    applied = []

    cfg = effects.get("brightness_contrast")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        b_lo, b_hi = _float_range(cfg.get("brightness"), (-0.05, 0.05))
        c_lo, c_hi = _float_range(cfg.get("contrast"), (0.9, 1.1))
        brightness = rng.uniform(b_lo, b_hi)
        contrast = rng.uniform(c_lo, c_hi)
        arr = out.astype(np.float32) * contrast + brightness * 255.0
        out = np.clip(arr, 0, 255).astype(np.uint8)
        applied.append({"effect": "brightness_contrast", "brightness": brightness, "contrast": contrast})

    cfg = effects.get("gamma")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("range"), (0.9, 1.1))
        gamma = max(0.05, rng.uniform(lo, hi))
        lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
        applied.append({"effect": "gamma", "gamma": gamma})

    cfg = effects.get("gaussian_noise")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("sigma"), (1.0, 6.0))
        sigma = max(0.0, rng.uniform(lo, hi))
        # Seed NumPy locally from deterministic Python RNG state.
        np_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise = np_rng.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        applied.append({"effect": "gaussian_noise", "sigma": sigma})

    cfg = effects.get("motion_blur")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        k = _odd_ksize(rng, cfg.get("ksize"), (3, 7))
        angle = rng.uniform(0.0, 180.0)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        matrix = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (k, k))
        total = float(kernel.sum())
        if total > 0:
            kernel /= total
            out = cv2.filter2D(out, -1, kernel)
            applied.append({"effect": "motion_blur", "ksize": k, "angle": angle})

    cfg = effects.get("downscale")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("scale"), (0.7, 0.95))
        scale = min(1.0, max(0.10, rng.uniform(lo, hi)))
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(2, int(round(w * scale))), max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        applied.append({"effect": "downscale", "scale": scale})

    cfg = effects.get("vignette")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("strength"), (0.05, 0.25))
        strength = min(0.95, max(0.0, rng.uniform(lo, hi)))
        h, w = out.shape[:2]
        yy, xx = np.ogrid[-1:1:complex(h), -1:1:complex(w)]
        radius = np.sqrt(xx * xx + yy * yy)
        mask = 1.0 - strength * np.clip(radius, 0.0, 1.0)
        out = np.clip(out.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
        applied.append({"effect": "vignette", "strength": strength})

    cfg = effects.get("jpeg")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _int_range(cfg.get("quality"), (75, 95))
        quality = max(5, min(100, rng.randint(lo, hi)))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded
                applied.append({"effect": "jpeg", "quality": quality})

    return out, applied
