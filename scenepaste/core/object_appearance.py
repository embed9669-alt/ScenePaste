"""Per-object appearance recipes applied to cutouts before color-match/blend.

Scene recipes (:mod:`scenepaste.core.recipes`) degrade the *finished* image.
Object appearance recipes diversify each pasted instance without changing
geometry, so Detect / Seg / OBB / Semantic / COCO labels stay aligned.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

SCHEMA = "scenepaste/object-appearance-recipe"
VERSION = 1

# ``legacy`` reproduces the v1.0 hard-coded HSV + optional blur ranges so
# existing runs remain comparable when users explicitly choose it.
BUILTIN_OBJECT_RECIPES: Dict[str, dict] = {
    "off": {
        "schema": SCHEMA, "version": VERSION, "name": "off", "effects": {},
    },
    "legacy": {
        "schema": SCHEMA, "version": VERSION, "name": "legacy",
        "effects": {
            "hsv_jitter": {
                "p": 1.0,
                # Historical OpenCV hue units (1 unit = 2 degrees).
                "hue": [-2.0, 2.0],
                "saturation": [0.90, 1.10],
                "value": [0.88, 1.12],
            },
            "gaussian_blur": {"p": None, "sigma": [0.25, 0.75]},
        },
    },
    "mild": {
        "schema": SCHEMA, "version": VERSION, "name": "mild",
        "effects": {
            "brightness_contrast": {
                "p": 0.85, "brightness": [-0.12, 0.12], "contrast": [0.88, 1.12],
            },
            "saturation": {"p": 0.70, "range": [0.82, 1.18]},
            # ``hue`` uses degrees; conversion to OpenCV HSV happens internally.
            "hue": {"p": 0.45, "range": [-5.0, 5.0]},
            "gamma": {"p": 0.35, "range": [0.88, 1.18]},
            "color_temperature": {"p": 0.35, "strength": [-12.0, 12.0]},
            "gaussian_blur": {"p": 0.25, "sigma": [0.20, 1.00]},
            "gaussian_noise": {"p": 0.20, "sigma": [1.0, 5.0]},
            "resolution_degrade": {"p": 0.35, "scale": [0.40, 0.90]},
            "sharpness": {"p": 0.18, "amount": [0.85, 1.18]},
        },
    },
    "surveillance-object": {
        "schema": SCHEMA, "version": VERSION, "name": "surveillance-object",
        "effects": {
            "brightness_contrast": {
                "p": 0.90, "brightness": [-0.18, 0.08], "contrast": [0.82, 1.15],
            },
            "saturation": {"p": 0.75, "range": [0.75, 1.20]},
            "hue": {"p": 0.40, "range": [-6.0, 6.0]},
            "gamma": {"p": 0.50, "range": [0.82, 1.25]},
            "color_temperature": {"p": 0.45, "strength": [-18.0, 18.0]},
            "gaussian_blur": {"p": 0.40, "sigma": [0.30, 1.40]},
            "motion_blur": {"p": 0.18, "ksize": [3, 7]},
            "gaussian_noise": {"p": 0.35, "sigma": [1.5, 8.0]},
            "resolution_degrade": {"p": 0.50, "scale": [0.30, 0.85]},
            "jpeg": {"p": 0.35, "quality": [55, 92]},
            "sharpness": {"p": 0.12, "amount": [0.75, 1.12]},
        },
    },
}

_EFFECT_KEYS = {
    "brightness_contrast": {"p", "brightness", "contrast"},
    "hsv_jitter": {"p", "hue", "saturation", "value"},
    "saturation": {"p", "range"},
    "hue": {"p", "range"},
    "gamma": {"p", "range"},
    "color_temperature": {"p", "strength", "kelvin_shift"},  # legacy alias accepted
    "gaussian_blur": {"p", "sigma"},
    "motion_blur": {"p", "ksize"},
    "gaussian_noise": {"p", "sigma"},
    "resolution_degrade": {"p", "scale"},
    "jpeg": {"p", "quality"},
    "sharpness": {"p", "amount"},
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


def _odd_ksize(rng: random.Random, value: object, default=(3, 7)) -> int:
    lo, hi = _int_range(value, default)
    lo, hi = max(1, lo), max(1, hi)
    candidates = [k for k in range(lo, hi + 1) if k % 2 == 1]
    return rng.choice(candidates or [3])


def _ensure_range(name: str, value: object, *, minimum: float, maximum: float) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是 [min, max]")
    try:
        values = [float(value[0]), float(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含数字") from exc
    if min(values) < minimum or max(values) > maximum:
        raise ValueError(f"{name} 必须在 {minimum}..{maximum} 范围内")


def _validate_effect(name: str, cfg: object, path: str) -> None:
    if name not in _EFFECT_KEYS:
        raise ValueError(f"未知 object appearance effect：{path}.{name}")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}.{name} 必须是对象")
    unknown = set(cfg) - _EFFECT_KEYS[name]
    if unknown:
        raise ValueError(f"{path}.{name} 包含未知字段：{', '.join(sorted(unknown))}")
    p = cfg.get("p", 0.0)
    if p is None:
        if name != "gaussian_blur":
            raise ValueError(f"{path}.{name}.p 不能为 null")
    else:
        try:
            pf = float(p)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.{name}.p 必须是 0..1 数字") from exc
        if not 0.0 <= pf <= 1.0:
            raise ValueError(f"{path}.{name}.p 必须在 0..1")

    specs = {
        "brightness_contrast": (("brightness", -1.0, 1.0), ("contrast", 0.05, 4.0)),
        "hsv_jitter": (("hue", -90.0, 90.0), ("saturation", 0.0, 4.0), ("value", 0.0, 4.0)),
        "saturation": (("range", 0.0, 4.0),),
        "hue": (("range", -180.0, 180.0),),
        "gamma": (("range", 0.05, 5.0),),
        "color_temperature": (),
        "gaussian_blur": (("sigma", 0.01, 20.0),),
        "motion_blur": (("ksize", 1.0, 31.0),),
        "gaussian_noise": (("sigma", 0.0, 100.0),),
        "resolution_degrade": (("scale", 0.05, 1.0),),
        "jpeg": (("quality", 5.0, 100.0),),
        "sharpness": (("amount", 0.0, 3.0),),
    }
    for key, lo, hi in specs[name]:
        if key in cfg:
            _ensure_range(f"{path}.{name}.{key}", cfg[key], minimum=lo, maximum=hi)
    if name == "color_temperature":
        value = cfg.get("strength", cfg.get("kelvin_shift", [-10.0, 10.0]))
        _ensure_range(f"{path}.{name}.strength", value, minimum=-100.0, maximum=100.0)


def validate_object_appearance_recipe(payload: Mapping[str, Any]) -> dict:
    """Validate and normalize a recipe.

    Unlike the early v1.0.1 implementation, unknown effect names and misspelled
    fields are rejected instead of being silently ignored. This makes long-run
    recipes auditable and prevents a typo from quietly disabling augmentation.
    """
    # Shallow-copy the top level. Validation never mutates nested effect configs,
    # avoiding an expensive JSON round-trip for every generated instance.
    data = dict(payload)
    if data.get("schema") != SCHEMA:
        raise ValueError("不是 ScenePaste object appearance recipe")
    version = int(data.get("version", 1))
    if version > VERSION:
        raise ValueError(f"object appearance recipe 版本 {version} 高于当前支持版本 {VERSION}")
    effects = data.get("effects", {})
    if not isinstance(effects, dict):
        raise ValueError("object appearance recipe effects 必须是对象")
    for effect_name, cfg in effects.items():
        _validate_effect(str(effect_name), cfg, "effects")

    by_class = data.get("by_class", {})
    if by_class is None:
        by_class = {}
    if not isinstance(by_class, dict):
        raise ValueError("object appearance recipe by_class 必须是对象")
    for label, override in by_class.items():
        if not isinstance(override, dict):
            raise ValueError(f"by_class[{label!r}] 必须是对象")
        nested = override.get("effects", override)
        if not isinstance(nested, dict):
            raise ValueError(f"by_class[{label!r}].effects 必须是对象")
        for effect_name, cfg in nested.items():
            _validate_effect(str(effect_name), cfg, f"by_class[{label!r}].effects")

    data["version"] = VERSION
    data.setdefault("name", "custom")
    data["effects"] = effects
    data["by_class"] = by_class
    return data


def load_object_appearance_recipe(
    source: Optional[Union[str, Path, Mapping[str, Any]]],
) -> Optional[dict]:
    """Load a built-in name, JSON path or already-loaded mapping."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return validate_object_appearance_recipe(source)
    text = str(source).strip()
    if not text:
        return None
    key = text.lower()
    if key in BUILTIN_OBJECT_RECIPES:
        return validate_object_appearance_recipe(BUILTIN_OBJECT_RECIPES[key])
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"object appearance recipe 不存在：{source}")
    return validate_object_appearance_recipe(json.loads(path.read_text(encoding="utf-8")))


def save_object_appearance_recipe(payload: Mapping[str, Any], path: Path) -> Path:
    data = validate_object_appearance_recipe(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_object_effects(
    recipe: Optional[Mapping[str, Any]],
    class_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge base effects with optional per-class overrides."""
    if recipe is None:
        return {}
    # Callers normally pass a validated/preloaded recipe. Keep validation here
    # for public API safety; worker code resolves once at startup.
    data = validate_object_appearance_recipe(recipe)
    effects = dict(data.get("effects") or {})
    if not class_label:
        return effects
    by_class = data.get("by_class") or {}
    override = by_class.get(class_label)
    if override is None:
        folded = class_label.casefold()
        for key, value in by_class.items():
            if str(key).casefold() == folded:
                override = value
                break
    if not isinstance(override, dict):
        return effects
    nested = override.get("effects", override)
    if isinstance(nested, dict):
        effects.update(nested)
    return effects


def _alpha3(alpha: np.ndarray) -> np.ndarray:
    return np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]


def _apply_masked(image: np.ndarray, alpha: np.ndarray, modified: np.ndarray) -> np.ndarray:
    """Apply an RGB-only effect smoothly across antialiased alpha edges."""
    weight = _alpha3(alpha)
    out = image.astype(np.float32) * (1.0 - weight) + modified.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_aware_filter(
    image: np.ndarray,
    alpha: np.ndarray,
    filter_image,
    filter_alpha,
) -> np.ndarray:
    """Filter RGB without pulling transparent/outside-source colors into edges."""
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    premul = image.astype(np.float32) * a[..., None]
    num = filter_image(premul)
    den = np.clip(filter_alpha(a), 1e-4, None)
    return np.clip(num / den[..., None], 0, 255).astype(np.uint8)


def _safe_jpeg_source(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fill transparent pixels with object mean before JPEG to avoid edge bleed."""
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    total = float(a.sum())
    if total <= 1e-6:
        return image
    mean = (image.astype(np.float32) * a[..., None]).sum(axis=(0, 1)) / total
    filled = image.astype(np.float32) * a[..., None] + mean.reshape(1, 1, 3) * (1.0 - a[..., None])
    return np.clip(filled, 0, 255).astype(np.uint8)


def apply_object_appearance(
    image: np.ndarray,
    alpha: np.ndarray,
    recipe: Optional[Mapping[str, Any]],
    rng: random.Random,
    *,
    class_label: Optional[str] = None,
    blur_prob_fallback: Optional[float] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Apply per-object RGB appearance effects; geometry and alpha are unchanged.

    ``gaussian_blur.p = null`` means ``blur_prob_fallback`` and exists only for
    the ``legacy`` built-in. ``hue.range`` uses **degrees**; the historical
    ``hsv_jitter.hue`` field remains in OpenCV hue units for v1.0 parity.
    """
    effects = resolve_object_effects(recipe, class_label)
    if not effects:
        return image, []

    out = image.copy()
    applied: List[dict] = []
    alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if not np.any(alpha_f > 0.10):
        return image, []

    cfg = effects.get("brightness_contrast")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        b_lo, b_hi = _float_range(cfg.get("brightness"), (-0.08, 0.08))
        c_lo, c_hi = _float_range(cfg.get("contrast"), (0.9, 1.1))
        brightness = rng.uniform(b_lo, b_hi)
        contrast = rng.uniform(c_lo, c_hi)
        modified = np.clip(out.astype(np.float32) * contrast + brightness * 255.0, 0, 255)
        out = _apply_masked(out, alpha_f, modified)
        applied.append({"effect": "brightness_contrast", "brightness": brightness, "contrast": contrast})

    cfg = effects.get("hsv_jitter")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p"), 1.0):
        h_lo, h_hi = _float_range(cfg.get("hue"), (-2.0, 2.0))
        s_lo, s_hi = _float_range(cfg.get("saturation"), (0.9, 1.1))
        v_lo, v_hi = _float_range(cfg.get("value"), (0.88, 1.12))
        hue_cv = rng.uniform(h_lo, h_hi)
        sat = rng.uniform(s_lo, s_hi)
        val = rng.uniform(v_lo, v_hi)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_cv, 180.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = _apply_masked(out, alpha_f, modified)
        applied.append({"effect": "hsv_jitter", "hue_opencv": hue_cv, "saturation": sat, "value": val})

    cfg = effects.get("saturation")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("range"), (0.85, 1.15))
        factor = rng.uniform(lo, hi)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = _apply_masked(out, alpha_f, modified)
        applied.append({"effect": "saturation", "factor": factor})

    cfg = effects.get("hue")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("range"), (-5.0, 5.0))
        shift_degrees = rng.uniform(lo, hi)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + shift_degrees / 2.0, 180.0)
        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = _apply_masked(out, alpha_f, modified)
        applied.append({"effect": "hue", "shift_degrees": shift_degrees})

    cfg = effects.get("gamma")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("range"), (0.9, 1.1))
        gamma = max(0.05, rng.uniform(lo, hi))
        lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        modified = cv2.LUT(out, lut)
        out = _apply_masked(out, alpha_f, modified)
        applied.append({"effect": "gamma", "gamma": gamma})

    cfg = effects.get("color_temperature")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        raw = cfg.get("strength", cfg.get("kelvin_shift"))
        lo, hi = _float_range(raw, (-10.0, 10.0))
        strength = rng.uniform(lo, hi)
        # Positive strength warms (more R, less B). This is a perceptual control,
        # not an absolute Kelvin delta, hence the neutral ``strength`` name.
        arr = out.astype(np.float32)
        arr[..., 2] += strength * 1.2
        arr[..., 0] -= strength
        out = _apply_masked(out, alpha_f, np.clip(arr, 0, 255))
        applied.append({"effect": "color_temperature", "strength": strength})

    cfg = effects.get("resolution_degrade")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("scale"), (0.4, 0.9))
        scale = min(1.0, max(0.05, rng.uniform(lo, hi)))
        h, w = out.shape[:2]
        sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        premul = out.astype(np.float32) * alpha_f[..., None]
        small_num = cv2.resize(premul, (sw, sh), interpolation=cv2.INTER_AREA)
        small_a = cv2.resize(alpha_f, (sw, sh), interpolation=cv2.INTER_AREA)
        restored_num = cv2.resize(small_num, (w, h), interpolation=cv2.INTER_LINEAR)
        restored_a = cv2.resize(small_a, (w, h), interpolation=cv2.INTER_LINEAR)
        restored = restored_num / np.clip(restored_a[..., None], 1e-4, None)
        out = _apply_masked(out, alpha_f, np.clip(restored, 0, 255))
        applied.append({"effect": "resolution_degrade", "scale": scale})

    cfg = effects.get("gaussian_noise")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("sigma"), (1.0, 5.0))
        sigma = max(0.0, rng.uniform(lo, hi))
        np_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise = np_rng.normal(0.0, sigma, out.shape).astype(np.float32)
        modified = np.clip(out.astype(np.float32) + noise, 0, 255)
        out = _apply_masked(out, alpha_f, modified)
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
            modified = _alpha_aware_filter(
                out, alpha_f,
                lambda arr: cv2.filter2D(arr, -1, kernel),
                lambda arr: cv2.filter2D(arr, -1, kernel),
            )
            out = _apply_masked(out, alpha_f, modified)
            applied.append({"effect": "motion_blur", "ksize": k, "angle": angle})

    cfg = effects.get("jpeg")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _int_range(cfg.get("quality"), (60, 92))
        quality = max(5, min(100, rng.randint(lo, hi)))
        source = _safe_jpeg_source(out, alpha_f)
        ok, enc = cv2.imencode(".jpg", source, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.shape[:2] == out.shape[:2]:
                out = _apply_masked(out, alpha_f, decoded)
                applied.append({"effect": "jpeg", "quality": quality})

    cfg = effects.get("gaussian_blur")
    if isinstance(cfg, dict):
        p_raw = cfg.get("p")
        prob = _clip_probability(blur_prob_fallback, 0.0) if p_raw is None else _clip_probability(p_raw)
        if rng.random() < prob:
            lo, hi = _float_range(cfg.get("sigma"), (0.25, 0.75))
            sigma = max(0.05, rng.uniform(lo, hi))
            modified = _alpha_aware_filter(
                out, alpha_f,
                lambda arr: cv2.GaussianBlur(arr, (0, 0), sigma),
                lambda arr: cv2.GaussianBlur(arr, (0, 0), sigma),
            )
            out = _apply_masked(out, alpha_f, modified)
            applied.append({"effect": "gaussian_blur", "sigma": sigma})

    cfg = effects.get("sharpness")
    if isinstance(cfg, dict) and rng.random() < _clip_probability(cfg.get("p")):
        lo, hi = _float_range(cfg.get("amount"), (0.85, 1.15))
        amount = rng.uniform(lo, hi)
        base = _alpha_aware_filter(
            out, alpha_f,
            lambda arr: cv2.GaussianBlur(arr, (0, 0), 0.8),
            lambda arr: cv2.GaussianBlur(arr, (0, 0), 0.8),
        ).astype(np.float32)
        # amount=1 is neutral. Above 1 applies unsharp-mask; below 1 softens.
        modified = out.astype(np.float32) + (amount - 1.0) * (out.astype(np.float32) - base)
        out = _apply_masked(out, alpha_f, np.clip(modified, 0, 255))
        applied.append({"effect": "sharpness", "amount": amount})

    return out, applied
