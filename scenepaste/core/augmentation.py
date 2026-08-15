"""Foreground augmentation and resizing for paste composition."""

from __future__ import annotations

import random
from typing import List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .models import ObjectAsset
from .object_appearance import apply_object_appearance, load_object_appearance_recipe


def resize_asset(
    asset: ObjectAsset,
    desired_height: int,
    flip: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize an asset to a target height (preserving aspect), optionally flip.

    Uses ``INTER_AREA`` when downsampling, ``INTER_LINEAR`` when upsampling.
    """
    source_h, source_w = asset.image.shape[:2]
    desired_height = max(6, desired_height)
    scale = desired_height / float(source_h)
    desired_width = max(3, int(round(source_w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(asset.image, (desired_width, desired_height), interpolation=interpolation)
    alpha = cv2.resize(asset.alpha, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0)
    if flip:
        image = cv2.flip(image, 1)
        alpha = cv2.flip(alpha, 1)
    return image, alpha


def _legacy_hsv_blur(
    image: np.ndarray,
    alpha: np.ndarray,
    rng: random.Random,
    blur_prob: float,
) -> Tuple[np.ndarray, List[dict]]:
    """Historical v1.0 light HSV jitter + optional blur (no recipe configured)."""
    mask = alpha > 0.10
    if not np.any(mask):
        return image, []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = rng.uniform(0.90, 1.10)
    val = rng.uniform(0.88, 1.12)
    hue = rng.uniform(-2.0, 2.0)
    hsv[..., 1] *= sat
    hsv[..., 2] *= val
    hsv[..., 0] = np.mod(hsv[..., 0] + hue, 180.0)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    foreground = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    applied: List[dict] = [{
        "effect": "hsv_jitter",
        "hue": hue,
        "saturation": sat,
        "value": val,
    }]
    if rng.random() < blur_prob:
        sigma = rng.uniform(0.25, 0.75)
        foreground = cv2.GaussianBlur(foreground, (0, 0), sigma)
        applied.append({"effect": "gaussian_blur", "sigma": sigma})
    return foreground, applied


def augment_foreground(
    image: np.ndarray,
    alpha: np.ndarray,
    background_patch: np.ndarray,
    rng: random.Random,
    color_match_strength: float,
    blur_prob: float,
    *,
    object_appearance_recipe: Optional[Mapping[str, object] | str] = None,
    class_label: Optional[str] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Apply object appearance + optional color-match.

    Returns ``(foreground_bgr, applied_effect_metadata)``.

    When ``object_appearance_recipe`` is set, photometric/degrade effects come
    from the recipe (``blur_prob`` only fills ``gaussian_blur.p = null`` for
    the ``legacy`` built-in). When unset, the historical HSV + blur path runs.
    Color matching remains a separate blend step after appearance.
    """
    mask = alpha > 0.10
    if not np.any(mask):
        return image, []

    recipe = object_appearance_recipe
    if isinstance(recipe, str) or recipe is None:
        # ``None`` keeps legacy path; string/path/name goes through the loader.
        loaded = load_object_appearance_recipe(recipe) if recipe else None
    else:
        loaded = recipe

    if loaded is None:
        foreground, applied = _legacy_hsv_blur(image, alpha, rng, blur_prob)
    else:
        foreground, applied = apply_object_appearance(
            image,
            alpha,
            loaded,
            rng,
            class_label=class_label,
            blur_prob_fallback=blur_prob,
        )

    foreground = foreground.astype(np.float32)
    if color_match_strength > 0 and background_patch.size:
        # Match only the pixels actually covered by the object. Using the
        # whole bounding rectangle biases thin/rotated objects toward unrelated
        # background colors and can produce visible color casts. Soft alpha
        # weighting also keeps antialiased edges consistent.
        weights = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        total = float(weights.sum())
        if total > 1e-6 and background_patch.shape[:2] == foreground.shape[:2]:
            target_mean = (foreground * weights[..., None]).sum(axis=(0, 1)) / total
            bg = background_patch.astype(np.float32)
            background_mean = (bg * weights[..., None]).sum(axis=(0, 1)) / total
            shift = (background_mean - target_mean) * color_match_strength
            foreground += shift.reshape(1, 1, 3)
            applied = list(applied) + [{
                "effect": "color_match",
                "strength": float(color_match_strength),
                "shift_bgr": [float(x) for x in shift.reshape(-1)],
            }]

    foreground = np.clip(foreground, 0, 255).astype(np.uint8)
    return foreground, applied


def rotate_asset(image: np.ndarray, alpha: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image+alpha around center while expanding the output canvas."""
    angle = float(angle)
    if abs(angle) < 1e-7:
        return image, alpha
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0]); sin = abs(matrix[0, 1])
    nw = max(1, int(round(h * sin + w * cos)))
    nh = max(1, int(round(h * cos + w * sin)))
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    out_img = cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out_alpha = cv2.warpAffine(alpha.astype(np.float32), matrix, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out_img, np.clip(out_alpha, 0.0, 1.0)
