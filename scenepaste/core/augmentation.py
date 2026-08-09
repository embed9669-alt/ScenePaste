"""Foreground augmentation and resizing for paste composition."""

from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np

from .models import ObjectAsset


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


def augment_foreground(
    image: np.ndarray,
    alpha: np.ndarray,
    background_patch: np.ndarray,
    rng: random.Random,
    color_match_strength: float,
    blur_prob: float,
) -> np.ndarray:
    """Apply light HSV jitter + optional color-match + optional blur.

    Color matching pulls the foreground mean toward the local background
    patch mean by ``color_match_strength`` in ``[0, 1]``.
    """
    foreground = image.astype(np.float32)
    mask = alpha > 0.10
    if not np.any(mask):
        return image

    # Light brightness/saturation jitter (avoids unnatural strong color shifts).
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.90, 1.10)
    hsv[..., 2] *= rng.uniform(0.88, 1.12)
    hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-2.0, 2.0), 180.0)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    foreground = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if color_match_strength > 0 and background_patch.size:
        target_pixels = foreground[mask]
        background_pixels = background_patch.reshape(-1, 3).astype(np.float32)
        if len(target_pixels) and len(background_pixels):
            target_mean = target_pixels.mean(axis=0)
            background_mean = background_pixels.mean(axis=0)
            shift = (background_mean - target_mean) * color_match_strength
            foreground += shift.reshape(1, 1, 3)

    foreground = np.clip(foreground, 0, 255).astype(np.uint8)
    if rng.random() < blur_prob:
        sigma = rng.uniform(0.25, 0.75)
        foreground = cv2.GaussianBlur(foreground, (0, 0), sigma)  # type: ignore[assignment]
    return foreground


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
