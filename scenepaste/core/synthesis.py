"""Single-instance paste composition: placement + collision avoidance + blending."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .augmentation import augment_foreground, resize_asset, rotate_asset
from .config import GenerationConfig
from .geometry import box_iou
from .models import ObjectAsset, PlacementSpec
from .sampling import sample_bottom_point

# How many random placement attempts before giving up on one instance.
PLACEMENT_MAX_ATTEMPTS = 40


def paste_one(
    canvas: np.ndarray,
    asset: ObjectAsset,
    existing_boxes: List[Tuple[int, int, int, int]],
    zone_mask: Optional[np.ndarray],
    config: GenerationConfig,
    rng: random.Random,
    class_mask: Optional[np.ndarray] = None,
    placement: Optional[PlacementSpec] = None,
) -> Optional[Tuple[Tuple[int, int, int, int], bool, tuple]]:
    """Try to paste ``asset`` onto ``canvas`` without exceeding ``config.max_iou``.

    Returns ``(box_xyxy, flipped, transform)`` on success, where ``transform``
    is ``(scale, x1, y1, flip, angle)`` carrying the *final* geometry (used by
    segmentation/COCO writers to derive visible masks accurately).

    When ``class_mask`` is provided, the pasted region's class id is written
    into it (z-order handles occlusion naturally).

    Returns ``None`` when no valid placement is found within
    :data:`PLACEMENT_MAX_ATTEMPTS` attempts.
    """
    height, width = canvas.shape[:2]
    for attempt in range(PLACEMENT_MAX_ATTEMPTS):
        if placement is not None and placement.center_x_ratio is not None and placement.bottom_y_ratio is not None:
            # Small retry jitter only after the requested position fails.
            jitter = 0.0 if attempt == 0 else min(0.03, 0.004 * attempt)
            xr = float(placement.center_x_ratio) + rng.uniform(-jitter, jitter)
            yr = float(placement.bottom_y_ratio) + rng.uniform(-jitter, jitter)
            bottom_x = int(round(np.clip(xr, 0.01, 0.99) * width))
            bottom_y = int(round(np.clip(yr, 0.01, 0.995) * height))
        else:
            bottom_x, bottom_y = sample_bottom_point(
                rng, width, height, config.y_min, config.y_max, zone_mask
            )
        y_ratio = bottom_y / max(1.0, float(height))
        if placement is not None and placement.height_ratio is not None:
            height_ratio = float(placement.height_ratio)
        else:
            perspective_t = np.clip(
                (y_ratio - config.y_min) / max(1e-6, config.y_max - config.y_min), 0.0, 1.0
            )
            height_ratio = config.far_height + perspective_t * (
                config.near_height - config.far_height
            )
            height_ratio *= rng.uniform(0.82, 1.18)
        desired_height = int(round(height_ratio * height))
        flip = bool(placement.flip) if placement is not None and placement.flip is not None else (rng.random() < config.flip_prob)
        angle = float(placement.angle) if placement is not None else 0.0
        foreground, alpha = resize_asset(asset, desired_height, flip)
        foreground, alpha = rotate_asset(foreground, alpha, angle)
        fg_h, fg_w = foreground.shape[:2]
        source_h, source_w = asset.image.shape[:2]
        effective_scale = float(desired_height) / float(max(1, source_h))

        if fg_w >= width * 0.90 or fg_h >= height * 0.90:
            shrink = min(width * 0.85 / fg_w, height * 0.85 / fg_h)
            foreground = cv2.resize(
                foreground,
                (max(3, int(fg_w * shrink)), max(6, int(fg_h * shrink))),
                interpolation=cv2.INTER_AREA,
            )
            alpha = cv2.resize(
                alpha,
                (foreground.shape[1], foreground.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            fg_h, fg_w = foreground.shape[:2]
            effective_scale *= float(shrink)

        x1 = int(round(bottom_x - fg_w / 2.0))
        y1 = int(round(bottom_y - fg_h))
        x1 = max(0, min(width - fg_w, x1))
        y1 = max(0, min(height - fg_h, y1))
        x2, y2 = x1 + fg_w, y1 + fg_h

        visible = alpha > 0.10
        ys, xs = np.where(visible)
        if len(xs) == 0:
            continue
        box = (
            x1 + int(xs.min()),
            y1 + int(ys.min()),
            x1 + int(xs.max()) + 1,
            y1 + int(ys.max()) + 1,
        )
        if box[2] - box[0] < 4 or box[3] - box[1] < 8:
            continue
        if not (placement is not None and placement.allow_overlap):
            if any(box_iou(box, previous) > config.max_iou for previous in existing_boxes):
                continue

        patch = canvas[y1:y2, x1:x2]
        foreground = augment_foreground(
            foreground,
            alpha,
            patch,
            rng,
            config.color_match_strength,
            config.blur_prob,
        )
        blend_alpha = alpha.astype(np.float32)
        if config.blend_mode == "hard":
            blend_alpha = (blend_alpha > 0.5).astype(np.float32)
        elif config.blend_mode == "gaussian":
            sigma = max(0.01, float(config.blend_sigma))
            blend_alpha = cv2.GaussianBlur(blend_alpha, (0, 0), sigma)
            blend_alpha = np.clip(blend_alpha, 0.0, 1.0)
        alpha_3 = blend_alpha[..., None]
        blended = foreground.astype(np.float32) * alpha_3 + patch.astype(np.float32) * (1.0 - alpha_3)
        canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        # Optional: simultaneously write the class-id mask (semantic seg, z-order
        # handles occlusion naturally).
        if class_mask is not None:
            write_mask = (alpha > 0.5)
            sub = class_mask[y1:y2, x1:x2]
            sub[write_mask] = asset.class_id
        # Return the full transform so segmentation/COCO can derive visible
        # masks from the *true* scale + translation rather than the bbox.
        # scale refers to the pre-rotation source scaling; angle makes the transform reconstructable.
        transform = (effective_scale, x1, y1, flip, angle)
        return box, flip, transform
    return None
