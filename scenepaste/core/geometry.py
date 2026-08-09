"""Geometry helpers: mask/polygon conversions, IoU, bbox clipping."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, simplify_ratio: float = 0.002,
                     min_area: float = 4.0) -> List[np.ndarray]:
    """Convert a binary mask to a list of simplified outer polygons.

    Polygons are returned largest-area first. Contours smaller than
    ``min_area`` are dropped.
    """
    u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        eps = max(0.5, simplify_ratio * cv2.arcLength(contour, True))
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) >= 3:
            polys.append(poly)
    polys.sort(key=lambda p: abs(float(cv2.contourArea(p.reshape(-1, 1, 2)))), reverse=True)
    return polys


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x1, y1, x2, y2)`` bbox of non-zero pixels, or ``None`` if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """IoU between two xyxy boxes; returns ``0.0`` when union is empty."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def clip_bbox(box: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clip a bbox to the canvas ``[0, width] x [0, height]`` and normalize order."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def visible_ratio(original_box: Sequence[int], clipped_box: Sequence[int]) -> float:
    """Ratio of clipped bbox area over original bbox area (0 if original is 0)."""
    ox1, oy1, ox2, oy2 = original_box
    cx1, cy1, cx2, cy2 = clipped_box
    orig_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
    if orig_area <= 0:
        return 0.0
    clip_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    return clip_area / float(orig_area)



def _rotation_matrix_expand(width: int, height: int, angle: float):
    """Return OpenCV affine matrix and expanded output size for ``angle``."""
    if abs(float(angle)) < 1e-7:
        return None, int(width), int(height)
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cos = abs(matrix[0, 0]); sin = abs(matrix[0, 1])
    nw = max(1, int(round(height * sin + width * cos)))
    nh = max(1, int(round(height * cos + width * sin)))
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    return matrix, nw, nh


def canvas_polygon(asset, box: Tuple[int, int, int, int],
                   flipped: bool, canvas_w: int, canvas_h: int,
                   transform: Optional[Tuple[float, int, int, bool]] = None) -> Optional[np.ndarray]:
    """Project the asset's local polygon onto the canvas using the paste transform.

    ``transform = (scale, x_offset, y_offset, flip)`` comes straight from
    :func:`scenepaste.core.synthesis.paste_one` (scale is the *final* fg height
    over src height, including any shrink). When ``transform`` is ``None`` the
    bbox is used to back out an approximate scale (legacy behaviour, less
    accurate due to padding).
    """
    if getattr(asset, "polygon", None) is None:
        return None
    local = np.asarray(asset.polygon, dtype=np.float64).copy()
    src_h, src_w = asset.image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return None

    if transform is not None:
        if len(transform) >= 5:
            scale, x_off, y_off, _flip_in_transform, angle = transform[:5]
        else:
            scale, x_off, y_off, _flip_in_transform = transform
            angle = 0.0
    else:
        box_x1, box_y1, box_x2, box_y2 = box
        scale = float(max(1, box_y2 - box_y1)) / float(src_h)
        x_off, y_off = box_x1, box_y1
        angle = 0.0

    out_w = src_w * scale
    poly = local.copy()
    poly[:, 0] *= scale
    poly[:, 1] *= scale
    if flipped:
        poly[:, 0] = out_w - poly[:, 0]
    if abs(float(angle)) >= 1e-7:
        out_h = src_h * scale
        matrix, _rw, _rh = _rotation_matrix_expand(int(round(out_w)), int(round(out_h)), float(angle))
        if matrix is not None:
            ones = np.ones((len(poly), 1), dtype=np.float64)
            poly = np.hstack([poly, ones]) @ matrix.T
    poly[:, 0] += x_off
    poly[:, 1] += y_off
    poly[:, 0] = np.clip(poly[:, 0], 0, canvas_w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, canvas_h - 1)
    return poly.astype(np.float32)


def annotation_canvas_mask(asset, transform: tuple,
                           canvas_w: int, canvas_h: int,
                           threshold: float = 0.5) -> np.ndarray:
    """Project a single asset's alpha mask onto a full-canvas boolean mask."""
    if len(transform) >= 5:
        scale, x_off, y_off, flipped, angle = transform[:5]
    else:
        scale, x_off, y_off, flipped = transform
        angle = 0.0
    src_h, src_w = asset.alpha.shape[:2]
    out_w = max(1, int(round(src_w * float(scale))))
    out_h = max(1, int(round(src_h * float(scale))))
    alpha = cv2.resize(asset.alpha.astype(np.float32), (out_w, out_h),
                       interpolation=cv2.INTER_LINEAR)
    if flipped:
        alpha = cv2.flip(alpha, 1)
    if abs(float(angle)) >= 1e-7:
        matrix, rw, rh = _rotation_matrix_expand(out_w, out_h, float(angle))
        alpha = cv2.warpAffine(alpha, matrix, (rw, rh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        out_w, out_h = rw, rh
    local = alpha >= float(threshold)
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    x1, y1 = int(x_off), int(y_off)
    x2, y2 = x1 + out_w, y1 + out_h
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(canvas_w, x2), min(canvas_h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return mask
    lx1, ly1 = cx1 - x1, cy1 - y1
    lx2, ly2 = lx1 + (cx2 - cx1), ly1 + (cy2 - cy1)
    mask[cy1:cy2, cx1:cx2] = local[ly1:ly2, lx1:lx2]
    return mask


def visible_instance_masks(annotations, canvas_w: int, canvas_h: int,
                           threshold: float = 0.5) -> List[np.ndarray]:
    """Compute each instance's visible region, z-order aware.

    Later entries in ``annotations`` are above earlier ones. Returns one
    boolean mask per annotation.
    """
    raw = [
        annotation_canvas_mask(asset, transform, canvas_w, canvas_h, threshold)
        for asset, _box, _flipped, transform in annotations
    ]
    covered = np.zeros((canvas_h, canvas_w), dtype=bool)
    visible = [np.zeros_like(covered) for _ in raw]
    for i in range(len(raw) - 1, -1, -1):
        visible[i] = raw[i] & ~covered
        covered |= raw[i]
    return visible
