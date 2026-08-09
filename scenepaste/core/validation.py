"""Validation helpers for bounding boxes and generation config."""

from __future__ import annotations

from typing import Tuple, Sequence

from .geometry import clip_bbox, visible_ratio


def is_valid_yolo_box(box: Sequence[int], width: int, height: int,
                      min_visible: float = 0.10,
                      min_size: int = 4) -> Tuple[bool, Tuple[int, int, int, int]]:
    """Validate a bbox: clip to canvas, check visible ratio + minimum size.

    Returns ``(is_valid, clipped_box)``. The clipped box is always returned
    even when validation fails, so callers can decide what to do with it.
    """
    clipped = clip_bbox(box, width, height)
    cx1, cy1, cx2, cy2 = clipped
    if cx2 - cx1 < min_size or cy2 - cy1 < max(2, min_size // 2):
        return False, clipped
    if visible_ratio(box, clipped) < min_visible:
        return False, clipped
    return True, clipped
