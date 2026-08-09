"""Core data models and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Supported image file extensions (lowercase, with dot).
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# LabelMe shape labels that mark allowed placement regions on a background.
PASTE_ZONE_LABELS = {"paste_zone", "ground", "road", "可粘贴区域", "地面"}
PASTE_ZONE_PREFIXES = ("paste_zone:", "ground:", "road:", "zone:", "可粘贴区域:", "地面:")


def is_paste_zone_label(label: str) -> bool:
    text = str(label).strip().casefold()
    return text in PASTE_ZONE_LABELS or any(text.startswith(prefix) for prefix in PASTE_ZONE_PREFIXES)


def paste_zone_target_label(label: str):
    """Return the class suffix for ``zone:class`` labels, else ``None``."""
    text = str(label).strip()
    folded = text.casefold()
    for prefix in PASTE_ZONE_PREFIXES:
        if folded.startswith(prefix):
            return text[len(prefix):].strip() or None
    return None


@dataclass
class ObjectAsset:
    """A cutout object ready to be pasted onto backgrounds.

    Attributes:
        label: Class name from the LabelMe annotation (or ``"auto"`` for
            auto-cutout mode).
        class_id: Integer class id resolved from ``class_map``.
        image: BGR cropped foreground image (uint8 ndarray).
        alpha: Per-pixel alpha in ``[0, 1]`` matching ``image`` shape.
        source_json: Path of the originating LabelMe JSON / image file.
        source_shape_index: Index of the shape inside the source JSON.
        polygon: Local-coordinate polygon (Nx2 float32). ``None`` when the
            asset has no usable polygon (segmentation output is skipped).
    """

    label: str
    class_id: int
    image: np.ndarray
    alpha: np.ndarray
    source_json: Path
    source_shape_index: int
    polygon: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PlacementSpec:
    """Optional deterministic placement sampled by a profile/template planner.

    Ratios are normalized to the destination canvas. ``bottom_y_ratio`` is
    the intended object foot / bbox-bottom location.  ``source_name`` pins a
    template to one portable cutout identity; ``same_class_random`` lets a
    parameterized template keep the semantic role but vary the appearance.
    """

    class_id: Optional[int] = None
    label: Optional[str] = None
    source_name: Optional[str] = None
    center_x_ratio: Optional[float] = None
    bottom_y_ratio: Optional[float] = None
    height_ratio: Optional[float] = None
    flip: Optional[bool] = None
    angle: float = 0.0
    allow_overlap: bool = False
    same_class_random: bool = False
