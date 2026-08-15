"""Small scene-understanding layer for placement regions.

V9 keeps manually annotated LabelMe placement regions as the highest-confidence
source, but exposes a stable policy layer so semantic/depth models can be added
later without changing the renderer.  ``auto`` falls back to a conservative
lower-image ground prior when no explicit region exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .io import load_json
from .labelme import load_paste_zone_masks, shape_to_mask

FORBIDDEN_LABELS = {"forbidden", "no_paste", "no-paste", "禁止放置", "禁放区域", "不可粘贴区域"}


def resolve_placement_regions(
    background_path: Path,
    height: int,
    width: int,
    *,
    mode: str = "auto",
    ground_start_ratio: float = 0.35,
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], dict]:
    mode = str(mode or "auto").lower()
    if mode not in {"auto", "explicit", "ground-prior", "none"}:
        raise ValueError("scene_region_mode must be auto / explicit / ground-prior / none")
    generic, by_class = (None, {}) if mode == "none" else load_paste_zone_masks(background_path, height, width)
    source = "explicit" if generic is not None or by_class else "none"

    if mode in {"auto", "ground-prior"} and generic is None:
        generic = np.zeros((height, width), dtype=np.uint8)
        y0 = int(round(np.clip(float(ground_start_ratio), 0.0, 0.95) * height))
        generic[y0:, :] = 255
        source = "ground-prior"

    # Explicit forbidden polygons subtract from both generic/class masks.
    forbidden: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    json_path = Path(background_path).with_suffix(".json")
    if json_path.is_file():
        try:
            data = load_json(json_path)
            for shape in data.get("shapes", []):
                if str(shape.get("label", "")).strip().casefold() not in FORBIDDEN_LABELS:
                    continue
                mask = shape_to_mask(shape, height, width)
                if mask is not None:
                    forbidden = cv2.bitwise_or(forbidden, np.asarray(mask, dtype=np.uint8))
        except Exception:
            pass
    if np.any(forbidden):
        inv = cv2.bitwise_not(forbidden)
        if generic is not None:
            generic = cv2.bitwise_and(generic, inv)
        by_class = {k: cv2.bitwise_and(v, inv) for k, v in by_class.items()}

    meta = {
        "source": source,
        "mode": mode,
        "forbidden_pixels": int(np.count_nonzero(forbidden)),
        "generic_pixels": int(np.count_nonzero(generic)) if generic is not None else 0,
        "class_specific": sorted(by_class),
    }
    return generic, by_class, meta
