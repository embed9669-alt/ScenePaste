"""LabelMe annotation loading: shape -> mask, asset extraction, paste zones."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .io import imread_unicode, load_json
from .models import (
    IMAGE_SUFFIXES, ObjectAsset, PASTE_ZONE_LABELS, is_paste_zone_label, paste_zone_target_label,
)
from ..errors import t


def shape_to_mask(shape: dict, height: int, width: int) -> Optional[np.ndarray]:
    """Rasterize a LabelMe shape (polygon or rectangle) to a uint8 mask.

    Returns ``None`` for unsupported shape types or empty masks.
    """
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return None

    shape_type = shape.get("shape_type", "polygon")
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape_type == "polygon" and len(points) >= 3:
        polygon = np.round(points).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 255)
    elif shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))
        y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    else:
        return None
    return mask if cv2.countNonZero(mask) > 0 else None


def resolve_labelme_image(json_path: Path, data: dict) -> Tuple[np.ndarray, str]:
    """Resolve the source image for a LabelMe JSON.

    Tries ``imageData`` (base64) first, then ``imagePath`` relative to the
    JSON, then any sibling image with the same stem. Raises ``FileNotFoundError``
    if nothing resolves.
    """
    image_data = data.get("imageData")
    if image_data:
        try:
            raw = base64.b64decode(image_data)
            array = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is not None:
                return image, "embedded:imageData"
        except (ValueError, TypeError):
            pass

    candidates: List[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        normalized = str(image_path).replace("\\", "/")
        candidates.append(json_path.parent / normalized)
        candidates.append(json_path.parent / Path(normalized).name)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(json_path.with_suffix(suffix))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            image = imread_unicode(candidate)
            if image is not None:
                return image, str(candidate)
    raise FileNotFoundError(t("assets.labelme_image_not_found", name=json_path.name))


def load_paste_zone_mask(background_path: Path, height: int, width: int) -> Optional[np.ndarray]:
    """Load the union of paste-zone polygons from a background's sibling JSON.

    Returns ``None`` when the background has no JSON, or no shape with a
    label in :data:`PASTE_ZONE_LABELS`.
    """
    json_path = background_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        data = load_json(json_path)
    except Exception:
        return None

    combined = np.zeros((height, width), dtype=np.uint8)
    found = False
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in PASTE_ZONE_LABELS:
            continue
        mask = shape_to_mask(shape, height, width)
        if mask is not None:
            combined = cv2.bitwise_or(combined, mask)  # type: ignore[assignment]
            found = True
    return combined if found else None


def load_paste_zone_masks(background_path: Path, height: int, width: int):
    """Load generic and class-specific placement-zone masks.

    Generic labels include ``paste_zone`` / ``ground`` / ``road``.  A label
    like ``paste_zone:person`` or ``zone:truck`` restricts that class to the
    corresponding polygons while other classes continue to use the generic
    zone (or the default ground band when no generic zone exists).
    """
    json_path = background_path.with_suffix(".json")
    if not json_path.exists():
        return None, {}
    try:
        data = load_json(json_path)
    except Exception:
        return None, {}
    generic = np.zeros((height, width), dtype=np.uint8)
    generic_found = False
    by_class = {}
    for shape in data.get("shapes", []):
        raw_label = str(shape.get("label", "")).strip()
        folded = raw_label.casefold()
        if not is_paste_zone_label(raw_label):
            continue
        mask = shape_to_mask(shape, height, width)
        if mask is None:
            continue
        target = paste_zone_target_label(raw_label)
        if target:
            key = target.casefold()
            current = by_class.get(key)
            by_class[key] = mask if current is None else cv2.bitwise_or(current, mask)
        elif folded in PASTE_ZONE_LABELS:
            generic = cv2.bitwise_or(generic, mask)
            generic_found = True
    return (generic if generic_found else None), by_class


def load_object_assets(
    objects_dir: Path,
    class_map: Dict[str, int],
    feather_sigma: float,
    log: Callable[[str], None],
) -> List[ObjectAsset]:
    """Load all labeled object assets from a LabelMe JSON directory.

    For each JSON, resolves its image, rasterizes every shape whose label is
    in ``class_map``, crops tightly with light padding, optionally feathers
    the alpha (to remove the original background halo), and stores the local
    polygon for later segmentation output.

    Raises ``RuntimeError`` if no JSON files exist or no usable assets were
    extracted.
    """
    assets: List[ObjectAsset] = []
    json_paths = sorted(objects_dir.rglob("*.json"))
    if not json_paths:
        raise RuntimeError(t("assets.no_json", path=objects_dir))

    skipped_unknown = set()
    for json_path in json_paths:
        try:
            data = load_json(json_path)
            image, _ = resolve_labelme_image(json_path, data)
        except Exception as exc:
            log(f"[跳过] {json_path.name}：{exc}")
            continue

        height, width = image.shape[:2]
        for shape_index, shape in enumerate(data.get("shapes", [])):
            label = str(shape.get("label", "")).strip()
            if label not in class_map:
                if label and not is_paste_zone_label(label):
                    skipped_unknown.add(label)
                continue
            mask = shape_to_mask(shape, height, width)
            if mask is None:
                log(f"[跳过] {json_path.name} 的第 {shape_index + 1} 个标注不是有效多边形/矩形")
                continue

            x, y, w, h = cv2.boundingRect(mask)
            if w < 3 or h < 3:
                continue
            padding = max(2, int(round(min(w, h) * 0.01)))
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            crop = image[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2].copy()

            # Local polygon (translated to crop origin) for later segmentation reuse.
            shape_type = str(shape.get("shape_type", "polygon"))
            pts = np.asarray(shape.get("points", []), dtype=np.float32)
            local_poly: Optional[np.ndarray] = None
            if shape_type == "polygon" and pts.ndim == 2 and pts.shape[0] >= 3:
                local_poly = pts.copy()
                local_poly[:, 0] -= x1
                local_poly[:, 1] -= y1
            elif shape_type == "rectangle" and pts.ndim == 2 and pts.shape[0] >= 2:
                rx1, ry1 = np.floor(pts.min(axis=0)).astype(np.float32)
                rx2, ry2 = np.ceil(pts.max(axis=0)).astype(np.float32)
                local_poly = np.array(
                    [[rx1 - x1, ry1 - y1],
                     [rx2 - x1, ry1 - y1],
                     [rx2 - x1, ry2 - y1],
                     [rx1 - x1, ry2 - y1]],
                    dtype=np.float32,
                )
            if local_poly is not None:
                local_poly[:, 0] = np.clip(local_poly[:, 0], 0, crop.shape[1] - 1)
                local_poly[:, 1] = np.clip(local_poly[:, 1], 0, crop.shape[0] - 1)

            if feather_sigma > 0:
                # Slight erode before feathering to suppress the bright halo
                # left by the original background.
                if min(crop_mask.shape) >= 10:
                    crop_mask = cv2.erode(crop_mask, np.ones((3, 3), np.uint8), iterations=1)
                crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), feather_sigma)

            alpha = crop_mask.astype(np.float32) / 255.0
            assets.append(
                ObjectAsset(
                    label=label,
                    class_id=class_map[label],
                    image=crop,
                    alpha=alpha,
                    source_json=json_path,
                    source_shape_index=shape_index,
                    polygon=local_poly,
                )
            )

    if skipped_unknown:
        log("[提示] 未在类别映射中的标注已忽略：" + ", ".join(sorted(skipped_unknown)))
    if not assets:
        raise RuntimeError(t("assets.none_extracted"))
    return assets
