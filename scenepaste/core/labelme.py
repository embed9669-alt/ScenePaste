"""LabelMe annotation loading: shape -> mask, asset extraction, paste zones."""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .io import imread_unicode, imread_with_exif, imwrite_unicode, load_json
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



def _refine_rectangle_mask_grabcut(
    image: np.ndarray,
    rect_xyxy: tuple[int, int, int, int],
    crop_xyxy: tuple[int, int, int, int],
    *,
    iterations: int = 5,
) -> tuple[Optional[np.ndarray], dict]:
    """Refine a detector-style rectangle into an object-shaped foreground mask.

    LabelMe rectangles are bounding boxes, not semantic masks. Treating the whole
    rectangle as alpha causes the source image background to be pasted as a hard
    rectangular tile. GrabCut gives us a dependency-free foreground estimate so
    detector annotations can still be used safely for copy/paste generation.

    The returned mask lives in crop coordinates and is strictly clipped to the
    annotated rectangle. If the estimate is implausible, ``None`` is returned so
    callers can reject the asset instead of silently producing rectangular junk.
    """
    cx1, cy1, cx2, cy2 = map(int, crop_xyxy)
    rx1, ry1, rx2, ry2 = map(int, rect_xyxy)
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 8 or cw < 8:
        return None, {"ok": False, "reason": "crop-too-small"}

    lx1 = int(np.clip(rx1 - cx1, 0, cw - 1))
    ly1 = int(np.clip(ry1 - cy1, 0, ch - 1))
    lx2 = int(np.clip(rx2 - cx1, lx1 + 1, cw))
    ly2 = int(np.clip(ry2 - cy1, ly1 + 1, ch))
    rw, rh = max(1, lx2 - lx1), max(1, ly2 - ly1)

    gc_mask = np.full((ch, cw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[ly1:ly2, lx1:lx2] = cv2.GC_PR_FGD

    # Give GrabCut a conservative definite-foreground seed near the box centre.
    # This is deliberately small: it stabilizes ordinary person/vehicle boxes
    # without forcing the full detector rectangle to foreground.
    inset_x = max(1, int(round(rw * 0.34)))
    inset_y = max(1, int(round(rh * 0.34)))
    sx1, sx2 = lx1 + inset_x, lx2 - inset_x
    sy1, sy2 = ly1 + inset_y, ly2 - inset_y
    if sx2 > sx1 and sy2 > sy1:
        gc_mask[sy1:sy2, sx1:sx2] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc_mask, None, bgd, fgd, int(max(1, iterations)), cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        return None, {"ok": False, "reason": f"grabcut-error:{exc}"}

    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    allowed = np.zeros_like(fg)
    allowed[ly1:ly2, lx1:lx2] = 255
    fg = cv2.bitwise_and(fg, allowed)

    # Remove isolated speckles while retaining several meaningful components
    # (e.g. wheels can become weakly disconnected from a motorcycle body).
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), 8)
    if count > 1:
        areas = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, count)]
        areas.sort(key=lambda x: x[1], reverse=True)
        if areas:
            largest = areas[0][1]
            keep = {i for i, area in areas if area >= max(12, int(largest * 0.06))}
            fg = np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)

    rect_area = float(rw * rh)
    fg_area = float(np.count_nonzero(fg))
    fill_ratio = fg_area / max(1.0, rect_area)
    ys, xs = np.where(fg > 0)
    if not len(xs):
        return None, {"ok": False, "reason": "empty", "fill_ratio": 0.0}
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    tight_area = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    rectangularity = fg_area / tight_area

    ok = 0.04 <= fill_ratio <= 0.90 and rectangularity < 0.94
    meta = {
        "ok": bool(ok),
        "fill_ratio": round(fill_ratio, 4),
        "rectangularity": round(rectangularity, 4),
        "method": "grabcut",
    }
    if not ok:
        meta["reason"] = "implausible-foreground-mask"
        return None, meta
    return fg, meta


def load_object_assets(
    objects_dir: Path,
    class_map: Dict[str, int],
    feather_sigma: float,
    log: Callable[[str], None],
    rectangle_mask_mode: str = "legacy",
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
    rectangle_mask_mode = str(rectangle_mask_mode or "legacy").strip().lower()
    if rectangle_mask_mode not in {"grabcut", "reject", "legacy"}:
        raise ValueError("rectangle_mask_mode 必须是 grabcut / reject / legacy")
    refined_rectangles = 0
    rejected_rectangles = 0
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
            shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
            # Rectangle annotations are detector boxes, not foreground masks.
            # In data-factory mode we refine them by default instead of pasting
            # the whole source rectangle (which creates the obvious "web tile" artifact).
            rect_mode_active = shape_type == "rectangle" and rectangle_mask_mode != "legacy"
            padding_ratio = 0.10 if rect_mode_active else 0.01
            padding = max(8 if rect_mode_active else 2, int(round(min(w, h) * padding_ratio)))
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            crop = image[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2].copy()

            if shape_type == "rectangle" and rectangle_mask_mode == "reject":
                rejected_rectangles += 1
                log(
                    f"[拒绝矩形标注] {json_path.name}#{shape_index + 1}："
                    "bbox 不是前景 Mask；请改用 polygon/segmentation，或启用 GrabCut 自动细化。"
                )
                continue
            if shape_type == "rectangle" and rectangle_mask_mode == "grabcut":
                refined, mask_meta = _refine_rectangle_mask_grabcut(
                    image, (x, y, x + w, y + h), (x1, y1, x2, y2)
                )
                if refined is None:
                    rejected_rectangles += 1
                    log(
                        f"[拒绝矩形标注] {json_path.name}#{shape_index + 1}："
                        f"GrabCut 无法得到可靠前景（{mask_meta}）。不会退回整块矩形贴图。"
                    )
                    continue
                crop_mask = refined
                refined_rectangles += 1

            # Local polygon (translated to crop origin) for later segmentation reuse.
            pts = np.asarray(shape.get("points", []), dtype=np.float32)
            local_poly: Optional[np.ndarray] = None
            if shape_type == "polygon" and pts.ndim == 2 and pts.shape[0] >= 3:
                local_poly = pts.copy()
                local_poly[:, 0] -= x1
                local_poly[:, 1] -= y1
            elif shape_type == "rectangle" and rectangle_mask_mode == "grabcut":
                contours, _ = cv2.findContours((crop_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    biggest = max(contours, key=cv2.contourArea)
                    if len(biggest) >= 3:
                        eps = 0.003 * cv2.arcLength(biggest, True)
                        local_poly = cv2.approxPolyDP(biggest, eps, True).reshape(-1, 2).astype(np.float32)
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
            if shape_type == "rectangle" and rectangle_mask_mode == "grabcut":
                mask_source = "rectangle-grabcut"
            elif shape_type == "rectangle":
                mask_source = "rectangle-legacy"
            else:
                mask_source = "polygon"
            assets.append(
                ObjectAsset(
                    label=label,
                    class_id=class_map[label],
                    image=crop,
                    alpha=alpha,
                    source_json=json_path,
                    source_shape_index=shape_index,
                    polygon=local_poly,
                    mask_source=mask_source,
                )
            )

    if skipped_unknown:
        log("[提示] 未在类别映射中的标注已忽略：" + ", ".join(sorted(skipped_unknown)))
    if rectangle_mask_mode == "grabcut" and (refined_rectangles or rejected_rectangles):
        log(f"矩形标注前景细化：成功 {refined_rectangles}，拒绝 {rejected_rectangles}（不会使用整块 bbox 当贴图）")
    if not assets:
        raise RuntimeError(t("assets.none_extracted"))
    return assets


def write_labelme_json(
    json_path: Path,
    *,
    image_path: str,
    image_height: int,
    image_width: int,
    shapes: List[dict],
    version: str = "4.5.6",
) -> Path:
    """Write a LabelMe-compatible annotation JSON (imageData left null).

    Each shape should contain at least ``label``, ``points`` (Nx2 list) and
    ``shape_type`` (``polygon`` / ``rectangle``).
    """
    import json

    json_path = Path(json_path)
    payload = {
        "version": version,
        "flags": {},
        "shapes": [],
        "imagePath": str(image_path),
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    for shape in shapes:
        points = shape.get("points") or []
        pts = [[float(x), float(y)] for x, y in points]
        payload["shapes"].append({
            "label": str(shape.get("label") or "object"),
            "score": shape.get("score", None),
            "points": pts,
            "group_id": shape.get("group_id", None),
            "shape_type": str(shape.get("shape_type") or "polygon"),
            "description": str(shape.get("description") or ""),
            "flags": dict(shape.get("flags") or {}),
        })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def polygon_shape(label: str, points) -> dict:
    """Build one LabelMe polygon shape dict from an Nx2 array-like."""
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return {
        "label": str(label),
        "points": [[float(x), float(y)] for x, y in arr],
        "shape_type": "polygon",
    }


def save_cutout_as_labelme(
    source_image: Path,
    out_dir: Path,
    *,
    label: str,
    polygon: np.ndarray,
    bgr: Optional[np.ndarray] = None,
) -> Path:
    """Save/copy one source image plus a sibling LabelMe polygon JSON.

    This function intentionally lives in the core layer so dataset preparation
    works on headless servers without importing PySide6.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_image = Path(source_image)
    stem = source_image.stem
    suffix = source_image.suffix.lower() or ".jpg"
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    dest_img = out_dir / f"{stem}{suffix}"
    if source_image.resolve() != dest_img.resolve():
        if bgr is not None:
            imwrite_unicode(dest_img, bgr)
        else:
            shutil.copy2(source_image, dest_img)
    elif bgr is not None:
        imwrite_unicode(dest_img, bgr)

    if bgr is None:
        bgr = imread_with_exif(dest_img)
    if bgr is None:
        raise RuntimeError(f"无法读取图像：{dest_img}")
    h, w = bgr.shape[:2]
    json_path = out_dir / f"{stem}.json"
    write_labelme_json(
        json_path,
        image_path=dest_img.name,
        image_height=h,
        image_width=w,
        shapes=[polygon_shape(label, polygon)],
    )
    return json_path
