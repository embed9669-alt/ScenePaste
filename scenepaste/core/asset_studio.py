"""Headless helpers for the ScenePaste Asset Studio.

The Asset Studio turns one real annotated image into reusable, human-reviewed
training assets:

* an editable semantic/instance mask,
* a transparent foreground cutout,
* an independently editable background-removal mask,
* a clean background created by deterministic inpainting,
* provenance metadata so every derived asset can be traced back.

The core functions intentionally have no Qt dependency so they can be tested and
used in batch/headless workflows.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import imwrite_unicode
from .labelme import polygon_shape, write_labelme_json


@dataclass(frozen=True)
class AssetStudioExport:
    foreground_path: Path
    foreground_json: Path
    background_path: Path
    bundle_dir: Path
    edited_annotation: Path


def binary_mask(mask: np.ndarray, *, threshold: int = 127) -> np.ndarray:
    """Return a contiguous uint8 mask with values 0/255."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype.kind == "f":
        # Accept masks in either 0..1 or 0..255 form.
        vmax = float(np.nanmax(arr)) if arr.size else 0.0
        arr = arr * (255.0 if vmax <= 1.0 else 1.0)
    return np.ascontiguousarray((arr > int(threshold)).astype(np.uint8) * 255)


def mask_to_polygons(
    mask: np.ndarray,
    *,
    min_area: float = 12.0,
    simplify_epsilon: float = 0.0025,
) -> List[np.ndarray]:
    """Convert a binary mask to external LabelMe-compatible polygons."""
    m = binary_mask(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        eps = max(0.5, perimeter * float(max(0.0, simplify_epsilon)))
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) >= 3:
            rows.append((area, approx))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [poly for _, poly in rows]


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes without changing the outer silhouette."""
    m = binary_mask(mask)
    h, w = m.shape[:2]
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(m, holes)


def morph_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate for positive pixels, erode for negative pixels."""
    m = binary_mask(mask)
    radius = abs(int(pixels))
    if radius <= 0:
        return m
    k = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if pixels > 0:
        return cv2.dilate(m, kernel, iterations=1)
    return cv2.erode(m, kernel, iterations=1)


def clean_mask(mask: np.ndarray, *, radius: int = 2, fill_holes: bool = True) -> np.ndarray:
    """Remove small edge noise while preserving the main object geometry."""
    m = binary_mask(mask)
    r = max(1, int(radius))
    k = r * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    if fill_holes:
        m = fill_mask_holes(m)
    return m


def mask_bbox(mask: np.ndarray, *, padding: int = 4) -> Tuple[int, int, int, int]:
    """Return clipped x1,y1,x2,y2 for nonzero mask pixels."""
    m = binary_mask(mask)
    ys, xs = np.where(m > 0)
    if not len(xs):
        raise ValueError("mask is empty")
    h, w = m.shape[:2]
    p = max(0, int(padding))
    x1 = max(0, int(xs.min()) - p)
    y1 = max(0, int(ys.min()) - p)
    x2 = min(w, int(xs.max()) + 1 + p)
    y2 = min(h, int(ys.max()) + 1 + p)
    return x1, y1, x2, y2


def make_foreground_rgba(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    feather_px: float = 1.0,
    crop: bool = True,
    crop_padding: int = 4,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Build a transparent BGRA cutout and return its top-left source offset."""
    m = binary_mask(mask)
    if not np.any(m):
        raise ValueError("mask is empty")
    alpha = m.astype(np.float32)
    sigma = max(0.0, float(feather_px))
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        # Keep the alpha support local; faint full-image blur tails are not useful.
        support = morph_mask(m, max(1, int(round(sigma * 3)))) > 0
        alpha[~support] = 0
    alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)
    bgra = np.dstack([np.asarray(bgr, dtype=np.uint8), alpha_u8])
    if not crop:
        return bgra, (0, 0)
    x1, y1, x2, y2 = mask_bbox(m, padding=crop_padding)
    return bgra[y1:y2, x1:x2].copy(), (x1, y1)


def make_clean_background(
    bgr: np.ndarray,
    removal_mask: np.ndarray,
    *,
    expand_px: int = 3,
    radius: float = 3.0,
    method: str = "telea",
) -> np.ndarray:
    """Remove the selected object region with deterministic OpenCV inpainting.

    This is intentionally the default background generator: unlike generative
    object editing it is deterministic, does not alter pixels outside the edited
    region and introduces no new foreground semantics.
    """
    mask = binary_mask(removal_mask)
    if expand_px:
        mask = morph_mask(mask, int(expand_px))
    if not np.any(mask):
        return np.asarray(bgr, dtype=np.uint8).copy()
    flag = cv2.INPAINT_TELEA if str(method).lower() != "ns" else cv2.INPAINT_NS
    return cv2.inpaint(np.asarray(bgr, dtype=np.uint8), mask, max(1.0, float(radius)), flag)


def _translated_polygons(mask: np.ndarray, offset: Tuple[int, int]) -> List[np.ndarray]:
    x0, y0 = map(float, offset)
    rows = []
    for poly in mask_to_polygons(mask):
        p = poly.copy()
        p[:, 0] -= x0
        p[:, 1] -= y0
        rows.append(p)
    return rows


def export_asset_bundle(
    *,
    source_image: Path,
    bgr: np.ndarray,
    label: str,
    instance_index: int,
    auto_mask: np.ndarray,
    edited_mask: np.ndarray,
    background_remove_mask: np.ndarray,
    objects_dir: Path,
    backgrounds_dir: Path,
    bundle_root: Path,
    feather_px: float = 1.0,
    background_expand_px: int = 3,
    background_inpaint_radius: float = 3.0,
    background_method: str = "telea",
) -> AssetStudioExport:
    """Export one reviewed object as foreground + clean background + provenance."""
    source_image = Path(source_image)
    objects_dir = Path(objects_dir)
    backgrounds_dir = Path(backgrounds_dir)
    bundle_root = Path(bundle_root)
    label = str(label or "object").strip() or "object"
    safe_label = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in label) or "object"
    stem = f"{source_image.stem}__i{int(instance_index):02d}__{safe_label}"

    edited = binary_mask(edited_mask)
    auto = binary_mask(auto_mask)
    bg_remove = binary_mask(background_remove_mask)
    if not np.any(edited):
        raise ValueError("edited foreground mask is empty")

    foreground, offset = make_foreground_rgba(
        bgr, edited, feather_px=feather_px, crop=True, crop_padding=max(4, int(round(feather_px * 3)))
    )
    x0, y0 = offset
    crop_mask = edited[y0 : y0 + foreground.shape[0], x0 : x0 + foreground.shape[1]]
    polys = mask_to_polygons(crop_mask)
    if not polys:
        raise ValueError("edited mask could not be converted to a valid polygon")

    object_out = objects_dir / "edited_assets" / safe_label
    object_out.mkdir(parents=True, exist_ok=True)
    fg_path = object_out / f"{stem}.png"
    if not imwrite_unicode(fg_path, foreground):
        raise RuntimeError(f"failed to write foreground: {fg_path}")
    fg_json = object_out / f"{stem}.json"
    write_labelme_json(
        fg_json,
        image_path=fg_path.name,
        image_height=foreground.shape[0],
        image_width=foreground.shape[1],
        shapes=[polygon_shape(label, p) for p in polys],
    )

    clean_bg = make_clean_background(
        bgr,
        bg_remove,
        expand_px=background_expand_px,
        radius=background_inpaint_radius,
        method=background_method,
    )
    backgrounds_dir.mkdir(parents=True, exist_ok=True)
    bg_path = backgrounds_dir / f"{source_image.stem}__clean_i{int(instance_index):02d}.jpg"
    if not imwrite_unicode(bg_path, clean_bg, jpeg_quality=95):
        raise RuntimeError(f"failed to write clean background: {bg_path}")

    bundle = bundle_root / stem
    bundle.mkdir(parents=True, exist_ok=True)
    suffix = source_image.suffix.lower() if source_image.suffix else ".jpg"
    original_copy = bundle / f"original{suffix}"
    try:
        if source_image.is_file():
            shutil.copy2(source_image, original_copy)
        else:
            imwrite_unicode(original_copy, bgr)
    except OSError:
        imwrite_unicode(original_copy, bgr)
    imwrite_unicode(bundle / "mask_auto.png", auto)
    imwrite_unicode(bundle / "mask_edited.png", edited)
    imwrite_unicode(bundle / "mask_background_remove.png", bg_remove)
    imwrite_unicode(bundle / "foreground.png", foreground)
    imwrite_unicode(bundle / "background_clean.jpg", clean_bg, jpeg_quality=95)

    # Human-reviewed annotation in original-image coordinates. Keep it outside
    # the object library to avoid accidentally double-loading the source image.
    edited_annotation = bundle / "annotation_edited.json"
    original_polys = mask_to_polygons(edited)
    write_labelme_json(
        edited_annotation,
        image_path=source_image.name,
        image_height=bgr.shape[0],
        image_width=bgr.shape[1],
        shapes=[polygon_shape(label, p) for p in original_polys],
    )

    metadata = {
        "schema": "scenepaste.asset_studio.v1",
        "source_image": str(source_image),
        "label": label,
        "instance_index": int(instance_index),
        "foreground_path": str(fg_path),
        "foreground_json": str(fg_json),
        "background_path": str(bg_path),
        "mask_area_px": int(np.count_nonzero(edited)),
        "background_remove_area_px": int(np.count_nonzero(bg_remove)),
        "foreground_offset_xy": [int(x0), int(y0)],
        "background_fill": {
            "backend": "opencv-inpaint",
            "method": str(background_method),
            "expand_px": int(background_expand_px),
            "radius": float(background_inpaint_radius),
        },
        "notes": "Masks are human-editable assets; generated training labels use the reviewed edited mask.",
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return AssetStudioExport(
        foreground_path=fg_path,
        foreground_json=fg_json,
        background_path=bg_path,
        bundle_dir=bundle,
        edited_annotation=edited_annotation,
    )
