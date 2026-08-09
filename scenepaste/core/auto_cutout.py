"""Auto-cutout via rembg (no LabelMe JSON required)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from .geometry import mask_to_polygons  # noqa: F401  (kept for re-export convenience)
from .io import imread_with_exif
from .models import IMAGE_SUFFIXES, ObjectAsset
from ..errors import t


def alpha_to_polygon(alpha_u8: np.ndarray, simplify_eps: float = 0.004):
    """Extract the largest external contour from an alpha mask as a polygon.

    Returns ``None`` when no usable contour exists.
    """
    bin_mask = (alpha_u8 > 64).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 3:
        return None
    eps = simplify_eps * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, eps, True).reshape(-1, 2).astype(np.float32)
    return approx if len(approx) >= 3 else None


def load_object_assets_auto(
    objects_dir: Path,
    class_map: Dict[str, int],
    log: Callable[[str], None],
    backend: str = "rembg",
    polygon_simplify_eps: float = 0.004,
) -> List[ObjectAsset]:
    """Auto-cutout every image in ``objects_dir`` using rembg.

    All cutouts are labeled ``"auto"`` (added to ``class_map`` if missing).
    Requires the ``rembg`` extra: ``pip install 'scenepaste[auto]'``.
    """
    try:
        from rembg import remove as _rembg_remove
    except ImportError as exc:
        raise ImportError(t("assets.rembg_missing")) from exc

    image_paths = sorted(
        p for p in objects_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(t("assets.image_dir_empty", path=objects_dir))

    if "auto" not in class_map:
        next_id = max(list(class_map.values()) + [-1]) + 1
        class_map["auto"] = next_id
    auto_id = class_map["auto"]

    assets: List[ObjectAsset] = []
    for path in image_paths:
        try:
            bgr = imread_with_exif(path)
            if bgr is None:
                log(f"[跳过] 无法读取：{path.name}")
                continue
            rgb = bgr[..., ::-1].copy()
            out = _rembg_remove(rgb)
            if out.shape[-1] != 4:
                log(f"[跳过] rembg 返回非 RGBA：{path.name}")
                continue
            out_bgr = out.copy()
            out_bgr[..., :3] = out[..., :3][..., ::-1]
            alpha_u8 = out_bgr[..., 3]
            ys, xs = np.where(alpha_u8 > 32)
            if len(xs) == 0:
                log(f"[跳过] {path.name}：rembg 输出全透明")
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            pad = 2
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(out_bgr.shape[1], x2 + pad), min(out_bgr.shape[0], y2 + pad)
            crop = out_bgr[y1:y2, x1:x2, :3].copy()
            alpha_crop = (alpha_u8[y1:y2, x1:x2].astype(np.float32) / 255.0)
            poly = alpha_to_polygon(alpha_u8[y1:y2, x1:x2], polygon_simplify_eps)
            assets.append(ObjectAsset(
                label="auto",
                class_id=auto_id,
                image=crop,
                alpha=alpha_crop,
                source_json=path,
                source_shape_index=0,
                polygon=poly,
            ))
        except Exception as exc:
            log(f"[跳过] {path.name}：{exc}")
            continue
    if not assets:
        raise RuntimeError(t("assets.auto_failed", path=objects_dir))
    log(f"自动抠图完成：{len(assets)} / {len(image_paths)} 张（class_id={auto_id}）")
    return assets
