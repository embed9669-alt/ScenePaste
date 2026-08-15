"""Auto-cutout via rembg (no LabelMe JSON required)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

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


def rembg_mask_and_polygon(
    bgr: np.ndarray,
    *,
    simplify_eps: float = 0.004,
    model_id: str = "rembg:u2net",
):
    """Run rembg (or another registered cutout model) on a BGR image.

    Returns ``(polygon Nx2|None, alpha HxW uint8)``. Polygon coordinates are
    in full image space. Requires ``pip install 'scenepaste[auto]'`` for rembg.
    """
    from .cutout_models import predict_cutout

    return predict_cutout(bgr, model_id, simplify_eps=simplify_eps)


def _ensure_label_id(class_map: Dict[str, int], label: str) -> int:
    label = str(label or "").strip() or "auto"
    if label not in class_map:
        next_id = max(list(class_map.values()) + [-1]) + 1
        class_map[label] = next_id
    return int(class_map[label])


def _label_for_path(
    path: Path,
    objects_dir: Path,
    *,
    default_label: str,
    label_from_subdir: bool,
) -> str:
    """Resolve the class name for one image.

    When ``label_from_subdir`` is True and the file lives under a subdirectory
    of ``objects_dir``, the first path component is used as the label
    (e.g. ``objects/person/a.jpg`` → ``person``). Otherwise ``default_label``.
    """
    if label_from_subdir:
        try:
            rel = path.resolve().relative_to(Path(objects_dir).resolve())
        except ValueError:
            rel = Path(path.name)
        if len(rel.parts) >= 2:
            return str(rel.parts[0]).strip() or default_label
    return default_label


def load_object_assets_auto(
    objects_dir: Path,
    class_map: Dict[str, int],
    log: Callable[[str], None],
    backend: str = "rembg",
    polygon_simplify_eps: float = 0.004,
    *,
    label: Optional[str] = None,
    label_from_subdir: bool = False,
) -> List[ObjectAsset]:
    """Auto-cutout every image in ``objects_dir`` using rembg.

    Class naming:
      * ``label`` — default class for every image (default ``"auto"``).
      * ``label_from_subdir`` — if True, use the first subdirectory name under
        ``objects_dir`` as the class when present.

    Requires the ``rembg`` extra: ``pip install 'scenepaste[auto]'``.
    """
    try:
        from rembg import remove as _rembg_remove
    except ImportError as exc:
        raise ImportError(t("assets.rembg_missing")) from exc

    objects_dir = Path(objects_dir)
    default_label = str(label or "").strip() or "auto"
    image_paths = sorted(
        p for p in objects_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(t("assets.image_dir_empty", path=objects_dir))

    assets: List[ObjectAsset] = []
    total = len(image_paths)
    for index, path in enumerate(image_paths, start=1):
        try:
            class_label = _label_for_path(
                path, objects_dir,
                default_label=default_label,
                label_from_subdir=bool(label_from_subdir),
            )
            class_id = _ensure_label_id(class_map, class_label)
            log(f"自动抠图 [{index}/{total}] {path.name} → {class_label}")
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
                label=class_label,
                class_id=class_id,
                image=crop,
                alpha=alpha_crop,
                source_json=path,
                source_shape_index=0,
                polygon=poly,
                mask_source="rembg",
            ))
        except Exception as exc:
            log(f"[跳过] {path.name}：{exc}")
            continue
    if not assets:
        raise RuntimeError(t("assets.auto_failed", path=objects_dir))
    labels_used = sorted({a.label for a in assets})
    log(
        f"自动抠图完成：{len(assets)} / {len(image_paths)} 张"
        f"（类别: {', '.join(labels_used)}）"
    )
    return assets
