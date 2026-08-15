"""Dataset indexing and annotation preview helpers for ScenePaste.

The module deliberately has no Qt dependency so annotation parsing and overlay
rendering can be tested in headless environments. The desktop window lives in
``compose_app_qt.explorer``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    split: str
    stem: str


@dataclass
class OverlayResult:
    image: Image.Image
    format_name: str
    object_count: int
    notes: List[str]


def index_dataset(root: Path) -> List[DatasetImage]:
    """Return images under ``images/{train,val,test}`` in deterministic order."""
    root = Path(root)
    items: List[DatasetImage] = []
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                items.append(DatasetImage(path=path, split=split, stem=path.stem))
    return items


def _read_classes(root: Path) -> Dict[int, str]:
    classes: Dict[int, str] = {}
    path = Path(root) / "classes.txt"
    if not path.exists():
        return classes
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        raw_id, name = line.split(":", 1)
        try:
            classes[int(raw_id.strip())] = name.strip()
        except ValueError:
            continue
    return classes


def _class_color(class_id: int) -> Tuple[int, int, int]:
    """Stable high-contrast RGB colour without maintaining a global palette."""
    i = int(class_id)
    return ((37 * i + 67) % 206 + 32, (83 * i + 113) % 206 + 32, (149 * i + 41) % 206 + 32)


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, color) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
        draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=color)
    except Exception:
        pass
    draw.text((x, y), text, fill=(255, 255, 255))


def _parse_label_lines(path: Path) -> List[Tuple[int, List[float]]]:
    rows: List[Tuple[int, List[float]]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        rows.append((class_id, values))
    return rows


def _render_yolo(base: Image.Image, rows: Iterable[Tuple[int, List[float]]], classes: Dict[int, str]) -> Tuple[str, int]:
    draw = ImageDraw.Draw(base, "RGBA")
    w, h = base.size
    fmt = "unknown"
    count = 0
    for class_id, values in rows:
        color = _class_color(class_id)
        label = classes.get(class_id, str(class_id))
        if len(values) == 4:  # detect
            fmt = "YOLO Detect"
            cx, cy, bw, bh = values
            x1 = int(round((cx - bw / 2) * w)); y1 = int(round((cy - bh / 2) * h))
            x2 = int(round((cx + bw / 2) * w)); y2 = int(round((cy + bh / 2) * h))
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
            _draw_label(draw, (x1 + 2, max(0, y1 - 16)), label, color + (220,))
        elif len(values) == 8:  # obb four normalized corners
            fmt = "YOLO OBB"
            pts = [(int(round(values[i] * w)), int(round(values[i + 1] * h))) for i in range(0, 8, 2)]
            draw.polygon(pts, outline=color + (255,))
            draw.line(pts + [pts[0]], fill=color + (255,), width=3)
            _draw_label(draw, pts[0], label, color + (220,))
        elif len(values) >= 6 and len(values) % 2 == 0:  # segmentation polygon
            fmt = "YOLO Segmentation"
            pts = [(int(round(values[i] * w)), int(round(values[i + 1] * h))) for i in range(0, len(values), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=color + (65,), outline=color + (255,))
                draw.line(pts + [pts[0]], fill=color + (255,), width=2)
                _draw_label(draw, pts[0], label, color + (220,))
        else:
            continue
        count += 1
    return fmt, count


@lru_cache(maxsize=8)
def _load_coco_cached(path_text: str, mtime_ns: int) -> Tuple[Dict[str, List[dict]], Dict[int, str]]:
    # ``mtime_ns`` is intentionally part of the cache key so refreshing a
    # dataset after generation invalidates stale COCO indexes automatically.
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    names = {int(c.get("id", -1)): str(c.get("name", c.get("id", "?"))) for c in data.get("categories", [])}
    image_names = {int(i.get("id", -1)): Path(str(i.get("file_name", ""))).name for i in data.get("images", [])}
    by_file: Dict[str, List[dict]] = {}
    for ann in data.get("annotations", []):
        filename = image_names.get(int(ann.get("image_id", -1)))
        if filename:
            by_file.setdefault(filename, []).append(ann)
    return by_file, names


def _load_coco(root: Path) -> Tuple[Dict[str, List[dict]], Dict[int, str]]:
    path = Path(root) / "instances_coco.json"
    if not path.exists():
        return {}, {}
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}, {}
    return _load_coco_cached(str(path.resolve()), mtime_ns)


def _render_coco(base: Image.Image, annotations: List[dict], names: Dict[int, str]) -> int:
    draw = ImageDraw.Draw(base, "RGBA")
    count = 0
    for ann in annotations:
        cid = int(ann.get("category_id", -1))
        color = _class_color(cid)
        label = names.get(cid, str(cid))
        seg = ann.get("segmentation", [])
        first_point: Optional[Tuple[int, int]] = None
        if isinstance(seg, list):
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = [(int(round(poly[i])), int(round(poly[i + 1]))) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=color + (60,), outline=color + (255,))
                    if first_point is None:
                        first_point = pts[0]
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, bw, bh = [int(round(float(v))) for v in bbox]
            draw.rectangle((x, y, x + bw, y + bh), outline=color + (255,), width=2)
            if first_point is None:
                first_point = (x, y)
        if first_point is not None:
            _draw_label(draw, first_point, label, color + (220,))
        count += 1
    return count


def _overlay_semantic(base: Image.Image, mask_path: Path) -> Tuple[Image.Image, int]:
    raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return base, 0
    if raw.ndim > 2:
        raw = raw[..., 0]
    if raw.shape[1] != base.width or raw.shape[0] != base.height:
        raw = cv2.resize(raw, base.size, interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    values = [int(v) for v in np.unique(raw).tolist() if int(v) != 0]
    for value in values:
        color = _class_color(value - 1)
        m = raw == value
        overlay[m, 0] = color[0]; overlay[m, 1] = color[1]; overlay[m, 2] = color[2]; overlay[m, 3] = 90
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB"), len(values)


def render_dataset_image(root: Path, item: DatasetImage, *, show_semantic: bool = True) -> OverlayResult:
    """Render all annotations ScenePaste can discover for ``item``."""
    root = Path(root)
    base = Image.open(item.path).convert("RGB")
    notes: List[str] = []
    classes = _read_classes(root)
    modality_counts: List[int] = []
    format_names: List[str] = []

    label_path = root / "labels" / item.split / f"{item.stem}.txt"
    rows = _parse_label_lines(label_path)
    if rows:
        fmt, n = _render_yolo(base, rows, classes)
        modality_counts.append(n)
        if fmt != "unknown":
            format_names.append(fmt)

    seg_path = root / "labels-seg" / item.split / f"{item.stem}.txt"
    seg_rows = _parse_label_lines(seg_path)
    if seg_rows:
        fmt, n = _render_yolo(base, seg_rows, classes)
        modality_counts.append(n)
        format_names.append("YOLO Segmentation (extra)" if fmt != "unknown" else "labels-seg")

    obb_path = root / "labels-obb" / item.split / f"{item.stem}.txt"
    obb_rows = _parse_label_lines(obb_path)
    if obb_rows:
        fmt, n = _render_yolo(base, obb_rows, classes)
        modality_counts.append(n)
        format_names.append("YOLO OBB (extra)" if fmt != "unknown" else "labels-obb")

    coco_by_file, coco_names = _load_coco(root)
    coco_anns = coco_by_file.get(item.path.name, [])
    if coco_anns:
        modality_counts.append(_render_coco(base, coco_anns, coco_names))
        format_names.append("COCO Instances")

    if show_semantic:
        mask_path = root / "masks" / item.split / f"{item.stem}.png"
        if mask_path.exists():
            base, semantic_classes = _overlay_semantic(base, mask_path)
            format_names.append("Semantic Mask")
            notes.append(f"semantic values: {semantic_classes}")

    if not format_names:
        format_names.append("No annotations found")
    object_count = max(modality_counts, default=0)
    return OverlayResult(base, " + ".join(dict.fromkeys(format_names)), object_count, notes)
