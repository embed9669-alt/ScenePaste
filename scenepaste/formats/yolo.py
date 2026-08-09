"""YOLO-family label writers: detect, seg, OBB, classes.txt, data.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..core.geometry import canvas_polygon, mask_to_polygons


def yolo_line(class_id: int, box: Sequence[int], width: int, height: int) -> str:
    """Format a YOLO detection line: ``class cx cy w h`` (normalized)."""
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def yolo_seg_line(class_id: int, polygon: Optional[np.ndarray],
                  width: int, height: int) -> str:
    """Format a YOLO-segmentation line from a polygon (``class x1 y1 x2 y2 ...``)."""
    if polygon is None or len(polygon) < 3 or width <= 0 or height <= 0:
        return ""
    pts = np.asarray(polygon, dtype=np.float64).copy()
    pts[:, 0] = np.clip(pts[:, 0] / width, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / height, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {coords}"


def ultralytics_obb_line(class_id: int, mask: np.ndarray,
                         width: int, height: int) -> str:
    """Format a Ultralytics YOLO-OBB line from a visible mask.

    Output: ``class x1 y1 x2 y2 x3 y3 x4 y4`` (normalized), with the 4 corner
    points ordered clockwise from the top-most (then left-most) corner so
    output is deterministic and test-friendly.
    """
    ys, xs = np.where(mask)
    if len(xs) < 3 or width <= 0 or height <= 0:
        return ""
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    corners = cv2.boxPoints(rect).astype(np.float64)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    start = int(np.lexsort((corners[:, 0], corners[:, 1]))[0])
    corners = np.roll(corners, -start, axis=0)
    corners[:, 0] = np.clip(corners[:, 0] / float(width), 0.0, 1.0)
    corners[:, 1] = np.clip(corners[:, 1] / float(height), 0.0, 1.0)
    return f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in corners.reshape(-1))


def write_classes_file(class_map: Dict[str, int], output_dir: Path) -> None:
    """Write ``classes.txt`` with ``id: label`` lines ordered by id."""
    names = sorted(class_map.items(), key=lambda item: item[1])
    with (output_dir / "classes.txt").open("w", encoding="utf-8") as file:
        for label, class_id in names:
            file.write(f"{class_id}: {label}\n")


def write_data_yaml(output_dir: Path, class_map: Dict[str, int],
                    val_split: Optional[str] = None) -> Path:
    """Write a Ultralytics-ready ``data.yaml``.

    ``val_split``: ``None`` points val at train (self-validation convenience);
    a path like ``"images/val"`` uses a user-prepared val directory.
    """
    names = sorted(class_map.items(), key=lambda kv: kv[1])
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        f"val: {val_split if val_split else 'images/train'}",
        "",
        "names:",
    ]
    for label, class_id in names:
        lines.append(f"  {class_id}: {label}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def seg_lines_for_annotations(annotations, visible_masks, width: int, height: int) -> List[str]:
    """Build the list of YOLO-seg label lines for one image.

    For each instance the largest external contour of its visible mask is
    used (Ultralytics ``.txt`` is single-polygon per line); falls back to the
    projected local polygon when no contour survives.
    """
    seg_lines: List[str] = []
    for idx, (asset, box, flipped, transform) in enumerate(annotations):
        polys = mask_to_polygons(visible_masks[idx]) if visible_masks is not None else []
        poly = polys[0] if polys else canvas_polygon(asset, box, flipped, width, height, transform)
        line = yolo_seg_line(asset.class_id, poly, width, height)
        if line:
            seg_lines.append(line)
    return seg_lines


def obb_lines_for_annotations(annotations, visible_masks, width: int, height: int) -> List[str]:
    """Build the list of YOLO-OBB label lines for one image."""
    obb_lines: List[str] = []
    for (asset, _box, _flipped, _transform), vis in zip(annotations, visible_masks):
        line = ultralytics_obb_line(asset.class_id, vis, width, height)
        if line:
            obb_lines.append(line)
    return obb_lines
