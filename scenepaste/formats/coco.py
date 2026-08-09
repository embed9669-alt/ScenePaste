"""COCO instances writer (high throughput, checkpointable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..core.geometry import canvas_polygon, mask_bbox, mask_to_polygons


class CocoWriter:
    """High-throughput COCO ``instances`` writer.

    - Image / annotation ids come from O(1) counters (no per-call scan of
      historical annotations).
    - ``add_*`` only mutates in memory — no full-file rewrite per image.
    - Optional ``checkpoint_interval`` flushes atomically every N images.
    - :meth:`finalize` writes the complete JSON exactly once.
    """

    def __init__(self, path: Path, categories: Optional[List[dict]] = None,
                 checkpoint_interval: int = 0):
        self.path = Path(path)
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self._since_checkpoint = 0
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.images = list(data.get("images", []))
                self.annotations = list(data.get("annotations", []))
                self.categories = list(data.get("categories", []))
            except Exception:
                self.images, self.annotations, self.categories = [], [], []
        else:
            self.images, self.annotations, self.categories = [], [], []
        self._next_image_id = max((int(i.get("id", 0)) for i in self.images), default=0) + 1
        self._next_annotation_id = max((int(a.get("id", 0)) for a in self.annotations), default=0) + 1
        if categories:
            existing = {c.get("name"): c.get("id") for c in self.categories}
            for cat in categories:
                if cat.get("name") not in existing:
                    self.categories.append(dict(cat))

    def add_image(self, file_name: str, width: int, height: int, **extra) -> int:
        """Register an image and return its id."""
        img_id = self._next_image_id
        self._next_image_id += 1
        entry = {"id": img_id, "file_name": file_name,
                 "width": int(width), "height": int(height)}
        entry.update(extra)
        self.images.append(entry)
        self._since_checkpoint += 1
        self.maybe_checkpoint()
        return img_id

    def add_annotation(self, image_id: int, category_id: int,
                       segmentation: List[List[float]], bbox: List[float],
                       area: float, **extra) -> int:
        """Register one annotation and return its id."""
        ann_id = self._next_annotation_id
        self._next_annotation_id += 1
        entry = {
            "id": ann_id,
            "image_id": int(image_id),
            "category_id": int(category_id),
            "segmentation": segmentation,
            "area": float(area),
            "bbox": [float(v) for v in bbox],
            "iscrowd": 0,
        }
        entry.update(extra)
        self.annotations.append(entry)
        return ann_id

    def ensure_category(self, name: str, class_id: int) -> int:
        """Return the id of an existing category or add a new one."""
        for c in self.categories:
            if c.get("name") == name:
                return int(c.get("id", class_id))
        self.categories.append({"id": int(class_id), "name": name,
                                "supercategory": "object"})
        return int(class_id)

    def maybe_checkpoint(self) -> None:
        if self.checkpoint_interval and self._since_checkpoint >= self.checkpoint_interval:
            self.flush()
            self._since_checkpoint = 0

    def flush(self) -> None:
        """Atomically write the current COCO state (crash-safe)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
        tmp.replace(self.path)

    def finalize(self) -> None:
        self.flush()
        self._since_checkpoint = 0


def append_coco(output_dir: Path, bg_path: Path, image_path: Path,
                stem: str, width: int, height: int, annotations,
                writer: Optional[CocoWriter] = None,
                visible_masks: Optional[Sequence[np.ndarray]] = None) -> CocoWriter:
    """Append one composed image to a COCO writer.

    When ``visible_masks`` is provided, segmentation/bbox/area are derived
    from the final visible pixels — so they match the real occlusion
    relationships produced by z-order compositing. The writer is reused
    across an entire run (no per-image JSON rewrite).
    """
    owned_writer = writer is None
    if writer is None:
        writer = CocoWriter(output_dir / "instances_coco.json")
    for asset, _box, _flipped, _transform in annotations:
        writer.ensure_category(asset.label, asset.class_id)
    img_id = writer.add_image(
        file_name=str(Path("images/train") / image_path.name),
        width=width, height=height, background=str(bg_path), stem=stem,
    )
    for idx, (asset, box, flipped, transform) in enumerate(annotations):
        x1, y1, x2, y2 = box
        segmentation: List[List[float]] = []
        area = 0.0
        coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if visible_masks is not None and idx < len(visible_masks):
            mask = visible_masks[idx]
            polys = mask_to_polygons(mask)
            segmentation = [[float(v) for v in poly.reshape(-1)] for poly in polys]
            area = float(np.count_nonzero(mask))
            mb = mask_bbox(mask)
            if mb is not None:
                mx1, my1, mx2, my2 = mb
                coco_box = [float(mx1), float(my1), float(mx2 - mx1), float(my2 - my1)]
        if not segmentation:
            poly = canvas_polygon(asset, box, flipped, width, height, transform)
            if poly is None:
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            segmentation = [[float(v) for v in poly.reshape(-1)]]
            area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                             - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
        writer.add_annotation(
            image_id=img_id,
            category_id=int(asset.class_id),
            segmentation=segmentation,
            bbox=coco_box,
            area=float(area),
        )
    if owned_writer:
        writer.finalize()
    return writer
