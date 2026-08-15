"""Learn and sample compact real-dataset distributions.

ScenePaste distribution profiles intentionally keep *statistics*, not source
images.  A profile can therefore be committed to a project and reused by a
large generation run without carrying the original dataset around.

Supported learning inputs:
- LabelMe folders
- YOLO detect / segmentation / OBB datasets
- COCO instances JSON
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .models import IMAGE_SUFFIXES

SCHEMA = "scenepaste/distribution-profile"
VERSION = 1


class _Histogram:
    def __init__(self, bins: int, lo: float, hi: float):
        self.bins = int(max(2, bins))
        self.lo = float(lo)
        self.hi = float(hi)
        self.counts = [0] * self.bins

    def observe(self, value: float) -> None:
        if not math.isfinite(value):
            return
        v = min(self.hi - 1e-12, max(self.lo, float(value)))
        t = (v - self.lo) / max(1e-12, self.hi - self.lo)
        idx = min(self.bins - 1, max(0, int(t * self.bins)))
        self.counts[idx] += 1

    def as_dict(self) -> dict:
        return {"bins": self.bins, "range": [self.lo, self.hi], "counts": self.counts}


class _ClassAccumulator:
    def __init__(self, bins: int):
        self.count = 0
        self.center_x = _Histogram(bins, 0.0, 1.0)
        self.center_y = _Histogram(bins, 0.0, 1.0)
        self.bottom_y = _Histogram(bins, 0.0, 1.0)
        self.width = _Histogram(bins, 0.0, 1.0)
        self.height = _Histogram(bins, 0.0, 1.0)
        self.area = _Histogram(bins, 0.0, 1.0)
        self.aspect = _Histogram(bins, 0.0, 5.0)
        self.overlap_iou = _Histogram(bins, 0.0, 1.0)
        self.visible_shape_fraction = _Histogram(bins, 0.0, 1.0)

    def observe(self, cx: float, cy: float, w: float, h: float) -> None:
        w = max(1e-8, float(w))
        h = max(1e-8, float(h))
        self.count += 1
        self.center_x.observe(cx)
        self.center_y.observe(cy)
        self.bottom_y.observe(cy + h / 2.0)
        self.width.observe(w)
        self.height.observe(h)
        self.area.observe(w * h)
        self.aspect.observe(w / h)

    def observe_context(self, overlap_iou: float = 0.0, visible_shape_fraction: Optional[float] = None) -> None:
        self.overlap_iou.observe(overlap_iou)
        if visible_shape_fraction is not None:
            self.visible_shape_fraction.observe(visible_shape_fraction)

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "center_x": self.center_x.as_dict(),
            "center_y": self.center_y.as_dict(),
            "bottom_y": self.bottom_y.as_dict(),
            "width": self.width.as_dict(),
            "height": self.height.as_dict(),
            "area": self.area.as_dict(),
            "aspect_ratio": self.aspect.as_dict(),
            "overlap_iou": self.overlap_iou.as_dict(),
            "visible_shape_fraction": self.visible_shape_fraction.as_dict(),
        }


def _weighted_choice(rng: random.Random, items: List[Tuple[object, float]]):
    total = sum(max(0.0, float(w)) for _x, w in items)
    if total <= 0:
        return items[rng.randrange(len(items))][0]
    r = rng.random() * total
    upto = 0.0
    for value, weight in items:
        upto += max(0.0, float(weight))
        if r <= upto:
            return value
    return items[-1][0]


def _sample_hist(hist: Mapping[str, object], rng: random.Random, fallback: float) -> float:
    counts = [int(x) for x in hist.get("counts", [])]  # type: ignore[arg-type]
    if not counts or sum(counts) <= 0:
        return float(fallback)
    lo, hi = [float(x) for x in hist.get("range", [0.0, 1.0])]  # type: ignore[arg-type]
    idx = int(_weighted_choice(rng, [(i, c) for i, c in enumerate(counts)]))
    width = (hi - lo) / len(counts)
    return lo + (idx + rng.random()) * width


@dataclass
class DistributionProfile:
    """Compact learned distribution used by the generation planner."""

    data: dict

    @classmethod
    def load(cls, path: Path) -> "DistributionProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"不是 ScenePaste distribution profile: {path}")
        return cls(data)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @property
    def classes(self) -> Mapping[str, dict]:
        return self.data.get("classes", {})

    def matching_labels(self, class_map: Mapping[str, int]) -> List[str]:
        return [name for name in class_map if name in self.classes and self.classes[name].get("count", 0) > 0]

    def sample_object_count(self, rng: random.Random, minimum: int, maximum: int) -> int:
        rows = self.data.get("object_count", {}).get("counts", {})
        candidates = []
        for key, count in rows.items():
            try:
                n = int(key)
            except (TypeError, ValueError):
                continue
            if minimum <= n <= maximum and int(count) > 0:
                candidates.append((n, int(count)))
        if not candidates:
            return rng.randint(minimum, maximum)
        return int(_weighted_choice(rng, candidates))

    def sample_label(self, rng: random.Random, class_map: Mapping[str, int]) -> Optional[str]:
        labels = self.matching_labels(class_map)
        if not labels:
            return None
        weights = [(name, float(self.classes[name].get("count", 0))) for name in labels]
        return str(_weighted_choice(rng, weights))

    def sample_placement(self, label: str, rng: random.Random) -> dict:
        row = self.classes.get(label, {})
        overlap = min(1.0, max(0.0, _sample_hist(row.get("overlap_iou", {}), rng, 0.0)))
        visible = min(1.0, max(0.0, _sample_hist(row.get("visible_shape_fraction", {}), rng, 1.0)))
        return {
            "center_x_ratio": min(0.99, max(0.01, _sample_hist(row.get("center_x", {}), rng, 0.5))),
            "bottom_y_ratio": min(0.995, max(0.01, _sample_hist(row.get("bottom_y", {}), rng, 0.75))),
            "height_ratio": min(0.90, max(0.01, _sample_hist(row.get("height", {}), rng, 0.20))),
            "overlap_iou": overlap,
            "visible_shape_fraction": visible,
            "allow_overlap": bool(overlap >= 0.05 or visible < 0.85),
        }


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1); bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    return inter / max(1e-12, aa + bb - inter)


def _polygon_area(xs, ys) -> float:
    if len(xs) < 3:
        return 0.0
    return 0.5 * abs(sum(xs[i] * ys[(i+1)%len(xs)] - ys[i] * xs[(i+1)%len(xs)] for i in range(len(xs))))


def _observe_context(entries, accum: Dict[str, _ClassAccumulator]) -> None:
    # entries: (label, bbox_xyxy_normalized, optional visible-shape fraction)
    for i, (label, box, visible_fraction) in enumerate(entries):
        max_iou = 0.0
        for j, (_label2, box2, _vf2) in enumerate(entries):
            if i != j:
                max_iou = max(max_iou, _bbox_iou(box, box2))
        accum[label].observe_context(max_iou, visible_fraction)


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        h, w = arr.shape[:2]
        return int(w), int(h)
    except Exception:
        return None


def _classes_txt(root: Path) -> Dict[int, str]:
    path = root / "classes.txt"
    if not path.exists():
        return {}
    result: Dict[int, str] = {}
    for fallback_id, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            try:
                cid = int(left.strip())
                result[cid] = right.strip() or str(cid)
                continue
            except ValueError:
                pass
        result[fallback_id] = line
    return result


def _observe_yolo(root: Path, bins: int, accum: Dict[str, _ClassAccumulator], object_counts: Dict[int, int], geometry_source: str = "auto") -> int:
    names = _classes_txt(root)
    total_images = 0
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        candidates = {
            "detect": root / "labels" / split,
            "seg": root / "labels-seg" / split,
            "obb": root / "labels-obb" / split,
        }
        if geometry_source == "auto":
            label_dir = next((candidates[k] for k in ("seg", "obb", "detect") if candidates[k].exists()), candidates["detect"])
        else:
            label_dir = candidates.get(geometry_source, candidates["detect"])
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            if _image_size(image_path) is None:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            entries = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    tok = line.split()
                    if not tok:
                        continue
                    try:
                        cid = int(float(tok[0])); vals = [float(v) for v in tok[1:]]
                    except ValueError:
                        continue
                    visible_fraction = None
                    if len(tok) == 5:  # detect: cx cy w h
                        cx, cy, bw, bh = vals[:4]
                    elif len(tok) == 9:  # OBB: four corners
                        xs = vals[0::2]; ys = vals[1::2]
                        cx, cy = (min(xs)+max(xs))/2.0, (min(ys)+max(ys))/2.0
                        bw, bh = max(xs)-min(xs), max(ys)-min(ys)
                        visible_fraction = min(1.0, _polygon_area(xs, ys) / max(1e-12, bw * bh))
                    elif len(tok) >= 7 and len(tok) % 2 == 1:  # seg polygon
                        xs = vals[0::2]; ys = vals[1::2]
                        if not xs or not ys:
                            continue
                        cx, cy = (min(xs)+max(xs))/2.0, (min(ys)+max(ys))/2.0
                        bw, bh = max(xs)-min(xs), max(ys)-min(ys)
                        visible_fraction = min(1.0, _polygon_area(xs, ys) / max(1e-12, bw * bh))
                    else:
                        continue
                    label = names.get(cid, str(cid))
                    accum.setdefault(label, _ClassAccumulator(bins)).observe(cx, cy, bw, bh)
                    entries.append((label, (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), visible_fraction))
            _observe_context(entries, accum)
            object_counts[len(entries)] = object_counts.get(len(entries), 0) + 1
            total_images += 1
    return total_images


def _observe_coco(root: Path, bins: int, accum: Dict[str, _ClassAccumulator], object_counts: Dict[int, int]) -> int:
    p = root / "instances_coco.json"
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    images = {int(i["id"]): i for i in data.get("images", []) if "id" in i}
    cats = {int(c["id"]): str(c.get("name", c["id"])) for c in data.get("categories", []) if "id" in c}
    grouped: Dict[int, list] = {iid: [] for iid in images}
    for ann in data.get("annotations", []):
        try:
            iid = int(ann["image_id"]); image = images[iid]
            iw, ih = float(image["width"]), float(image["height"])
            x, y, w, h = [float(v) for v in ann["bbox"]]
            label = cats.get(int(ann["category_id"]), str(ann["category_id"]))
        except Exception:
            continue
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            continue
        cx, cy, bw, bh = (x+w/2)/iw, (y+h/2)/ih, w/iw, h/ih
        visible_fraction = None
        try:
            area = float(ann.get("area", 0.0))
            if area > 0:
                visible_fraction = min(1.0, area / max(1e-12, w*h))
        except Exception:
            pass
        accum.setdefault(label, _ClassAccumulator(bins)).observe(cx, cy, bw, bh)
        grouped.setdefault(iid, []).append((label, (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), visible_fraction))
    for iid in images:
        entries = grouped.get(iid, [])
        _observe_context(entries, accum)
        object_counts[len(entries)] = object_counts.get(len(entries), 0) + 1
    return len(images)


def _observe_labelme(root: Path, bins: int, accum: Dict[str, _ClassAccumulator], object_counts: Dict[int, int]) -> int:
    total_images = 0
    for jp in sorted(root.rglob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Ignore ScenePaste configs/templates that may live next to data.
        if "shapes" not in data:
            continue
        iw = int(data.get("imageWidth") or 0); ih = int(data.get("imageHeight") or 0)
        if iw <= 0 or ih <= 0:
            image_path = jp.parent / str(data.get("imagePath", ""))
            size = _image_size(image_path) if image_path.exists() else None
            if size: iw, ih = size
        if iw <= 0 or ih <= 0:
            continue
        entries = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip(); pts = shape.get("points", [])
            if not label or not isinstance(pts, list) or len(pts) < 2:
                continue
            try:
                xs_px = [float(p[0]) for p in pts]; ys_px = [float(p[1]) for p in pts]
            except Exception:
                continue
            x1, x2, y1, y2 = min(xs_px), max(xs_px), min(ys_px), max(ys_px)
            w, h = max(1e-6, x2-x1), max(1e-6, y2-y1)
            cx, cy, bw, bh = ((x1+x2)/2)/iw, ((y1+y2)/2)/ih, w/iw, h/ih
            visible_fraction = None
            if len(pts) >= 3:
                visible_fraction = min(1.0, _polygon_area(xs_px, ys_px) / max(1e-12, w*h))
            accum.setdefault(label, _ClassAccumulator(bins)).observe(cx, cy, bw, bh)
            entries.append((label, (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), visible_fraction))
        _observe_context(entries, accum)
        object_counts[len(entries)] = object_counts.get(len(entries), 0) + 1
        total_images += 1
    return total_images


def learn_distribution_profile(root: Path, bins: int = 20, geometry_source: str = "auto") -> DistributionProfile:
    """Learn a compact placement/class/count profile from ``root``."""
    root = Path(root)
    geometry_source = str(geometry_source).lower()
    if geometry_source not in {"auto", "detect", "seg", "obb"}:
        raise ValueError("geometry_source must be auto / detect / seg / obb")
    if not root.exists():
        raise FileNotFoundError(root)
    accum: Dict[str, _ClassAccumulator] = {}
    object_counts: Dict[int, int] = {}
    source_type = "labelme"
    if (root / "instances_coco.json").exists():
        source_type = "coco"
        image_count = _observe_coco(root, bins, accum, object_counts)
    elif (root / "images").exists() and (root / "labels").exists():
        source_type = "yolo"
        image_count = _observe_yolo(root, bins, accum, object_counts, geometry_source=geometry_source)
    else:
        image_count = _observe_labelme(root, bins, accum, object_counts)
    total_objects = sum(a.count for a in accum.values())
    if image_count <= 0 or total_objects <= 0:
        raise RuntimeError(f"未能从 {root} 学到有效目标分布")
    classes = {name: acc.as_dict() for name, acc in sorted(accum.items())}
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "source": str(root.resolve()),
        "source_type": source_type,
        "geometry_source": geometry_source,
        "image_count": int(image_count),
        "object_count_total": int(total_objects),
        "object_count": {"counts": {str(k): int(v) for k, v in sorted(object_counts.items())}},
        "classes": classes,
        "class_probabilities": {name: row["count"] / total_objects for name, row in classes.items()},
        "notes": "Normalized geometry statistics. overlap_iou is a crowding/occlusion proxy; visible_shape_fraction is available for polygon/mask annotations.",
    }
    return DistributionProfile(payload)



def learn_yolo_profile_subset(root: Path, stems, bins: int = 20, geometry_source: str = "auto") -> DistributionProfile:
    """Learn a distribution profile from selected YOLO stems.

    Detect labels are preferred. If they are absent, ScenePaste falls back to
    labels-seg or labels-obb so Seg/OBB hard-mining can still create a reusable
    generation profile.
    """
    root = Path(root)
    wanted = {str(x) for x in stems}
    if not wanted:
        raise RuntimeError("hard-example subset is empty")
    names = _classes_txt(root)
    accum: Dict[str, _ClassAccumulator] = {}
    object_counts: Dict[int, int] = {}
    image_count = 0
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        dirs = {"detect": root / "labels" / split, "seg": root / "labels-seg" / split, "obb": root / "labels-obb" / split}
        if geometry_source == "auto":
            label_dir = next((dirs[k] for k in ("seg", "obb", "detect") if dirs[k].exists()), dirs["detect"])
        else:
            label_dir = dirs.get(geometry_source, dirs["detect"])
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            if image_path.stem not in wanted:
                continue
            entries = []
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    tok = line.split()
                    if not tok:
                        continue
                    try:
                        cid = int(float(tok[0])); vals = [float(v) for v in tok[1:]]
                    except ValueError:
                        continue
                    visible_fraction = None
                    if len(tok) == 5:
                        cx, cy, bw, bh = vals[:4]
                    elif len(tok) == 9:
                        xs, ys = vals[0::2], vals[1::2]
                        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                        bw, bh = max(xs)-min(xs), max(ys)-min(ys)
                        visible_fraction = min(1.0, _polygon_area(xs, ys) / max(1e-12, bw*bh))
                    elif len(tok) >= 7 and len(tok) % 2 == 1:
                        xs, ys = vals[0::2], vals[1::2]
                        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                        bw, bh = max(xs)-min(xs), max(ys)-min(ys)
                        visible_fraction = min(1.0, _polygon_area(xs, ys) / max(1e-12, bw*bh))
                    else:
                        continue
                    label = names.get(cid, str(cid))
                    accum.setdefault(label, _ClassAccumulator(bins)).observe(cx, cy, bw, bh)
                    entries.append((label, (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), visible_fraction))
            _observe_context(entries, accum)
            object_counts[len(entries)] = object_counts.get(len(entries), 0) + 1
            image_count += 1
    total_objects = sum(a.count for a in accum.values())
    if image_count <= 0 or total_objects <= 0:
        raise RuntimeError("selected hard examples contain no usable YOLO ground-truth objects")
    classes = {name: acc.as_dict() for name, acc in sorted(accum.items())}
    payload = {
        "schema": SCHEMA, "version": VERSION, "source": str(root.resolve()),
        "source_type": "yolo-hard-subset", "image_count": int(image_count),
        "object_count_total": int(total_objects),
        "object_count": {"counts": {str(k): int(v) for k, v in sorted(object_counts.items())}},
        "classes": classes,
        "class_probabilities": {name: row["count"] / total_objects for name, row in classes.items()},
        "notes": "Learned from model hard examples selected by ScenePaste hardmine.",
        "selected_stems": sorted(wanted),
    }
    return DistributionProfile(payload)

def _normalized_counts(row: Mapping[str, object]) -> List[float]:
    counts = np.asarray(row.get("counts", []), dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * len(counts)
    return (counts / total).tolist()


def compare_profiles(a: DistributionProfile, b: DistributionProfile) -> dict:
    """Return simple bounded distances (0 is identical, 1 is very different)."""
    labels = sorted(set(a.classes) | set(b.classes))
    ac = {k: float(a.classes.get(k, {}).get("count", 0)) for k in labels}
    bc = {k: float(b.classes.get(k, {}).get("count", 0)) for k in labels}
    at, bt = sum(ac.values()) or 1.0, sum(bc.values()) or 1.0
    class_tv = 0.5 * sum(abs(ac[k] / at - bc[k] / bt) for k in labels)
    metrics = []
    metric_by_name: Dict[str, List[float]] = {}
    for label in labels:
        if label not in a.classes or label not in b.classes:
            continue
        for key in ("center_x", "center_y", "bottom_y", "width", "height", "area", "aspect_ratio", "overlap_iou", "visible_shape_fraction"):
            pa = _normalized_counts(a.classes[label].get(key, {}))
            pb = _normalized_counts(b.classes[label].get(key, {}))
            if len(pa) == len(pb) and pa:
                dist = 0.5 * sum(abs(x - y) for x, y in zip(pa, pb))
                metrics.append(dist)
                metric_by_name.setdefault(key, []).append(dist)
    acounts = a.data.get("object_count", {}).get("counts", {})
    bcounts = b.data.get("object_count", {}).get("counts", {})
    count_keys = sorted(set(acounts) | set(bcounts), key=lambda x: int(x) if str(x).isdigit() else str(x))
    at_count = sum(max(0.0, float(acounts.get(k, 0))) for k in count_keys) or 1.0
    bt_count = sum(max(0.0, float(bcounts.get(k, 0))) for k in count_keys) or 1.0
    object_count_tv = 0.5 * sum(abs(float(acounts.get(k, 0)) / at_count - float(bcounts.get(k, 0)) / bt_count) for k in count_keys)
    return {
        "class_total_variation": float(class_tv),
        "object_count_total_variation": float(object_count_tv),
        "placement_histogram_distance": float(sum(metrics) / len(metrics)) if metrics else None,
        "metric_distances": {k: float(sum(v) / len(v)) for k, v in sorted(metric_by_name.items()) if v},
        "matched_classes": len([k for k in labels if k in a.classes and k in b.classes]),
    }


def _normalized_hist_counts(hist: Mapping[str, object]) -> np.ndarray:
    counts = np.asarray(hist.get("counts", []), dtype=np.float64)
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def mix_distribution_profiles(
    profiles: List[DistributionProfile],
    weights: Optional[List[float]] = None,
    pseudo_count: int = 100000,
) -> DistributionProfile:
    """Blend multiple learned domains into one reusable profile.

    Each input profile is normalized before mixing, so a very large source
    dataset does not dominate a smaller domain unless its explicit weight is
    larger. Histograms must use compatible binning (the default learner does).
    """
    if not profiles:
        raise ValueError("至少需要一个 distribution profile")
    if weights is None:
        weights = [1.0] * len(profiles)
    if len(weights) != len(profiles):
        raise ValueError("weights 数量必须与 profiles 一致")
    w = np.asarray([max(0.0, float(x)) for x in weights], dtype=np.float64)
    if float(w.sum()) <= 0:
        raise ValueError("profile weights 总和必须大于 0")
    w /= w.sum()
    scale = max(1000, int(pseudo_count))

    # Object-count distribution is mixed per source image distribution.
    object_probs: Dict[int, float] = {}
    for profile, weight in zip(profiles, w):
        rows = profile.data.get("object_count", {}).get("counts", {})
        total = sum(max(0.0, float(v)) for v in rows.values()) or 1.0
        for key, count in rows.items():
            try:
                n = int(key)
            except (TypeError, ValueError):
                continue
            object_probs[n] = object_probs.get(n, 0.0) + float(weight) * max(0.0, float(count)) / total

    labels = sorted(set().union(*(set(p.classes) for p in profiles)))
    class_probs: Dict[str, float] = {label: 0.0 for label in labels}
    for profile, weight in zip(profiles, w):
        total = sum(max(0.0, float(row.get("count", 0))) for row in profile.classes.values()) or 1.0
        for label in labels:
            row = profile.classes.get(label)
            if row:
                class_probs[label] += float(weight) * max(0.0, float(row.get("count", 0))) / total

    hist_keys = ("center_x", "center_y", "bottom_y", "width", "height", "area", "aspect_ratio", "overlap_iou", "visible_shape_fraction")
    classes = {}
    for label in labels:
        p_label = class_probs.get(label, 0.0)
        if p_label <= 0:
            continue
        out_row = {"count": max(1, int(round(p_label * scale)))}
        for key in hist_keys:
            template = None
            acc = None
            for profile, weight in zip(profiles, w):
                row = profile.classes.get(label)
                if not row or key not in row:
                    continue
                hist = row[key]
                counts = _normalized_hist_counts(hist)
                if not len(counts):
                    continue
                if template is None:
                    template = hist
                    acc = np.zeros_like(counts, dtype=np.float64)
                elif len(counts) != len(acc) or list(hist.get("range", [])) != list(template.get("range", [])):
                    raise ValueError(f"profile histogram bins 不兼容：{label}/{key}")
                source_total = sum(max(0.0, float(r.get("count", 0))) for r in profile.classes.values()) or 1.0
                source_class_prob = max(0.0, float(row.get("count", 0))) / source_total
                acc += float(weight) * source_class_prob * counts
            if template is not None and acc is not None:
                cond_total = float(acc.sum()) or 1.0
                counts_out = np.rint(acc / cond_total * max(1, out_row["count"])).astype(int).tolist()
                out_row[key] = {
                    "bins": int(template.get("bins", len(counts_out))),
                    "range": list(template.get("range", [0.0, 1.0])),
                    "counts": counts_out,
                }
        classes[label] = out_row

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "source": "mixture",
        "source_type": "mixture",
        "sources": [p.data.get("source", "") for p in profiles],
        "weights": [float(x) for x in w.tolist()],
        "image_count": scale,
        "object_count_total": sum(int(r["count"]) for r in classes.values()),
        "object_count": {"counts": {str(k): max(1, int(round(v * scale))) for k, v in sorted(object_probs.items()) if v > 0}},
        "classes": classes,
        "class_probabilities": {k: float(v) for k, v in class_probs.items() if v > 0},
        "notes": "Weighted mixture of ScenePaste distribution profiles.",
    }
    return DistributionProfile(payload)
