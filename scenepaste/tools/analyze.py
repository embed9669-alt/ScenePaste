#!/usr/bin/env python3
"""Dataset QA / statistics for ScenePaste.

Recognizes:
- YOLO detect labels (5 tokens/line)
- Ultralytics YOLO-OBB labels (9 tokens/line)
- YOLO segmentation labels (>=7 tokens/line, even coordinate count)
- semantic masks (masks/train|val)
- COCO instances_coco.json
- LabelMe JSON source datasets

Usage:
    scenepaste analyze ./generated
    scenepaste analyze ./generated --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for split in ("train", "val", "test"):
        d = root / "images" / split
        if d.exists():
            out.extend(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(out)


def _class_names(root: Path) -> Dict[int, str]:
    p = root / "classes.txt"
    result: Dict[int, str] = {}
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        cid, name = line.split(":", 1)
        try:
            result[int(cid.strip())] = name.strip()
        except ValueError:
            pass
    return result


def _class_imbalance(counter: Counter) -> Optional[float]:
    vals = [v for v in counter.values() if v > 0]
    if len(vals) < 2:
        return None
    return max(vals) / max(1, min(vals))


def _parse_yolo_file(path: Path) -> Tuple[str, Counter, int, int]:
    """Return (kind, class_counts, objects, invalid_lines)."""
    counts = Counter()
    objects = 0
    invalid = 0
    kinds = Counter()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "unknown", counts, 0, 1
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            invalid += 1
            continue
        counts[cid] += 1
        objects += 1
        if len(parts) == 5:
            kinds["detect"] += 1
            cx, cy, w, h = vals
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid += 1
        elif len(parts) == 9:
            kinds["obb"] += 1
            if any(v < 0 or v > 1 for v in vals):
                invalid += 1
        elif len(parts) >= 7 and len(vals) % 2 == 0:
            kinds["seg"] += 1
            if any(v < 0 or v > 1 for v in vals):
                invalid += 1
        else:
            kinds["unknown"] += 1
            invalid += 1
    kind = kinds.most_common(1)[0][0] if kinds else "empty"
    return kind, counts, objects, invalid


def _analyze_yolo_dir(root: Path) -> dict:
    details = {}
    total_objects = 0
    all_classes = Counter()
    invalid = 0
    label_files = 0
    kinds = Counter()
    missing_labels = 0

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = [p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES] if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        stems = {p.stem for p in labels}
        missing = sum(1 for p in images if p.stem not in stems)
        missing_labels += missing
        split_objects = 0
        split_classes = Counter()
        split_invalid = 0
        for lp in labels:
            kind, cc, n, bad = _parse_yolo_file(lp)
            kinds[kind] += 1
            split_classes.update(cc)
            split_objects += n
            split_invalid += bad
        details[split] = {
            "images": len(images), "labels": len(labels), "objects": split_objects,
            "invalid_lines": split_invalid, "missing_labels": missing,
            "by_class_id": dict(sorted(split_classes.items())),
        }
        label_files += len(labels)
        total_objects += split_objects
        all_classes.update(split_classes)
        invalid += split_invalid

    return {
        "present": label_files > 0,
        "dominant_format": kinds.most_common(1)[0][0] if kinds else None,
        "label_files": label_files,
        "total_objects": total_objects,
        "invalid_lines": invalid,
        "missing_labels": missing_labels,
        "by_class_id": dict(sorted(all_classes.items())),
        "class_imbalance": _class_imbalance(all_classes),
        "splits": details,
    }


def _analyze_extra_seg(root: Path) -> dict:
    total = 0
    objects = 0
    invalid = 0
    by_class = Counter()
    for split in ("train", "val", "test"):
        d = root / "labels-seg" / split
        if not d.exists():
            continue
        for lp in d.glob("*.txt"):
            total += 1
            kind, cc, n, bad = _parse_yolo_file(lp)
            by_class.update(cc)
            objects += n
            if kind not in ("seg", "empty"):
                bad += 1
            invalid += bad
    return {"present": total > 0, "label_files": total, "objects": objects,
            "invalid_lines": invalid, "by_class_id": dict(sorted(by_class.items()))}


def _object_count_hints(root: Path) -> Dict[str, int]:
    """Best-effort object-count hints keyed by generated stem.

    This lets semantic QA distinguish an intentional pure-background sample
    from an accidentally empty mask. ScenePaste's crash-safe fragments are
    preferred when present; exported text/COCO annotations provide a fallback.
    """
    hints: Dict[str, int] = {}

    # Crash-safe generation metadata is the strongest signal and also exists
    # for semantic-only runs where YOLO text labels are intentionally absent.
    fragment_root = root / ".scenepaste" / "fragments"
    if fragment_root.exists():
        for fp in fragment_root.glob("*/meta/*.json"):
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
                stem = str(payload.get("stem", "")).strip()
                if stem:
                    hints[stem] = int(payload.get("objects", 0))
            except Exception:
                continue

    # Text annotations are cheap and remain useful after fragments are removed.
    for base in ("labels", "labels-seg", "labels-obb"):
        for split in ("train", "val", "test"):
            d = root / base / split
            if not d.exists():
                continue
            for lp in d.glob("*.txt"):
                if lp.stem in hints:
                    continue
                try:
                    hints[lp.stem] = sum(1 for line in lp.read_text(encoding="utf-8").splitlines() if line.strip())
                except Exception:
                    pass

    coco_path = root / "instances_coco.json"
    if coco_path.exists():
        try:
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
            image_to_stem = {
                int(row.get("id", -1)): Path(str(row.get("file_name", ""))).stem
                for row in payload.get("images", [])
            }
            counts = Counter(int(a.get("image_id", -1)) for a in payload.get("annotations", []))
            for image_id, stem in image_to_stem.items():
                if stem and stem not in hints:
                    hints[stem] = int(counts.get(image_id, 0))
        except Exception:
            pass
    return hints


def _analyze_semantic(root: Path) -> dict:
    mask_files = []
    for split in ("train", "val", "test"):
        d = root / "masks" / split
        if d.exists():
            mask_files.extend(d.glob("*.png"))
    pixel_counts = Counter()
    empty = 0
    intentional_empty = 0
    suspicious_empty = 0
    unreadable = 0
    object_hints = _object_count_hints(root)
    for p in mask_files:
        arr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            unreadable += 1
            continue
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            pixel_counts[int(v)] += int(c)
        if not np.any(arr != 0):
            empty += 1
            if object_hints.get(p.stem) == 0:
                intentional_empty += 1
            else:
                suspicious_empty += 1
    mapping = None
    map_path = root / "semantic_classes.json"
    if map_path.exists():
        try:
            mapping = json.loads(map_path.read_text(encoding="utf-8"))
        except Exception:
            mapping = None
    return {"present": bool(mask_files), "mask_files": len(mask_files),
            "empty_masks": empty, "intentional_empty_masks": intentional_empty,
            "suspicious_empty_masks": suspicious_empty, "unreadable_masks": unreadable,
            "pixel_counts": dict(sorted(pixel_counts.items())), "classes": mapping}


def _analyze_coco(root: Path) -> dict:
    p = root / "instances_coco.json"
    if not p.exists():
        return {"present": False}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": True, "valid": False, "error": str(exc)}
    images = d.get("images", [])
    anns = d.get("annotations", [])
    cats = d.get("categories", [])
    image_ids = {int(i.get("id", -1)) for i in images}
    invalid = 0
    by_class = Counter()
    for a in anns:
        by_class[int(a.get("category_id", -1))] += 1
        if int(a.get("image_id", -1)) not in image_ids:
            invalid += 1
        if float(a.get("area", 0)) <= 0:
            invalid += 1
        if not a.get("segmentation"):
            invalid += 1
    return {"present": True, "valid": invalid == 0, "images": len(images),
            "annotations": len(anns), "categories": len(cats),
            "invalid_annotations": invalid, "by_category_id": dict(sorted(by_class.items()))}


def _latest_log(root: Path) -> Optional[Path]:
    legacy = root / "generation_log.csv"
    candidates = []
    if legacy.exists():
        candidates.append(legacy)
    candidates += list(root.glob("run_*_log.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _analyze_reuse(root: Path) -> dict:
    p = _latest_log(root)
    if p is None:
        return {"present": False}
    from collections import defaultdict
    bg_stems = defaultdict(set)
    src_stems = defaultdict(set)
    rows = 0
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows += 1
                stem = r.get("generated_stem") or r.get("stem") or str(rows)
                if r.get("background"):
                    bg_stems[r["background"]].add(stem)
                sj = r.get("source_json", "")
                si = r.get("shape_index", "")
                if sj:
                    src_stems[f"{sj}#{si}"].add(stem)
    except Exception:
        return {"present": True, "log": p.name, "error": "无法解析日志"}
    return {
        "present": True, "log": p.name, "rows": rows,
        "unique_backgrounds": len(bg_stems),
        "max_background_reuse": max((len(v) for v in bg_stems.values()), default=0),
        "unique_sources": len(src_stems),
        "max_source_reuse": max((len(v) for v in src_stems.values()), default=0),
    }


def count_synthetic(root: Path) -> dict:
    images = _image_files(root)
    yolo = _analyze_yolo_dir(root)
    extra_seg = _analyze_extra_seg(root)
    semantic = _analyze_semantic(root)
    coco = _analyze_coco(root)
    reuse = _analyze_reuse(root)
    warnings = []
    if yolo["invalid_lines"]:
        warnings.append(f"YOLO invalid lines: {yolo['invalid_lines']}")
    if yolo["missing_labels"] and yolo["present"]:
        warnings.append(f"images missing labels: {yolo['missing_labels']}")
    if semantic.get("suspicious_empty_masks"):
        warnings.append(f"unexpected empty semantic masks: {semantic['suspicious_empty_masks']}")
    if coco.get("present") and coco.get("invalid_annotations", 0):
        warnings.append(f"invalid COCO annotations: {coco['invalid_annotations']}")
    imbalance = yolo.get("class_imbalance")
    if imbalance is not None and imbalance >= 10:
        warnings.append(f"class imbalance is high: {imbalance:.1f}x")
    return {
        "type": "synthetic",
        "images": len(images),
        "classes": _class_names(root),
        "yolo": yolo,
        "extra_segmentation": extra_seg,
        "semantic": semantic,
        "coco": coco,
        "reuse": reuse,
        "warnings": warnings,
        "health": "ok" if not warnings else "warning",
    }


def count_real(root: Path) -> dict:
    jsons = sorted(root.rglob("*.json"))
    total_objects = 0
    by_label = Counter()
    skipped = 0
    image_refs = set()
    for jp in jsons:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        if d.get("imagePath"):
            image_refs.add(str(d["imagePath"]))
        for s in d.get("shapes", []):
            lbl = str(s.get("label", "")).strip()
            if lbl:
                by_label[lbl] += 1
                total_objects += 1
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return {"type": "labelme", "images": len(images), "labels": len(jsons),
            "skipped_json": skipped, "total_objects": total_objects,
            "by_label": dict(sorted(by_label.items())), "health": "ok" if skipped == 0 else "warning"}


def _detect_kind(root: Path) -> str:
    if (root / "images").exists() or (root / "instances_coco.json").exists() or (root / "masks").exists():
        return "synthetic"
    return "real" if any(root.rglob("*.json")) else "synthetic"


def _print_synthetic(name: str, info: dict) -> None:
    print(f"\n【{name}】 synthetic dataset")
    print(f"  Images: {info['images']}  Health: {info['health']}")
    y = info["yolo"]
    if y["present"]:
        print(f"  YOLO: {y['dominant_format']} · labels={y['label_files']} · objects={y['total_objects']} · invalid={y['invalid_lines']}")
        print(f"  Class distribution: {y['by_class_id']}")
    s = info["extra_segmentation"]
    if s["present"]:
        print(f"  Extra seg labels: {s['label_files']} · objects={s['objects']} · invalid={s['invalid_lines']}")
    m = info["semantic"]
    if m["present"]:
        print(f"  Semantic masks: {m['mask_files']} · empty={m['empty_masks']} (intentional={m.get('intentional_empty_masks',0)}, suspicious={m.get('suspicious_empty_masks',0)}) · values={list(m['pixel_counts'])}")
    c = info["coco"]
    if c.get("present"):
        print(f"  COCO: images={c.get('images', 0)} · anns={c.get('annotations', 0)} · invalid={c.get('invalid_annotations', 0)}")
    r = info["reuse"]
    if r.get("present"):
        print(f"  Reuse: source max={r.get('max_source_reuse', 0)} · background max={r.get('max_background_reuse', 0)}")
    for w in info["warnings"]:
        print(f"  ⚠ {w}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze/validate generated or LabelMe datasets")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    result = {}
    for p in args.paths:
        if not p.exists():
            print(f"[skip] not found: {p}", file=sys.stderr)
            continue
        info = count_synthetic(p) if _detect_kind(p) == "synthetic" else count_real(p)
        result[p.name] = info

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    for name, info in result.items():
        if info["type"] == "synthetic":
            _print_synthetic(name, info)
        else:
            print(f"\n【{name}】 LabelMe · images={info['images']} · objects={info['total_objects']} · classes={info['by_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
