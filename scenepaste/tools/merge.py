#!/usr/bin/env python3
"""Merge multiple generated datasets into one dataset without filename collisions.

Supports images, YOLO labels (detect/seg/OBB), extra labels-seg, semantic masks,
previews and COCO instances. Source stems are prefixed with a stable dataset key.

Usage:
    scenepaste merge ds_a ds_b --output merged
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _safe_prefix(path: Path, index: int) -> str:
    raw = re.sub(r"[^A-Za-z0-9_-]+", "_", path.name).strip("_") or f"ds{index}"
    return f"{index:02d}_{raw}"


def _copy_stem(src_root: Path, dst_root: Path, split: str, stem: str,
               new_stem: str) -> int:
    copied = 0
    pairs = [
        (src_root / "images" / split, dst_root / "images" / split, IMAGE_EXTS),
        (src_root / "labels" / split, dst_root / "labels" / split, (".txt",)),
        (src_root / "labels-seg" / split, dst_root / "labels-seg" / split, (".txt",)),
        (src_root / "labels-obb" / split, dst_root / "labels-obb" / split, (".txt",)),
        (src_root / "masks" / split, dst_root / "masks" / split, (".png", ".tif", ".tiff")),
    ]
    for src_dir, dst_dir, exts in pairs:
        if not src_dir.exists():
            continue
        for ext in exts:
            src = src_dir / f"{stem}{ext}"
            if src.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / f"{new_stem}{ext}")
                copied += 1
                break
    preview_dir = src_root / "previews"
    if preview_dir.exists():
        for ext in IMAGE_EXTS:
            src = preview_dir / f"{stem}{ext}"
            if src.exists():
                dst = dst_root / "previews"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / f"{new_stem}{ext}")
                copied += 1
                break
    return copied


def _class_map(root: Path) -> Dict[str, int]:
    p = root / "classes.txt"
    out: Dict[str, int] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        cid, name = line.split(":", 1)
        try:
            out[name.strip()] = int(cid.strip())
        except ValueError:
            pass
    return out


def _assert_compatible_classes(inputs: List[Path]) -> Dict[str, int]:
    reference: Optional[Dict[str, int]] = None
    for root in inputs:
        cm = _class_map(root)
        if not cm:
            continue
        if reference is None:
            reference = cm
        elif cm != reference:
            raise ValueError(
                f"类别映射不一致：{root}\nexpected={reference}\nactual={cm}\n"
                "请先统一 classes.txt / class ids 再合并。"
            )
    return reference or {}


def _merge_logs(inputs: List[Path], prefixes: Dict[Path, str], output: Path) -> int:
    out = output / "generation_log.csv"
    header = ["dataset", "stem", "background", "label", "class_id", "source_json",
              "shape_index", "flip", "angle", "box_xyxy"]
    rows = 0
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for root in inputs:
            logs = [root / "generation_log.csv"] + sorted(root.glob("run_*_log.csv"))
            log = next((p for p in logs if p.exists()), None)
            if log is None:
                continue
            with log.open("r", encoding="utf-8-sig", newline="") as src:
                for r in csv.DictReader(src):
                    stem = r.get("generated_stem") or r.get("stem") or ""
                    if not stem:
                        continue
                    writer.writerow({
                        "dataset": root.name,
                        "stem": f"{prefixes[root]}_{stem}",
                        "background": r.get("background", ""),
                        "label": r.get("label", ""),
                        "class_id": r.get("class_id", ""),
                        "source_json": r.get("source_json", ""),
                        "shape_index": r.get("shape_index", "0"),
                        "flip": r.get("flip", r.get("flipped", "")),
                        "angle": r.get("angle", ""),
                        "box_xyxy": r.get("box_xyxy", ""),
                    })
                    rows += 1
    return rows


def _merge_coco(inputs: List[Path], prefixes: Dict[Path, str], output: Path) -> Optional[dict]:
    datasets = []
    for root in inputs:
        p = root / "instances_coco.json"
        if p.exists():
            try:
                datasets.append((root, json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                pass
    if not datasets:
        return None

    categories_by_name: Dict[str, int] = {}
    for _, d in datasets:
        for c in d.get("categories", []):
            name = str(c.get("name", c.get("id")))
            if name not in categories_by_name:
                categories_by_name[name] = len(categories_by_name)
    cats = [{"id": cid, "name": name, "supercategory": "object"}
            for name, cid in sorted(categories_by_name.items(), key=lambda kv: kv[1])]

    images_out = []
    anns_out = []
    next_image_id = 1
    next_ann_id = 1
    for root, d in datasets:
        old_cat_name = {int(c.get("id", -1)): str(c.get("name", c.get("id")))
                        for c in d.get("categories", [])}
        image_id_map: Dict[int, int] = {}
        for img in d.get("images", []):
            old_id = int(img.get("id", -1))
            entry = dict(img)
            entry["id"] = next_image_id
            file_name = Path(str(entry.get("file_name", "")))
            new_name = f"{prefixes[root]}_{file_name.name}"
            entry["file_name"] = str(file_name.parent / new_name) if str(file_name.parent) != "." else new_name
            image_id_map[old_id] = next_image_id
            next_image_id += 1
            images_out.append(entry)
        for ann in d.get("annotations", []):
            old_image_id = int(ann.get("image_id", -1))
            if old_image_id not in image_id_map:
                continue
            entry = dict(ann)
            entry["id"] = next_ann_id
            next_ann_id += 1
            entry["image_id"] = image_id_map[old_image_id]
            old_cid = int(ann.get("category_id", -1))
            entry["category_id"] = categories_by_name[old_cat_name.get(old_cid, str(old_cid))]
            anns_out.append(entry)
    (output / "instances_coco.json").write_text(
        json.dumps({"images": images_out, "annotations": anns_out, "categories": cats},
                   ensure_ascii=False), encoding="utf-8")
    return {"images": len(images_out), "annotations": len(anns_out), "categories": len(cats)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge generated datasets")
    parser.add_argument("inputs", nargs="+", type=Path, help="dataset roots")
    parser.add_argument("--output", type=Path, required=True, help="merged dataset root")
    parser.add_argument("--force", action="store_true", help="allow non-empty output directory")
    args = parser.parse_args(argv)

    inputs = [p for p in args.inputs if p.is_dir()]
    if len(inputs) != len(args.inputs):
        missing = [str(p) for p in args.inputs if not p.is_dir()]
        print(f"Missing inputs: {missing}", file=sys.stderr)
        return 1
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        print("Output is not empty; pass --force to merge into it.", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        class_map = _assert_compatible_classes(inputs)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    prefixes = {root: _safe_prefix(root, i) for i, root in enumerate(inputs, 1)}

    image_count = 0
    file_count = 0
    for root in inputs:
        prefix = prefixes[root]
        for split in ("train", "val", "test"):
            img_dir = root / "images" / split
            if not img_dir.exists():
                continue
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                new_stem = f"{prefix}_{img.stem}"
                file_count += _copy_stem(root, args.output, split, img.stem, new_stem)
                image_count += 1

    if class_map:
        (args.output / "classes.txt").write_text(
            "".join(f"{cid}: {name}\n" for name, cid in sorted(class_map.items(), key=lambda kv: kv[1])),
            encoding="utf-8")
        try:
            import scenepaste.core as core
            val = "images/val" if (args.output / "images" / "val").exists() else None
            core.write_data_yaml(args.output, class_map, val_split=val)
        except Exception:
            pass

    log_rows = _merge_logs(inputs, prefixes, args.output)
    coco = _merge_coco(inputs, prefixes, args.output)
    report = {"inputs": [str(p) for p in inputs], "images": image_count,
              "files": file_count, "log_rows": log_rows, "coco": coco}
    (args.output / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {image_count} images from {len(inputs)} datasets -> {args.output}")
    if coco:
        print(f"COCO: {coco['images']} images / {coco['annotations']} annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
