#!/usr/bin/env python3
"""按源目标分组拆 train/val，避免 Copy-Paste 数据泄漏。

同一个 source object（source_json + shape_index）生成出来的所有图片会进入同一 split。
支持当前 run-id 日志、旧 generation_log.csv、YOLO detect/seg/OBB、semantic masks
以及 COCO instances JSON 的同步拆分。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _source_key(source_json: str, shape_index: str) -> str:
    return f"{source_json}#{shape_index}"


def resolve_log_path(output_dir: Path, run_id: str = "latest") -> Optional[Path]:
    """解析本次要拆分的 generation log，兼容新旧目录结构。"""
    output_dir = Path(output_dir)
    if run_id and run_id != "latest":
        candidates = [
            output_dir / "runs" / run_id / "generation_log.csv",
            output_dir / f"{run_id}_log.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # 新版 summary 指针优先。
    summary = output_dir / "latest_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            log_file = data.get("log_file")
            if log_file:
                p = output_dir / str(log_file)
                if p.exists():
                    return p
        except Exception:
            pass

    # 旧版 run_config 指针。
    cfg = output_dir / "run_config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            rid = data.get("latest_run_id")
            if rid:
                for p in (output_dir / f"{rid}_log.csv",
                          output_dir / "runs" / str(rid) / "generation_log.csv"):
                    if p.exists():
                        return p
        except Exception:
            pass

    legacy = output_dir / "generation_log.csv"
    if legacy.exists():
        return legacy
    logs = sorted(output_dir.glob("run_*_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _read_log(log_path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stem = r.get("generated_stem") or r.get("stem") or ""
            sj = r.get("source_json", "")
            si = r.get("shape_index", "0")
            if stem:
                rows.append((stem, sj, si))
    return rows


class _DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_images_by_source(rows: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    dsu = _DSU()
    source_to_stems: Dict[str, List[str]] = defaultdict(list)
    all_stems: Set[str] = set()
    for stem, sj, si in rows:
        all_stems.add(stem)
        # 无 source 信息的 GUI/旧日志，按 stem 自成组，不错误地把空 source 全 union。
        if not sj:
            dsu.find(stem)
            continue
        src = _source_key(sj, si)
        source_to_stems[src].append(stem)
        dsu.find(stem)
        if len(source_to_stems[src]) > 1:
            dsu.union(source_to_stems[src][0], stem)

    groups: Dict[str, Set[str]] = defaultdict(set)
    for stem in all_stems:
        groups[dsu.find(stem)].add(stem)
    return dict(groups)


def assign_splits(groups: Dict[str, Set[str]], val_ratio: float,
                  seed: int = 42) -> Tuple[Set[str], Set[str]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 [0, 1) 范围")

    def group_hash(gid: str) -> int:
        raw = f"{seed}:{gid}".encode("utf-8")
        return int(hashlib.sha256(raw).hexdigest()[:16], 16)

    items = sorted(groups.items(), key=lambda kv: (len(kv[1]), group_hash(kv[0])))
    total = sum(len(v) for v in groups.values())
    target = round(total * val_ratio)
    val: Set[str] = set()
    train: Set[str] = set()
    current = 0
    for gid, stems in items:
        size = len(stems)
        put_val = current < target and abs((current + size) - target) <= abs(current - target)
        if put_val:
            val |= stems
            current += size
        else:
            train |= stems
    return train, val


def _transfer_one(stem: str, src_dir: Path, dst_dir: Path, mode: str) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".txt"):
        src = src_dir / f"{stem}{ext}"
        if not src.exists():
            continue
        dst = dst_dir / src.name
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        moved += 1
    return moved


def apply_split(output_dir: Path, val_stems: Set[str], mode: str = "move") -> Dict[str, int]:
    """将 val stem 从 train 分流。OBB 与纯 seg 都在 labels/train，因此自动兼容。"""
    pairs = [
        ("images/train", "images/val"),
        ("labels/train", "labels/val"),
        ("labels-seg/train", "labels-seg/val"),
        ("labels-obb/train", "labels-obb/val"),
        ("masks/train", "masks/val"),
    ]
    files = 0
    for stem in val_stems:
        for src_sub, dst_sub in pairs:
            files += _transfer_one(stem, output_dir / src_sub, output_dir / dst_sub, mode)
    return {"files_transferred": files}


def split_coco(output_dir: Path, train_stems: Set[str], val_stems: Set[str]) -> Optional[Dict[str, int]]:
    src = output_dir / "instances_coco.json"
    if not src.exists():
        return None
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return None

    def stem_of(entry: dict) -> str:
        return Path(str(entry.get("file_name", ""))).stem

    train_images = [i for i in data.get("images", []) if stem_of(i) in train_stems]
    val_images = [i for i in data.get("images", []) if stem_of(i) in val_stems]
    train_ids = {int(i["id"]) for i in train_images}
    val_ids = {int(i["id"]) for i in val_images}
    anns = data.get("annotations", [])
    train_anns = [a for a in anns if int(a.get("image_id", -1)) in train_ids]
    val_anns = [a for a in anns if int(a.get("image_id", -1)) in val_ids]
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    common = {"categories": data.get("categories", [])}
    (ann_dir / "instances_train.json").write_text(
        json.dumps({"images": train_images, "annotations": train_anns, **common},
                   ensure_ascii=False), encoding="utf-8")
    (ann_dir / "instances_val.json").write_text(
        json.dumps({"images": val_images, "annotations": val_anns, **common},
                   ensure_ascii=False), encoding="utf-8")
    return {"train_images": len(train_images), "val_images": len(val_images),
            "train_annotations": len(train_anns), "val_annotations": len(val_anns)}


def _read_class_map(output_dir: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    p = output_dir / "classes.txt"
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        cid, name = line.split(":", 1)
        try:
            result[name.strip()] = int(cid.strip())
        except ValueError:
            pass
    return result


def update_data_yaml(output_dir: Path) -> None:
    class_map = _read_class_map(output_dir)
    if not class_map:
        return
    try:
        import scenepaste.core as core
        core.write_data_yaml(output_dir, class_map, val_split="images/val")
    except Exception:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="按 source object 拆 train/val，防止 Copy-Paste 数据泄漏")
    p.add_argument("--input", type=Path, required=True, help="生成器输出目录")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    p.add_argument("--seed", type=int, default=42, help="确定性分组种子")
    p.add_argument("--run-id", default="latest", help="拆分哪个 run；默认 latest")
    p.add_argument("--mode", choices=("move", "copy"), default="move",
                   help="move=原地拆分；copy=复制到 val（默认 move）")
    p.add_argument("--dry-run", action="store_true", help="只显示计划，不改文件")
    args = p.parse_args(argv)

    out_dir = args.input
    log_path = resolve_log_path(out_dir, args.run_id)
    if log_path is None:
        print(f"找不到可用 generation log：{out_dir}", file=sys.stderr)
        return 1
    rows = _read_log(log_path)
    if not rows:
        print(f"日志为空：{log_path}", file=sys.stderr)
        return 1

    groups = group_images_by_source(rows)
    train_stems, val_stems = assign_splits(groups, args.val_ratio, args.seed)
    total = len(train_stems) + len(val_stems)
    actual = len(val_stems) / total if total else 0.0

    source_to_split: Dict[str, str] = {}
    leaks = []
    for stem, sj, si in rows:
        if not sj:
            continue
        src = _source_key(sj, si)
        split = "val" if stem in val_stems else "train"
        if src in source_to_split and source_to_split[src] != split:
            leaks.append(src)
        source_to_split[src] = split

    print(f"日志：{log_path.name}")
    print(f"总图片：{total}  source groups：{len(groups)}")
    print(f"train={len(train_stems)}  val={len(val_stems)}  实际 val={actual:.1%}")
    print("✓ source 泄漏校验通过" if not leaks else f"⚠ source 泄漏：{len(set(leaks))}")
    if args.dry_run:
        print("[dry-run] 未修改任何文件")
        return 0

    transfer = apply_split(out_dir, val_stems, args.mode)
    coco = split_coco(out_dir, train_stems, val_stems)
    update_data_yaml(out_dir)
    report = {
        "log_file": str(log_path.name),
        "mode": args.mode,
        "total_images": total,
        "source_groups": len(groups),
        "train": len(train_stems),
        "val": len(val_stems),
        "target_val_ratio": args.val_ratio,
        "actual_val_ratio": actual,
        "source_leaks": len(set(leaks)),
        **transfer,
        "coco": coco,
        "stems_val": sorted(val_stems),
    }
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成：处理 {transfer['files_transferred']} 个文件")
    if coco:
        print(f"COCO：train {coco['train_images']} / val {coco['val_images']}")
    print(f"报告：{out_dir / 'split_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
