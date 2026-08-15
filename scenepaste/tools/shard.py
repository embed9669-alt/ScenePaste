"""Shard ScenePaste datasets into deterministic WebDataset-compatible tar files."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.models import IMAGE_SUFFIXES


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data); info.mtime = 0; info.mode = 0o444
    tf.addfile(info, io.BytesIO(data))


def _coco_by_stem(root: Path) -> Dict[str, dict]:
    path = root / "instances_coco.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    images = {int(i["id"]): i for i in data.get("images", []) if "id" in i}
    anns: Dict[int, List[dict]] = {}
    for a in data.get("annotations", []):
        try: anns.setdefault(int(a["image_id"]), []).append(a)
        except Exception: continue
    result = {}
    for iid, image in images.items():
        stem = Path(str(image.get("file_name", ""))).stem
        if stem:
            result[stem] = {"image": image, "annotations": anns.get(iid, []), "categories": data.get("categories", [])}
    return result



def _primary_label_kind(root: Path, split: str) -> str:
    label_dir = root / "labels" / split
    if not label_dir.exists():
        return "detect"
    for p in sorted(label_dir.glob("*.txt")):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            tok = line.split()
            if len(tok) == 5:
                return "detect"
            if len(tok) == 9:
                return "obb"
            if len(tok) >= 7 and len(tok) % 2 == 1:
                return "seg"
    # Empty-only datasets: consult latest run config when available.
    latest = root / "latest_summary.json"
    if latest.exists():
        try:
            run_id = json.loads(latest.read_text(encoding="utf-8")).get("run_id")
            cfg = root / f"{run_id}_config.json"
            if cfg.exists():
                fmt = str(json.loads(cfg.read_text(encoding="utf-8")).get("output_format", "detect"))
                if fmt in {"seg", "obb"}:
                    return fmt
        except Exception:
            pass
    return "detect"

def _sample_payloads(root: Path, split: str, image: Path, coco_map: Dict[str, dict], primary_kind: str) -> List[Tuple[str, bytes]]:
    stem = image.stem
    rows: List[Tuple[str, bytes]] = [(image.suffix.lower().lstrip("."), image.read_bytes())]
    mapping = [
        (root / "labels" / split / f"{stem}.txt", f"{primary_kind}.txt"),
        (root / "labels-seg" / split / f"{stem}.txt", "seg.txt"),
        (root / "labels-obb" / split / f"{stem}.txt", "obb.txt"),
        (root / "masks" / split / f"{stem}.png", "semantic.png"),
    ]
    modalities = []
    for p, suffix in mapping:
        if p.exists():
            rows.append((suffix, p.read_bytes())); modalities.append(suffix)
    if stem in coco_map:
        rows.append(("coco.json", json.dumps(coco_map[stem], ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        modalities.append("coco.json")
    meta = {"schema": "scenepaste/webdataset-sample", "version": 1, "stem": stem,
            "split": split, "image_name": image.name, "modalities": modalities}
    rows.append(("json", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    return rows


def build_webdataset_shards(dataset: Path, output: Path, split: str = "train",
                            max_samples: int = 10000, max_bytes: int = 0) -> dict:
    dataset = Path(dataset); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    image_dir = dataset / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    coco_map = _coco_by_stem(dataset)
    primary_kind = _primary_label_kind(dataset, split)
    max_samples = max(1, int(max_samples)); max_bytes = max(0, int(max_bytes))
    shards = []; tf: Optional[tarfile.TarFile] = None; shard_path: Optional[Path] = None
    shard_samples = shard_bytes = total = 0; shard_index = 0

    def open_shard() -> Tuple[tarfile.TarFile, Path]:
        path = output / f"{split}-{shard_index:06d}.tar"
        return tarfile.open(path, "w"), path

    def close_shard() -> None:
        nonlocal tf, shard_samples, shard_bytes, shard_path
        if tf is not None and shard_path is not None:
            tf.close()
            h = hashlib.sha256()
            with shard_path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            shards.append({"file": shard_path.name, "samples": shard_samples,
                           "bytes": shard_path.stat().st_size, "sha256": h.hexdigest()})
        tf = None; shard_path = None; shard_samples = 0; shard_bytes = 0

    for image in images:
        payloads = _sample_payloads(dataset, split, image, coco_map, primary_kind)
        estimated = sum(len(data) for _suffix, data in payloads)
        if tf is None:
            tf, shard_path = open_shard()
        elif shard_samples >= max_samples or (max_bytes and shard_samples > 0 and shard_bytes + estimated > max_bytes):
            close_shard(); shard_index += 1; tf, shard_path = open_shard()
        assert tf is not None
        key = image.stem
        for suffix, data in payloads:
            _add_bytes(tf, f"{key}.{suffix}", data)
        shard_samples += 1; shard_bytes += estimated; total += 1
    close_shard()
    manifest = {
        "schema": "scenepaste/webdataset-shards", "version": 1,
        "dataset": str(dataset.resolve()), "split": split, "samples": total,
        "max_samples": max_samples, "max_bytes": max_bytes, "primary_label_kind": primary_kind, "shards": shards,
        "pattern": f"{split}-{{000000..{max(0, len(shards)-1):06d}}}.tar" if shards else "",
    }
    for name in ("classes.txt", "data.yaml", "semantic_classes.json"):
        src = dataset / name
        if src.exists():
            shutil.copy2(src, output / name)
    manifest_path = output / f"{split}-shards.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic WebDataset tar shards from ScenePaste output")
    parser.add_argument("dataset", type=Path); parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-bytes", type=int, default=0, help="optional uncompressed payload byte cap per shard")
    args = parser.parse_args(argv)
    report = build_webdataset_shards(args.dataset, args.output, args.split, args.max_samples, args.max_bytes)
    print(f"WebDataset shards: {len(report['shards'])} files, {report['samples']} samples")
    print(args.output / f"{args.split}-shards.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
