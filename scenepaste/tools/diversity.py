"""Dataset diversity analysis and representative sample selection."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

from ..core.similarity import diversity_summary, iter_dataset_images, select_diverse


def analyze_diversity(dataset: Path, limit: int = 1000, select: int = 0, embedding_backend: str = "cv-lite-v1") -> dict:
    rows = iter_dataset_images(Path(dataset))
    paths = [p for _split, p in rows]
    report = {
        "schema": "scenepaste/diversity-report", "version": 1,
        "dataset": str(Path(dataset).resolve()),
        "summary": diversity_summary(paths, limit=limit, backend=embedding_backend),
        "selected": [],
    }
    if select > 0:
        chosen = select_diverse(paths, count=select, limit=max(limit, select), backend=embedding_backend)
        report["selected"] = [{"image": str(p), "score": score} for p, score in chosen]
    return report



def export_selected_dataset(dataset: Path, selected: Sequence[Path], output: Path) -> dict:
    dataset = Path(dataset); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected_stems = set()
    copied = 0
    for image in selected:
        split = image.parent.name if image.parent.name in {"train", "val", "test"} else "train"
        stem = image.stem; selected_stems.add(stem)
        dst_img = output / "images" / split / image.name; dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dst_img); copied += 1
        for src_sub, dst_sub, ext in [
            ("labels", "labels", ".txt"), ("labels-seg", "labels-seg", ".txt"),
            ("labels-obb", "labels-obb", ".txt"), ("masks", "masks", ".png"),
        ]:
            src = dataset / src_sub / split / f"{stem}{ext}"
            if src.exists():
                dst = output / dst_sub / split / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    for name in ("classes.txt", "data.yaml", "semantic_classes.json"):
        src = dataset / name
        if src.exists(): shutil.copy2(src, output / name)
    # Filter COCO if available.
    coco = dataset / "instances_coco.json"
    if coco.exists():
        try:
            data = json.loads(coco.read_text(encoding="utf-8"))
            images = [i for i in data.get("images", []) if Path(str(i.get("file_name", ""))).stem in selected_stems]
            ids = {int(i["id"]) for i in images if "id" in i}
            anns = [a for a in data.get("annotations", []) if int(a.get("image_id", -1)) in ids]
            (output / "instances_coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": data.get("categories", [])}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return {"images": copied, "output": str(output)}

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze visual diversity and select maximally diverse samples")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--select", type=int, default=0, help="select N diverse images")
    parser.add_argument("--embedding-backend", choices=("cv-lite-v1","clip","dinov2"), default="cv-lite-v1")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--copy-selected", type=Path, default=None, help="copy selected images only")
    parser.add_argument("--export-dataset", type=Path, default=None, help="export selected images with matching labels/masks")
    args = parser.parse_args(argv)
    report = analyze_diversity(args.dataset, args.limit, args.select, args.embedding_backend)
    output = args.output or (args.dataset / "diversity_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["selected"]:
        csv_path = output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["image", "score"]); w.writeheader(); w.writerows(report["selected"])
        selected_paths = [Path(row["image"]) for row in report["selected"]]
        if args.copy_selected:
            args.copy_selected.mkdir(parents=True, exist_ok=True)
            for p in selected_paths:
                shutil.copy2(p, args.copy_selected / p.name)
        if args.export_dataset:
            export_selected_dataset(args.dataset, selected_paths, args.export_dataset)
    s = report["summary"]
    print(f"Diversity ({s.get('backend')}): samples={s.get('samples')} mean_uniqueness={s.get('mean_uniqueness')}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
