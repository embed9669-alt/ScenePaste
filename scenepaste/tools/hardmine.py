"""Model-driven hard-example mining for Detect, Segmentation, and OBB.

Prediction text is intentionally compatible with common Ultralytics exports:
- detect: ``class cx cy w h [confidence]``
- obb: ``class x1 y1 x2 y2 x3 y3 x4 y4 [confidence]``
- seg: ``class x1 y1 x2 y2 ... [confidence]``

All coordinates are normalized. Geometry is matched using rasterized polygon
IoU, so boxes, oriented boxes, and polygons share the same scoring path.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.distribution import learn_yolo_profile_subset
from ..core.models import IMAGE_SUFFIXES


@dataclass(frozen=True)
class Shape:
    cls: int
    polygon: Tuple[Tuple[float, float], ...]
    conf: float = 1.0


# Backwards-compatible detection helper used by external callers.
@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: Tuple[float, float, float, float]
    conf: float = 1.0


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _box_shape(cls: int, cx: float, cy: float, w: float, h: float, conf: float = 1.0) -> Shape:
    x1, y1, x2, y2 = _xywh_to_xyxy(cx, cy, w, h)
    return Shape(cls, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)), conf)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy; bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, aa + ba - inter)


def _shape_iou(a: Shape, b: Shape, raster: int = 256) -> float:
    if a.cls != b.cls or not a.polygon or not b.polygon:
        return 0.0
    scale = max(32, int(raster)) - 1
    ma = np.zeros((scale + 1, scale + 1), dtype=np.uint8)
    mb = np.zeros_like(ma)
    pa = np.asarray([[min(1.0, max(0.0, x)) * scale, min(1.0, max(0.0, y)) * scale]
                     for x, y in a.polygon], dtype=np.float32).round().astype(np.int32)
    pb = np.asarray([[min(1.0, max(0.0, x)) * scale, min(1.0, max(0.0, y)) * scale]
                     for x, y in b.polygon], dtype=np.float32).round().astype(np.int32)
    if len(pa) < 3 or len(pb) < 3:
        return 0.0
    cv2.fillPoly(ma, [pa], 1); cv2.fillPoly(mb, [pb], 1)
    inter = int(np.count_nonzero((ma != 0) & (mb != 0)))
    union = int(np.count_nonzero((ma != 0) | (mb != 0)))
    return inter / union if union else 0.0


def _infer_conf(values: List[float], task: str, prediction: bool) -> Tuple[List[float], float]:
    if not prediction:
        return values, 1.0
    if task == "detect":
        return (values[:4], float(values[4])) if len(values) >= 5 else (values[:4], 1.0)
    if task == "obb":
        return (values[:8], float(values[8])) if len(values) >= 9 else (values[:8], 1.0)
    # Seg polygons have variable length. Ultralytics appends confidence when
    # save_conf=True, yielding an odd number of coordinate/conf values.
    if len(values) >= 7 and len(values) % 2 == 1:
        return values[:-1], float(values[-1])
    return values, 1.0


def _parse_line(line: str, task: str, prediction: bool) -> Optional[Shape]:
    tok = line.split()
    if not tok:
        return None
    try:
        cls = int(float(tok[0])); values = [float(v) for v in tok[1:]]
    except ValueError:
        return None
    values, conf = _infer_conf(values, task, prediction)
    if task == "detect":
        if len(values) < 4:
            return None
        return _box_shape(cls, *values[:4], conf=conf)
    if task == "obb":
        if len(values) < 8:
            return None
        pts = tuple((values[i], values[i + 1]) for i in range(0, 8, 2))
        return Shape(cls, pts, conf)
    if task == "seg":
        if len(values) < 6 or len(values) % 2:
            return None
        pts = tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))
        return Shape(cls, pts, conf)
    raise ValueError(f"unsupported hardmine task: {task}")


def _read_shapes(path: Path, task: str, prediction: bool = False) -> List[Shape]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        shape = _parse_line(line, task, prediction)
        if shape is not None:
            rows.append(shape)
    return rows


def _read_gt(path: Path) -> List[Box]:
    rows = []
    for shape in _read_shapes(path, "detect", prediction=False):
        xs = [p[0] for p in shape.polygon]; ys = [p[1] for p in shape.polygon]
        rows.append(Box(shape.cls, (min(xs), min(ys), max(xs), max(ys)), 1.0))
    return rows


def _read_pred(path: Path) -> List[Box]:
    rows = []
    for shape in _read_shapes(path, "detect", prediction=True):
        xs = [p[0] for p in shape.polygon]; ys = [p[1] for p in shape.polygon]
        rows.append(Box(shape.cls, (min(xs), min(ys), max(xs), max(ys)), shape.conf))
    return rows


def _prediction_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "labels").is_dir():
        return path / "labels"
    return path


def score_shapes(gt: List[Shape], pred: List[Shape], match_iou: float = 0.5,
                 hard_conf: float = 0.5, fp_conf: float = 0.25,
                 localization_iou: float = 0.75) -> dict:
    preds = sorted(pred, key=lambda b: b.conf, reverse=True)
    used_gt = set(); matches = []; false_pos = []
    for pi, p in enumerate(preds):
        best = (-1, 0.0)
        for gi, g in enumerate(gt):
            if gi in used_gt or g.cls != p.cls:
                continue
            iou = _shape_iou(g, p)
            if iou > best[1]: best = (gi, iou)
        if best[0] >= 0 and best[1] >= match_iou:
            used_gt.add(best[0]); matches.append((best[0], pi, best[1], p.conf))
        elif p.conf >= fp_conf:
            false_pos.append((pi, p.conf))
    false_neg = [gi for gi in range(len(gt)) if gi not in used_gt]
    fn_classes: Dict[str, int] = {}
    for gi in false_neg:
        key = str(gt[gi].cls); fn_classes[key] = fn_classes.get(key, 0) + 1
    fp_classes: Dict[str, int] = {}
    for pi, _conf in false_pos:
        key = str(preds[pi].cls); fp_classes[key] = fp_classes.get(key, 0) + 1
    low_conf_tp = [m for m in matches if m[3] < hard_conf]
    poor_loc = [m for m in matches if m[2] < localization_iou]
    loc_penalty = sum(max(0.0, localization_iou - m[2]) / max(1e-6, localization_iou) for m in poor_loc)
    score = 3.0 * len(false_neg) + 2.0 * len(false_pos) + len(low_conf_tp) + 1.5 * loc_penalty
    if not gt and false_pos: score += len(false_pos)
    return {
        "score": float(score), "gt": len(gt), "pred": len(preds),
        "false_negatives": len(false_neg), "false_positives": len(false_pos),
        "false_negative_classes": fn_classes, "false_positive_classes": fp_classes,
        "low_confidence_true_positives": len(low_conf_tp), "poor_localization": len(poor_loc),
        "mean_matched_iou": (sum(m[2] for m in matches) / len(matches)) if matches else None,
        "mean_matched_confidence": (sum(m[3] for m in matches) / len(matches)) if matches else None,
    }


def score_image(gt: List[Box], pred: List[Box], match_iou: float = 0.5,
                hard_conf: float = 0.5, fp_conf: float = 0.25,
                localization_iou: float = 0.75) -> dict:
    """Backwards-compatible detect-only scoring API."""
    def cvt(b: Box) -> Shape:
        x1, y1, x2, y2 = b.xyxy
        return Shape(b.cls, ((x1,y1),(x2,y1),(x2,y2),(x1,y2)), b.conf)
    return score_shapes([cvt(x) for x in gt], [cvt(x) for x in pred], match_iou, hard_conf, fp_conf, localization_iou)


def _gt_dir(dataset: Path, split: str, task: str) -> Path:
    if task == "seg":
        p = dataset / "labels-seg" / split
        if p.is_dir(): return p
    if task == "obb":
        p = dataset / "labels-obb" / split
        if p.is_dir(): return p
    return dataset / "labels" / split


def mine_hard_examples(dataset: Path, predictions: Path, split: str = "val", top: int = 200,
                       match_iou: float = 0.5, hard_conf: float = 0.5,
                       fp_conf: float = 0.25, localization_iou: float = 0.75,
                       task: str = "detect") -> dict:
    task = str(task).lower()
    if task not in {"detect", "seg", "obb"}:
        raise ValueError("task must be detect / seg / obb")
    dataset = Path(dataset); predictions = _prediction_dir(Path(predictions))
    image_dir = dataset / "images" / split; gt_dir = _gt_dir(dataset, split, task)
    if not image_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"{task} split not found: {image_dir} / {gt_dir}")
    rows = []
    for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        gt = _read_shapes(gt_dir / f"{image.stem}.txt", task, prediction=False)
        pred = _read_shapes(predictions / f"{image.stem}.txt", task, prediction=True)
        stats = score_shapes(gt, pred, match_iou, hard_conf, fp_conf, localization_iou)
        stats.update({"stem": image.stem, "image": str(image), "split": split, "task": task})
        reasons = []
        if stats["false_negatives"]: reasons.append(f"FN:{stats['false_negatives']}")
        if stats["false_positives"]: reasons.append(f"FP:{stats['false_positives']}")
        if stats["low_confidence_true_positives"]: reasons.append(f"low-conf:{stats['low_confidence_true_positives']}")
        if stats["poor_localization"]: reasons.append(f"geometry:{stats['poor_localization']}")
        stats["reasons"] = reasons; rows.append(stats)
    rows.sort(key=lambda r: (-float(r["score"]), r["stem"]))
    hard = [r for r in rows if r["score"] > 0][: max(0, int(top))]
    fn_by_class: Dict[str, int] = {}; fp_by_class: Dict[str, int] = {}
    for row in rows:
        for key, value in row.get("false_negative_classes", {}).items(): fn_by_class[key] = fn_by_class.get(key, 0) + int(value)
        for key, value in row.get("false_positive_classes", {}).items(): fp_by_class[key] = fp_by_class.get(key, 0) + int(value)
    return {
        "schema": "scenepaste/hard-example-report", "version": 2, "task": task,
        "dataset": str(dataset.resolve()), "predictions": str(predictions.resolve()), "split": split,
        "images": len(rows), "hard_images": len(hard),
        "total_false_negatives": sum(r["false_negatives"] for r in rows),
        "total_false_positives": sum(r["false_positives"] for r in rows),
        "total_low_confidence_true_positives": sum(r["low_confidence_true_positives"] for r in rows),
        "total_poor_localization": sum(r["poor_localization"] for r in rows),
        "false_negatives_by_class_id": fn_by_class, "false_positives_by_class_id": fp_by_class,
        "thresholds": {"match_iou": match_iou, "hard_conf": hard_conf, "fp_conf": fp_conf, "localization_iou": localization_iou},
        "hard_examples": hard,
    }


def render_hardmine_html(report: dict) -> str:
    rows = []
    for row in report.get("hard_examples", [])[:100]:
        rows.append("<tr>" + f"<td>{html.escape(str(row.get('stem','')))}</td>" +
                    f"<td>{float(row.get('score',0)):.2f}</td><td>{int(row.get('false_negatives',0))}</td>" +
                    f"<td>{int(row.get('false_positives',0))}</td><td>{int(row.get('low_confidence_true_positives',0))}</td>" +
                    f"<td>{int(row.get('poor_localization',0))}</td><td>{html.escape(', '.join(row.get('reasons',[])))}</td></tr>")
    fn_classes = html.escape(json.dumps(report.get("false_negatives_by_class_id", {}), ensure_ascii=False))
    fp_classes = html.escape(json.dumps(report.get("false_positives_by_class_id", {}), ensure_ascii=False))
    task = html.escape(str(report.get("task", "detect")))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>ScenePaste Hard Example Mining</title>
<style>body{{font-family:Inter,Segoe UI,Arial;background:#101215;color:#e9edf1;margin:0}}.wrap{{max-width:1200px;margin:auto;padding:28px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{background:#181c21;border:1px solid #2a3038;border-radius:12px;padding:16px}}.big{{font-size:28px;font-weight:700;margin-top:6px}}table{{width:100%;border-collapse:collapse;background:#181c21}}th,td{{padding:10px;border-bottom:1px solid #2a3038;text-align:left}}th{{background:#20252c}}.muted{{color:#919aa5}}code{{color:#9dd1ff}}</style></head><body><div class="wrap">
<h1>ScenePaste — Hard Example Mining</h1><p class="muted">Task: <b>{task}</b><br>Dataset: {html.escape(report.get('dataset',''))}<br>Predictions: {html.escape(report.get('predictions',''))}</p>
<div class="grid"><div class="card">Hard images<div class="big">{report.get('hard_images',0)}</div></div><div class="card">False negatives<div class="big">{report.get('total_false_negatives',0)}</div></div><div class="card">False positives<div class="big">{report.get('total_false_positives',0)}</div></div><div class="card">Low-conf TP<div class="big">{report.get('total_low_confidence_true_positives',0)}</div></div><div class="card">Poor geometry<div class="big">{report.get('total_poor_localization',0)}</div></div></div>
<h2>Failure classes</h2><div class="card">FN by class id: <code>{fn_classes}</code><br>FP by class id: <code>{fp_classes}</code></div>
<h2>Top hard examples</h2><table><thead><tr><th>Stem</th><th>Score</th><th>FN</th><th>FP</th><th>Low conf</th><th>Geometry</th><th>Reasons</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></body></html>'''


def write_hardmine_outputs(report: dict, output_dir: Path, make_profile: bool = True) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hard_examples.json"; html_path = output_dir / "hardmine_dashboard.html"
    csv_path = output_dir / "hard_examples.csv"; list_path = output_dir / "hard_examples.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_hardmine_html(report), encoding="utf-8")
    fields = ["stem","task","score","gt","pred","false_negatives","false_positives","low_confidence_true_positives","poor_localization","mean_matched_iou","mean_matched_confidence","reasons","image"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in report.get("hard_examples", []):
            out = dict(row); out["reasons"] = ";".join(row.get("reasons", [])); w.writerow({k: out.get(k, "") for k in fields})
    list_path.write_text("\n".join(r["image"] for r in report.get("hard_examples", [])) + ("\n" if report.get("hard_examples") else ""), encoding="utf-8")
    negative_path = output_dir / "hard_negative_backgrounds.txt"
    negatives = [r["image"] for r in report.get("hard_examples", []) if int(r.get("gt",0)) == 0 and int(r.get("false_positives",0)) > 0]
    negative_path.write_text("\n".join(negatives) + ("\n" if negatives else ""), encoding="utf-8")
    profile_path: Optional[Path] = None
    if make_profile:
        stems = [r["stem"] for r in report.get("hard_examples", []) if int(r.get("gt",0)) > 0]
        if stems:
            try:
                profile = learn_yolo_profile_subset(Path(report["dataset"]), stems, geometry_source=str(report.get("task", "auto")))
                profile.data["hardmine"] = {"task": report.get("task","detect"), "source_report": str(json_path), "selected_images": len(stems)}
                profile_path = output_dir / "hard_distribution_profile.json"; profile.save(profile_path)
            except Exception:
                profile_path = None
    result = dict(report); result.update({"json_path":str(json_path),"html_path":str(html_path),"csv_path":str(csv_path),"list_path":str(list_path),"profile_path":str(profile_path) if profile_path else None,"hard_negative_backgrounds":str(negative_path),"hard_negative_count":len(negatives)})
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mine Detect/Seg/OBB model failures and build a hard-generation profile")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=("detect","seg","obb"), default="detect")
    parser.add_argument("--split", default="val", choices=("train","val","test"))
    parser.add_argument("--top", type=int, default=200); parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--hard-conf", type=float, default=0.5); parser.add_argument("--fp-conf", type=float, default=0.25)
    parser.add_argument("--localization-iou", type=float, default=0.75); parser.add_argument("-o","--output",type=Path,default=None)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)
    report = mine_hard_examples(args.dataset,args.predictions,args.split,args.top,args.match_iou,args.hard_conf,args.fp_conf,args.localization_iou,args.task)
    out = args.output or (args.dataset / f"hardmine-{args.task}"); result = write_hardmine_outputs(report,out,make_profile=not args.no_profile)
    print(f"Hard mining [{args.task}]: {report['hard_images']}/{report['images']} hard images | FN={report['total_false_negatives']} FP={report['total_false_positives']}")
    print(result["json_path"])
    if result.get("profile_path"): print(f"Hard generation profile: {result['profile_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
