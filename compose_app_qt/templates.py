"""Qt document adapter for ScenePaste parameterized scene templates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compose_app.models import Instance
from scenepaste.core.templates import (
    build_payload_from_layout,
    load_template_data,
    parameterize_payload,
    portable_source_name,
    save_template_data,
)


def save_template(doc, path: Path, parameters: Optional[Dict[str, object]] = None) -> int:
    if doc.bg_size == (0, 0):
        raise RuntimeError("请先加载一个背景")
    if not doc.instances:
        raise RuntimeError("请先放置至少一个目标")
    bw, bh = doc.bg_size
    rows = []
    for inst in doc.instances:
        if not (0 <= inst.cutout_index < len(doc.cutouts)):
            continue
        cut = doc.cutouts[inst.cutout_index]
        rows.append({
            "source": cut.source,
            "source_name": portable_source_name(cut.source),
            "label": cut.label,
            "class_id": int(cut.class_id),
            "cx_ratio": float(inst.cx) / max(1, bw),
            "cy_ratio": float(inst.cy) / max(1, bh),
            "h_ratio": float(inst.h_ratio),
            "flip": bool(inst.flip),
            "angle": float(inst.angle),
        })
    payload = build_payload_from_layout(bw, bh, rows)
    if parameters:
        payload = parameterize_payload(payload, **parameters)
    save_template_data(payload, Path(path))
    return len(rows)


def parameterize_template_file(path: Path, **parameters) -> Path:
    data = load_template_data(path)
    data = parameterize_payload(data, **parameters)
    return save_template_data(data, path)


def load_template(doc, path: Path) -> Tuple[List[Instance], List[str]]:
    if not doc.cutouts:
        raise RuntimeError("请先加载目标素材库")
    if doc.bg_size == (0, 0):
        raise RuntimeError("请先加载一个背景")
    data = load_template_data(path)
    rows = data.get("instances", [])
    exact = {(c.source, c.label): i for i, c in enumerate(doc.cutouts)}
    by_source = {c.source: i for i, c in enumerate(doc.cutouts)}
    by_portable: Dict[str, List[int]] = {}
    by_label: Dict[str, List[int]] = {}
    for i, c in enumerate(doc.cutouts):
        by_portable.setdefault(portable_source_name(c.source), []).append(i)
        by_label.setdefault(c.label, []).append(i)
    bw, bh = doc.bg_size
    restored: List[Instance] = []
    missing: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")); label = str(row.get("label", ""))
        source_name = str(row.get("source_name", "")) or portable_source_name(source)
        idx = exact.get((source, label), by_source.get(source))
        if idx is None:
            matches = by_portable.get(source_name, [])
            if len(matches) == 1: idx = matches[0]
        if idx is None and len(by_label.get(label, [])) == 1:
            idx = by_label[label][0]
        if idx is None:
            missing.append(f"{label} ({source})"); continue
        restored.append(Instance(
            cutout_index=int(idx), cx=float(row.get("cx_ratio", .5))*bw,
            cy=float(row.get("cy_ratio", .5))*bh,
            h_ratio=max(.01, min(1.5, float(row.get("h_ratio", .3)))),
            flip=bool(row.get("flip", False)), angle=float(row.get("angle", 0.0)),
            uid=doc.next_uid(),
        ))
    if not restored:
        raise RuntimeError("模板中的目标无法与当前素材库匹配")
    return restored, missing
