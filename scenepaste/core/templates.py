"""Parameterized ScenePaste scene-template schema and sampling.

Version 2 keeps the original portable nominal layout and adds stochastic
variation.  Version-1 templates are upgraded in memory automatically.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

SCHEMA = "scenepaste/scene-template"
VERSION = 2


@dataclass(frozen=True)
class TemplatePlacement:
    label: str
    class_id: int
    source_name: str
    cx_ratio: float
    cy_ratio: float
    h_ratio: float
    flip: bool
    angle: float
    slot_id: str = ""
    same_class_random: bool = False
    allow_overlap: bool = True

    @property
    def bottom_y_ratio(self) -> float:
        return float(self.cy_ratio + self.h_ratio / 2.0)


def portable_source_name(source: str) -> str:
    text = str(source)
    if "#" in text:
        path_text, shape = text.rsplit("#", 1)
        return f"{Path(path_text).name}#{shape}"
    return Path(text).name


def _clip(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(v)))


def _range_around(value: float, jitter: float, lo: float, hi: float) -> List[float]:
    return [_clip(value - jitter, lo, hi), _clip(value + jitter, lo, hi)]


def parameterize_payload(
    payload: dict,
    *,
    position_jitter_x: float = 0.03,
    position_jitter_y: float = 0.02,
    scale_jitter: float = 0.10,
    angle_jitter: float = 5.0,
    instance_probability: float = 1.0,
    flip_probability: Optional[float] = None,
    same_class_random: bool = False,
    allow_overlap: bool = True,
) -> dict:
    """Upgrade/copy a template and attach default variation ranges."""
    data = upgrade_template(payload)
    out = json.loads(json.dumps(data))
    out["version"] = VERSION
    out["parameters"] = {
        "position_jitter_x": float(max(0.0, position_jitter_x)),
        "position_jitter_y": float(max(0.0, position_jitter_y)),
        "scale_jitter": float(max(0.0, scale_jitter)),
        "angle_jitter": float(max(0.0, angle_jitter)),
        "instance_probability": _clip(instance_probability, 0.0, 1.0),
        "flip_probability": None if flip_probability is None else _clip(flip_probability, 0.0, 1.0),
        "same_class_random": bool(same_class_random),
        "allow_overlap": bool(allow_overlap),
    }
    for row in out.get("instances", []):
        cx = float(row.get("cx_ratio", 0.5))
        cy = float(row.get("cy_ratio", 0.5))
        hr = float(row.get("h_ratio", 0.2))
        ang = float(row.get("angle", 0.0))
        row["variation"] = {
            "cx_range": _range_around(cx, position_jitter_x, 0.0, 1.0),
            "cy_range": _range_around(cy, position_jitter_y, 0.0, 1.0),
            "h_range": [_clip(hr * (1.0 - scale_jitter), 0.01, 1.5),
                        _clip(hr * (1.0 + scale_jitter), 0.01, 1.5)],
            "angle_range": [ang - angle_jitter, ang + angle_jitter],
            "enabled_probability": _clip(instance_probability, 0.0, 1.0),
            "flip_probability": None if flip_probability is None else _clip(flip_probability, 0.0, 1.0),
            "same_class_random": bool(same_class_random),
            "allow_overlap": bool(allow_overlap),
        }
    return out


def upgrade_template(data: Mapping[str, Any]) -> dict:
    if data.get("schema") != SCHEMA:
        raise ValueError("不是 ScenePaste scene template")
    version = int(data.get("version", 1))
    if version > VERSION:
        raise ValueError(f"模板版本 {version} 高于当前支持版本 {VERSION}")
    out = json.loads(json.dumps(dict(data)))
    out.setdefault("canvas", {"width": 1, "height": 1})
    rows = out.get("instances", [])
    if not isinstance(rows, list):
        raise ValueError("模板中的 instances 必须是数组")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row.setdefault("id", f"slot_{index + 1:03d}")
        source = str(row.get("source", ""))
        row.setdefault("source_name", portable_source_name(source))
        row.setdefault("flip", False)
        row.setdefault("angle", 0.0)
    out["version"] = VERSION
    out.setdefault("constraints", [])
    out.setdefault("parameters", {
        "position_jitter_x": 0.0,
        "position_jitter_y": 0.0,
        "scale_jitter": 0.0,
        "angle_jitter": 0.0,
        "instance_probability": 1.0,
        "flip_probability": None,
        "same_class_random": False,
        "allow_overlap": True,
    })
    return out


def load_template_data(path: Path) -> dict:
    return upgrade_template(json.loads(Path(path).read_text(encoding="utf-8")))


def save_template_data(data: Mapping[str, Any], path: Path) -> Path:
    out = upgrade_template(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sample_range(row: Mapping[str, Any], key: str, fallback: float, rng: random.Random) -> float:
    values = row.get(key)
    if isinstance(values, (list, tuple)) and len(values) == 2:
        lo, hi = float(values[0]), float(values[1])
        if hi < lo:
            lo, hi = hi, lo
        return rng.uniform(lo, hi)
    return float(fallback)


def _distance(a: TemplatePlacement, b: TemplatePlacement) -> float:
    return ((a.cx_ratio - b.cx_ratio) ** 2 + (a.cy_ratio - b.cy_ratio) ** 2) ** 0.5


def template_constraints_satisfied(placements: List[TemplatePlacement], constraints) -> bool:
    """Evaluate lightweight 2D relation constraints between template slots.

    Supported types: ``min_distance``, ``max_distance``, ``left_of``,
    ``right_of``, ``above`` and ``below``. Distances/margins are normalized to
    the destination image. Missing/disabled slots make a constraint inactive.
    """
    by_id = {p.slot_id: p for p in placements if p.slot_id}
    for raw in constraints or []:
        if not isinstance(raw, Mapping):
            continue
        a = by_id.get(str(raw.get("a", "")))
        b = by_id.get(str(raw.get("b", "")))
        if a is None or b is None:
            continue
        kind = str(raw.get("type", "")).strip().lower()
        margin = max(0.0, float(raw.get("margin", 0.0) or 0.0))
        if kind == "min_distance" and _distance(a, b) < float(raw.get("value", 0.0) or 0.0):
            return False
        if kind == "max_distance" and _distance(a, b) > float(raw.get("value", 1.0) or 1.0):
            return False
        if kind == "left_of" and not (a.cx_ratio + margin < b.cx_ratio):
            return False
        if kind == "right_of" and not (a.cx_ratio - margin > b.cx_ratio):
            return False
        if kind == "above" and not (a.cy_ratio + margin < b.cy_ratio):
            return False
        if kind == "below" and not (a.cy_ratio - margin > b.cy_ratio):
            return False
    return True


def _sample_template_once(tpl: Mapping[str, Any], rng: random.Random, class_map: Mapping[str, int]) -> List[TemplatePlacement]:
    global_params = tpl.get("parameters", {})
    result: List[TemplatePlacement] = []
    for raw in tpl.get("instances", []):
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", ""))
        if label not in class_map:
            continue
        var = dict(raw.get("variation") or {})
        enabled = float(var.get("enabled_probability", global_params.get("instance_probability", 1.0)))
        if rng.random() > _clip(enabled, 0.0, 1.0):
            continue
        cx0 = float(raw.get("cx_ratio", 0.5))
        cy0 = float(raw.get("cy_ratio", 0.5))
        h0 = float(raw.get("h_ratio", 0.2))
        a0 = float(raw.get("angle", 0.0))
        if "cx_range" not in var:
            var["cx_range"] = _range_around(cx0, float(global_params.get("position_jitter_x", 0.0)), 0.0, 1.0)
        if "cy_range" not in var:
            var["cy_range"] = _range_around(cy0, float(global_params.get("position_jitter_y", 0.0)), 0.0, 1.0)
        if "h_range" not in var:
            sj = float(global_params.get("scale_jitter", 0.0))
            var["h_range"] = [_clip(h0 * (1.0 - sj), 0.01, 1.5), _clip(h0 * (1.0 + sj), 0.01, 1.5)]
        if "angle_range" not in var:
            aj = float(global_params.get("angle_jitter", 0.0))
            var["angle_range"] = [a0 - aj, a0 + aj]
        fp = var.get("flip_probability", global_params.get("flip_probability"))
        if fp is None:
            flip = bool(raw.get("flip", False))
        else:
            flip = rng.random() < _clip(float(fp), 0.0, 1.0)
        result.append(TemplatePlacement(
            label=label,
            class_id=int(class_map[label]),
            source_name=str(raw.get("source_name", portable_source_name(str(raw.get("source", ""))))),
            cx_ratio=_clip(_sample_range(var, "cx_range", cx0, rng), 0.0, 1.0),
            cy_ratio=_clip(_sample_range(var, "cy_range", cy0, rng), 0.0, 1.0),
            h_ratio=_clip(_sample_range(var, "h_range", h0, rng), 0.01, 1.5),
            flip=flip,
            angle=_sample_range(var, "angle_range", a0, rng),
            slot_id=str(raw.get("id", "")),
            same_class_random=bool(var.get("same_class_random", global_params.get("same_class_random", False))),
            allow_overlap=bool(var.get("allow_overlap", global_params.get("allow_overlap", True))),
        ))
    return result


def sample_template(
    data: Mapping[str, Any],
    rng: random.Random,
    class_map: Mapping[str, int],
) -> List[TemplatePlacement]:
    """Sample one stochastic layout and enforce optional relation constraints."""
    tpl = upgrade_template(data)
    constraints = tpl.get("constraints", [])
    tries = max(1, int(tpl.get("parameters", {}).get("constraint_tries", 50) or 50))
    last: List[TemplatePlacement] = []
    for _ in range(tries):
        last = _sample_template_once(tpl, rng, class_map)
        if template_constraints_satisfied(last, constraints):
            return last
    # Returning an empty plan makes the generation task fail visibly instead
    # of silently violating a user-declared scene relation.
    return []


def build_payload_from_layout(canvas_width: int, canvas_height: int, rows: Iterable[Mapping[str, Any]]) -> dict:
    """Build a v2 exact template from portable nominal layout rows."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "canvas": {"width": int(canvas_width), "height": int(canvas_height)},
        "constraints": [],
        "parameters": {
            "position_jitter_x": 0.0,
            "position_jitter_y": 0.0,
            "scale_jitter": 0.0,
            "angle_jitter": 0.0,
            "instance_probability": 1.0,
            "flip_probability": None,
            "same_class_random": False,
            "allow_overlap": True,
        },
        "instances": [dict({"id": f"slot_{i + 1:03d}"}, **dict(r)) for i, r in enumerate(rows)],
    }
