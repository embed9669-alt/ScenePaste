"""Formal label-first scene planning for ScenePaste V9.

Older ScenePaste versions sometimes delegated class/position choice to the
renderer by emitting an empty PlacementSpec.  V9 makes the plan explicit before
pixels are rendered so every target has a class, normalized location and scale.
That plan can be audited, serialized and later used by generative backends.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .distribution import DistributionProfile
from .models import PlacementSpec
from .templates import sample_template


BUILTIN_HARDCASE_RECIPES = {
    "small-object": {
        "height_scale": [0.35, 0.65],
        "bottom_y_bias": "far",
        "overlap_probability": 0.05,
        "cluster_probability": 0.0,
        "count_bias": "normal",
    },
    "far-occluded": {
        "height_scale": [0.35, 0.70],
        "bottom_y_bias": "far",
        "overlap_probability": 0.65,
        "cluster_probability": 0.70,
        "count_bias": "max",
    },
    "crowded": {
        "height_scale": [0.75, 1.10],
        "bottom_y_bias": "normal",
        "overlap_probability": 0.45,
        "cluster_probability": 0.55,
        "count_bias": "max",
    },
}


def load_hardcase_recipe(source: Optional[object]) -> Optional[dict]:
    if source is None or source == "" or str(source).lower() in {"off", "none"}:
        return None
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        text = str(source)
        if text in BUILTIN_HARDCASE_RECIPES:
            data = dict(BUILTIN_HARDCASE_RECIPES[text])
            data["name"] = text
        else:
            path = Path(text).expanduser()
            if not path.is_file():
                raise ValueError(f"unknown hard-case recipe or missing JSON: {source}")
            data = json.loads(path.read_text(encoding="utf-8"))
    scale = data.get("height_scale", [1.0, 1.0])
    if not isinstance(scale, Sequence) or len(scale) != 2:
        raise ValueError("hard-case height_scale must be [min,max]")
    lo, hi = float(scale[0]), float(scale[1])
    if lo <= 0 or hi < lo:
        raise ValueError("invalid hard-case height_scale")
    data["height_scale"] = [lo, hi]
    for key in ("overlap_probability", "cluster_probability"):
        v = float(data.get(key, 0.0))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hard-case {key} must be in 0..1")
        data[key] = v
    return data


def _default_position(cfg, rng: random.Random, hard: Optional[dict], previous: list[PlacementSpec]):
    y_min, y_max = float(cfg.y_min), float(cfg.y_max)
    bias = str((hard or {}).get("bottom_y_bias", "normal"))
    if bias == "far":
        upper = y_min + (y_max - y_min) * 0.48
        bottom_y = rng.uniform(y_min, upper)
    elif bias == "near":
        lower = y_min + (y_max - y_min) * 0.52
        bottom_y = rng.uniform(lower, y_max)
    else:
        bottom_y = rng.uniform(y_min, y_max)

    cluster_probability = float((hard or {}).get("cluster_probability", 0.0))
    if previous and rng.random() < cluster_probability and previous[-1].center_x_ratio is not None:
        cx = float(previous[-1].center_x_ratio) + rng.uniform(-0.08, 0.08)
        cx = min(0.96, max(0.04, cx))
    else:
        cx = rng.uniform(0.06, 0.94)

    t = max(0.0, min(1.0, (bottom_y - y_min) / max(1e-6, y_max - y_min)))
    h = float(cfg.far_height) + t * (float(cfg.near_height) - float(cfg.far_height))
    h *= rng.uniform(0.82, 1.18)
    if hard:
        lo, hi = hard.get("height_scale", [1.0, 1.0])
        h *= rng.uniform(float(lo), float(hi))
    h = min(0.85, max(0.01, h))
    return cx, bottom_y, h


def plan_label_first(
    cfg,
    rng: random.Random,
    profile: Optional[DistributionProfile] = None,
    template_data: Optional[dict] = None,
    hardcase_recipe: Optional[dict] = None,
) -> list[PlacementSpec]:
    """Return explicit placement specs for one sample before rendering."""

    if cfg.empty_scene_probability > 0 and rng.random() < cfg.empty_scene_probability:
        return []
    if template_data is not None:
        sampled = sample_template(template_data, rng, cfg.class_map)
        if not sampled:
            sampled = sample_template(template_data, rng, cfg.class_map)
        return [
            PlacementSpec(
                class_id=p.class_id,
                label=p.label,
                source_name=p.source_name,
                center_x_ratio=p.cx_ratio,
                bottom_y_ratio=p.bottom_y_ratio,
                height_ratio=p.h_ratio,
                flip=p.flip,
                angle=p.angle,
                allow_overlap=p.allow_overlap,
                same_class_random=p.same_class_random,
                difficulty=(hardcase_recipe or {}).get("name"),
            )
            for p in sampled
        ]

    use_profile = profile is not None and rng.random() < float(cfg.profile_strength)
    if use_profile:
        n = profile.sample_object_count(rng, cfg.min_objects, cfg.max_objects)
        rows: list[PlacementSpec] = []
        for _ in range(n):
            label = profile.sample_label(rng, cfg.class_map)
            if label is None:
                label = rng.choice([k for k, _v in sorted(cfg.class_map.items(), key=lambda kv: kv[1])])
                cx, by, hr = _default_position(cfg, rng, hardcase_recipe, rows)
            else:
                pos = profile.sample_placement(label, rng)
                cx, by, hr = pos["center_x_ratio"], pos["bottom_y_ratio"], pos["height_ratio"]
                if hardcase_recipe:
                    lo, hi = hardcase_recipe.get("height_scale", [1.0, 1.0])
                    hr = float(hr) * rng.uniform(float(lo), float(hi))
            rows.append(
                PlacementSpec(
                    class_id=int(cfg.class_map[label]),
                    label=label,
                    center_x_ratio=float(cx),
                    bottom_y_ratio=float(by),
                    height_ratio=float(hr),
                    allow_overlap=bool(rng.random() < float((hardcase_recipe or {}).get("overlap_probability", 0.0))),
                    difficulty=(hardcase_recipe or {}).get("name"),
                )
            )
        return rows

    lo, hi = int(cfg.min_objects), int(cfg.max_objects)
    if hardcase_recipe and str(hardcase_recipe.get("count_bias")) == "max":
        n = hi
    else:
        n = rng.randint(lo, hi)
    labels = [k for k, _v in sorted(cfg.class_map.items(), key=lambda kv: kv[1])]
    fallback_rows: list[PlacementSpec] = []
    for _ in range(n):
        label = rng.choice(labels)
        cx, by, hr = _default_position(cfg, rng, hardcase_recipe, fallback_rows)
        fallback_rows.append(
            PlacementSpec(
                class_id=int(cfg.class_map[label]),
                label=label,
                center_x_ratio=cx,
                bottom_y_ratio=by,
                height_ratio=hr,
                flip=bool(rng.random() < float(cfg.flip_prob)),
                allow_overlap=bool(rng.random() < float((hardcase_recipe or {}).get("overlap_probability", 0.0))),
                difficulty=(hardcase_recipe or {}).get("name"),
            )
        )
    return fallback_rows


def placement_to_dict(spec: PlacementSpec) -> dict:
    return {
        "class_id": spec.class_id,
        "label": spec.label,
        "source_name": spec.source_name,
        "center_x_ratio": spec.center_x_ratio,
        "bottom_y_ratio": spec.bottom_y_ratio,
        "height_ratio": spec.height_ratio,
        "flip": spec.flip,
        "angle": spec.angle,
        "allow_overlap": spec.allow_overlap,
        "same_class_random": spec.same_class_random,
        "difficulty": spec.difficulty,
    }
