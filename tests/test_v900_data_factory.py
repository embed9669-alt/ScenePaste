from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

try:
    from scenepaste.core.planning import load_hardcase_recipe, plan_label_first
except ModuleNotFoundError as exc:  # package may still be mid-refactor
    pytest.skip(f"scenepaste import unavailable: {exc}", allow_module_level=True)


def _planner_cfg():
    return SimpleNamespace(
        class_map={"person": 0, "vehicle": 1},
        min_objects=2,
        max_objects=3,
        y_min=0.35,
        y_max=0.95,
        far_height=0.08,
        near_height=0.32,
        flip_prob=0.5,
        empty_scene_probability=0.0,
        profile_strength=0.0,
    )


def test_v9_default_planner_is_explicit_label_first():
    rows = plan_label_first(_planner_cfg(), random.Random(7))
    assert 2 <= len(rows) <= 3
    assert all(x.class_id is not None and x.label for x in rows)
    assert all(x.center_x_ratio is not None for x in rows)
    assert all(x.bottom_y_ratio is not None for x in rows)
    assert all(x.height_ratio is not None and x.height_ratio > 0 for x in rows)


def test_small_object_hardcase_reduces_height():
    cfg = _planner_cfg()
    base = plan_label_first(cfg, random.Random(3))
    hard = plan_label_first(cfg, random.Random(3), hardcase_recipe=load_hardcase_recipe("small-object"))
    assert max(x.height_ratio for x in hard) < max(x.height_ratio for x in base)
    assert all(x.difficulty == "small-object" for x in hard)
