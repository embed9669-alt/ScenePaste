"""Advanced deterministic planner + resumable multiprocessing generation.

ScenePaste routes every dataset generation call through this engine, even
when ``workers=1``. Each sample index is deterministic and independently writable,
which keeps one-worker and multi-worker runs consistent while enabling crash-safe
resume and bounded multiprocessing queues.
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ..formats import (
    obb_lines_for_annotations,
    seg_lines_for_annotations,
    write_classes_file,
    write_data_yaml,
    write_semantic_classes,
    yolo_line,
)
from .auto_cutout import load_object_assets_auto
from .config import GenerationConfig, validate_config
from .distribution import DistributionProfile
from .geometry import annotation_canvas_mask, canvas_polygon, mask_bbox, mask_to_polygons
from .io import imread_with_exif, imwrite_unicode, list_backgrounds, log_default, save_cutouts
from .labelme import load_object_assets
from .models import ObjectAsset, PlacementSpec
from .runstate import RunStateStore, find_latest_resumable_run, stable_config_hash
from .sampling import BackgroundCache, BackgroundSampler, build_asset_groups, sample_asset
from .synthesis import paste_one
from .templates import load_template_data, portable_source_name
from .validation import is_valid_yolo_box
from .recipes import apply_scene_recipe, load_augmentation_recipe
from .object_appearance import load_object_appearance_recipe
from .planning import load_hardcase_recipe, placement_to_dict, plan_label_first
from .scene_understanding import resolve_placement_regions


@dataclass(frozen=True)
class PlannedObject:
    placement: PlacementSpec


@dataclass(frozen=True)
class SamplePlan:
    index: int
    background: str
    seed: int
    objects: List[PlannedObject]


@dataclass
class TaskResult:
    index: int
    ok: bool
    stem: str = ""
    objects: int = 0
    error: str = ""


@dataclass
class _WorkerContext:
    config: GenerationConfig
    assets: List[ObjectAsset]
    groups: Dict[int, List[ObjectAsset]]
    by_identity: Dict[str, List[ObjectAsset]]
    bg_cache: Optional[BackgroundCache]
    run_id: str
    fragment_root: Path
    augmentation_recipe: Optional[dict]
    object_appearance_recipe: Optional[dict]


_CTX: Optional[_WorkerContext] = None


def _task_seed(seed: int, index: int) -> int:
    return (int(seed) * 1000003 + int(index) * 9176 + 0x5EED123) & 0xFFFFFFFF


def _load_assets(config: GenerationConfig, log: Callable[[str], None]) -> List[ObjectAsset]:
    if config.auto_cutout:
        return load_object_assets_auto(
            config.objects_dir, config.class_map, log,
            label=config.auto_cutout_label,
            label_from_subdir=bool(config.auto_cutout_label_from_subdir),
        )
    return load_object_assets(
        config.objects_dir, config.class_map, config.feather_sigma, log,
        rectangle_mask_mode=config.rectangle_mask_mode,
    )


def _init_worker(config: GenerationConfig, run_id: str, fragment_root: str) -> None:
    global _CTX
    assets = _load_assets(config, lambda _msg: None)
    groups = build_asset_groups(assets)
    by_identity: Dict[str, List[ObjectAsset]] = {}
    for asset in assets:
        identity = portable_source_name(f"{asset.source_json}#{asset.source_shape_index}")
        by_identity.setdefault(identity, []).append(asset)
    _CTX = _WorkerContext(
        config=config,
        assets=assets,
        groups=groups,
        by_identity=by_identity,
        bg_cache=BackgroundCache(config.background_cache_size) if config.background_cache_size > 0 else None,
        run_id=run_id,
        fragment_root=Path(fragment_root),
        augmentation_recipe=load_augmentation_recipe(config.augmentation_recipe),
        object_appearance_recipe=load_object_appearance_recipe(config.object_appearance_recipe),
    )


def _asset_for_plan(ctx: _WorkerContext, spec: PlacementSpec, rng: random.Random) -> ObjectAsset:
    if spec.source_name and not spec.same_class_random:
        matches = ctx.by_identity.get(spec.source_name, [])
        if spec.label:
            labeled = [a for a in matches if a.label == spec.label]
            if labeled:
                return rng.choice(labeled)
        if matches:
            return rng.choice(matches)
    if spec.class_id is not None and spec.class_id in ctx.groups:
        return rng.choice(ctx.groups[spec.class_id])
    return sample_asset(rng, ctx.assets, ctx.groups, ctx.config.asset_sampling)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _coco_fragment(index: int, stem: str, bg_path: Path, image_path: Path,
                   width: int, height: int, annotations, visible_masks) -> dict:
    anns = []
    for idx, (asset, box, flipped, transform) in enumerate(annotations):
        x1, y1, x2, y2 = box
        segmentation = []
        area = 0.0
        coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if visible_masks is not None and idx < len(visible_masks):
            mask = visible_masks[idx]
            polys = mask_to_polygons(mask)
            segmentation = [[float(v) for v in p.reshape(-1)] for p in polys]
            area = float(np.count_nonzero(mask))
            mb = mask_bbox(mask)
            if mb is not None:
                mx1, my1, mx2, my2 = mb
                coco_box = [float(mx1), float(my1), float(mx2 - mx1), float(my2 - my1)]
        if not segmentation:
            poly = canvas_polygon(asset, box, flipped, width, height, transform)
            if poly is None:
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            segmentation = [[float(v) for v in poly.reshape(-1)]]
            area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                             - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
        anns.append({"category_id": int(asset.class_id), "segmentation": segmentation,
                     "bbox": coco_box, "area": float(area), "iscrowd": 0})
    return {
        "index": int(index),
        "image": {"file_name": str(Path("images/train") / image_path.name),
                  "width": int(width), "height": int(height),
                  "background": str(bg_path), "stem": stem},
        "annotations": anns,
    }


def _visible_masks_from_raw(raw_masks: List[np.ndarray]) -> List[np.ndarray]:
    """Apply z-order occlusion to already-canvas-aligned masks."""
    if not raw_masks:
        return []
    h, w = raw_masks[0].shape[:2]
    occupied = np.zeros((h, w), dtype=bool)
    visible: List[np.ndarray] = [np.zeros((h, w), dtype=bool) for _ in raw_masks]
    for idx in range(len(raw_masks) - 1, -1, -1):
        raw = np.asarray(raw_masks[idx]) > 0
        visible[idx] = raw & ~occupied
        occupied |= raw
    return visible


def _write_task(plan: SamplePlan) -> TaskResult:
    global _CTX
    if _CTX is None:
        return TaskResult(plan.index, False, error="worker context not initialized")
    ctx = _CTX
    cfg = ctx.config
    rng = random.Random(plan.seed)
    try:
        bg_path = Path(plan.background)
        background = ctx.bg_cache.get(bg_path) if ctx.bg_cache is not None else imread_with_exif(bg_path)
        if background is None:
            raise RuntimeError(f"无法读取背景：{bg_path}")
        canvas = background.copy()
        height, width = canvas.shape[:2]
        zone_mask, class_zone_masks, scene_region_meta = resolve_placement_regions(
            bg_path, height, width, mode=cfg.scene_region_mode, ground_start_ratio=cfg.y_min
        )
        boxes = []
        annotations_ext = []

        for object_index, planned in enumerate(plan.objects):
            spec = planned.placement
            asset = _asset_for_plan(ctx, spec, rng)
            effective_zone = class_zone_masks.get(asset.label.casefold(), zone_mask)
            result = paste_one(
                canvas, asset, boxes, effective_zone, cfg, rng, placement=spec,
                object_appearance_recipe=ctx.object_appearance_recipe,
            )
            if result is None:
                continue
            box, flipped, transform, appearance = result
            boxes.append(box)
            annotations_ext.append((asset, box, flipped, transform, appearance))

        intentional_empty = len(plan.objects) == 0
        if not annotations_ext and not intentional_empty:
            raise RuntimeError("未找到合适的粘贴位置")

        valid = []
        for row in annotations_ext:
            asset, box, flipped, transform, appearance = row
            ok, clipped = is_valid_yolo_box(box, width, height, min_visible=cfg.min_visible_ratio)
            if ok:
                valid.append((asset, clipped, flipped, transform, appearance))
        annotations_ext = valid
        if not annotations_ext and not intentional_empty:
            raise RuntimeError("所有目标越界/过小")

        appearance_by_idx = [row[4] for row in annotations_ext]
        annotations = [(a, b, f, t) for a, b, f, t, _app in annotations_ext]

        # Image-only domain recipe runs last; geometry remains unchanged.
        canvas, applied_effects = apply_scene_recipe(canvas, ctx.augmentation_recipe, rng)

        prefix = f"{ctx.run_id}_{cfg.seed}"
        stem = f"{prefix}_{plan.index:06d}"
        out = cfg.output_dir
        image_path = out / "images" / "train" / f"{stem}.jpg"
        label_path = out / "labels" / "train" / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(image_path, canvas, jpeg_quality=95):
            raise RuntimeError("图像写入失败")

        fmt = (cfg.output_format or "detect").lower()
        visible_masks = None
        visibility_rows = []
        if fmt in {"seg", "both", "coco", "semantic", "obb", "all"}:
            raw_masks = []
            for asset, _box, _f, transform in annotations:
                raw_masks.append(annotation_canvas_mask(asset, transform, width, height) > 0)
            visible_masks = _visible_masks_from_raw(raw_masks)
            for raw, vis in zip(raw_masks, visible_masks):
                raw_area = int(np.count_nonzero(raw)); vis_area = int(np.count_nonzero(vis))
                visibility_rows.append(vis_area / raw_area if raw_area else 0.0)

        if fmt in {"detect", "both", "coco", "all"}:
            detect_lines = [yolo_line(a.class_id, box, width, height) for a, box, _f, _t in annotations]
            label_path.write_text(("\n".join(detect_lines) + "\n") if detect_lines else "", encoding="utf-8")
        if fmt in {"seg", "both", "all"}:
            seg_dir = out / "labels" / "train" if fmt == "seg" else out / "labels-seg" / "train"
            seg_dir.mkdir(parents=True, exist_ok=True)
            lines = seg_lines_for_annotations(annotations, visible_masks, width, height)
            (seg_dir / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        if fmt in {"obb", "all"}:
            lines = obb_lines_for_annotations(annotations, visible_masks, width, height)
            obb_path = label_path if fmt == "obb" else (out / "labels-obb" / "train" / f"{stem}.txt")
            obb_path.parent.mkdir(parents=True, exist_ok=True)
            obb_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        if fmt in {"semantic", "all"}:
            max_value = max(cfg.class_map.values(), default=0) + 1
            dtype = np.uint8 if max_value <= 255 else np.uint16
            semantic = np.zeros((height, width), dtype=dtype)
            for (asset, _box, _f, _t), vis in zip(annotations, visible_masks or []):
                semantic[vis] = int(asset.class_id) + 1
            masks = out / "masks" / "train"
            masks.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(masks / f"{stem}.png", semantic)
        if fmt in {"coco", "all"}:
            frag = _coco_fragment(plan.index, stem, bg_path, image_path, width, height, annotations, visible_masks)
            _atomic_json(ctx.fragment_root / "coco" / f"{plan.index:09d}.json", frag)

        if cfg.save_previews and cfg.preview_ratio > 0 and rng.random() < cfg.preview_ratio:
            preview = canvas.copy()
            for asset, box, _f, _t in annotations:
                x1, y1, x2, y2 = box
                color = (40, 220, 40) if asset.class_id == 0 else (40, 160, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(preview, asset.label, (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            previews = out / "previews"
            previews.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(previews / f"{stem}.jpg", preview, jpeg_quality=92)

        log_rows = []
        for ann_idx, (asset, box, flipped, transform) in enumerate(annotations):
            angle = float(transform[4]) if len(transform) >= 5 else 0.0
            object_effects = appearance_by_idx[ann_idx] if ann_idx < len(appearance_by_idx) else []
            log_rows.append({
                "generated_stem": stem, "background": str(bg_path), "label": asset.label,
                "class_id": int(asset.class_id), "source_json": str(asset.source_json),
                "shape_index": int(asset.source_shape_index), "flipped": 1 if flipped else 0,
                "angle": angle, "box_xyxy": ",".join(map(str, box)),
                "used_paste_zone": 1 if (zone_mask is not None or asset.label.casefold() in class_zone_masks) else 0,
                "scene_effects": json.dumps(applied_effects, ensure_ascii=False, separators=(",", ":")),
                "object_effects": json.dumps(object_effects, ensure_ascii=False, separators=(",", ":")),
                "visible_ratio": float(visibility_rows[ann_idx]) if ann_idx < len(visibility_rows) else None,
                "run_id": ctx.run_id,
                "task_index": int(plan.index),
            })
        _atomic_json(
            ctx.fragment_root / "meta" / f"{plan.index:09d}.json",
            {
                "schema": "scenepaste/sample-metadata",
                "version": 9,
                "index": plan.index,
                "stem": stem,
                "objects": len(annotations),
                "intentional_empty": bool(intentional_empty),
                "planned_objects": [placement_to_dict(p.placement) for p in plan.objects],
                "scene_region": scene_region_meta,
                "visible_ratios": visibility_rows,
                "scene_effects": applied_effects,
                "object_effects": appearance_by_idx,
                "rows": log_rows,
            },
        )
        return TaskResult(plan.index, True, stem=stem, objects=len(annotations))
    except Exception as exc:
        return TaskResult(plan.index, False, error=str(exc))


def _finalize_log(fragment_root: Path, path: Path, completed: bytearray) -> None:
    # Keep the v0.5 CSV contract stable; resume-specific task metadata lives in SQLite/fragments.
    header = ["generated_stem", "background", "label", "class_id", "source_json",
              "shape_index", "flipped", "box_xyxy", "used_paste_zone", "run_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for idx, done in enumerate(completed):
            if not done:
                continue
            p = fragment_root / "meta" / f"{idx:09d}.json"
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                for row in payload.get("rows", []):
                    writer.writerow({k: row.get(k, "") for k in header})
            except Exception:
                continue


def _finalize_generation_diagnostics(fragment_root: Path, path: Path, completed: bytearray) -> dict:
    bins = [0] * 10
    by_class: Dict[str, list] = {}
    effects: Dict[str, int] = {}
    object_effects: Dict[str, int] = {}
    images = objects = negatives = 0
    for idx, done in enumerate(completed):
        if not done:
            continue
        p = fragment_root / "meta" / f"{idx:09d}.json"
        if not p.exists():
            continue
        try: payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        images += 1; objects += int(payload.get("objects", 0))
        if payload.get("intentional_empty"): negatives += 1
        for effect in payload.get("scene_effects", []) or []:
            if isinstance(effect, dict):
                key = str(effect.get("effect") or effect.get("name") or effect)
            else:
                key = str(effect)
            effects[key] = effects.get(key, 0) + 1
        for instance_effects in payload.get("object_effects", []) or []:
            for effect in instance_effects or []:
                if isinstance(effect, dict):
                    key = str(effect.get("effect") or effect.get("name") or effect)
                else:
                    key = str(effect)
                object_effects[key] = object_effects.get(key, 0) + 1
        rows = payload.get("rows", []) or []
        ratios = payload.get("visible_ratios", []) or []
        for i, ratio in enumerate(ratios):
            try: value = min(1.0, max(0.0, float(ratio)))
            except Exception: continue
            bins[min(9, int(value * 10))] += 1
            label = str(rows[i].get("label", "unknown")) if i < len(rows) else "unknown"
            by_class.setdefault(label, []).append(value)
    result = {
        "schema": "scenepaste/generation-diagnostics", "version": 1,
        "images": images, "objects": objects, "intentional_empty_images": negatives,
        "visibility_samples": int(sum(bins)),
        "visible_ratio": {"bins": 10, "range": [0.0, 1.0], "counts": bins},
        "visible_ratio_by_class": {
            k: {"count": len(v), "mean": float(sum(v)/len(v)), "min": float(min(v)), "max": float(max(v))}
            for k, v in sorted(by_class.items()) if v
        },
        "scene_effect_counts": dict(sorted(effects.items())),
        "object_effect_counts": dict(sorted(object_effects.items())),
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _finalize_coco(fragment_root: Path, path: Path, completed: bytearray, class_map: Dict[str, int]) -> None:
    """Stream a standard COCO JSON from per-image fragments without loading all annotations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    valid = []
    for idx, done in enumerate(completed):
        p = fragment_root / "coco" / f"{idx:09d}.json"
        if done and p.exists():
            valid.append(p)
    with tmp.open("w", encoding="utf-8") as f:
        f.write('{"images":[')
        first = True
        image_id = 1
        image_ids = {}
        for p in valid:
            payload = json.loads(p.read_text(encoding="utf-8"))
            image = dict(payload["image"])
            image["id"] = image_id
            image_ids[p.name] = image_id
            if not first: f.write(",")
            json.dump(image, f, ensure_ascii=False, separators=(",", ":"))
            first = False
            image_id += 1
        f.write('],"annotations":[')
        first = True
        ann_id = 1
        for p in valid:
            payload = json.loads(p.read_text(encoding="utf-8"))
            iid = image_ids[p.name]
            for raw in payload.get("annotations", []):
                ann = dict(raw); ann["id"] = ann_id; ann["image_id"] = iid
                if not first: f.write(",")
                json.dump(ann, f, ensure_ascii=False, separators=(",", ":"))
                first = False; ann_id += 1
        f.write('],"categories":')
        cats = [{"id": cid, "name": name, "supercategory": "object"}
                for name, cid in sorted(class_map.items(), key=lambda kv: kv[1])]
        json.dump(cats, f, ensure_ascii=False, separators=(",", ":"))
        f.write("}")
    tmp.replace(path)


def _config_payload(cfg: GenerationConfig) -> dict:
    payload = {
        "objects_dir": str(Path(cfg.objects_dir).resolve()),
        "backgrounds_dir": str(Path(cfg.backgrounds_dir).resolve()),
        "class_map": dict(cfg.class_map), "count": cfg.count,
        "min_objects": cfg.min_objects, "max_objects": cfg.max_objects,
        "y_min": cfg.y_min, "y_max": cfg.y_max, "far_height": cfg.far_height,
        "near_height": cfg.near_height, "max_iou": cfg.max_iou, "flip_prob": cfg.flip_prob,
        "blur_prob": cfg.blur_prob, "color_match_strength": cfg.color_match_strength,
        "feather_sigma": cfg.feather_sigma, "seed": cfg.seed,
        "output_format": cfg.output_format, "auto_cutout": cfg.auto_cutout,
        "rectangle_mask_mode": cfg.rectangle_mask_mode,
        "min_visible_ratio": cfg.min_visible_ratio, "asset_sampling": cfg.asset_sampling,
        "background_sampling": cfg.background_sampling, "profile_strength": cfg.profile_strength,
        "distribution_profile": str(Path(cfg.distribution_profile).resolve()) if cfg.distribution_profile else None,
        "scene_template": str(Path(cfg.scene_template).resolve()) if cfg.scene_template else None,
        "empty_scene_probability": cfg.empty_scene_probability,
        "augmentation_recipe": cfg.augmentation_recipe,
        "object_appearance_recipe": cfg.object_appearance_recipe,
        "blend_mode": cfg.blend_mode, "blend_sigma": cfg.blend_sigma,
        "scene_region_mode": cfg.scene_region_mode,
        "hardcase_recipe": cfg.hardcase_recipe,
    }
    # Include profile/template content so editing a file cannot silently resume a stale plan.
    import hashlib
    for key, p in (("distribution_profile_sha256", cfg.distribution_profile),
                   ("scene_template_sha256", cfg.scene_template)):
        payload[key] = hashlib.sha256(Path(p).read_bytes()).hexdigest() if p else None
    if cfg.augmentation_recipe and Path(str(cfg.augmentation_recipe)).is_file():
        payload["augmentation_recipe_sha256"] = hashlib.sha256(Path(str(cfg.augmentation_recipe)).read_bytes()).hexdigest()
    else:
        payload["augmentation_recipe_sha256"] = None
    if cfg.object_appearance_recipe and Path(str(cfg.object_appearance_recipe)).is_file():
        payload["object_appearance_recipe_sha256"] = hashlib.sha256(
            Path(str(cfg.object_appearance_recipe)).read_bytes()
        ).hexdigest()
    else:
        payload["object_appearance_recipe_sha256"] = None
    if cfg.hardcase_recipe and Path(str(cfg.hardcase_recipe)).is_file():
        payload["hardcase_recipe_sha256"] = hashlib.sha256(
            Path(str(cfg.hardcase_recipe)).read_bytes()
        ).hexdigest()
    else:
        payload["hardcase_recipe_sha256"] = None
    return payload


def _plan_objects(cfg: GenerationConfig, rng: random.Random,
                  profile: Optional[DistributionProfile], template_data: Optional[dict],
                  hardcase_recipe: Optional[dict] = None) -> List[PlannedObject]:
    """Compatibility wrapper around the V9 explicit label-first planner."""
    return [PlannedObject(spec) for spec in plan_label_first(
        cfg, rng, profile=profile, template_data=template_data, hardcase_recipe=hardcase_recipe
    )]



def _write_live_status(output: Path, run_id: str, *, status: str, completed: int,
                       failed: int, requested: int, objects: int, started: float,
                       workers: int) -> dict:
    elapsed = max(1e-6, time.monotonic() - started)
    rate = completed / elapsed
    remaining = max(0, requested - completed)
    eta = (remaining / rate) if rate > 1e-9 else None
    try:
        free_bytes = shutil.disk_usage(output).free
    except Exception:
        free_bytes = None
    payload = {
        "schema": "scenepaste/run-status", "version": 1,
        "run_id": run_id, "status": status, "completed": int(completed),
        "failed": int(failed), "requested": int(requested), "objects": int(objects),
        "workers": int(workers), "elapsed_seconds": float(elapsed),
        "images_per_second": float(rate), "eta_seconds": float(eta) if eta is not None else None,
        "disk_free_bytes": int(free_bytes) if free_bytes is not None else None,
    }
    path = output / ".scenepaste" / "status" / f"{run_id}.json"
    _atomic_json(path, payload)
    return payload

def generate_dataset_advanced(config: GenerationConfig, log: Callable[[str], None] = log_default,
                              cancel_event=None) -> dict:
    validate_config(config)
    output = Path(config.output_dir)
    for d in (output / "images" / "train", output / "labels" / "train", output / "previews"):
        d.mkdir(parents=True, exist_ok=True)
    if config.output_format in {"semantic", "all"}:
        (output / "masks" / "train").mkdir(parents=True, exist_ok=True)

    if config.resume:
        run_id = config.run_id or find_latest_resumable_run(output)
        if not run_id:
            raise RuntimeError("没有找到可恢复的 ScenePaste run；请指定 --run-id 或先启动一次生成")
    else:
        run_id = config.run_id or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    config.run_id = run_id

    log("正在读取目标素材与背景……")
    assets = _load_assets(config, log)
    backgrounds = list_backgrounds(config.backgrounds_dir)
    if not backgrounds:
        raise RuntimeError(f"背景目录为空：{config.backgrounds_dir}")
    save_cutouts(assets, output)
    write_classes_file(config.class_map, output)
    write_data_yaml(output, config.class_map)
    if config.output_format in {"semantic", "all"}:
        write_semantic_classes(config.class_map, output)
    log(f"提取 {len(assets)} 个目标，找到 {len(backgrounds)} 张背景")
    # Parent does not need decoded cutouts after setup; workers own their copies.
    del assets; gc.collect()

    profile = DistributionProfile.load(config.distribution_profile) if config.distribution_profile else None
    template_data = load_template_data(config.scene_template) if config.scene_template else None
    hardcase_recipe = load_hardcase_recipe(config.hardcase_recipe)
    if profile:
        target_data = json.loads(json.dumps(profile.data))
        target_data["classes"] = {k: v for k, v in target_data.get("classes", {}).items() if k in config.class_map}
        target_total = sum(int(v.get("count", 0)) for v in target_data["classes"].values())
        target_data["object_count_total"] = target_total
        target_data["class_probabilities"] = {
            k: (float(v.get("count", 0)) / target_total if target_total else 0.0)
            for k, v in target_data["classes"].items()
        }
        (output / "target_distribution_profile.json").write_text(
            json.dumps(target_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if template_data:
        (output / "active_scene_template.json").write_text(
            json.dumps(template_data, ensure_ascii=False, indent=2), encoding="utf-8")

    fragment_root = output / ".scenepaste" / "fragments" / run_id
    (fragment_root / "meta").mkdir(parents=True, exist_ok=True)
    if config.output_format in {"coco", "all"}:
        (fragment_root / "coco").mkdir(parents=True, exist_ok=True)

    payload = _config_payload(config)
    cfg_hash = stable_config_hash(payload)
    state = RunStateStore(output, run_id)
    state.initialize(config_hash=cfg_hash, count=config.count, config_payload=payload)
    completed = state.completed_bitmap(config.count)
    already = int(sum(completed))
    if already:
        log(f"恢复 Run {run_id}：已完成 {already}/{config.count}，只补剩余任务")

    bg_rng = random.Random(config.seed ^ 0x3B9ACA07)
    bg_sampler = BackgroundSampler(backgrounds, bg_rng, config.background_sampling)

    def plans():
        for idx in range(config.count):
            bg = bg_sampler.next()  # advance even for completed tasks => deterministic resume
            if completed[idx]:
                continue
            rng = random.Random(_task_seed(config.seed, idx))
            objs = _plan_objects(config, rng, profile, template_data, hardcase_recipe)
            yield SamplePlan(idx, str(bg), _task_seed(config.seed, idx), objs)

    workers = int(config.workers)
    if workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, workers)
    queue_depth = int(config.queue_depth) or max(2, workers * 2)
    done_now = failed_now = 0
    interrupted = False
    started_monotonic = time.monotonic()
    _write_live_status(output, run_id, status="running", completed=already, failed=0,
                       requested=config.count, objects=state.counts().get("objects", 0),
                       started=started_monotonic, workers=workers)

    def handle(result: TaskResult):
        nonlocal done_now, failed_now
        if result.ok:
            state.record_completed(result.index, result.stem, result.objects)
            completed[result.index] = 1
            done_now += 1
        else:
            state.record_failed(result.index, result.error)
            failed_now += 1
            log(f"[失败 index={result.index}] {result.error}")
        total_done = already + done_now
        if total_done == 1 or total_done % 50 == 0 or total_done == config.count:
            counts_now = state.counts()
            live = _write_live_status(output, run_id, status="running", completed=total_done,
                                      failed=counts_now.get("failed", failed_now), requested=config.count,
                                      objects=counts_now.get("objects", 0), started=started_monotonic,
                                      workers=workers)
            eta_text = "n/a" if live["eta_seconds"] is None else f"{live['eta_seconds']/60:.1f} min"
            free_text = "n/a" if live["disk_free_bytes"] is None else f"{live['disk_free_bytes']/1024**3:.1f} GB"
            log(f"已完成 {total_done}/{config.count}，失败 {failed_now}，{live['images_per_second']:.2f} img/s，ETA {eta_text}，磁盘剩余 {free_text}")

    try:
        if workers == 1:
            _init_worker(config, run_id, str(fragment_root))
            for plan in plans():
                if cancel_event is not None and cancel_event.is_set():
                    interrupted = True; break
                handle(_write_task(plan))
        else:
            ctx = mp.get_context("spawn")
            iterator = iter(plans())
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(config, run_id, str(fragment_root))) as ex:
                in_flight = {}
                exhausted = False
                while in_flight or not exhausted:
                    if cancel_event is not None and cancel_event.is_set():
                        interrupted = True
                        break
                    while not exhausted and len(in_flight) < queue_depth:
                        try:
                            plan = next(iterator)
                        except StopIteration:
                            exhausted = True; break
                        fut = ex.submit(_write_task, plan)
                        in_flight[fut] = plan.index
                    if not in_flight:
                        continue
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        in_flight.pop(fut, None)
                        try:
                            handle(fut.result())
                        except Exception as exc:
                            failed_now += 1; log(f"[worker exception] {exc}")
                if interrupted:
                    for fut in in_flight:
                        fut.cancel()
    except KeyboardInterrupt:
        interrupted = True
        log("收到中断信号；已完成任务已写入恢复状态。")

    counts = state.counts()
    final_completed = state.completed_bitmap(config.count)
    log_path = output / f"{run_id}_log.csv"
    _finalize_log(fragment_root, log_path, final_completed)
    if config.output_format in {"coco", "all"}:
        run_coco = output / f"instances_{run_id}.json"
        _finalize_coco(fragment_root, run_coco, final_completed, config.class_map)
        # Latest-run compatibility path used by explorer/analyzer.
        (output / "instances_coco.json").write_bytes(run_coco.read_bytes())

    diagnostics_path = output / f"{run_id}_generation_diagnostics.json"
    diagnostics = _finalize_generation_diagnostics(fragment_root, diagnostics_path, final_completed)
    (output / "latest_generation_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "interrupted" if interrupted else ("completed" if counts["completed"] >= config.count else "partial")
    state.mark_status(status)
    _write_live_status(
        output, run_id, status=status, completed=counts["completed"],
        failed=counts["failed"], requested=config.count, objects=counts["objects"],
        started=started_monotonic, workers=workers,
    )
    summary = {
        "run_id": run_id, "status": status,
        "generated_images": counts["completed"], "generated_objects": counts["objects"],
        "failed_images": counts["failed"], "requested_images": config.count,
        "resumed_images": already, "workers": workers,
        "output_dir": str(output.resolve()), "log_file": log_path.name,
        "state_db": str(state.path.relative_to(output)),
        "distribution_profile": str(config.distribution_profile) if config.distribution_profile else None,
        "scene_template": str(config.scene_template) if config.scene_template else None,
        "empty_scene_probability": config.empty_scene_probability,
        "augmentation_recipe": config.augmentation_recipe,
        "blend_mode": config.blend_mode,
        "hardcase_recipe": config.hardcase_recipe,
        "status_file": str((Path(".scenepaste") / "status" / f"{run_id}.json")),
        "generation_diagnostics": diagnostics_path.name,
    }
    state.close()
    (output / f"{run_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run_cfg = dict(payload); run_cfg.update({"run_id": run_id, "workers": workers, "resume": config.resume})
    (output / f"{run_id}_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps({"latest_run_id": run_id, "config": f"{run_id}_config.json"}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Run {run_id}：{status}，完成 {counts['completed']}/{config.count} 张，目标 {counts['objects']} 个")
    return summary
