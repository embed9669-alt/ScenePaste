"""Dataset generation pipeline (orchestration).

This module assembles asset loading, background sampling, paste composition,
format-specific output writers, and run metadata. Behaviour matches the
historical monolithic ``copy_paste_dataset_tool.generate_dataset``; the only
change is internal modularization.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from ..formats import (
    CocoWriter,
    append_coco,
    obb_lines_for_annotations,
    seg_lines_for_annotations,
    write_classes_file,
    write_data_yaml,
    write_semantic_classes,
    yolo_line,
)
from .augmentation import augment_foreground, resize_asset  # noqa: F401 (re-export)
from .auto_cutout import load_object_assets_auto
from .config import GenerationConfig, validate_config
from ..errors import t
from .geometry import (
    box_iou,  # noqa: F401 (re-export)
    canvas_polygon,  # noqa: F401 (re-export)
    visible_instance_masks,
)
from .io import (
    imread_unicode,  # noqa: F401 (re-export)
    imread_with_exif,  # noqa: F401 (re-export)
    imwrite_unicode,
    list_backgrounds,
    load_json,  # noqa: F401 (re-export)
    log_default,
    save_cutouts,
)
from .labelme import (
    load_object_assets,
    load_paste_zone_mask,
    resolve_labelme_image,  # noqa: F401 (re-export)
    shape_to_mask,  # noqa: F401 (re-export)
)
from .models import IMAGE_SUFFIXES, PASTE_ZONE_LABELS, ObjectAsset  # noqa: F401
from .sampling import (
    BackgroundCache,
    BackgroundSampler,
    build_asset_groups,
    sample_asset,
)
from .synthesis import paste_one
from .validation import (
    clip_bbox,  # noqa: F401 (re-export)
    is_valid_yolo_box,
    visible_ratio,  # noqa: F401 (re-export)
)


def generate_dataset(
    config: GenerationConfig,
    log: Callable[[str], None] = log_default,
) -> dict:
    """Run a full dataset generation pass.

    Steps:
    1. Validate config and create output directory layout.
    2. Load object assets (LabelMe or auto-cutout) and the background list.
    3. Persist cutouts / classes.txt / data.yaml / semantic mapping once.
    4. Stream composed images, per-format labels, and the run CSV log.
    5. Finalize COCO (if requested) and write per-run + latest summary JSON.

    Returns a summary dict (also persisted as ``<run_id>_summary.json`` and
    ``latest_summary.json``).
    """
    validate_config(config)
    # Unified engine: deterministic per-index planning, SQLite resume and
    # bounded multiprocessing are used even when workers=1. This guarantees
    # the same seed produces the same plans regardless of worker count.
    from .advanced_pipeline import generate_dataset_advanced
    return generate_dataset_advanced(config, log=log)

    # Legacy implementation retained below for source compatibility/reference.
    rng = random.Random(config.seed)
    preview_rng = random.Random(config.seed ^ 0x5A17C0DE)
    output_dir = config.output_dir
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    previews_dir = output_dir / "previews"
    init_dirs = [images_dir, labels_dir, previews_dir]
    if (config.output_format or "detect").lower() == "semantic":
        init_dirs.append(output_dir / "masks" / "train")
    for directory in init_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    log("正在读取标注目标……")
    if config.auto_cutout:
        assets = load_object_assets_auto(
            config.objects_dir, config.class_map, log,
            label=config.auto_cutout_label,
            label_from_subdir=bool(config.auto_cutout_label_from_subdir),
        )
    else:
        assets = load_object_assets(
            config.objects_dir, config.class_map, config.feather_sigma, log,
            rectangle_mask_mode=config.rectangle_mask_mode,
        )
    backgrounds = list_backgrounds(config.backgrounds_dir)
    if not backgrounds:
        raise RuntimeError(t("backgrounds.empty", path=config.backgrounds_dir))
    asset_groups = build_asset_groups(assets)
    background_rng = random.Random(config.seed ^ 0x3B9ACA07)
    background_sampler = BackgroundSampler(backgrounds, background_rng, config.background_sampling)
    save_cutouts(assets, output_dir)
    write_classes_file(config.class_map, output_dir)
    if (config.output_format or "detect").lower() == "semantic":
        write_semantic_classes(config.class_map, output_dir)
    yaml_path = write_data_yaml(output_dir, config.class_map)
    log(f"data.yaml 已生成：{yaml_path.name}（可直接 yolo train data=data.yaml）")
    log(f"提取到 {len(assets)} 个目标，找到 {len(backgrounds)} 张背景图")

    generated = 0
    failed = 0
    total_objects = 0
    run_id = config.run_id or _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    prefix = f"{run_id}_{config.seed}"
    log(f"Run ID：{run_id}")

    run_config_path = output_dir / f"{run_id}_config.json"
    latest_config_path = output_dir / "run_config.json"
    try:
        with run_config_path.open("w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "seed": config.seed,
                "count": config.count,
                "min_objects": config.min_objects,
                "max_objects": config.max_objects,
                "class_map": {str(k): v for k, v in config.class_map.items()},
                "output_format": config.output_format,
                "asset_sampling": config.asset_sampling,
                "background_sampling": config.background_sampling,
                "auto_cutout": config.auto_cutout,
                "objects_dir": str(config.objects_dir),
                "backgrounds_dir": str(config.backgrounds_dir),
                "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }, f, ensure_ascii=False, indent=2)
        latest_config_path.write_text(
            json.dumps({"latest_run_id": run_id,
                        "config": str(run_config_path.name)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    log_path = output_dir / f"{run_id}_log.csv"
    log_fp = log_path.open("w", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(
        [
            "generated_stem",
            "background",
            "label",
            "class_id",
            "source_json",
            "shape_index",
            "flipped",
            "box_xyxy",
            "used_paste_zone",
            "run_id",
        ]
    )

    coco_writer: Optional[CocoWriter] = None
    if (config.output_format or "detect").lower() == "coco":
        coco_writer = CocoWriter(
            output_dir / "instances_coco.json",
            categories=[{"id": cid, "name": name, "supercategory": "object"}
                        for name, cid in sorted(config.class_map.items(), key=lambda kv: kv[1])],
            checkpoint_interval=config.coco_checkpoint_interval,
        )

    bg_cache = BackgroundCache(config.background_cache_size) if config.background_cache_size > 0 else None

    for index in range(config.count):
        background_path = background_sampler.next()
        background = bg_cache.get(background_path) if bg_cache is not None else imread_with_exif(background_path)
        if background is None:
            failed += 1
            log(f"[跳过] 无法读取背景：{background_path}")
            continue
        fmt = (config.output_format or "detect").lower()
        canvas = background.copy()
        height, width = canvas.shape[:2]
        zone_mask = load_paste_zone_mask(background_path, height, width)
        object_count = rng.randint(config.min_objects, config.max_objects)
        boxes: List[Tuple[int, int, int, int]] = []
        annotations = []

        for _ in range(object_count):
            asset = sample_asset(rng, assets, asset_groups, config.asset_sampling)
            result = paste_one(canvas, asset, boxes, zone_mask, config, rng)
            if result is None:
                continue
            box, flipped, transform, _appearance = result
            boxes.append(box)
            annotations.append((asset, box, flipped, transform))

        if not annotations:
            failed += 1
            log(f"[跳过] 第 {index + 1} 张未找到合适的粘贴位置")
            continue

        stem = f"{prefix}_{index:06d}"
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        if not imwrite_unicode(image_path, canvas, jpeg_quality=95):
            failed += 1
            continue

        # Unified clip + visibility check: drop badly-clipped instances.
        valid_annotations = []
        dropped = 0
        for ann in annotations:
            asset, box, flipped, transform = ann
            ok, clipped = is_valid_yolo_box(
                box, width, height, min_visible=config.min_visible_ratio)
            if ok:
                valid_annotations.append((asset, clipped, flipped, transform))
            else:
                dropped += 1
        if dropped:
            log(f"[{stem}] 丢弃 {dropped} 个越界/过小目标")
        annotations = valid_annotations
        if not annotations:
            failed += 1
            log(f"[跳过] 第 {index + 1} 张所有目标越界被丢弃")
            try:
                image_path.unlink()
            except OSError:
                pass
            continue

        # Mask-first: seg / COCO / semantic / OBB all derive from final pixels.
        visible_masks: Optional[List[np.ndarray]] = None
        if fmt in ("seg", "both", "coco", "semantic", "obb"):
            visible_masks = visible_instance_masks(annotations, width, height)

        if fmt in ("detect", "both", "coco"):
            with label_path.open("w", encoding="utf-8") as file:
                for asset, box, _flipped, _transform in annotations:
                    file.write(yolo_line(asset.class_id, box, width, height) + "\n")

        # YOLO-seg: derive contour from final visible mask. Single-polygon per
        # line is an Ultralytics .txt compatibility trade-off; COCO keeps the
        # full multi-polygon segmentation.
        if fmt in ("seg", "both"):
            # Pure seg writes to labels/train (Ultralytics-ready); both mode
            # diverts to labels-seg/train to avoid clobbering detect labels.
            seg_dir = labels_dir if fmt == "seg" else (output_dir / "labels-seg" / "train")
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_lines = seg_lines_for_annotations(annotations, visible_masks, width, height)
            (seg_dir / f"{stem}.txt").write_text(
                ("\n".join(seg_lines) + "\n") if seg_lines else "",
                encoding="utf-8")

        if fmt == "coco":
            coco_writer = append_coco(
                output_dir, background_path, image_path, stem, width, height, annotations,
                writer=coco_writer, visible_masks=visible_masks,
            )

        # Semantic: 0 is always background; real class c is encoded c+1.
        if fmt == "semantic" and visible_masks is not None:
            max_value = max(config.class_map.values(), default=0) + 1
            dtype = np.uint8 if max_value <= 255 else np.uint16
            semantic_mask = np.zeros((height, width), dtype=dtype)
            for (asset, _box, _flipped, _transform), vis in zip(annotations, visible_masks):
                semantic_mask[vis] = int(asset.class_id) + 1
            masks_dir = output_dir / "masks" / "train"
            masks_dir.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(masks_dir / f"{stem}.png", semantic_mask)

        # Ultralytics YOLO-OBB: labels still go to labels/train, format class x1 y1 ... x4 y4.
        if fmt == "obb" and visible_masks is not None:
            obb_lines = obb_lines_for_annotations(annotations, visible_masks, width, height)
            labels_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text(("\n".join(obb_lines) + "\n") if obb_lines else "", encoding="utf-8")

        if config.save_previews and config.preview_ratio > 0 and preview_rng.random() < config.preview_ratio:
            preview = canvas.copy()
            for asset, box, _flipped, _transform in annotations:
                x1, y1, x2, y2 = box
                color = (40, 220, 40) if asset.class_id == 0 else (40, 160, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    preview,
                    asset.label,
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            imwrite_unicode(previews_dir / f"{stem}.jpg", preview, jpeg_quality=92)

        for asset, box, flipped, _transform in annotations:
            log_writer.writerow(
                [
                    stem,
                    str(background_path),
                    asset.label,
                    str(asset.class_id),
                    str(asset.source_json),
                    str(asset.source_shape_index),
                    "1" if flipped else "0",
                    ",".join(map(str, box)),
                    "1" if zone_mask is not None else "0",
                    run_id,
                ]
            )
        generated += 1
        total_objects += len(annotations)
        # Flush every 50 images so a crash loses at most 50 rows of metadata.
        if generated % 50 == 0:
            log_fp.flush()
        if generated == 1 or generated % 50 == 0 or index == config.count - 1:
            log(f"已生成 {generated}/{config.count} 张，目标总数 {total_objects}")

    log_fp.close()
    if coco_writer is not None:
        coco_writer.finalize()

    summary = {
        "run_id": run_id,
        "generated_images": generated,
        "generated_objects": total_objects,
        "failed_images": failed,
        "output_dir": str(output_dir.resolve()),
        "log_file": str(log_path.name),
        "preview_ratio": config.preview_ratio if config.save_previews else 0.0,
        "background_cache": bg_cache.stats() if bg_cache is not None else {"capacity": 0},
    }
    with (output_dir / f"{run_id}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (output_dir / "latest_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    log(
        f"完成：生成 {generated} 张、{total_objects} 个目标；"
        f"建议抽查 previews 并运行 `scenepaste analyze <output>` 做质量检查。"
    )
    return summary
