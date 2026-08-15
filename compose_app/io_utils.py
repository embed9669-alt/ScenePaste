"""IO 与类别映射相关的纯函数。

不依赖 tkinter，方便测试。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import scenepaste.core as core

from .models import Cutout
from .rendering import pil_from_asset
from . import segmentation as seg


def scan_labels(objects_dir: Path) -> List[str]:
    """扫描 objects 目录所有 JSON 里实际出现的标注类别（不含 paste_zone 等）。"""
    labels: set = set()
    for jp in sorted(objects_dir.rglob("*.json")):
        try:
            d = core.load_json(jp)
        except Exception:
            continue
        for s in d.get("shapes", []):
            lbl = str(s.get("label", "")).strip()
            if lbl and lbl not in core.PASTE_ZONE_LABELS:
                labels.add(lbl)
    return sorted(labels)


def build_class_map(found_labels: List[str], current_text: str) -> Dict[str, int]:
    """基于扫描到的 label 自动构建/补齐 class_map，保证 id 从 0 连续。

    用户已显式写在 current_text 里的优先保留；其余按字母序追加。
    """
    try:
        base = core.parse_class_map(current_text)
    except Exception:
        base = {}
    result = dict(base)
    next_id = max(list(result.values()) + [-1]) + 1
    for lbl in found_labels:
        if lbl not in result:
            result[lbl] = next_id
            next_id += 1
    # 重排为 0..N-1 连续（parse_class_map 强制连续）
    items = sorted(result.items(), key=lambda kv: kv[1])
    return {lbl: i for i, (lbl, _) in enumerate(items)}


def format_class_map(class_map: Dict[str, int]) -> str:
    """把 dict 序列化成输入框里的 "label=id,label=id" 文本。"""
    return ",".join(f"{lbl}={i}" for lbl, i in sorted(class_map.items(), key=lambda x: x[1]))


def load_cutouts(objects_dir: Path, class_map: Dict[str, int],
                 feather_sigma: float = 0.8) -> List[Cutout]:
    """读取目标素材，返回 Cutout 列表（不带缩略图 PhotoImage，由调用方补）。

    在工作线程中调用，避免阻塞 UI。PhotoImage 必须在主线程创建。
    """
    assets = core.load_object_assets(objects_dir, class_map, feather_sigma=feather_sigma,
                                     log=lambda m: None, rectangle_mask_mode="grabcut")
    cutouts: List[Cutout] = []
    for a in assets:
        rgba = pil_from_asset(a).convert("RGBA")
        cutouts.append(Cutout(
            label=a.label,
            class_id=a.class_id,
            source=f"{Path(a.source_json).name}#{a.source_shape_index}",
            rgba=rgba,
            polygon=getattr(a, "polygon", None),
            thumb=None,
        ))
    return cutouts


def save_composite(bg_image, bg_path: Path, instances, cutouts: List[Cutout],
                   output_dir: Path, stem: str,
                   apply_shadow: bool = False, apply_color_match: bool = False,
                   composite_and_boxes=None,
                   output_format: str = "detect",
                   coco_writer: Optional[object] = None) -> Tuple[Path, Optional[Path], List[tuple]]:
    """保存合成图与标注。

    支持：detect / seg / both / coco / semantic / obb。
    分割、COCO、Semantic、OBB 均从最终渲染 alpha 的 *可见 mask* 派生，
    因而目标互相遮挡后标签会同步更新。
    """
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    seg_dir = output_dir / "labels-seg" / "train"
    masks_dir = output_dir / "masks" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    if composite_and_boxes is not None:
        composite, native_boxes = composite_and_boxes
    else:
        from .rendering import bbox_of_rendered
        composite = bg_image.convert("RGBA")
        native_boxes = []
        for inst in instances:
            if not getattr(inst, "visible", True):
                # Hidden layers are skipped by both composite and annotation paths.
                native_boxes.append(None)
                continue
            cut = cutouts[inst.cutout_index]
            bh = bg_image.size[1]
            h = max(4, int(round(inst.h_ratio * bh)))
            rendered = inst.get_rendered(cut.rgba, h)
            x1 = int(round(inst.cx - rendered.width / 2.0))
            y1 = int(round(inst.cy - rendered.height / 2.0))
            composite.alpha_composite(rendered, (x1, y1))
            bx1, by1, bx2, by2 = bbox_of_rendered(rendered)
            native_boxes.append((x1 + bx1, y1 + by1, x1 + bx2, y1 + by2,
                                 cut.label, cut.class_id))

    bw_n, bh_n = bg_image.size
    arr = np.asarray(composite.convert("RGB"))[..., ::-1].copy()
    image_path = images_dir / f"{stem}.jpg"
    core.imwrite_unicode(image_path, arr, jpeg_quality=95)

    fmt = (output_format or "detect").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"不支持的输出格式：{fmt}")

    label_path = labels_dir / f"{stem}.txt"
    min_visible = 0.10
    valid_boxes = []
    valid_instances = []
    for inst, box in zip(instances, native_boxes):
        if box is None or not getattr(inst, "visible", True):
            continue
        bx1, by1, bx2, by2, label, cls_id = box
        ok, (cx1, cy1, cx2, cy2) = core.is_valid_yolo_box(
            (bx1, by1, bx2, by2), bw_n, bh_n, min_visible=min_visible)
        if ok:
            valid_boxes.append((cx1, cy1, cx2, cy2, label, cls_id))
            valid_instances.append(inst)
    native_boxes = valid_boxes
    instances = valid_instances
    if not native_boxes:
        try:
            image_path.unlink()
        except OSError:
            pass
        return image_path, None, []

    # 最终可见 mask：后放置的实例遮挡前面的实例。
    visible_masks = None
    if fmt in ("seg", "both", "coco", "semantic", "obb"):
        visible_masks = seg.visible_instance_masks(cutouts, instances, bw_n, bh_n)

    primary_path: Optional[Path] = label_path
    if fmt in ("detect", "both", "coco"):
        labels_dir.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8") as f:
            for bx1, by1, bx2, by2, _label, cls_id in native_boxes:
                f.write(core.yolo_line(cls_id, (bx1, by1, bx2, by2), bw_n, bh_n) + "\n")

    if fmt in ("seg", "both"):
        target_seg_dir = labels_dir if fmt == "seg" else seg_dir
        target_seg_dir.mkdir(parents=True, exist_ok=True)
        seg_path = target_seg_dir / f"{stem}.txt"
        lines: List[str] = []
        for idx, (inst, box) in enumerate(zip(instances, native_boxes)):
            bx1, by1, bx2, by2, _label, cls_id = box
            polys = core.mask_to_polygons(visible_masks[idx]) if visible_masks is not None else []
            if polys:
                poly = polys[0]
            else:
                cut = cutouts[inst.cutout_index]
                poly = seg.instance_canvas_polygon(cut, inst, bw_n, bh_n)
                if poly is None:
                    poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                                    dtype=np.float32)
            line = seg.yolo_seg_line(cls_id, poly, bw_n, bh_n)
            if line:
                lines.append(line)
        seg_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        if fmt == "seg":
            primary_path = seg_path

    if fmt == "coco":
        _append_coco_instances(
            output_dir, bg_path, image_path, stem, bw_n, bh_n,
            instances, cutouts, native_boxes, writer=coco_writer,
            visible_masks=visible_masks,
        )

    if fmt == "semantic" and visible_masks is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
        max_class = max((c.class_id for c in cutouts), default=0) + 1
        dtype = np.uint8 if max_class <= 255 else np.uint16
        semantic_mask = np.zeros((bh_n, bw_n), dtype=dtype)
        class_map = {c.label: int(c.class_id) for c in cutouts}
        for inst, vis in zip(instances, visible_masks):
            cut = cutouts[inst.cutout_index]
            semantic_mask[vis] = int(cut.class_id) + 1
        mask_path = masks_dir / f"{stem}.png"
        core.imwrite_unicode(mask_path, semantic_mask)
        core.write_semantic_classes(class_map, output_dir)
        primary_path = mask_path

    if fmt == "obb" and visible_masks is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for inst, vis in zip(instances, visible_masks):
            cut = cutouts[inst.cutout_index]
            line = core.ultralytics_obb_line(cut.class_id, vis, bw_n, bh_n)
            if line:
                lines.append(line)
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        primary_path = label_path

    # GUI 日志采用流式 append，不持有全量 rows。
    log_path = output_dir / "generation_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["stem", "background", "label", "class_id", "source_json",
                        "shape_index", "flip", "angle", "box_xyxy"])
        for inst, (bx1, by1, bx2, by2, label, cls_id) in zip(instances, native_boxes):
            cut = cutouts[inst.cutout_index]
            source_json, _, shape_index = cut.source.rpartition("#")
            w.writerow([stem, str(bg_path), label, cls_id, source_json, shape_index or "0",
                        "1" if inst.flip else "0", f"{inst.angle:.1f}",
                        ",".join(map(str, (bx1, by1, bx2, by2)))])

    return image_path, primary_path, native_boxes


# 输出格式 → 稳定枚举。GUI 与 CLI 共用相同任务集合。
OUTPUT_FORMATS = ("detect", "seg", "both", "coco", "semantic", "obb")


def _append_coco_instances(output_dir: Path, bg_path: Path, image_path: Path,
                           stem: str, bw: int, bh: int,
                           instances, cutouts: List[Cutout], native_boxes,
                           writer: Optional[object] = None,
                           visible_masks=None):
    """追加当前图到 COCO writer；传 writer 时不逐图 flush。"""
    coco_path = output_dir / "instances_coco.json"
    owned_writer = writer is None
    if writer is None:
        writer = core.CocoWriter(coco_path)
    for cut in cutouts:
        writer.ensure_category(cut.label, cut.class_id)
    img_id = writer.add_image(
        file_name=str(Path("images/train") / image_path.name),
        width=bw, height=bh, background=str(bg_path), stem=stem,
    )
    for idx, (inst, box) in enumerate(zip(instances, native_boxes)):
        bx1, by1, bx2, by2, _label, cls_id = box
        segmentation = []
        area = 0.0
        bbox = [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)]
        if visible_masks is not None and idx < len(visible_masks):
            vis = visible_masks[idx]
            polys = core.mask_to_polygons(vis)
            segmentation = [[float(v) for v in p.reshape(-1)] for p in polys]
            area = float(np.count_nonzero(vis))
            mb = core.mask_bbox(vis)
            if mb is not None:
                x1, y1, x2, y2 = mb
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if not segmentation:
            cut = cutouts[inst.cutout_index]
            poly = seg.instance_canvas_polygon(cut, inst, bw, bh)
            if poly is None:
                poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                                dtype=np.float32)
            segmentation = [seg.coco_polygon(poly)]
            area = float(seg.polygon_area(poly))
        writer.add_annotation(
            image_id=img_id,
            category_id=int(cls_id),
            segmentation=segmentation,
            bbox=bbox,
            area=area,
        )
    if owned_writer:
        writer.finalize()
    return writer

def _empty_coco(cutouts: List[Cutout]) -> dict:
    cats = []
    seen = set()
    for cut in cutouts:
        if cut.label not in seen:
            cats.append({"id": cut.class_id, "name": cut.label,
                         "supercategory": "object"})
            seen.add(cut.label)
    cats.sort(key=lambda c: c["id"])
    return {"images": [], "annotations": [], "categories": cats}
