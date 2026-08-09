"""Background workers (QThread) for loading cutouts and saving composites.

Heavy IO (JPEG decoding, asset loading) must not run on the GUI thread.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from compose_app.models import Cutout
from compose_app.rendering import pil_from_asset, make_thumbnail
import scenepaste.core as core
from scenepaste.core.models import ObjectAsset


def _cutout_to_asset(cutout: Cutout) -> ObjectAsset:
    """Build a minimal ObjectAsset (with .alpha) for the format writers.

    The Cutout only stores RGBA as a PIL image; geometry / format helpers
    expect ObjectAsset.alpha (float ndarray 0..1). We extract the alpha
    channel and convert the RGB channels back to BGR ndarray.
    """
    import numpy as np
    rgba = np.asarray(cutout.rgba)
    rgb = rgba[..., :3]
    bgr = rgb[..., ::-1].copy()
    alpha = (rgba[..., 3].astype(np.float32) / 255.0)
    return ObjectAsset(
        label=cutout.label,
        class_id=cutout.class_id,
        image=bgr,
        alpha=alpha,
        source_json=Path(cutout.source.split("#")[0]),
        source_shape_index=int(cutout.source.split("#")[1]) if "#" in cutout.source else 0,
        polygon=cutout.polygon,
    )


def _asset_to_cutout(asset: core.ObjectAsset) -> Cutout:
    rgba = pil_from_asset(asset)
    thumb = make_thumbnail(rgba)
    return Cutout(
        label=asset.label,
        class_id=asset.class_id,
        source=f"{asset.source_json}#{asset.source_shape_index}",
        rgba=rgba,
        polygon=asset.polygon,
        thumb=thumb,
    )


class LoadCutoutsWorker(QThread):
    """Load object assets off the GUI thread.

    Emits :attr:`cutouts_ready` as ``(cutouts, class_map_text)``. The class
    map is auto-extended from labels found on disk so unknown categories are
    not silently dropped.
    """

    cutouts_ready = Signal(list, str)
    failed = Signal(str)

    def __init__(self, objects_dir: Path, class_map_text: str, parent=None):
        super().__init__(parent)
        self._objects_dir = Path(objects_dir)
        self._class_map_text = class_map_text

    def run(self) -> None:
        try:
            from compose_app.io_utils import (
                build_class_map, format_class_map, scan_labels,
            )
            found = scan_labels(self._objects_dir)
            class_map = build_class_map(found, self._class_map_text)
            class_map_text = format_class_map(class_map) if class_map else self._class_map_text
            assets = core.load_object_assets(
                self._objects_dir, class_map, feather_sigma=0.8,
                log=lambda _msg: None,
            )
            cutouts = [_asset_to_cutout(a) for a in assets]
            self.cutouts_ready.emit(cutouts, class_map_text)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.failed.emit(str(exc))


def composite_from_doc(doc, instances=None) -> Image.Image:
    """Render the current document into a PIL RGBA composite (background + instances).

    Honours ``doc.do_shadow`` / ``doc.do_color_match`` when present.
    """
    from compose_app.rendering import (
        apply_color_match, bbox_of_rendered, draw_shadow,
    )

    bg_path = doc.current_background()
    if bg_path is None:
        raise RuntimeError("没有加载任何背景图")
    bgr = core.imread_with_exif(bg_path)
    if bgr is None:
        raise RuntimeError(f"无法读取背景：{bg_path}")
    h, w = bgr.shape[:2]
    rgb = bgr[..., ::-1]
    composite = Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("RGBA")

    do_shadow = bool(getattr(doc, "do_shadow", False))
    do_color_match = bool(getattr(doc, "do_color_match", False))
    insts = doc.instances if instances is None else instances
    for inst in insts:
        if not (0 <= inst.cutout_index < len(doc.cutouts)):
            continue
        cutout = doc.cutouts[inst.cutout_index]
        target_h = max(4, int(round(inst.h_ratio * h)))
        rendered = inst.get_rendered(cutout.rgba, target_h, class_label=cutout.label)
        x = int(round(inst.cx - rendered.width / 2.0))
        y = int(round(inst.cy - rendered.height / 2.0))
        if do_shadow:
            bx1, by1, bx2, by2 = bbox_of_rendered(rendered)
            draw_shadow(composite, x + (bx1 + bx2) // 2, y + by2, max(4, bx2 - bx1))
        if do_color_match:
            rendered = apply_color_match(rendered, composite, x, y)
        composite.alpha_composite(rendered, (x, y))
    return composite


@dataclass
class _InstanceSnap:
    """Relative-coord snapshot of a placed instance, portable across background sizes."""

    cutout_index: int
    cx_ratio: float
    cy_ratio: float
    h_ratio: float
    flip: bool
    angle: float
    uid: int = 0
    appearance_enabled: bool = False
    appearance_recipe: str = "mild"
    appearance_seed: int = 0
    appearance_brightness: float = 0.0
    appearance_contrast: float = 1.0
    appearance_saturation: float = 1.0
    appearance_blur: float = 0.0

    @classmethod
    def from_instance(cls, inst, bg_w: int, bg_h: int) -> "_InstanceSnap":
        return cls(
            cutout_index=inst.cutout_index,
            cx_ratio=float(inst.cx) / max(1, bg_w),
            cy_ratio=float(inst.cy) / max(1, bg_h),
            h_ratio=float(inst.h_ratio),
            flip=bool(inst.flip),
            angle=float(inst.angle),
            uid=int(inst.uid),
            appearance_enabled=bool(getattr(inst, "appearance_enabled", False)),
            appearance_recipe=str(getattr(inst, "appearance_recipe", "mild") or "mild"),
            appearance_seed=int(getattr(inst, "appearance_seed", 0) or 0),
            appearance_brightness=float(getattr(inst, "appearance_brightness", 0.0) or 0.0),
            appearance_contrast=float(getattr(inst, "appearance_contrast", 1.0) or 1.0),
            appearance_saturation=float(getattr(inst, "appearance_saturation", 1.0) or 1.0),
            appearance_blur=float(getattr(inst, "appearance_blur", 0.0) or 0.0),
        )

    def to_instance(self, bg_w: int, bg_h: int, uid: int = 0):
        from compose_app.models import Instance
        return Instance(
            cutout_index=self.cutout_index,
            cx=self.cx_ratio * bg_w,
            cy=self.cy_ratio * bg_h,
            h_ratio=self.h_ratio,
            flip=self.flip,
            angle=self.angle,
            uid=uid if uid else self.uid,
            appearance_enabled=bool(self.appearance_enabled),
            appearance_recipe=str(self.appearance_recipe or "mild"),
            appearance_seed=int(self.appearance_seed),
            appearance_brightness=float(self.appearance_brightness),
            appearance_contrast=float(self.appearance_contrast),
            appearance_saturation=float(self.appearance_saturation),
            appearance_blur=float(self.appearance_blur),
        )


class BatchApplyWorker(QThread):
    """Apply the current layout to a list of backgrounds and save each.

    The worker mirrors what :func:`scenepaste.core.generate_dataset` does for
    a single image, but instead of running ``paste_one`` it reuses the
    pre-placed instance snapshot (relative coords). All formats supported by
    the CLI are emitted; the COCO writer is reused across the whole batch.

    Signals:
        progress(int, int, str): (done, total, current_bg_name)
        finished(dict): summary {generated, failed, output_dir}
        failed(str): top-level error (e.g. bad output dir)
    """

    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, cutouts, instance_snaps, background_paths, output_dir: Path,
                 class_map_text: str, output_format: str = "detect",
                 run_id: Optional[str] = None, do_shadow: bool = False,
                 do_color_match: bool = False, scene_recipe: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.cutouts = list(cutouts)
        self.snaps = list(instance_snaps)
        self.background_paths = list(background_paths)
        self.output_dir = Path(output_dir)
        self.class_map_text = class_map_text
        self.output_format = (output_format or "detect").lower()
        self.run_id = run_id or _dt.datetime.now().strftime("batch_%Y%m%d_%H%M%S")
        self.do_shadow = bool(do_shadow)
        self.do_color_match = bool(do_color_match)
        self.scene_recipe = str(scene_recipe or "").strip()
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation; the worker finishes the current image then exits."""
        self._cancel = True

    def run(self) -> None:
        try:
            class_map = core.parse_class_map(self.class_map_text)
            out = self.output_dir
            (out / "images" / "train").mkdir(parents=True, exist_ok=True)
            (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (out / "previews").mkdir(parents=True, exist_ok=True)
            from compose_app.rendering import (
                apply_color_match, bbox_of_rendered, draw_shadow,
            )
            from compose_app import segmentation as gui_seg
            from scenepaste.formats import CocoWriter, write_data_yaml

            coco_writer: Optional[CocoWriter] = None
            if self.output_format == "coco":
                coco_writer = CocoWriter(
                    out / "instances_coco.json",
                    categories=[{"id": cid, "name": name, "supercategory": "object"}
                                for name, cid in sorted(class_map.items(), key=lambda kv: kv[1])],
                )
            write_data_yaml(out, class_map)

            total = len(self.background_paths)
            generated = failed = 0
            for i, bg_path in enumerate(self.background_paths):
                if self._cancel:
                    break
                self.progress.emit(i, total, Path(bg_path).name)
                try:
                    bgr = core.imread_with_exif(bg_path)
                    if bgr is None:
                        failed += 1
                        continue
                    h, w = bgr.shape[:2]
                    rgb = bgr[..., ::-1]
                    composite = Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("RGBA")
                    annotations = []  # list of (asset, box_xyxy, flipped, transform)
                    placed_instances = []
                    for snap in self.snaps:
                        if not (0 <= snap.cutout_index < len(self.cutouts)):
                            continue
                        cutout = self.cutouts[snap.cutout_index]
                        asset = _cutout_to_asset(cutout)
                        inst = snap.to_instance(w, h, uid=len(placed_instances) + 1)
                        target_h = max(4, int(round(inst.h_ratio * h)))
                        rendered = inst.get_rendered(
                            cutout.rgba, target_h, class_label=cutout.label,
                        )
                        x = int(round(inst.cx - rendered.width / 2.0))
                        y = int(round(inst.cy - rendered.height / 2.0))
                        if self.do_shadow:
                            bx1, by1, bx2, by2 = bbox_of_rendered(rendered)
                            draw_shadow(
                                composite, x + (bx1 + bx2) // 2, y + by2, max(4, bx2 - bx1),
                            )
                        if self.do_color_match:
                            rendered = apply_color_match(rendered, composite, x, y)
                        composite.alpha_composite(rendered, (x, y))
                        # Tight alpha bbox (matches CLI / io_utils.save_composite).
                        bx1, by1, bx2, by2 = bbox_of_rendered(rendered)
                        box = (x + bx1, y + by1, x + bx2, y + by2)
                        scale = target_h / max(1, cutout.rgba.height)
                        annotations.append((asset, box, inst.flip, (scale, x, y, inst.flip)))
                        placed_instances.append(inst)

                    if self.scene_recipe:
                        from scenepaste.core.recipes import apply_scene_recipe, load_augmentation_recipe
                        import random as _rng
                        recipe = load_augmentation_recipe(self.scene_recipe)
                        if recipe is not None:
                            bgr_out = np.asarray(composite.convert("RGB"))[..., ::-1].copy()
                            bgr_out, _effects = apply_scene_recipe(
                                bgr_out, recipe, _rng.Random(hash((self.run_id, i)) & 0xFFFFFFFF),
                            )
                            composite = Image.fromarray(bgr_out[..., ::-1], "RGB").convert("RGBA")

                    stem = f"{self.run_id}_{i:06d}"
                    jpg_path = out / "images" / "train" / f"{stem}.jpg"
                    composite.convert("RGB").save(str(jpg_path), quality=95)

                    # Masks from rendered alpha (includes rotation), not unrotated asset.alpha.
                    self._write_labels(
                        out, stem, w, h, annotations, class_map, coco_writer,
                        cutouts=self.cutouts, instances=placed_instances,
                        mask_fn=gui_seg.visible_instance_masks,
                    )
                    self._write_preview(out, stem, composite, annotations)
                    generated += 1
                except Exception:
                    # Surface the underlying error for debugging; production
                    # runs just count it as failed and move on.
                    import os as _os
                    if _os.environ.get("SCENEPASTE_DEBUG"):
                        import traceback as _tb
                        _tb.print_exc()
                    failed += 1
                    continue

            if coco_writer is not None:
                coco_writer.finalize()

            summary = {
                "generated": generated,
                "failed": failed,
                "cancelled": self._cancel,
                "output_dir": str(out.resolve()),
                "run_id": self.run_id,
            }
            self.finished.emit(summary)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.failed.emit(str(exc))

    # ------------------------------------------------------- per-format writers
    def _write_labels(self, out: Path, stem: str, w: int, h: int,
                      annotations, class_map, coco_writer,
                      cutouts=None, instances=None, mask_fn=None) -> None:
        from scenepaste.formats import (
            append_coco,
            seg_lines_for_annotations, obb_lines_for_annotations,
            write_semantic_classes,
        )
        from scenepaste.core.geometry import visible_instance_masks
        fmt = self.output_format
        label_path = out / "labels" / "train" / f"{stem}.txt"

        visible_masks = None
        if fmt in ("seg", "both", "coco", "semantic", "obb") and annotations:
            if mask_fn is not None and cutouts is not None and instances is not None:
                visible_masks = mask_fn(cutouts, instances, w, h)
            else:
                visible_masks = visible_instance_masks(annotations, w, h)

        if fmt in ("detect", "both", "coco"):
            lines = []
            for asset, box, _flipped, _transform in annotations:
                lines.append(core.yolo_line(asset.class_id, box, w, h))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                                  encoding="utf-8")

        if fmt == "coco" and coco_writer is not None:
            jpg_rel = out / "images" / "train" / f"{stem}.jpg"
            coco_writer = append_coco(
                out, "", jpg_rel, stem, w, h, annotations,
                writer=coco_writer, visible_masks=visible_masks,
            )

        if fmt in ("seg", "both"):
            seg_dir = out / "labels" / "train" if fmt == "seg" else (out / "labels-seg" / "train")
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_lines = seg_lines_for_annotations(annotations, visible_masks, w, h)
            (seg_dir / f"{stem}.txt").write_text(
                "\n".join(seg_lines) + ("\n" if seg_lines else ""), encoding="utf-8")

        if fmt == "obb" and visible_masks:
            obb_lines = obb_lines_for_annotations(annotations, visible_masks, w, h)
            label_path.write_text("\n".join(obb_lines) + ("\n" if obb_lines else ""),
                                  encoding="utf-8")

        if fmt == "semantic" and visible_masks:
            from scenepaste.core.io import imwrite_unicode as _imw
            max_value = max(class_map.values(), default=0) + 1
            dtype = np.uint8 if max_value <= 255 else np.uint16
            mask = np.zeros((h, w), dtype=dtype)
            for (asset, _box, _flipped, _transform), vis in zip(annotations, visible_masks):
                mask[vis] = int(asset.class_id) + 1
            masks_dir = out / "masks" / "train"
            masks_dir.mkdir(parents=True, exist_ok=True)
            _imw(masks_dir / f"{stem}.png", mask)
            write_semantic_classes(class_map, out)

    def _write_preview(self, out: Path, stem: str, composite: Image.Image,
                       annotations) -> None:
        preview = composite.convert("RGB").copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(preview)
        for asset, box, _flipped, _transform in annotations:
            x1, y1, x2, y2 = box
            color = (40, 220, 40) if asset.class_id == 0 else (40, 160, 255)
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=2)
        preview.save(str(out / "previews" / f"{stem}.jpg"), quality=92)
