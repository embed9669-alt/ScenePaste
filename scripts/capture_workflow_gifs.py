#!/usr/bin/env python3
"""Capture a short multi-GIF tour of the ScenePaste workflow.

Produces under ``docs/images/``:

  gif_00_cutout.gif        — Cutout Studio: folder browser + polygon
  gif_01_compose.gif       — place / scale / rotate cutouts
  gif_02_appearance.gif    — object appearance preview
  gif_03_batch_defaults.gif— main-window generation defaults
  gif_04_explorer.gif      — Dataset Explorer browse
  gif_05_data_loop.gif     — Data Loop Center tabs
  ui_demo.gif              — alias of appearance clip for README hero

Usage::

    python scripts/capture_workflow_gifs.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
SAMPLES = ROOT / "samples"
DEMO = ROOT / "generated" / "_workflow_gif_demo"

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter  # noqa: E402

from compose_app_qt.app import MainWindow  # noqa: E402


def _drain(app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.015)


def _wait_ready(win: MainWindow, app: QApplication, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        time.sleep(0.04)
        if len(win.doc.cutouts) >= 1 and win.doc.bg_size[0] > 0:
            return True
    return False


def _prefer_background(win: MainWindow, keyword: str = "road") -> None:
    for i, p in enumerate(win.doc.background_paths):
        if keyword in Path(p).stem.lower():
            if i != win.doc.background_index:
                win.doc.background_index = i
                win.doc.instances = []
                win.doc.selected_uid = None
                win.doc._emit()
            return


def _qpixmap_to_pil(pix) -> Image.Image:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pix.save(buf, "PNG")
    return Image.open(BytesIO(bytes(buf.data()))).convert("RGB")


def _grab(widget, width: int = 960) -> Image.Image:
    img = _qpixmap_to_pil(widget.grab())
    if width and img.width > width:
        ratio = width / float(img.width)
        img = img.resize((width, max(1, int(round(img.height * ratio)))), Image.LANCZOS)
    return img


def _save_gif(frames: list[Image.Image], path: Path, fps: float = 8.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"no frames for {path}")
    ms = max(40, int(round(1000.0 / max(1.0, fps))))
    tmp = path.with_suffix(".raw.gif")
    frames[0].save(
        tmp, save_all=True, append_images=frames[1:], duration=ms,
        loop=0, optimize=True, disposal=2,
    )
    # Palette-optimized smaller GIF for blog / README.
    opt = path.with_suffix(".opt.gif")
    cmd = [
        "ffmpeg", "-y", "-i", str(tmp),
        "-vf",
        f"fps={fps},scale={frames[0].width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        str(opt),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        opt.replace(path)
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.replace(path)
        opt.unlink(missing_ok=True)
    kb = path.stat().st_size / 1024.0
    print(f"OK {path.name} · {len(frames)} frames · {kb:.0f} KB · {frames[0].width}x{frames[0].height}")
    return path


def _bias_splitter(win: MainWindow) -> None:
    for sp in win.findChildren(QSplitter):
        if sp.count() >= 3:
            sp.setSizes([170, 760, 430])


def _show_right_tab(win: MainWindow, index: int) -> None:
    """0=实例, 1=变换·外观, 2=批量默认."""
    tabs = getattr(win, "_right_tabs", None)
    if tabs is not None and 0 <= index < tabs.count():
        tabs.setCurrentIndex(index)


def _fit_canvas(win: MainWindow) -> None:
    if win.canvas._bg_item is not None:
        win.canvas.fitInView(win.canvas._bg_item, Qt.KeepAspectRatio)


def _cutout_order(win: MainWindow) -> list[int]:
    by_label = {c.label.lower(): i for i, c in enumerate(win.doc.cutouts)}
    order = []
    for key in ("person", "truck", "motorcycle"):
        if key in by_label:
            order.append(by_label[key])
    for i in range(len(win.doc.cutouts)):
        if i not in order:
            order.append(i)
    return order


def _ensure_demo_dataset(app: QApplication) -> Path:
    """Small all-format dataset for Explorer / Data Loop GIFs."""
    marker = DEMO / "images" / "train"
    if marker.is_dir() and any(marker.glob("*.jpg")):
        return DEMO
    from scenepaste import GenerationConfig, generate_dataset

    # Prefer the angled template used by docs screenshots.
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from capture_docs_screenshots import _write_angled_template  # type: ignore
        tpl = _write_angled_template()
    except Exception:
        import json
        tpl = ROOT / "generated" / "_docs_angled_template.json"
        payload = {
            "schema": "scenepaste/scene-template", "version": 2,
            "canvas": {"width": 800, "height": 600},
            "global": {"enabled_probability": 1.0, "flip_probability": 0.0, "same_class_random": False},
            "instances": [
                {"source": "sample_person.json#0", "label": "person", "class_id": 0,
                 "cx_ratio": 0.28, "cy_ratio": 0.72, "h_ratio": 0.38, "flip": False, "angle": 18.0},
                {"source": "sample_truck.json#0", "label": "truck", "class_id": 1,
                 "cx_ratio": 0.62, "cy_ratio": 0.74, "h_ratio": 0.28, "flip": False, "angle": -16.0},
                {"source": "sample_motorcycle.json#0", "label": "motorcycle", "class_id": 2,
                 "cx_ratio": 0.82, "cy_ratio": 0.78, "h_ratio": 0.16, "flip": False, "angle": 22.0},
            ],
        }
        tpl.parent.mkdir(parents=True, exist_ok=True)
        tpl.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if DEMO.exists():
        shutil.rmtree(DEMO, ignore_errors=True)
    generate_dataset(GenerationConfig(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=DEMO,
        class_map={"person": 0, "truck": 1, "motorcycle": 2},
        count=4, min_objects=3, max_objects=3, workers=1, seed=11,
        output_format="all", save_previews=True, preview_ratio=1.0,
        scene_template=tpl, object_appearance_recipe="mild",
        empty_scene_probability=0.0, run_id="workflow_gif",
    ))
    _drain(app, 0.1)
    return DEMO


def record_cutout_studio(app: QApplication) -> Path:
    """Folder browser + thumbnail list + polygon canvas in Cutout Studio."""
    from compose_app_qt.cutout_studio import ObjectCutoutStudioDialog
    from compose_app_qt.theme import apply_theme

    objects = SAMPLES / "objects"
    images = sorted(
        p for p in objects.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    dlg = ObjectCutoutStudioDialog(
        objects_dir=objects,
        class_map_text="person=0,truck=1,motorcycle=2",
        parent=None,
    )
    apply_theme(dlg, "dark")
    dlg.resize(1280, 780)
    dlg.show(); dlg.raise_(); _drain(app, 0.25)

    frames: list[Image.Image] = []
    frames.append(_grab(dlg, width=1000))
    if images:
        dlg._set_folder_images(images, root=objects, select=images[0])
        _drain(app, 0.8)
        frames.append(_grab(dlg, width=1000))
        # Browse through samples so the left list selection is obvious.
        for row in range(min(len(images), 3)):
            dlg.file_list.setCurrentRow(row)
            _drain(app, 0.35)
            frames.append(_grab(dlg, width=1000))
            frames.append(_grab(dlg, width=1000))
        # Emphasize person + closed polygon (LabelMe JSON or fallback).
        for i, p in enumerate(images):
            if "person" in p.stem.lower():
                dlg.file_list.setCurrentRow(i)
                break
        _drain(app, 0.3)
        if dlg.view.polygon_array() is None and dlg._bgr is not None:
            h, w = dlg._bgr.shape[:2]
            pts = [
                (w * 0.35, h * 0.25),
                (w * 0.55, h * 0.22),
                (w * 0.65, h * 0.45),
                (w * 0.60, h * 0.85),
                (w * 0.32, h * 0.82),
            ]
            for n in range(3, len(pts) + 1):
                dlg.view.set_polygon(pts[:n], closed=(n == len(pts)))
                _drain(app, 0.12)
                frames.append(_grab(dlg, width=1000))
        # Toggle interaction mode label for the last beats.
        idx = dlg.mode_combo.findData(dlg.view.MODE_CLICK)
        if idx >= 0:
            dlg.mode_combo.setCurrentIndex(idx)
            _drain(app, 0.2)
            frames.append(_grab(dlg, width=1000))
            dlg.mode_combo.setCurrentIndex(0)
            _drain(app, 0.15)
            frames.append(_grab(dlg, width=1000))
    for _ in range(3):
        frames.append(_grab(dlg, width=1000)); _drain(app, 0.08)
    dlg.close()
    return _save_gif(frames, OUT / "gif_00_cutout.gif", fps=6)


def record_compose(app: QApplication) -> Path:
    win = MainWindow(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=ROOT / "generated" / "_gif_compose",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    win.resize(1460, 900)
    win.show(); win.raise_(); _drain(app, 0.3)
    _wait_ready(win, app)
    _prefer_background(win, "road")
    win.doc.replace_instances([])
    win.doc.scene_recipe = "camera-mild"
    win.doc.object_appearance_recipe = "mild"
    win.doc.blend_mode = "gaussian"
    win.gen_defaults.reflect(win.doc)
    win.doc._emit()
    _drain(app, 0.25)
    _fit_canvas(win); _bias_splitter(win)
    _show_right_tab(win, 0)  # 实例 / layers
    _drain(app, 0.2)

    frames = []
    w, h = win.doc.bg_size
    order = _cutout_order(win)
    layout = [
        (0.24, 0.76, 0.38, 0.0),
        (0.55, 0.74, 0.30, 0.0),
        (0.82, 0.80, 0.18, 0.0),
    ]
    frames.append(_grab(win))
    placed = []
    for idx, (xr, yr, hr, ang) in zip(order, layout):
        win._add_instance(idx, w * xr, h * yr)
        inst = win.doc.instances[-1]
        inst.h_ratio = hr
        inst.angle = ang
        inst.invalidate_cache()
        win.doc.select(inst.uid)
        win.doc._emit(); _drain(app, 0.12)
        frames.append(_grab(win))
        placed.append(inst.uid)
        # Animate scale + rotate on the newest instance.
        for hr2, ang2 in ((hr * 0.9, ang + 8), (hr, ang + 18), (hr, ang + 16)):
            inst.h_ratio = hr2
            inst.angle = ang2
            inst.invalidate_cache()
            win.doc._emit(); _drain(app, 0.08)
            frames.append(_grab(win))

    # Layer ops: send the back-most instance to the front (visible restack + list).
    if len(placed) >= 2:
        back_uid = placed[0]
        win.doc.select(back_uid)
        win.doc._emit(); _drain(app, 0.1)
        frames.append(_grab(win))
        win.controller.bring_to_front(back_uid)
        _drain(app, 0.15)
        frames.append(_grab(win))
        win.controller.send_to_back(back_uid)
        _drain(app, 0.12)
        frames.append(_grab(win))
        win.controller.move_layer_up(back_uid)
        _drain(app, 0.12)
        for _ in range(3):
            frames.append(_grab(win)); _drain(app, 0.08)

    # Hold final pose.
    for _ in range(3):
        frames.append(_grab(win)); _drain(app, 0.08)
    win.close()
    return _save_gif(frames, OUT / "gif_01_compose.gif", fps=7)


def record_appearance(app: QApplication) -> Path:
    win = MainWindow(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=ROOT / "generated" / "_gif_appearance",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    win.resize(1460, 900)
    win.show(); _drain(app, 0.25)
    _wait_ready(win, app)
    _prefer_background(win, "road")
    win.doc.replace_instances([])
    w, h = win.doc.bg_size
    order = _cutout_order(win)
    layout = [(0.24, 0.76, 0.38, 16.0), (0.55, 0.74, 0.30, -20.0), (0.82, 0.80, 0.18, 26.0)]
    for idx, (xr, yr, hr, ang) in zip(order, layout):
        win._add_instance(idx, w * xr, h * yr)
        inst = win.doc.instances[-1]
        inst.h_ratio = hr; inst.angle = ang; inst.invalidate_cache()
    win.doc.scene_recipe = "camera-mild"
    win.doc.object_appearance_recipe = "mild"
    win.doc.blend_mode = "gaussian"
    win.doc.empty_scene_prob = 0.10
    win.gen_defaults.reflect(win.doc)
    person = win.doc.instances[0]
    win.doc.select(person.uid)
    win.doc._emit(); _fit_canvas(win); _bias_splitter(win)
    _show_right_tab(win, 1)  # 变换·外观
    win.controls.reflect(win.doc)
    _drain(app, 0.25)

    frames = []
    steps = [
        {},
        {"appearance_enabled": True, "appearance_recipe": "off", "appearance_seed": 7,
         "appearance_brightness": 0.12},
        {"appearance_contrast": 1.15, "appearance_saturation": 1.18},
        {"appearance_hue": 12.0, "appearance_temperature": 14.0},
        {"appearance_blur": 1.2, "appearance_noise": 5.0, "appearance_sharpness": 1.25},
        # Reset then show a recipe-bake path (random → sliders).
        {"appearance_brightness": 0.0, "appearance_contrast": 1.0,
         "appearance_saturation": 1.0, "appearance_hue": 0.0,
         "appearance_temperature": 0.0, "appearance_blur": 0.0,
         "appearance_noise": 0.0, "appearance_sharpness": 1.0,
         "appearance_recipe": "mild", "appearance_seed": 21},
    ]
    for state in steps:
        if state:
            for k, v in state.items():
                setattr(person, k, v)
            person.invalidate_cache()
            win.doc._emit()
            win.controls.reflect(win.doc)
            _drain(app, 0.1)
        for _ in range(3):
            frames.append(_grab(win)); _drain(app, 0.06)

    # Explicit resample bake: fills sliders from mild sample, recipe → off.
    from compose_app.rendering import sample_recipe_into_sliders
    person.appearance_recipe = "mild"
    person.appearance_seed = 88
    sample_recipe_into_sliders(
        person, win.doc.cutouts[person.cutout_index].rgba,
        class_label=win.doc.cutouts[person.cutout_index].label,
        recipe_name="mild",
    )
    win.doc._emit()
    win.controls.reflect(win.doc)
    _drain(app, 0.15)
    for _ in range(5):
        frames.append(_grab(win)); _drain(app, 0.07)

    win.close()
    path = _save_gif(frames, OUT / "gif_02_appearance.gif", fps=8)
    shutil.copyfile(path, OUT / "ui_demo.gif")
    print(f"OK ui_demo.gif (copy of {path.name})")
    return path


def record_batch_defaults(app: QApplication) -> Path:
    win = MainWindow(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=ROOT / "generated" / "_gif_batch",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    win.resize(1460, 900)
    win.show(); _drain(app, 0.25)
    _wait_ready(win, app)
    _prefer_background(win, "road")
    win.doc.replace_instances([])
    w, h = win.doc.bg_size
    order = _cutout_order(win)
    layout = [(0.28, 0.74, 0.36, 12.0), (0.58, 0.74, 0.28, -14.0), (0.82, 0.80, 0.17, 20.0)]
    for idx, (xr, yr, hr, ang) in zip(order, layout):
        win._add_instance(idx, w * xr, h * yr)
        inst = win.doc.instances[-1]
        inst.h_ratio = hr; inst.angle = ang
        inst.appearance_enabled = True
        inst.appearance_recipe = "mild"
        inst.appearance_seed = 3
        inst.invalidate_cache()
    win.doc.select(win.doc.instances[0].uid)
    win.doc._emit(); _fit_canvas(win); _bias_splitter(win)
    _show_right_tab(win, 2)  # 批量默认
    _drain(app, 0.2)

    frames = []
    presets = [
        ("", "mild", "alpha", 0.0),
        ("camera-mild", "mild", "gaussian", 0.10),
        ("surveillance", "surveillance-object", "gaussian", 0.15),
        ("low-light", "mild", "hard", 0.05),
        ("camera-mild", "mild", "gaussian", 0.10),
    ]
    for scene, obj, blend, empty in presets:
        win.doc.scene_recipe = scene
        win.doc.object_appearance_recipe = obj
        win.doc.blend_mode = blend
        win.doc.empty_scene_prob = empty
        win.gen_defaults.reflect(win.doc)
        # Apply default appearance onto instances for visible effect.
        for inst in win.doc.instances:
            if obj and obj != "off":
                inst.appearance_enabled = True
                inst.appearance_recipe = obj
                inst.appearance_seed = (inst.uid * 17 + 5) % 1000
                inst.invalidate_cache()
        win.doc._emit(); _drain(app, 0.12)
        for _ in range(4):
            frames.append(_grab(win)); _drain(app, 0.06)
    win.close()
    return _save_gif(frames, OUT / "gif_03_batch_defaults.gif", fps=7)


def record_explorer(app: QApplication, demo: Path) -> Path:
    from compose_app_qt.explorer import DatasetExplorerWindow

    explorer = DatasetExplorerWindow(dataset_root=demo)
    explorer.resize(1280, 820)
    explorer.show(); _drain(app, 0.6)
    frames = []
    n = min(len(explorer.items), 4)
    for i in range(n):
        explorer.file_list.setCurrentRow(i)
        _drain(app, 0.25)
        for _ in range(5):
            frames.append(_grab(explorer, width=1000)); _drain(app, 0.05)
    explorer.close()
    return _save_gif(frames, OUT / "gif_04_explorer.gif", fps=6)


def record_data_loop(app: QApplication, demo: Path) -> Path:
    from compose_app_qt.data_loop import DataLoopCenterDialog

    loop = DataLoopCenterDialog(dataset_root=demo, output_root=demo / "shards")
    loop.resize(960, 740)
    loop.show(); _drain(app, 0.4)
    frames = []
    for i in range(loop.tabs.count()):
        loop.tabs.setCurrentIndex(i)
        _drain(app, 0.2)
        for _ in range(4):
            frames.append(_grab(loop, width=900)); _drain(app, 0.05)
    loop.close()
    return _save_gif(frames, OUT / "gif_05_data_loop.gif", fps=6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    # Only re-record editor-facing clips by default (compose / appearance / batch).
    # Pass --all to also refresh cutout / explorer / data-loop GIFs.
    only_editor = "--all" not in sys.argv
    print("Recording workflow GIFs…", "(editor clips)" if only_editor else "(all)")
    if not only_editor:
        record_cutout_studio(app)
    record_compose(app)
    record_appearance(app)
    record_batch_defaults(app)
    if not only_editor:
        demo = _ensure_demo_dataset(app)
        record_explorer(app, demo)
        record_data_loop(app, demo)
    print("Done →", OUT)
    print("Files:", ", ".join(p.name for p in sorted(OUT.glob("gif_*.gif"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
