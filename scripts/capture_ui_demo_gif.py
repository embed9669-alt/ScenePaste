#!/usr/bin/env python3
"""Record a short animated GIF of the ScenePaste editor UI.

Demonstrates batch-generation defaults + live object appearance preview
over a few seconds. Frames are captured via ``QWidget.grab()`` so the
result is reproducible without a separate screen recorder.

Usage::

    python scripts/capture_ui_demo_gif.py
    python scripts/capture_ui_demo_gif.py -o docs/images/ui_demo.gif --seconds 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "docs" / "images" / "ui_demo.gif"
SAMPLES = ROOT / "samples"

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter  # noqa: E402

from compose_app_qt.app import MainWindow  # noqa: E402


def _drain(app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def _wait_ready(win: MainWindow, app: QApplication, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        time.sleep(0.05)
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


def _compose_scene(win: MainWindow) -> None:
    _prefer_background(win, "road")
    w, h = win.doc.bg_size
    if w <= 0 or h <= 0 or not win.doc.cutouts:
        return
    win.doc.replace_instances([])
    by_label = {c.label.lower(): i for i, c in enumerate(win.doc.cutouts)}
    order = []
    for key in ("person", "truck", "motorcycle"):
        if key in by_label:
            order.append(by_label[key])
    for i in range(len(win.doc.cutouts)):
        if i not in order:
            order.append(i)
    layout = [
        (0.24, 0.76, 0.38, 16.0, False),
        (0.55, 0.74, 0.30, -20.0, False),
        (0.82, 0.80, 0.18, 26.0, False),
    ]
    for idx, (xr, yr, hr, ang, flip) in zip(order, layout):
        win._add_instance(idx, w * xr, h * yr)
        inst = win.doc.instances[-1]
        inst.h_ratio = hr
        inst.angle = ang
        inst.flip = flip
        inst.appearance_enabled = False
        inst.invalidate_cache()

    win.doc.scene_recipe = "camera-mild"
    win.doc.object_appearance_recipe = "mild"
    win.doc.blend_mode = "gaussian"
    win.doc.empty_scene_prob = 0.10
    win.gen_defaults.reflect(win.doc)
    if win.doc.instances:
        win.doc.select(win.doc.instances[0].uid)
    win.doc._emit()


def _qpixmap_to_pil(pix) -> Image.Image:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pix.save(buf, "PNG")
    data = bytes(buf.data())
    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGBA")


def _grab(win: MainWindow, width: int) -> Image.Image:
    img = _qpixmap_to_pil(win.grab())
    if width and img.width > width:
        ratio = width / float(img.width)
        img = img.resize((width, max(1, int(round(img.height * ratio)))), Image.LANCZOS)
    # GIF palette works better from RGB.
    return img.convert("RGB")


def _set_appearance(win: MainWindow, **kwargs) -> None:
    inst = win.doc.selected()
    if inst is None and win.doc.instances:
        win.doc.select(win.doc.instances[0].uid)
        inst = win.doc.selected()
    if inst is None:
        return
    for key, value in kwargs.items():
        setattr(inst, key, value)
    inst.invalidate_cache()
    win.doc._emit()
    win.controls.reflect(win.doc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seconds", type=float, default=4.5, help="approx GIF duration")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1100, help="output width in pixels")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=ROOT / "generated" / "_gif_demo",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    win.resize(1480, 920)
    win.setWindowTitle("ScenePaste")
    win.show()
    win.raise_()
    win.activateWindow()

    frames: list[Image.Image] = []
    frame_ms = max(40, int(round(1000.0 / max(1.0, args.fps))))

    def record() -> None:
        if not _wait_ready(win, app):
            print("WARN: UI not fully ready", file=sys.stderr)
        _compose_scene(win)
        _drain(app, 0.5)
        if win.canvas._bg_item is not None:
            win.canvas.fitInView(win.canvas._bg_item, Qt.KeepAspectRatio)
        for sp in win.findChildren(QSplitter):
            if sp.count() >= 3:
                sp.setSizes([180, 780, 420])
        _drain(app, 0.35)

        # Timeline of UI state changes (seconds into the clip).
        timeline = [
            (0.0, dict()),  # hold initial layout
            (0.6, {"appearance_enabled": True, "appearance_recipe": "mild", "appearance_seed": 7}),
            (1.2, {"appearance_brightness": 0.12, "appearance_saturation": 1.0}),
            (1.8, {"appearance_brightness": -0.08, "appearance_saturation": 1.18}),
            (2.4, {"appearance_brightness": 0.05, "appearance_contrast": 1.15, "appearance_blur": 0.6}),
            (3.0, {"appearance_seed": 99, "appearance_brightness": 0.0, "appearance_contrast": 1.0,
                   "appearance_saturation": 1.0, "appearance_blur": 0.0}),
            (3.6, {"appearance_recipe": "surveillance-object", "appearance_seed": 21}),
            (4.2, {"appearance_recipe": "mild", "appearance_seed": 21, "appearance_saturation": 1.1}),
        ]
        # Stretch/compress timeline to requested duration.
        scale = float(args.seconds) / max(0.1, timeline[-1][0] + 0.4)
        timeline = [(t * scale, state) for t, state in timeline]

        t = 0.0
        step = 1.0 / max(1.0, args.fps)
        state_idx = 0
        applied = {}
        while t <= args.seconds + 1e-6:
            while state_idx < len(timeline) and t + 1e-6 >= timeline[state_idx][0]:
                applied.update(timeline[state_idx][1])
                if timeline[state_idx][1]:
                    _set_appearance(win, **applied)
                    _drain(app, 0.08)
                state_idx += 1
            frames.append(_grab(win, args.width))
            _drain(app, step * 0.55)
            t += step

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Slightly dithered palette GIF; keep file size blog-friendly.
        first, rest = frames[0], frames[1:]
        first.save(
            args.output,
            save_all=True,
            append_images=rest,
            duration=frame_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        size_kb = args.output.stat().st_size / 1024.0
        print(
            f"OK {args.output} · {len(frames)} frames · "
            f"{args.seconds:.1f}s @ ~{args.fps:.0f}fps · {size_kb:.0f} KB · "
            f"{first.width}x{first.height}"
        )
        app.quit()

    QTimer.singleShot(250, record)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
