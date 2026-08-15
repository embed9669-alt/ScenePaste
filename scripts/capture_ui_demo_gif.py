#!/usr/bin/env python3
"""Record a hero GIF of the ScenePaste welcome / home page.

Intended for the README hero shot. Keeps the window on the Welcome Home
stack (not the compose workspace) and lightly animates focus across the
primary actions so the page feels alive.

Usage::

    python3 scripts/capture_ui_demo_gif.py
    python3 scripts/capture_ui_demo_gif.py -o docs/images/ui_demo.gif
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "docs" / "images" / "ui_demo.gif"
SAMPLES = ROOT / "samples"

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app_qt.app import MainWindow  # noqa: E402
from compose_app_qt.theme import qss_for  # noqa: E402


def _drain(app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.015)


def _qpixmap_to_pil(pix) -> Image.Image:
    from PySide6.QtCore import QBuffer, QIODevice

    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pix.save(buf, "PNG")
    return Image.open(BytesIO(bytes(buf.data()))).convert("RGB")


def _grab(win: MainWindow, width: int) -> Image.Image:
    img = _qpixmap_to_pil(win.grab())
    if width and img.width > width:
        ratio = width / float(img.width)
        img = img.resize((width, max(1, int(round(img.height * ratio)))), Image.LANCZOS)
    return img


def _save_gif(frames: list[Image.Image], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("no frames")
    ms = max(50, int(round(1000.0 / max(1.0, fps))))
    # Quantize for a README-friendly size.
    out_frames = []
    for frame in frames:
        out_frames.append(frame.quantize(colors=112, method=Image.MEDIANCUT))
    out_frames[0].save(
        path,
        save_all=True,
        append_images=out_frames[1:],
        duration=ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _focus_button(btn) -> None:
    if btn is None:
        return
    btn.setDefault(True)
    btn.setFocus(Qt.OtherFocusReason)
    btn.setDown(False)


def _pulse_down(btn, down: bool) -> None:
    if btn is None:
        return
    btn.setDown(bool(down))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1100)
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv[:1])

    # Do NOT load sample assets — stay on Welcome Home.
    win = MainWindow(
        objects_dir=None,
        backgrounds_dir=None,
        output_dir=ROOT / "generated" / "_gif_welcome",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    win.resize(1440, 900)
    win.setWindowTitle("ScenePaste")
    win.show()
    win.raise_()
    win.activateWindow()

    frames: list[Image.Image] = []

    def record() -> None:
        # Force welcome page even if settings restored a previous session.
        if hasattr(win, "_home_stack") and hasattr(win, "_welcome_home"):
            win._home_stack.setCurrentWidget(win._welcome_home)

        # Seed one example recent project so the home card looks populated.
        example = SAMPLES / "scenepaste.project.example.json"
        if example.is_file():
            win._remember_recent_project(example)
        win._refresh_recent_projects()
        _drain(app, 0.45)

        buttons = [
            win._banner_btn_asset,
            win._banner_btn_factory,
            win._banner_btn_open,
            win._banner_btn_objects,
            win._banner_btn_sample,
            win._banner_btn_docs,
        ]

        # Beat 1: hold the full welcome page.
        for _ in range(5):
            frames.append(_grab(win, args.width))
            _drain(app, 0.1)

        # Beat 2: walk focus across primary / secondary actions.
        for btn in buttons:
            for other in buttons:
                other.setDefault(other is btn)
                other.setDown(False)
            _focus_button(btn)
            _drain(app, 0.12)
            frames.append(_grab(win, args.width))
            _pulse_down(btn, True)
            _drain(app, 0.1)
            frames.append(_grab(win, args.width))
            _pulse_down(btn, False)
            _drain(app, 0.08)
            frames.append(_grab(win, args.width))

        # Beat 3: brief light-theme flash, then back to dark (shows polish).
        win.setStyleSheet(qss_for("light"))
        _drain(app, 0.2)
        for _ in range(3):
            frames.append(_grab(win, args.width))
            _drain(app, 0.1)
        win.setStyleSheet(qss_for("dark"))
        _drain(app, 0.2)

        # Beat 4: end on the primary CTA focused.
        for other in buttons:
            other.setDefault(other is win._banner_btn_asset)
            other.setDown(False)
        _focus_button(win._banner_btn_asset)
        _drain(app, 0.15)
        for _ in range(5):
            frames.append(_grab(win, args.width))
            _drain(app, 0.1)

        _save_gif(frames, args.output, args.fps)
        size_kb = args.output.stat().st_size / 1024.0
        print(
            f"OK {args.output} · {len(frames)} frames · "
            f"~{args.fps:.0f}fps · {size_kb:.0f} KB · "
            f"{frames[0].width}x{frames[0].height}"
        )
        win.close()
        app.quit()

    QTimer.singleShot(250, record)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
