#!/usr/bin/env python3
"""Capture Qt GUI screenshots into docs/images/ for README / docs.

Requires a working display (or Qt offscreen). Prefer a real DISPLAY so fonts
and window chrome look natural.

Usage::

    python scripts/capture_ui_screenshots.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
SAMPLES = ROOT / "samples"

# Prefer the session display; allow headless fallback.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app_qt.app import MainWindow  # noqa: E402
from compose_app_qt.panels import ProjectSettingsDialog  # noqa: E402
from compose_app_qt.theme import qss_for  # noqa: E402


def _drain(app: QApplication, seconds: float = 2.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.03)


def _wait_ready(win: MainWindow, app: QApplication, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        time.sleep(0.05)
        if len(win.doc.cutouts) >= 1 and win.doc.bg_size[0] > 0:
            return True
    return False


def _prefer_background(win: MainWindow, keyword: str = "road") -> None:
    """Switch to a more photogenic sample background when available."""
    paths = win.doc.background_paths
    for i, p in enumerate(paths):
        if keyword in Path(p).stem.lower():
            if i != win.doc.background_index:
                win.doc.background_index = i
                win.doc.instances = []
                win.doc.selected_uid = None
                win.doc._emit()
            return


def _compose_demo_scene(win: MainWindow) -> None:
    """Place cutouts and enable appearance so the right panel demos clearly."""
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
        inst.invalidate_cache()

    # Main-window batch defaults (always-visible right panel).
    win.doc.scene_recipe = "camera-mild"
    win.doc.object_appearance_recipe = "mild"
    win.doc.blend_mode = "gaussian"
    win.doc.empty_scene_prob = 0.10
    win.gen_defaults.reflect(win.doc)

    # Selected instance shows Object Appearance Preview controls.
    if win.doc.instances:
        person = win.doc.instances[0]
        person.appearance_enabled = True
        person.appearance_recipe = "mild"
        person.appearance_seed = 42
        person.appearance_brightness = 0.08
        person.appearance_saturation = 1.12
        person.appearance_blur = 0.4
        person.invalidate_cache()
        win.doc.select(person.uid)
    win.doc._emit()


def _save(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    ok = pix.save(str(path), "PNG")
    print(f"{'OK' if ok else 'FAIL'} {path} ({pix.width()}x{pix.height()})")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    OUT.mkdir(parents=True, exist_ok=True)

    win = MainWindow(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=ROOT / "generated" / "_screenshot",
        class_map_text="person=0,truck=1,motorcycle=2",
        theme_mode="dark",
    )
    # Wider right panel so batch defaults + appearance preview are readable.
    win.resize(1560, 960)
    win.setMinimumSize(1280, 820)
    win.setWindowTitle("ScenePaste")
    win.show()
    win.raise_()
    win.activateWindow()

    def capture() -> None:
        if not _wait_ready(win, app):
            print("WARN: cutouts/background not ready; capturing anyway", file=sys.stderr)
        _compose_demo_scene(win)
        _drain(app, 0.6)
        if win.canvas._bg_item is not None:
            win.canvas.fitInView(win.canvas._bg_item, Qt.KeepAspectRatio)
        # Bias splitter toward the right panel for documentation shots.
        if hasattr(win, "centralWidget"):
            splitters = win.findChildren(__import__("PySide6.QtWidgets", fromlist=["QSplitter"]).QSplitter)
            for sp in splitters:
                if sp.count() >= 3:
                    sp.setSizes([200, 820, 420])
        _drain(app, 0.5)

        _save(win, OUT / "ui_overview.png")

        win.setStyleSheet(qss_for("light"))
        _drain(app, 0.3)
        _save(win, OUT / "ui_overview_light.png")
        win.setStyleSheet(qss_for("dark"))
        _drain(app, 0.2)

        dlg = ProjectSettingsDialog(win.doc, parent=win)
        dlg.setWindowTitle("ScenePaste — 项目设置")
        dlg.resize(480, 520)
        dlg.show()
        _drain(app, 0.3)
        _save(dlg, OUT / "ui_settings.png")
        dlg.close()

        app.quit()

    QTimer.singleShot(200, capture)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
