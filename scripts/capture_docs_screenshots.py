#!/usr/bin/env python3
"""Capture README/doc screenshots beyond the main editor.

Produces:
  docs/images/example_formats.png   — Detect / Seg / OBB / Semantic collage
  docs/images/example_qa.png        — compact QA metrics card from qa_report.json
  docs/images/ui_explorer.png       — Dataset Explorer window
  docs/images/ui_data_loop.png      — Data Loop Center dialog

Also refreshes the editor shots via ``capture_ui_screenshots`` helpers when requested.

Usage::

    python scripts/capture_docs_screenshots.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
SAMPLES = ROOT / "samples"
DEMO = ROOT / "generated" / "_docs_demo"

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _drain(app: QApplication, seconds: float = 0.4) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def _font(size: int = 18):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _write_angled_template() -> Path:
    """Template with visible rotations so OBB differs clearly from Detect."""
    import json

    tpl = {
        "schema": "scenepaste/scene-template",
        "version": 2,
        "canvas": {"width": 800, "height": 600},
        "global": {"enabled_probability": 1.0, "flip_probability": 0.0, "same_class_random": False},
        "instances": [
            {
                "source": "sample_person.json#0",
                "label": "person",
                "class_id": 0,
                "cx_ratio": 0.28,
                "cy_ratio": 0.72,
                "h_ratio": 0.38,
                "flip": False,
                "angle": 18.0,
                "variation": {
                    "cx_range": [0.28, 0.28],
                    "cy_range": [0.72, 0.72],
                    "h_range": [0.38, 0.38],
                    "angle_range": [18.0, 18.0],
                    "enabled_probability": 1.0,
                    "flip_probability": 0.0,
                    "same_class_random": False,
                    "allow_overlap": True,
                },
            },
            {
                "source": "sample_truck.json#0",
                "label": "truck",
                "class_id": 1,
                "cx_ratio": 0.58,
                "cy_ratio": 0.74,
                "h_ratio": 0.30,
                "flip": False,
                "angle": -22.0,
                "variation": {
                    "cx_range": [0.58, 0.58],
                    "cy_range": [0.74, 0.74],
                    "h_range": [0.30, 0.30],
                    "angle_range": [-22.0, -22.0],
                    "enabled_probability": 1.0,
                    "flip_probability": 0.0,
                    "same_class_random": False,
                    "allow_overlap": True,
                },
            },
            {
                "source": "sample_motorcycle.json#0",
                "label": "motorcycle",
                "class_id": 2,
                "cx_ratio": 0.82,
                "cy_ratio": 0.78,
                "h_ratio": 0.20,
                "flip": False,
                "angle": 28.0,
                "variation": {
                    "cx_range": [0.82, 0.82],
                    "cy_range": [0.78, 0.78],
                    "h_range": [0.20, 0.20],
                    "angle_range": [28.0, 28.0],
                    "enabled_probability": 1.0,
                    "flip_probability": 0.0,
                    "same_class_random": False,
                    "allow_overlap": True,
                },
            },
        ],
    }
    path = DEMO.parent / "_docs_angled_template.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _generate_demo_dataset() -> Path:
    from scenepaste import GenerationConfig, generate_dataset

    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True, exist_ok=True)
    template = _write_angled_template()
    cfg = GenerationConfig(
        objects_dir=SAMPLES / "objects",
        backgrounds_dir=SAMPLES / "backgrounds",
        output_dir=DEMO,
        class_map={"person": 0, "truck": 1, "motorcycle": 2},
        count=4,
        min_objects=3,
        max_objects=3,
        output_format="all",
        seed=11,
        preview_ratio=1.0,
        save_previews=True,
        workers=1,
        scene_template=template,
        empty_scene_probability=0.0,
    )
    summary = generate_dataset(cfg)
    print(f"generated demo dataset → {DEMO} ({summary})")
    return DEMO


def _refresh_before_after(root: Path) -> None:
    """Update example_before_after.jpg from a generated preview + background."""
    previews = sorted((root / "previews").rglob("*.jpg"))
    bgs = sorted((SAMPLES / "backgrounds").glob("*.jpg"))
    if not previews or not bgs:
        print("WARN: skip before/after (missing preview or background)")
        return
    # Prefer road background if present.
    bg_path = next((p for p in bgs if "road" in p.stem), bgs[0])
    left = Image.open(bg_path).convert("RGB")
    right = Image.open(previews[0]).convert("RGB")
    h = 360
    left.thumbnail((640, h), Image.Resampling.LANCZOS)
    right.thumbnail((640, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (left.width + right.width + 12, max(left.height, right.height) + 40), (22, 22, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Background", fill=(200, 200, 205), font=_font(16))
    draw.text((left.width + 24, 10), "ScenePaste (multi-format labels)", fill=(200, 200, 205), font=_font(16))
    canvas.paste(left, (0, 40))
    canvas.paste(right, (left.width + 12, 40))
    out = OUT / "example_before_after.jpg"
    canvas.save(out, "JPEG", quality=90)
    print(f"OK {out}")


def _render_modality(root: Path, item, modality: str) -> Image.Image:
    """Render a single annotation modality for a clearer format gallery."""
    from scenepaste.explorer import (
        _load_coco,
        _overlay_semantic,
        _parse_label_lines,
        _read_classes,
        _render_coco,
        _render_yolo,
    )

    base = Image.open(item.path).convert("RGB")
    classes = _read_classes(root)
    if modality == "detect":
        rows = _parse_label_lines(root / "labels" / item.split / f"{item.stem}.txt")
        if rows:
            _render_yolo(base, rows, classes)
    elif modality == "seg":
        rows = _parse_label_lines(root / "labels-seg" / item.split / f"{item.stem}.txt")
        if rows:
            _render_yolo(base, rows, classes)
    elif modality == "obb":
        rows = _parse_label_lines(root / "labels-obb" / item.split / f"{item.stem}.txt")
        if rows:
            _render_yolo(base, rows, classes)
    elif modality == "semantic":
        mask = root / "masks" / item.split / f"{item.stem}.png"
        if mask.exists():
            base, _ = _overlay_semantic(base, mask)
    elif modality == "coco":
        coco_by_file, coco_names = _load_coco(root)
        anns = coco_by_file.get(item.path.name, [])
        if anns:
            _render_coco(base, anns, coco_names)
    return base


def _captioned(img: Image.Image, title: str, size=(640, 400)) -> Image.Image:
    img = img.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 36), (24, 24, 28))
    x = (size[0] - img.width) // 2
    y = 36 + (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0], 34), fill=(40, 44, 52))
    draw.text((12, 8), title, fill=(230, 230, 235), font=_font(16))
    return canvas


def build_format_collage(root: Path) -> Path:
    from scenepaste.explorer import index_dataset

    items = index_dataset(root)
    if not items:
        raise RuntimeError(f"no images indexed under {root}")
    # Prefer a scene with multiple objects when possible.
    item = items[0]
    panels = [
        ("YOLO Detect", _render_modality(root, item, "detect")),
        ("YOLO Segmentation", _render_modality(root, item, "seg")),
        ("YOLO OBB", _render_modality(root, item, "obb")),
        ("Semantic Mask", _render_modality(root, item, "semantic")),
    ]
    caps = [_captioned(im, title) for title, im in panels]
    w, h = caps[0].size
    collage = Image.new("RGB", (w * 2 + 8, h * 2 + 8), (18, 18, 20))
    collage.paste(caps[0], (0, 0))
    collage.paste(caps[1], (w + 8, 0))
    collage.paste(caps[2], (0, h + 8))
    collage.paste(caps[3], (w + 8, h + 8))
    out = OUT / "example_formats.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    collage.save(out, "PNG")
    print(f"OK {out} ({collage.width}x{collage.height})")
    return out


def build_qa_card(root: Path) -> Path:
    from scenepaste.tools.qa import write_qa_dashboard

    html_path = root / "qa_dashboard.html"
    json_path = root / "qa_report.json"
    report = write_qa_dashboard(root, html_path=html_path, json_path=json_path)
    integrity = report.get("integrity") or {}
    summary = report.get("summary") or {}
    yolo = summary.get("yolo") or {}
    distro = report.get("distribution") or {}
    classes = distro.get("classes") or {}
    lines = [
        "ScenePaste QA Dashboard",
        f"health: {report.get('health', '—')}",
        f"images: {summary.get('images', yolo.get('images', '—'))}",
        f"exact duplicates: {integrity.get('duplicate_images', 0)}",
        f"pHash near-duplicates: {integrity.get('near_duplicate_images', 0)}",
        f"unreadable images: {integrity.get('unreadable_images', 0)}",
        "covers Detect / Seg / OBB / Semantic / COCO integrity",
    ]
    if classes:
        top = ", ".join(
            f"{name}:{row.get('count', row) if isinstance(row, dict) else row}"
            for name, row in list(classes.items())[:5]
        )
        lines.append(f"classes: {top}")
    warns = report.get("warnings") or []
    if warns:
        lines.append(f"warnings: {len(warns)}")

    card = Image.new("RGB", (900, 420), (28, 30, 36))
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, 900, 48), fill=(50, 90, 60))
    draw.text((20, 12), "QA · integrity / duplicates / distributions", fill=(235, 245, 230), font=_font(20))
    y = 70
    for line in lines[:12]:
        draw.text((28, y), str(line), fill=(220, 220, 225), font=_font(18))
        y += 28
    draw.text(
        (28, 370),
        "Full interactive report: scenepaste qa ./generated  →  qa_dashboard.html",
        fill=(160, 170, 150),
        font=_font(14),
    )
    out = OUT / "example_qa.png"
    card.save(out, "PNG")
    print(f"OK {out} (also wrote {html_path.name})")
    return out


def capture_qt_windows(root: Path) -> None:
    from compose_app_qt.data_loop import DataLoopCenterDialog
    from compose_app_qt.explorer import DatasetExplorerWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])

    explorer = DatasetExplorerWindow(dataset_root=root)
    explorer.resize(1280, 820)
    explorer.show()
    _drain(app, 0.8)
    if explorer.items:
        explorer.file_list.setCurrentRow(0)
        _drain(app, 0.5)
    pix = explorer.grab()
    path = OUT / "ui_explorer.png"
    pix.save(str(path), "PNG")
    print(f"OK {path} ({pix.width()}x{pix.height()})")
    explorer.close()

    loop = DataLoopCenterDialog(dataset_root=root, output_root=root)
    loop.resize(960, 740)
    # Show the Quality tab if present.
    for i in range(loop.tabs.count()):
        if "QA" in loop.tabs.tabText(i) or "质量" in loop.tabs.tabText(i) or "Quality" in loop.tabs.tabText(i):
            loop.tabs.setCurrentIndex(i)
            break
    loop.show()
    _drain(app, 0.5)
    pix = loop.grab()
    path = OUT / "ui_data_loop.png"
    pix.save(str(path), "PNG")
    print(f"OK {path} ({pix.width()}x{pix.height()})")
    loop.close()
    app.quit()


def _refresh_editor_shots() -> None:
    """Run the editor capture script for overview / settings images."""
    import subprocess

    script = ROOT / "scripts" / "capture_ui_screenshots.py"
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT), env=env)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    root = _generate_demo_dataset()
    _refresh_before_after(root)
    build_format_collage(root)
    try:
        build_qa_card(root)
    except Exception as exc:
        print(f"WARN: QA card skipped: {exc}", file=sys.stderr)
    capture_qt_windows(root)
    try:
        _refresh_editor_shots()
    except Exception as exc:
        print(f"WARN: editor shots skipped: {exc}", file=sys.stderr)
    print("Done. Images under", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
