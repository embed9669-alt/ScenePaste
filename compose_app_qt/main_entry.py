"""Console-script entry point for the Qt GUI.

Usage::

    scenepaste gui
    scenepaste gui --objects ./o --backgrounds ./b --output ./out
    python -m compose_app_qt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


def main_entry(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ScenePaste Qt scene compositor")
    parser.add_argument("--project", type=Path, default=None, help="ScenePaste project manifest")
    parser.add_argument("--objects", type=Path, default=Path("objects"),
                        help="目标 JSON + 原图所在目录（默认 ./objects）")
    parser.add_argument("--backgrounds", type=Path, default=Path("backgrounds"),
                        help="背景图目录（默认 ./backgrounds）")
    parser.add_argument("--output", type=Path, default=Path("generated"),
                        help="合成结果输出目录（默认 ./generated）")
    parser.add_argument("--class-map", default="person=0,vehicle=1",
                        help="初始类别映射")
    parser.add_argument("--theme", default=None, choices=("dark", "light"),
                        help="UI 主题；省略时沿用上次选择")
    # Accepted as a no-op for older scripts that still pass --qt.
    parser.add_argument("--qt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.project is not None:
        try:
            from scenepaste.project import ScenePasteProject
            project = ScenePasteProject.load(args.project)
            if project.objects_dir is not None: args.objects = project.objects_dir
            if project.backgrounds_dir is not None: args.backgrounds = project.backgrounds_dir
            if project.output_dir is not None: args.output = project.output_dir
            if project.class_map:
                args.class_map = ",".join(f"{k}={v}" for k,v in sorted(project.class_map.items(), key=lambda kv: kv[1]))
        except Exception as exc:
            print(f"Project manifest 加载失败：{exc}", file=sys.stderr)
            return 1

    # PySide6 needs a real (or offscreen) display; honour the standard env.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        print(
            f"缺少 PySide6：{exc}\n"
            "请安装：python -m pip install 'scenepaste[gui-qt]'",
            file=sys.stderr,
        )
        return 1

    from .app import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow(
        objects_dir=args.objects if args.objects.exists() else None,
        backgrounds_dir=args.backgrounds if args.backgrounds.exists() else None,
        output_dir=args.output,
        class_map_text=args.class_map,
        theme_mode=args.theme,
    )
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main_entry())
