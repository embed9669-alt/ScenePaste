"""ScenePaste Qt main window.

Toolbar + cutout gallery + canvas + instance controls + status bar.
Document model is the source of truth; undo goes through QUndoStack.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import List, Optional

import scenepaste.core as core
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QUndoStack
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from compose_app.models import Instance

from .canvas import CanvasView
from .panels import ControlsPanel, CutoutThumbWidget, GenerationDefaultsPanel, InstanceListWidget
from .state import Document
from .theme import qss_for
from .undo import AddInstanceCommand, DeleteSelectedCommand, TransformCommand
from .workers import LoadCutoutsWorker, composite_from_doc


def _instance_snapshots_equal(a: List[Instance], b: List[Instance]) -> bool:
    """True when two undo snapshots describe the same layout."""
    if len(a) != len(b):
        return False
    by_uid = {inst.uid: inst for inst in b}
    for left in a:
        right = by_uid.get(left.uid)
        if right is None:
            return False
        if (
            left.cutout_index != right.cutout_index
            or abs(left.cx - right.cx) > 1e-6
            or abs(left.cy - right.cy) > 1e-6
            or abs(left.h_ratio - right.h_ratio) > 1e-9
            or bool(left.flip) != bool(right.flip)
            or abs((left.angle % 360.0) - (right.angle % 360.0)) > 1e-6
            or bool(getattr(left, "appearance_enabled", False))
            != bool(getattr(right, "appearance_enabled", False))
            or str(getattr(left, "appearance_recipe", ""))
            != str(getattr(right, "appearance_recipe", ""))
            or int(getattr(left, "appearance_seed", 0))
            != int(getattr(right, "appearance_seed", 0))
            or abs(float(getattr(left, "appearance_brightness", 0.0))
                   - float(getattr(right, "appearance_brightness", 0.0))) > 1e-6
            or abs(float(getattr(left, "appearance_contrast", 1.0))
                   - float(getattr(right, "appearance_contrast", 1.0))) > 1e-6
            or abs(float(getattr(left, "appearance_saturation", 1.0))
                   - float(getattr(right, "appearance_saturation", 1.0))) > 1e-6
            or abs(float(getattr(left, "appearance_blur", 0.0))
                   - float(getattr(right, "appearance_blur", 0.0))) > 1e-6
        ):
            return False
    return True


class MainWindow(QMainWindow):
    """Assembles all panels and routes signals to controllers / undo stack."""

    def __init__(self, objects_dir: Optional[Path] = None,
                 backgrounds_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 class_map_text: str = "person=0,vehicle=1",
                 theme_mode: str = "dark"):
        super().__init__()
        self.setWindowTitle("ScenePaste — Scene Compositor")
        self.resize(1280, 820)

        self.doc = Document(
            output_dir=output_dir,
            class_map_text=class_map_text,
        )
        self._objects_dir = objects_dir
        self._backgrounds_dir = backgrounds_dir
        self._theme_mode = theme_mode
        self._worker: Optional[LoadCutoutsWorker] = None

        self.setStyleSheet(qss_for(theme_mode))

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        self._bind_shortcuts()

        # Auto-load initial directories if provided.
        if objects_dir and Path(objects_dir).is_dir():
            self.load_objects(Path(objects_dir))
        if backgrounds_dir and Path(backgrounds_dir).is_dir():
            self.load_backgrounds(Path(backgrounds_dir))

    # ------------------------------------------------------------- building
    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        self._empty_banner = QLabel(
            "尚未加载数据。点工具栏「★ 加载示例」可 30 秒上手，"
            "或使用「加载目标 / 加载背景」导入自己的数据。"
        )
        self._empty_banner.setWordWrap(True)
        self._empty_banner.setStyleSheet(
            "padding: 10px 14px; background: #3a4a2a; color: #e8f0d8; border-bottom: 1px solid #556644;"
        )
        outer.addWidget(self._empty_banner)

        splitter = QSplitter(Qt.Horizontal)
        self.thumbs = CutoutThumbWidget(self.doc)
        self.thumbs.canvas_place_requested.connect(self._place_at_centre)
        splitter.addWidget(self.thumbs)

        self.canvas = CanvasView(self.doc)
        self.canvas.transform_committed.connect(self._on_canvas_transform_committed)
        self.canvas.place_at_requested.connect(
            lambda idx, pos: self._add_instance(idx, float(pos.x()), float(pos.y()))
        )
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.instance_list = InstanceListWidget(self.doc)
        right_layout.addWidget(QLabel("实例"))
        right_layout.addWidget(self.instance_list, 1)

        self.gen_defaults = GenerationDefaultsPanel()
        self.gen_defaults.changed.connect(self._on_generation_defaults_changed)
        self.gen_defaults.apply_to_instances.connect(self._apply_default_appearance_to_instances)
        self.gen_defaults.reflect(self.doc)
        right_layout.addWidget(self.gen_defaults)

        self.controls = ControlsPanel()
        self.controls.scale_changed.connect(self._on_scale_changed)
        self.controls.angle_changed.connect(self._on_angle_changed)
        self.controls.flip_toggled.connect(self._on_flip)
        self.controls.delete_clicked.connect(self.delete_selected)
        self.controls.appearance_preview.connect(self._on_appearance_preview)
        self.controls.appearance_committed.connect(self._on_appearance_committed)
        self.controls.appearance_resample.connect(self._on_appearance_resample)
        self._appearance_before = None
        # Live wheel/Shift-drag must refresh spinboxes without a full doc emit.
        self.canvas.transform_preview.connect(lambda: self.controls.reflect(self.doc))
        right_layout.addWidget(self.controls)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 700, 280])
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        # Subscribe to doc changes for side-effects.
        self.doc.subscribe(self._after_doc_change)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)

        self.act_load_sample = QAction("★ 加载示例", self)
        self.act_load_sample.setToolTip("加载 samples/ 数据，30 秒上手")
        self.act_load_sample.triggered.connect(self.load_sample_dataset)
        tb.addAction(self.act_load_sample)

        self.act_open_project = QAction("📁 打开工程", self)
        self.act_open_project.setToolTip("打开 scenepaste.project.json")
        self.act_open_project.triggered.connect(self.open_project_manifest)
        tb.addAction(self.act_open_project)
        self.act_save_project = QAction("💼 保存工程", self)
        self.act_save_project.setToolTip("保存当前路径/类别/默认配置到项目 Manifest")
        self.act_save_project.triggered.connect(self.save_project_manifest)
        tb.addAction(self.act_save_project)

        tb.addSeparator()

        self.act_load_objects = QAction("加载目标", self)
        self.act_load_objects.triggered.connect(self._pick_objects_dir)
        tb.addAction(self.act_load_objects)

        self.act_load_bgs = QAction("加载背景", self)
        self.act_load_bgs.triggered.connect(self._pick_backgrounds_dir)
        tb.addAction(self.act_load_bgs)

        tb.addSeparator()
        self.act_prev_bg = QAction("◀ 上一张", self)
        self.act_prev_bg.triggered.connect(lambda: self._step_background(-1))
        tb.addAction(self.act_prev_bg)
        self.act_next_bg = QAction("下一张 ▶", self)
        self.act_next_bg.triggered.connect(lambda: self._step_background(+1))
        tb.addAction(self.act_next_bg)

        tb.addSeparator()
        self.act_save = QAction("💾 保存当前", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.triggered.connect(self.save_current)
        tb.addAction(self.act_save)

        self.act_batch = QAction("⚡ 批量套用", self)
        self.act_batch.setToolTip("把当前布局套用到剩余背景并保存")
        self.act_batch.triggered.connect(self.apply_to_next)
        tb.addAction(self.act_batch)

        self.act_large_generate = QAction("🚀 大规模生成", self)
        self.act_large_generate.setToolTip("真实分布驱动 / 参数化模板 / 多进程 / 断点恢复")
        self.act_large_generate.triggered.connect(self.open_large_generation)
        tb.addAction(self.act_large_generate)

        self.act_data_loop = QAction("🧠 数据闭环", self)
        self.act_data_loop.setToolTip("Hard Mining / QA / 泄漏检测 / 真实合成对比 / 多样性策展 / Sharding")
        self.act_data_loop.triggered.connect(self.open_data_loop_center)
        tb.addAction(self.act_data_loop)

        tb.addSeparator()
        self.act_settings = QAction("⚙ 项目设置", self)
        self.act_settings.triggered.connect(self.open_project_settings)
        tb.addAction(self.act_settings)

        self.act_explorer = QAction("🔎 数据集浏览", self)
        self.act_explorer.setToolTip("浏览已生成数据并叠加 Detect / Seg / OBB / COCO / Semantic 标注")
        self.act_explorer.triggered.connect(self.open_dataset_explorer)
        tb.addAction(self.act_explorer)

        self.act_random = QAction("🎲 随机铺满", self)
        self.act_random.setToolTip("随机铺满当前背景（替换现有实例）")
        self.act_random.triggered.connect(self.random_fill)
        tb.addAction(self.act_random)

        self.act_save_tpl = QAction("📝 存模板", self)
        self.act_save_tpl.triggered.connect(self.save_scene_template)
        tb.addAction(self.act_save_tpl)
        self.act_load_tpl = QAction("📂 读模板", self)
        self.act_load_tpl.triggered.connect(self.load_scene_template)
        tb.addAction(self.act_load_tpl)

        tb.addSeparator()
        self.act_undo = QAction("↶ 撤销", self)
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_undo.triggered.connect(self.undo)
        tb.addAction(self.act_undo)
        self.act_redo = QAction("↷ 重做", self)
        self.act_redo.setShortcut("Ctrl+Y")
        self.act_redo.triggered.connect(self.redo)
        tb.addAction(self.act_redo)

        tb.addSeparator()
        self.act_theme = QAction("浅色主题", self)
        self.act_theme.setCheckable(True)
        self.act_theme.setChecked(self._theme_mode == "light")
        self.act_theme.setToolTip("切换深色 / 浅色界面")
        self.act_theme.toggled.connect(self._toggle_theme)
        tb.addAction(self.act_theme)

        tb.addSeparator()
        self.act_about = QAction("关于", self)
        self.act_about.triggered.connect(self.show_about)
        tb.addAction(self.act_about)

    def open_project_manifest(self) -> None:
        from scenepaste.project import ScenePasteProject
        path, _ = QFileDialog.getOpenFileName(self, "打开 ScenePaste 工程", str(Path.cwd()), "ScenePaste Project (*.json)")
        if not path:
            return
        try:
            project = ScenePasteProject.load(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "无法打开工程", str(exc)); return
        self._project_manifest = Path(path)
        self._active_scene_template = project.scene_template
        if project.class_map:
            self.doc.class_map_text = ",".join(f"{k}={v}" for k,v in sorted(project.class_map.items(), key=lambda kv: kv[1]))
        if project.output_dir is not None:
            self.doc.output_dir = project.output_dir
        defaults = dict(project.defaults or {})
        if defaults.get("output_format"):
            self.doc.output_format = str(defaults["output_format"])
        if "augmentation_recipe" in defaults:
            self.doc.scene_recipe = str(defaults.get("augmentation_recipe") or "")
        if "object_appearance_recipe" in defaults:
            self.doc.object_appearance_recipe = str(defaults.get("object_appearance_recipe") or "")
        if defaults.get("blend_mode"):
            self.doc.blend_mode = str(defaults["blend_mode"])
        if "empty_scene_prob" in defaults:
            try:
                self.doc.empty_scene_prob = float(defaults["empty_scene_prob"])
            except (TypeError, ValueError):
                pass
        self.gen_defaults.reflect(self.doc)
        if project.objects_dir and project.objects_dir.is_dir():
            self.load_objects(project.objects_dir)
        if project.backgrounds_dir and project.backgrounds_dir.is_dir():
            self.load_backgrounds(project.backgrounds_dir)
        self.statusBar().showMessage(f"已打开工程：{project.name}", 5000)

    def save_project_manifest(self) -> None:
        from scenepaste.project import ScenePasteProject
        from scenepaste.core.config import parse_class_map
        default = str(getattr(self, "_project_manifest", Path.cwd() / "scenepaste.project.json"))
        path, _ = QFileDialog.getSaveFileName(self, "保存 ScenePaste 工程", default, "ScenePaste Project (*.json)")
        if not path:
            return
        try:
            project_path = Path(path)
            if project_path.exists():
                project = ScenePasteProject.load(project_path)
            else:
                project = ScenePasteProject(
                    path=project_path,
                    name=project_path.parent.name or "ScenePaste Project",
                )
            project.path = project_path
            project.update_generation(
                objects_dir=self._objects_dir,
                backgrounds_dir=self._backgrounds_dir,
                output_dir=self.doc.output_dir,
                class_map=parse_class_map(self.doc.class_map_text),
                scene_template=getattr(self, "_active_scene_template", None),
                defaults={
                    "workers": 0,
                    "preview_ratio": 0.01,
                    **self.doc.generation_defaults(),
                },
            )
            project.save()
            self._project_manifest = project_path
            self.statusBar().showMessage(f"工程已保存：{path}", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "保存工程失败", str(exc))

    def open_data_loop_center(self) -> None:
        from .data_loop import DataLoopCenterDialog
        self._data_loop = DataLoopCenterDialog(
            dataset_root=self.doc.output_dir, output_root=(Path(self.doc.output_dir) / "shards") if self.doc.output_dir else None, parent=self)
        self._data_loop.show(); self._data_loop.raise_()

    def open_large_generation(self) -> None:
        from .large_generate import LargeGenerationDialog
        self.gen_defaults.apply_to_document(self.doc)
        defaults = dict(self.doc.generation_defaults())
        distribution_profile = None
        scene_template = getattr(self, "_active_scene_template", None)
        project_path = getattr(self, "_project_manifest", None)
        if project_path and Path(project_path).is_file():
            try:
                from scenepaste.project import ScenePasteProject
                project = ScenePasteProject.load(Path(project_path))
                # Main-window defaults win; fill only missing keys from project.
                for key, value in dict(project.defaults or {}).items():
                    defaults.setdefault(key, value)
                distribution_profile = project.distribution_profile
                scene_template = project.scene_template or scene_template
            except Exception:
                # The generation dialog remains usable even if an optional
                # project manifest was moved or edited outside ScenePaste.
                pass
        self._large_generation = LargeGenerationDialog(
            objects_dir=self._objects_dir, backgrounds_dir=self._backgrounds_dir,
            output_dir=self.doc.output_dir, class_map_text=self.doc.class_map_text,
            output_format=self.doc.output_format, project_defaults=defaults,
            distribution_profile=distribution_profile, scene_template=scene_template, parent=self)
        self._large_generation.show()
        self._large_generation.raise_()

    def open_dataset_explorer(self) -> None:
        """Open the generated-dataset QA browser in a separate window."""
        from .explorer import DatasetExplorerWindow

        root = self.doc.output_dir
        if root is None or not (Path(root) / "images").exists():
            picked = QFileDialog.getExistingDirectory(
                self, "选择 ScenePaste 数据集", str(root or Path.cwd())
            )
            if not picked:
                return
            root = Path(picked)
        self._dataset_explorer = DatasetExplorerWindow(Path(root), parent=self)
        self._dataset_explorer.show()
        self._dataset_explorer.raise_()

    def _toggle_theme(self, checked: bool) -> None:
        self._theme_mode = "light" if checked else "dark"
        self.setStyleSheet(qss_for(self._theme_mode))
        # Keep the empty-state banner readable on both themes.
        if self._theme_mode == "light":
            self._empty_banner.setStyleSheet(
                "padding: 10px 14px; background: #e8f0d0; color: #334422; border-bottom: 1px solid #ccd4aa;"
            )
        else:
            self._empty_banner.setStyleSheet(
                "padding: 10px 14px; background: #3a4a2a; color: #e8f0d8; border-bottom: 1px solid #556644;"
            )

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._status_bg = QLabel("无背景")
        self._status_hint = QLabel(
            "提示: 滚轮缩放 · Shift+拖动旋转 · Ctrl+滚轮画布 · 方向键微调 · "
            "PgUp/PgDn换背景 · F翻转 · Ctrl+D复制 · Delete删除"
        )
        self._status_count = QLabel("实例 0")
        sb.addWidget(self._status_bg)
        sb.addWidget(self._status_hint, 1)
        sb.addPermanentWidget(self._status_count)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Delete"), self, activated=self.delete_selected)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.duplicate_selected)
        # Arrow keys nudge the selection; PageUp/PageDown change background.
        for dx, dy, keys in [(-1, 0, "Left"), (1, 0, "Right"),
                             (0, -1, "Up"), (0, 1, "Down")]:
            QShortcut(QKeySequence(keys), self,
                      activated=lambda dx=dx, dy=dy: self.nudge(dx, dy))
        QShortcut(QKeySequence("PgUp"), self, activated=lambda: self._step_background(-1))
        QShortcut(QKeySequence("PgDown"), self, activated=lambda: self._step_background(+1))
        QShortcut(QKeySequence("F"), self, activated=self._on_flip)

    def show_about(self) -> None:
        import scenepaste
        QMessageBox.about(
            self,
            "关于 ScenePaste",
            (
                f"<b>ScenePaste</b> {scenepaste.__version__}<br><br>"
                "Scene-first controllable Copy-Paste dataset studio.<br>"
                "MIT License — "
                "<a href='https://github.com/embed9669-alt/ScenePaste'>"
                "github.com/embed9669-alt/ScenePaste</a><br><br>"
                "GUI: PySide6 · Engine: <code>scenepaste.core</code>"
            ),
        )

    def _after_doc_change(self) -> None:
        bg = self.doc.current_background()
        self._status_bg.setText(
            f"背景 {self.doc.background_index + 1}/{len(self.doc.background_paths)}: "
            f"{bg.name if bg else '无'}"
        )
        self._status_count.setText(f"实例 {len(self.doc.instances)}")
        self.controls.reflect(self.doc)
        has_data = bool(self.doc.cutouts) and bool(self.doc.background_paths)
        self._empty_banner.setVisible(not has_data)

    def _on_generation_defaults_changed(self) -> None:
        self.gen_defaults.apply_to_document(self.doc)

    def _apply_default_appearance_to_instances(self) -> None:
        """Copy the main-window default object recipe onto all placed instances."""
        import random
        self.gen_defaults.apply_to_document(self.doc)
        recipe = str(self.doc.object_appearance_recipe or "").strip()
        if not recipe or recipe.lower() == "off":
            QMessageBox.information(
                self, "目标外观",
                "请先在「批量生成默认」里选择一个目标外观 Recipe（例如 mild）。",
            )
            return
        if not self.doc.instances:
            QMessageBox.information(self, "目标外观", "当前画布没有实例。")
            return
        before = self.doc.snapshot()
        for inst in self.doc.instances:
            inst.appearance_enabled = True
            inst.appearance_recipe = recipe
            if int(inst.appearance_seed or 0) == 0:
                inst.appearance_seed = int(random.randint(1, 2**31 - 1))
            inst.invalidate_cache()
        self._push_transform(before)
        self.doc._emit()
        self.statusBar().showMessage(f"已将外观 Recipe「{recipe}」应用到全部实例", 4000)

    # ------------------------------------------------------------ undo stack
    @property
    def _undo_stack(self):
        # Lazily create on first use; can't be a property attribute since
        # QUndoStack needs a parent.
        if not hasattr(self, "_undo_stack_"):
            self._undo_stack_ = QUndoStack(self)
        return self._undo_stack_

    def undo(self) -> None:
        if self._undo_stack.canUndo():
            self._undo_stack.undo()

    def redo(self) -> None:
        if self._undo_stack.canRedo():
            self._undo_stack.redo()

    # --------------------------------------------------------------- loading
    def _pick_objects_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目标素材目录")
        if path:
            self.load_objects(Path(path))

    def _pick_backgrounds_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择背景图目录")
        if path:
            self.load_backgrounds(Path(path))

    def load_objects(self, path: Path) -> None:
        self._objects_dir = path
        self.statusBar().showMessage(f"加载目标中: {path}", 3000)
        self._worker = LoadCutoutsWorker(path, self.doc.class_map_text, self)
        self._worker.cutouts_ready.connect(self._on_cutouts_loaded)
        self._worker.failed.connect(self._on_load_failed)
        self._worker.start()

    def _on_cutouts_loaded(self, cutouts, class_map_text: str = "") -> None:
        if class_map_text and class_map_text != self.doc.class_map_text:
            self.doc.class_map_text = class_map_text
        self.doc.set_cutouts(cutouts)
        labels = sorted({c.label for c in cutouts})
        extra = f"（类别: {', '.join(labels)}）" if labels else ""
        self.statusBar().showMessage(f"已加载 {len(cutouts)} 个目标{extra}", 4000)

    def _on_load_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "加载失败", msg)

    def load_backgrounds(self, path: Path) -> None:
        self._backgrounds_dir = path
        paths = core.list_backgrounds(path)
        if not paths:
            QMessageBox.warning(self, "无背景", f"目录里没有图片：{path}")
            return
        self.doc.set_backgrounds(paths, index=0)

    # ---------------------------------------------------------- one-click demo
    # ---------------------------------------------------------- settings dialog
    def open_project_settings(self) -> None:
        """Open the project settings dialog and apply changes on accept."""
        from .panels import ProjectSettingsDialog
        old_map = self.doc.class_map_text
        dlg = ProjectSettingsDialog(self.doc, self)
        if not dlg.apply_to_document():
            return
        self.gen_defaults.reflect(self.doc)
        self.statusBar().showMessage("项目设置已更新", 3000)
        # Class-map edits only take effect after reloading objects.
        if (
            self.doc.class_map_text != old_map
            and self._objects_dir is not None
            and Path(self._objects_dir).is_dir()
        ):
            reply = QMessageBox.question(
                self, "重新加载目标？",
                "类别映射已更改。是否按新映射重新加载目标素材？\n"
                "（当前画布上的实例会失效，建议先保存。）",
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.load_objects(Path(self._objects_dir))

    # ------------------------------------------------------------ random fill
    def random_fill(self) -> None:
        """Clear the canvas and randomly place 1-3 grounded instances.

        Each instance is "foot-grounded": cy is chosen so its bottom edge
        lands in the bottom half of the background. IoU > 0.2 against
        existing placements is rejected. Pushed as one undo entry.
        """
        import random as _random
        from .undo import _SnapshotCommand
        from compose_app.rendering import render_instance

        if not self.doc.cutouts:
            QMessageBox.information(self, "随机铺满", "请先加载目标素材")
            return
        if self.doc.bg_size == (0, 0):
            QMessageBox.information(self, "随机铺满", "当前背景未就绪")
            return

        bw, bh = self.doc.bg_size
        rng = _random.Random()
        n = rng.randint(1, 3)
        before = self.doc.snapshot()
        new_instances = []
        placed = []
        attempts = 0
        while len(new_instances) < n and attempts < 60:
            attempts += 1
            cut_idx = rng.randrange(len(self.doc.cutouts))
            cutout = self.doc.cutouts[cut_idx]
            cx = rng.uniform(0.1, 0.9) * bw
            h_ratio = rng.uniform(0.10, 0.30)
            h = max(4, int(h_ratio * bh))
            rendered = render_instance(cutout.rgba, h, False, 0.0)
            bottom_y = rng.uniform(0.55, 0.92) * bh
            cy = bottom_y - rendered.height / 2.0
            x1 = int(cx - rendered.width / 2.0)
            y1 = int(cy - rendered.height / 2.0)
            box = (x1, y1, x1 + rendered.width, y1 + rendered.height)
            if any(core.box_iou(box, p) > 0.2 for p in placed):
                continue
            placed.append(box)
            new_instances.append(Instance(
                cutout_index=cut_idx,
                cx=float(cx), cy=float(cy),
                h_ratio=float(h_ratio),
                flip=rng.random() < 0.5,
                angle=0.0,
                uid=self.doc.next_uid(),
            ))
        if not new_instances:
            self.statusBar().showMessage("随机铺满：未找到合适位置", 3000)
            return
        self._undo_stack.push(_SnapshotCommand(self.doc, before, new_instances,
                                               "random fill"))
        self.doc.select(new_instances[-1].uid)
        self.statusBar().showMessage(f"随机铺满：放置 {len(new_instances)} 个目标", 3000)

    # ---------------------------------------------------------- one-click demo
    def load_sample_dataset(self) -> None:
        """Load the bundled samples/ — zero-config way to try the tool."""
        samples_root = self._find_samples_root()
        if samples_root is None:
            QMessageBox.warning(
                self, "找不到示例",
                "当前安装缺少内置示例资源。请重新安装官方 ScenePaste wheel/source package。",
            )
            return
        self.doc.class_map_text = "person=0,motorcycle=1,truck=2"
        self.load_objects(samples_root / "objects")
        self.load_backgrounds(samples_root / "backgrounds")
        self.statusBar().showMessage("已加载示例数据：3 个目标 / 4 张背景", 4000)

    @staticmethod
    def _find_samples_root() -> Optional[Path]:
        """Return bundled package samples, with source-tree fallback."""
        from scenepaste.sample_data import find_samples_root
        return find_samples_root()

    def _step_background(self, delta: int) -> None:
        if not self.doc.background_paths:
            return
        if self.doc.auto_save and self.doc.instances and self.doc.output_dir is not None:
            try:
                self.save_current()
            except Exception as exc:
                self.statusBar().showMessage(f"自动保存失败：{exc}", 5000)

        old_w, old_h = self.doc.bg_size
        snaps = None
        if self.doc.keep_position and self.doc.instances and old_w > 0 and old_h > 0:
            from .workers import _InstanceSnap
            snaps = [_InstanceSnap.from_instance(i, old_w, old_h) for i in self.doc.instances]

        # Background switch is a new editing context; old undo entries refer to
        # a different scene (and may use absolute coords / stale UIDs).
        self._undo_stack.clear()
        self.canvas.reset_gestures()

        new_index = (self.doc.background_index + delta) % len(self.doc.background_paths)
        self.doc.background_index = new_index
        self.doc.instances = []
        self.doc.selected_uid = None
        self.doc._emit()  # canvas reloads bg and refreshes bg_size

        if snaps:
            new_w, new_h = self.doc.bg_size
            if new_w > 0 and new_h > 0:
                restored = [
                    s.to_instance(new_w, new_h, uid=s.uid) for s in snaps
                ]
                self.doc.replace_instances(restored)

    # ----------------------------------------------------------- placement
    def _place_at_centre(self, cutout_index: int) -> None:
        cx, cy = self.doc.bg_size[0] / 2.0, self.doc.bg_size[1] / 2.0
        self._add_instance(cutout_index, cx, cy)

    def _add_instance(self, cutout_index: int, cx: float, cy: float) -> None:
        if not (0 <= cutout_index < len(self.doc.cutouts)):
            return
        cutout = self.doc.cutouts[cutout_index]
        bg_h = self.doc.bg_size[1] or cutout.rgba.height
        inst = Instance(
            cutout_index=cutout_index,
            cx=cx, cy=cy,
            h_ratio=max(0.05, min(0.5, cutout.rgba.height / max(1, bg_h) * 0.5)),
            uid=self.doc.next_uid(),
        )
        self._undo_stack.push(AddInstanceCommand(self.doc, inst))

    # ------------------------------------------------------- transformations
    def _on_scale_changed(self, value: float) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        if abs(float(inst.h_ratio) - float(value)) < 1e-6:
            return
        before = self.doc.snapshot()
        inst.h_ratio = float(value)
        inst.invalidate_cache()
        self._push_transform(before)

    def _on_angle_changed(self, value: float) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        angle = float(value) % 360.0
        if abs(float(inst.angle) % 360.0 - angle) < 1e-6:
            return
        before = self.doc.snapshot()
        inst.angle = angle
        inst.invalidate_cache()
        self._push_transform(before)

    def _on_canvas_transform_committed(self, before) -> None:
        """Undo entry for wheel-scale / Shift-drag rotate / move gestures."""
        if before is None:
            return
        self._push_transform(before)

    def _on_flip(self) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        before = self.doc.snapshot()
        inst.flip = not inst.flip
        inst.invalidate_cache()
        self._push_transform(before)

    def _apply_appearance_from_controls(self) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        vals = self.controls.appearance_values()
        inst.appearance_enabled = bool(vals["enabled"])
        inst.appearance_recipe = str(vals["recipe"] or "mild")
        inst.appearance_brightness = float(vals["brightness"])
        inst.appearance_contrast = float(vals["contrast"])
        inst.appearance_saturation = float(vals["saturation"])
        inst.appearance_blur = float(vals["blur"])
        inst.invalidate_cache()

    def _on_appearance_preview(self) -> None:
        """Live RGB preview while dragging sliders (undo committed on release)."""
        if self.doc.selected() is None:
            return
        if self._appearance_before is None:
            self._appearance_before = self.doc.snapshot()
        self._apply_appearance_from_controls()
        self.doc._emit()

    def _on_appearance_committed(self) -> None:
        if self.doc.selected() is None:
            self._appearance_before = None
            return
        if self._appearance_before is None:
            self._appearance_before = self.doc.snapshot()
            self._apply_appearance_from_controls()
            self.doc._emit()
        self._push_transform(self._appearance_before)
        self._appearance_before = None

    def _on_appearance_resample(self) -> None:
        import random
        inst = self.doc.selected()
        if inst is None:
            return
        before = self.doc.snapshot()
        inst.appearance_enabled = True
        inst.appearance_seed = int(random.randint(0, 2**31 - 1))
        if not str(inst.appearance_recipe or "").strip():
            inst.appearance_recipe = "mild"
        inst.invalidate_cache()
        self._push_transform(before)
        self.controls.reflect(self.doc)
        self.doc._emit()

    def _push_transform(self, before) -> None:
        """Push one undo entry for a completed transform; skip no-ops."""
        after = self.doc.snapshot()
        if _instance_snapshots_equal(before, after):
            return
        self._undo_stack.push(TransformCommand(self.doc, before))

    def delete_selected(self) -> None:
        if self.doc.selected_uid is None:
            return
        self._undo_stack.push(DeleteSelectedCommand(self.doc))

    def duplicate_selected(self) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        clone = inst.clone()
        clone.cx += 16
        clone.cy += 16
        clone.uid = self.doc.next_uid()
        self._undo_stack.push(AddInstanceCommand(self.doc, clone))

    def nudge(self, dx: int, dy: int) -> None:
        inst = self.doc.selected()
        if inst is None:
            return
        before = self.doc.snapshot()
        inst.cx += dx
        inst.cy += dy
        self._push_transform(before)

    # ----------------------------------------------------------------- save
    def save_current(self) -> None:
        if self.doc.output_dir is None:
            picked = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if not picked:
                return
            self.doc.output_dir = Path(picked)
        out_dir = self.doc.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        bg_path = self.doc.current_background()
        if bg_path is None:
            QMessageBox.warning(self, "保存失败", "没有加载任何背景图")
            return
        try:
            composite = composite_from_doc(self.doc)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return

        from compose_app.io_utils import save_composite
        from compose_app.rendering import bbox_of_rendered, render_instance

        stem = _dt.datetime.now().strftime("manual_%Y%m%d_%H%M%S_%f")
        bw, bh = composite.size
        native_boxes = []
        for inst in self.doc.instances:
            if not (0 <= inst.cutout_index < len(self.doc.cutouts)):
                continue
            cutout = self.doc.cutouts[inst.cutout_index]
            target_h = max(4, int(round(inst.h_ratio * bh)))
            rendered = render_instance(cutout.rgba, target_h, inst.flip, inst.angle)
            x1 = int(round(inst.cx - rendered.width / 2.0))
            y1 = int(round(inst.cy - rendered.height / 2.0))
            bx1, by1, bx2, by2 = bbox_of_rendered(rendered)
            native_boxes.append(
                (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2, cutout.label, cutout.class_id)
            )

        bg_image = Image.new("RGB", (bw, bh))
        try:
            image_path, _label_path, _boxes = save_composite(
                bg_image,
                Path(bg_path),
                self.doc.instances,
                self.doc.cutouts,
                out_dir,
                stem,
                composite_and_boxes=(composite, native_boxes),
                output_format=self.doc.output_format,
            )
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return

        # Preview with tight boxes.
        from PIL import ImageDraw
        preview = composite.convert("RGB").copy()
        draw = ImageDraw.Draw(preview)
        for bx1, by1, bx2, by2, _label, cls_id in native_boxes:
            color = (40, 220, 40) if cls_id == 0 else (40, 160, 255)
            draw.rectangle([bx1, by1, max(bx1, bx2 - 1), max(by1, by2 - 1)],
                           outline=color, width=2)
        preview_path = out_dir / "previews" / f"{stem}.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(str(preview_path), quality=92)
        self.statusBar().showMessage(f"已保存 {image_path.name}", 4000)

    # ------------------------------------------------- batch apply_to_next
    def apply_to_next(self) -> None:
        """Apply the current layout to all (or N) remaining backgrounds.

        Snapshots instance positions as relative coords so the layout
        survives background size differences. A modal progress dialog shows
        per-image progress and supports mid-run cancellation.
        """
        if not self.doc.background_paths:
            QMessageBox.information(self, "批量套用", "请先加载背景")
            return
        if not self.doc.instances:
            QMessageBox.information(self, "批量套用", "当前画布为空，请先摆放目标")
            return
        if self.doc.bg_size == (0, 0):
            QMessageBox.information(self, "批量套用", "当前背景未就绪")
            return

        remaining = len(self.doc.background_paths) - self.doc.background_index - 1
        if remaining <= 0:
            QMessageBox.information(self, "批量套用", "已是最后一张背景")
            return

        # Confirm output dir.
        if self.doc.output_dir is None:
            picked = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if not picked:
                return
            self.doc.output_dir = Path(picked)

        # Confirm scope (5 / 10 / 15 / 20 / all remaining).
        from PySide6.QtWidgets import QInputDialog
        choices = []
        choice_counts = []
        for n in (5, 10, 15, 20):
            if n < remaining:
                choices.append(f"后续 {n} 张")
                choice_counts.append(n)
        choices.append(f"全部剩余 {remaining} 张")
        choice_counts.append(remaining)
        picked, ok = QInputDialog.getItem(
            self, "批量套用",
            f"将当前布局套用到后续背景并保存。\n输出目录：{self.doc.output_dir}\n\n套用范围:",
            choices, len(choices) - 1, False,
        )
        if not ok or not picked:
            return
        count = choice_counts[choices.index(picked)]

        self.gen_defaults.apply_to_document(self.doc)
        bg_w, bg_h = self.doc.bg_size
        from .workers import BatchApplyWorker, _InstanceSnap
        import random as _random
        snaps = []
        default_recipe = str(self.doc.object_appearance_recipe or "").strip()
        for inst in self.doc.instances:
            snap = _InstanceSnap.from_instance(inst, bg_w, bg_h)
            if (
                default_recipe
                and default_recipe.lower() != "off"
                and not snap.appearance_enabled
            ):
                snap.appearance_enabled = True
                snap.appearance_recipe = default_recipe
                if int(snap.appearance_seed or 0) == 0:
                    snap.appearance_seed = int(_random.randint(1, 2**31 - 1))
            snaps.append(snap)
        targets = self.doc.background_paths[
            self.doc.background_index + 1: self.doc.background_index + 1 + count
        ]

        self._batch_worker = BatchApplyWorker(
            cutouts=self.doc.cutouts,
            instance_snaps=snaps,
            background_paths=targets,
            output_dir=self.doc.output_dir,
            class_map_text=self.doc.class_map_text,
            output_format=self.doc.output_format,
            do_shadow=self.doc.do_shadow,
            do_color_match=self.doc.do_color_match,
            scene_recipe=self.doc.scene_recipe,
        )
        from .panels import BatchProgressDialog
        self._batch_dialog = BatchProgressDialog(total=len(targets),
                                                  scope_text=f"套用到剩余 {len(targets)} 张背景")
        self._batch_dialog.set_worker(self._batch_worker)
        self._batch_worker.progress.connect(self._batch_dialog.on_progress)
        self._batch_worker.finished.connect(self._batch_dialog.on_finished)
        self._batch_worker.failed.connect(self._batch_dialog.on_failed)
        self._batch_worker.start()
        self._batch_dialog.exec()

    # ----------------------------------------------------- scene templates
    def save_scene_template(self) -> None:
        """Save the current layout as a resolution-portable JSON template."""
        from . import templates
        path, _ = QFileDialog.getSaveFileName(
            self, "保存场景模板", "scene_template.json",
            "Scene template (*.json);;All files (*)")
        if not path:
            return
        try:
            from .panels import TemplateParametersDialog
            param_dialog = TemplateParametersDialog(self)
            if param_dialog.exec() != param_dialog.Accepted:
                return
            count = templates.save_template(self.doc, Path(path), parameters=param_dialog.parameters())
        except RuntimeError as exc:
            QMessageBox.warning(self, "无法保存模板", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._active_scene_template = Path(path)
        self.statusBar().showMessage(f"场景模板已保存（{count} 个目标）", 4000)

    def load_scene_template(self) -> None:
        """Load a template and resolve its cutouts against the current library."""
        from . import templates
        path, _ = QFileDialog.getOpenFileName(
            self, "加载场景模板", "", "Scene template (*.json);;All files (*)")
        if not path:
            return
        try:
            restored, missing = templates.load_template(self.doc, Path(path))
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "无法加载模板", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "模板无效", str(exc))
            return
        # Replace the current layout (not append) — matches "load template" UX.
        from .undo import _SnapshotCommand
        before = self.doc.snapshot()
        self._undo_stack.push(_SnapshotCommand(self.doc, before, restored, "load template"))
        msg = f"已加载模板：{len(restored)} 个目标"
        if missing:
            msg += f"，{len(missing)} 个素材未匹配"
            QMessageBox.warning(self, "部分素材未匹配",
                                msg + "\n\n" + "\n".join(missing[:8]))
        self._active_scene_template = Path(path)
        self.statusBar().showMessage(msg, 5000)
