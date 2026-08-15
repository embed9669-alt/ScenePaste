"""ScenePaste Qt main window.

Toolbar + cutout gallery + canvas + instance controls + status bar.
Document model is the source of truth; undo goes through QUndoStack.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import scenepaste.core as core
from PIL import Image
from PySide6.QtCore import Qt, QSize, QSettings, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut, QUndoStack
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from compose_app.models import Instance

from .canvas import CanvasView
from .controllers import MainWindowController, _instance_snapshots_equal  # noqa: F401 (re-exported for back-compat)
from .panels import ControlsPanel, CutoutThumbWidget, GenerationDefaultsPanel, InstanceListWidget
from .state import Document
from .theme import qss_for
from .workers import LoadCutoutsWorker, composite_from_doc


class MainWindow(QMainWindow):
    """Assembles all panels and routes signals to controllers / undo stack."""

    def __init__(self, objects_dir: Optional[Path] = None,
                 backgrounds_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 class_map_text: str = "person=0,vehicle=1",
                 theme_mode: Optional[str] = None):
        super().__init__()
        self._settings = QSettings("ScenePaste", "ScenePaste")
        persisted_theme = str(self._settings.value("ui/theme", "dark") or "dark")
        theme_mode = theme_mode or persisted_theme
        self.setWindowTitle("ScenePaste — Controllable Vision Data Factory")
        self.resize(1280, 820)

        self.doc = Document(
            output_dir=output_dir,
            class_map_text=class_map_text,
        )
        self._objects_dir = objects_dir
        self._backgrounds_dir = backgrounds_dir
        self._theme_mode = theme_mode
        self._worker: Optional[LoadCutoutsWorker] = None
        self._load_progress: Optional[QProgressDialog] = None

        self.setStyleSheet(qss_for(theme_mode))

        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        # Controller owns undo-aware mutations (transform / appearance / layer ops).
        self.controller = MainWindowController(self)
        self._bind_shortcuts()
        self._wire_controller()
        self._restore_ui_state()
        self._refresh_recent_projects()

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

        # Workspace splitter. A dedicated welcome home is shown before any
        # project/assets are loaded, so first-time users immediately understand
        # the main workflow instead of staring at an empty canvas.
        splitter = QSplitter(Qt.Horizontal)
        self._main_splitter = splitter
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
        # Tabs own their inner padding; keep the chrome tight against the splitter.
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(0)

        # Right panel is now a 3-tab widget: 实例 / 变换·外观 / 批量默认.
        self._right_tabs = QTabWidget()
        self._right_tabs.setTabPosition(QTabWidget.North)
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setElideMode(Qt.ElideNone)

        # Tab 1: 实例 (layer list + layer ops + per-row visibility/lock).
        self.instance_list = InstanceListWidget(self.doc)
        self._right_tabs.addTab(self.instance_list, "实例")

        # Tab 2: 变换·外观.
        self.controls = ControlsPanel()
        self._right_tabs.addTab(self.controls, "变换·外观")

        # Tab 3: batch Copy-Paste defaults (does not start a run).
        self.gen_defaults = GenerationDefaultsPanel()
        self.gen_defaults.reflect(self.doc)
        self._right_tabs.addTab(self.gen_defaults, "批量默认")

        right_layout.addWidget(self._right_tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)
        splitter.setSizes([210, 740, 320])

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.addWidget(splitter, 1)
        self._workspace = workspace

        self._welcome_home = self._build_welcome_home()
        # Backward-compatible alias used by older GUI tests/extensions.
        self._empty_banner = self._welcome_home
        self._home_stack = QStackedWidget()
        self._home_stack.addWidget(self._welcome_home)
        self._home_stack.addWidget(workspace)
        self._home_stack.setCurrentWidget(self._welcome_home)
        outer.addWidget(self._home_stack, 1)

        self.setCentralWidget(central)

        # Subscribe to doc changes for side-effects.
        self.doc.subscribe(self._after_doc_change)

    def _build_welcome_home(self) -> QWidget:
        """Friendly first-run home for open-source users."""
        page = QWidget()
        page.setObjectName("welcomePage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(48, 36, 48, 36)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("welcomeCard")
        card.setMaximumWidth(860)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 30, 36, 30)
        card_layout.setSpacing(14)

        brand = QLabel("ScenePaste")
        brand.setObjectName("welcomeTitle")
        brand.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Controllable Vision Data Factory")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        intro = QLabel("从真实数据出发：分割/人工修正 → 前景与干净背景资产库 → 可控合成 → 自动标签 → QA。")
        intro.setObjectName("welcomeDescription")
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(brand)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(intro)

        self._banner_btn_asset = QPushButton("打开素材工作室")
        self._banner_btn_asset.setObjectName("primaryButton")
        self._banner_btn_asset.setMinimumHeight(40)
        card_layout.addWidget(self._banner_btn_asset)

        secondary = QHBoxLayout()
        self._banner_btn_factory = QPushButton("开始批量数据合成")
        self._banner_btn_open = QPushButton("打开已有工程…")
        self._banner_btn_objects = QPushButton("加载目标素材…")
        secondary.addWidget(self._banner_btn_factory)
        secondary.addWidget(self._banner_btn_open)
        secondary.addWidget(self._banner_btn_objects)
        card_layout.addLayout(secondary)

        steps = QLabel("<b>推荐流程</b>　① 自动/已有分割　→　② 人工修 Mask　→　③ 保存前景 + 干净背景　→　④ 可控 Copy-Paste 合成")
        steps.setObjectName("welcomeSteps")
        steps.setWordWrap(True)
        steps.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(steps)

        recent_title = QLabel("最近工程")
        recent_title.setObjectName("sectionHeadline")
        card_layout.addWidget(recent_title)
        self._recent_projects_box = QWidget()
        self._recent_projects_layout = QVBoxLayout(self._recent_projects_box)
        self._recent_projects_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_projects_layout.setSpacing(5)
        card_layout.addWidget(self._recent_projects_box)

        bottom = QHBoxLayout()
        self._banner_btn_sample = QPushButton("加载示例数据")
        self._banner_btn_docs = QPushButton("查看快速开始")
        bottom.addStretch(1)
        bottom.addWidget(self._banner_btn_sample)
        bottom.addWidget(self._banner_btn_docs)
        bottom.addStretch(1)
        card_layout.addLayout(bottom)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)
        return page

    def _build_empty_banner(self) -> QWidget:
        """Compatibility wrapper retained for third-party extensions."""
        return self._build_welcome_home()

    def _wire_controller(self) -> None:
        """Connect panel signals to controller methods (called after both built)."""
        # Per-instance transform / appearance.
        self.controls.scale_changed.connect(self.controller.on_scale_changed)
        self.controls.angle_changed.connect(self.controller.on_angle_changed)
        self.controls.flip_toggled.connect(self.controller.on_flip)
        self.controls.delete_clicked.connect(self.controller.delete_selected)
        self.controls.appearance_preview.connect(self.controller.on_appearance_preview)
        self.controls.appearance_committed.connect(self.controller.on_appearance_committed)
        self.controls.appearance_resample.connect(self.controller.on_appearance_resample)
        # Live wheel/Shift-drag must refresh spinboxes without a full doc emit.
        self.canvas.transform_preview.connect(lambda: self.controls.reflect(self.doc))

        # Instance-list panel → controller (layer ops + visibility + lock + reorder).
        self.instance_list.selection_clicked.connect(self.doc.select)
        self.instance_list.layer_op_requested.connect(
            lambda op, uid: {
                "bring_front": self.controller.bring_to_front,
                "send_back": self.controller.send_to_back,
                "forward": self.controller.move_layer_up,
                "backward": self.controller.move_layer_down,
            }[op](uid)
        )
        self.instance_list.visibility_toggled.connect(self.controller.set_visibility)
        self.instance_list.lock_toggled.connect(self.controller.set_locked)
        self.instance_list.reorder_requested.connect(self.controller.reorder_to_uids)

        # Generation-defaults panel.
        self.gen_defaults.changed.connect(self._on_generation_defaults_changed)
        self.gen_defaults.apply_to_instances.connect(self._apply_default_appearance_to_instances)

        # Empty-state banner buttons share the same QActions used by menus/toolbar.
        self._banner_btn_sample.clicked.connect(self.act_load_sample.trigger)
        self._banner_btn_open.clicked.connect(self.act_open_project.trigger)
        self._banner_btn_objects.clicked.connect(self.act_load_objects.trigger)
        self._banner_btn_asset.clicked.connect(self.act_asset_studio.trigger)
        self._banner_btn_factory.clicked.connect(self.act_large_generate.trigger)
        self._banner_btn_docs.clicked.connect(self.show_quick_start)

    def _build_actions(self) -> None:
        """Create all QActions once; menus and toolbar share them."""
        self.act_load_sample = QAction("加载示例", self)
        self.act_load_sample.setToolTip("加载 samples/ 数据，快速上手")
        self.act_load_sample.triggered.connect(self.load_sample_dataset)

        self.act_open_project = QAction("打开工程…", self)
        self.act_open_project.setToolTip("打开 scenepaste.project.json")
        self.act_open_project.triggered.connect(self.open_project_manifest)

        self.act_save_project = QAction("保存工程", self)
        self.act_save_project.setToolTip("保存路径/类别/默认配置到项目 Manifest")
        self.act_save_project.triggered.connect(self.save_project_manifest)

        self.act_load_objects = QAction("加载目标…", self)
        self.act_load_objects.setToolTip("加载 LabelMe / X-AnyLabeling 标注的目标素材")
        self.act_load_objects.triggered.connect(self._pick_objects_dir)

        self.act_asset_studio = QAction("素材工作室…", self)
        self.act_asset_studio.setToolTip(
            "人工编辑分割 Mask 与背景移除 Mask，导出透明前景、干净背景和审核记录"
        )
        self.act_asset_studio.triggered.connect(self.open_asset_studio)

        self.act_cutout_studio = QAction("快速抠图…", self)
        self.act_cutout_studio.setToolTip(
            "手绘 / 点选 / 自动得到初始分割；需要精修 Mask 与干净背景时请用「素材工作室」"
        )
        self.act_cutout_studio.triggered.connect(self.open_cutout_studio)

        self.act_auto_cutout = QAction("整目录 rembg 加载…", self)
        self.act_auto_cutout.setToolTip(
            "不打开抠图工作室，直接用 rembg 批量加载普通原图目录"
            "（需 pip install 'scenepaste[auto]'；要交互精修请用「快速抠图」）"
        )
        self.act_auto_cutout.triggered.connect(self._pick_auto_cutout_dir)

        self.act_load_bgs = QAction("加载背景…", self)
        self.act_load_bgs.triggered.connect(self._pick_backgrounds_dir)

        self.act_prev_bg = QAction("上一张背景", self)
        self.act_prev_bg.triggered.connect(lambda: self._step_background(-1))
        self.act_next_bg = QAction("下一张背景", self)
        self.act_next_bg.triggered.connect(lambda: self._step_background(+1))

        self.act_save = QAction("保存当前", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.triggered.connect(self.save_current)

        self.act_batch = QAction("批量套用", self)
        self.act_batch.setToolTip("把当前布局套用到剩余背景并保存")
        self.act_batch.triggered.connect(self.apply_to_next)

        self.act_large_generate = QAction("批量数据工厂…", self)
        self.act_large_generate.setToolTip("以可控 Copy-Paste 批量合成训练集：自动标签与困难样本策略")
        self.act_large_generate.triggered.connect(self.open_large_generation)

        self.act_data_loop = QAction("数据闭环…", self)
        self.act_data_loop.setToolTip("Hard Mining / QA / 泄漏 / 对比 / 策展 / Sharding")
        self.act_data_loop.triggered.connect(self.open_data_loop_center)

        self.act_dataset_tools = QAction("数据集工具…", self)
        self.act_dataset_tools.setToolTip("analyze / split / merge / recipe / project")
        self.act_dataset_tools.triggered.connect(self.open_dataset_tools)

        self.act_settings = QAction("项目设置…", self)
        self.act_settings.triggered.connect(self.open_project_settings)

        self.act_explorer = QAction("数据集浏览…", self)
        self.act_explorer.setToolTip("浏览已生成数据并叠加标注")
        self.act_explorer.triggered.connect(self.open_dataset_explorer)

        self.act_random = QAction("随机铺满", self)
        self.act_random.setToolTip("随机铺满当前背景（替换现有实例）")
        self.act_random.triggered.connect(self.random_fill)

        self.act_save_tpl = QAction("存模板…", self)
        self.act_save_tpl.triggered.connect(self.save_scene_template)
        self.act_load_tpl = QAction("读模板…", self)
        self.act_load_tpl.triggered.connect(self.load_scene_template)

        self.act_undo = QAction("撤销", self)
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_undo.triggered.connect(self.undo)
        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcut("Ctrl+Y")
        self.act_redo.triggered.connect(self.redo)

        self.act_theme = QAction("浅色主题", self)
        self.act_theme.setCheckable(True)
        self.act_theme.setChecked(self._theme_mode == "light")
        self.act_theme.setToolTip("切换深色 / 浅色界面")
        self.act_theme.toggled.connect(self._toggle_theme)

        self.act_toggle_left = QAction("显示目标素材栏", self)
        self.act_toggle_left.setCheckable(True); self.act_toggle_left.setChecked(True)
        self.act_toggle_left.toggled.connect(lambda checked: self.thumbs.setVisible(checked))
        self.act_toggle_right = QAction("显示属性栏", self)
        self.act_toggle_right.setCheckable(True); self.act_toggle_right.setChecked(True)
        self.act_toggle_right.toggled.connect(lambda checked: self._right_tabs.parentWidget().setVisible(checked))
        self.act_reset_layout = QAction("恢复默认布局", self)
        self.act_reset_layout.triggered.connect(self._reset_layout)

        self.act_quick_start = QAction("快速开始", self)
        self.act_quick_start.triggered.connect(self.show_quick_start)
        self.act_docs = QAction("使用文档", self)
        self.act_docs.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/embed9669-alt/ScenePaste#readme")))
        self.act_asset_docs = QAction("素材工作室说明", self)
        self.act_asset_docs.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/embed9669-alt/ScenePaste/blob/main/docs/ASSET_STUDIO.md")))
        self.act_shortcuts = QAction("快捷键", self)
        self.act_shortcuts.triggered.connect(self.show_shortcuts)
        self.act_github = QAction("GitHub 项目主页", self)
        self.act_github.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/embed9669-alt/ScenePaste")))
        self.act_issue = QAction("报告问题…", self)
        self.act_issue.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/embed9669-alt/ScenePaste/issues")))
        self.act_about = QAction("关于 ScenePaste", self)
        self.act_about.triggered.connect(self.show_about)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        file_m = mb.addMenu("文件")
        file_m.addAction(self.act_open_project)
        file_m.addAction(self.act_save_project)
        file_m.addAction(self.act_save)
        file_m.addSeparator()
        file_m.addAction(self.act_settings)
        file_m.addSeparator()
        file_m.addAction("退出", self.close)

        assets_m = mb.addMenu("素材")
        assets_m.addAction(self.act_load_objects)
        assets_m.addAction(self.act_load_bgs)
        assets_m.addSeparator()
        assets_m.addAction(self.act_asset_studio)
        assets_m.addSeparator()
        assets_m.addAction(self.act_cutout_studio)
        assets_m.addAction(self.act_auto_cutout)
        assets_m.addSeparator()
        assets_m.addAction(self.act_load_sample)

        scene_m = mb.addMenu("场景")
        scene_m.addAction(self.act_random)
        scene_m.addSeparator()
        scene_m.addAction(self.act_prev_bg)
        scene_m.addAction(self.act_next_bg)
        scene_m.addSeparator()
        scene_m.addAction(self.act_save_tpl)
        scene_m.addAction(self.act_load_tpl)

        edit_m = mb.addMenu("编辑")
        edit_m.addAction(self.act_undo)
        edit_m.addAction(self.act_redo)

        gen_m = mb.addMenu("合成")
        gen_m.addAction(self.act_large_generate)
        gen_m.addSeparator()
        gen_m.addAction(self.act_batch)

        data_m = mb.addMenu("数据")
        data_m.addAction(self.act_explorer)
        data_m.addAction(self.act_data_loop)
        data_m.addAction(self.act_dataset_tools)

        view_m = mb.addMenu("视图")
        view_m.addAction(self.act_toggle_left)
        view_m.addAction(self.act_toggle_right)
        view_m.addAction(self.act_reset_layout)
        view_m.addSeparator()
        view_m.addAction(self.act_theme)

        help_m = mb.addMenu("帮助")
        help_m.addAction(self.act_quick_start)
        help_m.addAction(self.act_docs)
        help_m.addAction(self.act_asset_docs)
        help_m.addAction(self.act_shortcuts)
        help_m.addSeparator()
        help_m.addAction(self.act_github)
        help_m.addAction(self.act_issue)
        help_m.addSeparator()
        help_m.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        """Focused toolbar: project/assets → Asset Studio / batch factory → edit."""
        tb = QToolBar("main", self)
        self._main_toolbar = tb
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        tb.addAction(self.act_open_project)
        tb.addSeparator()
        tb.addAction(self.act_load_objects)
        tb.addAction(self.act_load_bgs)
        tb.addSeparator()
        asset_btn = QPushButton("素材工作室")
        asset_btn.setObjectName("primaryButton")
        asset_btn.setToolTip(self.act_asset_studio.toolTip())
        asset_btn.clicked.connect(self.act_asset_studio.trigger)
        tb.addWidget(asset_btn)
        factory_btn = QPushButton("批量数据工厂")
        factory_btn.setToolTip(self.act_large_generate.toolTip())
        factory_btn.clicked.connect(self.act_large_generate.trigger)
        tb.addWidget(factory_btn)
        tb.addSeparator()
        tb.addAction(self.act_save)
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)

    def _set_asset_actions_enabled(self, enabled: bool) -> None:
        for act in (
            self.act_load_objects,
            self.act_auto_cutout,
            self.act_asset_studio,
            self.act_cutout_studio,
            self.act_load_bgs,
            self.act_load_sample,
            self.act_batch,
            self.act_large_generate,
        ):
            act.setEnabled(enabled)

    def open_project_manifest(self) -> None:
        start = str(self._settings.value("paths/last_project_dir", str(Path.cwd())) or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "打开 ScenePaste 工程", start, "ScenePaste Project (*.json)")
        if path:
            self._load_project_manifest_path(Path(path))

    def _load_project_manifest_path(self, path: Path) -> None:
        from scenepaste.project import ScenePasteProject
        try:
            project = ScenePasteProject.load(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "无法打开工程", str(exc)); return
        self._project_manifest = Path(path)
        self._settings.setValue("paths/last_project_dir", str(Path(path).parent))
        self._remember_recent_project(Path(path))
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
            self._remember_recent_project(project_path)
            self._settings.setValue("paths/last_project_dir", str(project_path.parent))
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
        self._settings.setValue("ui/theme", self._theme_mode)
        self.setStyleSheet(qss_for(self._theme_mode))
        for studio in (getattr(self, "_cutout_studio", None), getattr(self, "_asset_studio", None)):
            if studio is not None:
                try:
                    from .theme import apply_theme
                    apply_theme(studio, self._theme_mode)
                except RuntimeError:
                    pass

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._status_bg = QLabel("无背景")
        self._status_bg.setObjectName("statusLabel")
        full_hint = (
            "滚轮缩放 · Shift+拖动旋转 · Ctrl+滚轮画布 · [ / ] 缩放 · "
            "方向键微调 · Ctrl+Shift+[ / ] 层级 · Home/End 置顶/置底 · "
            "PgUp/PgDn 换背景 · F 翻转 · Ctrl+D 复制 · Delete 删除"
        )
        self._status_hint = QLabel("滚轮缩放 · Shift+旋转 · [ / ] 缩放 · 方向键微调 · Home/End 层级")
        self._status_hint.setObjectName("statusLabel")
        self._status_hint.setToolTip(full_hint)
        self._status_count = QLabel("实例 0")
        self._status_count.setObjectName("statusLabel")
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
        # Scale shortcuts: [ shrinks, ] enlarges (factor 1.10 per press).
        QShortcut(QKeySequence("["), self,
                  activated=lambda: self.controller.scale_selected(1.0 / self.controller._scale_step))
        QShortcut(QKeySequence("]"), self,
                  activated=lambda: self.controller.scale_selected(self.controller._scale_step))
        # Layer ordering shortcuts.
        QShortcut(QKeySequence("Ctrl+Shift+]"), self,
                  activated=lambda: self.controller.move_layer_up())
        QShortcut(QKeySequence("Ctrl+Shift+["), self,
                  activated=lambda: self.controller.move_layer_down())
        QShortcut(QKeySequence("Home"), self,
                  activated=lambda: self.controller.bring_to_front())
        QShortcut(QKeySequence("End"), self,
                  activated=lambda: self.controller.send_to_back())

    def _restore_ui_state(self) -> None:
        geometry = self._settings.value("ui/geometry")
        if geometry is not None:
            try: self.restoreGeometry(geometry)
            except Exception: pass
        splitter = self._settings.value("ui/main_splitter")
        if splitter is not None:
            try: self._main_splitter.restoreState(splitter)
            except Exception: pass
        self.act_toggle_left.setChecked(self._settings.value("ui/show_left", True, type=bool))
        self.act_toggle_right.setChecked(self._settings.value("ui/show_right", True, type=bool))

    def _reset_layout(self) -> None:
        self.resize(1280, 820)
        self._main_splitter.setSizes([210, 740, 320])
        self.act_toggle_left.setChecked(True)
        self.act_toggle_right.setChecked(True)
        self.statusBar().showMessage("已恢复默认布局", 2500)

    def _recent_projects(self) -> list[str]:
        value = self._settings.value("recent/projects", [])
        if isinstance(value, str):
            value = [value]
        return [str(x) for x in (value or []) if str(x)]

    def _remember_recent_project(self, path: Path) -> None:
        resolved = str(Path(path).expanduser().resolve())
        rows = [x for x in self._recent_projects() if x != resolved and Path(x).exists()]
        rows.insert(0, resolved)
        self._settings.setValue("recent/projects", rows[:5])
        self._refresh_recent_projects()

    def _refresh_recent_projects(self) -> None:
        layout = getattr(self, "_recent_projects_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None: widget.deleteLater()
        rows = [x for x in self._recent_projects() if Path(x).is_file()]
        if not rows:
            label = QLabel("暂无最近工程 · 可以先进入素材工作室整理前景/背景，或加载示例数据")
            label.setObjectName("mutedLabel")
            layout.addWidget(label)
            return
        for path in rows[:5]:
            p = Path(path)
            btn = QPushButton(f"{p.stem}   ·   {p.parent}")
            btn.setObjectName("recentProjectButton")
            btn.setToolTip(path)
            btn.clicked.connect(lambda _checked=False, q=p: self._load_project_manifest_path(q))
            layout.addWidget(btn)

    def show_quick_start(self) -> None:
        QMessageBox.information(
            self, "ScenePaste 快速开始",
            "1. 用已有 LabelMe/分割结果，或先通过“快速抠图”得到初始 Mask\n"
            "2. 打开“素材工作室”，人工修正前景 Mask\n"
            "3. 独立编辑背景移除 Mask，预览并保存干净背景\n"
            "4. 将审核后的透明前景 + 真实/干净背景加入素材库\n"
            "5. 打开“批量数据工厂”，用可控 Copy-Paste 合成训练数据。"
        )

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self, "快捷键",
            "Ctrl+S 保存当前　Ctrl+Z/Y 撤销/重做\n"
            "PgUp/PgDn 切换背景　F 翻转　Ctrl+D 复制　Delete 删除\n"
            "[ / ] 缩放　方向键微调　Home/End 置顶/置底\n"
            "滚轮缩放目标　Shift+拖动旋转　Ctrl+滚轮缩放画布"
        )

    def closeEvent(self, event):  # pragma: no cover - GUI lifecycle
        try:
            self._settings.setValue("ui/geometry", self.saveGeometry())
            self._settings.setValue("ui/main_splitter", self._main_splitter.saveState())
            self._settings.setValue("ui/theme", self._theme_mode)
            self._settings.setValue("ui/show_left", self.act_toggle_left.isChecked())
            self._settings.setValue("ui/show_right", self.act_toggle_right.isChecked())
        finally:
            super().closeEvent(event)

    def show_about(self) -> None:
        import scenepaste
        QMessageBox.about(
            self,
            "关于 ScenePaste",
            (
                f"<b>ScenePaste</b> {scenepaste.__version__}<br><br>"
                "Human-review-first controllable training-data asset and composition toolkit for computer vision.<br>"
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
        has_any_data = bool(self.doc.cutouts) or bool(self.doc.background_paths)
        if hasattr(self, "_home_stack"):
            self._home_stack.setCurrentWidget(self._workspace if has_any_data else self._welcome_home)

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
                "请先在右侧「批量默认」里选择一个目标外观 Recipe（例如 mild）。",
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
        start = str(self._settings.value("paths/objects", str(Path.cwd())) or Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "选择目标素材目录", start)
        if path:
            self.load_objects(Path(path), auto_cutout=False)

    def _pick_auto_cutout_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择普通原图目录（自动抠图 rembg）"
        )
        if not path:
            return
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
            QLabel, QVBoxLayout,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("自动抠图 — 选择类别")
        dlg.resize(420, 180)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "抠图只负责 mask；这里指定写入标注的类别名。\n"
            "也可勾选「按子文件夹名」：objects/person/*.jpg → person。"
        ))
        form = QFormLayout()
        label_box = QComboBox()
        label_box.setEditable(True)
        known = []
        for part in (self.doc.class_map_text or "").replace("，", ",").split(","):
            part = part.strip()
            if "=" in part:
                known.append(part.split("=", 1)[0].strip())
        for name in known + ["person", "truck", "motorcycle", "vehicle", "auto"]:
            if name and label_box.findText(name) < 0:
                label_box.addItem(name)
        label_box.setCurrentText(known[0] if known else "person")
        form.addRow("类别名:", label_box)
        subdir = QCheckBox("按子文件夹名作为类别（可覆盖上面的默认名）")
        form.addRow("", subdir)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        self.load_objects(
            Path(path),
            auto_cutout=True,
            auto_cutout_label=label_box.currentText().strip() or "auto",
            auto_cutout_label_from_subdir=subdir.isChecked(),
        )

    def open_asset_studio(self) -> None:
        from .asset_studio import AssetStudioDialog

        existing = getattr(self, "_asset_studio", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_(); existing.activateWindow(); return
            except RuntimeError:
                pass
        dlg = AssetStudioDialog(
            objects_dir=self._objects_dir,
            backgrounds_dir=self._backgrounds_dir,
            class_map_text=self.doc.class_map_text,
            parent=self,
        )
        self._asset_studio = dlg
        dlg.assets_saved.connect(
            lambda: self.statusBar().showMessage(
                "素材工作室已保存人工审核前景/背景，可重新加载素材库使用。", 6000
            )
        )
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def open_cutout_studio(self) -> None:
        from .cutout_studio import ObjectCutoutStudioDialog

        # Reuse one studio window so downloads keep running while using the main UI.
        existing = getattr(self, "_cutout_studio", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.activateWindow()
                    return
            except RuntimeError:
                self._cutout_studio = None

        dlg = ObjectCutoutStudioDialog(
            objects_dir=Path(self._objects_dir) if self._objects_dir else Path.cwd() / "objects",
            class_map_text=self.doc.class_map_text,
            parent=self,
        )
        dlg.setModal(False)
        dlg.setWindowModality(Qt.NonModal)

        def _on_saved():
            out = Path(dlg.out_dir.text() or "")
            if out.is_dir():
                self.statusBar().showMessage(f"抠图已保存到 {out}（关闭工作室后可重新加载目标）", 5000)

        def _on_finished(_result=None):
            out = Path(dlg.out_dir.text() or "")
            # Reload once when the studio closes if anything was saved.
            if getattr(dlg, "_saved_once", False) and out.is_dir():
                self.load_objects(out, auto_cutout=False)
            self._cutout_studio = None

        def _mark_saved():
            dlg._saved_once = True
            _on_saved()

        dlg.assets_saved.connect(_mark_saved)
        dlg.finished.connect(_on_finished)
        self._cutout_studio = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _pick_backgrounds_dir(self) -> None:
        start = str(self._settings.value("paths/backgrounds", str(Path.cwd())) or Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "选择背景图目录", start)
        if path:
            self.load_backgrounds(Path(path))

    def load_objects(
        self,
        path: Path,
        *,
        auto_cutout: bool = False,
        auto_cutout_label: str | None = None,
        auto_cutout_label_from_subdir: bool = False,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "请稍候", "正在加载目标，请等待当前任务结束。")
            return
        self._objects_dir = path
        self._settings.setValue("paths/objects", str(Path(path).resolve()))
        mode = "自动抠图" if auto_cutout else "目标"
        self.statusBar().showMessage(f"加载{mode}中: {path}", 0)
        self._set_asset_actions_enabled(False)

        self._load_progress = QProgressDialog(
            f"正在加载{mode}…\n首次自动抠图可能需下载/加载 U2Net 模型。",
            None,
            0,
            0,
            self,
        )
        self._load_progress.setWindowTitle("加载中")
        self._load_progress.setWindowModality(Qt.WindowModal)
        self._load_progress.setMinimumDuration(0)
        self._load_progress.setCancelButton(None)
        self._load_progress.show()

        self._worker = LoadCutoutsWorker(
            path, self.doc.class_map_text, self,
            auto_cutout=auto_cutout,
            auto_cutout_label=auto_cutout_label,
            auto_cutout_label_from_subdir=auto_cutout_label_from_subdir,
        )
        self._worker.cutouts_ready.connect(self._on_cutouts_loaded)
        self._worker.failed.connect(self._on_load_failed)
        self._worker.progress.connect(self._on_load_progress)
        self._worker.start()

    def _on_load_progress(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 0)
        if self._load_progress is not None:
            self._load_progress.setLabelText(msg)

    def _finish_load_ui(self) -> None:
        self._set_asset_actions_enabled(True)
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None

    def _on_cutouts_loaded(self, cutouts, class_map_text: str = "") -> None:
        self._finish_load_ui()
        if class_map_text and class_map_text != self.doc.class_map_text:
            self.doc.class_map_text = class_map_text
        self.doc.set_cutouts(cutouts)
        labels = sorted({c.label for c in cutouts})
        extra = f"（类别: {', '.join(labels)}）" if labels else ""
        self.statusBar().showMessage(f"已加载 {len(cutouts)} 个目标{extra}", 4000)

    def _on_load_failed(self, msg: str) -> None:
        self._finish_load_ui()
        tip = msg
        low = msg.lower()
        if "rembg" in low or "自动抠图需要" in msg:
            tip = (
                f"{msg}\n\n"
                "安装：pip install 'scenepaste[auto]'\n"
                "若模型下载失败，请将 u2net.onnx 放到 ~/.u2net/ "
                "（可用 hf-mirror 等镜像）。"
            )
        QMessageBox.warning(self, "加载失败", tip)

    def open_dataset_tools(self) -> None:
        from .dataset_tools import DatasetToolsDialog
        self._dataset_tools = DatasetToolsDialog(
            dataset_root=self.doc.output_dir, parent=self,
        )
        self._dataset_tools.show()
        self._dataset_tools.raise_()

    def load_backgrounds(self, path: Path) -> None:
        self._backgrounds_dir = path
        self._settings.setValue("paths/backgrounds", str(Path(path).resolve()))
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
        self.controller.place_at_centre(cutout_index)

    def _add_instance(self, cutout_index: int, cx: float, cy: float) -> None:
        self.controller.add_instance(cutout_index, cx, cy)

    # ----------------------------------------------- back-compat proxy methods
    # These delegate to MainWindowController so existing tests / signal wiring
    # keep working. New code should call the controller directly.
    def _on_scale_changed(self, value: float) -> None:
        self.controller.on_scale_changed(value)

    def _on_angle_changed(self, value: float) -> None:
        self.controller.on_angle_changed(value)

    def _on_canvas_transform_committed(self, before) -> None:
        self.controller.on_canvas_transform_committed(before)

    def _on_flip(self) -> None:
        self.controller.on_flip()

    def _apply_appearance_from_controls(self) -> None:
        self.controller._apply_appearance_from_controls()

    def _on_appearance_preview(self) -> None:
        self.controller.on_appearance_preview()

    def _on_appearance_committed(self) -> None:
        self.controller.on_appearance_committed()

    def _on_appearance_resample(self) -> None:
        self.controller.on_appearance_resample()

    def _push_transform(self, before) -> None:
        self.controller._push_transform(before)

    def delete_selected(self) -> None:
        self.controller.delete_selected()

    def duplicate_selected(self) -> None:
        self.controller.duplicate_selected()

    def nudge(self, dx: int, dy: int) -> None:
        self.controller.nudge(dx, dy)

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
        visible_instances = []
        for inst in self.doc.instances:
            if not getattr(inst, "visible", True):
                continue
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
            visible_instances.append(inst)

        bg_image = Image.new("RGB", (bw, bh))
        try:
            image_path, _label_path, _boxes = save_composite(
                bg_image,
                Path(bg_path),
                visible_instances,
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
