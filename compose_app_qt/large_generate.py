"""Qt dialog for resumable multi-process generation.

Uses QProcess rather than nesting Python ProcessPool inside a QThread. This is
more reliable on Windows/macOS and keeps the GUI process isolated from worker
crashes.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .widgets import CollapsibleSection, MetricCard, PathRow

# CLI generate defaults — only emit advanced flags when the user changes them.
_ADV_DEFAULTS = {
    "min_objects": 1,
    "max_objects": 3,
    "seed": 42,
    "y_min": 0.35,
    "y_max": 0.95,
    "far_height": 0.08,
    "near_height": 0.32,
    "max_iou": 0.15,
    "asset_sampling": "balanced",
    "background_sampling": "balanced",
    "blend_sigma": 1.5,
    "background_cache_size": 16,
    "queue_depth": 0,
}


class LargeGenerationDialog(QDialog):
    def __init__(self, objects_dir=None, backgrounds_dir=None, output_dir=None,
                 class_map_text="person=0,vehicle=1", output_format="detect",
                 project_defaults=None, distribution_profile=None, scene_template=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 批量数据工厂")
        self._settings = QSettings("ScenePaste", "ScenePaste")
        self.resize(1180, 760)
        self.setMinimumSize(980, 680)
        self._active_output_dir: Path | None = None
        self._active_requested_count = 0
        self._trial_mode = False
        self._active_run_id = ""
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self._poll_status)
        defaults = dict(project_defaults or {})
        self._has_project_defaults = bool(project_defaults)

        layout = QVBoxLayout(self)
        header = QFrame()
        header.setObjectName("dialogHero")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(3)
        title = QLabel("批量数据工厂")
        title.setObjectName("dialogTitle")
        subtitle = QLabel("以人工审核素材为基础：规划目标 → 可控 Copy-Paste 合成 → 自动标签 → 输出训练集。")
        subtitle.setObjectName("dialogSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        self._main_splitter = splitter
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        self._tabs = tabs
        left_layout.addWidget(tabs, 1)

        # ---- Tab: 数据与任务 ----
        basic = self._build_basic_tab(
            defaults, objects_dir, backgrounds_dir, output_dir, class_map_text,
            output_format, distribution_profile, scene_template,
        )
        tabs.addTab(basic, "数据与任务")

        # ---- Tab: 高级 ----
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.NoFrame)
        adv_body = QWidget()
        adv_form = QFormLayout(adv_body)
        tip = QLabel("仅当与 CLI 默认值不同时才会写入命令行，保持日志简洁。")
        tip.setWordWrap(True)
        adv_form.addRow(tip)

        self.min_objects = QSpinBox()
        self.min_objects.setRange(0, 64)
        self.min_objects.setValue(int(defaults.get("min_objects", _ADV_DEFAULTS["min_objects"])))
        adv_form.addRow("最少目标数:", self.min_objects)
        self.max_objects = QSpinBox()
        self.max_objects.setRange(1, 64)
        self.max_objects.setValue(int(defaults.get("max_objects", _ADV_DEFAULTS["max_objects"])))
        adv_form.addRow("最多目标数:", self.max_objects)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(int(defaults.get("seed", _ADV_DEFAULTS["seed"])))
        adv_form.addRow("随机种子:", self.seed)
        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(0, 1)
        self.y_min.setDecimals(3)
        self.y_min.setValue(float(defaults.get("y_min", _ADV_DEFAULTS["y_min"])))
        adv_form.addRow("落地点 y-min:", self.y_min)
        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(0, 1)
        self.y_max.setDecimals(3)
        self.y_max.setValue(float(defaults.get("y_max", _ADV_DEFAULTS["y_max"])))
        adv_form.addRow("落地点 y-max:", self.y_max)
        self.far_height = QDoubleSpinBox()
        self.far_height.setRange(0.01, 1)
        self.far_height.setDecimals(3)
        self.far_height.setValue(float(defaults.get("far_height", _ADV_DEFAULTS["far_height"])))
        adv_form.addRow("远处高度比:", self.far_height)
        self.near_height = QDoubleSpinBox()
        self.near_height.setRange(0.01, 1)
        self.near_height.setDecimals(3)
        self.near_height.setValue(float(defaults.get("near_height", _ADV_DEFAULTS["near_height"])))
        adv_form.addRow("近处高度比:", self.near_height)
        self.max_iou = QDoubleSpinBox()
        self.max_iou.setRange(0, 1)
        self.max_iou.setDecimals(3)
        self.max_iou.setValue(float(defaults.get("max_iou", _ADV_DEFAULTS["max_iou"])))
        adv_form.addRow("最大 IoU:", self.max_iou)
        self.asset_sampling = QComboBox()
        self.asset_sampling.addItems(["balanced", "random"])
        self.asset_sampling.setCurrentText(
            str(defaults.get("asset_sampling", _ADV_DEFAULTS["asset_sampling"]))
        )
        adv_form.addRow("目标采样:", self.asset_sampling)
        self.background_sampling = QComboBox()
        self.background_sampling.addItems(["balanced", "random"])
        self.background_sampling.setCurrentText(
            str(defaults.get("background_sampling", _ADV_DEFAULTS["background_sampling"]))
        )
        adv_form.addRow("背景采样:", self.background_sampling)
        self.blend_sigma = QDoubleSpinBox()
        self.blend_sigma.setRange(0.1, 32)
        self.blend_sigma.setDecimals(2)
        self.blend_sigma.setValue(float(defaults.get("blend_sigma", _ADV_DEFAULTS["blend_sigma"])))
        adv_form.addRow("Blend sigma:", self.blend_sigma)
        self.no_preview = QCheckBox("不保存带框预览")
        adv_form.addRow("", self.no_preview)
        self.queue_depth = QSpinBox()
        self.queue_depth.setRange(0, 10_000)
        self.queue_depth.setValue(int(defaults.get("queue_depth", _ADV_DEFAULTS["queue_depth"])))
        self.queue_depth.setSpecialValueText("默认")
        adv_form.addRow("队列深度:", self.queue_depth)
        self.bg_cache = QSpinBox()
        self.bg_cache.setRange(0, 1024)
        self.bg_cache.setValue(
            int(defaults.get("background_cache_size", _ADV_DEFAULTS["background_cache_size"]))
        )
        adv_form.addRow("背景缓存大小:", self.bg_cache)
        adv_scroll.setWidget(adv_body)
        tabs.addTab(adv_scroll, "运行与性能")

        splitter.addWidget(left)

        # ---- right: task dashboard ----
        right = self._build_task_dashboard()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 430])

        row = QHBoxLayout()
        row.addStretch(1)
        self.trial_btn = QPushButton("试生成 10 张")
        self.trial_btn.setToolTip("先生成 10 张到独立试验目录，检查效果后再正式生成。")
        self.trial_btn.clicked.connect(self.start_trial)
        self.start_btn = QPushButton("开始正式生成")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("中止（可恢复）")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(self.trial_btn)
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(close)
        layout.addLayout(row)
        self._restore_dialog_state()

    def _restore_dialog_state(self) -> None:
        geometry = self._settings.value("factory/geometry")
        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass
        splitter = self._settings.value("factory/splitter")
        if splitter is not None:
            try:
                self._main_splitter.restoreState(splitter)
            except Exception:
                pass
        try:
            tab = int(self._settings.value("factory/tab", self._tabs.currentIndex()))
            if 0 <= tab < self._tabs.count():
                self._tabs.setCurrentIndex(tab)
        except Exception:
            pass

    def _build_basic_tab(self, defaults, objects_dir, backgrounds_dir, output_dir,
                         class_map_text, output_format, distribution_profile, scene_template):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(10, 10, 10, 18)
        root.setSpacing(12)

        tip = QLabel(
            "本对话框仅做可控 Copy-Paste 批量合成。"
            "请先在「素材工作室」审核前景/背景资产；放置区域与困难样本在下方「数据策略与增强」中配置。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("mutedLabel")
        root.addWidget(tip)

        source_box = QGroupBox("1  ·  数据源")
        form = QFormLayout(source_box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.objects = PathRow(self, objects_dir, directory=True)
        form.addRow("目标素材:", self.objects.widget)
        self.backgrounds = PathRow(self, backgrounds_dir, directory=True)
        form.addRow("真实背景:", self.backgrounds.widget)
        self.output = PathRow(self, output_dir, directory=True)
        form.addRow("输出目录:", self.output.widget)
        self.class_map = QLineEdit(class_map_text)
        self.class_map.setPlaceholderText("person=0,vehicle=1")
        form.addRow("类别映射:", self.class_map)

        self.auto_cutout = QCheckBox("普通原图自动抠图（rembg）")
        self.auto_cutout.setToolTip("目标目录没有 LabelMe Mask 时可启用；需要安装 scenepaste[auto]。")
        form.addRow("", self.auto_cutout)
        cutout_options = QWidget()
        cutout_layout = QHBoxLayout(cutout_options)
        cutout_layout.setContentsMargins(0, 0, 0, 0)
        cutout_layout.setSpacing(8)
        self.auto_cutout_label = QComboBox()
        self.auto_cutout_label.setEditable(True)
        for name in ["person", "truck", "motorcycle", "vehicle", "auto"]:
            self.auto_cutout_label.addItem(name)
        for part in class_map_text.replace("，", ",").split(","):
            part = part.strip()
            if "=" in part:
                name = part.split("=", 1)[0].strip()
                if name:
                    self.auto_cutout_label.setCurrentText(name)
                    break
        self.auto_cutout_subdir = QCheckBox("子目录名作为类别")
        cutout_layout.addWidget(QLabel("默认类别"))
        cutout_layout.addWidget(self.auto_cutout_label, 1)
        cutout_layout.addWidget(self.auto_cutout_subdir)
        form.addRow("抠图选项:", cutout_options)
        root.addWidget(source_box)

        task_box = QGroupBox("2  ·  合成任务")
        task = QFormLayout(task_box)
        task.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.count = QSpinBox()
        self.count.setRange(1, 10_000_000)
        self.count.setValue(int(defaults.get("count", 10000)))
        self.count.setToolTip("建议先用 50~200 张小任务验证生成质量，再启动大规模任务。")
        task.addRow("生成数量:", self.count)
        self.format = QComboBox()
        self.format.addItems(["detect", "seg", "both", "coco", "semantic", "obb", "all"])
        self.format.setCurrentText(str(defaults.get("output_format", output_format)))
        task.addRow("标注输出:", self.format)
        self.workers = QSpinBox()
        self.workers.setRange(0, 256)
        self.workers.setValue(int(defaults.get("workers", 0)))
        self.workers.setSpecialValueText("自动")
        self.workers.setToolTip("并行 worker 数量；0 表示自动选择。")
        task.addRow("并行进程:", self.workers)
        self.preview = QDoubleSpinBox()
        self.preview.setRange(0, 1)
        self.preview.setDecimals(3)
        self.preview.setSingleStep(0.01)
        self.preview.setValue(float(defaults.get("preview_ratio", 0.01)))
        task.addRow("带框预览比例:", self.preview)
        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        self.run_id = QLineEdit()
        self.run_id.setPlaceholderText("留空自动生成")
        self.resume = QCheckBox("恢复未完成任务")
        run_layout.addWidget(self.run_id, 1)
        run_layout.addWidget(self.resume)
        task.addRow("Run ID:", run_row)
        root.addWidget(task_box)

        strategy = CollapsibleSection(
            "数据策略与增强",
            expanded=False,
            subtitle="真实分布 Profile、负样本、增强 Recipe 和 Scene Template 属于进阶能力；首次使用可保持默认。",
        )
        s = QFormLayout()
        self.empty_scene = QDoubleSpinBox()
        self.empty_scene.setRange(0, 1)
        self.empty_scene.setDecimals(2)
        self.empty_scene.setSingleStep(0.05)
        self.empty_scene.setValue(float(defaults.get("empty_scene_prob", 0)))
        s.addRow("纯背景负样本:", self.empty_scene)
        self.recipe = QComboBox()
        self.recipe.setEditable(True)
        self.recipe.addItems(["", "camera-mild", "surveillance", "low-light", "clean"])
        self.recipe.setCurrentText(str(defaults.get("augmentation_recipe", "") or ""))
        s.addRow("场景增强 Recipe:", self.recipe)
        self.object_recipe = QComboBox()
        self.object_recipe.setEditable(True)
        self.object_recipe.addItems(["", "mild", "surveillance-object", "legacy", "off"])
        self.object_recipe.setCurrentText(str(defaults.get("object_appearance_recipe", "") or ""))
        s.addRow("目标外观 Recipe:", self.object_recipe)
        self.blend = QComboBox()
        self.blend.addItems(["alpha", "gaussian", "hard"])
        self.blend.setCurrentText(str(defaults.get("blend_mode", "alpha")))
        s.addRow("传统融合方式:", self.blend)
        self.rectangle_mask_mode = QComboBox()
        self.rectangle_mask_mode.addItem("GrabCut 自动前景细化（推荐）", "grabcut")
        self.rectangle_mask_mode.addItem("拒绝矩形 bbox", "reject")
        self.rectangle_mask_mode.addItem("旧版整块矩形（不推荐）", "legacy")
        rect_default = str(defaults.get("rectangle_mask_mode", "grabcut"))
        rect_idx = self.rectangle_mask_mode.findData(rect_default)
        self.rectangle_mask_mode.setCurrentIndex(max(0, rect_idx))
        self.rectangle_mask_mode.setToolTip(
            "LabelMe rectangle 是检测框，不是前景 Mask。推荐 GrabCut 自动细化，避免把原图背景矩形一起贴入。"
        )
        s.addRow("矩形标注处理:", self.rectangle_mask_mode)
        self.profile = PathRow(self, distribution_profile, directory=False, filter_text="Distribution profile (*.json)")
        s.addRow("分布 Profile:", self.profile.widget)
        profile_row = QWidget()
        profile_row_l = QHBoxLayout(profile_row)
        profile_row_l.setContentsMargins(0, 0, 0, 0)
        self.profile_strength = QDoubleSpinBox()
        self.profile_strength.setRange(0, 1)
        self.profile_strength.setDecimals(2)
        self.profile_strength.setValue(float(defaults.get("profile_strength", 1)))
        learn_btn = QPushButton("从真实数据学习…")
        learn_btn.clicked.connect(self._learn_profile)
        profile_row_l.addWidget(self.profile_strength)
        profile_row_l.addWidget(learn_btn)
        profile_row_l.addStretch(1)
        s.addRow("Profile 强度:", profile_row)
        self.template = PathRow(self, scene_template, directory=False, filter_text="Scene template (*.json)")
        s.addRow("Scene Template:", self.template.widget)
        self.scene_region_mode = QComboBox()
        self.scene_region_mode.addItems(["auto", "explicit", "ground-prior", "none"])
        self.scene_region_mode.setCurrentText(str(defaults.get("scene_region_mode", "auto")))
        self.scene_region_mode.setToolTip("可放置区域：显式 LabelMe paste-zone 优先，或使用地面先验。")
        s.addRow("可放置区域:", self.scene_region_mode)
        self.hardcase_recipe = QComboBox()
        self.hardcase_recipe.setEditable(True)
        self.hardcase_recipe.addItems(["", "small-object", "far-occluded", "crowded"])
        self.hardcase_recipe.setCurrentText(str(defaults.get("hardcase_recipe", "") or ""))
        self.hardcase_recipe.setToolTip("主动难例 Recipe；留空则按常规规划摆放。")
        s.addRow("困难样本策略:", self.hardcase_recipe)
        strategy.set_content_layout(s)
        root.addWidget(strategy)
        root.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _build_task_dashboard(self):
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(10)

        title = QLabel("任务状态")
        title.setObjectName("sectionHeadline")
        layout.addWidget(title)
        self.live_status = QLabel("准备就绪 · 配置左侧参数后即可开始")
        self.live_status.setObjectName("statusPill")
        self.live_status.setWordWrap(True)
        layout.addWidget(self.live_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("尚未开始")
        layout.addWidget(self.progress)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)
        self.metric_completed = MetricCard("已完成", "0")
        self.metric_objects = MetricCard("目标数", "0")
        self.metric_failed = MetricCard("失败", "0")
        self.metric_speed = MetricCard("速度", "—")
        metrics.addWidget(self.metric_completed, 0, 0)
        metrics.addWidget(self.metric_objects, 0, 1)
        metrics.addWidget(self.metric_failed, 1, 0)
        metrics.addWidget(self.metric_speed, 1, 1)
        layout.addLayout(metrics)

        preview_box = QGroupBox("最近生成结果")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel("开始合成后，这里会显示最近样本预览（若已保存带框预览）。")
        self.preview_label.setObjectName("previewSurface")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.preview_label, 1)
        self.preview_path_label = QLabel("暂无结果")
        self.preview_path_label.setObjectName("mutedLabel")
        self.preview_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_layout.addWidget(self.preview_path_label)
        preview_actions = QHBoxLayout()
        self.open_debug_btn = QPushButton("查看大图")
        self.open_debug_btn.clicked.connect(self._open_latest_debug)
        open_output = QPushButton("打开输出目录")
        open_output.clicked.connect(self._open_output_dir)
        preview_actions.addWidget(self.open_debug_btn)
        preview_actions.addWidget(open_output)
        preview_actions.addStretch(1)
        preview_layout.addLayout(preview_actions)
        layout.addWidget(preview_box, 1)

        self.log_section = CollapsibleSection(
            "详细日志", expanded=False, subtitle="出现问题时再展开；正常生成无需关注命令行输出。"
        )
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 4, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setPlaceholderText("运行日志会显示在这里…")
        self.log.setMinimumHeight(150)
        log_layout.addWidget(self.log)
        self.log_section.set_content_layout(log_layout)
        layout.addWidget(self.log_section)
        return right

    def _current_run_output(self) -> Path | None:
        if self._active_output_dir is not None:
            return self._active_output_dir
        text = self.output.text().strip()
        return Path(text) if text else None

    def _open_output_dir(self):
        current = self._current_run_output()
        out = str(current) if current is not None else ""
        if not out:
            QMessageBox.information(self, "输出目录", "请先设置输出目录。")
            return
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _refresh_latest_debug_preview(self):
        current = self._current_run_output()
        out = str(current) if current is not None else ""
        if not out:
            return
        root = Path(out) / ".scenepaste" / "debug"
        if not root.exists():
            return
        files = sorted(
            root.rglob("*_compare.jpg"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not files:
            return
        latest = files[0]
        if getattr(self, "_latest_preview_path", None) == str(latest):
            return
        pix = QPixmap(str(latest))
        if pix.isNull():
            return
        self._latest_preview_path = str(latest)
        target = self.preview_label.size()
        if target.width() < 100 or target.height() < 100:
            target = self.preview_label.minimumSizeHint()
        self.preview_label.setPixmap(pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_path_label.setText(latest.name)

    def _learn_profile(self):
        root = QFileDialog.getExistingDirectory(
            self, "选择真实数据集（LabelMe / YOLO / COCO）", str(Path.cwd()),
        )
        if not root:
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "保存 Distribution Profile", "distribution_profile.json", "JSON (*.json)",
        )
        if not out:
            return
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.finished.connect(lambda code, _status: self._profile_learned(proc, out, code))
        proc.start(sys.executable, ["-m", "scenepaste", "profile", "learn", root, "-o", out])
        self.log.appendPlainText(f"学习真实分布：{root}")

    def _profile_learned(self, proc, out, code):
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())
        if code == 0:
            self.profile.setText(out)
            QMessageBox.information(self, "Distribution Profile", f"已生成：{out}")
        else:
            QMessageBox.warning(self, "Distribution Profile", "学习失败，请查看日志。")
        proc.deleteLater()

    def _maybe(self, args: list, flag: str, value, default) -> None:
        if isinstance(default, float):
            if abs(float(value) - float(default)) > 1e-9:
                args += [flag, str(value)]
        elif value != default:
            args += [flag, str(value)]

    def _args(self, *, count_override: int | None = None, output_override: Path | None = None, run_id_override: str | None = None):
        required = [
            ("目标目录", self.objects.text()),
            ("背景目录", self.backgrounds.text()),
            ("输出目录", self.output.text()),
        ]
        missing = [name for name, value in required if not value.strip()]
        if missing:
            raise ValueError("缺少：" + "、".join(missing))
        args = [
            "-m", "scenepaste", "generate",
            "--objects", self.objects.text(),
            "--backgrounds", self.backgrounds.text(),
            "--output", str(output_override or Path(self.output.text())),
            "--count", str(count_override if count_override is not None else self.count.value()),
            "--class-map", self.class_map.text(),
            "--workers", str(self.workers.value()),
            "--output-format", self.format.currentText(),
            "--rectangle-mask-mode", str(self.rectangle_mask_mode.currentData() or "grabcut"),
            "--preview-ratio", str(self.preview.value()),
            "--profile-strength", str(self.profile_strength.value()),
            "--empty-scene-prob", str(self.empty_scene.value()),
            "--blend-mode", self.blend.currentText(),
            "--scene-region-mode", self.scene_region_mode.currentText(),
        ]
        self._maybe(args, "--min-objects", self.min_objects.value(), _ADV_DEFAULTS["min_objects"])
        self._maybe(args, "--max-objects", self.max_objects.value(), _ADV_DEFAULTS["max_objects"])
        self._maybe(args, "--seed", self.seed.value(), _ADV_DEFAULTS["seed"])
        self._maybe(args, "--y-min", self.y_min.value(), _ADV_DEFAULTS["y_min"])
        self._maybe(args, "--y-max", self.y_max.value(), _ADV_DEFAULTS["y_max"])
        self._maybe(args, "--far-height", self.far_height.value(), _ADV_DEFAULTS["far_height"])
        self._maybe(args, "--near-height", self.near_height.value(), _ADV_DEFAULTS["near_height"])
        self._maybe(args, "--max-iou", self.max_iou.value(), _ADV_DEFAULTS["max_iou"])
        self._maybe(
            args, "--asset-sampling", self.asset_sampling.currentText(),
            _ADV_DEFAULTS["asset_sampling"],
        )
        self._maybe(
            args, "--background-sampling", self.background_sampling.currentText(),
            _ADV_DEFAULTS["background_sampling"],
        )
        self._maybe(args, "--blend-sigma", self.blend_sigma.value(), _ADV_DEFAULTS["blend_sigma"])
        self._maybe(
            args, "--background-cache-size", self.bg_cache.value(),
            _ADV_DEFAULTS["background_cache_size"],
        )
        if self.queue_depth.value() != _ADV_DEFAULTS["queue_depth"]:
            args += ["--queue-depth", str(self.queue_depth.value())]
        if self.no_preview.isChecked():
            args += ["--no-preview"]
        if self.auto_cutout.isChecked():
            args += ["--auto-cutout"]
            label = self.auto_cutout_label.currentText().strip() or "auto"
            args += ["--auto-cutout-label", label]
            if self.auto_cutout_subdir.isChecked():
                args += ["--auto-cutout-label-from-subdir"]
        if self.recipe.currentText().strip():
            args += ["--augmentation-recipe", self.recipe.currentText().strip()]
        if self.object_recipe.currentText().strip():
            args += ["--object-appearance-recipe", self.object_recipe.currentText().strip()]
        hard = self.hardcase_recipe.currentText().strip()
        if hard:
            args += ["--hardcase-recipe", hard]
        if self.profile.text():
            args += ["--distribution-profile", self.profile.text()]
        if self.template.text():
            args += ["--scene-template", self.template.text()]
        effective_run_id = run_id_override or self.run_id.text().strip()
        if effective_run_id:
            args += ["--run-id", effective_run_id]
        if self.resume.isChecked() and not (effective_run_id and effective_run_id.startswith("trial_")):
            args += ["--resume"]
        return args

    def start_trial(self):
        """Generate 10 isolated samples before a long run."""
        if self.process.state() != QProcess.NotRunning:
            return
        base = self.output.text().strip()
        if not base:
            QMessageBox.warning(self, "无法试生成", "请先设置输出目录。")
            return
        run_id = dt.datetime.now().strftime("trial_%Y%m%d_%H%M%S")
        trial_out = Path(base) / "_trials" / run_id
        self._start_generation(count_override=10, output_override=trial_out, run_id_override=run_id, trial=True)

    def start(self):
        if self.process.state() != QProcess.NotRunning:
            return
        if not self.run_id.text().strip():
            self.run_id.setText(dt.datetime.now().strftime("gui_%Y%m%d_%H%M%S"))
        self._start_generation(
            count_override=self.count.value(),
            output_override=Path(self.output.text()),
            run_id_override=self.run_id.text().strip(),
            trial=False,
        )

    def _start_generation(self, *, count_override: int, output_override: Path, run_id_override: str, trial: bool):
        try:
            args = self._args(
                count_override=count_override,
                output_override=output_override,
                run_id_override=run_id_override,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self._active_output_dir = Path(output_override)
        self._active_requested_count = int(count_override)
        self._trial_mode = bool(trial)
        self._active_run_id = run_id_override
        self._latest_preview_path = None
        self.log.appendPlainText("$ " + sys.executable + " " + " ".join(args))
        self.start_btn.setEnabled(False)
        self.trial_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setRange(0, count_override)
        self.progress.setValue(0)
        self.progress.setFormat("启动中…")
        self.live_status.setText(("试生成" if trial else "正式生成") + "任务正在启动…")
        self.metric_completed.set_value("0", f"目标 {count_override} 张")
        self.metric_objects.set_value("0")
        self.metric_failed.set_value("0")
        self.metric_speed.set_value("—")
        self.status_timer.start()
        self.process.start(sys.executable, args)

    def _open_latest_debug(self):
        current = self._current_run_output()
        out = str(current) if current is not None else ""
        if not out:
            QMessageBox.information(self, "结果预览", "请先设置输出目录。")
            return
        debug_root = Path(out) / ".scenepaste" / "debug"
        files = sorted(
            debug_root.rglob("*_compare.jpg"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not files:
            QMessageBox.information(self, "结果预览", "没有找到对比图。可先完成一次生成后再查看。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(files[0].resolve())))

    def closeEvent(self, event):  # pragma: no cover - GUI only
        try:
            self._settings.setValue("factory/geometry", self.saveGeometry())
            self._settings.setValue("factory/splitter", self._main_splitter.saveState())
            self._settings.setValue("factory/tab", self._tabs.currentIndex())
        except Exception:
            pass
        return super().closeEvent(event)

    def _poll_status(self):
        run_id = self._active_run_id or self.run_id.text().strip()
        current = self._current_run_output()
        out = str(current) if current is not None else ""
        if not run_id or not out:
            return
        path = Path(out) / ".scenepaste" / "status" / f"{run_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        done = int(data.get("completed", 0))
        total = int(data.get("requested", self.count.value()))
        failed = int(data.get("failed", 0))
        objects = int(data.get("objects", 0))
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(min(done, total))
        pct = 100.0 * done / max(1, total)
        self.progress.setFormat(f"{done:,} / {total:,}  ·  {pct:.1f}%")
        rate = float(data.get("images_per_second", 0.0) or 0.0)
        eta = data.get("eta_seconds")
        free = data.get("disk_free_bytes")
        eta_text = "计算中" if eta is None else (
            f"{float(eta)/60:.1f} 分钟" if float(eta) >= 60 else f"{float(eta):.0f} 秒"
        )
        free_text = "未知" if free is None else f"{int(free)/1024**3:.1f} GB"
        status = str(data.get("status", "running"))
        status_cn = {
            "running": "生成中",
            "completed": "已完成",
            "partial": "部分完成",
            "interrupted": "已中止",
        }.get(status, status)
        self.live_status.setText(f"{status_cn} · ETA {eta_text} · 磁盘剩余 {free_text}")
        self.metric_completed.set_value(f"{done:,}", f"目标 {total:,} 张")
        self.metric_objects.set_value(f"{objects:,}")
        self.metric_failed.set_value(f"{failed:,}", "0 为最佳")
        self.metric_speed.set_value(f"{rate:.2f}", "img/s")
        self._refresh_latest_debug_preview()

    def cancel(self):
        if self.process.state() != QProcess.NotRunning:
            self.log.appendPlainText("请求中止；可用恢复选项再次启动…")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())

    def _finished(self, code, _status):
        self._read_output()
        self._poll_status()
        self.status_timer.stop()
        self.start_btn.setEnabled(True)
        self.trial_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._refresh_latest_debug_preview()
        if code == 0:
            self.log.appendPlainText("生成完成")
            self.live_status.setText(("试生成完成" if self._trial_mode else "生成完成") + " · 可以查看最近生成结果")
            self.progress.setFormat("生成完成 · 100%")
            if self._trial_mode:
                QMessageBox.information(
                    self, "试生成完成",
                    "10 张试验样本已经生成到独立目录。建议先查看右侧预览与输出目录，"
                    "确认效果后再点击“开始正式生成”。",
                )
        else:
            self.live_status.setText(f"任务退出 · code={code} · 可使用恢复功能继续")
            self.log.appendPlainText(
                f"生成进程退出：code={code}。如任务未完成，可勾选恢复后再次启动。"
            )
