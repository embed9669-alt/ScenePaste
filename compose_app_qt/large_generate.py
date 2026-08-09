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

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QVBoxLayout,
)


class LargeGenerationDialog(QDialog):
    def __init__(self, objects_dir=None, backgrounds_dir=None, output_dir=None,
                 class_map_text="person=0,vehicle=1", output_format="detect",
                 project_defaults=None, distribution_profile=None, scene_template=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 大规模生成")
        self.resize(760, 650)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.status_timer = QTimer(self); self.status_timer.setInterval(1000); self.status_timer.timeout.connect(self._poll_status)
        layout = QVBoxLayout(self)
        intro = QLabel("支持真实分布驱动、参数化 Scene Template、多进程与断点恢复。中止后可使用相同输出目录 + Run ID 恢复。")
        intro.setWordWrap(True); layout.addWidget(intro)
        form = QFormLayout()
        self.objects = self._path_row(form, "目标目录:", objects_dir, directory=True)
        self.backgrounds = self._path_row(form, "背景目录:", backgrounds_dir, directory=True)
        self.output = self._path_row(form, "输出目录:", output_dir, directory=True)
        self.class_map = QLineEdit(class_map_text); form.addRow("类别映射:", self.class_map)
        defaults = dict(project_defaults or {})
        self.count = QSpinBox(); self.count.setRange(1, 10_000_000); self.count.setValue(int(defaults.get("count", 10000))); form.addRow("生成数量:", self.count)
        self.workers = QSpinBox(); self.workers.setRange(0, 256); self.workers.setValue(int(defaults.get("workers", 0))); self.workers.setSpecialValueText("自动 (CPU-1)"); form.addRow("进程数:", self.workers)
        self.format = QComboBox(); self.format.addItems(["detect","seg","both","coco","semantic","obb","all"]); self.format.setCurrentText(str(defaults.get("output_format", output_format))); form.addRow("输出格式:", self.format)
        self.preview = QDoubleSpinBox(); self.preview.setRange(0,1); self.preview.setDecimals(3); self.preview.setSingleStep(.01); self.preview.setValue(float(defaults.get("preview_ratio", .01))); form.addRow("Preview 比例:", self.preview)
        self.empty_scene = QDoubleSpinBox(); self.empty_scene.setRange(0,1); self.empty_scene.setDecimals(2); self.empty_scene.setSingleStep(.05); self.empty_scene.setValue(float(defaults.get("empty_scene_prob", 0))); form.addRow("纯背景负样本:", self.empty_scene)
        self.recipe = QComboBox(); self.recipe.setEditable(True); self.recipe.addItems(["", "camera-mild", "surveillance", "low-light", "clean"]); self.recipe.setCurrentText(str(defaults.get("augmentation_recipe", "") or "")); self.recipe.setToolTip("整图相机域增强；可输入自定义 JSON 路径"); form.addRow("场景 Recipe:", self.recipe)
        self.object_recipe = QComboBox(); self.object_recipe.setEditable(True); self.object_recipe.addItems(["", "mild", "surveillance-object", "legacy", "off"]); self.object_recipe.setCurrentText(str(defaults.get("object_appearance_recipe", "") or "")); self.object_recipe.setToolTip("单个贴图外观增强；空=保持 v1.0 兼容轻量 HSV；推荐先尝试 mild；可输入自定义 JSON"); form.addRow("目标外观 Recipe:", self.object_recipe)
        self.blend = QComboBox(); self.blend.addItems(["alpha", "gaussian", "hard"]); self.blend.setCurrentText(str(defaults.get("blend_mode", "alpha"))); form.addRow("边缘 Blend:", self.blend)
        self.profile = self._path_row(form, "分布 Profile:", distribution_profile, directory=False, file_filter="Distribution profile (*.json)")
        self.profile_strength = QDoubleSpinBox(); self.profile_strength.setRange(0,1); self.profile_strength.setDecimals(2); self.profile_strength.setValue(float(defaults.get("profile_strength", 1))); form.addRow("Profile 强度:", self.profile_strength)
        self.template = self._path_row(form, "Scene Template:", scene_template, directory=False, file_filter="Scene template (*.json)")
        self.run_id = QLineEdit(); self.run_id.setPlaceholderText("留空自动生成；恢复指定 run 时填写"); form.addRow("Run ID:", self.run_id)
        self.resume = QCheckBox("恢复未完成 Run（Run ID 留空时自动找最新未完成）"); form.addRow("", self.resume)
        layout.addLayout(form)

        learn_row = QHBoxLayout()
        self.learn_btn = QPushButton("从真实数据学习 Profile…"); self.learn_btn.clicked.connect(self._learn_profile)
        learn_row.addWidget(self.learn_btn); learn_row.addStretch(1); layout.addLayout(learn_row)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.hide(); layout.addWidget(self.progress)
        self.live_status = QLabel("等待任务…"); self.live_status.setWordWrap(True); layout.addWidget(self.live_status)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000); layout.addWidget(self.log, 1)
        row = QHBoxLayout(); row.addStretch(1)
        self.start_btn = QPushButton("🚀 开始生成"); self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("中止（可恢复）"); self.cancel_btn.clicked.connect(self.cancel); self.cancel_btn.setEnabled(False)
        close = QPushButton("关闭"); close.clicked.connect(self.close)
        row.addWidget(self.start_btn); row.addWidget(self.cancel_btn); row.addWidget(close); layout.addLayout(row)

    def _path_row(self, form, label, value, directory=True, file_filter="JSON (*.json)"):
        edit = QLineEdit(str(value or "")); btn = QPushButton("选择…")
        box = QHBoxLayout(); box.addWidget(edit, 1); box.addWidget(btn)
        def pick():
            if directory:
                p = QFileDialog.getExistingDirectory(self, label, edit.text() or str(Path.cwd()))
            else:
                p, _ = QFileDialog.getOpenFileName(self, label, edit.text() or str(Path.cwd()), file_filter)
            if p: edit.setText(p)
        btn.clicked.connect(pick); form.addRow(label, box); return edit

    def _learn_profile(self):
        root = QFileDialog.getExistingDirectory(self, "选择真实数据集（LabelMe / YOLO / COCO）", str(Path.cwd()))
        if not root: return
        out, _ = QFileDialog.getSaveFileName(self, "保存 Distribution Profile", "distribution_profile.json", "JSON (*.json)")
        if not out: return
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.finished.connect(lambda code, _status: self._profile_learned(proc, out, code))
        proc.start(sys.executable, ["-m", "scenepaste", "profile", "learn", root, "-o", out])
        self.log.appendPlainText(f"学习真实分布：{root}")

    def _profile_learned(self, proc, out, code):
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text: self.log.appendPlainText(text.rstrip())
        if code == 0:
            self.profile.setText(out); QMessageBox.information(self, "Distribution Profile", f"已生成：{out}")
        else:
            QMessageBox.warning(self, "Distribution Profile", "学习失败，请查看日志。")
        proc.deleteLater()

    def _args(self):
        required = [("目标目录", self.objects.text()), ("背景目录", self.backgrounds.text()), ("输出目录", self.output.text())]
        missing = [name for name, value in required if not value.strip()]
        if missing: raise ValueError("缺少：" + "、".join(missing))
        args = ["-m", "scenepaste", "generate", "--objects", self.objects.text(),
                "--backgrounds", self.backgrounds.text(), "--output", self.output.text(),
                "--count", str(self.count.value()), "--class-map", self.class_map.text(),
                "--workers", str(self.workers.value()), "--output-format", self.format.currentText(),
                "--preview-ratio", str(self.preview.value()), "--profile-strength", str(self.profile_strength.value()),
                "--empty-scene-prob", str(self.empty_scene.value()), "--blend-mode", self.blend.currentText()]
        if self.recipe.currentText().strip(): args += ["--augmentation-recipe", self.recipe.currentText().strip()]
        if self.object_recipe.currentText().strip():
            args += ["--object-appearance-recipe", self.object_recipe.currentText().strip()]
        if self.profile.text().strip(): args += ["--distribution-profile", self.profile.text().strip()]
        if self.template.text().strip(): args += ["--scene-template", self.template.text().strip()]
        if self.run_id.text().strip(): args += ["--run-id", self.run_id.text().strip()]
        if self.resume.isChecked(): args += ["--resume"]
        return args

    def start(self):
        if self.process.state() != QProcess.NotRunning: return
        if not self.run_id.text().strip():
            self.run_id.setText(dt.datetime.now().strftime("gui_%Y%m%d_%H%M%S"))
        try: args = self._args()
        except ValueError as exc: QMessageBox.warning(self, "无法开始", str(exc)); return
        self.log.appendPlainText("$ " + sys.executable + " " + " ".join(args))
        self.start_btn.setEnabled(False); self.cancel_btn.setEnabled(True); self.progress.setRange(0, self.count.value()); self.progress.setValue(0); self.progress.show()
        self.live_status.setText("正在启动…"); self.status_timer.start()
        self.process.start(sys.executable, args)


    def _poll_status(self):
        run_id = self.run_id.text().strip()
        out = self.output.text().strip()
        if not run_id or not out:
            return
        path = Path(out) / ".scenepaste" / "status" / f"{run_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        done = int(data.get("completed", 0)); total = int(data.get("requested", self.count.value()))
        self.progress.setRange(0, max(1, total)); self.progress.setValue(min(done, total))
        rate = float(data.get("images_per_second", 0.0) or 0.0)
        eta = data.get("eta_seconds"); free = data.get("disk_free_bytes")
        eta_text = "n/a" if eta is None else f"{float(eta)/60:.1f} min"
        free_text = "n/a" if free is None else f"{int(free)/1024**3:.1f} GB"
        self.live_status.setText(
            f"{data.get('status','running')} · {done}/{total} · {rate:.2f} img/s · ETA {eta_text} · "
            f"失败 {int(data.get('failed',0))} · 磁盘剩余 {free_text}"
        )

    def cancel(self):
        if self.process.state() != QProcess.NotRunning:
            self.log.appendPlainText("请求中止；已确认任务可用 --resume 恢复…")
            self.process.terminate()
            if not self.process.waitForFinished(3000): self.process.kill()

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text: self.log.appendPlainText(text.rstrip())

    def _finished(self, code, _status):
        self._read_output(); self._poll_status(); self.status_timer.stop(); self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if code == 0: self.log.appendPlainText("✓ 生成完成")
        else: self.log.appendPlainText(f"生成进程退出：code={code}。如任务未完成，可勾选恢复后再次启动。")
