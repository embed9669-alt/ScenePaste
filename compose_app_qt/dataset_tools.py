"""Dataset tools dialog — GUI wrappers for CLI subcommands without a dedicated UI.

Covers: analyze / split / merge / recipe / project init|show|validate.
Long jobs run via QProcess against ``python -m scenepaste …``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .widgets import PathRow


class DatasetToolsDialog(QDialog):
    def __init__(self, dataset_root=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 数据集工具")
        self.resize(920, 640)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("对齐 CLI：analyze · split · merge · recipe · project"))

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        left_l.addWidget(self.tabs, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setPlaceholderText("命令输出…")
        right_l.addWidget(self.log, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 380])

        self._build_analyze(dataset_root)
        self._build_split(dataset_root)
        self._build_merge()
        self._build_recipe()
        self._build_project()

        row = QHBoxLayout()
        self.status = QLabel("就绪")
        row.addWidget(self.status, 1)
        self.stop_btn = QPushButton("中止")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)

    def _tab(self, name):
        w = QWidget()
        f = QFormLayout(w)
        self.tabs.addTab(w, name)
        return f

    def _button(self, form, text, callback):
        b = QPushButton(text)
        b.clicked.connect(callback)
        form.addRow("", b)
        return b

    def _build_analyze(self, root):
        f = self._tab("Analyze")
        self.an_paths = QLineEdit(str(root or ""))
        self.an_paths.setPlaceholderText("一个或多个数据集路径，空格分隔")
        f.addRow("数据集路径:", self.an_paths)
        self.an_json = QCheckBox("输出 JSON")
        f.addRow("", self.an_json)
        self._button(f, "运行 analyze", self._analyze)

    def _build_split(self, root):
        f = self._tab("Split")
        self.sp_input = PathRow(self, root)
        f.addRow("输入目录:", self.sp_input.widget)
        self.sp_val = QDoubleSpinBox()
        self.sp_val.setRange(0.01, 0.95)
        self.sp_val.setDecimals(3)
        self.sp_val.setValue(0.2)
        f.addRow("验证集比例:", self.sp_val)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_147_483_647)
        self.sp_seed.setValue(42)
        f.addRow("种子:", self.sp_seed)
        self.sp_run = QLineEdit("latest")
        f.addRow("Run ID:", self.sp_run)
        self.sp_mode = QComboBox()
        self.sp_mode.addItems(["move", "copy"])
        f.addRow("模式:", self.sp_mode)
        self.sp_dry = QCheckBox("仅预览")
        f.addRow("", self.sp_dry)
        self._button(f, "运行 split", self._split)

    def _build_merge(self):
        f = self._tab("Merge")
        self.mg_inputs = QLineEdit()
        self.mg_inputs.setPlaceholderText("多个数据集路径，空格分隔")
        f.addRow("输入数据集:", self.mg_inputs)
        self.mg_output = PathRow(self, "merged", directory=True)
        f.addRow("输出目录:", self.mg_output.widget)
        self.mg_force = QCheckBox("允许非空输出")
        f.addRow("", self.mg_force)
        self._button(f, "运行 merge", self._merge)

    def _build_recipe(self):
        f = self._tab("Recipe")
        self.rc_kind = QComboBox()
        self.rc_kind.addItems(["scene", "object"])
        f.addRow("类型:", self.rc_kind)
        self.rc_name = QLineEdit()
        self.rc_name.setPlaceholderText("内置名或 JSON 路径")
        f.addRow("Recipe:", self.rc_name)
        self.rc_export = PathRow(self, "", directory=False, filter_text="JSON (*.json)")
        f.addRow("导出路径:", self.rc_export.widget)
        self._button(f, "列出", self._recipe_list)
        self._button(f, "查看", self._recipe_show)
        self._button(f, "导出", self._recipe_export)

    def _build_project(self):
        f = self._tab("Project")
        self.pj_path = QLineEdit("scenepaste.project.json")
        pj_browse = QPushButton("选择…")

        def _pick_project():
            p, _ = QFileDialog.getOpenFileName(
                self, "选择工程文件",
                self.pj_path.text() or str(Path.cwd()),
                "ScenePaste Project (*.json);;All (*)",
            )
            if p:
                self.pj_path.setText(p)

        pj_browse.clicked.connect(_pick_project)
        pj_box = QWidget()
        pj_row = QHBoxLayout(pj_box)
        pj_row.setContentsMargins(0, 0, 0, 0)
        pj_row.addWidget(self.pj_path, 1)
        pj_row.addWidget(pj_browse)
        f.addRow("工程路径:", pj_box)
        self.pj_name = QLineEdit()
        f.addRow("名称:", self.pj_name)
        self.pj_objects = PathRow(self, "", directory=True)
        f.addRow("目标目录:", self.pj_objects.widget)
        self.pj_backgrounds = PathRow(self, "", directory=True)
        f.addRow("背景目录:", self.pj_backgrounds.widget)
        self.pj_output = PathRow(self, "", directory=True)
        f.addRow("输出目录:", self.pj_output.widget)
        self.pj_class_map = QLineEdit("person=0,vehicle=1")
        f.addRow("类别映射:", self.pj_class_map)
        self.pj_generation = QCheckBox("validate 时要求 objects/backgrounds/output")
        f.addRow("", self.pj_generation)
        self._button(f, "初始化", self._project_init)
        self._button(f, "显示", self._project_show)
        self._button(f, "校验", self._project_validate)

    def _analyze(self):
        paths = self.an_paths.text().split()
        if not paths:
            QMessageBox.warning(self, "参数不完整", "请填写至少一个数据集路径。")
            return
        args = ["analyze"] + paths
        if self.an_json.isChecked():
            args.append("--json")
        self._run(args)

    def _split(self):
        args = [
            "split",
            "--input", self.sp_input.text(),
            "--val-ratio", str(self.sp_val.value()),
            "--seed", str(self.sp_seed.value()),
            "--run-id", self.sp_run.text().strip() or "latest",
            "--mode", self.sp_mode.currentText(),
        ]
        if self.sp_dry.isChecked():
            args.append("--dry-run")
        self._run(args)

    def _merge(self):
        inputs = self.mg_inputs.text().split()
        if not inputs or not self.mg_output.text():
            QMessageBox.warning(self, "参数不完整", "请填写输入路径与输出目录。")
            return
        args = ["merge"] + inputs + ["--output", self.mg_output.text()]
        if self.mg_force.isChecked():
            args.append("--force")
        self._run(args)

    def _recipe_list(self):
        self._run(["recipe", "--kind", self.rc_kind.currentText(), "list"])

    def _recipe_show(self):
        if not self.rc_name.text().strip():
            QMessageBox.warning(self, "参数不完整", "请填写 Recipe 名称或路径。")
            return
        self._run([
            "recipe", "--kind", self.rc_kind.currentText(),
            "show", self.rc_name.text().strip(),
        ])

    def _recipe_export(self):
        if not self.rc_name.text().strip() or not self.rc_export.text():
            QMessageBox.warning(self, "参数不完整", "请填写 Recipe 与导出路径。")
            return
        self._run([
            "recipe", "--kind", self.rc_kind.currentText(),
            "export", self.rc_name.text().strip(),
            "-o", self.rc_export.text(),
        ])

    def _project_init(self):
        args = ["project", "init", self.pj_path.text().strip() or "."]
        if self.pj_name.text().strip():
            args += ["--name", self.pj_name.text().strip()]
        if self.pj_objects.text():
            args += ["--objects", self.pj_objects.text()]
        if self.pj_backgrounds.text():
            args += ["--backgrounds", self.pj_backgrounds.text()]
        if self.pj_output.text():
            args += ["--output", self.pj_output.text()]
        if self.pj_class_map.text().strip():
            args += ["--class-map", self.pj_class_map.text().strip()]
        self._run(args)

    def _project_show(self):
        path = self.pj_path.text().strip() or "scenepaste.project.json"
        self._run(["project", "show", path])

    def _project_validate(self):
        path = self.pj_path.text().strip() or "scenepaste.project.json"
        args = ["project", "validate", path]
        if self.pj_generation.isChecked():
            args.append("--generation")
        self._run(args)

    def _run(self, args):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "任务进行中", "请先等待当前任务结束或中止。")
            return
        if any(x == "" for x in args):
            QMessageBox.warning(self, "参数不完整", "请先填写必要路径。")
            return
        cmd = ["-m", "scenepaste"] + list(args)
        self.log.appendPlainText("$ " + sys.executable + " " + " ".join(cmd))
        self.status.setText("运行中…")
        self.stop_btn.setEnabled(True)
        self.process.start(sys.executable, cmd)

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())

    def _finished(self, code, _status):
        self._read_output()
        self.stop_btn.setEnabled(False)
        self.status.setText("完成" if code == 0 else f"退出 code={code}")

    def stop(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
