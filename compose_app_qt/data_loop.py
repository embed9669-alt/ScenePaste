"""Unified ScenePaste data-loop center.

A thin Qt orchestration layer over the tested CLI. Long-running operations use
QProcess, keeping the editor responsive and ensuring CLI/GUI behavior stays in
sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .widgets import PathRow


class DataLoopCenterDialog(QDialog):
    def __init__(self, dataset_root=None, output_root=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 数据闭环中心")
        self.resize(1000, 700)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        layout = QVBoxLayout(self)
        title = QLabel("难例 → QA/泄漏 → 真实/合成对比 → 策展 / 分片（参数对齐 CLI）")
        title.setWordWrap(True)
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setPlaceholderText("命令输出…")
        right_layout.addWidget(self.log, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([580, 400])

        self._build_project()
        self._build_profile(dataset_root)
        self._build_hardmine(dataset_root)
        self._build_quality(dataset_root)
        self._build_compare(dataset_root)
        self._build_curation(dataset_root)
        self._build_publish(dataset_root, output_root)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        w = QWidget()
        f = QFormLayout(w)
        f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        scroll.setWidget(w)
        self.tabs.addTab(scroll, name)
        return f

    def _button(self, form, text, callback):
        b = QPushButton(text)
        b.clicked.connect(callback)
        form.addRow("", b)
        return b

    def _build_project(self):
        f = self._tab("工程")
        self.project_path = PathRow(
            self, "scenepaste.project.json", directory=False,
            filter_text="ScenePaste Project (*.json)",
        )
        f.addRow("Project Manifest:", self.project_path.widget)
        self._button(f, "加载工程并填充工作流", self._load_project)
        self._button(f, "把当前闭环路径写回工程", self._save_project_workflow)

    def _load_project(self):
        from scenepaste.project import ScenePasteProject
        try:
            project = ScenePasteProject.load(Path(self.project_path.text()))
        except Exception as exc:
            QMessageBox.warning(self, "工程加载失败", str(exc))
            return
        if project.real_dataset:
            self.profile_dataset.setText(str(project.real_dataset))
            self.real_ds.setText(str(project.real_dataset))
        if project.validation_dataset:
            self.hm_dataset.setText(str(project.validation_dataset))
        if project.predictions_dir:
            self.hm_pred.setText(str(project.predictions_dir))
        if project.output_dir:
            for row in (self.qa_dataset, self.synth_ds, self.cur_dataset, self.shard_dataset):
                row.setText(str(project.output_dir))
        self.status.setText(f"已加载工程：{project.name}")

    def _save_project_workflow(self):
        from scenepaste.project import ScenePasteProject
        try:
            project = ScenePasteProject.load(Path(self.project_path.text()))
        except Exception as exc:
            QMessageBox.warning(self, "工程加载失败", str(exc))
            return
        project.real_dataset = Path(self.real_ds.text()).resolve() if self.real_ds.text() else None
        project.validation_dataset = Path(self.hm_dataset.text()).resolve() if self.hm_dataset.text() else None
        project.predictions_dir = Path(self.hm_pred.text()).resolve() if self.hm_pred.text() else None
        if self.qa_dataset.text():
            project.output_dir = Path(self.qa_dataset.text()).resolve()
        if self.profile_out.text():
            project.distribution_profile = Path(self.profile_out.text()).resolve()
        project.save()
        self.status.setText("闭环路径已写回 Project Manifest")

    def _build_profile(self, root):
        f = self._tab("分布 Profile")
        self.profile_dataset = PathRow(self, root)
        f.addRow("真实数据集:", self.profile_dataset.widget)
        self.profile_out = PathRow(
            self, "distribution_profile.json", directory=False, filter_text="JSON (*.json)",
        )
        f.addRow("输出 Profile:", self.profile_out.widget)
        self.profile_bins = QSpinBox()
        self.profile_bins.setRange(4, 128)
        self.profile_bins.setValue(20)
        f.addRow("Bins:", self.profile_bins)
        self.profile_geom = QComboBox()
        self.profile_geom.addItems(["auto", "detect", "seg", "obb"])
        f.addRow("几何来源:", self.profile_geom)
        self._button(f, "学习真实分布", self._profile_learn)

        self.profile_show = PathRow(
            self, "distribution_profile.json", directory=False, filter_text="JSON (*.json)",
        )
        f.addRow("查看 Profile:", self.profile_show.widget)
        self._button(f, "显示 Profile", self._profile_show)

        self.profile_mix_a = PathRow(self, "", directory=False, filter_text="JSON (*.json)")
        self.profile_mix_b = PathRow(self, "", directory=False, filter_text="JSON (*.json)")
        self.profile_mix_out = PathRow(
            self, "mixed_distribution_profile.json", directory=False, filter_text="JSON (*.json)",
        )
        self.profile_mix_w = QLineEdit("0.5 0.5")
        f.addRow("混合 Profile A:", self.profile_mix_a.widget)
        f.addRow("混合 Profile B:", self.profile_mix_b.widget)
        f.addRow("权重:", self.profile_mix_w)
        f.addRow("混合输出:", self.profile_mix_out.widget)
        self._button(f, "混合 Profile", self._profile_mix)

    def _profile_learn(self):
        self._run([
            "profile", "learn", self.profile_dataset.text(),
            "-o", self.profile_out.text(),
            "--bins", str(self.profile_bins.value()),
            "--geometry-source", self.profile_geom.currentText(),
        ])

    def _profile_show(self):
        self._run(["profile", "show", self.profile_show.text()])

    def _profile_mix(self):
        weights = self.profile_mix_w.text().strip().split()
        args = [
            "profile", "mix",
            self.profile_mix_a.text(), self.profile_mix_b.text(),
            "-o", self.profile_mix_out.text(),
        ]
        if weights:
            args += ["-w"] + weights
        self._run(args)

    def _build_hardmine(self, root):
        f = self._tab("Hard Mining")
        self.hm_dataset = PathRow(self, root)
        f.addRow("验证数据集:", self.hm_dataset.widget)
        self.hm_pred = PathRow(self, None)
        f.addRow("预测 labels:", self.hm_pred.widget)
        self.hm_task = QComboBox()
        self.hm_task.addItems(["detect", "seg", "obb"])
        f.addRow("任务:", self.hm_task)
        self.hm_split = QComboBox()
        self.hm_split.addItems(["val", "train", "test"])
        f.addRow("Split:", self.hm_split)
        self.hm_top = QSpinBox()
        self.hm_top.setRange(1, 100_000)
        self.hm_top.setValue(200)
        f.addRow("Top:", self.hm_top)
        self.hm_match_iou = QDoubleSpinBox()
        self.hm_match_iou.setRange(0, 1)
        self.hm_match_iou.setDecimals(3)
        self.hm_match_iou.setValue(0.5)
        f.addRow("Match IoU:", self.hm_match_iou)
        self.hm_hard_conf = QDoubleSpinBox()
        self.hm_hard_conf.setRange(0, 1)
        self.hm_hard_conf.setDecimals(3)
        self.hm_hard_conf.setValue(0.5)
        f.addRow("Hard conf:", self.hm_hard_conf)
        self.hm_fp_conf = QDoubleSpinBox()
        self.hm_fp_conf.setRange(0, 1)
        self.hm_fp_conf.setDecimals(3)
        self.hm_fp_conf.setValue(0.25)
        f.addRow("FP conf:", self.hm_fp_conf)
        self.hm_loc_iou = QDoubleSpinBox()
        self.hm_loc_iou.setRange(0, 1)
        self.hm_loc_iou.setDecimals(3)
        self.hm_loc_iou.setValue(0.75)
        f.addRow("Localization IoU:", self.hm_loc_iou)
        self.hm_no_profile = QCheckBox("不写 Hard Profile")
        f.addRow("", self.hm_no_profile)
        self.hm_out = PathRow(self, "hardmine", directory=True)
        f.addRow("输出目录:", self.hm_out.widget)
        self._button(f, "分析难例并生成 Hard Profile", self._hardmine)

    def _build_quality(self, root):
        f = self._tab("QA / 泄漏")
        self.qa_dataset = PathRow(self, root)
        f.addRow("数据集:", self.qa_dataset.widget)
        self.embed_backend = QComboBox()
        self.embed_backend.addItems(["cv-lite-v1", "clip", "dinov2"])
        f.addRow("Embedding:", self.embed_backend)
        self.qa_html = PathRow(self, "", directory=False, filter_text="HTML (*.html)")
        f.addRow("QA HTML:", self.qa_html.widget)
        self.qa_json = PathRow(self, "", directory=False, filter_text="JSON (*.json)")
        f.addRow("QA JSON:", self.qa_json.widget)
        self.qa_target_profile = PathRow(self, "", directory=False, filter_text="JSON (*.json)")
        f.addRow("Target Profile:", self.qa_target_profile.widget)
        self.qa_dup_limit = QSpinBox()
        self.qa_dup_limit.setRange(0, 10_000_000)
        self.qa_dup_limit.setValue(5000)
        f.addRow("Duplicate limit:", self.qa_dup_limit)
        self.qa_near_th = QSpinBox()
        self.qa_near_th.setRange(0, 64)
        self.qa_near_th.setValue(6)
        f.addRow("Near-dup threshold:", self.qa_near_th)
        self.qa_embed_limit = QSpinBox()
        self.qa_embed_limit.setRange(0, 10_000_000)
        self.qa_embed_limit.setValue(500)
        f.addRow("Embedding limit:", self.qa_embed_limit)
        self.qa_leak_th = QDoubleSpinBox()
        self.qa_leak_th.setRange(0, 1)
        self.qa_leak_th.setDecimals(4)
        self.qa_leak_th.setValue(0.995)
        f.addRow("Leakage embed threshold:", self.qa_leak_th)
        self._button(f, "生成 QA Dashboard", self._qa)

        self.leak_out = PathRow(self, "", directory=True)
        f.addRow("泄漏报告目录:", self.leak_out.widget)
        self.leak_phash = QSpinBox()
        self.leak_phash.setRange(0, 64)
        self.leak_phash.setValue(6)
        f.addRow("pHash threshold:", self.leak_phash)
        self.leak_embed_th = QDoubleSpinBox()
        self.leak_embed_th.setRange(0, 1)
        self.leak_embed_th.setDecimals(4)
        self.leak_embed_th.setValue(0.995)
        f.addRow("Embedding threshold:", self.leak_embed_th)
        self.leak_embed_limit = QSpinBox()
        self.leak_embed_limit.setRange(0, 10_000_000)
        self.leak_embed_limit.setValue(1000)
        f.addRow("Leakage embed limit:", self.leak_embed_limit)
        self._button(f, "检查 train/val/test 泄漏", self._leakage)

    def _build_compare(self, root):
        f = self._tab("真实 vs 合成")
        self.real_ds = PathRow(self, None)
        f.addRow("真实数据集:", self.real_ds.widget)
        self.synth_ds = PathRow(self, root)
        f.addRow("合成数据集:", self.synth_ds.widget)
        self.compare_out = PathRow(self, "comparison", directory=True)
        f.addRow("报告目录:", self.compare_out.widget)
        self.compare_backend = QComboBox()
        self.compare_backend.addItems(["cv-lite-v1", "clip", "dinov2"])
        f.addRow("Embedding:", self.compare_backend)
        self.compare_embed_limit = QSpinBox()
        self.compare_embed_limit.setRange(0, 10_000_000)
        self.compare_embed_limit.setValue(1000)
        f.addRow("Embedding limit:", self.compare_embed_limit)
        self._button(f, "生成对比 Dashboard", self._compare)

    def _build_curation(self, root):
        f = self._tab("多样性策展")
        self.cur_dataset = PathRow(self, root)
        f.addRow("数据集:", self.cur_dataset.widget)
        self.cur_count = QSpinBox()
        self.cur_count.setRange(1, 10_000_000)
        self.cur_count.setValue(1000)
        f.addRow("选择代表样本:", self.cur_count)
        self.cur_limit = QSpinBox()
        self.cur_limit.setRange(0, 10_000_000)
        self.cur_limit.setValue(1000)
        f.addRow("扫描上限:", self.cur_limit)
        self.cur_out = PathRow(self, "diverse_subset", directory=True)
        f.addRow("导出数据集:", self.cur_out.widget)
        self.cur_report = PathRow(self, "", directory=True)
        f.addRow("报告目录:", self.cur_report.widget)
        self.cur_backend = QComboBox()
        self.cur_backend.addItems(["cv-lite-v1", "clip", "dinov2"])
        f.addRow("Embedding:", self.cur_backend)
        self.cur_copy = PathRow(self, "", directory=True)
        f.addRow("仅复制选中图:", self.cur_copy.widget)
        self._button(f, "选择并导出高多样性子集", self._diversity)

    def _build_publish(self, root, output_root):
        f = self._tab("发布 / Sharding")
        self.shard_dataset = PathRow(self, root)
        f.addRow("数据集:", self.shard_dataset.widget)
        self.shard_out = PathRow(self, output_root or "shards", directory=True)
        f.addRow("Shard 输出:", self.shard_out.widget)
        self.shard_split = QComboBox()
        self.shard_split.addItems(["train", "val", "test"])
        f.addRow("Split:", self.shard_split)
        self.shard_size = QSpinBox()
        self.shard_size.setRange(1, 1_000_000)
        self.shard_size.setValue(10000)
        f.addRow("每 shard 样本数:", self.shard_size)
        self.shard_bytes = QSpinBox()
        self.shard_bytes.setRange(0, 2_000_000_000)
        self.shard_bytes.setValue(0)
        self.shard_bytes.setSpecialValueText("不限制")
        f.addRow("每 shard 字节上限:", self.shard_bytes)
        self._button(f, "构建 WebDataset shards", self._shard)

    def _hardmine(self):
        args = [
            "curate", "hardmine", self.hm_dataset.text(),
            "--predictions", self.hm_pred.text(),
            "--task", self.hm_task.currentText(),
            "--split", self.hm_split.currentText(),
            "--top", str(self.hm_top.value()),
            "--match-iou", str(self.hm_match_iou.value()),
            "--hard-conf", str(self.hm_hard_conf.value()),
            "--fp-conf", str(self.hm_fp_conf.value()),
            "--localization-iou", str(self.hm_loc_iou.value()),
            "-o", self.hm_out.text(),
        ]
        if self.hm_no_profile.isChecked():
            args.append("--no-profile")
        self._run(args)

    def _qa(self):
        args = [
            "qa", self.qa_dataset.text(),
            "--embedding-backend", self.embed_backend.currentText(),
            "--duplicate-limit", str(self.qa_dup_limit.value()),
            "--near-duplicate-threshold", str(self.qa_near_th.value()),
            "--embedding-limit", str(self.qa_embed_limit.value()),
            "--leakage-embedding-threshold", str(self.qa_leak_th.value()),
        ]
        if self.qa_html.text():
            args += ["--html", self.qa_html.text()]
        if self.qa_json.text():
            args += ["--json", self.qa_json.text()]
        if self.qa_target_profile.text():
            args += ["--target-profile", self.qa_target_profile.text()]
        self._run(args)

    def _leakage(self):
        args = [
            "curate", "leakage", self.qa_dataset.text(),
            "--embedding-backend", self.embed_backend.currentText(),
            "--phash-threshold", str(self.leak_phash.value()),
            "--embedding-threshold", str(self.leak_embed_th.value()),
            "--embedding-limit", str(self.leak_embed_limit.value()),
        ]
        if self.leak_out.text():
            args += ["-o", self.leak_out.text()]
        self._run(args)

    def _compare(self):
        self._run([
            "compare", self.real_ds.text(), self.synth_ds.text(),
            "--embedding-backend", self.compare_backend.currentText(),
            "--embedding-limit", str(self.compare_embed_limit.value()),
            "-o", self.compare_out.text(),
        ])

    def _diversity(self):
        args = [
            "curate", "diversity", self.cur_dataset.text(),
            "--select", str(self.cur_count.value()),
            "--limit", str(self.cur_limit.value()),
            "--embedding-backend", self.cur_backend.currentText(),
            "--export-dataset", self.cur_out.text(),
        ]
        if self.cur_report.text():
            args += ["-o", self.cur_report.text()]
        if self.cur_copy.text():
            args += ["--copy-selected", self.cur_copy.text()]
        self._run(args)

    def _shard(self):
        args = [
            "shard", self.shard_dataset.text(),
            "-o", self.shard_out.text(),
            "--split", self.shard_split.currentText(),
            "--max-samples", str(self.shard_size.value()),
        ]
        if self.shard_bytes.value() > 0:
            args += ["--max-bytes", str(self.shard_bytes.value())]
        self._run(args)

    def _run(self, args):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "任务进行中", "请先等待当前任务结束或中止。")
            return
        if any(x == "" for x in args):
            QMessageBox.warning(self, "参数不完整", "请先填写必要路径。")
            return
        self._current_args = list(args)
        cmd = ["-m", "scenepaste"] + self._current_args
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
        args = getattr(self, "_current_args", [])
        quality_warning = code == 1 and (
            (args and args[0] == "qa")
            or (len(args) >= 2 and args[0] == "curate" and args[1] == "leakage")
        )
        if code == 0:
            self.status.setText("完成")
        elif quality_warning:
            self.status.setText("完成（发现 QA / 泄漏警告，请查看报告）")
        else:
            self.status.setText(f"退出 code={code}")

    def stop(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()


def launch_data_loop_center(dataset_root=None, output_root=None) -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dlg = DataLoopCenterDialog(dataset_root=dataset_root, output_root=output_root)
    dlg.show()
    return int(app.exec())
