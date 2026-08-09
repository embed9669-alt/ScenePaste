"""Unified ScenePaste data-loop center.

A thin Qt orchestration layer over the tested CLI. Long-running operations use
QProcess, keeping the editor responsive and ensuring CLI/GUI behavior stays in
sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)


class _PathRow:
    def __init__(self, parent, value="", directory=True, filter_text="All files (*)"):
        self.edit = QLineEdit(str(value or "")); self.button = QPushButton("选择…")
        self.widget = QWidget(); row = QHBoxLayout(self.widget); row.setContentsMargins(0,0,0,0)
        row.addWidget(self.edit, 1); row.addWidget(self.button)
        def choose():
            if directory:
                p = QFileDialog.getExistingDirectory(parent, "选择目录", self.edit.text() or str(Path.cwd()))
            else:
                p, _ = QFileDialog.getOpenFileName(parent, "选择文件", self.edit.text() or str(Path.cwd()), filter_text)
            if p: self.edit.setText(p)
        self.button.clicked.connect(choose)

    def text(self): return self.edit.text().strip()


class DataLoopCenterDialog(QDialog):
    def __init__(self, dataset_root=None, output_root=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ScenePaste — 数据闭环中心")
        self.resize(900, 720)
        self.process = QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output); self.process.finished.connect(self._finished)
        layout = QVBoxLayout(self)
        title = QLabel("模型 → 难例 → 针对性生成 → QA/泄漏 → 真实/合成对比 → 数据策展/发布")
        title.setWordWrap(True); layout.addWidget(title)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self._build_project(); self._build_profile(dataset_root); self._build_hardmine(dataset_root); self._build_quality(dataset_root)
        self._build_compare(dataset_root); self._build_curation(dataset_root); self._build_publish(dataset_root, output_root)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(3000); layout.addWidget(self.log, 1)
        row = QHBoxLayout(); self.status = QLabel("就绪"); row.addWidget(self.status, 1)
        self.stop_btn = QPushButton("中止"); self.stop_btn.clicked.connect(self.stop); self.stop_btn.setEnabled(False); row.addWidget(self.stop_btn)
        close = QPushButton("关闭"); close.clicked.connect(self.close); row.addWidget(close); layout.addLayout(row)

    def _tab(self, name):
        w=QWidget(); f=QFormLayout(w); self.tabs.addTab(w,name); return f

    def _button(self, form, text, callback):
        b=QPushButton(text); b.clicked.connect(callback); form.addRow("", b); return b

    def _build_project(self):
        f=self._tab("工程")
        self.project_path=_PathRow(self,"scenepaste.project.json",directory=False,filter_text="ScenePaste Project (*.json)"); f.addRow("Project Manifest:",self.project_path.widget)
        self._button(f,"加载工程并填充工作流",self._load_project)
        self._button(f,"把当前闭环路径写回工程",self._save_project_workflow)

    def _load_project(self):
        from scenepaste.project import ScenePasteProject
        try: project=ScenePasteProject.load(Path(self.project_path.text()))
        except Exception as exc: QMessageBox.warning(self,"工程加载失败",str(exc)); return
        if project.real_dataset:
            self.profile_dataset.edit.setText(str(project.real_dataset)); self.real_ds.edit.setText(str(project.real_dataset))
        if project.validation_dataset: self.hm_dataset.edit.setText(str(project.validation_dataset))
        if project.predictions_dir: self.hm_pred.edit.setText(str(project.predictions_dir))
        if project.output_dir:
            for row in (self.qa_dataset,self.synth_ds,self.cur_dataset,self.shard_dataset): row.edit.setText(str(project.output_dir))
        self.status.setText(f"已加载工程：{project.name}")

    def _save_project_workflow(self):
        from scenepaste.project import ScenePasteProject
        try: project=ScenePasteProject.load(Path(self.project_path.text()))
        except Exception as exc: QMessageBox.warning(self,"工程加载失败",str(exc)); return
        project.real_dataset=Path(self.real_ds.text()).resolve() if self.real_ds.text() else None
        project.validation_dataset=Path(self.hm_dataset.text()).resolve() if self.hm_dataset.text() else None
        project.predictions_dir=Path(self.hm_pred.text()).resolve() if self.hm_pred.text() else None
        if self.qa_dataset.text():
            project.output_dir=Path(self.qa_dataset.text()).resolve()
        if self.profile_out.text():
            project.distribution_profile=Path(self.profile_out.text()).resolve()
        project.save()
        self.status.setText("闭环路径已写回 Project Manifest")

    def _build_profile(self, root):
        f=self._tab("分布 Profile")
        self.profile_dataset=_PathRow(self,root); f.addRow("真实数据集:",self.profile_dataset.widget)
        self.profile_out=_PathRow(self,"distribution_profile.json",directory=False,filter_text="JSON (*.json)"); f.addRow("输出 Profile:",self.profile_out.widget)
        self._button(f,"学习真实分布",lambda:self._run(["profile","learn",self.profile_dataset.text(),"-o",self.profile_out.text()]))

    def _build_hardmine(self, root):
        f=self._tab("Hard Mining")
        self.hm_dataset=_PathRow(self,root); f.addRow("验证数据集:",self.hm_dataset.widget)
        self.hm_pred=_PathRow(self,None); f.addRow("预测 labels:",self.hm_pred.widget)
        self.hm_task=QComboBox(); self.hm_task.addItems(["detect","seg","obb"]); f.addRow("任务:",self.hm_task)
        self.hm_split=QComboBox(); self.hm_split.addItems(["val","train","test"]); f.addRow("Split:",self.hm_split)
        self.hm_out=_PathRow(self,"hardmine",directory=True); f.addRow("输出目录:",self.hm_out.widget)
        self._button(f,"分析难例并生成 Hard Profile",self._hardmine)

    def _build_quality(self, root):
        f=self._tab("QA / 泄漏")
        self.qa_dataset=_PathRow(self,root); f.addRow("数据集:",self.qa_dataset.widget)
        self.embed_backend=QComboBox(); self.embed_backend.addItems(["cv-lite-v1","clip","dinov2"]); f.addRow("Embedding:",self.embed_backend)
        self._button(f,"生成 QA Dashboard",self._qa)
        self._button(f,"检查 train/val/test 泄漏",self._leakage)

    def _build_compare(self, root):
        f=self._tab("真实 vs 合成")
        self.real_ds=_PathRow(self,None); f.addRow("真实数据集:",self.real_ds.widget)
        self.synth_ds=_PathRow(self,root); f.addRow("合成数据集:",self.synth_ds.widget)
        self.compare_out=_PathRow(self,"comparison",directory=True); f.addRow("报告目录:",self.compare_out.widget)
        self.compare_backend=QComboBox(); self.compare_backend.addItems(["cv-lite-v1","clip","dinov2"]); f.addRow("Embedding:",self.compare_backend)
        self._button(f,"生成对比 Dashboard",self._compare)

    def _build_curation(self, root):
        f=self._tab("多样性策展")
        self.cur_dataset=_PathRow(self,root); f.addRow("数据集:",self.cur_dataset.widget)
        self.cur_count=QSpinBox(); self.cur_count.setRange(1,10_000_000); self.cur_count.setValue(1000); f.addRow("选择代表样本:",self.cur_count)
        self.cur_out=_PathRow(self,"diverse_subset",directory=True); f.addRow("导出数据集:",self.cur_out.widget)
        self.cur_backend=QComboBox(); self.cur_backend.addItems(["cv-lite-v1","clip","dinov2"]); f.addRow("Embedding:",self.cur_backend)
        self._button(f,"选择并导出高多样性子集",self._diversity)

    def _build_publish(self, root, output_root):
        f=self._tab("发布 / Sharding")
        self.shard_dataset=_PathRow(self,root); f.addRow("数据集:",self.shard_dataset.widget)
        self.shard_out=_PathRow(self,output_root or "shards",directory=True); f.addRow("Shard 输出:",self.shard_out.widget)
        self.shard_size=QSpinBox(); self.shard_size.setRange(1,1_000_000); self.shard_size.setValue(10000); f.addRow("每 shard 样本数:",self.shard_size)
        self._button(f,"构建 WebDataset shards",self._shard)

    def _hardmine(self):
        self._run(["curate","hardmine",self.hm_dataset.text(),"--predictions",self.hm_pred.text(),"--task",self.hm_task.currentText(),"--split",self.hm_split.currentText(),"-o",self.hm_out.text()])
    def _qa(self):
        self._run(["qa",self.qa_dataset.text(),"--embedding-backend",self.embed_backend.currentText()])
    def _leakage(self):
        self._run(["curate","leakage",self.qa_dataset.text(),"--embedding-backend",self.embed_backend.currentText()])
    def _compare(self):
        self._run(["compare",self.real_ds.text(),self.synth_ds.text(),"--embedding-backend",self.compare_backend.currentText(),"-o",self.compare_out.text()])
    def _diversity(self):
        self._run(["curate","diversity",self.cur_dataset.text(),"--select",str(self.cur_count.value()),"--embedding-backend",self.cur_backend.currentText(),"--export-dataset",self.cur_out.text()])
    def _shard(self):
        self._run(["shard",self.shard_dataset.text(),"-o",self.shard_out.text(),"--max-samples",str(self.shard_size.value())])

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
        text=bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace")
        if text:self.log.appendPlainText(text.rstrip())
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
        if self.process.state()!=QProcess.NotRunning:self.process.terminate()


def launch_data_loop_center(dataset_root=None, output_root=None) -> int:
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dlg = DataLoopCenterDialog(dataset_root=dataset_root, output_root=output_root)
    dlg.show()
    return int(app.exec())
