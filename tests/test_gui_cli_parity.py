"""GUI CLI-parity smoke tests for auto-cutout load and large-generate args."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from compose_app_qt.large_generate import LargeGenerationDialog
from compose_app_qt.workers import LoadCutoutsWorker
from scenepaste.core.models import ObjectAsset


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_asset(label: str = "auto", class_id: int = 0) -> ObjectAsset:
    h, w = 40, 30
    image = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.ones((h, w), dtype=np.float32)
    return ObjectAsset(
        label=label,
        class_id=class_id,
        image=image,
        alpha=alpha,
        source_json=Path("fake.jpg"),
        source_shape_index=0,
        polygon=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32),
    )


def test_load_cutouts_worker_auto_cutout_path(qapp, tmp_path):
    objects = tmp_path / "raw"
    objects.mkdir()
    (objects / "a.jpg").write_bytes(b"not-a-real-image")

    result = {}

    def _on_ready(cutouts, class_map_text):
        result["cutouts"] = cutouts
        result["class_map_text"] = class_map_text

    def _on_fail(msg):
        result["error"] = msg

    worker = LoadCutoutsWorker(
        objects, "person=0", auto_cutout=True, auto_cutout_label="person",
    )
    worker.cutouts_ready.connect(_on_ready)
    worker.failed.connect(_on_fail)

    with patch(
        "compose_app_qt.workers.core.load_object_assets_auto",
    ) as mocked:
        def _side_effect(path, class_map, log, label=None, label_from_subdir=False):
            assert label == "person"
            class_map.setdefault("person", 0)
            log("自动抠图完成：1 / 1")
            return [_fake_asset(label="person", class_id=0)]

        mocked.side_effect = _side_effect
        worker.run()

    assert "error" not in result, result.get("error")
    assert len(result["cutouts"]) == 1
    assert result["cutouts"][0].label == "person"
    assert "person=" in result["class_map_text"]
    mocked.assert_called_once()


def test_auto_cutout_label_from_subdir_helper(tmp_path):
    from scenepaste.core.auto_cutout import _label_for_path

    root = tmp_path / "objects"
    person = root / "person"
    person.mkdir(parents=True)
    img = person / "a.jpg"
    img.write_bytes(b"x")
    assert _label_for_path(img, root, default_label="auto", label_from_subdir=True) == "person"
    flat = root / "b.jpg"
    flat.write_bytes(b"x")
    assert _label_for_path(flat, root, default_label="vehicle", label_from_subdir=True) == "vehicle"


def test_large_generation_args_include_auto_cutout_label(qapp, tmp_path):
    dlg = LargeGenerationDialog(
        objects_dir=tmp_path / "obj",
        backgrounds_dir=tmp_path / "bg",
        output_dir=tmp_path / "out",
        class_map_text="person=0",
    )
    dlg.objects.setText(str(tmp_path / "obj"))
    dlg.backgrounds.setText(str(tmp_path / "bg"))
    dlg.output.setText(str(tmp_path / "out"))
    dlg.auto_cutout.setChecked(True)
    dlg.auto_cutout_label.setCurrentText("person")
    dlg.auto_cutout_subdir.setChecked(True)
    args = dlg._args()
    assert "--auto-cutout" in args
    assert args[args.index("--auto-cutout-label") + 1] == "person"
    assert "--auto-cutout-label-from-subdir" in args
    dlg.close()


def test_load_cutouts_worker_auto_cutout_missing_rembg(qapp, tmp_path):
    objects = tmp_path / "raw"
    objects.mkdir()
    result = {}
    worker = LoadCutoutsWorker(objects, "auto=0", auto_cutout=True)
    worker.failed.connect(lambda msg: result.setdefault("error", msg))
    worker.cutouts_ready.connect(lambda *_: result.setdefault("ok", True))

    with patch(
        "compose_app_qt.workers.core.load_object_assets_auto",
        side_effect=ImportError("自动抠图需要 rembg"),
    ):
        worker.run()

    assert "error" in result
    assert "rembg" in result["error"].lower() or "自动抠图" in result["error"]


def test_large_generation_args_include_auto_cutout_and_advanced(qapp, tmp_path):
    dlg = LargeGenerationDialog(
        objects_dir=tmp_path / "obj",
        backgrounds_dir=tmp_path / "bg",
        output_dir=tmp_path / "out",
        class_map_text="auto=0",
    )
    dlg.objects.setText(str(tmp_path / "obj"))
    dlg.backgrounds.setText(str(tmp_path / "bg"))
    dlg.output.setText(str(tmp_path / "out"))
    dlg.auto_cutout.setChecked(True)
    dlg.min_objects.setValue(2)
    dlg.max_objects.setValue(4)
    dlg.seed.setValue(123)
    dlg.no_preview.setChecked(True)
    dlg.queue_depth.setValue(8)
    args = dlg._args()
    assert "--auto-cutout" in args
    assert args[args.index("--min-objects") + 1] == "2"
    assert args[args.index("--max-objects") + 1] == "4"
    assert args[args.index("--seed") + 1] == "123"
    assert "--no-preview" in args
    assert args[args.index("--queue-depth") + 1] == "8"
    # Defaults omitted when unchanged:
    dlg2 = LargeGenerationDialog(
        objects_dir=tmp_path / "obj",
        backgrounds_dir=tmp_path / "bg",
        output_dir=tmp_path / "out",
    )
    dlg2.objects.setText(str(tmp_path / "obj"))
    dlg2.backgrounds.setText(str(tmp_path / "bg"))
    dlg2.output.setText(str(tmp_path / "out"))
    sparse = dlg2._args()
    assert "--min-objects" not in sparse
    assert "--seed" not in sparse
    dlg.close()
    dlg2.close()


def test_dataset_tools_analyze_args(qapp, tmp_path):
    from compose_app_qt.dataset_tools import DatasetToolsDialog

    dlg = DatasetToolsDialog(dataset_root=tmp_path)
    dlg.an_paths.setText(f"{tmp_path} {tmp_path / 'b'}")
    dlg.an_json.setChecked(True)
    captured = {}

    def _fake_start(exe, cmd):
        captured["cmd"] = list(cmd)
        return True

    dlg.process.start = _fake_start  # type: ignore[method-assign]
    dlg._analyze()
    assert captured["cmd"][:3] == ["-m", "scenepaste", "analyze"]
    assert "--json" in captured["cmd"]
    dlg.close()


def test_data_loop_hardmine_includes_top(qapp, tmp_path):
    from compose_app_qt.data_loop import DataLoopCenterDialog

    dlg = DataLoopCenterDialog(dataset_root=tmp_path, output_root=tmp_path / "shards")
    dlg.hm_dataset.setText(str(tmp_path))
    dlg.hm_pred.setText(str(tmp_path / "pred"))
    dlg.hm_out.setText(str(tmp_path / "hard"))
    dlg.hm_top.setValue(50)
    captured = {}

    def _fake_start(exe, cmd):
        captured["cmd"] = list(cmd)
        return True

    dlg.process.start = _fake_start  # type: ignore[method-assign]
    dlg._hardmine()
    assert "--top" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--top") + 1] == "50"
    dlg.close()


def test_mainwindow_has_menus(qapp):
    from compose_app_qt.app import MainWindow

    win = MainWindow(theme_mode="dark")
    names = [a.text() for a in win.menuBar().actions()]
    assert "文件" in names
    assert "合成" in names
    assert "数据" in names
    assert win.act_auto_cutout is not None
    win.close()
