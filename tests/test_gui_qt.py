"""Headless Qt GUI smoke tests using the offscreen QPA platform.

These exercise the new PySide6 ``compose_app_qt`` package end-to-end on a
virtual display. They cover the core flow only (load → place → save);
deeper coverage will follow as the Qt port matures.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Skip the entire module when PySide6 isn't installed or no display is
# available. CI sets QT_QPA_PLATFORM=offscreen, so this works in headless
# environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pyside6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OBJECTS = REPO_ROOT / "samples" / "objects"
SAMPLE_BACKGROUNDS = REPO_ROOT / "samples" / "backgrounds"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _drain(qapp, iterations: int = 40, delay: float = 0.03) -> None:
    """Process Qt events until the worker thread has had time to finish."""
    for _ in range(iterations):
        qapp.processEvents()
        time.sleep(delay)


def test_main_window_constructs_and_loads(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow

    win = MainWindow(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    assert len(win.doc.cutouts) == 3
    assert len(win.doc.background_paths) == 4
    assert win.doc.bg_size[0] > 0


def test_place_instance_and_save(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow

    win = MainWindow(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    win._place_at_centre(0)
    assert len(win.doc.instances) == 1
    win.save_current()

    jpgs = list((tmp_path / "images" / "train").glob("*.jpg"))
    labels = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(jpgs) == 1
    assert len(labels) == 1
    # Label should be a valid YOLO detect line.
    tokens = labels[0].read_text(encoding="utf-8").split()
    assert len(tokens) == 5
    cls = int(tokens[0])
    assert 0 <= cls <= 2


def test_undo_redo_after_delete(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow

    win = MainWindow(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    win._place_at_centre(0)
    assert len(win.doc.instances) == 1
    win.delete_selected()
    assert len(win.doc.instances) == 0
    win.undo()
    assert len(win.doc.instances) == 1
    win.redo()
    assert len(win.doc.instances) == 0


def test_transform_persists_in_model(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow

    win = MainWindow(
        objects_dir=SAMPLE_OBJECTS,
        backgrounds_dir=SAMPLE_BACKGROUNDS,
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    win._place_at_centre(0)
    inst = win.doc.selected()
    assert inst is not None
    win.controls.scale_spin.setValue(0.25)
    assert abs(win.doc.selected().h_ratio - 0.25) < 1e-3
    win.controls.angle_spin.setValue(45.0)
    assert abs(win.doc.selected().angle - 45.0) < 1e-3
