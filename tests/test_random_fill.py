"""Tests for MainWindow.random_fill (Qt port)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _drain(qapp, iterations: int = 40, delay: float = 0.03) -> None:
    for _ in range(iterations):
        qapp.processEvents()
        time.sleep(delay)


def _ready_window(qapp, tmp_path: Path):
    from compose_app_qt.app import MainWindow
    win = MainWindow(
        objects_dir=REPO_ROOT / "samples" / "objects",
        backgrounds_dir=REPO_ROOT / "samples" / "backgrounds",
        output_dir=tmp_path,
        class_map_text="person=0,motorcycle=1,truck=2",
    )
    _drain(qapp)
    return win


def test_random_fill_places_between_1_and_3(qapp, tmp_path: Path):
    win = _ready_window(qapp, tmp_path)
    win.random_fill()
    assert 1 <= len(win.doc.instances) <= 3
    # Each instance grounded: cy / bh is in the bottom-half band.
    bw, bh = win.doc.bg_size
    for inst in win.doc.instances:
        rendered_h = max(4, int(round(inst.h_ratio * bh)))
        bottom_y = inst.cy + rendered_h / 2.0
        assert 0.50 <= (bottom_y / bh) <= 0.95


def test_random_fill_is_undoable(qapp, tmp_path: Path):
    win = _ready_window(qapp, tmp_path)
    win.random_fill()
    n = len(win.doc.instances)
    assert n >= 1
    win.undo()
    assert len(win.doc.instances) == 0
    win.redo()
    assert len(win.doc.instances) == n


def test_random_fill_warns_without_cutouts(qapp, tmp_path: Path, monkeypatch):
    """Empty cutouts → warning shown (mocked) → no instances added."""
    from compose_app_qt.app import MainWindow
    from PySide6.QtWidgets import QMessageBox

    # Stub QMessageBox.information so it doesn't block in headless mode.
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)

    win = MainWindow(
        objects_dir=Path("/nonexistent"),
        backgrounds_dir=REPO_ROOT / "samples" / "backgrounds",
        output_dir=tmp_path,
        class_map_text="person=0",
    )
    _drain(qapp)
    win.random_fill()
    assert win.doc.instances == []
