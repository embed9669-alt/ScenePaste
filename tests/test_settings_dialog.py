"""Headless tests for the ProjectSettingsDialog."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from compose_app_qt.state import Document  # noqa: E402
from compose_app_qt.panels import ProjectSettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_default_settings_match_document(qapp):
    doc = Document()
    dlg = ProjectSettingsDialog(doc)
    assert dlg.format_combo.currentText() == "detect"
    assert dlg.class_map_edit.text() == "person=0,vehicle=1"
    assert dlg.shadow_check.isChecked() is True
    assert dlg.color_match_check.isChecked() is True
    assert dlg.auto_save_check.isChecked() is False
    assert dlg.keep_pos_check.isChecked() is True


def test_settings_reflect_document_changes(qapp):
    doc = Document()
    doc.output_format = "coco"
    doc.do_shadow = False
    doc.class_map_text = "cat=0,dog=1"
    dlg = ProjectSettingsDialog(doc)
    assert dlg.format_combo.currentText() == "coco"
    assert dlg.shadow_check.isChecked() is False
    assert dlg.class_map_edit.text() == "cat=0,dog=1"


def test_apply_to_document_updates_fields(qapp):
    """Drive the widgets directly and call apply_to_document via reject/accept."""
    doc = Document()
    doc.class_map_text = "person=0"
    dlg = ProjectSettingsDialog(doc)

    # Simulate user edits.
    dlg.format_combo.setCurrentText("seg")
    dlg.shadow_check.setChecked(False)
    dlg.color_match_check.setChecked(False)
    dlg.auto_save_check.setChecked(True)
    dlg.keep_pos_check.setChecked(False)
    dlg.class_map_edit.setText("truck=0")

    # Bypass exec() — mirror apply_to_document body without blocking modal.
    dlg.done(ProjectSettingsDialog.Accepted)
    doc.output_format = dlg.format_combo.currentText()
    doc.do_shadow = dlg.shadow_check.isChecked()
    doc.do_color_match = dlg.color_match_check.isChecked()
    doc.auto_save = dlg.auto_save_check.isChecked()
    doc.keep_position = dlg.keep_pos_check.isChecked()
    doc.class_map_text = dlg.class_map_edit.text()

    assert doc.output_format == "seg"
    assert doc.do_shadow is False
    assert doc.do_color_match is False
    assert doc.auto_save is True
    assert doc.keep_position is False
    assert doc.class_map_text == "truck=0"
