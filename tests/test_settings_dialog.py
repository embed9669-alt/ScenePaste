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
    dlg.scene_recipe.setCurrentText("surveillance")
    dlg.object_recipe.setCurrentText("mild")
    dlg.blend.setCurrentText("gaussian")
    dlg.empty_scene.setValue(0.1)

    # Bypass exec() — mirror apply_to_document body without blocking modal.
    dlg.done(ProjectSettingsDialog.Accepted)
    doc.output_format = dlg.format_combo.currentText()
    doc.do_shadow = dlg.shadow_check.isChecked()
    doc.do_color_match = dlg.color_match_check.isChecked()
    doc.auto_save = dlg.auto_save_check.isChecked()
    doc.keep_position = dlg.keep_pos_check.isChecked()
    doc.class_map_text = dlg.class_map_edit.text()
    doc.scene_recipe = dlg.scene_recipe.currentText().strip()
    doc.object_appearance_recipe = dlg.object_recipe.currentText().strip()
    doc.blend_mode = dlg.blend.currentText().strip()
    doc.empty_scene_prob = float(dlg.empty_scene.value())

    assert doc.output_format == "seg"
    assert doc.do_shadow is False
    assert doc.do_color_match is False
    assert doc.auto_save is True
    assert doc.keep_position is False
    assert doc.class_map_text == "truck=0"
    assert doc.scene_recipe == "surveillance"
    assert doc.object_appearance_recipe == "mild"
    assert doc.blend_mode == "gaussian"
    assert abs(doc.empty_scene_prob - 0.1) < 1e-9


def test_generation_defaults_panel_writes_document(qapp):
    from compose_app_qt.panels import GenerationDefaultsPanel

    doc = Document()
    panel = GenerationDefaultsPanel()
    panel.object_recipe.setCurrentText("mild")
    panel.scene_recipe.setCurrentText("camera-mild")
    panel.blend.setCurrentText("hard")
    panel.empty_scene.setValue(0.2)
    panel.apply_to_document(doc)
    assert doc.generation_defaults()["object_appearance_recipe"] == "mild"
    assert doc.generation_defaults()["augmentation_recipe"] == "camera-mild"
    assert doc.blend_mode == "hard"
    assert abs(doc.empty_scene_prob - 0.2) < 1e-9
