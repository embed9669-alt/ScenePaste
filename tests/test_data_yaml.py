"""data.yaml 输出测试。"""

from __future__ import annotations

from pathlib import Path

import scenepaste.core as core


def test_write_data_yaml_basic(tmp_path: Path):
    out = tmp_path / "dataset"
    out.mkdir()
    yaml_path = core.write_data_yaml(out, {"person": 0, "truck": 1, "moto": 2})
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    # 关键字段
    assert "path: " in text
    assert "train: images/train" in text
    assert "val: images/train" in text  # 默认 val=train（自验证）
    assert "names:" in text
    assert "0: person" in text
    assert "1: truck" in text
    assert "2: moto" in text


def test_write_data_yaml_custom_val(tmp_path: Path):
    out = tmp_path / "dataset"
    out.mkdir()
    yaml_path = core.write_data_yaml(out, {"person": 0}, val_split="images/val")
    text = yaml_path.read_text(encoding="utf-8")
    assert "val: images/val" in text


def test_write_data_yaml_absolute_path(tmp_path: Path):
    """path 字段必须是绝对路径，YOLO 训练时才不会因 cwd 失败。"""
    out = tmp_path / "ds"
    out.mkdir()
    yaml_path = core.write_data_yaml(out, {"person": 0})
    text = yaml_path.read_text(encoding="utf-8")
    assert text.startswith(f"path: {out.resolve().as_posix()}")
