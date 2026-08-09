from pathlib import Path
import zipfile

from scripts.build_release import FILE_EXCLUDE_NAMES, build, verify_zip


def test_release_builder_excludes_repository_artifacts(tmp_path: Path):
    output = tmp_path / "ScenePaste.zip"
    build(output)
    verify_zip(output)
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert any(name.endswith("/README.md") for name in names)
    assert any("/scenepaste/explorer.py" in name for name in names)
    assert any("/scenepaste/resources/samples/objects/sample_person.json" in name for name in names)
    assert any("/scripts/__init__.py" in name for name in names)
    assert all("/.git/" not in name for name in names)
    assert all("__pycache__" not in name for name in names)
    assert all(".egg-info" not in name for name in names)
    assert all("/generated/" not in name for name in names)


def test_release_builder_excludes_local_project_manifest():
    assert "scenepaste.project.json" in FILE_EXCLUDE_NAMES
