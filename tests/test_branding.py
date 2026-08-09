from pathlib import Path

import scenepaste
from scenepaste import GenerationConfig, generate_dataset
from scenepaste.cli import main as cli_main


def test_public_api_version_and_exports():
    assert scenepaste.__version__ == "1.0.0"
    assert GenerationConfig is not None
    assert callable(generate_dataset)


def test_unified_cli_help_and_version(capsys):
    assert cli_main(["--version"]) == 0
    assert "ScenePaste" in capsys.readouterr().out
    assert cli_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "scenepaste generate" in out
    assert "scenepaste gui" in out


def test_public_docs_use_scenepaste_brand():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README_zh.md", "pyproject.toml"):
        text = (root / name).read_text(encoding="utf-8")
        assert "Copy-Paste Dataset Studio" not in text
        assert "copy-paste-gui" not in text
        assert "copy-paste-cli" not in text


def test_v1_metadata_and_docs_are_release_consistent():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_zh = (root / "README_zh.md").read_text(encoding="utf-8")
    assert 'version = "1.0.0"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert "Development Status :: 5 - Production/Stable" in pyproject
    assert "paginated thumbnails" not in readme.lower()
    assert "分页缩略图" not in readme_zh
    assert "v1.0.0 Stable" in readme
    assert "v1.0.0 Stable" in readme_zh
