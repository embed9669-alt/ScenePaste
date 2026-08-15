"""CLI subprocess smoke test for the full generate -> analyze -> split -> merge chain."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OBJECTS = REPO_ROOT / "samples" / "objects"
SAMPLE_BACKGROUNDS = REPO_ROOT / "samples" / "backgrounds"


def _run(args, **kwargs) -> subprocess.CompletedProcess:
    """Run a scenepaste CLI command and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "scenepaste", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), **kwargs,
    )


def test_version_and_help():
    out = _run(["--version"])
    assert out.returncode == 0
    assert "ScenePaste" in out.stdout
    out = _run(["--help"])
    assert out.returncode == 0
    assert "generate" in out.stdout
    assert "analyze" in out.stdout
    assert "split" in out.stdout
    assert "merge" in out.stdout
    assert "explore" in out.stdout


def test_generate_subcommand_writes_dataset(tmp_path: Path):
    out_dir = tmp_path / "ds"
    result = _run([
        "generate",
        "--objects", str(SAMPLE_OBJECTS),
        "--backgrounds", str(SAMPLE_BACKGROUNDS),
        "--output", str(out_dir),
        "--count", "3",
        "--class-map", "person=0",
        "--seed", "42",
    ])
    assert result.returncode == 0, result.stderr
    assert (out_dir / "images" / "train").is_dir()
    images = list((out_dir / "images" / "train").glob("*.jpg"))
    assert len(images) == 3
    assert (out_dir / "data.yaml").is_file()


def test_analyze_subcommand_reads_generated(tmp_path: Path):
    out_dir = tmp_path / "ds"
    _run([
        "generate",
        "--objects", str(SAMPLE_OBJECTS),
        "--backgrounds", str(SAMPLE_BACKGROUNDS),
        "--output", str(out_dir),
        "--count", "2",
        "--class-map", "person=0",
        "--seed", "7",
    ])
    result = _run(["analyze", str(out_dir), "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "images" in payload or "summary" in payload or payload  # any structured output


def test_split_subcommand_partitions_dataset(tmp_path: Path):
    out_dir = tmp_path / "ds"
    _run([
        "generate",
        "--objects", str(SAMPLE_OBJECTS),
        "--backgrounds", str(SAMPLE_BACKGROUNDS),
        "--output", str(out_dir),
        "--count", "8",
        "--min-objects", "1",
        "--max-objects", "1",
        "--class-map", "person=0,motorcycle=1,truck=2",
        "--seed", "11",
    ])
    result = _run(["split", "--input", str(out_dir), "--val-ratio", "0.4"])
    assert result.returncode == 0, result.stderr
    # Split should produce val directories and a report.
    assert (out_dir / "images" / "val").is_dir()
    assert (out_dir / "labels" / "val").is_dir()
    assert (out_dir / "split_report.json").is_file()


def test_merge_subcommand_combines_two_datasets(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    merged = tmp_path / "merged"
    for d, seed in ((a, 1), (b, 2)):
        _run([
            "generate",
            "--objects", str(SAMPLE_OBJECTS),
            "--backgrounds", str(SAMPLE_BACKGROUNDS),
            "--output", str(d),
            "--count", "3",
            "--class-map", "person=0",
            "--seed", str(seed),
        ])
    result = _run(["merge", str(a), str(b), "--output", str(merged)])
    assert result.returncode == 0, result.stderr
    a_imgs = len(list((a / "images" / "train").glob("*.jpg")))
    b_imgs = len(list((b / "images" / "train").glob("*.jpg")))
    m_imgs = len(list((merged / "images" / "train").glob("*.jpg")))
    assert m_imgs == a_imgs + b_imgs, "merge lost images"


def test_generate_rejects_bad_class_map(tmp_path: Path):
    result = _run([
        "generate",
        "--objects", str(SAMPLE_OBJECTS),
        "--backgrounds", str(SAMPLE_BACKGROUNDS),
        "--output", str(tmp_path / "x"),
        "--class-map", "bad_format",
    ])
    assert result.returncode == 1
    assert "错误" in result.stderr or "Error" in result.stderr or "error" in result.stderr.lower()


def test_gui_strips_qt_flag_anywhere(monkeypatch):
    """`--qt` may appear mid-argv and must not reach argparse as unknown."""
    from scenepaste import cli as cli_mod
    import compose_app_qt.main_entry as me
    seen = {}

    def fake_entry(argv=None):
        seen["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(me, "main_entry", fake_entry)
    code = cli_mod.main(["gui", "--objects", "o", "--qt", "--output", "out"])
    assert code == 0
    assert "--qt" not in seen["argv"]
    assert "--objects" in seen["argv"]


def test_generate_no_args_uses_qt_entry(monkeypatch):
    from scenepaste import cli as cli_mod
    import compose_app_qt.main_entry as me
    called = {"n": 0}

    def fake_entry(argv=None):
        called["n"] += 1
        return 0

    monkeypatch.setattr(me, "main_entry", fake_entry)
    code = cli_mod.generate_main([])
    assert code == 0
    assert called["n"] == 1


def test_project_manifest_can_drive_headless_generate(tmp_path: Path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    out_dir = project_dir / "generated"
    init = _run([
        "project", "init", str(project_dir),
        "--objects", str(SAMPLE_OBJECTS),
        "--backgrounds", str(SAMPLE_BACKGROUNDS),
        "--output", str(out_dir),
        "--class-map", "person=0",
    ])
    assert init.returncode == 0, init.stderr
    manifest = project_dir / "scenepaste.project.json"
    assert manifest.is_file()

    result = _run([
        "generate", "--project", str(manifest),
        "--count", "2", "--workers", "1", "--no-preview", "--seed", "19",
    ])
    assert result.returncode == 0, result.stderr
    assert len(list((out_dir / "images" / "train").glob("*.jpg"))) == 2


def test_project_defaults_do_not_override_explicit_cli_values(tmp_path: Path):
    from scenepaste.cli import build_generate_parser, config_from_args
    from scenepaste.project import init_project

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output = project_dir / "generated"
    project = init_project(
        project_dir,
        objects=SAMPLE_OBJECTS,
        backgrounds=SAMPLE_BACKGROUNDS,
        output=output,
        class_map={"person": 0},
    )
    project.defaults.update({"workers": 8, "output_format": "all", "preview_ratio": 0.01})
    project.save()

    raw = ["--project", str(project.path), "--workers", "1", "--output-format", "detect"]
    args = build_generate_parser().parse_args(raw)
    args._provided_options = {token.split("=", 1)[0] for token in raw if token.startswith("--")}
    config = config_from_args(args)
    assert config.workers == 1
    assert config.output_format == "detect"
    assert config.preview_ratio == 0.01
