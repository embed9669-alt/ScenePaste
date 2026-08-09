#!/usr/bin/env python3
"""Build and verify a clean ScenePaste source ZIP."""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_EXCLUDE = {
    ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__",
    "build", "dist", ".venv", "venv", "generated", "output", "objects",
    "backgrounds", "new_data", "newdata2", "scenepaste.egg-info", ".scenepaste", "hardmine", "comparison", "shards",
}
DIR_EXCLUDE = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".scenepaste"}
FILE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
FILE_EXCLUDE_NAMES = {".coverage", "coverage.xml", ".DS_Store", "Thumbs.db", "scenepaste.project.json", "qa_dashboard.html", "qa_report.json", "distribution_profile.json",
                      "target_distribution_profile.json", "active_scene_template.json", "leakage_report.json", "diversity_report.json", "diversity_report.csv", "real_vs_synthetic.json", "real_vs_synthetic.html", "hard_examples.json", "hard_examples.csv", "hard_examples.txt", "hardmine_dashboard.html", "hard_negative_backgrounds.txt"}
FORBIDDEN_RELEASE_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".scenepaste"}


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("pyproject.toml does not contain a project version")
    return match.group(1)


def should_skip(rel: Path) -> bool:
    if not rel.parts:
        return False
    if rel.parts[0] in TOP_LEVEL_EXCLUDE:
        return True
    if any(part in DIR_EXCLUDE or part.endswith(".egg-info") for part in rel.parts):
        return True
    if rel.suffix.lower() in FILE_EXCLUDE_SUFFIXES:
        return True
    if rel.name in FILE_EXCLUDE_NAMES or rel.name.endswith(".tmp"):
        return True
    return False


def verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    if not names:
        raise RuntimeError("release ZIP is empty")
    bad = []
    for name in names:
        parts = Path(name).parts
        if any(part in FORBIDDEN_RELEASE_PARTS or part.endswith(".egg-info") for part in parts):
            bad.append(name)
        if Path(name).name in FILE_EXCLUDE_NAMES:
            bad.append(name)
    if bad:
        raise RuntimeError(f"release ZIP contains forbidden files: {bad[:8]}")
    required = {"README.md", "README_zh.md", "LICENSE", "pyproject.toml", "scenepaste/cli.py", "scenepaste/resources/samples/objects/sample_person.json", "scripts/__init__.py"}
    top = names[0].split("/", 1)[0]
    available = {n[len(top) + 1:] for n in names if n.startswith(top + "/")}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"release ZIP missing required files: {missing}")


def build(output: Path) -> Path:
    version = project_version()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package_root_name = f"scenepaste-{version}"
    with tempfile.TemporaryDirectory(prefix="scenepaste_release_") as tmp:
        stage = Path(tmp) / package_root_name
        stage.mkdir()
        for src in ROOT.rglob("*"):
            rel = src.relative_to(ROOT)
            if should_skip(rel):
                continue
            dst = stage / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage.parent).as_posix())
    verify_zip(output)
    return output


def main() -> int:
    version = project_version()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"ScenePaste-v{version}.zip")
    args = parser.parse_args()
    path = build(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
