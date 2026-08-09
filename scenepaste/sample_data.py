"""Access to the tiny sample dataset bundled inside the installed package."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional


def bundled_samples_root() -> Path:
    """Return the installed package's bundled ``samples`` directory.

    ScenePaste wheels are installed as ordinary directories by pip, so the
    returned path is directly usable by OpenCV/Pillow/Qt file APIs. A clear
    error is raised instead of silently falling back to a source checkout.
    """
    node = resources.files("scenepaste.resources").joinpath("samples")
    path = Path(str(node))
    if not (path / "objects").is_dir() or not (path / "backgrounds").is_dir():
        raise RuntimeError("ScenePaste bundled sample resources are missing from this installation")
    return path


def find_samples_root() -> Optional[Path]:
    """Prefer package resources, then fall back to a source-tree ``samples/``."""
    try:
        return bundled_samples_root()
    except Exception:
        pass
    candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    seen = set()
    for start in candidates:
        start = start.resolve() if start.exists() else start
        for _ in range(6):
            key = str(start)
            if key in seen:
                break
            seen.add(key)
            p = start / "samples"
            if (p / "objects").is_dir() and (p / "backgrounds").is_dir():
                return p
            if start.parent == start:
                break
            start = start.parent
    return None
