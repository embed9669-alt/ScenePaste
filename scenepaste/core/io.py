"""Unicode-safe image I/O and JSON helpers.

OpenCV's ``imread``/``imwrite`` cannot handle non-ASCII paths on Windows; the
helpers here go through ``np.fromfile`` / ``cv2.imencode`` + ``tofile`` to
support any path. EXIF orientation is applied via PIL for cross-version
consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .models import IMAGE_SUFFIXES


def log_default(message: str) -> None:
    """Default logger: prints to stdout, flushed immediately."""
    print(message, flush=True)


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from a path that may contain non-ASCII characters.

    Returns ``None`` on read failure or empty file.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except (OSError, ValueError):
        return None


def imwrite_unicode(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> bool:
    """Write an image to a (possibly non-ASCII) path, creating parent dirs.

    Returns ``True`` on success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params: List[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def imread_with_exif(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image and apply EXIF orientation (mobile photos).

    Uses ``PIL.ImageOps.exif_transpose`` to avoid OpenCV's version-dependent
    EXIF handling (OpenCV >=4.5 applies EXIF on ``imread`` by default, which
    would otherwise be applied twice). Returns BGR ndarray; falls back to
    ``imread_unicode`` on any error.
    """
    try:
        from PIL import Image, ImageOps
        with Image.open(str(path)) as opened:
            pil = ImageOps.exif_transpose(opened)  # type: ignore[assignment]
            mode = "RGB" if (flags == cv2.IMREAD_COLOR or flags == 1) else None
            if mode is None:
                pil = pil.convert("RGB")  # type: ignore[assignment]
            else:
                pil = pil.convert(mode)  # type: ignore[assignment]
            rgb = np.asarray(pil)
            if rgb.ndim != 3:
                rgb = np.stack([rgb] * 3, axis=-1)
            return rgb[..., ::-1].copy()  # RGB -> BGR
    except Exception:
        return imread_unicode(path, flags)


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_backgrounds(backgrounds_dir: Path) -> List[Path]:
    """Return a sorted list of image files under ``backgrounds_dir`` (recursive)."""
    return sorted(
        path
        for path in backgrounds_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def save_cutouts(assets, output_dir: Path) -> None:
    """Write all assets as transparent PNG cutouts under ``cutouts/``."""
    cutouts_dir = output_dir / "cutouts"
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    for index, asset in enumerate(assets):
        alpha_u8 = np.clip(asset.alpha * 255.0, 0, 255).astype(np.uint8)
        bgra = np.dstack([asset.image, alpha_u8])
        name = f"{asset.label}_{index:05d}.png"
        imwrite_unicode(cutouts_dir / name, bgra)
