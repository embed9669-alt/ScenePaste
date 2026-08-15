"""Headless-safe Cutout Studio facade.

Pure LabelMe save I/O is exported without Qt. GUI classes are loaded only when
PySide6 is available, fixing server/CI imports while preserving the historical
``compose_app_qt.cutout_studio`` import path.
"""
from __future__ import annotations

from scenepaste.core.labelme import save_cutout_as_labelme

_QT_IMPORT_ERROR = None
try:  # pragma: no cover - availability is environment dependent
    from .cutout_studio_gui import AnnotateView, ObjectCutoutStudioDialog
except ModuleNotFoundError as exc:  # headless/core-only installation
    if exc.name and (exc.name == "PySide6" or exc.name.startswith("PySide6.")):
        _QT_IMPORT_ERROR = exc
    else:
        raise


def require_qt() -> None:
    if _QT_IMPORT_ERROR is not None:
        raise ImportError(
            "Cutout Studio GUI requires PySide6; install `scenepaste[gui-qt]`"
        ) from _QT_IMPORT_ERROR


__all__ = ["save_cutout_as_labelme", "require_qt"]
if _QT_IMPORT_ERROR is None:
    __all__ += ["AnnotateView", "ObjectCutoutStudioDialog"]
