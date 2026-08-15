"""Headless-safe facade for the Qt Asset Studio."""

from __future__ import annotations

try:
    from .asset_studio_gui import AssetStudioDialog, MaskPaintView
except ImportError as exc:  # pragma: no cover - optional GUI dependency
    _IMPORT_ERROR = exc

    def require_qt():
        raise RuntimeError("Asset Studio GUI requires PySide6; install `scenepaste[gui-qt]`") from _IMPORT_ERROR

    __all__ = ["require_qt"]
else:
    def require_qt():
        return True

    __all__ = ["AssetStudioDialog", "MaskPaintView", "require_qt"]
