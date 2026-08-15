"""Shared scene-composition helpers used by the Qt UI (``compose_app_qt``).

The historical tkinter ``ComposeApp`` was removed in v0.5.0; the GUI now
lives in :mod:`compose_app_qt`. This package keeps the GUI-agnostic
helpers that the Qt port still imports:

- :mod:`compose_app.models`       – Cutout / Instance dataclasses
- :mod:`compose_app.rendering`    – pure PIL/numpy render functions
- :mod:`compose_app.segmentation` – polygon transforms / serialization
- :mod:`compose_app.auto_mask`    – optional rembg/SAM cutouts
- :mod:`compose_app.io_utils`     – class-map scan, composite save

Headless / server code should ``import scenepaste`` and never need to
touch this package.
"""

__version__ = "10.0.0"

__all__ = ["__version__"]
