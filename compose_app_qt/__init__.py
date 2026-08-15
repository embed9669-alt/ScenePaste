"""ScenePaste Qt GUI (PySide6).

The only desktop editor. Shared helpers live in :mod:`compose_app`
(models / rendering / segmentation / io_utils).

Architecture:

- :mod:`compose_app_qt.state`      – document model + selection
- :mod:`compose_app_qt.undo`       – QUndoCommand subclasses
- :mod:`compose_app_qt.canvas`     – QGraphicsView + InstanceItem
- :mod:`compose_app_qt.panels`     – left/right/top panels
- :mod:`compose_app_qt.workers`    – QThread workers (load/batch save)
- :mod:`compose_app_qt.templates`  – scene template save/load
- :mod:`compose_app_qt.theme`      – QSS dark/light themes
- :mod:`compose_app_qt.app`        – MainWindow
- :mod:`compose_app_qt.main_entry` – console-script entry point

Import the entry point as ``from compose_app_qt.main_entry import main_entry``
so the ``main_entry`` submodule is not shadowed.
"""

__all__ = []
