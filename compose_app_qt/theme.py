"""QSS theme strings for the Qt main window."""

DARK_QSS = """
QWidget { background-color: #1f1f22; color: #e4e4e7; font-size: 12px; }
QGraphicsView { background-color: #1c1c1e; border: 1px solid #2c2c2e; }
QPushButton {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    padding: 6px 12px; border-radius: 4px;
}
QPushButton:hover { background-color: #3a3a3c; }
QPushButton:disabled { color: #6e6e72; background-color: #262628; }
QListWidget {
    background-color: #26262a; border: 1px solid #2c2c2e;
    border-radius: 4px;
}
QListWidget::item:selected { background-color: #0a84ff; color: #ffffff; }
QListWidgetItem { padding: 4px; }
QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    padding: 3px 6px; border-radius: 3px;
}
QLabel { color: #c4c4c8; }
QGroupBox {
    border: 1px solid #2c2c2e; border-radius: 4px;
    margin-top: 10px; padding-top: 10px;
}
QGroupBox::title {
    color: #a0a0a4; subcontrol-origin: margin;
    left: 10px; padding: 0 4px;
}
QStatusBar { background-color: #26262a; color: #a0a0a4; }
QMenuBar { background-color: #26262a; color: #e4e4e7; }
QMenuBar::item:selected { background-color: #3a3a3c; }
QMenu { background-color: #2c2c2e; border: 1px solid #3a3a3c; }
QMenu::item:selected { background-color: #0a84ff; color: #ffffff; }
"""

LIGHT_QSS = """
QWidget { background-color: #fafafa; color: #1c1c1e; font-size: 12px; }
QGraphicsView { background-color: #ffffff; border: 1px solid #d0d0d4; }
QPushButton {
    background-color: #ffffff; border: 1px solid #c0c0c4;
    padding: 6px 12px; border-radius: 4px;
}
QPushButton:hover { background-color: #e8e8ea; }
QListWidget { background-color: #ffffff; border: 1px solid #d0d0d4; }
QListWidget::item:selected { background-color: #0a84ff; color: #ffffff; }
"""


def qss_for(mode: str) -> str:
    """Return the QSS stylesheet for ``"dark"`` or ``"light"`` mode."""
    return LIGHT_QSS if mode == "light" else DARK_QSS
