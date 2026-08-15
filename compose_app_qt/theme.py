"""QSS theme strings for the Qt main window and tools."""

DARK_QSS = """
QWidget { background-color: #1f1f22; color: #e4e4e7; font-size: 12px; }
QDialog { background-color: #1f1f22; }
QGraphicsView { background-color: #1c1c1e; border: 1px solid #2c2c2e; border-radius: 4px; }
QPushButton {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    padding: 5px 10px; border-radius: 4px;
}
QPushButton:hover { background-color: #3a3a3c; }
QPushButton:disabled { color: #6e6e72; background-color: #262628; }
QPushButton#primaryButton {
    background-color: #0a84ff; border: 1px solid #0a84ff; color: #ffffff; font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #409cff; border-color: #409cff; }
QPushButton#primaryButton:disabled { background-color: #2a4a6a; border-color: #2a4a6a; color: #9ab; }
QPushButton#dangerButton {
    background-color: #3a2a2a; border: 1px solid #5a3a3a; color: #ff8f8f;
}
QPushButton#dangerButton:hover { background-color: #4a3232; }
QPushButton#layerButton, QPushButton#iconToggleButton {
    padding: 3px 6px; min-height: 22px; font-size: 11px;
}
QPushButton#iconToggleButton {
    padding: 2px 0; min-width: 26px; max-width: 28px;
}
QListWidget {
    background-color: #26262a; border: 1px solid #2c2c2e;
    border-radius: 4px; outline: none; padding: 2px;
}
QListWidget::item {
    padding: 2px 4px; margin: 1px 0; border-radius: 3px; min-height: 28px;
}
QListWidget::item:selected { background-color: #0a84ff; color: #ffffff; }
QListWidget::item:hover:!selected { background-color: #323236; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    padding: 4px 8px; border-radius: 3px; min-height: 22px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    selection-background-color: #0a84ff;
}
QCheckBox { spacing: 6px; color: #c4c4c8; }
QSlider::groove:horizontal {
    height: 4px; background: #3a3a3c; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -4px 0;
    background: #0a84ff; border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #0a84ff; border-radius: 2px; }
QLabel { color: #c4c4c8; }
QLabel#mutedLabel { color: #8e8e93; font-size: 11px; }
QLabel#statusLabel { color: #a0a0a4; font-size: 11px; }
QLabel#readyLabel {
    color: #30d158; font-size: 12px; font-weight: 600;
    padding: 6px; background-color: #1a2e1f; border: 1px solid #2d5a38;
    border-radius: 4px;
}
QLabel#busyLabel {
    color: #ffd60a; font-size: 12px;
    padding: 4px;
}
QWidget#bannerLabel {
    background-color: #1a2e1f;
    border-bottom: 1px solid #2d5a38;
}
QLabel#bannerLabel {
    background-color: transparent;
    color: #c8e6c9;
    padding: 0;
}
QProgressBar#readyBar::chunk { background-color: #30d158; }
QGroupBox {
    border: 1px solid #2c2c2e; border-radius: 6px;
    margin-top: 10px; padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    color: #a0a0a4; subcontrol-origin: margin;
    left: 10px; padding: 0 4px;
}
QTabWidget::pane {
    border: 1px solid #2c2c2e; border-radius: 4px;
    top: -1px; background-color: #1f1f22;
}
QTabBar::tab {
    background-color: #26262a; color: #a0a0a4;
    border: 1px solid #2c2c2e; border-bottom: none;
    padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px;
    border-top-right-radius: 4px; min-width: 64px;
}
QTabBar::tab:selected {
    background-color: #1f1f22; color: #e4e4e7; font-weight: 600;
}
QTabBar::tab:hover:!selected { background-color: #323236; color: #d0d0d4; }
QToolBar {
    background-color: #26262a; border-bottom: 1px solid #2c2c2e;
    spacing: 4px; padding: 4px 6px;
}
QToolBar::separator {
    background: #3a3a3c; width: 1px; margin: 4px 6px;
}
QProgressBar {
    background-color: #2c2c2e; border: 1px solid #3a3a3c;
    border-radius: 4px; text-align: center; min-height: 16px; max-height: 18px;
    color: #e4e4e7;
}
QProgressBar::chunk {
    background-color: #0a84ff; border-radius: 3px;
}
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #2c2c2e; width: 1px; }
QStatusBar { background-color: #26262a; color: #a0a0a4; }
QStatusBar QLabel { padding: 0 6px; }
QMenuBar { background-color: #26262a; color: #e4e4e7; padding: 2px 0; }
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background-color: #3a3a3c; }
QMenu { background-color: #2c2c2e; border: 1px solid #3a3a3c; }
QMenu::item { padding: 5px 24px 5px 12px; }
QMenu::item:selected { background-color: #0a84ff; color: #ffffff; }
"""

LIGHT_QSS = """
QWidget { background-color: #f5f5f7; color: #1c1c1e; font-size: 12px; }
QDialog { background-color: #f5f5f7; }
QGraphicsView { background-color: #ffffff; border: 1px solid #d0d0d4; border-radius: 4px; }
QPushButton {
    background-color: #ffffff; border: 1px solid #c0c0c4;
    padding: 5px 10px; border-radius: 4px;
}
QPushButton:hover { background-color: #e8e8ea; }
QPushButton:disabled { color: #8e8e93; background-color: #f0f0f2; }
QPushButton#primaryButton {
    background-color: #0071e3; border: 1px solid #0071e3; color: #ffffff; font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #0077ed; border-color: #0077ed; }
QPushButton#primaryButton:disabled { background-color: #9ec3f0; border-color: #9ec3f0; color: #f5f9ff; }
QPushButton#dangerButton {
    background-color: #fff5f5; border: 1px solid #f0c0c0; color: #c0392b;
}
QPushButton#dangerButton:hover { background-color: #ffe8e8; }
QPushButton#layerButton, QPushButton#iconToggleButton {
    padding: 3px 6px; min-height: 22px; font-size: 11px;
}
QPushButton#iconToggleButton {
    padding: 2px 0; min-width: 26px; max-width: 28px;
}
QListWidget {
    background-color: #ffffff; border: 1px solid #d0d0d4;
    border-radius: 4px; outline: none; padding: 2px;
}
QListWidget::item {
    padding: 2px 4px; margin: 1px 0; border-radius: 3px; min-height: 28px;
}
QListWidget::item:selected { background-color: #0071e3; color: #ffffff; }
QListWidget::item:hover:!selected { background-color: #ececef; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background-color: #ffffff; border: 1px solid #c0c0c4;
    padding: 4px 8px; border-radius: 3px; min-height: 22px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #ffffff; border: 1px solid #c0c0c4;
    selection-background-color: #0071e3; selection-color: #ffffff;
}
QCheckBox { spacing: 6px; color: #3a3a3c; }
QSlider::groove:horizontal {
    height: 4px; background: #d0d0d4; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -4px 0;
    background: #0071e3; border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #0071e3; border-radius: 2px; }
QLabel { color: #3a3a3c; }
QLabel#mutedLabel { color: #8e8e93; font-size: 11px; }
QLabel#statusLabel { color: #6e6e73; font-size: 11px; }
QLabel#readyLabel {
    color: #1b7f3a; font-size: 12px; font-weight: 600;
    padding: 6px; background-color: #e8f8ee; border: 1px solid #a8dfb8;
    border-radius: 4px;
}
QLabel#busyLabel {
    color: #9a6700; font-size: 12px;
    padding: 4px;
}
QWidget#bannerLabel {
    background-color: #e8f8ee;
    border-bottom: 1px solid #a8dfb8;
}
QLabel#bannerLabel {
    background-color: transparent;
    color: #1b7f3a;
    padding: 0;
}
QProgressBar#readyBar::chunk { background-color: #34c759; }
QGroupBox {
    border: 1px solid #d8d8dc; border-radius: 6px;
    margin-top: 10px; padding: 12px 10px 10px 10px;
    font-weight: 600; background-color: #fafafa;
}
QGroupBox::title {
    color: #6e6e73; subcontrol-origin: margin;
    left: 10px; padding: 0 4px;
}
QTabWidget::pane {
    border: 1px solid #d0d0d4; border-radius: 4px;
    top: -1px; background-color: #f5f5f7;
}
QTabBar::tab {
    background-color: #ececef; color: #6e6e73;
    border: 1px solid #d0d0d4; border-bottom: none;
    padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px;
    border-top-right-radius: 4px; min-width: 64px;
}
QTabBar::tab:selected {
    background-color: #f5f5f7; color: #1c1c1e; font-weight: 600;
}
QTabBar::tab:hover:!selected { background-color: #e4e4e8; color: #3a3a3c; }
QToolBar {
    background-color: #ececef; border-bottom: 1px solid #d0d0d4;
    spacing: 4px; padding: 4px 6px;
}
QToolBar::separator {
    background: #c0c0c4; width: 1px; margin: 4px 6px;
}
QProgressBar {
    background-color: #e8e8ea; border: 1px solid #d0d0d4;
    border-radius: 4px; text-align: center; min-height: 16px; max-height: 18px;
    color: #1c1c1e;
}
QProgressBar::chunk {
    background-color: #0071e3; border-radius: 3px;
}
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #d0d0d4; width: 1px; }
QStatusBar { background-color: #ececef; color: #6e6e73; }
QStatusBar QLabel { padding: 0 6px; }
QMenuBar { background-color: #ececef; color: #1c1c1e; padding: 2px 0; }
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background-color: #d8d8dc; }
QMenu { background-color: #ffffff; border: 1px solid #c0c0c4; }
QMenu::item { padding: 5px 24px 5px 12px; }
QMenu::item:selected { background-color: #0071e3; color: #ffffff; }
"""



DARK_POLISH_QSS = """
QWidget { font-family: "Inter", "Segoe UI", "Microsoft YaHei UI", sans-serif; }
QFrame#dialogHero, QFrame#factoryHero {
    background-color: #26262a; border: 1px solid #34343a; border-radius: 10px;
}
QLabel#dialogTitle { color: #f5f5f7; font-size: 20px; font-weight: 700; }
QLabel#dialogSubtitle { color: #a8a8ad; font-size: 12px; }
QLabel#sectionHeadline { color: #f0f0f3; font-size: 14px; font-weight: 700; }
QLabel#statusPill {
    background-color: #252b34; border: 1px solid #354052; color: #c9d8ed;
    border-radius: 8px; padding: 8px 10px; font-weight: 600;
}
QLabel#modelStatusCard {
    background-color: #252b34; border: 1px solid #354052; color: #c9d8ed;
    border-radius: 8px; padding: 10px;
}
QLabel#errorLabel {
    color: #ff9f9f; background-color: #382428; border: 1px solid #6a343d;
    border-radius: 8px; padding: 9px; font-weight: 600;
}
QFrame#metricCard {
    background-color: #27272b; border: 1px solid #34343a; border-radius: 9px;
}
QLabel#metricTitle { color: #8f8f96; font-size: 11px; }
QLabel#metricValue { color: #f5f5f7; font-size: 21px; font-weight: 700; }
QLabel#metricSubtitle { color: #77777e; font-size: 10px; }
QLabel#previewSurface {
    background-color: #171719; border: 1px dashed #3a3a40; border-radius: 9px;
    color: #77777e; padding: 10px;
}
QToolButton#sectionToggle {
    background: transparent; border: none; color: #d8d8dc; text-align: left;
    padding: 7px 4px; font-size: 13px; font-weight: 650;
}
QToolButton#sectionToggle:hover { background-color: #29292d; border-radius: 6px; }
QGroupBox { background-color: #222225; border-color: #34343a; border-radius: 8px; }
QPushButton { min-height: 26px; padding: 5px 11px; border-radius: 6px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 25px; border-radius: 5px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #4a4a50; min-height: 28px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background-color: #333338; color: #f5f5f7; border: 1px solid #4a4a50; padding: 5px; }

QWidget#welcomePage { background-color: #1b1b1e; }
QFrame#welcomeCard { background-color: #242428; border: 1px solid #35353b; border-radius: 16px; }
QLabel#welcomeTitle { color: #f7f7fa; font-size: 34px; font-weight: 800; }
QLabel#welcomeSubtitle { color: #66aaff; font-size: 16px; font-weight: 650; }
QLabel#welcomeDescription { color: #aaaab0; font-size: 13px; }
QLabel#welcomeSteps { color: #c6c6cb; background-color: #1f2730; border: 1px solid #304257; border-radius: 8px; padding: 10px; }
QPushButton#recentProjectButton { text-align: left; padding: 8px 10px; background-color: #29292e; border-color: #3a3a40; }
QPushButton#recentProjectButton:hover { background-color: #323239; border-color: #4b4b54; }
"""

LIGHT_POLISH_QSS = """
QWidget { font-family: "Inter", "Segoe UI", "Microsoft YaHei UI", sans-serif; }
QFrame#dialogHero, QFrame#factoryHero {
    background-color: #ffffff; border: 1px solid #dedee3; border-radius: 10px;
}
QLabel#dialogTitle { color: #171719; font-size: 20px; font-weight: 700; }
QLabel#dialogSubtitle { color: #6e6e73; font-size: 12px; }
QLabel#sectionHeadline { color: #1c1c1e; font-size: 14px; font-weight: 700; }
QLabel#statusPill {
    background-color: #eef5ff; border: 1px solid #c6dbf8; color: #245785;
    border-radius: 8px; padding: 8px 10px; font-weight: 600;
}
QLabel#modelStatusCard {
    background-color: #eef5ff; border: 1px solid #c6dbf8; color: #245785;
    border-radius: 8px; padding: 10px;
}
QLabel#errorLabel {
    color: #b42318; background-color: #fff2f0; border: 1px solid #f2b8b2;
    border-radius: 8px; padding: 9px; font-weight: 600;
}
QFrame#metricCard { background-color: #ffffff; border: 1px solid #dedee3; border-radius: 9px; }
QLabel#metricTitle { color: #77777e; font-size: 11px; }
QLabel#metricValue { color: #1c1c1e; font-size: 21px; font-weight: 700; }
QLabel#metricSubtitle { color: #8e8e93; font-size: 10px; }
QLabel#previewSurface {
    background-color: #fbfbfc; border: 1px dashed #c9c9cf; border-radius: 9px;
    color: #8e8e93; padding: 10px;
}
QToolButton#sectionToggle {
    background: transparent; border: none; color: #2c2c2e; text-align: left;
    padding: 7px 4px; font-size: 13px; font-weight: 650;
}
QToolButton#sectionToggle:hover { background-color: #ededf0; border-radius: 6px; }
QGroupBox { background-color: #ffffff; border-color: #dedee3; border-radius: 8px; }
QPushButton { min-height: 26px; padding: 5px 11px; border-radius: 6px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 25px; border-radius: 5px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c4c4ca; min-height: 28px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background-color: #ffffff; color: #1c1c1e; border: 1px solid #c9c9cf; padding: 5px; }

QWidget#welcomePage { background-color: #f3f4f7; }
QFrame#welcomeCard { background-color: #ffffff; border: 1px solid #dadde4; border-radius: 16px; }
QLabel#welcomeTitle { color: #171719; font-size: 34px; font-weight: 800; }
QLabel#welcomeSubtitle { color: #0071e3; font-size: 16px; font-weight: 650; }
QLabel#welcomeDescription { color: #6e6e73; font-size: 13px; }
QLabel#welcomeSteps { color: #344054; background-color: #eef5ff; border: 1px solid #c6dbf8; border-radius: 8px; padding: 10px; }
QPushButton#recentProjectButton { text-align: left; padding: 8px 10px; background-color: #fbfbfc; border-color: #d8dbe2; }
QPushButton#recentProjectButton:hover { background-color: #f0f4fa; border-color: #b8c7dc; }
"""

def qss_for(mode: str) -> str:
    """Return the QSS stylesheet for ``"dark"`` or ``"light"`` mode."""
    return (LIGHT_QSS + LIGHT_POLISH_QSS) if mode == "light" else (DARK_QSS + DARK_POLISH_QSS)


def apply_theme(widget, mode: str = "dark") -> None:
    """Apply the shared ScenePaste theme to a top-level widget."""
    widget.setStyleSheet(qss_for(mode))


def style_primary(button) -> None:
    button.setObjectName("primaryButton")


def style_danger(button) -> None:
    button.setObjectName("dangerButton")
