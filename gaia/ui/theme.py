from __future__ import annotations

GAIA_STYLE = """
QMainWindow, QWidget {
    background: #05070a;
    color: #e5e7eb;
    font-family: Inter, Segoe UI, Arial;
    font-size: 10pt;
}
QLabel {
    color: #d1d5db;
}
QPushButton {
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #1d4ed8;
    border: 1px solid #38bdf8;
    color: #ffffff;
}
QPushButton:pressed {
    background: #0f172a;
    border: 1px solid #22c55e;
    color: #86efac;
}
QPushButton:disabled {
    color: #6b7280;
    border-color: #1f2937;
    background: #0b1117;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0b1117;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    border-radius: 5px;
    padding: 5px 8px;
    min-height: 24px;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border: 1px solid #2563eb;
}
QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #05070a;
    border-left: 1px solid #1f2937;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #0b1117;
    color: #e5e7eb;
    border: 1px solid #2563eb;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
}
QCheckBox {
    color: #d1d5db;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    background: #0b1117;
    border: 1px solid #1f2937;
    border-radius: 3px;
}
QCheckBox::indicator:hover {
    border: 1px solid #2563eb;
}
QCheckBox::indicator:checked {
    background: #22c55e;
    border: 1px solid #86efac;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #0b1117;
    border: 1px solid #1f2937;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
    background: #2563eb;
    border: 1px solid #38bdf8;
}
QSlider::sub-page:horizontal {
    background: #22c55e;
    border-radius: 3px;
}
QProgressBar {
    background: #0b1117;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background: #22c55e;
    border-radius: 4px;
}
QTableWidget {
    background: #05070a;
    alternate-background-color: #0b1117;
    color: #e5e7eb;
    gridline-color: #1f2937;
    selection-background-color: #1d4ed8;
    selection-color: #ffffff;
    border: 1px solid #1f2937;
}
QTabWidget::pane {
    border: 1px solid #1f2937;
    background: #05070a;
}
QTabBar::tab {
    background: #0b1117;
    color: #d1d5db;
    border: 1px solid #1f2937;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background: #111827;
    color: #86efac;
}
QLineEdit, QTextEdit {
    background: #05070a;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    border-radius: 5px;
    padding: 6px;
    selection-background-color: #1d4ed8;
}
QHeaderView::section {
    background: #111827;
    color: #86efac;
    border: 1px solid #1f2937;
    padding: 6px;
    font-weight: 600;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #05070a;
    border: 1px solid #0b1117;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #2563eb66;
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #22c55e99;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: #0b1117;
    border: 1px solid #1f2937;
}
QListWidget {
    background: #05070a;
    color: #d1d5db;
    border: 1px solid #1f2937;
}
QListWidget::item {
    padding: 8px;
}
QListWidget::item:selected {
    background: #1d4ed8;
    color: #ffffff;
}
"""
