"""Main application: top-level window with sidebar navigation."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services import ConfigService
from hindibabynet_gui.utils import PROJECT_ROOT
from hindibabynet_gui.widgets.annotation_page import AnnotationPage
from hindibabynet_gui.widgets.config_page import ConfigPage
from hindibabynet_gui.widgets.diagnostics_page import DiagnosticsPage
from hindibabynet_gui.widgets.help_page import HelpPage
from hindibabynet_gui.widgets.home_page import HomePage
from hindibabynet_gui.widgets.logs_page import LogsPage
from hindibabynet_gui.widgets.outputs_page import OutputsPage
from hindibabynet_gui.widgets.run_page import RunPage


# Navigation entries: (label, icon_char)
NAV_ITEMS = [
    ("Home", "🏠"),
    ("Configuration", "⚙"),
    ("Run Pipeline", "▶"),
    ("Outputs", "📂"),
    ("Logs", "📋"),
    ("Annotation", "🏷"),
    ("Diagnostics", "🔍"),
    ("Help", "❓"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HindiBabyNet Pipeline")
        self.resize(1100, 750)

        self._config_service = ConfigService()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Sidebar ──
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(170)
        self._sidebar.setStyleSheet("""
            QListWidget {
                background: #2c3e50;
                color: #ecf0f1;
                border: none;
                font-size: 13px;
                padding-top: 8px;
            }
            QListWidget::item {
                padding: 10px 14px;
                border-bottom: 1px solid #34495e;
            }
            QListWidget::item:selected {
                background: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background: #34495e;
            }
        """)
        for label, icon in NAV_ITEMS:
            item = QListWidgetItem(f"{icon}  {label}")
            item.setData(Qt.ItemDataRole.UserRole, label)
            self._sidebar.addItem(item)
        self._sidebar.currentRowChanged.connect(self._on_nav)
        main_layout.addWidget(self._sidebar)

        # ── Page stack ──
        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}

        self._pages["Home"] = HomePage()
        self._pages["Configuration"] = ConfigPage(self._config_service)
        self._pages["Run Pipeline"] = RunPage()
        self._pages["Outputs"] = OutputsPage()
        self._pages["Logs"] = LogsPage()
        self._pages["Annotation"] = AnnotationPage()
        self._pages["Diagnostics"] = DiagnosticsPage()
        self._pages["Help"] = HelpPage()

        for label, _ in NAV_ITEMS:
            self._stack.addWidget(self._pages[label])

        main_layout.addWidget(self._stack, 1)

        # ── Status bar ──
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage(f"Project: {PROJECT_ROOT}")

        # ── Menu bar ──
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Select Home
        self._sidebar.setCurrentRow(0)

    def _on_nav(self, idx: int):
        if 0 <= idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
            label = NAV_ITEMS[idx][0]
            self._statusbar.showMessage(f"{label} — {PROJECT_ROOT}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("HindiBabyNet")
    app.setStyle("Fusion")

    # Set a clean default font
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
