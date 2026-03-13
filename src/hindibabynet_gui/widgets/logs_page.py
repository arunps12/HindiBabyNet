"""Logs viewer page: live output + saved log browser."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from hindibabynet_gui.utils import LOGS_DIR
from hindibabynet_gui.services import history_service


class LogsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter logs…")
        self._search.textChanged.connect(self._filter_text)
        toolbar.addWidget(self._search, 1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_list)
        toolbar.addWidget(btn_refresh)

        btn_copy = QPushButton("Copy All")
        btn_copy.clicked.connect(self._copy_all)
        toolbar.addWidget(btn_copy)
        layout.addLayout(toolbar)

        # Splitter: log list | viewer
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: log file list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Log Directories"))
        self._log_list = QListWidget()
        self._log_list.currentItemChanged.connect(self._on_log_selected)
        left_layout.addWidget(self._log_list)
        splitter.addWidget(left)

        # Right: log content
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # File selector within a log dir
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._file_combo = QComboBox()
        self._file_combo.currentIndexChanged.connect(self._load_file)
        file_row.addWidget(self._file_combo, 1)
        right_layout.addLayout(file_row)

        self._viewer = QPlainTextEdit()
        self._viewer.setReadOnly(True)
        self._viewer.setStyleSheet("font-family: 'Monospace', 'Courier New'; font-size: 11px;")
        self._viewer.setMaximumBlockCount(100000)
        right_layout.addWidget(self._viewer)
        splitter.addWidget(right)

        splitter.setSizes([200, 600])
        layout.addWidget(splitter, 1)

        # Run history
        layout.addWidget(QLabel("Recent GUI Runs"))
        self._history_list = QPlainTextEdit()
        self._history_list.setReadOnly(True)
        self._history_list.setMaximumHeight(150)
        self._history_list.setStyleSheet("font-family: 'Monospace', 'Courier New'; font-size: 10px;")
        layout.addWidget(self._history_list)

        self._refresh_list()
        self._refresh_history()

    def _refresh_list(self):
        self._log_list.clear()
        if not LOGS_DIR.exists():
            return
        for d in sorted(LOGS_DIR.iterdir(), reverse=True):
            if d.is_dir():
                item = QListWidgetItem(d.name)
                item.setData(Qt.ItemDataRole.UserRole, str(d))
                self._log_list.addItem(item)

    def _on_log_selected(self, current: QListWidgetItem, _prev):
        if not current:
            return
        log_dir = Path(current.data(Qt.ItemDataRole.UserRole))
        self._file_combo.clear()
        if log_dir.exists():
            files = sorted(log_dir.rglob("*"))
            for f in files:
                if f.is_file():
                    self._file_combo.addItem(f.name, str(f))

    def _load_file(self, idx: int):
        if idx < 0:
            return
        path_str = self._file_combo.itemData(idx)
        if not path_str:
            return
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            self._viewer.setPlainText(text)
            self._highlight_errors()
        except Exception as exc:
            self._viewer.setPlainText(f"Error reading file: {exc}")

    def _filter_text(self, query: str):
        if not query:
            return
        # Simple find
        cursor = self._viewer.document().find(query)
        if not cursor.isNull():
            self._viewer.setTextCursor(cursor)

    def _highlight_errors(self):
        """Scroll to first ERROR/WARNING if present."""
        text = self._viewer.toPlainText()
        for keyword in ("ERROR", "CRITICAL", "Traceback"):
            idx = text.find(keyword)
            if idx >= 0:
                cursor = self._viewer.textCursor()
                cursor.setPosition(idx)
                self._viewer.setTextCursor(cursor)
                self._viewer.ensureCursorVisible()
                break

    def _copy_all(self):
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._viewer.toPlainText())

    def _refresh_history(self):
        records = history_service.get_history(20)
        lines = []
        for r in records:
            ts = r.get("timestamp", "?")[:19]
            st = r.get("status", "?")
            mode = r.get("mode", "?")
            ec = r.get("exit_code", "")
            lines.append(f"{ts}  {st:<10}  {mode:<25}  exit={ec}")
        self._history_list.setPlainText("\n".join(lines) if lines else "No GUI runs recorded yet.")
