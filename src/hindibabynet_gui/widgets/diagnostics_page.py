"""Diagnostics page: environment and prerequisite checks."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services.diagnostics_service import CheckResult, run_all_checks


STATUS_COLORS = {
    "ok": "#27ae60",
    "warn": "#f39c12",
    "fail": "#e74c3c",
}

STATUS_ICONS = {
    "ok": "●",
    "warn": "▲",
    "fail": "✖",
}


class DiagnosticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        btn_run = QPushButton("Run Diagnostics")
        btn_run.clicked.connect(self.run_checks)
        toolbar.addWidget(btn_run)
        self._summary_label = QLabel("")
        toolbar.addWidget(self._summary_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Status", "Check", "Result", "Detail"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, 1)

        self.run_checks()

    def run_checks(self):
        results = run_all_checks()
        self._table.setRowCount(len(results))

        n_ok = n_warn = n_fail = 0
        for i, r in enumerate(results):
            # Status icon
            icon = STATUS_ICONS.get(r.status, "?")
            color = STATUS_COLORS.get(r.status, "#333")
            status_item = QTableWidgetItem(icon)
            status_item.setForeground(Qt.GlobalColor.white)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setBackground(self._qcolor(color))
            self._table.setItem(i, 0, status_item)

            self._table.setItem(i, 1, QTableWidgetItem(r.name))
            self._table.setItem(i, 2, QTableWidgetItem(r.message))
            self._table.setItem(i, 3, QTableWidgetItem(r.detail))

            if r.status == "ok":
                n_ok += 1
            elif r.status == "warn":
                n_warn += 1
            else:
                n_fail += 1

        self._summary_label.setText(
            f"<span style='color:#27ae60'>● {n_ok} OK</span> &nbsp; "
            f"<span style='color:#f39c12'>▲ {n_warn} Warnings</span> &nbsp; "
            f"<span style='color:#e74c3c'>✖ {n_fail} Failures</span>"
        )
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)

    @staticmethod
    def _qcolor(hex_color: str):
        from PySide6.QtGui import QColor
        return QColor(hex_color)
