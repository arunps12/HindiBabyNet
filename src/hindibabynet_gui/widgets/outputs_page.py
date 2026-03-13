"""Output browser page: scan and explore participant outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services.outputs_service import scan_outputs, ParticipantStatus


class OutputsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._participants: list[ParticipantStatus] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Outputs")
        btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(btn_refresh)
        self._count_label = QLabel("")
        toolbar.addWidget(self._count_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Splitter: table | detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: participant table
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Participant", "Female", "Male", "Child", "Background",
            "TextGrid", "Segments", "Summary",
        ])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.currentCellChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        # Right: detail panel
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(4, 4, 4, 4)

        self._detail_label = QLabel("Select a participant to view details")
        self._detail_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        detail_layout.addWidget(self._detail_label)

        btn_row = QHBoxLayout()
        self._btn_open_folder = QPushButton("Open Folder")
        self._btn_open_folder.clicked.connect(self._open_folder)
        btn_row.addWidget(self._btn_open_folder)
        self._btn_open_summary = QPushButton("View Summary JSON")
        self._btn_open_summary.clicked.connect(self._view_summary)
        btn_row.addWidget(self._btn_open_summary)
        btn_row.addStretch()
        detail_layout.addLayout(btn_row)

        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setStyleSheet("font-family: 'Monospace', 'Courier New'; font-size: 11px;")
        detail_layout.addWidget(self._detail_text)
        splitter.addWidget(detail)

        splitter.setSizes([500, 400])
        layout.addWidget(splitter, 1)

        self.refresh()

    def refresh(self):
        self._participants = scan_outputs()
        self._table.setRowCount(len(self._participants))
        for i, p in enumerate(self._participants):
            self._table.setItem(i, 0, QTableWidgetItem(p.participant_id))
            self._set_bool_cell(i, 1, p.has_main_female)
            self._set_bool_cell(i, 2, p.has_main_male)
            self._set_bool_cell(i, 3, p.has_child)
            self._set_bool_cell(i, 4, p.has_background)
            self._set_bool_cell(i, 5, p.has_textgrid)
            self._set_bool_cell(i, 6, p.has_segments_parquet)
            self._set_bool_cell(i, 7, p.has_summary_json)
        self._count_label.setText(f"{len(self._participants)} participant(s)")

    def _set_bool_cell(self, row: int, col: int, val: bool):
        item = QTableWidgetItem("✓" if val else "—")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if val:
            item.setForeground(Qt.GlobalColor.darkGreen)
        else:
            item.setForeground(Qt.GlobalColor.gray)
        self._table.setItem(row, col, item)

    def _selected_participant(self) -> ParticipantStatus | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._participants):
            return self._participants[row]
        return None

    def _on_row_selected(self, row: int, *_):
        p = self._selected_participant()
        if not p:
            return
        self._detail_label.setText(p.participant_id)
        lines = [f"Output directory: {p.output_dir}"]
        if p.output_dir and p.output_dir.exists():
            for f in sorted(p.output_dir.iterdir()):
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                lines.append(f"  {f.name:40s}  {size_str}")
        self._detail_text.setPlainText("\n".join(lines))

    def _open_folder(self):
        p = self._selected_participant()
        if not p or not p.output_dir:
            return
        path = str(p.output_dir)
        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            os.startfile(path)

    def _view_summary(self):
        p = self._selected_participant()
        if not p or not p.has_summary_json or not p.output_dir:
            QMessageBox.information(self, "No Summary", "No summary JSON available for this participant.")
            return
        summary_path = p.output_dir / f"{p.participant_id}_summary.json"
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            self._detail_text.setPlainText(json.dumps(data, indent=2))
        except Exception as exc:
            self._detail_text.setPlainText(f"Error reading summary: {exc}")
