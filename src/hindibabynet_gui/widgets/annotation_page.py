"""Annotation integration page: wrapper for ADS/IDS annotation tool."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services.annotation_service import list_annotatable_participants
from hindibabynet_gui.services.process_service import ProcessRunner
from hindibabynet_gui.utils.command_builder import build_annotate


class AnnotationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = ProcessRunner(self)
        self._participants: list[dict] = []
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Participant table ──
        table_group = QGroupBox("Annotatable Participants")
        tl = QVBoxLayout(table_group)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_row.addWidget(btn_refresh)

        btn_status = QPushButton("Check Status (all)")
        btn_status.clicked.connect(self._check_status)
        btn_row.addWidget(btn_status)
        btn_row.addStretch()
        tl.addLayout(btn_row)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Participant", "Female WAV", "Male WAV", "Female Prog.", "Male Prog."
        ])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        tl.addWidget(self._table)
        layout.addWidget(table_group)

        # ── Launch controls ──
        ctrl_group = QGroupBox("Launch Annotation")
        cl = QHBoxLayout(ctrl_group)

        cl.addWidget(QLabel("Speaker:"))
        self._speaker_combo = QComboBox()
        self._speaker_combo.addItems(["both", "female", "male"])
        cl.addWidget(self._speaker_combo)

        self._resume_cb = QCheckBox("Resume")
        self._resume_cb.setToolTip("Resume from last saved position")
        cl.addWidget(self._resume_cb)

        self._export_only_cb = QCheckBox("Export only")
        self._export_only_cb.setToolTip("Skip annotation, just export WAVs from existing CSV")
        cl.addWidget(self._export_only_cb)

        self._btn_launch = QPushButton("▶  Launch Annotation")
        self._btn_launch.setStyleSheet("font-weight: bold;")
        self._btn_launch.clicked.connect(self._launch)
        cl.addWidget(self._btn_launch)

        self._btn_cancel = QPushButton("■  Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        cl.addWidget(self._btn_cancel)
        layout.addWidget(ctrl_group)

        # ── Output ──
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(20000)
        self._output.setStyleSheet("font-family: 'Monospace', 'Courier New'; font-size: 11px;")
        layout.addWidget(self._output, 1)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self.refresh()

    def _connect_signals(self):
        self._runner.started.connect(lambda cmd: self._set_running(True))
        self._runner.output_ready.connect(self._on_output)
        self._runner.error_ready.connect(self._on_output)
        self._runner.finished.connect(self._on_finished)

    def refresh(self):
        self._participants = list_annotatable_participants()
        self._table.setRowCount(len(self._participants))
        for i, p in enumerate(self._participants):
            self._table.setItem(i, 0, QTableWidgetItem(p["participant_id"]))
            self._table.setItem(i, 1, QTableWidgetItem("✓" if p["has_female"] else "—"))
            self._table.setItem(i, 2, QTableWidgetItem("✓" if p["has_male"] else "—"))
            self._table.setItem(i, 3, QTableWidgetItem(p["female_progress"]))
            self._table.setItem(i, 4, QTableWidgetItem(p["male_progress"]))

    def _selected_pid(self) -> str | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._participants):
            return self._participants[row]["participant_id"]
        return None

    def _launch(self):
        if self._runner.is_running:
            QMessageBox.warning(self, "Busy", "Annotation process already running.")
            return
        pid = self._selected_pid()
        if not pid:
            QMessageBox.warning(self, "No Selection", "Select a participant first.")
            return
        speaker = self._speaker_combo.currentText()
        if speaker == "both":
            speaker = None
        cmd = build_annotate(
            participant=pid,
            speaker=speaker,
            resume=self._resume_cb.isChecked(),
            export_only=self._export_only_cb.isChecked(),
        )
        self._output.clear()
        self._runner.run(cmd)

    def _cancel(self):
        self._runner.cancel()

    def _check_status(self):
        if self._runner.is_running:
            return
        cmd = build_annotate(show_status=True)
        self._output.clear()
        self._runner.run(cmd)

    def _set_running(self, running: bool):
        self._btn_launch.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._status_label.setText("Running…" if running else "")

    def _on_output(self, text: str):
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)

    def _on_finished(self, exit_code: int, status: str):
        self._set_running(False)
        self._status_label.setText(f"Finished: {status} (exit {exit_code})")
        if status == "success":
            self._status_label.setStyleSheet("color: green;")
        else:
            self._status_label.setStyleSheet("color: red;")
