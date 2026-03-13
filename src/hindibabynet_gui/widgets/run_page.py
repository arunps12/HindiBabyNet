"""Run Pipeline page: choose mode, configure, execute, and monitor."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services.process_service import ProcessRunner
from hindibabynet_gui.services import history_service
from hindibabynet_gui.utils.command_builder import (
    build_full_pipeline,
    build_stage_01,
    build_stage_02_from_parquet,
    build_stage_02_single_wav,
    build_stage_03,
    build_stage_03_vad,
    build_stage_04_diarization,
    build_stage_05_intersection,
    build_stage_06_classification,
)


# ── Mode definitions ──────────────────────────────────────────────────────
MODES = [
    ("Full Pipeline (all stages)", "full"),
    ("Stage 01 — Data Ingestion", "stage01"),
    ("Stage 02 — Audio Prep (batch from parquet)", "stage02_parquet"),
    ("Stage 02 — Audio Prep (single WAV)", "stage02_single"),
    ("Stage 03 — Speaker Classification (single WAV)", "stage03_wav"),
    ("Stage 03 — Speaker Classification (analysis dir)", "stage03_dir"),
    ("Stage 03 — Speaker Classification (from parquet)", "stage03_parquet"),
    ("Stage 03 — VAD only", "stage03_vad"),
    ("Stage 04 — Diarization", "stage04"),
    ("Stage 05 — Intersection", "stage05"),
    ("Stage 06 — Classification export", "stage06"),
]


class RunPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = ProcessRunner(self)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Top controls ──
        ctrl_group = QGroupBox("Run Configuration")
        ctrl = QVBoxLayout(ctrl_group)

        # Mode
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        for label, _ in MODES:
            self._mode_combo.addItem(label)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._mode_combo, 1)
        ctrl.addLayout(row1)

        # Backend
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["xgb", "vtc"])
        row2.addWidget(self._backend_combo)

        row2.addWidget(QLabel("Participant Limit:"))
        self._limit_spin = QSpinBox()
        self._limit_spin.setRange(0, 99999)
        self._limit_spin.setSpecialValueText("All")
        self._limit_spin.setToolTip("0 = process all participants")
        row2.addWidget(self._limit_spin)
        row2.addStretch()
        ctrl.addLayout(row2)

        # Input path row
        row3 = QHBoxLayout()
        self._input_label = QLabel("Input Path:")
        row3.addWidget(self._input_label)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Browse or type path…")
        row3.addWidget(self._input_edit, 1)
        self._browse_btn = QToolButton()
        self._browse_btn.setText("…")
        self._browse_btn.clicked.connect(self._browse_input)
        row3.addWidget(self._browse_btn)
        ctrl.addLayout(row3)

        # Recording ID (for single WAV)
        row4 = QHBoxLayout()
        self._rid_label = QLabel("Recording ID:")
        row4.addWidget(self._rid_label)
        self._rid_edit = QLineEdit()
        self._rid_edit.setPlaceholderText("Optional custom recording ID")
        row4.addWidget(self._rid_edit, 1)
        ctrl.addLayout(row4)

        # Run ID
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Run ID:"))
        self._run_id_edit = QLineEdit()
        self._run_id_edit.setPlaceholderText("Auto-generated if empty")
        row5.addWidget(self._run_id_edit, 1)
        ctrl.addLayout(row5)

        layout.addWidget(ctrl_group)

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 6px 20px;")
        self._run_btn.clicked.connect(self._start_run)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_run)
        btn_row.addWidget(self._cancel_btn)

        self._status_label = QLabel("")
        btn_row.addWidget(self._status_label, 1)
        layout.addLayout(btn_row)

        # ── Live output ──
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(50000)
        self._output.setStyleSheet("font-family: 'Monospace', 'Courier New'; font-size: 11px;")
        layout.addWidget(self._output, 1)

        # ── Command preview ──
        cmd_row = QHBoxLayout()
        cmd_row.addWidget(QLabel("Command:"))
        self._cmd_preview = QLineEdit()
        self._cmd_preview.setReadOnly(True)
        self._cmd_preview.setStyleSheet("color: #666;")
        cmd_row.addWidget(self._cmd_preview, 1)
        layout.addLayout(cmd_row)

        self._on_mode_changed(0)

    def _connect_signals(self):
        self._runner.started.connect(self._on_started)
        self._runner.output_ready.connect(self._on_output)
        self._runner.error_ready.connect(self._on_error)
        self._runner.finished.connect(self._on_finished)
        # Update preview when fields change
        self._mode_combo.currentIndexChanged.connect(self._update_preview)
        self._backend_combo.currentIndexChanged.connect(self._update_preview)
        self._limit_spin.valueChanged.connect(self._update_preview)
        self._input_edit.textChanged.connect(self._update_preview)
        self._rid_edit.textChanged.connect(self._update_preview)
        self._run_id_edit.textChanged.connect(self._update_preview)

    def _on_mode_changed(self, idx: int):
        _, mode = MODES[idx]
        needs_input = mode in ("stage02_parquet", "stage02_single", "stage03_wav",
                               "stage03_dir", "stage03_parquet", "stage03_vad",
                               "stage04", "stage05", "stage06")
        self._input_edit.setVisible(needs_input)
        self._input_label.setVisible(needs_input)
        self._browse_btn.setVisible(needs_input)

        needs_rid = mode == "stage02_single"
        self._rid_label.setVisible(needs_rid)
        self._rid_edit.setVisible(needs_rid)

        needs_backend = mode in ("stage03_wav", "stage03_dir", "stage03_parquet", "full")
        self._backend_combo.setEnabled(needs_backend)

        needs_limit = mode not in ("stage02_single", "stage03_wav")
        self._limit_spin.setEnabled(needs_limit)

        # Update placeholder
        placeholders = {
            "stage02_parquet": "Path to recordings.parquet",
            "stage02_single": "Path to input .wav file",
            "stage03_wav": "Path to prepared .wav file",
            "stage03_dir": "Path to analysis directory",
            "stage03_parquet": "Path to recordings.parquet",
            "stage03_vad": "Path to recordings.parquet",
            "stage04": "Path to recordings.parquet",
            "stage05": "Path to recordings.parquet",
            "stage06": "Path to recordings.parquet",
        }
        self._input_edit.setPlaceholderText(placeholders.get(mode, ""))
        self._update_preview()

    def _browse_input(self):
        _, mode = MODES[self._mode_combo.currentIndex()]
        if mode in ("stage02_single", "stage03_wav"):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select WAV file", "", "WAV files (*.wav *.WAV);;All (*)"
            )
        elif mode == "stage03_dir":
            path = QFileDialog.getExistingDirectory(self, "Select analysis directory")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select parquet file", "", "Parquet (*.parquet);;All (*)"
            )
        if path:
            self._input_edit.setText(path)

    def _build_command(self) -> list[str]:
        _, mode = MODES[self._mode_combo.currentIndex()]
        limit = self._limit_spin.value() or None
        backend = self._backend_combo.currentText()
        input_path = self._input_edit.text().strip()
        rid = self._rid_edit.text().strip() or None
        run_id = self._run_id_edit.text().strip() or None

        if mode == "full":
            return build_full_pipeline(run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"), limit=limit)
        elif mode == "stage01":
            return build_stage_01(run_id=run_id)
        elif mode == "stage02_parquet":
            return build_stage_02_from_parquet(input_path, run_id=run_id, limit=limit)
        elif mode == "stage02_single":
            return build_stage_02_single_wav(input_path, recording_id=rid)
        elif mode == "stage03_wav":
            return build_stage_03(wav=input_path, backend=backend, run_id=run_id)
        elif mode == "stage03_dir":
            return build_stage_03(analysis_dir=input_path, backend=backend, run_id=run_id, limit=limit)
        elif mode == "stage03_parquet":
            return build_stage_03(recordings_parquet=input_path, backend=backend, run_id=run_id, limit=limit)
        elif mode == "stage03_vad":
            return build_stage_03_vad(recordings_parquet=input_path, run_id=run_id, limit=limit)
        elif mode == "stage04":
            return build_stage_04_diarization(recordings_parquet=input_path, run_id=run_id, limit=limit)
        elif mode == "stage05":
            return build_stage_05_intersection(recordings_parquet=input_path, run_id=run_id, limit=limit)
        elif mode == "stage06":
            return build_stage_06_classification(recordings_parquet=input_path, run_id=run_id, limit=limit)
        return []

    def _update_preview(self):
        try:
            cmd = self._build_command()
            self._cmd_preview.setText(" ".join(cmd))
        except Exception:
            self._cmd_preview.setText("")

    def _start_run(self):
        if self._runner.is_running:
            QMessageBox.warning(self, "Busy", "A process is already running.")
            return

        cmd = self._build_command()
        if not cmd:
            QMessageBox.warning(self, "Invalid", "Cannot build command for this mode.")
            return

        _, mode = MODES[self._mode_combo.currentIndex()]
        # Confirm long runs
        if mode == "full":
            reply = QMessageBox.question(
                self, "Confirm",
                "Run the full pipeline? This may take a long time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._output.clear()
        self._runner.run(cmd)
        history_service.add_record(
            command=" ".join(cmd),
            mode=mode,
            backend=self._backend_combo.currentText(),
            input_paths=[self._input_edit.text()] if self._input_edit.text() else [],
            participant_limit=self._limit_spin.value() or None,
            run_id=self._run_id_edit.text() or None,
        )

    def _cancel_run(self):
        self._runner.cancel()

    # ── Slots ─────────────────────────────────────────────
    def _on_started(self, cmd_str: str):
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Running…")
        self._status_label.setStyleSheet("color: blue;")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._output.appendPlainText(f"[{timestamp}] Started: {cmd_str}\n")

    def _on_output(self, text: str):
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)

    def _on_error(self, text: str):
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)

    def _on_finished(self, exit_code: int, status: str):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._output.appendPlainText(f"\n[{timestamp}] Finished — {status} (exit code {exit_code})")
        history_service.update_last(status, exit_code)
        if status == "success":
            self._status_label.setText(f"Done ✓ (exit {exit_code})")
            self._status_label.setStyleSheet("color: green;")
        elif status == "cancelled":
            self._status_label.setText("Cancelled")
            self._status_label.setStyleSheet("color: orange;")
        else:
            self._status_label.setText(f"Failed (exit {exit_code})")
            self._status_label.setStyleSheet("color: red;")
