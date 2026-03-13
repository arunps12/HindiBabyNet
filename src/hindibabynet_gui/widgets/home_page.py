"""Home / Overview page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from hindibabynet_gui.services.outputs_service import scan_artifact_runs, scan_outputs
from hindibabynet_gui.utils import PROJECT_ROOT


class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("HindiBabyNet Pipeline")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Automatic speaker classification for long-form Hindi child–caregiver audio recordings."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(subtitle)

        # Quick stats
        self._stats_group = QGroupBox("Quick Overview")
        stats_layout = QVBoxLayout(self._stats_group)
        self._stats_label = QLabel("Loading…")
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(self._stats_group)

        # Project info
        info_group = QGroupBox("Project")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel(f"Root: {PROJECT_ROOT}"))
        layout.addWidget(info_group)

        # Quick-start guide
        guide = QGroupBox("Quick Start")
        guide_layout = QVBoxLayout(guide)
        steps = [
            "1. Go to <b>Diagnostics</b> to verify your environment",
            "2. Go to <b>Configuration</b> to review and edit settings",
            "3. Go to <b>Run Pipeline</b> to process audio files",
            "4. Go to <b>Outputs</b> to browse results per participant",
            "5. Go to <b>Annotation</b> to label ADS/IDS segments",
        ]
        for s in steps:
            lbl = QLabel(s)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            guide_layout.addWidget(lbl)
        layout.addWidget(guide)

        layout.addStretch()
        self.refresh()

    def refresh(self):
        try:
            runs = scan_artifact_runs()
            participants = scan_outputs()
            n_complete = sum(
                1 for p in participants
                if p.has_main_female and p.has_main_male and p.has_textgrid
            )
            lines = [
                f"Artifact runs: <b>{len(runs)}</b>",
                f"Participants with outputs: <b>{len(participants)}</b>",
                f"Fully complete (female + male + TextGrid): <b>{n_complete}</b>",
            ]
            if runs:
                lines.append(f"Latest run: <b>{runs[0]['run_id']}</b> — stages: {', '.join(runs[0]['stages'])}")
            self._stats_label.setText("<br>".join(lines))
        except Exception as exc:
            self._stats_label.setText(f"Could not load stats: {exc}")
