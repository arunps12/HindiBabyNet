"""Help / About page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from hindibabynet_gui import __version__
from hindibabynet_gui.utils import PROJECT_ROOT


HELP_TEXT = """\
<h2>HindiBabyNet GUI — Help</h2>

<h3>Overview</h3>
<p>This GUI provides a graphical interface to the HindiBabyNet speaker classification pipeline.
It wraps the existing CLI scripts and Python modules, allowing researchers to configure,
run, and monitor the pipeline without using the terminal.</p>

<h3>Pages</h3>
<ul>
  <li><b>Home</b> — Quick overview and recent stats</li>
  <li><b>Configuration</b> — Load, edit, and save <code>configs/config.yaml</code></li>
  <li><b>Run Pipeline</b> — Execute pipeline stages with live log output</li>
  <li><b>Outputs</b> — Browse participant output files (WAVs, TextGrids, etc.)</li>
  <li><b>Logs</b> — View saved log files and run history</li>
  <li><b>Annotation</b> — Launch the ADS/IDS annotation tool</li>
  <li><b>Diagnostics</b> — Check system prerequisites and configuration</li>
</ul>

<h3>Pipeline Stages</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Stage</th><th>Name</th><th>Input</th><th>Output</th></tr>
<tr><td>01</td><td>Data Ingestion</td><td>Raw audio directory</td><td>recordings.parquet</td></tr>
<tr><td>02</td><td>Audio Preparation</td><td>recordings.parquet / WAV</td><td>Prepared WAVs</td></tr>
<tr><td>03</td><td>VAD</td><td>Prepared WAVs</td><td>VAD parquet</td></tr>
<tr><td>04</td><td>Diarization</td><td>Prepared WAVs</td><td>Diarization parquet</td></tr>
<tr><td>05</td><td>Intersection</td><td>VAD + Diarization</td><td>Speech segments</td></tr>
<tr><td>06</td><td>Classification</td><td>Speech segments + WAV</td><td>Speaker WAVs, TextGrid</td></tr>
</table>

<h3>Backends</h3>
<ul>
  <li><b>xgb</b> — Built-in HindiBabyNet pipeline (eGeMAPS + XGBoost)</li>
  <li><b>vtc</b> — External VTC 2.0 voice type classifier</li>
</ul>

<h3>Keyboard Shortcuts</h3>
<ul>
  <li><code>Ctrl+Q</code> — Quit</li>
</ul>

<h3>Troubleshooting</h3>
<ul>
  <li>Check the <b>Diagnostics</b> page for environment issues</li>
  <li>Ensure <code>HF_TOKEN</code> is set for pyannote model access</li>
  <li>Ensure <code>ffmpeg</code> is installed for audio processing</li>
  <li>For VTC backend, ensure the VTC repository is cloned at <code>vtc.repo_path</code></li>
</ul>
"""


class HelpPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QLabel(HELP_TEXT)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setWordWrap(True)
        content.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.setContentsMargins(16, 16, 16, 16)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # About
        about = QLabel(
            f"<p><b>HindiBabyNet GUI</b> v{__version__}<br>"
            f"Project: {PROJECT_ROOT}<br>"
            f"Built with PySide6</p>"
        )
        about.setTextFormat(Qt.TextFormat.RichText)
        about.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(about)
