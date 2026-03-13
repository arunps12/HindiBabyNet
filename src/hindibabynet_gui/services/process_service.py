"""Process runner service: executes pipeline commands asynchronously via QProcess."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, Signal

from hindibabynet_gui.utils import PROJECT_ROOT


class ProcessRunner(QObject):
    """Wraps QProcess to run pipeline commands with live stdout/stderr streaming."""

    output_ready = Signal(str)        # incremental text chunk
    error_ready = Signal(str)         # stderr chunk
    finished = Signal(int, str)       # exit_code, status ("success"/"failed"/"cancelled")
    started = Signal(str)             # command string

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._cmd_str: str = ""
        self._start_time: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def command_str(self) -> str:
        return self._cmd_str

    @property
    def start_time(self) -> Optional[datetime]:
        return self._start_time

    def run(self, cmd: list[str], env_override: dict[str, str] | None = None) -> None:
        """Start a command asynchronously. *cmd* is e.g. ["python", "-m", "..."]."""
        if self.is_running:
            return

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))

        # Merge env
        env = QProcess.systemEnvironment()
        if env_override:
            for k, v in env_override.items():
                # Remove existing key then add
                env = [e for e in env if not e.startswith(f"{k}=")]
                env.append(f"{k}={v}")
        proc_env = self._process.processEnvironment()
        # Rebuild from list
        from PySide6.QtCore import QProcessEnvironment
        new_env = QProcessEnvironment()
        for entry in env:
            if "=" in entry:
                k, v = entry.split("=", 1)
                new_env.insert(k, v)
        if env_override:
            for k, v in env_override.items():
                new_env.insert(k, v)

        # Ensure PYTHONUNBUFFERED for live output
        new_env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(new_env)

        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        program = cmd[0]
        args = cmd[1:]
        self._cmd_str = " ".join(cmd)
        self._start_time = datetime.now()

        self.started.emit(self._cmd_str)
        self._process.start(program, args)

    def cancel(self) -> None:
        """Terminate the running process."""
        if self._process and self.is_running:
            self._process.kill()

    # ── Private slots ─────────────────────────────────────
    def _on_stdout(self) -> None:
        if self._process:
            data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
            if data:
                self.output_ready.emit(data)

    def _on_stderr(self) -> None:
        if self._process:
            data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
            if data:
                self.error_ready.emit(data)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if exit_status == QProcess.ExitStatus.CrashExit:
            status = "cancelled"
        elif exit_code == 0:
            status = "success"
        else:
            status = "failed"
        self.finished.emit(exit_code, status)
        self._process = None
