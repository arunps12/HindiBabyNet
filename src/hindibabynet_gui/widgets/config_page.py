"""Configuration editor page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from hindibabynet_gui.services import ConfigService, FieldMeta


class ConfigPage(QWidget):
    def __init__(self, config_service: ConfigService, parent=None):
        super().__init__(parent)
        self._svc = config_service
        self._editors: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        self._status = QLabel("")
        toolbar.addWidget(self._status)
        toolbar.addStretch()

        btn_reload = QPushButton("Reload")
        btn_reload.setToolTip("Reload config.yaml from disk")
        btn_reload.clicked.connect(self._reload)
        toolbar.addWidget(btn_reload)

        btn_validate = QPushButton("Validate")
        btn_validate.setToolTip("Validate path fields")
        btn_validate.clicked.connect(self._validate)
        toolbar.addWidget(btn_validate)

        btn_save = QPushButton("Save")
        btn_save.setToolTip("Save changes to config.yaml")
        btn_save.setStyleSheet("font-weight: bold;")
        btn_save.clicked.connect(self._save)
        toolbar.addWidget(btn_save)
        outer.addLayout(toolbar)

        # Scroll area for fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._form_layout = QVBoxLayout(container)
        self._form_layout.setSpacing(4)

        current_group = ""
        current_form: QFormLayout | None = None

        for fdef in ConfigService.field_definitions():
            if fdef.group != current_group:
                current_group = fdef.group
                grp = QGroupBox(current_group)
                current_form = QFormLayout(grp)
                current_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                self._form_layout.addWidget(grp)

            editor = self._make_editor(fdef)
            self._editors[fdef.key] = editor
            label = QLabel(fdef.label)
            label.setToolTip(fdef.tooltip)
            if current_form:
                current_form.addRow(label, editor)

        self._form_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

        self._populate()

    def _make_editor(self, fdef: FieldMeta) -> QWidget:
        if fdef.kind == "path":
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setToolTip(fdef.tooltip)
            le.setObjectName(fdef.key)
            hl.addWidget(le)
            btn = QToolButton()
            btn.setText("…")
            btn.setToolTip("Browse")
            btn.clicked.connect(lambda checked=False, k=fdef.key: self._browse(k))
            hl.addWidget(btn)
            return row
        elif fdef.kind == "int":
            sb = QSpinBox()
            sb.setRange(0, 999999)
            sb.setToolTip(fdef.tooltip)
            return sb
        elif fdef.kind == "float":
            dsb = QDoubleSpinBox()
            dsb.setRange(-1000.0, 999999.0)
            dsb.setDecimals(2)
            dsb.setToolTip(fdef.tooltip)
            return dsb
        elif fdef.kind == "bool":
            cb = QCheckBox()
            cb.setToolTip(fdef.tooltip)
            return cb
        elif fdef.kind == "choice":
            combo = QComboBox()
            combo.addItems(fdef.choices)
            combo.setToolTip(fdef.tooltip)
            return combo
        else:  # str, list
            le = QLineEdit()
            le.setToolTip(fdef.tooltip)
            return le

    def _populate(self):
        for fdef in ConfigService.field_definitions():
            val = self._svc.get(fdef.key)
            editor = self._editors.get(fdef.key)
            if editor is None:
                continue
            self._set_editor_value(editor, fdef, val)
        self._status.setText(f"Loaded: {self._svc.path}")

    def _set_editor_value(self, editor: QWidget, fdef: FieldMeta, val):
        if fdef.kind == "path":
            le = editor.findChild(QLineEdit)
            if le:
                le.setText(str(val) if val is not None else "")
        elif fdef.kind == "int":
            editor.setValue(int(val) if val is not None else 0)
        elif fdef.kind == "float":
            editor.setValue(float(val) if val is not None else 0.0)
        elif fdef.kind == "bool":
            editor.setChecked(bool(val))
        elif fdef.kind == "choice":
            idx = editor.findText(str(val)) if val else 0
            editor.setCurrentIndex(max(idx, 0))
        elif fdef.kind == "list":
            editor.setText(str(val) if val is not None else "")
        else:
            editor.setText(str(val) if val is not None else "")

    def _get_editor_value(self, editor: QWidget, fdef: FieldMeta):
        if fdef.kind == "path":
            le = editor.findChild(QLineEdit)
            return le.text() if le else ""
        elif fdef.kind == "int":
            return editor.value()
        elif fdef.kind == "float":
            return editor.value()
        elif fdef.kind == "bool":
            return editor.isChecked()
        elif fdef.kind == "choice":
            return editor.currentText()
        elif fdef.kind == "list":
            text = editor.text().strip()
            if text.startswith("["):
                # Parse JSON-like list
                import json
                try:
                    return json.loads(text.replace("'", '"'))
                except Exception:
                    return [s.strip().strip("'\"") for s in text.strip("[]").split(",") if s.strip()]
            return text
        else:
            return editor.text()

    def _browse(self, key: str):
        editor = self._editors.get(key)
        if not editor:
            return
        le = editor.findChild(QLineEdit)
        current = le.text() if le else ""
        path = QFileDialog.getExistingDirectory(self, f"Select directory for {key}", current)
        if path and le:
            le.setText(path)

    def _collect(self):
        """Collect editor values back into the config data."""
        for fdef in ConfigService.field_definitions():
            editor = self._editors.get(fdef.key)
            if editor is None:
                continue
            val = self._get_editor_value(editor, fdef)
            self._svc.set(fdef.key, val)

    def _save(self):
        self._collect()
        try:
            self._svc.save()
            self._status.setText("Saved ✓")
            self._status.setStyleSheet("color: green;")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            self._status.setText("Save failed")
            self._status.setStyleSheet("color: red;")

    def _reload(self):
        self._svc.reload()
        self._populate()
        self._status.setText("Reloaded from disk")
        self._status.setStyleSheet("color: blue;")

    def _validate(self):
        self._collect()
        warnings = self._svc.validate_paths()
        if warnings:
            QMessageBox.warning(self, "Validation Warnings", "\n".join(warnings))
            self._status.setText(f"{len(warnings)} warning(s)")
            self._status.setStyleSheet("color: orange;")
        else:
            self._status.setText("All paths OK ✓")
            self._status.setStyleSheet("color: green;")
