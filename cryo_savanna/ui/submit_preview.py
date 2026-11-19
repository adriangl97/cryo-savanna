# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The job editor's "Preview submission script…": the run_submit.script RELION would write for the job, and what
may be wrong with it (project/submit_script.py). The template is read and SLURM asked in the background."""

from __future__ import annotations

import html
from collections.abc import Callable

from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from cryo_savanna.project.submit_script import Preview
from cryo_savanna.viewers.tasks import TaskError, Tasks


class SubmitPreviewDialog(QDialog):
    def __init__(self, job: str, make: Callable[[], Preview], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Submission script of {job.rstrip('/')}")
        self.resize(820, 620)
        self.result: Preview | None = None
        self.checks = QLabel("Reading the template and asking SLURM…")
        self.checks.setWordWrap(True)
        self.script = QPlainTextEdit()
        self.script.setReadOnly(True)
        self.script.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.script.setLineWrapMode(QPlainTextEdit.NoWrap)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.checks)
        lay.addWidget(self.script, 1)
        lay.addWidget(QLabel(f"<span style='color:gray'>As RELION writes it to {html.escape(job)}run_submit.script "
                             "when it sends the job, from the settings in the editor now.</span>"))
        lay.addWidget(buttons)
        self.tasks = Tasks(self)
        self.tasks.run(make, self._show)

    def _show(self, result) -> None:
        if isinstance(result, TaskError):
            self.checks.setText(f"<span style='color:#d43c3c'>Could not make the preview: {html.escape(str(result))}"
                                "</span>")
            return
        self.result = result
        self.script.setPlainText(result.script)
        parts = [f"<li style='color:#d43c3c'>{html.escape(p)}</li>" for p in result.problems]
        parts += [f"<li style='color:gray'>{html.escape(n)}</li>" for n in result.notes]
        head = ("<b style='color:#d43c3c'>Check these before submitting:</b>" if result.problems
                else "<b style='color:#2e9e4f'>Nothing wrong found.</b>")
        self.checks.setText(head + (f"<ul style='margin:2px'>{''.join(parts)}</ul>" if parts else ""))
