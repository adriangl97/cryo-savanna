# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Review of jobs read from their directories (project/rebuild.py) before they go in the pipeline.

Used for a folder whose default_pipeline.star is missing (generate it) and for Project → Add jobs
run outside the pipeline. Each job can be left out and its status changed; a folder that is not a
<Type>/jobNNN/ directory (a command run by hand with its own --o) can be added as a job.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.project.ops import decompose
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.rebuild import RECENT_S, FoundJob, RebuildError, Scan, adopt_folder, scan, write_pipeline
from cryo_savanna.relion_consts import JOB_TYPES, STATUS_LABELS
from cryo_savanna.ui.busy import run_busy

COLUMNS = ("Job", "Type", "Status", "Inputs from", "Outputs", "Read from", "Notes")


def _busy(fn: Callable, text: str = "Reading the job folders…", parent: QWidget | None = None):
    """Reading (or writing) every job folder takes a while over a network file system: in a thread,
    with the window alive meanwhile (ui/busy.py)."""
    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
    try:
        return run_busy(fn, text, parent)
    finally:
        QApplication.restoreOverrideCursor()


def _short(name: str) -> str:
    return name.rstrip("/")


class RebuildDialog(QDialog):
    """The jobs of a Scan, to check, leave out or give another status before writing them."""

    def __init__(self, project_dir: Path, result: Scan, rescan: Callable[[], Scan], new: bool,
                 type_names: dict[str, str] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.result = result
        self.rescan = rescan
        self.new = new
        self.type_names = type_names or {}
        self.setWindowTitle("Generate default_pipeline.star" if new else "Add jobs run outside the pipeline")
        self.resize(1100, 520)

        if new:
            text = (f"<b>{self.project_dir}</b> has no default_pipeline.star. These jobs were found in its folders "
                    "and read from their job.star, job_pipeline.star, note.txt and log files. Check them, then "
                    "write a new default_pipeline.star with the jobs that are ticked.")
        else:
            text = ("These job folders are not in the project's pipeline (run from the command line, or added "
                    "by hand). Check them, then add the jobs that are ticked. A command run with its own output "
                    "folder (not &lt;Type&gt;/jobNNN/) can be added with <i>Add folder as job…</i>.")
        intro = QLabel(text)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.summary = QLabel()
        self.summary.setWordWrap(True)

        self.adopt_btn = QPushButton("Add folder as job…")
        self.adopt_btn.setToolTip("Move a folder written by a command run by hand to <Type>/jobNNN/ (a link is "
                                  "left at its old place) and add it as a job")
        self.adopt_btn.clicked.connect(self.adopt)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Write default_pipeline.star" if new else "Add jobs")
        self.buttons.addButton(self.adopt_btn, QDialogButtonBox.ActionRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.summary)
        lay.addWidget(self.buttons)
        self.table.itemChanged.connect(lambda _: self._update_ok())
        self.fill()

    # -- contents
    def fill(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for fj in self.result.jobs:
            self._add_row(fj)
        self.table.blockSignals(False)
        skipped = "".join(f"<br>Left out: {_short(d)} ({why})" for d, why in self.result.skipped)
        self.summary.setText(f"{len(self.result.jobs)} job(s) found.{skipped}" if self.result.jobs or skipped
                             else "No job folders outside the pipeline were found.")
        self._update_ok()

    def _add_row(self, fj: FoundJob) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        name = _short(fj.name) + (f"  ({fj.alias.rstrip('/').split('/', 1)[-1]})" if fj.alias != "None" else "")
        item = QTableWidgetItem(name)
        item.setData(Qt.UserRole, fj.name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.table.setItem(r, 0, item)
        type_text = self.type_names.get(fj.type_label, fj.type_label) + (" (tomo)" if fj.is_tomo else "")
        self.table.setItem(r, 1, QTableWidgetItem(type_text))
        combo = QComboBox()
        combo.addItems(list(STATUS_LABELS))
        combo.setCurrentText(fj.status)
        self.table.setCellWidget(r, 2, combo)
        parents = []
        for n in fj.inputs:
            parts = decompose(n)
            p = _short(parts[0] + parts[1]) if parts else n
            if p not in parents:
                parents.append(p)
        self.table.setItem(r, 3, QTableWidgetItem(", ".join(parents) or "—"))
        self.table.setItem(r, 4, QTableWidgetItem(str(len(fj.outputs))))
        self.table.setItem(r, 5, QTableWidgetItem(", ".join(fj.sources)))
        self.table.setItem(r, 6, QTableWidgetItem("; ".join(fj.notes)))
        tip = (f"{fj.name}\n\nInputs:\n" + ("\n".join(f"  {n}" for n in fj.inputs) or "  none")
               + "\n\nOutputs:\n" + ("\n".join(f"  {n}  [{label}]" for n, label, _ in fj.outputs) or "  none")
               + ("\n\nNotes:\n" + "\n".join(f"  {t}" for t in fj.notes) if fj.notes else ""))
        for c in range(len(COLUMNS)):
            it = self.table.item(r, c)
            if it is not None:
                it.setToolTip(tip)

    def _update_ok(self) -> None:
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.chosen()))

    def chosen(self) -> list[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def statuses(self) -> dict[str, str]:
        return {self.table.item(r, 0).data(Qt.UserRole): self.table.cellWidget(r, 2).currentText()
                for r in range(self.table.rowCount())}

    # -- a folder of a command run by hand
    def adopt(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder to add as a job", str(self.project_dir))
        if not d:
            return
        folder = Path(d)
        try:
            rel = os.path.relpath(folder.resolve(), self.project_dir.resolve()).replace(os.sep, "/")
        except ValueError:
            rel = "../"
        if rel.startswith("../") or rel == ".":
            QMessageBox.warning(self, "Add folder as job", f"{folder} is not inside the project.")
            return
        labels = [f"{t.dirname}  ({t.label})" for t in JOB_TYPES]
        guess = next((i for i, t in enumerate(JOB_TYPES) if t.dirname == rel.split("/", 1)[0]), 0)
        choice, ok = QInputDialog.getItem(self, "Add folder as job", f"Job type of {rel}:", labels, guess, False)
        if not ok:
            return
        jt = JOB_TYPES[labels.index(choice)]
        try:
            newest = max((e.stat().st_mtime for e in os.scandir(folder) if e.is_file()), default=0.0)
        except OSError:
            newest = 0.0
        warn = ("\n\nIts files changed in the last minutes: if the command is still writing there, wait for it to "
                "finish first." if time.time() - newest < RECENT_S else "")
        if QMessageBox.question(
                self, "Add folder as job",
                f"Move {rel} to {jt.dirname}/jobNNN/ (the next free job number)?\n\nA link is left at {rel}, so "
                f"paths to its files in other files still work.{warn}") != QMessageBox.Yes:
            return
        try:
            name = adopt_folder(self.project_dir, folder, jt.label)
        except (RuntimeError, OSError) as e:  # also the pipeline locked by RELION, or not readable just now
            QMessageBox.warning(self, "Add folder as job", str(e))
            return
        self.result = _busy(self.rescan, parent=self)
        self.fill()
        QMessageBox.information(self, "Add folder as job", f"{rel} is now {_short(name)}.")


def run_rebuild(project_dir: Path, defs, new: bool, pipeline: Pipeline | None = None,
                type_names: dict[str, str] | None = None, parent: QWidget | None = None,
                write: Callable[[Scan, list[str], dict[str, str]], object] | None = None) -> list[str] | None:
    """Scan, review and write. ``new``: a new default_pipeline.star; else add to ``pipeline``.

    ``write``: how to write the chosen jobs (default: now, here); the main window passes its own,
    which writes in the background. Returns the jobs chosen, or None when cancelled."""
    project_dir = Path(project_dir)

    def rescan() -> Scan:
        current = None if new else Pipeline.read(project_dir)
        return scan(project_dir, current, defs=defs)

    result = _busy(rescan, parent=parent) if pipeline is None or new else \
        _busy(lambda: scan(project_dir, pipeline, defs=defs), parent=parent)
    dlg = RebuildDialog(project_dir, result, rescan, new, type_names, parent)
    if dlg.exec() != QDialog.Accepted:
        return None
    jobs, statuses = dlg.chosen(), dlg.statuses()
    if write is not None:
        write(dlg.result, jobs, statuses)
        return jobs
    try:
        return _busy(lambda: write_pipeline(project_dir, dlg.result, jobs, statuses, new=new),
                     "Writing default_pipeline.star…", parent)
    except (RebuildError, OSError, RuntimeError) as e:
        QMessageBox.critical(parent, "cryo-savanna", f"The pipeline was not written: {e}")
        return None


def ask_generate(folder: Path, n_jobs: int, readonly: bool, parent: QWidget | None = None,
                 other_folder: bool = True) -> str:
    """'generate', 'other' or 'quit' (or 'cancel' without the other-folder buttons)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("cryo-savanna: no default_pipeline.star")
    box.setText(f"{folder}\n\nhas {n_jobs} RELION job folder(s) but no default_pipeline.star (it was lost, or the "
                "jobs were run from the command line).")
    buttons = {}
    if readonly:
        box.setInformativeText("With --readonly the pipeline cannot be generated here. Run cryo-savanna without "
                               "--readonly to generate it from the job folders.")
    else:
        box.setInformativeText("Generate default_pipeline.star from the job folders and their logs? You can "
                               "check the jobs found before anything is written.")
        buttons[box.addButton("Yes, generate it…", QMessageBox.ButtonRole.AcceptRole)] = "generate"
    if other_folder:
        other = box.addButton("No, open a different folder…", QMessageBox.ButtonRole.ActionRole)
        buttons[other] = "other"
        quit_ = box.addButton("No, close cryo-savanna", QMessageBox.ButtonRole.RejectRole)
        buttons[quit_] = "quit"
    else:
        quit_ = box.addButton(QMessageBox.Cancel)
        buttons[quit_] = "cancel"
    box.setDefaultButton(next(iter(buttons)))
    box.setEscapeButton(quit_)
    box.exec()
    return buttons.get(box.clickedButton(), "quit" if other_folder else "cancel")
