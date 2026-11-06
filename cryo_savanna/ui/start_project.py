# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""At startup, in a directory that is not a RELION project: start one there, open another, or quit.

A directory with RELION job folders but no default_pipeline.star (lost, or the jobs were run from the
command line) is offered a pipeline generated from those folders instead (project/rebuild.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from cryo_savanna.project.new_project import NewProjectError, create_project, is_project
from cryo_savanna.project.rebuild import find_job_dirs

NEW, OTHER, QUIT, GENERATE = "new", "other", "quit", "generate"


def ask(folder: Path, readonly: bool, parent: QWidget | None = None) -> str:
    """The question RELION's GUI asks, with a third answer: open a different folder."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("cryo-savanna: not a RELION project")
    box.setText(f"{folder}\n\ndoes not look like a RELION project directory (it has no default_pipeline.star).")
    if readonly:
        box.setInformativeText("With --readonly no project can be started here. Open a different folder?")
    else:
        box.setInformativeText("Do you want to start a new RELION project here?")
    buttons = {}
    if not readonly:
        buttons[box.addButton("Yes, start a new project here", QMessageBox.ButtonRole.AcceptRole)] = NEW
    other = box.addButton("No, open a different folder…", QMessageBox.ButtonRole.ActionRole)
    buttons[other] = OTHER
    quit_ = box.addButton("No, close cryo-savanna", QMessageBox.ButtonRole.RejectRole)
    buttons[quit_] = QUIT
    box.setDefaultButton(other if readonly else next(iter(buttons)))
    box.setEscapeButton(quit_)
    box.exec()
    return buttons.get(box.clickedButton(), QUIT)


def pick_folder(folder: Path, parent: QWidget | None = None) -> Path | None:
    d = QFileDialog.getExistingDirectory(parent, "Open RELION project (folder with default_pipeline.star)",
                                         str(folder))
    return Path(d) if d else None


def show_error(text: str, parent: QWidget | None = None) -> None:
    QMessageBox.critical(parent, "cryo-savanna: new project", text)


def ask_generate(folder: Path, n_jobs: int, readonly: bool) -> str:
    from cryo_savanna.ui.rebuild_dialog import ask_generate as ask

    return ask(folder, n_jobs, readonly)


def generate(folder: Path) -> bool:
    """Review the jobs found in ``folder`` and write its default_pipeline.star; False when cancelled."""
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.ui.rebuild_dialog import run_rebuild

    return run_rebuild(folder, JobDefs.load(), new=True) is not None


def choose_project(
    folder: Path,
    readonly: bool = False,
    ask: Callable[[Path, bool], str] = ask,
    pick_folder: Callable[[Path], Path | None] = pick_folder,
    create: Callable[[Path], str] = create_project,
    show_error: Callable[[str], None] = show_error,
    ask_generate: Callable[[Path, int, bool], str] = ask_generate,
    generate: Callable[[Path], bool] = generate,
) -> Path | None:
    """The project to open, starting from ``folder``; None to close cryo-savanna.

    A different folder that is not a project either gets the same question; cancelling the folder
    dialog asks again about the folder before. A folder with job folders is offered a generated
    pipeline instead of a new project (its job numbers would clash with the new project's).
    """
    folder = Path(folder)
    while not is_project(folder):
        jobs = find_job_dirs(folder)
        answer = ask_generate(folder, len(jobs), readonly) if jobs else ask(folder, readonly)
        if answer == GENERATE and jobs and not readonly:
            generate(folder)  # cancelled or failed: asked again
        elif answer == NEW and not readonly and not jobs:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))  # relion takes a few seconds
            try:
                how, error = create(folder), None
            except NewProjectError as e:
                how, error = None, str(e)
            finally:
                QApplication.restoreOverrideCursor()
            if error is None and not is_project(folder):
                error = f"No default_pipeline.star was made in {folder}."
            if error is not None:
                show_error(error)  # and ask again
                continue
            print(f"cryo-savanna: new RELION project in {folder} ({how})", file=sys.stderr)
        elif answer == OTHER:
            picked = pick_folder(folder)
            if picked is not None:
                folder = picked.resolve()
        else:
            return None
    return folder

