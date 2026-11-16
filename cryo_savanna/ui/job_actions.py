# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Everything that changes a project: new / clone / continue / run / schedule jobs, abort, clean,
alias, mark finished or failed, delete and undelete. Also the "CLI tools" menu: RELION's
command-line programs as External jobs (jobdefs/cli_tools.py), and the "Additional external jobs" menu.

``JobActions`` owns the menus and enables what makes sense; the actions themselves are in ui/actions/, by
group (forms, selection windows, running, pipeline operations), with what they share in ui/actions/base.py.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QAction, QMainWindow, QMenu

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import aretomo_jobs, tomomask_jobs
from cryo_savanna.jobdefs.cli_catalog import CATEGORIES, describe
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import DESCRIPTIONS as PYTOM_DESCRIPTIONS
from cryo_savanna.jobdefs.pytom_jobs import EXTRACT as PYTOM_EXTRACT
from cryo_savanna.jobdefs.pytom_jobs import MATCH as PYTOM_MATCH
from cryo_savanna.ui.actions.base import ActionsBase, EditorWindow
from cryo_savanna.ui.actions.forms import FormActions
from cryo_savanna.ui.actions.pipeline_ops import PipelineActions
from cryo_savanna.ui.actions.running import RunActions, RunScheduledDialog
from cryo_savanna.ui.actions.selection import SelectionActions

__all__ = ["JobActions", "EditorWindow", "RunScheduledDialog"]


class JobActions(FormActions, SelectionActions, RunActions, PipelineActions, ActionsBase):
    """Owns the job menus, the editor dock and the background operations of a MainWindow."""

    def __init__(self, window: QMainWindow, defs: JobDefs, env: RelionEnv, type_names: dict[str, str],
                 tomo: bool, readonly: bool, flag_source=None):
        super().__init__(window, defs, env, type_names, tomo, readonly, flag_source)
        self._build_actions()

    # ------------------------------------------------------------------ actions and menus
    def _act(self, text: str, slot: Callable, shortcut: str | None = None, tip: str = "") -> QAction:
        a = QAction(text, self.window)
        if shortcut:
            a.setShortcut(shortcut)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(slot)
        self.window.addAction(a)
        return a

    def _fill_new_menu(self) -> None:
        self.new_menu.clear()
        for label, variant in self.defs.menu(self.tomo):
            a = self.new_menu.addAction(label)
            a.triggered.connect(lambda _=False, v=variant: self.new_job(v.key))

    def set_tomo(self, tomo: bool) -> None:
        """Switch the New job menu between RELION's SPA and tomography job lists."""
        if tomo != self.tomo:
            self.tomo = tomo
            self._fill_new_menu()

    def _build_actions(self) -> None:
        self.new_menu = QMenu("New job", self.window)
        self._fill_new_menu()
        self.act_clone = self._act("Clone…", self.clone_job, "Ctrl+D", "New job with the settings of the selected job")
        self.act_continue = self._act("Continue…", self.continue_job, tip="Continue the selected job")
        self.act_overwrite = self._act("Overwrite…", self.overwrite_job,
                                       tip="Run the selected job again with new settings in its own directory "
                                           "(the current one goes to Trash/)")
        self.act_run = self._act("Run scheduled…", self.run_scheduled, "Ctrl+R", "Run scheduled jobs")
        self.act_stop_runs = self._act("Stop repeating runs…", self.stop_runs,
                                       tip="Stop a repeating run of scheduled jobs after its current job")
        self.act_abort = self._act("Abort…", self.abort_job,
                                   tip="Ask the selected running job to stop; a job run on this machine (not in a "
                                       "queue) can also be killed (force abort)")
        self.act_alias = self._act("Set alias…", self.set_alias, "F2")
        self.act_star = self._act("Star", lambda on: self.current and self.star.emit(self.current, on), "Ctrl+B",
                                  "Mark the job, to find it with the Starred filter (for everyone who opens "
                                  "the project)")
        self.act_star.setCheckable(True)
        self.act_finished = self._act("Mark as finished", lambda: self.mark(False))
        self.act_failed = self._act("Mark as failed", lambda: self.mark(True))
        self.act_gentle = self._act("Clean (gentle)", lambda: self.clean(False),
                                    tip="Delete intermediate files RELION does not need any more")
        self.act_harsh = self._act("Clean (harsh)", lambda: self.clean(True),
                                   tip="Also delete files that are only needed to continue the job")
        self.act_delete = self._act("Delete…", lambda: self.delete(False), "Del", "Move the job to Trash/")
        self.act_delete_all = self._act("Delete with all downstream jobs…", lambda: self.delete(True))
        self.act_undelete = self._act("Undelete…", self.undelete, tip="Put a job back from Trash/")
        self.act_disk = self._act("Disk usage…", self.show_disk_usage,
                                  tip="How much space each job, job type and Trash/ take; clean jobs, empty Trash/")
        self.act_import = self._act("Add jobs run outside the pipeline…", self.import_jobs,
                                    tip="Add job folders the pipeline does not have (jobs run from the command line), "
                                        "read from their job.star and logs")

        self.cli_menu = QMenu("CLI tools", self.window)
        self.cli_menu.setToolTipsVisible(True)
        subs = {c: self.cli_menu.addMenu(c) for c in CATEGORIES}
        for program in self.cli.programs():
            category, description = describe(program)
            a = subs[category].addAction(program)
            a.setToolTip(description)
            a.setStatusTip(description)
            a.triggered.connect(lambda _=False, p=program: self.new_cli_job(p))
        for m in subs.values():
            m.setToolTipsVisible(True)

        # programs that are not RELION's, run as External jobs (a button of their own in the main toolbar)
        self.external_menu = QMenu("Additional external jobs", self.window)
        self.external_menu.setToolTipsVisible(True)
        for kind, text in ((PYTOM_MATCH, "pytom: template matching"), (PYTOM_EXTRACT, "pytom: extract candidates")):
            a = self.external_menu.addAction(text)
            a.setToolTip(PYTOM_DESCRIPTIONS[kind])
            a.setStatusTip(PYTOM_DESCRIPTIONS[kind])
            a.triggered.connect(lambda _=False, k=kind: self.new_pytom_job(k))
        for kind, text in tomomask_jobs.MENU_TEXT.items():  # pytom's tomogram masks
            a = self.external_menu.addAction(text)
            a.setToolTip(tomomask_jobs.DESCRIPTIONS[kind])
            a.setStatusTip(tomomask_jobs.DESCRIPTIONS[kind])
            a.triggered.connect(lambda _=False, k=kind: self.new_tomomask_job(k))
        self.external_menu.addSeparator()
        for kind, text in aretomo_jobs.MENU_TEXT.items():
            a = self.external_menu.addAction(text)
            a.setToolTip(aretomo_jobs.DESCRIPTIONS[kind])
            a.setStatusTip(aretomo_jobs.DESCRIPTIONS[kind])
            a.triggered.connect(lambda _=False, k=kind: self.new_aretomo_job(k))

        self.job_menu = QMenu("Job", self.window)
        self.job_menu.addMenu(self.new_menu)
        for a in (self.act_clone, self.act_continue, self.act_overwrite, self.act_run, self.act_stop_runs, None,
                  self.act_abort, self.act_alias, self.act_star, self.act_finished, self.act_failed, None,
                  self.act_gentle, self.act_harsh, None, self.act_delete, self.act_delete_all, self.act_undelete):
            self.job_menu.addSeparator() if a is None else self.job_menu.addAction(a)

    def context_menu(self, job: str | None, pos: QPoint) -> None:
        if job is not None:
            self.select.emit(job)
        m = QMenu(self.window)
        m.addMenu(self.new_menu)
        for a in (self.act_clone, self.act_continue, self.act_overwrite, self.act_run, None, self.act_abort, self.act_alias,
                  self.act_star, self.act_finished, self.act_failed, None, self.act_gentle, self.act_harsh, None,
                  self.act_delete, self.act_delete_all):
            m.addSeparator() if a is None else m.addAction(a)
        m.exec(pos)
        m.deleteLater()  # a child of the window: it would stay until the window closes

    def update_state(self) -> None:
        """Enable what makes sense for the selected job, the project and this shell."""
        snap, job = self.snapshot, self.current
        info = snap.jobs.get(job) if snap is not None and job else None
        status = info.status if info is not None else None
        writable = not self.readonly and snap is not None
        has_pipeliner = self.pipeliner is not None and self.pipeliner.available
        why = ("read-only (--readonly)" if self.readonly else
               "" if has_pipeliner else "relion_pipeliner not found: load RELION 5.1")
        self.new_menu.setEnabled(writable)
        self.cli_menu.setEnabled(writable)
        self.external_menu.setEnabled(writable)
        self.act_clone.setEnabled(writable and info is not None)
        self.act_continue.setEnabled(writable and has_pipeliner and info is not None and status != "Running")
        self.act_overwrite.setEnabled(self.act_continue.isEnabled())
        scheduled = snap is not None and any(j.status == "Scheduled" for j in snap.jobs.values())
        self.act_run.setEnabled(writable and has_pipeliner and scheduled)
        self.act_stop_runs.setEnabled(writable and self._has_repeating)
        self.act_abort.setEnabled(writable and status == "Running")
        for a in (self.act_alias, self.act_finished, self.act_failed, self.act_delete, self.act_delete_all):
            a.setEnabled(writable and info is not None)
        self.act_gentle.setEnabled(writable and has_pipeliner and info is not None and status != "Running")
        self.act_harsh.setEnabled(self.act_gentle.isEnabled())
        self.act_undelete.setEnabled(writable and self._has_trash)
        self.act_import.setEnabled(writable)
        self.act_disk.setEnabled(snap is not None)  # read-only too: only to look
        self.act_star.setEnabled(info is not None)  # read-only too: for the session only (project/stars.py)
        self.act_star.setChecked(info is not None and info.starred)
        self.editor.set_permissions(self.readonly, has_pipeliner, why)
