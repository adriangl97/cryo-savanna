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
from functools import partial

from qtpy.QtCore import QPoint
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAction, QMainWindow, QMenu

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import aretomo_jobs, csparc_jobs, membrain_jobs, split_jobs, tomomask_jobs
from cryo_savanna.jobdefs.cli_catalog import CATEGORIES, describe
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import DESCRIPTIONS as PYTOM_DESCRIPTIONS
from cryo_savanna.jobdefs.pytom_jobs import MENU_TEXT as PYTOM_MENU_TEXT
from cryo_savanna.ui.actions.base import ActionsBase, EditorWindow
from cryo_savanna.ui.actions.forms import FormActions
from cryo_savanna.ui.actions.pipeline_ops import PipelineActions
from cryo_savanna.ui.actions.running import RunActions, RunScheduledDialog
from cryo_savanna.ui.actions.selection import SelectionActions
from cryo_savanna.ui.new_job_menu import Entry, NewJobMenu

__all__ = ["JobActions", "EditorWindow", "RunScheduledDialog"]


def explain(action: QAction, tip: str, why_not: str = "") -> None:
    """Set an action's tooltip: what it does, its shortcut, and (while it is greyed out) why."""
    keys = action.shortcut().toString(QKeySequence.NativeText)
    text = f"{tip} ({keys})" if keys else tip
    if why_not and not action.isEnabled():
        text += f"\nNot now: {why_not}"
    action.setToolTip(text)


class JobActions(FormActions, SelectionActions, RunActions, PipelineActions, ActionsBase):
    """Owns the job menus, the editor dock and the background operations of a MainWindow."""

    def __init__(self, window: QMainWindow, defs: JobDefs, env: RelionEnv, type_names: dict[str, str],
                 tomo: bool, readonly: bool, flag_source=None):
        super().__init__(window, defs, env, type_names, tomo, readonly, flag_source)
        self._tips: dict[QAction, str] = {}  # what each action does: its tooltip, before the shortcut and why not
        self._build_actions()

    # ------------------------------------------------------------------ actions and menus
    def _act(self, text: str, slot: Callable, shortcut: str | None = None, tip: str = "") -> QAction:
        a = QAction(text, self.window)
        if shortcut:
            a.setShortcut(shortcut)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        self._tips[a] = tip or text.rstrip("…")
        a.triggered.connect(slot)
        self.window.addAction(a)
        return a

    def _fill_new_menu(self) -> None:
        """RELION's jobs (SPA or tomography), the other programs and the command-line tools (new_job_menu.py)."""
        jobs = [Entry(f"job:{v.key}", label, partial(self.new_job, v.key)) for label, v in self.defs.menu(self.tomo)]
        self.new_menu.set_entries(jobs, self._other_entries, self._cli_entries, self.new_cli_menu)

    def set_tomo(self, tomo: bool) -> None:
        """Switch the New job menu between RELION's SPA and tomography job lists."""
        if tomo != self.tomo:
            self.tomo = tomo
            self._fill_new_menu()

    def _build_actions(self) -> None:
        self.new_menu = NewJobMenu("New job", self.window)
        # the toolbar's button (it opens new_menu): enabled, and its tooltip, as the other actions. Not
        # QAction.setMenu: new_menu is also in the menu bar, and macOS's menus take a menu in one place only
        self.act_new = QAction("New job", self.window)
        self._tips[self.act_new] = "RELION's jobs, other programs (pytom, AreTomo3, ...) and RELION's command-line tools"
        self.act_clone = self._act("Clone…", self.clone_job, "Ctrl+D", "New job with the settings of the selected job")
        self.act_continue = self._act("Continue…", self.continue_job, tip="Continue the selected job")
        self.act_overwrite = self._act("Overwrite…", self.overwrite_job,
                                       tip="Run the selected job again with new settings in its own directory "
                                           "(the current one goes to Trash/)")
        self.act_run = self._act("Run scheduled…", self.run_scheduled, "Ctrl+R", "Run scheduled jobs")
        self.act_rerun = self._act("Run out-of-date jobs again…", lambda: self.rerun_out_of_date(),
                                   tip="Run finished jobs again, with their own settings, whose inputs changed after "
                                       "they ran (or that depend on such a job)")
        self.act_stop_runs = self._act("Stop runs…", self.stop_runs,
                                       tip="Stop a repeating run after its current job, or a run that waits for "
                                           "the input files of its next job")
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
        # the same tools in New job: a menu of its own (macOS's menus take a menu in one place only)
        self.new_cli_menu = QMenu("RELION command-line tools", self.window)
        self.new_cli_menu.setToolTipsVisible(True)
        new_subs = {c: self.new_cli_menu.addMenu(c) for c in CATEGORIES}
        for m in new_subs.values():
            m.setToolTipsVisible(True)
        self._cli_entries: list[Entry] = []
        for program in self.cli.programs():
            category, description = describe(program)
            a = subs[category].addAction(program)
            a.setToolTip(description)
            a.setStatusTip(description)
            a.triggered.connect(lambda _=False, p=program: self.new_cli_job(p))
            a.triggered.connect(lambda _=False, p=program: self.new_menu.remember(f"cli:{p}"))
            new_subs[category].addAction(a)
            self._cli_entries.append(Entry(f"cli:{program}", program, partial(self.new_cli_job, program),
                                           description))
        for m in subs.values():
            m.setToolTipsVisible(True)

        # programs that are not RELION's, run as External jobs: in New job, and a menu of their own in the menu bar
        self.external_menu = QMenu("Additional external jobs", self.window)
        self.external_menu.setToolTipsVisible(True)
        self._other_entries: list[Entry] = []
        groups = [[(f"pytom:{k}", t, PYTOM_DESCRIPTIONS[k], partial(self.new_pytom_job, k))
                   for k, t in PYTOM_MENU_TEXT.items()]  # template matching, with the extraction of its candidates
                  + [(f"tomomask:{k}", t, tomomask_jobs.DESCRIPTIONS[k], partial(self.new_tomomask_job, k))
                     for k, t in tomomask_jobs.MENU_TEXT.items()],  # pytom's tomogram masks
                  [(f"aretomo:{k}", t, aretomo_jobs.DESCRIPTIONS[k], partial(self.new_aretomo_job, k))
                   for k, t in aretomo_jobs.MENU_TEXT.items()]
                  + [(f"membrain:{k}", t, membrain_jobs.DESCRIPTIONS[k], partial(self.new_membrain_job, k))
                     for k, t in membrain_jobs.MENU_TEXT.items()],  # membranes in the tomograms
                  [(f"split:{k}", t, split_jobs.DESCRIPTIONS[k], partial(self.new_split_job, k))
                   for k, t in split_jobs.MENU_TEXT.items()]  # the particles a failed refinement stopped on
                  + [(f"csparc:{k}", t, csparc_jobs.DESCRIPTIONS[k], partial(self.new_csparc_job, k))
                     for k, t in csparc_jobs.MENU_TEXT.items()]]  # CryoSPARC's particles into the project
        for n, group in enumerate(groups):
            if n:
                self.external_menu.addSeparator()
            for entry_id, text, tip, start in group:
                a = self.external_menu.addAction(text)
                a.setToolTip(tip)
                a.setStatusTip(tip)
                a.triggered.connect(lambda _=False, f=start: f())
                a.triggered.connect(lambda _=False, i=entry_id: self.new_menu.remember(i))
                self._other_entries.append(Entry(entry_id, text, start, tip))
        self._fill_new_menu()

        self.job_menu = QMenu("Job", self.window)
        self.job_menu.addMenu(self.new_menu)
        for a in (self.act_clone, self.act_continue, self.act_overwrite, self.act_run, self.act_rerun,
                  self.act_stop_runs, None,
                  self.act_abort, self.act_alias, self.act_star, self.act_finished, self.act_failed, None,
                  self.act_gentle, self.act_harsh, None, self.act_delete, self.act_delete_all, self.act_undelete):
            self.job_menu.addSeparator() if a is None else self.job_menu.addAction(a)

    def context_menu(self, job: str | None, pos: QPoint) -> None:
        if job is not None:
            self.select.emit(job)
        m = QMenu(self.window)
        m.addMenu(self.new_menu)
        for a in (self.act_clone, self.act_continue, self.act_overwrite, self.act_run, self.act_rerun, None,
                  self.act_abort, self.act_alias,
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
        self.act_new.setEnabled(writable)
        self.cli_menu.setEnabled(writable)
        self.new_cli_menu.setEnabled(writable)
        self.external_menu.setEnabled(writable)
        self.act_clone.setEnabled(writable and info is not None)
        self.act_continue.setEnabled(writable and has_pipeliner and info is not None and status != "Running")
        self.act_overwrite.setEnabled(self.act_continue.isEnabled())
        scheduled = sum(1 for j in snap.jobs.values() if j.status == "Scheduled") if snap is not None else 0
        self.act_run.setEnabled(writable and has_pipeliner and scheduled > 0)
        self.act_run.setText(f"Run scheduled ({scheduled})…" if scheduled else "Run scheduled…")
        self.act_rerun.setEnabled(writable and has_pipeliner and bool(self.stale))
        self.act_stop_runs.setEnabled(writable and self._stoppable > 0)
        self.act_stop_runs.setText(f"Stop runs ({self._stoppable})…" if self._stoppable else "Stop runs…")
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

        # tooltips: what the action does, its shortcut, and why it cannot be used now
        no_project = "No project open" if snap is None else ""
        readonly = "The project is open read-only (--readonly)" if self.readonly else ""
        no_pipeliner = "" if has_pipeliner else "relion_pipeliner not found: load RELION 5.1"
        no_job = "Select a job first" if info is None else ""
        running = "The selected job is running" if status == "Running" else ""
        reasons = {
            self.act_new: (no_project, readonly),
            self.act_clone: (no_project, readonly, no_job),
            self.act_continue: (no_project, readonly, no_pipeliner, no_job, running),
            self.act_overwrite: (no_project, readonly, no_pipeliner, no_job, running),
            self.act_run: (no_project, readonly, no_pipeliner, "" if scheduled else "No job is scheduled"),
            self.act_stop_runs: (no_project, readonly, "" if self._stoppable else "No run that can be stopped is going"),
            self.act_rerun: (no_project, readonly, no_pipeliner, "" if self.stale else "No job is out of date"),
            self.act_abort: (no_project, readonly, no_job, "" if status in (None, "Running") else "The job is not running"),
        }
        for a in self._tips:
            explain(a, self._tips[a], next((r for r in reasons.get(a, ()) if r), ""))
        self.editor.set_permissions(self.readonly, has_pipeliner, why)
