# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Everything that changes a project: new / clone / continue / run / schedule jobs, abort, clean,
alias, mark finished or failed, delete and undelete. Also the "CLI tools" menu: RELION's
command-line programs as External jobs (jobdefs/cli_tools.py).

Jobs are added and run through relion_pipeliner (project/pipeliner.py); the few operations it
cannot do go through PipelineOps (project/ops.py). Both can wait for RELION's lock, so they run
in a thread pool and report back with signals; the window stays responsive meanwhile.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import QObject, QPoint, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.relion_env import RelionEnv, RelionEnvError
from cryo_savanna.jobdefs.cli_catalog import CATEGORIES, describe
from cryo_savanna.jobdefs.cli_tools import CliCatalog, CliVariant, cli_program_of
from cryo_savanna.jobdefs.model import JobDefs, JobVariant
from cryo_savanna.jobdefs.pytom_jobs import TITLES as PYTOM_TITLES
from cryo_savanna.jobdefs.pytom_jobs import warm_defaults as warm_pytom_defaults
from cryo_savanna.jobdefs.pytom_jobs import (DESCRIPTIONS as PYTOM_DESCRIPTIONS, EXTRACT as PYTOM_EXTRACT,
                                             MATCH as PYTOM_MATCH, PytomCatalog, PytomVariant, last_settings,
                                             pytom_kind_of, quoted_values)
from cryo_savanna.project.jobstar import JobStar, read_last_used
from cryo_savanna.project.ops import OpsError, PipelineOps
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import Pipeliner, PipelinerError
from cryo_savanna.project.rebuild import RebuildError
from cryo_savanna.project.scratch import ENV_VAR as SCRATCH_VAR
from cryo_savanna.project.scratch import SCRATCH_KEY
from cryo_savanna.project.scratch import assign as assign_scratch
from cryo_savanna.project.selection import (
    SelectionError,
    interactive_kind,
    read_backup_selection,
    read_class_input,
    read_tilt_exclusion,
    read_tilt_input,
    relion_command,
    selected_numbers,
    write_class_selection,
    write_tilt_exclusion,
)
from cryo_savanna.project.snapshot import ProjectSnapshot
from cryo_savanna.relion_consts import RELION_JOB_EXIT_ABORTED, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_SUCCESS
from cryo_savanna.ui import style
from cryo_savanna.ui.busy import run_busy
from cryo_savanna.ui.job_editor import MODE_CONTINUE, MODE_OVERWRITE, JobEditor


class EditorWindow(QWidget):
    """The job editor in its own window (Preferences → Job editor), instead of the dock."""

    shown = Signal(bool)

    def __init__(self, window):
        super().__init__(window, Qt.Window)  # its own window, closed with the main one
        self.setWindowTitle("Job editor")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.placed = False  # given a size and a place (settings, or next to the main window)

    def showEvent(self, e) -> None:
        if not self.placed:
            self.placed = True
            main = self.parentWidget().frameGeometry()
            self.resize(760, max(600, min(900, main.height())))
            self.move(main.center() - self.rect().center())
        super().showEvent(e)
        self.shown.emit(True)

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        if not e.spontaneous():  # not when minimised
            self.shown.emit(False)


# ---------------------------------------------------------------------------- background tasks

class _Signals(QObject):
    done = Signal(object)
    error = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable, signals: _Signals):
        super().__init__()
        self.fn, self.signals = fn, signals

    def run(self) -> None:
        try:
            result = self.fn()
        except (OpsError, PipelinerError, RebuildError) as e:
            self.signals.error.emit(str(e))
        except Exception as e:  # anything else: report, never kill the GUI
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        else:
            self.signals.done.emit(result)


# ---------------------------------------------------------------------------- dialogs

class RunScheduledDialog(QDialog):
    """Pick scheduled jobs to run (in pipeline order) and how often, like RELION's scheduler."""

    def __init__(self, parent, jobs: list[tuple[str, str]], preselect: str | None):
        super().__init__(parent)
        self.setWindowTitle("Run scheduled jobs")
        self.list = QListWidget()
        for name, label in jobs:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if preselect in (None, name) else Qt.Unchecked)
            self.list.addItem(it)
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 10000)
        self.min_wait = QSpinBox()
        self.min_wait.setRange(0, 100000)
        self.min_wait_before = QSpinBox()
        self.min_wait_before.setRange(0, 100000)
        self.sec_wait_after = QSpinBox()
        self.sec_wait_after.setRange(1, 3600)
        self.sec_wait_after.setValue(10)
        form = QFormLayout()
        form.addRow("Run the jobs this many times:", self.repeat)
        form.addRow("Wait at least (min) between repeats:", self.min_wait)
        form.addRow("Wait (min) before the first job:", self.min_wait_before)
        form.addRow("Check every (s) whether a job finished:", self.sec_wait_after)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Jobs run one after the other, each waiting for its input files. "
                             "The run stops at the first failed or aborted job."))
        lay.addWidget(self.list)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def selected(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def run_args(self) -> dict:
        return dict(repeat=self.repeat.value(), min_wait=self.min_wait.value(),
                    min_wait_before=self.min_wait_before.value(), sec_wait_after=self.sec_wait_after.value())


# ---------------------------------------------------------------------------- controller

class JobActions(QObject):
    """Owns the job menus, the editor dock and the background operations of a MainWindow."""

    changed = Signal()  # the project was changed: refresh now
    select = Signal(str)  # select this job once it shows up
    message = Signal(str)  # short status message
    star = Signal(str, bool)  # star or unstar this job

    def __init__(self, window: QMainWindow, defs: JobDefs, env: RelionEnv, type_names: dict[str, str],
                 tomo: bool, readonly: bool, flag_source=None):
        super().__init__(window)
        self.window, self.defs, self.env, self.type_names = window, defs, env, type_names
        self.tomo, self.readonly = tomo, readonly
        self.project_dir: Path | None = None
        self.pipeliner: Pipeliner | None = None
        self.ops: PipelineOps | None = None
        self.snapshot: ProjectSnapshot | None = None
        self.current: str | None = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)  # one change at a time, in the order asked
        self._pending: set[_Signals] = set()
        self._structure = None
        self._has_trash = False  # looked at when the project changes, not on every selection
        self._has_repeating = False
        self._look_pool = QThreadPool(self)  # those looks read Trash/ and the run records: off the GUI thread
        self._look_pool.setMaxThreadCount(1)
        self._look_signals = _Signals()
        self._look_signals.done.connect(self._on_look)
        self._look_signals.error.connect(lambda msg: self._on_look(None))
        self._looking = self._look_again = False
        self.slurm = None  # slurm.SlurmMonitor of the project
        self._marked_lost: set[tuple[str, str]] = set()  # (job, SLURM id) already marked (mark_lost)
        self.queue = None  # slurm.QueueState, from the main window's monitor

        self.editor = JobEditor(env, type_names, flag_source=flag_source)
        self.cli = CliCatalog(defs, env, self.editor.flag_source)
        self.pytom = PytomCatalog(defs, env)
        warm_pytom_defaults()  # "module avail cuda", in the background: not when the first form opens
        self.editor.submit.connect(self._on_submit)
        # the editor goes in its own window or in the dock (Preferences → Job editor: set_editor_place)
        self.dock = QDockWidget("Job editor", window)
        self.dock.setObjectName("job_editor_dock")
        self.dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                              | QDockWidget.DockWidgetClosable)
        window.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self.editor_window = EditorWindow(window)
        self.editor_place = ""
        self.act_editor = QAction("Job editor", window)  # shows or hides it, wherever it is
        self.act_editor.setCheckable(True)
        self.act_editor.triggered.connect(lambda on: self._show_editor() if on else self.hide_editor())
        self.editor_window.addAction(self.act_editor)  # Ctrl+E there too (set by the main window)
        self.dock.visibilityChanged.connect(lambda _: self._sync_editor_action())
        self.editor_window.shown.connect(lambda _: self._sync_editor_action())
        self.set_editor_place("window")
        self.editor.closed.connect(self.hide_editor)
        self._edits = 0  # times the editor was filled: a job started from it closes it only if still shown

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
        self.act_abort = self._act("Abort", self.abort_job, tip="Ask the selected running job to stop")
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
        self.act_import = self._act("Add jobs run outside the pipeline…", self.import_jobs,
                                    tip="Add job folders the pipeline does not have (jobs run from the command line), "
                                        "read from their job.star and logs")
        self.tools: list[QMainWindow] = []  # open selection windows

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

    def set_project(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.pipeliner = Pipeliner(project_dir)
        self.ops = PipelineOps(project_dir)
        self.snapshot, self.current, self._structure = None, None, None
        self.editor.set_project(self.project_dir, None)
        self.hide_editor()
        self._look_at_project()
        self.update_state()

    def set_snapshot(self, snap: ProjectSnapshot) -> None:
        # a snapshot with the same pipeline only has new log stamps of running jobs (every poll while one
        # writes): Trash/ and the runs only change with the pipeline file
        pipeline_changed = self.snapshot is None or snap.pipeline is not self.snapshot.pipeline
        self.snapshot = snap
        key = snap.structure_key()
        if key != self._structure:  # new or changed jobs: refresh the input lists
            self._structure = key
            self.editor.update_pipeline(snap.pipeline)
        if pipeline_changed:
            self._look_at_project()
        self.update_state()

    def _look_at_project(self) -> None:
        """What Undelete and Stop repeating runs need: once per snapshot (a change), not per click.

        Read in the background (a big Trash/ over a slow file system); a change while a look is
        going is looked at again once it is done."""
        if self._looking:
            self._look_again = True
            return
        self._looking, self._look_again = True, False
        project, ops, pl = self.project_dir, self.ops, self.pipeliner
        self._look_pool.start(_Task(lambda: (project, *self._project_state(ops, pl)), self._look_signals))

    @staticmethod
    def _project_state(ops: PipelineOps | None, pl: Pipeliner | None) -> tuple[bool, bool]:
        """(anything to undelete, any repeating run going)."""
        return ops is not None and bool(ops.trash()), pl is not None and bool(pl.repeating_runs())

    def look_at_project_now(self) -> None:
        """_look_at_project, done at once (tests)."""
        self._has_trash, self._has_repeating = self._project_state(self.ops, self.pipeliner)
        self.update_state()

    def _on_look(self, result) -> None:
        """``result``: (project, has_trash, has_repeating), or None when the look failed (kept as it was)."""
        self._looking = False
        project, has_trash, has_repeating = result if result is not None else (None, False, False)
        if result is not None and project == self.project_dir:  # not one of the project open before
            self._has_trash, self._has_repeating = has_trash, has_repeating
            self.update_state()
        if self._look_again or (result is not None and project != self.project_dir):
            self._look_at_project()

    def set_slurm(self, monitor) -> None:
        self.slurm, self.queue = monitor, None

    def set_queue(self, state) -> None:
        self.queue = state

    def set_current(self, job: str | None) -> None:
        self.current = job
        self.update_state()

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
        self.act_star.setEnabled(info is not None)  # read-only too: for the session only (project/stars.py)
        self.act_star.setChecked(info is not None and info.starred)
        self.editor.set_permissions(self.readonly, has_pipeliner, why)

    # ------------------------------------------------------------------ helpers
    def _run(self, what: str, fn: Callable, done: Callable | None = None, on_error: Callable | None = None) -> None:
        signals = _Signals()
        self._pending.add(signals)

        def finished(result):
            self._pending.discard(signals)
            self.message.emit(f"{what}: done")
            if done is not None:
                done(result)
            self.changed.emit()

        def failed(msg: str):
            self._pending.discard(signals)
            self.message.emit(f"{what}: failed")
            if on_error is not None:
                on_error(msg)
            QMessageBox.warning(self.window, what, msg)
            self.changed.emit()

        signals.done.connect(finished)
        signals.error.connect(failed)
        self.message.emit(f"{what}…")
        self.pool.start(_Task(fn, signals))

    def wait(self, ms: int = 60000) -> bool:
        """Wait for queued operations (closing the window, tests); -1: as long as it takes."""
        return self.pool.waitForDone(ms)

    def _trash_warning(self, jobs: list[str]) -> str:
        """Text for a confirmation that moves ``jobs`` to Trash/, when that replaces earlier copies there."""
        old = self.ops.in_trash(jobs)
        if not old:
            return ""
        return ("\n\nWARNING: Trash/ already holds an earlier copy of " + ", ".join(j.rstrip("/") for j in old)
                + " (from an earlier overwrite or delete). It will be deleted for good and cannot be undeleted.")

    def _label(self, job: str) -> str:
        info = self.snapshot.jobs.get(job) if self.snapshot else None
        return style.job_label(job, info.process.alias) if info else job

    # ------------------------------------------------------------------ where the editor is
    def editor_container(self) -> QWidget:
        return self.dock if self.editor_place == "docked" else self.editor_window

    def editor_shown(self) -> bool:
        return not self.editor_container().isHidden()

    def set_editor_place(self, place: str) -> None:
        """"window" (its own window) or "docked" (in the main window); an open editor moves there."""
        place = "docked" if place == "docked" else "window"
        if place == self.editor_place:
            return
        was_shown = bool(self.editor_place) and self.editor_shown()
        self.hide_editor()
        self.editor_place = place
        if place == "docked":
            self.dock.setWidget(self.editor)
        else:
            self.dock.setWidget(None)
            self.editor_window.layout().addWidget(self.editor)
        if was_shown:
            self._show_editor()
        self._sync_editor_action()

    def hide_editor(self) -> None:
        self.dock.hide()
        self.editor_window.hide()

    def _show_editor(self) -> None:
        c = self.editor_container()
        c.show()
        c.raise_()
        if c is self.editor_window:
            c.activateWindow()
        self._sync_editor_action()

    def _sync_editor_action(self) -> None:
        self.act_editor.setChecked(self.editor_shown())

    def _edit(self, title: str, fn: Callable, *args) -> None:
        """Fill the editor with fn(*args) and show it; a RELION_* value RELION itself refuses is reported instead."""
        try:
            fn(*args)
        except RelionEnvError as e:
            self._env_error(title, e)
            return
        self._edits += 1
        self._show_editor()

    def _started(self, edit: int, text: str) -> None:
        """A job submitted from the editor started: close the editor, unless another job was opened in it
        meanwhile (Schedule keeps it open: see the callers)."""
        self.message.emit(text)
        self.editor.status.setText(text)
        if edit == self._edits:
            self.hide_editor()

    def _env_error(self, title: str, e: Exception) -> None:
        QMessageBox.warning(self.window, title, f"{e}.\n\nCorrect it in the shell (or the RELION module) that "
                            "starts cryo-savanna, then start it again.")

    # ------------------------------------------------------------------ new / clone / continue
    def new_job(self, variant_key: str) -> None:
        variant = self.defs.variants[variant_key]
        last = read_last_used(self.project_dir, variant.label, variant.is_tomo) if self.project_dir else None
        if last is not None and (cli_program_of(last) or pytom_kind_of(last)):  # remembered: a CLI tool's / pytom's
            last = None
        note = ("Settings of the last job of this type (from RELION's .gui files); 'Defaults' resets them."
                if last is not None else "")
        values = dict(last.options) if last else None
        if values and self.env.get(SCRATCH_VAR) and not self.env.from_shell(SCRATCH_VAR):
            values.pop(SCRATCH_KEY, None)  # the scratch directory of the Preferences, not the last job's
        self._edit("New job", self.editor.new_job, variant, values, note)

    def new_cli_job(self, program: str) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)  # the first time, the program's --help is read
        try:
            variant = self.cli.variant(program, self.tomo)
        except RelionEnvError as e:
            self._env_error(program, e)
            return
        finally:
            QApplication.restoreOverrideCursor()
        if variant is None:
            QMessageBox.warning(self.window, program, f"Cannot read the options of {program} (its --help).")
            return
        mpi = " With MPI, it runs only locally (not on the queue)." if variant.has_mpi else ""
        self._edit(program, self.editor.new_job, variant, None,
                   f"{variant.description}. Options from {variant.source}.{mpi}")

    def new_pytom_job(self, kind: str) -> None:
        """A pytom job (template matching or extraction): the form, with the pytom environment of the last one."""
        try:
            variant = self.pytom.variant(kind)
        except RelionEnvError as e:
            self._env_error("pytom", e)
            return
        values = variant.default_values(self.env)
        pipeline = self.snapshot.pipeline if self.snapshot is not None else None
        if self.project_dir is not None:
            # the snapshot knows what every External job runs: only the pytom ones are read
            pytom = [n for n, j in self.snapshot.jobs.items() if j.program in PYTOM_TITLES.values()] \
                if self.snapshot is not None else None
            values.update(last_settings(self.project_dir, pipeline, kind, pytom))
        if pipeline is not None:  # the newest input of the right kind, as a start
            values.update(_newest_inputs(pipeline, kind))
        self._edit(variant.program, self.editor.new_job, variant, values,
                   f"{variant.description}. Runs as a RELION External job (pytom from its conda environment).")

    def _variant_for(self, js: JobStar) -> JobVariant:
        """The form for a job: its job type's, or a CLI tool's or pytom's for External jobs made from those menus."""
        kind = pytom_kind_of(js)
        if kind is not None:
            return self.pytom.variant(kind)
        program = cli_program_of(js)
        if program is not None:
            v = self.cli.variant(program, js.is_tomo)
            if v is not None:
                return v
        return self.defs.variant_for_jobstar(js)

    def clone_job(self) -> None:
        job = self.current
        try:
            js = JobStar.read(self.project_dir / job / "job.star")
            variant = self._variant_for(js)
            values = (variant.values_from_jobstar(js, self.env)[0] if isinstance(variant, (CliVariant, PytomVariant))
                      else js.options)
        except RelionEnvError as e:
            self._env_error("Clone", e)
            return
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self.window, "Clone", f"Cannot read {job}job.star: {e}")
            return
        self._edit("Clone", self.editor.new_job, variant, values, f"Settings copied from {self._label(job)}.", job)

    def continue_job(self) -> None:
        job = self.current
        try:
            js = JobStar.read(self.project_dir / job / "job.star")
            variant = self._variant_for(js)
        except RelionEnvError as e:
            self._env_error("Continue", e)
            return
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self.window, "Continue", f"Cannot read {job}job.star: {e}")
            return
        self._edit("Continue", self.editor.continue_job, job, js, variant)

    def overwrite_job(self) -> None:
        job = self.current
        try:
            js = JobStar.read(self.project_dir / job / "job.star")
            variant = self._variant_for(js)
        except RelionEnvError as e:
            self._env_error("Overwrite", e)
            return
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self.window, "Overwrite", f"Cannot read {job}job.star: {e}")
            return
        pr = self.snapshot.jobs[job].process
        alias = pr.alias.rstrip("/").split("/", 1)[-1] if pr.has_alias else ""
        self._edit("Overwrite", self.editor.overwrite_job, job, js, variant, alias)

    def _submit_overwrite(self, js: JobStar, alias: str, schedule_only: bool) -> None:
        """RELION's "Overwrite!": ask, then replace the job by a new one in the same directory."""
        pl, job = self.pipeliner, self.editor.job
        # the latest snapshot, not a read of the pipeline here (a slow file system would freeze the
        # window); Pipeliner.overwrite_job checks again with the pipeline as it is then
        snap = self.snapshot
        if snap is None or job not in snap.jobs:
            QMessageBox.warning(self.window, "Overwrite", f"{job.rstrip('/')} is not in the pipeline any more.")
            return
        p = snap.pipeline
        if snap.jobs[job].status == "Running":
            QMessageBox.warning(self.window, "Overwrite", f"{self._label(job)} is running: abort it first.")
            return
        if alias:
            others = Pipeline(job_counter=p.job_counter, processes={n: pr for n, pr in p.processes.items() if n != job})
            try:
                PipelineOps.check_alias(others, job, alias)
            except OpsError as e:
                QMessageBox.warning(self.window, "Alias", str(e))
                return
        ask = (f"Overwrite {self._label(job)}?\n\nIts directory and files go to Trash/, and the job "
               f"{'is scheduled' if schedule_only else 'runs'} again as {job.rstrip('/')} with these settings.")
        users = p.children(job)
        if users:
            ask += ("\n\nWARNING: these jobs use its outputs, so their results will no longer match it:\n  "
                    + "\n  ".join(self._label(c) for c in users))
        ask += self._trash_warning([job])
        if QMessageBox.question(self.window, "Overwrite", ask) != QMessageBox.Yes:
            return

        scratch: list[str | None] = [None]  # the job's own scratch directory, once given
        own_scratch = self.editor.own_scratch

        def overwrite_and_run():
            if own_scratch:
                scratch[0] = self._own_scratch(js, p)
            res = pl.overwrite_job(job, js, alias or None)
            rec = None if schedule_only else pl.run([res.job])
            return res, rec

        edit = self._edits

        def done(result):
            res, rec = result
            self.select.emit(res.job)
            verb = "scheduled" if rec is None else "started"
            text = f"{res.job.rstrip('/')} overwritten and {verb}." + (" " + " ".join(res.warnings) if res.warnings else "")
            if scratch[0]:
                text += f" Scratch directory: {scratch[0]}"
            if rec is None:
                self.editor.status.setText(text)
            else:
                self._started(edit, text)

        self._run(f"Overwrite {self._label(job)}", overwrite_and_run, done)

    def _own_scratch(self, js: JobStar, pipeline: Pipeline | None) -> str | None:
        """Before a job is added (in the worker: it reads the project's job.star files): its scratch
        directory becomes a savanna_NNN of its own, so no two jobs wipe each other's relion_volatile/.
        Only with Preferences → "A subdirectory of its own for each job" (editor.own_scratch)."""
        labels = {v.label for v in self.defs.variants.values() if SCRATCH_KEY in v.options}
        try:
            pipeline = Pipeline.read(self.project_dir)  # with jobs added since the last snapshot
        except (OSError, ValueError, KeyError):
            pass
        return assign_scratch(js.options, self.project_dir, pipeline, labels)

    def _on_submit(self, js: JobStar, alias: str, schedule_only: bool) -> None:
        pl = self.pipeliner
        kind = interactive_kind(js.type_label, js.options)
        if kind and self.editor.native_selector and not schedule_only:
            job = self.editor.job if self.editor.mode in (MODE_CONTINUE, MODE_OVERWRITE) else None
            self.open_tool(kind, js, alias, self.editor.mode, job)
            return
        if isinstance(self.editor.variant, PytomVariant):
            bad = quoted_values(self.editor.values(), self.editor.variant.own_keys)
            if bad:
                QMessageBox.warning(self.window, self.editor.variant.program,
                                    "These options cannot have a double quote (\") in them (RELION's job.star "
                                    "keeps them in double quotes): " + ", ".join(
                                        self.editor.variant.options[k].label.rstrip(": ") for k in bad))
                return
        if isinstance(self.editor.variant, CliVariant):
            v = self.editor.values()
            if v.get("nr_mpi", "1") not in ("", "1") and v.get("do_queue") == "Yes":
                QMessageBox.warning(self.window, self.editor.variant.program,
                                    "MPI runs of CLI tools can only run locally: RELION's External jobs cannot ask "
                                    "the queue for MPI slots. Use 1 MPI process, or switch off 'Submit to queue'.")
                return
            try:
                limit = self.env.error_local_mpi
            except RelionEnvError as e:  # RELION would refuse to start
                self._env_error(self.editor.variant.program, e)
                return
            try:
                nr_mpi = int(float(v.get("nr_mpi") or 1))
            except ValueError:
                nr_mpi = 1  # cli_runner reports it
            if nr_mpi > limit:  # as relion_pipeliner refuses it for RELION's own jobs (cli_runner checks again)
                QMessageBox.warning(self.window, self.editor.variant.program,
                                    f"You're submitting a local job with {nr_mpi} parallel MPI processes. That's more "
                                    f"than allowed by the RELION_ERROR_LOCAL_MPI environment variable ({limit}).")
                return
        if self.editor.mode == MODE_OVERWRITE:
            self._submit_overwrite(js, alias, schedule_only)
            return
        if self.editor.mode == MODE_CONTINUE:
            job = self.editor.job
            info = self.snapshot.jobs.get(job) if self.snapshot else None
            if info is not None and info.status == "Running":
                QMessageBox.warning(self.window, "Continue", f"{self._label(job)} is still running.")
                return
            edit = self._edits
            self._run(f"Continue {self._label(job)}", lambda: pl.continue_job(job, js),
                      lambda _: self._started(edit, f"{job.rstrip('/')} continued."))
            return
        if alias and self.snapshot is not None:  # relion_pipeliner --setJobAlias checks again
            try:
                PipelineOps.check_alias(self.snapshot.pipeline, f"{self.editor.variant.dirname}/job000/", alias)
            except OpsError as e:
                QMessageBox.warning(self.window, "Alias", str(e))
                return

        variant, values = self.editor.variant, self.editor.values()
        predicted = self.editor.job_dir() if isinstance(variant, CliVariant) else None
        pipeline = self.editor.pipeline

        scratch: list[str | None] = [None]  # the job's own scratch directory, once given
        own_scratch = self.editor.own_scratch

        def add_and_run():
            if own_scratch:
                scratch[0] = self._own_scratch(js, pipeline)
            res = pl.add_job(js, alias or None)
            if predicted is not None and res.job != predicted:  # outputs must go to its own directory
                variant.make_jobstar(values, False, res.job, pipeline).write(self.project_dir / res.job / "job.star")
            rec = None if schedule_only else pl.run([res.job])
            return res, rec

        edit = self._edits

        def done(result):
            res, rec = result
            self.select.emit(res.job)
            verb = "scheduled" if rec is None else "started"
            text = f"{res.job.rstrip('/')} {verb}." + (" " + " ".join(res.warnings) if res.warnings else "")
            if scratch[0]:
                text += f" Scratch directory: {scratch[0]}"
            if rec is None:  # scheduled: kept open, to schedule the next job from it
                self.editor.status.setText(text)
            else:
                self._started(edit, text)

        self._run("Schedule job" if schedule_only else "Run job", add_and_run, done)

    # ------------------------------------------------------------------ selection tools
    def request_tool(self, kind: str, arg: str) -> None:
        """From a viewer: "classes" on an optimiser/model file, "tilts" on a tilt-series set (each makes
        a new job), or "change" for the Select / Exclude tilt images job ``arg``."""
        if self.readonly or self.project_dir is None:
            self.message.emit("read-only: no new jobs")
            return
        if kind == "change":
            try:
                js = JobStar.read(self.project_dir / arg / "job.star")
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self.window, "Selection", f"Cannot read {arg}job.star: {e}")
                return
            k = interactive_kind(js.type_label, js.options)
            if k is None:
                QMessageBox.information(self.window, "Selection", f"{self._label(arg)} is not an interactive selection.")
                return
            self.open_tool(k, js, "", MODE_CONTINUE, arg)
            return
        label, key = ("relion.select", "fn_model") if kind == "classes" else ("relion.excludetilts", "in_tiltseries")
        variant = self.defs.variant(label, kind == "tilts")
        last = read_last_used(self.project_dir, variant.label, variant.is_tomo)
        try:
            js = last if last is not None and interactive_kind(last.type_label, {**last.options, key: arg}) else \
                variant.new_jobstar(self.env)
        except RelionEnvError as e:
            self._env_error("Selection", e)
            return
        js.options[key] = arg
        if kind == "classes":
            for k in ("do_class_ranker", "do_filaments", "do_remove_duplicates", "do_select_values", "do_discard",
                      "do_split"):
                if k in js.options:
                    js.options[k] = "No"
        self.open_tool(kind, js, "", None, None)

    def open_tool(self, kind: str, js: JobStar, alias: str, mode, job: str | None) -> None:
        from cryo_savanna.viewers.selectors import ClassSelector, TiltExcluder
        o = js.options
        extra = o.get("other_args", "").strip()
        if extra:  # RELION passes them to its own tool; this window cannot follow them
            program = "relion_display" if kind == "classes" else "relion_python_tomo_exclude_tilt_images"
            ask = (f"This job's additional arguments ({extra}) are for {program}, which RELION runs for it. "
                   f"cryo-savanna's selection window does not use them, so what it saves can differ from what "
                   f"{program} would save (for example with --max_nr_parts_per_class).\n\nOpen cryo-savanna's "
                   f"window anyway, without them? To have them used, choose RELION's own tool under "
                   f"\"Selection window\" on the job editor's Running tab.")
            if QMessageBox.question(self.window, "Selection", ask) != QMessageBox.Yes:
                return
        what = (f"Change {self._label(job)}" if mode == MODE_CONTINUE else
                f"Overwrite {self._label(job)}" if mode == MODE_OVERWRITE else "New job")
        try:
            if kind == "classes":
                src = o["fn_model"]
                # the model and one STAR file per tilt series are read in a thread (ui/busy.py)
                inp = run_busy(lambda: read_class_input(self.project_dir, src), "Reading the classes…", self.window)
                pre = read_backup_selection(self.project_dir, job, len(inp.classes)) if job else None
                w = ClassSelector(self.project_dir, src, selected_numbers(inp, pre),
                                  recenter=o.get("do_recenter") == "Yes",
                                  regroup=int(float(o.get("nr_groups", "0") or 0)) if o.get("do_regroup") == "Yes" else 0,
                                  title=f"Select classes — {src} — {what}", parent=self.window)
            else:
                src = o["in_tiltseries"]
                kept = run_busy(lambda: read_tilt_exclusion(read_tilt_input(self.project_dir, src), job),
                                "Reading the tilt series…", self.window) if job else None
                w = TiltExcluder(self.project_dir, src, kept, title=f"Exclude tilt images — {src} — {what}",
                                 parent=self.window)
        except (OSError, ValueError, KeyError, SelectionError) as e:
            QMessageBox.warning(self.window, "Selection", str(e))
            return
        w.saved.connect(lambda choice, win=w: self._save_tool(kind, js, alias, mode, job, choice, win))
        self.tools.append(w)
        w.destroyed.connect(lambda *_, win=w: self.tools.remove(win) if win in self.tools else None)
        w.show()
        w.raise_()

    def _save_tool(self, kind: str, js: JobStar, alias: str, mode, job: str | None, choice, win) -> None:
        project, pl = self.project_dir, self.pipeliner
        o = dict(js.options)
        if kind == "classes":
            selected, recenter, regroup = choice
            js.options["do_recenter"] = "Yes" if recenter else "No"
            js.options["do_regroup"] = "Yes" if regroup > 0 else "No"
            if regroup > 0:
                js.options["nr_groups"] = str(regroup)
            src = o["fn_model"]

            def write(j):
                return write_class_selection(read_class_input(project, src), j, selected, recenter, regroup)
            command = lambda j: relion_command("classes", o, j, recenter, regroup)  # noqa: E731
        else:
            src = o["in_tiltseries"]

            def write(j):
                return write_tilt_exclusion(read_tilt_input(project, src), j, choice)
            command = lambda j: relion_command("tilts", o, j)  # noqa: E731

        def work():
            if mode == MODE_CONTINUE:
                # the job's settings follow the choice (recenter, regroup), so that RELION's GUI
                # continues it with them too
                path = project / job / "job.star"
                if JobStar.read(path).options != js.options:
                    js.write(path)
                pl.finish_done_here(job, write, command)
                return job
            if mode == MODE_OVERWRITE:
                return pl.overwrite_done_here(job, js, write, command, alias or None).job
            return pl.add_done_here(js, write, command, alias or None).job

        warning = self._trash_warning([job]) if mode == MODE_OVERWRITE else ""
        if warning and QMessageBox.question(win, "Overwrite", f"Overwrite {self._label(job)}?{warning}") \
                != QMessageBox.Yes:
            return

        def done(new_job):
            win.close()
            self.select.emit(new_job)
            self.editor.status.setText(f"{new_job.rstrip('/')}: selection saved.")

        win.set_busy(True, "Saving…")
        self._run("Save selection", work, done, on_error=lambda msg: win.set_busy(False, f"⚠ {msg}"))

    # ------------------------------------------------------------------ running
    def run_scheduled(self) -> None:
        jobs = [(n, self._label(n)) for n, i in self.snapshot.jobs.items() if i.status == "Scheduled"]
        dlg = RunScheduledDialog(self.window, jobs, self.current if self.current in dict(jobs) else None)
        if dlg.exec() != QDialog.Accepted or not dlg.selected():
            return
        selected, args = dlg.selected(), dlg.run_args()
        self._run("Run scheduled jobs", lambda: self.pipeliner.run(selected, **args))

    def stop_runs(self) -> None:
        """Stop a repeating run after the job it is running (RELION's pipeliner checks its flag file)."""
        runs = self.pipeliner.repeating_runs()
        if not runs:
            self._look_at_project()
            self.update_state()
            self.message.emit("No repeating run is going.")
            return
        labels = [f"{time.strftime('%d %b %H:%M', time.localtime(r.started))} · {r.repeat}× · "
                  + ", ".join(self._label(j) for j in r.jobs) for r in runs]
        label, ok = QInputDialog.getItem(self.window, "Stop repeating runs",
                                         "Stop this run after its current job:", labels, 0, False)
        if not ok:
            return
        rec = runs[labels.index(label)]
        self._run("Stop repeating run", lambda: self.pipeliner.stop_schedule(rec))

    def abort_job(self) -> None:
        job = self.current
        q = self.queue.live.get(job) if self.queue is not None else None
        if q is not None and self.slurm is not None:
            choice = self._choose_abort(job, q)
            if choice == "cancel":
                self._run(f"Cancel {self._label(job)} in SLURM", lambda: self._scancel(job, q.id))
            elif choice == "stop":
                self._run(f"Abort {self._label(job)}", lambda: self.pipeliner.abort(job))
            return
        if QMessageBox.question(self.window, "Abort", f"Ask {self._label(job)} to stop?") != QMessageBox.Yes:
            return
        self._run(f"Abort {self._label(job)}", lambda: self.pipeliner.abort(job))

    def _choose_abort(self, job: str, q) -> str:
        """"cancel" (scancel), "stop" (RELION's abort file) or "" for a job in the SLURM queue."""
        label = self._label(job)
        if q.pending:
            ask = (f"{label} is waiting in the queue ({q.describe()}).\n\nCancel it? This runs scancel {q.id}, "
                   "and the job is marked as aborted.")
            return "cancel" if QMessageBox.question(self.window, "Abort", ask) == QMessageBox.Yes else ""
        box = QMessageBox(QMessageBox.Question, "Abort",
                          f"{label} is running in SLURM ({q.describe()}).\n\n"
                          "Ask it to stop: RELION's programs stop at their next check and save what they can.\n"
                          f"Cancel now: scancel {q.id} ends it at once; the job is marked as aborted.",
                          parent=self.window)
        stop = box.addButton("Ask it to stop", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel now", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(stop)
        box.exec()
        return {stop: "stop", cancel: "cancel"}.get(box.clickedButton(), "")

    def _scancel(self, job: str, slurm_id: str) -> None:
        """Cancel a job's SLURM job, and tell RELION it was aborted (nothing else will: a cancelled
        job writes no exit file, and the pipeliner waiting for it would wait for ever)."""
        self.slurm.client.cancel([slurm_id])
        d = self.project_dir / job
        if not any((d / f).exists() for f in (RELION_JOB_EXIT_SUCCESS, RELION_JOB_EXIT_FAILURE,
                                                RELION_JOB_EXIT_ABORTED)):
            (d / RELION_JOB_EXIT_ABORTED).touch()

    def mark_lost(self, lost: dict) -> None:
        """Jobs that SLURM ended without RELION's exit file (time limit, out of memory, node failure, a
        failed or cancelled batch job): write the exit file their program would have written, FAILURE
        (ABORTED when cancelled), so that this window, RELION's GUI and a relion_pipeliner waiting for the
        job see it as finished. Read-only: nothing is written (the job stays marked in the Queue column)."""
        if self.readonly or self.project_dir is None:
            return
        for job, acct in lost.items():
            if (job, acct.id) in self._marked_lost:
                continue
            self._marked_lost.add((job, acct.id))
            what = "aborted" if acct.state.startswith("CANCELLED") else "failed"
            self._run(f"Mark {self._label(job)} as {what} (SLURM: {acct.state.lower().replace('_', ' ')})",
                      lambda job=job, acct=acct: self._write_lost(job, acct))

    def _write_lost(self, job: str, acct) -> None:
        d = self.project_dir / job
        exits = (RELION_JOB_EXIT_SUCCESS, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_ABORTED)
        if not d.is_dir() or any((d / f).exists() for f in exits):  # meanwhile written, or the job is gone
            return
        marker = RELION_JOB_EXIT_ABORTED if acct.state.startswith("CANCELLED") else RELION_JOB_EXIT_FAILURE
        details = style.lost_details(acct)
        with open(d / "run.err", "a") as fh:  # where RELION's GUI and this window show why a job failed
            fh.write(f"\n ++++ cryo-savanna: SLURM job {acct.id} ended {acct.state}{f' ({details})' if details else ''}, "
                     f"without RELION's exit file: {marker} written, {time.strftime('%a %b %d %H:%M:%S %Y')}\n")
        (d / marker).touch()

    def clean(self, harsh: bool) -> None:
        job = self.current
        kind = "harsh" if harsh else "gentle"
        if QMessageBox.question(self.window, "Clean", f"Do a {kind} clean of {self._label(job)}?\n"
                                "Deleted files cannot be recovered.") != QMessageBox.Yes:
            return
        self._run(f"Clean {self._label(job)}", lambda: self.pipeliner.clean(job, harsh))

    # ------------------------------------------------------------------ pipeline operations
    def set_alias(self) -> None:
        job = self.current
        info = self.snapshot.jobs[job]
        current = (info.process.alias.rstrip("/").split("/", 1)[-1] if info.process.has_alias
                   else job.rstrip("/").split("/")[-1])
        alias, ok = QInputDialog.getText(self.window, "Set alias", f"Alias for {job} (empty removes it):",
                                         text=current)
        if not ok or alias == current:
            return
        self._run(f"Set alias of {job}", lambda: self.ops.set_alias(job, alias))

    def mark(self, failed: bool) -> None:
        job = self.current
        what = "failed" if failed else "finished"
        ask = f"Mark {self._label(job)} as {what}?"
        q = self.queue.live.get(job) if self.queue is not None else None
        if q is not None:  # marking it does not stop it
            ask += (f"\n\nIt is still in the SLURM queue ({q.describe()}) and keeps going: use Abort to cancel "
                    "it first.")
        if QMessageBox.question(self.window, f"Mark as {what}", ask) != QMessageBox.Yes:
            return
        self._run(f"Mark {self._label(job)} as {what}", lambda: self.ops.mark_finished(job, failed))

    def delete(self, recursive: bool) -> None:
        job = self.current
        procs, _ = self.ops.delete_plan(job, recursive, self.snapshot.pipeline)
        text = "Move these jobs to Trash/?\n\n" + "\n".join(f"  {self._label(p)}" for p in procs)
        running = [p for p in procs if self.snapshot.jobs[p].status == "Running"]
        if running:
            text += "\n\nStill running: " + ", ".join(self._label(p) for p in running)
        text += self._trash_warning(procs)
        if QMessageBox.question(self.window, "Delete", text) != QMessageBox.Yes:
            return
        self._run(f"Delete {self._label(job)}", lambda: self.ops.delete(job, recursive))

    def import_jobs(self) -> None:
        """Project → Add jobs run outside the pipeline: job folders the pipeline lacks, reviewed, then added."""
        from cryo_savanna.project.rebuild import write_pipeline
        from cryo_savanna.ui.rebuild_dialog import run_rebuild

        project = self.project_dir

        def write(result, jobs, statuses):
            def done(added):
                if added:
                    self.select.emit(added[-1])
            self._run(f"Add {len(jobs)} job(s)", lambda: write_pipeline(project, result, jobs, statuses), done)

        run_rebuild(project, self.defs, new=False, pipeline=self.snapshot.pipeline if self.snapshot else None,
                    type_names=self.type_names, parent=self.window, write=write)

    def undelete(self) -> None:
        entries = self.ops.trash()
        labels = [str(e.parent.relative_to(self.project_dir / "Trash")) for e in entries]
        label, ok = QInputDialog.getItem(self.window, "Undelete", "Put back from Trash/:", labels, 0, False)
        if not ok:
            return
        fn = entries[labels.index(label)]

        def done(restored):
            if restored:
                self.select.emit(restored[0])

        self._run(f"Undelete {label}", lambda: self.ops.undelete(fn), done)


def _newest_inputs(pipeline: Pipeline, kind: str) -> dict[str, str]:
    """The newest tomogram set (template matching) or pytom match summary (extraction) of the pipeline."""
    from cryo_savanna.pytom_runner import SUMMARY
    for name in reversed(list(pipeline.nodes)):
        n = pipeline.nodes[name]
        if not n.producer:
            continue
        if kind == PYTOM_MATCH and n.kind == "TomogramGroupMetadata" and name.endswith("tomograms.star"):
            return {"in_tomograms": name}
        if kind == PYTOM_EXTRACT and name.endswith("/" + SUMMARY):
            return {"in_match": name}
    return {}
