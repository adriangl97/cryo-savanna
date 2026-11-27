# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What the job actions share: the project, its pipeliner and operations, the background tasks they run in,
the job editor's window or dock, and helpers. Jobs are added and run through relion_pipeliner
(project/pipeliner.py); the few operations it cannot do go through PipelineOps (project/ops.py). Both can wait
for RELION's lock, so they run in a thread pool and report back with signals; the window stays responsive.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QAction, QDockWidget, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from cryo_savanna.env.relion_env import RelionEnv, RelionEnvError
from cryo_savanna.jobdefs import external_jobs
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import warm_defaults as warm_pytom_defaults
from cryo_savanna.project import disk_usage
from cryo_savanna.project.gpu_queue import GpuQueue
from cryo_savanna.project.ops import OpsError, PipelineOps
from cryo_savanna.project.pipeliner import Pipeliner, PipelinerError
from cryo_savanna.project.rebuild import RebuildError
from cryo_savanna.project.slurm import QueueState, SlurmMonitor
from cryo_savanna.project.snapshot import ProjectSnapshot
from cryo_savanna.project.staleness import StalenessChecker
from cryo_savanna.ui import style
from cryo_savanna.ui.job_editor import JobEditor

if TYPE_CHECKING:
    from cryo_savanna.ui.disk_usage import DiskUsageWindow


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
        except (OpsError, PipelinerError, RebuildError, disk_usage.TrashError) as e:
            self.signals.error.emit(str(e))
        except Exception as e:  # anything else: report, never kill the GUI
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        else:
            self.signals.done.emit(result)



# ---------------------------------------------------------------------------- shared state and helpers

class ActionsBase(QObject):
    """The state and helpers of JobActions (ui/job_actions.py) that its groups of actions share."""

    changed = Signal()  # the project was changed: refresh now
    select = Signal(str)  # select this job once it shows up
    message = Signal(str)  # short status message
    star = Signal(str, bool)  # star or unstar this job
    editor_inputs = Signal(object)  # {job: [field labels]} the shown job editor takes inputs from ({}: none, or closed)
    runs = Signal(object)  # {job: Held}: the jobs runs still going are to run (Pipeliner.held_jobs), at each look
    stale_jobs = Signal(object)  # {job: Stale}: the finished jobs that are out of date (project/staleness.py)

    def __init__(self, window: QMainWindow, defs: JobDefs, env: RelionEnv, type_names: dict[str, str],
                 tomo: bool, readonly: bool, flag_source=None):
        super().__init__(window)
        self.window, self.defs, self.env, self.type_names = window, defs, env, type_names
        self.tomo, self.readonly = tomo, readonly
        self.project_dir: Path | None = None
        self.pipeliner: Pipeliner | None = None
        self.ops: PipelineOps | None = None
        self.staleness: StalenessChecker | None = None
        self.stale: dict = {}  # job -> Stale, at the last look
        # Preferences → local GPU and thread queues: runs here wait for their GPUs / threads (set_local_queue)
        self.gpu_queue_on = True
        self.max_threads = 0  # the thread queue's limit; 0: off
        self.auto_continue = False  # Preferences: continue relion_refine jobs that hit the SLURM time limit
        self._continuing: dict[str, tuple[str, float]] = {}  # job -> (SLURM id, until when): to continue (running.py)
        self.snapshot: ProjectSnapshot | None = None
        self.current: str | None = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)  # one change at a time, in the order asked
        self._pending: set[_Signals] = set()
        self._structure: object = None  # snapshot.structure_key() of the last snapshot
        self._nodes: object = None  # the pipeline's nodes (name, type, producer) when the input lists were filled
        self._has_trash = False  # looked at when the project changes, not on every selection
        self._stoppable = 0  # runs Stop runs can stop
        self._look_pool = QThreadPool(self)  # those looks read Trash/ and the run records: off the GUI thread
        self._look_pool.setMaxThreadCount(1)
        self._look_signals = _Signals()
        self._look_signals.done.connect(self._on_look)
        self._look_signals.error.connect(lambda msg: self._on_look(None))
        self._looking = self._look_again = False
        self.slurm: SlurmMonitor | None = None  # of the project
        self._marked_lost: set[tuple[str, str]] = set()  # (job, SLURM id) already marked (mark_lost)
        self.queue: QueueState | None = None  # from the main window's monitor

        self.editor = JobEditor(env, type_names, flag_source=flag_source, relion_version=defs.version)
        self._shown_inputs: dict[str, list[str]] = {}  # as last told (editor_inputs)
        self.editor.inputs_changed.connect(lambda _: self._tell_editor_inputs())
        # the forms of the External jobs made here (CLI tools, pytom, AreTomo3, tomogram masks, particle splits,
        # MemBrain-seg, csparc2star),
        # built once
        self.forms = external_jobs.Forms(defs, env, self.editor.flag_source)
        self.cli, self.pytom = self.forms.cli, self.forms.pytom
        self.aretomo, self.tomomask = self.forms.aretomo, self.forms.tomomask
        self.particle_split = self.forms.split
        self.membrain = self.forms.membrain
        self.csparc = self.forms.csparc
        warm_pytom_defaults()  # "module avail cuda", in the background: not when the first form opens
        self.editor.submit.connect(self._on_submit)
        self.editor.submit_many.connect(self._on_submit_many)
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
        self._disk_window: DiskUsageWindow | None = None
        self.tools: list[QMainWindow] = []  # open selection windows

    def update_state(self) -> None:
        """Enable what makes sense for the selected job, the project and this shell (JobActions)."""
        raise NotImplementedError

    def _kill_to_overwrite(self, job: str) -> Callable[[], None] | None:
        """What ends a running job before it is overwritten, after asking (RunActions)."""
        raise NotImplementedError

    def _ready(self) -> tuple[Path, Pipeliner, PipelineOps]:
        """The open project, its pipeliner and its operations (set together, by set_project; the actions that
        change a project are enabled only once it is)."""
        if self.project_dir is None or self.pipeliner is None or self.ops is None:
            raise RuntimeError("no project open")
        return self.project_dir, self.pipeliner, self.ops

    # ------------------------------------------------------------------ the project
    def set_project(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.pipeliner = Pipeliner(project_dir)
        self.pipeliner.gpu_queue = self._local_queue()
        self.pipeliner.own_scratch = self.editor.own_scratch
        self.ops = PipelineOps(project_dir)
        self.staleness = StalenessChecker(project_dir)
        self.stale = {}
        self.snapshot, self.current, self._structure, self._nodes = None, None, None, None
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
        # new or changed jobs, or new outputs (an External job's, RELION_OUTPUT_NODES.star, are only added when
        # it finishes, a status change): refresh the input lists
        nodes = self._nodes
        if pipeline_changed:
            nodes = tuple((n.name, n.type_label, n.producer) for n in snap.pipeline.nodes.values())
        if key != self._structure or nodes != self._nodes:
            self._structure, self._nodes = key, nodes
            self.editor.update_pipeline(snap.pipeline)
        elif pipeline_changed:
            self.editor.pipeline = snap.pipeline  # same nodes: no need to fill the lists again
        if pipeline_changed:
            self._look_at_project()
        self.update_state()

    def _look_at_project(self) -> None:
        """What Undelete and Stop runs need, and the jobs runs still going hold: once per snapshot (a change),
        not per click.

        Read in the background (a big Trash/ over a slow file system); a change while a look is
        going is looked at again once it is done."""
        if self._looking:
            self._look_again = True
            return
        self._looking, self._look_again = True, False
        project, ops, pl, checker = self.project_dir, self.ops, self.pipeliner, self.staleness
        pipeline = self.snapshot.pipeline if self.snapshot is not None else None  # not changed once made
        self._look_pool.start(_Task(lambda: (project, *self._project_state(ops, pl, pipeline, checker)),
                                    self._look_signals))

    @staticmethod
    def _project_state(ops: PipelineOps | None, pl: Pipeliner | None, pipeline=None,
                       checker: StalenessChecker | None = None) -> tuple[bool, int, dict, dict]:
        """(anything to undelete, how many runs Stop runs can stop, the jobs runs still going hold, the jobs out
        of date)."""
        held = pl.held_jobs(pipeline) if pl is not None else {}
        stoppable = len({r.id for r in pl.repeating_runs()}  # (as Pipeliner.stoppable_runs)
                        | {h.run.id for h in held.values() if h.stoppable or not h.current}
                        | {r.id for r, _ in pl.stuck_runs(pipeline, held)}) if pl is not None else 0
        stale = checker.check(pipeline) if checker is not None and pipeline is not None else {}
        return ops is not None and bool(ops.trash()), stoppable, held, stale

    def look_at_project_now(self) -> None:
        """_look_at_project, done at once (tests)."""
        pipeline = self.snapshot.pipeline if self.snapshot is not None else None
        self._has_trash, self._stoppable, held, self.stale = self._project_state(
            self.ops, self.pipeliner, pipeline, self.staleness)
        self.runs.emit(held)
        self.stale_jobs.emit(self.stale)
        self.update_state()

    def _held_now(self) -> dict:
        """Pipeliner.held_jobs for the latest snapshot, looked at now ({} when it cannot be told)."""
        pl, snap = self.pipeliner, self.snapshot
        if pl is None or snap is None:
            return {}
        try:
            return pl.held_jobs(snap.pipeline)
        except (OSError, ValueError, RuntimeError):
            return {}

    def _on_look(self, result) -> None:
        """``result``: (project, has_trash, stoppable runs, held, stale), or None when the look failed (kept as it
        was)."""
        self._looking = False
        project, has_trash, stoppable, held, stale = result if result is not None else (None, False, 0, {}, {})
        if result is not None and project == self.project_dir:  # not one of the project open before
            self._has_trash, self._stoppable, self.stale = has_trash, stoppable, stale
            self.runs.emit(held)
            self.stale_jobs.emit(stale)
            self.update_state()
        if self._look_again or (result is not None and project != self.project_dir):
            self._look_at_project()

    def _local_queue(self) -> GpuQueue | None:
        if not self.gpu_queue_on and self.max_threads <= 0:
            return None
        return GpuQueue(gpus=self.gpu_queue_on, max_threads=self.max_threads)

    def set_local_queue(self, gpus: bool, max_threads: int = 0) -> None:
        """Preferences → local GPU queue: runs of jobs that use GPUs here wait until those GPUs are free; local thread
        queue (``max_threads`` > 0): runs of jobs that compute here wait until enough of the threads are free."""
        self.gpu_queue_on, self.max_threads = gpus, max(0, max_threads)
        if self.pipeliner is not None:
            self.pipeliner.gpu_queue = self._local_queue()

    def set_gpu_queue(self, on: bool) -> None:
        """The GPU queue alone on or off (the thread queue as it is)."""
        self.set_local_queue(on, self.max_threads)

    def set_own_scratch(self, on: bool) -> None:
        """Preferences → a subdirectory of its own when the scratch directory is in use (project/scratch.py):
        decided when a job starts (Pipeliner.run)."""
        self.editor.own_scratch = on
        if self.pipeliner is not None:
            self.pipeliner.own_scratch = on

    def set_slurm(self, monitor) -> None:
        self.slurm, self.queue = monitor, None

    def set_queue(self, state) -> None:
        self.queue = state

    def set_current(self, job: str | None) -> None:
        self.current = job
        self.update_state()

    # ------------------------------------------------------------------ helpers
    def _run(self, what: str, fn: Callable, done: Callable | None = None, on_error: Callable | None = None) -> None:
        signals = _Signals()
        self._pending.add(signals)

        def finished(result):
            self._pending.discard(signals)
            self.message.emit(f"{what}: done")
            self._warn_oversubscribed()
            if done is not None:
                done(result)
            self.changed.emit()

        def failed(msg: str):
            self._pending.discard(signals)
            self._warn_oversubscribed()
            self.message.emit(f"{what}: failed")
            if on_error is not None:
                on_error(msg)
            self.changed.emit()  # before the message: what did change (part of the jobs) shows while it is read
            QMessageBox.warning(self.window, what, msg)

        signals.done.connect(finished)
        signals.error.connect(failed)
        self.message.emit(f"{what}…")
        self.pool.start(_Task(fn, signals))

    def _warn_oversubscribed(self) -> None:
        """Runs just started that ask for more threads than the thread queue's limit: say they will oversubscribe
        the machine (they wait until no RELION job runs here, then start: gpu_wait.py)."""
        pl = self.pipeliner
        if pl is None or not getattr(pl, "oversubscribed", None):  # a stand-in pipeliner (tests) has none
            return
        recs, pl.oversubscribed = pl.oversubscribed, []
        lines = "\n".join(f"  {', '.join(j.rstrip('/') for j in r.jobs)}: {r.threads} threads (MPI procs × threads)"
                          for r in recs)
        limit = recs[0].thread_limit
        box = QMessageBox(QMessageBox.Warning, "More threads than the limit",
                          f"These runs ask for more threads than the {limit} of the local thread queue "
                          f"(Preferences):\n\n{lines}\n\nEach waits until no other RELION job runs on this machine, "
                          "then starts and oversubscribes it. Stop it with Job → Stop runs… and lower its MPI procs or "
                          "threads if that is not what you want.", QMessageBox.Ok, self.window)
        box.setObjectName("oversubscribed")
        box.open()

    def wait(self, ms: int = 60000) -> bool:
        """Wait for queued operations (closing the window, tests); -1: as long as it takes."""
        return self.pool.waitForDone(ms)

    def _trash_warning(self, jobs: list[str]) -> str:
        """Text for a confirmation that moves ``jobs`` to Trash/, when that replaces earlier copies there."""
        old = self._ready()[2].in_trash(jobs)
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
        self._tell_editor_inputs()

    def _tell_editor_inputs(self) -> None:
        """editor_inputs, when they change: the editor's input jobs while it is shown, none once it is closed."""
        inputs = self.editor.input_jobs() if self.editor_shown() else {}
        if inputs != self._shown_inputs:
            self._shown_inputs = inputs
            self.editor_inputs.emit(inputs)

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

    @staticmethod
    def _env_message(e: Exception) -> str:
        return f"{e}.\n\nCorrect it in the shell (or the RELION module) that starts cryo-savanna, then start it again."

    def _env_error(self, title: str, e: Exception) -> None:
        QMessageBox.warning(self.window, title, self._env_message(e))
