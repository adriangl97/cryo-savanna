# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Running jobs and what follows them: run scheduled jobs, stop repeating runs, abort (with SLURM: scancel; on
this machine: force abort, its processes killed), mark the jobs SLURM ended without RELION's exit file, clean,
empty Trash/, and the disk usage window."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from cryo_savanna.project import autocontinue, disk_usage, processes
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import rerun_refused
from cryo_savanna.project.staleness import downstream
from cryo_savanna.relion_consts import (
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
    STATUS_RUNNING,
)
from cryo_savanna.ui import style

if TYPE_CHECKING:  # checked as the actions they are part of; at run time a plain class (no Qt diamond)
    from cryo_savanna.ui.actions.base import ActionsBase as _Base
else:
    _Base = object


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



CONTINUE_WAIT_S = 600.0  # how long a job that hit the time limit waits for the run that sent it to the queue to let go
CONTINUE_RETRY_S = 15.0


class RerunDialog(QDialog):
    """Pick out-of-date jobs to run again: (job, label, why it is out of date, why it cannot be, checked)."""

    def __init__(self, parent, rows: list[tuple[str, str, str, str, bool]]):
        super().__init__(parent)
        self.setWindowTitle("Run out-of-date jobs again")
        self.list = QListWidget()
        for name, label, why, refused, checked in rows:
            it = QListWidgetItem(f"{label} — {refused}" if refused else label)
            it.setData(Qt.UserRole, name)
            it.setToolTip(f"Out of date: {why}")
            if refused:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            else:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.list.addItem(it)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run again")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        text = QLabel("These finished jobs ran before an input of theirs changed (or depend on one that did). Each "
                      "job ticked is overwritten (its files go to Trash/, as Overwrite does) and runs again with its "
                      "own settings, in one run: each after the jobs it reads from, which it waits for.")
        text.setWordWrap(True)
        lay.addWidget(text)
        lay.addWidget(self.list)
        lay.addWidget(buttons)

    def selected(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).flags() & Qt.ItemIsUserCheckable and self.list.item(i).checkState() == Qt.Checked]


class RunActions(_Base):
    """Run scheduled jobs, stop repeating runs, abort, mark lost SLURM jobs, clean, empty Trash/."""

    def rerun_out_of_date(self, dialog_hook=None) -> None:
        """Run out-of-date jobs again (project/staleness.py): those after the selected one, if it is out of date,
        else all of them, chosen in a dialog. ``dialog_hook(dlg)`` (tests) answers it instead of the user."""
        snap = self.snapshot
        project, pl = self._ready()[0], self._ready()[1]
        if snap is None:
            return
        stale = {n: st for n, st in self.stale.items() if n in snap.jobs and snap.jobs[n].status == "Succeeded"}
        if not stale:
            QMessageBox.information(self.window, "Run out-of-date jobs again", "No finished job is out of date.")
            return
        held = self._held_now()
        current = self.current
        chosen = set(downstream(snap.pipeline, current, stale)) if current is not None and current in stale \
            else set(stale)
        rows = []
        for name, st in stale.items():
            try:
                refused = rerun_refused(JobStar.read(project / name / "job.star"))
            except (OSError, ValueError, KeyError) as e:
                refused = f"its job.star cannot be read ({e})"
            if name in held:
                refused = "is in a run that is still going"
            rows.append((name, self._label(name), st.text(), refused, name in chosen and not refused))
        dlg = RerunDialog(self.window, rows)
        if (dialog_hook(dlg) if dialog_hook is not None else dlg.exec()) != QDialog.Accepted:
            return
        jobs = dlg.selected()
        if not jobs:
            return
        warning = self._trash_warning(jobs)
        if warning and dialog_hook is None and QMessageBox.question(
                self.window, "Run out-of-date jobs again", warning.strip() + "\n\nRun them again?") != QMessageBox.Yes:
            return

        def done(result):
            ran, _rec = result
            self.select.emit(ran[0])
            self.message.emit("Running again: " + ", ".join(j.rstrip("/") for j in ran))
        what = f"Run {len(jobs)} out-of-date job{'s' if len(jobs) != 1 else ''} again"
        self._run(what, lambda: pl.rerun(jobs), done)

    def run_scheduled(self) -> None:
        snap, pl = self.snapshot, self._ready()[1]
        if snap is None:
            return
        held = self._held_now()  # a job a run still going is to run cannot be run by a second one
        jobs = [(n, self._label(n)) for n, i in snap.jobs.items() if i.status == "Scheduled" and n not in held]
        if not jobs:
            QMessageBox.information(self.window, "Run scheduled jobs", "Every scheduled job is already in a run "
                                    "that is still going (Job → Stop runs… stops one that waits for its inputs).")
            return
        dlg = RunScheduledDialog(self.window, jobs, self.current if self.current in dict(jobs) else None)
        if dlg.exec() != QDialog.Accepted or not dlg.selected():
            return
        selected, args = dlg.selected(), dlg.run_args()
        self._run("Run scheduled jobs", lambda: pl.run(selected, **args))

    def stop_runs(self) -> None:
        """Stop a run: a repeating one after the job it runs (RELION's pipeliner checks its flag file), one that
        waits for the input files of its next job now (its relion_pipeliner is ended; the job stays Scheduled)."""
        pl, snap = self._ready()[1], self.snapshot
        runs = pl.stoppable_runs(snap.pipeline if snap is not None else None)
        if not runs:
            self._look_at_project()
            self.update_state()
            self.message.emit("No run that can be stopped is going.")
            return

        stuck = dict((r.id, j) for r, j in pl.stuck_runs(snap.pipeline if snap is not None else None)) \
            if any(k == "stuck" for _, k in runs) else {}

        def label(rec, kind) -> str:
            if kind == "repeating":
                what = f"{rec.repeat}× · stops after its current job"
            elif kind == "stuck":
                what = (f"{self._label(stuck[rec.id])} is not running any more, but its relion_pipeliner "
                        f"({rec.pid}) still waits for it · ends now, the job marked aborted")
            else:
                what = "waits for its inputs or its GPUs · stops now"
            return (f"{time.strftime('%d %b %H:%M', time.localtime(rec.started))} · {what} · "
                    + ", ".join(self._label(j) for j in rec.jobs))
        runs = [(r, k) for r, k in runs if k != "stuck" or r.id in stuck]  # (one that ended meanwhile: not shown)
        if not runs:
            self.message.emit("No run that can be stopped is going.")
            return
        labels = [label(r, k) for r, k in runs]
        choice, ok = QInputDialog.getItem(self.window, "Stop runs", "Stop this run:", labels, 0, False)
        if not ok:
            return
        rec, kind = runs[labels.index(choice)]
        if kind == "repeating":
            self._run("Stop repeating run", lambda: pl.stop_schedule(rec))
        elif kind == "stuck":
            job = stuck[rec.id]
            self._run(f"End the run of {self._label(job)}", lambda: pl.end_stuck(rec, job))
        else:
            self._run("Stop run", lambda: pl.end_run(rec))

    def abort_job(self) -> None:
        job = self.current
        if job is None:
            return
        pl = self._ready()[1]
        q = self.queue.live.get(job) if self.queue is not None else None
        if q is not None and self.slurm is not None:
            choice = self._choose_abort(job, q)
            if choice == "cancel":
                self._run(f"Cancel {self._label(job)} in SLURM", lambda: self._scancel(job, q.id))
            elif choice == "stop":
                self._run(f"Abort {self._label(job)}", lambda: pl.abort(job))
            return
        if self._queued(job):  # in a queue cryo-savanna does not watch: its processes are not here
            if QMessageBox.question(self.window, "Abort", f"Ask {self._label(job)} to stop?") == QMessageBox.Yes:
                self._run(f"Abort {self._label(job)}", lambda: pl.abort(job))
            return
        choice = self._choose_local_abort(job)
        if choice == "stop":
            self._run(f"Abort {self._label(job)}", lambda: pl.abort(job))
        elif choice == "force":
            self.force_abort(job)

    def _queued(self, job: str) -> bool:
        """Whether the job was submitted to a queue (its job.star's "Submit to queue?")."""
        try:
            return JobStar.read(self._ready()[0] / job / "job.star").options.get("do_queue", "No") == "Yes"
        except (OSError, ValueError, KeyError):
            return False

    def _choose_local_abort(self, job: str) -> str:
        """"stop" (RELION's abort file), "force" (kill its processes here) or "" for a job run on a machine."""
        box = QMessageBox(QMessageBox.Question, "Abort",
                          f"{self._label(job)} is running (not in a queue).\n\n"
                          "Ask it to stop: RELION's programs stop at their next check and save what they can.\n"
                          "Force abort: its processes on this machine are killed now, and the job is marked as "
                          "aborted. What it was writing may be left incomplete.",
                          parent=self.window)
        stop = box.addButton("Ask it to stop", QMessageBox.AcceptRole)
        force = box.addButton("Force abort…", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(stop)
        box.exec()
        return {stop: "stop", force: "force"}.get(box.clickedButton(), "")

    def force_abort(self, job: str) -> None:
        """Kill the job's processes on this host, after showing which (project/processes.py)."""
        pl, label = self._ready()[1], self._label(job)
        try:
            procs, host = pl.force_abort_targets(job)
        except processes.ForceAbortError as e:
            QMessageBox.warning(self.window, "Force abort", str(e))
            return
        if not procs:
            rec = self._waiting_run(job)
            if rec is not None:  # its processes are gone, its relion_pipeliner still waits for its exit file
                if self._confirm_end_run(label, rec, "Force abort", "End the run"):
                    self._run(f"End the run of {label}", lambda: pl.end_stuck(rec, job))
                return
            where = (f"It was started on {host}: force abort it from a cryo-savanna there." if host else
                     "It may run on another machine, or it has just ended. If it is not running anywhere, use "
                     "Job → Mark as failed.")
            QMessageBox.information(self.window, "Force abort", f"No process of {label} was found on this machine.\n\n"
                                    + where)
            return
        if not self._confirm_kill(label, procs):
            return
        self._run(f"Force abort {label}", lambda: pl.force_abort(job, procs))

    def _waiting_run(self, job: str):
        """The run started here that is at ``job`` and still going (its relion_pipeliner waits for the job's
        exit file), or None."""
        h = self._held_now().get(job)
        return h.run if h is not None and h.current and not h.stoppable else None

    def _confirm_end_run(self, label: str, rec, title: str, button: str, then: str = "") -> bool:
        box = QMessageBox(QMessageBox.Warning, title,
                          f"No process of {label} is left on this machine, but the relion_pipeliner that ran it "
                          f"({rec.pid}) still waits for it to end (it ended without writing RELION's exit file: a "
                          "crash, the out-of-memory killer, or killed by hand).\n\n"
                          f"End that run{then}? The job is marked as aborted, and its relion_pipeliner ends (it is "
                          "stopped if it does not by itself). The jobs after it in that run are not run.",
                          parent=self.window)
        end = box.addButton(button, QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is end

    def _confirm_kill(self, label: str, procs: list, title: str = "Force abort", then: str = "",
                      button: str = "Kill") -> bool:
        shown = [f"{p.pid:>7}  {p.args if len(p.args) <= 110 else p.args[:107] + '...'}" for p in procs[:12]]
        if len(procs) > len(shown):
            shown.append(f"... and {len(procs) - len(shown)} more")
        box = QMessageBox(QMessageBox.Warning, title,
                          f"Kill these {len(procs)} process(es) of {label}{then}?\n\n" + "\n".join(shown) +
                          f"\n\nThey get SIGTERM, then SIGKILL after {processes.TERM_WAIT_S:g} s.",
                          parent=self.window)
        box.setDetailedText("\n".join(f"{p.pid}  (parent {p.ppid})  {p.args}" for p in procs))
        kill = box.addButton(button, QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is kill

    def _kill_to_overwrite(self, job: str) -> Callable[[], None] | None:
        """For an Overwrite of a running job: after asking, what ends it now (its processes on this machine
        killed, or its SLURM job cancelled) and waits until it can be overwritten (Pipeliner.wait_released), to
        run before the overwrite; None when it cannot be ended from here or the user said no."""
        pl, label = self._ready()[1], self._label(job)
        q = self.queue.live.get(job) if self.queue is not None else None
        if q is not None and self.slurm is not None:
            ask = (f"{label} is {'waiting' if q.pending else 'running'} in SLURM ({q.describe()}).\n\n"
                   f"Cancel it now (scancel {q.id}) and overwrite it?")
            if QMessageBox.question(self.window, "Overwrite", ask) != QMessageBox.Yes:
                return None
            slurm_id = q.id

            def cancel() -> None:
                self._scancel(job, slurm_id)
                pl.wait_released(job)
            return cancel
        if self._queued(job):  # in a queue cryo-savanna does not watch: its processes are not here
            QMessageBox.warning(self.window, "Overwrite", f"{label} is running in a queue: abort it first, and "
                                "overwrite it once it has stopped.")
            return None
        try:
            procs, host = pl.force_abort_targets(job)
        except processes.ForceAbortError as e:
            QMessageBox.warning(self.window, "Overwrite", str(e))
            return None
        if not procs:
            rec = self._waiting_run(job)
            if rec is not None:  # its processes are gone, its relion_pipeliner still waits for its exit file
                if not self._confirm_end_run(label, rec, "Overwrite", "End the run and overwrite",
                                             then=" and overwrite the job"):
                    return None

                def end_run() -> None:
                    pl.end_stuck(rec, job)
                    pl.wait_released(job)
                return end_run
            where = (f"It was started on {host}: abort it from a cryo-savanna there." if host else
                     "It may run on another machine, or it has just ended. If it is not running anywhere, use "
                     "Job → Mark as failed, then overwrite it.")
            QMessageBox.information(self.window, "Overwrite", f"{label} is running, and no process of it was found "
                                    "on this machine.\n\n" + where)
            return None
        if not self._confirm_kill(label, procs, title="Overwrite", then=" and overwrite it",
                                  button="Kill and overwrite"):
            return None

        def kill() -> None:
            pl.force_abort(job, procs)
            pl.wait_released(job)
        return kill

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
        assert self.slurm is not None  # abort_job offers it only for a job in the SLURM queue
        self.slurm.client.cancel([slurm_id])
        d = self._ready()[0] / job
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
                      lambda job=job, acct=acct: self._write_lost(job, acct),
                      lambda timed_out, job=job, acct=acct: self._after_time_limit(job, acct) if timed_out else None)

    def _write_lost(self, job: str, acct) -> bool:
        """Write the exit file of a job SLURM ended; True when it was its time limit."""
        d = self._ready()[0] / job
        exits = (RELION_JOB_EXIT_SUCCESS, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_ABORTED)
        if not d.is_dir() or any((d / f).exists() for f in exits):  # meanwhile written, or the job is gone
            return False
        if self.slurm is not None and self.slurm.ended_before_submission(job, acct):
            return False  # sent to the queue again since that look (Continue): this run is not the one that ended
        marker = RELION_JOB_EXIT_ABORTED if acct.state.startswith("CANCELLED") else RELION_JOB_EXIT_FAILURE
        details = style.lost_details(acct)
        with open(d / "run.err", "a") as fh:  # where RELION's GUI and this window show why a job failed
            fh.write(f"\n ++++ cryo-savanna: SLURM job {acct.id} ended {acct.state}{f' ({details})' if details else ''}, "
                     f"without RELION's exit file: {marker} written, {time.ctime()}\n")
        (d / marker).touch()
        return acct.state.startswith("TIMEOUT")

    # ------------------------------------------------------------------ continuing after the time limit
    def _after_time_limit(self, job: str, acct) -> None:
        """A relion_refine job SLURM ended at its time limit: continued from its latest iteration (Preferences →
        "Continue jobs that hit the SLURM time limit"), else the status bar says how."""
        info = self.snapshot.jobs.get(job) if self.snapshot is not None else None
        if info is None or not autocontinue.continuable(info.process.type_label):
            return
        if not self.auto_continue:
            self.message.emit(f"{self._label(job)} hit the SLURM time limit: Job → Continue… continues it from its "
                              "latest iteration")
            return
        self._continuing[job] = (acct.id, time.monotonic() + CONTINUE_WAIT_S)
        self._try_continue(job)

    def _try_continue(self, job: str) -> None:
        """Continue the job once the run that waited for it has seen it end (then it is free to run again)."""
        hit = self._continuing.get(job)
        if hit is None:
            return
        acct_id, deadline = hit
        label = self._label(job)

        def done(result: str) -> None:
            if result == "wait" and time.monotonic() < deadline:
                QTimer.singleShot(int(CONTINUE_RETRY_S * 1000), lambda: self._try_continue(job))
                return
            self._continuing.pop(job, None)
            if result == "wait":
                self.message.emit(f"{label} was not continued: the run that sent it to the queue still waits for it")
            elif result == "limit":
                self.message.emit(f"{label} hit the time limit again: continued {autocontinue.MAX_TIMES} times "
                                  "already, so not again (give it a longer time limit)")
            elif result:
                self.message.emit(f"{label} {result}")

        self._run(f"Continue {label} after the SLURM time limit", lambda: self._auto_continue(job, acct_id), done,
                  on_error=lambda _msg: self._continuing.pop(job, None))

    def _auto_continue(self, job: str, slurm_id: str) -> str:
        """(In the worker.) "wait" while a run still holds the job, "limit" when it was continued often enough,
        "" when it is gone, else what was done."""
        project, pl, ops = self._ready()
        p = Pipeline.read(project)
        pr = p.processes.get(job)
        if pr is None:
            return ""
        if job in pl.held_jobs(p):
            return "wait"
        if autocontinue.times_continued(project, job) >= autocontinue.MAX_TIMES:
            return "limit"
        js = autocontinue.continue_jobstar(project, job)
        if pr.status == STATUS_RUNNING:  # no run waits for it: marked failed first, as RELION's GUI would
            ops.mark_finished(job, failed=True)
        pl.continue_job(job, js)
        n = autocontinue.note_continued(project, job, slurm_id)
        with open(project / job / "run.err", "a") as fh:
            fh.write(f"\n ++++ cryo-savanna: continued from {js.options['fn_cont']} after SLURM job {slurm_id} hit "
                     f"its time limit ({n} of at most {autocontinue.MAX_TIMES} times), {time.ctime()}\n")
        return f"continued from {js.options['fn_cont']} after the SLURM time limit ({n} of {autocontinue.MAX_TIMES})"

    def clean(self, harsh: bool) -> None:
        job = self.current
        if job is None:
            return
        pl = self._ready()[1]
        kind = "harsh" if harsh else "gentle"
        if QMessageBox.question(self.window, "Clean", f"Do a {kind} clean of {self._label(job)}?\n\n"
                                "RELION moves the files it cleans to Trash/: they take space until Trash/ is "
                                "emptied (Project → Disk usage…).") != QMessageBox.Yes:
            return
        self._run(f"Clean {self._label(job)}", lambda: pl.clean(job, harsh))

    def clean_jobs(self, jobs: list[str], harsh: bool, done: Callable | None = None) -> None:
        """Clean several jobs, one after another (Disk usage window; it asked already)."""
        kind = "harsh" if harsh else "gentle"
        pl = self._ready()[1]

        def work() -> None:
            for job in jobs:
                pl.clean(job, harsh)

        what = f"Clean ({kind}) {self._label(jobs[0])}" if len(jobs) == 1 else f"Clean ({kind}) {len(jobs)} jobs"
        self._run(what, work, done=lambda _: done() if done else None, on_error=lambda _: done() if done else None)

    def empty_trash(self, entries: list[str], done: Callable | None = None) -> None:
        """Delete entries of Trash/ for good (Disk usage window; it asked already)."""
        project = self._ready()[0]

        def work() -> None:
            errors = disk_usage.empty_trash(project, entries)
            if errors:
                raise disk_usage.TrashError("Not everything could be removed:\n" + "\n".join(errors[:20]))

        self._run("Empty Trash/", work, done=lambda _: done() if done else None,
                  on_error=lambda _: done() if done else None)

    def show_disk_usage(self) -> None:
        from cryo_savanna.ui import disk_usage as window
        self._disk_window = window.show(self, self.window, self._disk_window)
