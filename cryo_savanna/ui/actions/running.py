# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Running jobs and what follows them: run scheduled jobs, stop repeating runs, abort (with SLURM: scancel),
mark the jobs SLURM ended without RELION's exit file, clean, empty Trash/, and the disk usage window."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
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

from cryo_savanna.project import disk_usage
from cryo_savanna.relion_consts import RELION_JOB_EXIT_ABORTED, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_SUCCESS
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



class RunActions(_Base):
    """Run scheduled jobs, stop repeating runs, abort, mark lost SLURM jobs, clean, empty Trash/."""

    def run_scheduled(self) -> None:
        snap, pl = self.snapshot, self._ready()[1]
        if snap is None:
            return
        jobs = [(n, self._label(n)) for n, i in snap.jobs.items() if i.status == "Scheduled"]
        dlg = RunScheduledDialog(self.window, jobs, self.current if self.current in dict(jobs) else None)
        if dlg.exec() != QDialog.Accepted or not dlg.selected():
            return
        selected, args = dlg.selected(), dlg.run_args()
        self._run("Run scheduled jobs", lambda: pl.run(selected, **args))

    def stop_runs(self) -> None:
        """Stop a repeating run after the job it is running (RELION's pipeliner checks its flag file)."""
        pl = self._ready()[1]
        runs = pl.repeating_runs()
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
        self._run("Stop repeating run", lambda: pl.stop_schedule(rec))

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
        if QMessageBox.question(self.window, "Abort", f"Ask {self._label(job)} to stop?") != QMessageBox.Yes:
            return
        self._run(f"Abort {self._label(job)}", lambda: pl.abort(job))

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
                      lambda job=job, acct=acct: self._write_lost(job, acct))

    def _write_lost(self, job: str, acct) -> None:
        d = self._ready()[0] / job
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
