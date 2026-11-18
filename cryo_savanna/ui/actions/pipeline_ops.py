# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Changes to the pipeline itself: alias, mark finished or failed, delete (to Trash/) and undelete, and adding
job folders the pipeline does not have (jobs run outside it)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QInputDialog, QMessageBox

from cryo_savanna.relion_consts import STATUS_RUNNING

if TYPE_CHECKING:  # checked as the actions they are part of; at run time a plain class (no Qt diamond)
    from cryo_savanna.ui.actions.base import ActionsBase as _Base
else:
    _Base = object


class PipelineActions(_Base):
    """Alias, mark finished / failed, delete, undelete, add jobs run outside the pipeline."""

    def set_alias(self) -> None:
        job, snap = self.current, self.snapshot
        if job is None or snap is None or job not in snap.jobs:
            return
        ops = self._ready()[2]
        info = snap.jobs[job]
        current = (info.process.alias.rstrip("/").split("/", 1)[-1] if info.process.has_alias
                   else job.rstrip("/").split("/")[-1])
        alias, ok = QInputDialog.getText(self.window, "Set alias", f"Alias for {job} (empty removes it):",
                                         text=current)
        if not ok or alias == current:
            return
        self._run(f"Set alias of {job}", lambda: ops.set_alias(job, alias))

    def mark(self, failed: bool) -> None:
        job = self.current
        if job is None:
            return
        ops = self._ready()[2]
        what = "failed" if failed else "finished"
        ask = f"Mark {self._label(job)} as {what}?"
        q = self.queue.live.get(job) if self.queue is not None else None
        if q is not None:  # marking it does not stop it
            ask += (f"\n\nIt is still in the SLURM queue ({q.describe()}) and keeps going: use Abort to cancel "
                    "it first.")
        if QMessageBox.question(self.window, f"Mark as {what}", ask) != QMessageBox.Yes:
            return
        self._run(f"Mark {self._label(job)} as {what}", lambda: ops.mark_finished(job, failed))

    def delete(self, recursive: bool) -> None:
        job, snap = self.current, self.snapshot
        if job is None or snap is None:
            return
        ops = self._ready()[2]
        procs, _ = ops.delete_plan(job, recursive, snap.pipeline)
        text = "Move these jobs to Trash/?\n\n" + "\n".join(f"  {self._label(p)}" for p in procs)
        running = [p for p in procs if snap.jobs[p].status == "Running"]
        if running:
            text += "\n\nStill running: " + ", ".join(self._label(p) for p in running)
        held = {p: h for p, h in self._held_now().items() if p in procs}
        if held:
            text += ("\n\nWARNING: in a run that is still going: " + "; ".join(
                f"{self._label(p)} ({h.describe()})" for p, h in held.items()) + ". RELION's pipeliner looks its "
                "jobs up by name when the run ends and crashes if one is gone, which can leave default_pipeline.star "
                "empty and locked. Stop that run first (Job → Stop runs…, or abort its running job).")
        moved = moved_running(snap, procs)
        if moved:
            text += ("\n\nWARNING: these running jobs come after it in the pipeline: "
                     + ", ".join(self._label(p) for p in moved) + ". RELION's pipeliner keeps the jobs it waits for by "
                     "their place in the pipeline, and deleting moves them up, so it may mark the wrong job as "
                     "finished (or never mark the one that ended). Better delete once they have finished.")
        text += self._trash_warning(procs)
        default = QMessageBox.No if held or moved else QMessageBox.Yes
        if QMessageBox.question(self.window, "Delete", text, QMessageBox.Yes | QMessageBox.No, default) != QMessageBox.Yes:
            return
        self._run(f"Delete {self._label(job)}", lambda: ops.delete(job, recursive))

    def import_jobs(self) -> None:
        """Project → Add jobs run outside the pipeline: job folders the pipeline lacks, reviewed, then added."""
        from cryo_savanna.project.rebuild import write_pipeline
        from cryo_savanna.ui.rebuild_dialog import run_rebuild

        project = self._ready()[0]

        def write(result, jobs, statuses):
            def done(added):
                if added:
                    self.select.emit(added[-1])
            self._run(f"Add {len(jobs)} job(s)", lambda: write_pipeline(project, result, jobs, statuses), done)

        run_rebuild(project, self.defs, new=False, pipeline=self.snapshot.pipeline if self.snapshot else None,
                    type_names=self.type_names, parent=self.window, write=write)

    def undelete(self) -> None:
        project, _, ops = self._ready()
        entries = ops.trash()
        if not entries:  # emptied since the last look at the project
            self._look_at_project()
            self.message.emit("Nothing to undelete: Trash/ has no deleted jobs.")
            return
        labels = [str(e.parent.relative_to(project / "Trash")) for e in entries]
        label, ok = QInputDialog.getItem(self.window, "Undelete", "Put back from Trash/:", labels, 0, False)
        if not ok:
            return
        fn = entries[labels.index(label)]

        def done(restored):
            if restored:
                self.select.emit(restored[0])

        self._run(f"Undelete {label}", lambda: ops.undelete(fn), done)


def moved_running(snap, procs: list[str]) -> list[str]:
    """Jobs the pipeline has as Running that deleting ``procs`` moves up in it (they come after the first one)."""
    names = list(snap.pipeline.processes)
    first = min((names.index(p) for p in procs if p in snap.pipeline.processes), default=len(names))
    gone = set(procs)
    return [n for n in names[first + 1:]
            if n not in gone and snap.pipeline.processes[n].status == STATUS_RUNNING]
