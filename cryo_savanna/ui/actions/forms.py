# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""New, clone, continue and overwrite jobs, and what happens when the job editor submits one: RELION's job
types, the CLI tools (jobdefs/cli_tools.py) and cryo-savanna's own External jobs (pytom, AreTomo3, tomogram
masks: jobdefs/external_jobs.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMessageBox

from cryo_savanna.env.relion_env import RelionEnvError
from cryo_savanna.jobdefs import aretomo_jobs, external_jobs, tomomask_jobs
from cryo_savanna.jobdefs.cli_tools import CliVariant
from cryo_savanna.jobdefs.model import JobVariant
from cryo_savanna.jobdefs.pytom_jobs import TITLES as PYTOM_TITLES
from cryo_savanna.jobdefs.pytom_jobs import (EXTRACT as PYTOM_EXTRACT, MATCH as PYTOM_MATCH, PytomVariant,
                                             last_settings, quoted_values)
from cryo_savanna.project.jobstar import JobStar, read_last_used
from cryo_savanna.project.ops import OpsError, PipelineOps
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import PipelinerError
from cryo_savanna.project.scratch import ENV_VAR as SCRATCH_VAR
from cryo_savanna.project.scratch import SCRATCH_KEY
from cryo_savanna.project.scratch import assign as assign_scratch
from cryo_savanna.project.selection import interactive_kind
from cryo_savanna.ui.job_editor import MODE_CONTINUE, MODE_OVERWRITE

if TYPE_CHECKING:  # checked as the actions they are part of; at run time a plain class (no Qt diamond)
    from cryo_savanna.ui.actions.base import ActionsBase as _Base
else:
    _Base = object


class FormActions(_Base):
    """New / clone / continue / overwrite, and submitting the job editor."""

    def new_job(self, variant_key: str) -> None:
        variant = self.defs.variants[variant_key]
        last = read_last_used(self.project_dir, variant.label, variant.is_tomo) if self.project_dir else None
        if last is not None and external_jobs.kind_of(last) is not None:  # a CLI tool's, a wrapper's: not External's
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

    def new_aretomo_job(self, kind: str) -> None:
        """An AreTomo3 job: the form, with the modules and executable of the project's last one."""
        try:
            variant = self.aretomo.variant(kind)
        except RelionEnvError as e:
            self._env_error("AreTomo3", e)
            return
        values = variant.default_values(self.env)
        pipeline = self.snapshot.pipeline if self.snapshot is not None else None
        if self.project_dir is not None:
            jobs = [n for n, j in self.snapshot.jobs.items() if j.program in aretomo_jobs.TITLES.values()] \
                if self.snapshot is not None else None
            values.update(aretomo_jobs.last_settings(self.project_dir, pipeline, jobs))
        if pipeline is not None:  # the newest tilt-series set, as a start
            values.update(aretomo_jobs.newest_input(pipeline))
        self._edit(variant.program, self.editor.new_job, variant, values,
                   f"{variant.description}. Runs as a RELION External job (AreTomo3 from its executable).")

    def new_tomomask_job(self, kind: str = tomomask_jobs.MASKS) -> None:
        """A tomogram mask job: the form, with the newest tomogram set."""
        try:
            variant = self.tomomask.variant(kind)
        except RelionEnvError as e:
            self._env_error("Tomogram masks", e)
            return
        values = variant.default_values(self.env)
        pipeline = self.snapshot.pipeline if self.snapshot is not None else None
        if pipeline is not None:
            values.update(tomomask_jobs.newest_input(pipeline))
        self._edit(variant.program, self.editor.new_job, variant, values,
                   f"{variant.description}. Runs as a RELION External job; the masks are then drawn in its "
                   "Masks view.")

    def _variant_for(self, js: JobStar) -> JobVariant:
        """The form for a job: its job type's, or a CLI tool's, pytom's or AreTomo3's for External jobs made
        from those menus."""
        v = self.forms.variant_for(js)  # (None also for a CLI tool whose --help cannot be read)
        return v if v is not None else self.defs.variant_for_jobstar(js)

    def clone_job(self) -> None:
        job, project = self.current, self.project_dir
        if job is None or project is None:
            return
        try:
            js = JobStar.read(project / job / "job.star")
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
        job, project = self.current, self.project_dir
        if job is None or project is None:
            return
        try:
            js = JobStar.read(project / job / "job.star")
            variant = self._variant_for(js)
        except RelionEnvError as e:
            self._env_error("Continue", e)
            return
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self.window, "Continue", f"Cannot read {job}job.star: {e}")
            return
        self._edit("Continue", self.editor.continue_job, job, js, variant)

    def overwrite_job(self) -> None:
        job, project = self.current, self.project_dir
        if job is None or project is None:
            return
        try:
            js = JobStar.read(project / job / "job.star")
            variant = self._variant_for(js)
        except RelionEnvError as e:
            self._env_error("Overwrite", e)
            return
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self.window, "Overwrite", f"Cannot read {job}job.star: {e}")
            return
        info = self.snapshot.jobs.get(job) if self.snapshot is not None else None
        alias = (info.process.alias.rstrip("/").split("/", 1)[-1]
                 if info is not None and info.process.has_alias else "")
        self._edit("Overwrite", self.editor.overwrite_job, job, js, variant, alias)

    def _submit_overwrite(self, js: JobStar, alias: str, schedule_only: bool) -> None:
        """RELION's "Overwrite!": ask, then replace the job by a new one in the same directory."""
        pl, job = self._ready()[1], self.editor.job
        # the latest snapshot, not a read of the pipeline here (a slow file system would freeze the
        # window); Pipeliner.overwrite_job checks again with the pipeline as it is then
        snap = self.snapshot
        if snap is None or job is None or job not in snap.jobs:
            QMessageBox.warning(self.window, "Overwrite", f"{(job or 'The job').rstrip('/')} is not in the pipeline any more.")
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
        project = self._ready()[0]
        try:
            pipeline = Pipeline.read(project)  # with jobs added since the last snapshot
        except (OSError, ValueError, KeyError):
            pass
        return assign_scratch(js.options, project, pipeline, labels)

    def _check_values(self, variant: JobVariant | None, v: dict[str, str]) -> tuple[str, str] | None:
        """(title, what is wrong) when these options of this form cannot be submitted, else None: what RELION
        or the runner would refuse later, said now."""
        if isinstance(variant, PytomVariant):
            bad = quoted_values(v, variant.own_keys)
            if bad:
                return variant.program, ("These options cannot have a double quote (\") in them (RELION's job.star "
                                         "keeps them in double quotes): " + ", ".join(
                                             variant.options[k].label.rstrip(": ") for k in bad))
        if isinstance(variant, CliVariant):
            if v.get("nr_mpi", "1") not in ("", "1") and v.get("do_queue") == "Yes":
                return variant.program, ("MPI runs of CLI tools can only run locally: RELION's External jobs cannot "
                                         "ask the queue for MPI slots. Use 1 MPI process, or switch off 'Submit to "
                                         "queue'.")
            try:
                limit = self.env.error_local_mpi
            except RelionEnvError as e:  # RELION would refuse to start
                return variant.program, self._env_message(e)
            try:
                nr_mpi = int(float(v.get("nr_mpi") or 1))
            except ValueError:
                nr_mpi = 1  # cli_runner reports it
            if nr_mpi > limit:  # as relion_pipeliner refuses it for RELION's own jobs (cli_runner checks again)
                return variant.program, (f"You're submitting a local job with {nr_mpi} parallel MPI processes. "
                                         "That's more than allowed by the RELION_ERROR_LOCAL_MPI environment "
                                         f"variable ({limit}).")
        return None

    def _on_submit(self, js: JobStar, alias: str, schedule_only: bool) -> None:
        pl = self._ready()[1]
        kind = interactive_kind(js.type_label, js.options)
        if kind and self.editor.native_selector and not schedule_only:
            job = self.editor.job if self.editor.mode in (MODE_CONTINUE, MODE_OVERWRITE) else None
            self.open_tool(kind, js, alias, self.editor.mode, job)
            return
        problem = self._check_values(self.editor.variant, self.editor.values())
        if problem is not None:
            QMessageBox.warning(self.window, *problem)
            return
        if self.editor.mode == MODE_OVERWRITE:
            self._submit_overwrite(js, alias, schedule_only)
            return
        if self.editor.mode == MODE_CONTINUE:
            job = self.editor.job
            if job is None:  # a continue is always of a job
                return
            info = self.snapshot.jobs.get(job) if self.snapshot else None
            if info is not None and info.status == "Running":
                QMessageBox.warning(self.window, "Continue", f"{self._label(job)} is still running.")
                return
            edit = self._edits
            self._run(f"Continue {self._label(job)}", lambda: pl.continue_job(job, js),
                      lambda _: self._started(edit, f"{job.rstrip('/')} continued."))
            return
        if alias and self.snapshot is not None and self.editor.variant is not None:  # --setJobAlias checks again
            try:
                PipelineOps.check_alias(self.snapshot.pipeline, f"{self.editor.variant.dirname}/job000/", alias)
            except OpsError as e:
                QMessageBox.warning(self.window, "Alias", str(e))
                return

        variant, values = self.editor.variant, self.editor.values()
        cli = variant if isinstance(variant, CliVariant) else None  # its outputs go to the job's directory
        predicted = self.editor.job_dir() if cli is not None else None
        project = self._ready()[0]
        pipeline = self.editor.pipeline

        scratch: list[str | None] = [None]  # the job's own scratch directory, once given
        own_scratch = self.editor.own_scratch

        def add_and_run():
            if own_scratch:
                scratch[0] = self._own_scratch(js, pipeline)
            res = pl.add_job(js, alias or None)
            if cli is not None and res.job != predicted:  # outputs must go to its own directory
                cli.make_jobstar(values, False, res.job, pipeline).write(project / res.job / "job.star")
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
    def _on_submit_many(self, jobs: list[tuple[JobStar, str, dict[str, str]]], schedule_only: bool) -> None:
        """Variations…: add these new jobs, in order, then run each in its own relion_pipeliner (so they run at
        the same time, each to the queue if it is submitted there) or leave them scheduled."""
        if not jobs:
            return
        pl = self._ready()[1]
        project = self._ready()[0]
        variant = self.editor.variant
        for js, alias, values in jobs:
            problem = self._check_values(variant, values)
            if problem is not None:
                QMessageBox.warning(self.window, problem[0], f"{alias or js.type_label}: {problem[1]}")
                return
        if self.snapshot is not None and variant is not None:  # --setJobAlias checks again
            for _, alias, _ in jobs:
                try:
                    PipelineOps.check_alias(self.snapshot.pipeline, f"{variant.dirname}/job000/", alias)
                except OpsError as e:
                    QMessageBox.warning(self.window, "Alias", f"{alias}: {e}")
                    return
        cli = variant if isinstance(variant, CliVariant) else None  # its outputs go to the job's directory
        pipeline = self.editor.pipeline
        own_scratch = self.editor.own_scratch

        def add_and_run():
            added: list[str] = []
            try:
                for js, alias, values in jobs:
                    if own_scratch:
                        self._own_scratch(js, pipeline)
                    res = pl.add_job(js, alias or None)
                    if cli is not None:  # outputs must go to its own directory (the one guessed may be taken)
                        cli.make_jobstar(values, False, res.job, pipeline).write(project / res.job / "job.star")
                    added.append(res.job)
            except (PipelinerError, OSError) as e:
                raise PipelinerError(str(e) + (f"\n\nAdded before this (scheduled, not run): {', '.join(added)}"
                                               if added else ""))
            for job in [] if schedule_only else added:  # each its own pipeliner: they run side by side
                pl.run([job])
            return added

        edit = self._edits

        def done(added):
            if added:
                self.select.emit(added[0])
            verb = "scheduled" if schedule_only else "started"
            text = f"{len(added)} jobs {verb}: " + ", ".join(j.rstrip("/") for j in added) + "."
            if schedule_only:
                self.editor.status.setText(text)
            else:
                self._started(edit, text)

        what = f"{'Schedule' if schedule_only else 'Run'} {len(jobs)} variations"
        self._run(what, add_and_run, done)



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
