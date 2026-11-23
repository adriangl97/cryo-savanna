# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Split off zero-weight particles as a job ("Additional external jobs"): the particles a refinement stopped on.

When relion_refine stops because a particle got all-zero weights (RELION's "A particle image was compared to
the reference and resulted in all-zero weights", project/failure_hints.py), this job takes the failed job's
folder, finds the particles its run.err names among its input particles, and writes accepted_particles.star
and rejected_particles.star (and for tomography optimisation_set.star with the accepted ones), to run the
refinement again on. The form can also reject every particle of those particles' tomograms (tomography) or
micrographs (SPA).

Like the tomogram masks (jobdefs/tomomask_jobs.py), a RELION External job: the executable is
"python3 .cryo-savanna/split_runner.py zero_weights" (a copy of cryo_savanna/split_runner.py in the project,
with runner_common.py), the form's options go to External's "Additional arguments" as ``--key value``. The
failed job's input particles (or optimisation set) go into External's input fields, so RELION links them; the
job is drawn under the failed job (its main input: the failed job's folder).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from cryo_savanna import split_runner as S
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import runner_files
from cryo_savanna.jobdefs.cli_tools import INPUT_SLOTS
from cryo_savanna.jobdefs.model import JobDefs, OptionDef, Row, Tab
from cryo_savanna.jobdefs.pytom_jobs import EXTERNAL, PytomVariant, _o, _yn, add_running_tab, decode
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(S.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/split_runner.py"
ZERO_WEIGHTS = S.ZERO_WEIGHTS
TITLES = {ZERO_WEIGHTS: "Split off zero-weight particles"}
DESCRIPTIONS = {
    ZERO_WEIGHTS: "For a refinement or classification that stopped because a particle got all-zero weights "
                  "(\"A particle image was compared to the reference and resulted in all-zero weights\"): the "
                  "particles its run.err names go to rejected_particles.star, the others to "
                  "accepted_particles.star (and, for tomography, optimisation_set.star), to run it again on",
}
MENU_TEXT = {ZERO_WEIGHTS: "Split off zero-weight particles (of a failed refinement)"}
MAIN_INPUTS = {ZERO_WEIGHTS: "in_job"}
# the jobs that run relion_refine (base type labels)
REFINE_TYPES = ("relion.refine3d", "relion.class3d", "relion.class2d", "relion.initialmodel", "relion.multibody")
_LINKED = ("TomoOptimisationSet", "ParticleGroupMetadata")  # the failed job's inputs this job links, in this order


def split_options(tomo: bool = False) -> tuple[list[OptionDef], list[Tab]]:
    """The form's own options, with whether whole tomograms (tomography) or micrographs (SPA) go."""
    opts = [
        _o("in_job", "Failed job:", "", "The folder of the job that stopped with RELION's all-zero weights error "
           "(3D auto-refine, 3D or 2D classification, initial model), e.g. Refine3D/job012/. Its run.err names "
           "the particles (a 'fn_img=' line before the error, also for the copies on a scratch disk); they are "
           "looked up in the job's input particles (its job.star, else note.txt): for tomography the particles "
           "of its optimisation set, or of its direct entries. Every attempt of the job adds to run.err, so the "
           "particles of all of them are split off.", kind="folder"),
    ]
    rows = [Row(("in_job",))]
    if tomo:
        opts.append(_yn(S.WHOLE_TOMOGRAMS, "Reject their whole tomograms?", "No",
                        "Yes: every particle of the tomograms (rlnTomoName) the named particles are in goes to "
                        "rejected_particles.star, e.g. when a tomogram's alignment or CTF is off and its other "
                        "particles would stop the next run too. No: only the named particles."))
        rows.append(Row((S.WHOLE_TOMOGRAMS,)))
    else:
        opts.append(_yn(S.WHOLE_MICROGRAPHS, "Reject their whole micrographs?", "No",
                        "Yes: every particle of the micrographs (rlnMicrographName) the named particles are in goes "
                        "to rejected_particles.star, e.g. when a micrograph is mostly ice, carbon or aggregates and "
                        "its other particles would stop the next run too. No: only the named particles."))
        rows.append(Row((S.WHOLE_MICROGRAPHS,)))
    return opts, [Tab("Input", tuple(rows))]


def normalise_job(value: str) -> str:
    """A job folder as RELION names jobs ("Refine3D/job012/"), or "" for nothing."""
    v = value.strip()
    while v.startswith("./"):
        v = v[2:]
    return (v.rstrip("/") + "/") if v.strip("/") else ""


@dataclass
class SplitVariant(PytomVariant):
    """The form: a PytomVariant (an own form written to an External job.star) with this runner, for SPA or
    tomography (the job.star's is-tomo flag follows the form, not always tomo)."""

    def make_jobstar(self, values: dict[str, str], is_continue: bool, job_dir: str,
                     pipeline: Pipeline | None) -> JobStar:
        values = dict(values)
        values["in_job"] = normalise_job(values.get("in_job", ""))
        js = super().make_jobstar(values, is_continue, job_dir, pipeline)
        js.is_tomo = self.is_tomo
        return js

    def _fill_inputs(self, o: dict[str, str], values: dict[str, str], pipeline: Pipeline) -> None:
        """The failed job's optimisation set or particles (its pipeline inputs) go in External's input fields."""
        for f in linked_inputs(pipeline, normalise_job(values.get("in_job", ""))):
            kind = pipeline.nodes[f].kind
            slot = next((k for k, want in INPUT_SLOTS if want == kind and not o.get(k)), None)
            slot = slot or next((k for k, _ in INPUT_SLOTS if not o.get(k)), None)
            if slot is not None:
                o[slot] = f


def linked_inputs(pipeline: Pipeline, job: str) -> list[str]:
    """The inputs of the failed job that hold its particles (its optimisation set, its particles), the first
    of each kind."""
    pr = pipeline.processes.get(job)
    if pr is None:
        return []
    out = []
    for kind in _LINKED:
        f = next((n for n in pr.inputs if n in pipeline.nodes and pipeline.nodes[n].kind == kind
                  and pipeline.nodes[n].producer), None)
        if f is not None:
            out.append(f)
    return out


def runner_command(kind: str = ZERO_WEIGHTS) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def split_kind(fn_exe: str) -> str | None:
    """zero_weights for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in S.MODES and \
            any(Path(p.strip("'\"")).name == "split_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def split_kind_of(js: JobStar) -> str | None:
    return split_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def split_main_input(js: JobStar) -> str | None:
    """The failed job's folder (the form's main input), else None: the job hangs under the failed job."""
    kind = split_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    return normalise_job(own.get(MAIN_INPUTS[kind], "")) or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's split_runner.py and the runner_common.py it imports into the project."""
    return runner_files.install_runner(project_dir, RUNNER, RUNNER_IN_PROJECT)


class SplitCatalog:
    """The form, built once per RELION environment (its queue settings), for SPA and for tomography."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[tuple[str, bool], SplitVariant] = {}

    def variant(self, kind: str = ZERO_WEIGHTS, tomo: bool = False) -> SplitVariant:
        if (kind, tomo) not in self._cache:
            self._cache[kind, tomo] = self._build(kind, tomo)
        return self._cache[kind, tomo]

    def _build(self, kind: str, tomo: bool) -> SplitVariant:
        if kind not in TITLES:
            raise KeyError(f"no split job {kind!r}")
        external = self.defs.variant(EXTERNAL, tomo)
        ext = external.resolved_options(self.env)
        own, tabs = split_options(tomo)
        options = {o.key: o for o in own}
        options["nr_threads"] = replace(ext["nr_threads"], help=(
            "CPU threads to reserve on the queue (XXXthreadsXXX). The job reads and writes STAR files only: 1 "
            "is enough, and it is quick enough to run locally."))
        tabs, groups = add_running_tab(options, tabs, {}, ext)
        return SplitVariant(key=f"split:{kind}" + (":tomo" if tomo else ""), label=EXTERNAL,
                            proc_id=external.proc_id, dirname=external.dirname, is_tomo=tomo, options=options,
                            tabs=tabs, groups=groups, kind=kind, program=TITLES[kind],
                            description=DESCRIPTIONS[kind], own_keys=tuple(o.key for o in own), external=external,
                            env=self.env, fn_exe=runner_command(kind))


def failed_job(pipeline: Pipeline, current: str | None) -> dict[str, str]:
    """The form's input to start with: the selected job if it is a failed refinement, else the newest one."""
    def failed_refine(name: str | None) -> bool:
        pr = pipeline.processes.get(name) if name else None
        return pr is not None and pr.status == "Failed" and any(
            pr.type_label == t or pr.type_label.startswith(t + ".") for t in REFINE_TYPES)

    if failed_refine(current):
        return {"in_job": str(current)}
    for name in reversed(list(pipeline.processes)):
        if failed_refine(name):
            return {"in_job": name}
    return {}
