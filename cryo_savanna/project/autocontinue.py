# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Continuing a relion_refine job (Class2D, Class3D, InitialModel, Refine3D, MultiBody) from its latest iteration
(no Qt): for the job editor's Continue (the file to continue from filled in), and after SLURM ended the job at its
time limit (Preferences → "Continue jobs that hit the SLURM time limit"; ui/actions/running.py).

A job is continued after the time limit at most MAX_TIMES times per run of it (a job overwritten starts again
from none): a job that never gets through an iteration in the time it is given would otherwise be sent to the
queue for ever. How often each was continued is kept in .cryo-savanna/auto_continue.json.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.shared_json import read_json, update_json
from cryo_savanna.relion_consts import job_type_for_label

CONTINUABLE = {"relion.class2d", "relion.class3d", "relion.initialmodel", "relion.refine3d", "relion.multibody"}
MAX_TIMES = 3
RECORD = Path(".cryo-savanna") / "auto_continue.json"


def continuable(type_label: str) -> bool:
    jt = job_type_for_label(type_label)
    return (jt.label if jt is not None else type_label) in CONTINUABLE


def latest_optimiser(project_dir: str | os.PathLike, job: str) -> str | None:
    """The optimiser file of the latest iteration the job finished (of its latest run), relative to the project:
    what its Continue takes ("Continue from here"). None when it finished none yet."""
    from cryo_savanna.results.classes import iterations
    project = Path(project_dir)
    for _, model in reversed(iterations(project / job)):
        opt = model.with_name(model.name.replace("_half1_model.star", "_optimiser.star")
                              .replace("_model.star", "_optimiser.star"))
        if opt.is_file():  # written after the iteration's other files
            return os.path.relpath(opt, project)
    return None


def continue_jobstar(project_dir: str | os.PathLike, job: str) -> JobStar:
    """The job's job.star for a continue from its latest iteration. ValueError when it cannot be continued."""
    js = JobStar.read(Path(project_dir) / job / "job.star")
    if not continuable(js.type_label):
        raise ValueError(f"{job.rstrip('/')} is not a job that continues from an iteration")
    fn = latest_optimiser(project_dir, job)
    if fn is None:
        raise ValueError(f"{job.rstrip('/')} has not finished an iteration yet: there is nothing to continue from")
    js.is_continue = True
    js.options["fn_cont"] = fn
    return js


def _run_key(project_dir: str | os.PathLike, job: str) -> str:
    """The job and when it first ran (note.txt): a job overwritten since is another run of it."""
    from cryo_savanna.project.slurm import _first_start
    first = _first_start(Path(project_dir) / job / "note.txt")
    return f"{job}@{int(first) if first else 0}"


def times_continued(project_dir: str | os.PathLike, job: str) -> int:
    runs = read_json(Path(project_dir) / RECORD).get("runs", {})
    entry = runs.get(_run_key(project_dir, job), []) if isinstance(runs, dict) else []
    return len(entry) if isinstance(entry, list) else 0


def note_continued(project_dir: str | os.PathLike, job: str, slurm_id: str) -> int:
    """Remember that the job was continued after SLURM job ``slurm_id`` hit its time limit; how often it was."""
    key = _run_key(project_dir, job)
    count = [0]

    def change(data: dict) -> dict:
        found = data.get("runs")
        runs: dict = found if isinstance(found, dict) else {}
        entry = [i for i in runs.get(key, []) if isinstance(i, str)] if isinstance(runs.get(key), list) else []
        if slurm_id not in entry:
            entry.append(slurm_id)
        runs[key] = entry
        count[0] = len(entry)
        return {**data, "runs": runs}
    update_json(Path(project_dir) / RECORD, change)
    return count[0]
