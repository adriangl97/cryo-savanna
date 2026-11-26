# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Optimisation sets for Select jobs on tomography particles (no Qt).

RELION 5's Select writes particles.star (particles_split<N>.star when it splits into subsets) and no
optimisation set (pipeline_jobs.cpp, getCommandsSelectJob), so a job after it (Refine3D, Class3D, Reconstruct
particle, CTF refinement, frame alignment...) needs direct entries: the particles, and the tomograms and
trajectories looked up by hand. cryo-savanna writes an optimisation set next to each particle file
(optimisation_set.star, optimisation_set_split<N>.star), as relion_refine writes one next to its data.star
(MlOptimiser::write): the selected particles, with the tomograms and the trajectories of the particles they were
selected from. It adds them to the pipeline as outputs of the Select job, so a job's "Input optimisation set"
offers them.

The tomograms and trajectories are those of the optimisation set next to Select's input (Class3D's
run_it025_optimiser.star: run_it025_optimisation_set.star; Extract's particles.star: optimisation_set.star), else
of an optimisation set the input's job has in the pipeline, else of what that job took as input (its job.star:
its optimisation set, or its direct entries). Nothing else of that set is kept: the reference maps, mask and FSC
of a refinement are of all its particles, not of those selected.

Done when the selection is saved in cryo-savanna's window (Pipeliner.finish_done_here) and, for a Select RELION
ran, when the window sees it finished (ui/main_window.py). The set is in the pipeline from then on: a job
scheduled before that, in the same run as the Select, takes direct entries as before.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.star import StarBlock, StarError, read_star, write_star
from cryo_savanna.relion_consts import STATUS_SUCCEEDED, job_type_for_label

PARTICLES_FILE, TOMOGRAMS_FILE, TRAJECTORIES_FILE = "rlnTomoParticlesFile", "rlnTomoTomogramsFile", \
    "rlnTomoTrajectoriesFile"
LABEL = "TomoOptimisationSet.star.relion.tomo.select"
PARTICLES_KIND, SET_KIND = "ParticleGroupMetadata", "TomoOptimisationSet"
_PARTICLES = re.compile(r"particles(_split\d+)?\.star$")
_ON_DATA = ("do_remove_duplicates", "do_select_values", "do_discard", "do_split")  # modes on fn_data, not fn_model


def is_select(type_label: str) -> bool:
    jt = job_type_for_label(type_label)
    return (jt.label if jt is not None else type_label) == "relion.select"


def set_for(particles: str) -> str:
    """The optimisation set written for a particle file of a Select job (``Select/job020/particles_split2.star``:
    ``Select/job020/optimisation_set_split2.star``)."""
    head, name = particles.rsplit("/", 1) if "/" in particles else ("", particles)
    m = _PARTICLES.match(name)
    split = (m.group(1) or "") if m else ""
    return (head + "/" if head else "") + f"optimisation_set{split}.star"


def particle_outputs(pipeline: Pipeline, job: str) -> list[str]:
    """The particle files a Select job has in the pipeline (particles.star, particles_split<N>.star)."""
    pr = pipeline.processes.get(job)
    if pr is None:
        return []
    return [n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == PARTICLES_KIND
            and _PARTICLES.search(n)]


def candidates(pipeline: Pipeline) -> list[str]:
    """Finished Select jobs with particles: those that may need optimisation sets (without reading a file)."""
    return [n for n, pr in pipeline.processes.items()
            if pr.status == STATUS_SUCCEEDED and is_select(pr.type_label) and particle_outputs(pipeline, n)]


def missing(project_dir: str | os.PathLike, pipeline: Pipeline, job: str) -> list[str]:
    """The particle files of a Select job without their optimisation set (as a file, or in the pipeline)."""
    project = Path(project_dir)
    out = []
    for parts in particle_outputs(pipeline, job):
        opt = set_for(parts)
        if not (project / opt).is_file() or opt not in pipeline.nodes:
            out.append(parts)
    return out


# ---------------------------------------------------------------------------- where the tomograms come from

def _values(path: Path) -> dict[str, str]:
    """{label: value} of an optimisation set (a list, or a loop with one row); {} if it is not one."""
    try:
        blocks = read_star(path)
    except (OSError, StarError, UnicodeDecodeError):
        return {}
    for b in blocks.values():
        if b.has(TOMOGRAMS_FILE) and b.rows:
            return {k: str(v) for k, v in b.as_dict().items()}
    return {}


def select_input(options: dict[str, str]) -> str:
    """The particles or classes the Select job selected from (as getCommandsSelectJob picks them): fn_data for
    duplicate removal, values and subsets, else fn_model; "" when it selected micrographs."""
    if any(options.get(k) == "Yes" for k in _ON_DATA):
        return "" if options.get("fn_mic", "").strip() else options.get("fn_data", "").strip()
    return options.get("fn_model", "").strip() or options.get("fn_data", "").strip()


def _beside(src: str) -> list[str]:
    """Optimisation sets written next to a file, by name: relion_refine's <root>_optimisation_set.star next to
    <root>_optimiser.star, _data.star or _model.star; optimisation_set.star next to particles.star."""
    head, name = src.rsplit("/", 1) if "/" in src else ("", src)
    pre = head + "/" if head else ""
    for suffix in ("_optimiser.star", "_data.star", "_half1_model.star", "_half2_model.star", "_model.star"):
        if name.endswith(suffix):
            return [pre + name[: -len(suffix)] + "_optimisation_set.star"]
    if _PARTICLES.match(name):
        return [set_for(src)]
    return []


def source_values(project_dir: str | os.PathLike, src: str, pipeline: Pipeline | None = None) -> dict[str, str]:
    """{rlnTomoTomogramsFile, rlnTomoTrajectoriesFile (if any)} of the particles in ``src`` (a particle,
    optimiser or model file); {} when none is found (not tomography)."""
    project = Path(project_dir)
    tried: list[str] = list(_beside(src))
    job = "/".join(src.split("/")[:2]) + "/" if src.count("/") >= 2 else ""
    if pipeline is not None and job in pipeline.processes:
        tried += [n for n in reversed(pipeline.processes[job].outputs)
                  if n in pipeline.nodes and pipeline.nodes[n].kind == SET_KIND]
    for rel in dict.fromkeys(tried):
        v = _values(project / rel)
        if v.get(TOMOGRAMS_FILE):
            return _keep(v)
    if not job:
        return {}
    try:  # what the input's job took
        o = JobStar.read(project / job / "job.star").options
    except (OSError, ValueError, KeyError):
        return {}
    if o.get("use_direct_entries", "No") != "Yes" and o.get("in_optimisation", "").strip():
        v = _values(project / o["in_optimisation"].strip())
        if v.get(TOMOGRAMS_FILE):
            return _keep(v)
    tomograms = o.get("in_tomograms", "").strip() or o.get("in_tomoset", "").strip()
    if tomograms:
        return _keep({TOMOGRAMS_FILE: tomograms, TRAJECTORIES_FILE: o.get("in_trajectories", "").strip()})
    return {}


def _keep(v: dict[str, str]) -> dict[str, str]:
    out = {TOMOGRAMS_FILE: v[TOMOGRAMS_FILE]}
    traj = v.get(TRAJECTORIES_FILE, "").strip()
    if traj and traj != "None":
        out[TRAJECTORIES_FILE] = traj
    return out


# ---------------------------------------------------------------------------- writing them

def write_sets(project_dir: str | os.PathLike, job: str, particles: list[str],
               pipeline: Pipeline | None = None) -> list[str]:
    """Write the optimisation set of each of these particle files of the Select ``job``; the sets written (none
    when the job's input is not tomography, or is no longer there)."""
    project = Path(project_dir)
    try:
        options = JobStar.read(project / job / "job.star").options
    except (OSError, ValueError, KeyError):
        return []
    src = select_input(options)
    values = source_values(project, src, pipeline) if src else {}
    if not values:
        return []
    out = []
    for parts in particles:
        if not (project / parts).is_file():
            continue
        opt = set_for(parts)
        write_star(project / opt, [StarBlock.from_dict("", {PARTICLES_FILE: parts, **values})])
        out.append(opt)
    return out


def complete(project_dir: str | os.PathLike, job: str, pipeline: Pipeline | None = None) -> list[str]:
    """Write the optimisation sets a finished Select job is missing and add them to the pipeline as its outputs;
    the sets added. Nothing for another job, or one not on tomography particles. ``pipeline``: the project's,
    as read a moment ago (else read here); it is not changed."""
    from cryo_savanna.project.ops import PipelineOps
    project = Path(project_dir)
    pipeline = pipeline or Pipeline.read(project)
    pr = pipeline.processes.get(job)
    if pr is None or pr.status != STATUS_SUCCEEDED or not is_select(pr.type_label):
        return []
    todo = missing(project, pipeline, job)
    if not todo:
        return []
    written = write_sets(project, job, todo, pipeline)
    if written:
        PipelineOps(project).add_outputs(job, [(w, LABEL) for w in written])
    return written
