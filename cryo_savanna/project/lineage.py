# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Main data lineage of a pipeline: which parent a job is drawn under in the job tree.

The main input of a job is, in this order:

1. the one chosen for that job in cryo-savanna (``main_inputs``: job -> input node, kept by
   project/main_inputs.py), if it is still one of the job's inputs;
2. for a pytom or AreTomo3 job (External jobs, jobdefs/pytom_jobs.py, aretomo_jobs.py), the file of its
   form's main input field (its tomograms, its matching job, its tilt series), whichever External field it
   went in. When the pipeline has no
   such input (an extraction added while its matching job ran: the matching job's output was no
   pipeline node yet, so RELION could not link it), the job sits under the job whose directory holds
   that file;
3. its tomography optimisation set (TomoOptimisationSet), else its particle set
   (ParticleGroupMetadata): the data a job works on. With several, the first in the job window's
   order;
4. else the first input-node option of the job's own job.star (window order, with the curated
   LINEAGE_INPUTS overrides) whose value is one of its pipeline inputs: RELION records inputs in the
   order its command builder adds them, which is not the data path (PostProcess lists the mask
   before the half map);
5. else the first pipeline input.

Other inputs are secondary links.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.external_jobs import main_input as own_form_main_input
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline


MAIN_INPUT_KINDS = ("TomoOptimisationSet", "ParticleGroupMetadata")  # preferred, in this order


def input_order(
    pipeline: Pipeline,
    process_name: str,
    project_dir: str | os.PathLike,
    defs: JobDefs,
    read_jobstar: Callable[[Path], JobStar] = JobStar.read,
) -> list[str]:
    """A job's pipeline inputs in the job window's order (curated overrides first), then the rest."""
    pr = pipeline.processes[process_name]
    inputs = set(pr.inputs)
    ordered: list[str] = []
    js: JobStar | None = None
    variant = None
    try:
        js = read_jobstar(Path(project_dir) / process_name / "job.star")
        variant = defs.variant_for_jobstar(js)
    except (OSError, ValueError, KeyError):
        js = variant = None
    if variant is not None and js is not None:
        for key in variant.lineage_inputs():
            value = js.options.get(key, "")
            if value in inputs and value not in ordered:
                ordered.append(value)
    return ordered + [n for n in pr.inputs if n not in ordered]


def default_input(pipeline: Pipeline, ordered: list[str]) -> str | None:
    """The main input when none was chosen: an optimisation set, else a particle set, else the first."""
    for kind in MAIN_INPUT_KINDS:
        for n in ordered:
            node = pipeline.nodes.get(n)
            if node is not None and node.kind == kind:
                return n
    return ordered[0] if ordered else None


def _own_main_input(
    process_name: str, project_dir: str | os.PathLike, read_jobstar: Callable[[Path], JobStar]
) -> str | None:
    """The main input a job's own form names (pytom, AreTomo3 and tomogram mask jobs), else None."""
    try:
        js = read_jobstar(Path(project_dir) / process_name / "job.star")
        return own_form_main_input(js)
    except (OSError, ValueError, KeyError):
        return None


def _job_holding(pipeline: Pipeline, path: str) -> str | None:
    """The job whose directory holds ``path`` (``Type/jobNNN/...``), if it is in the pipeline."""
    parts = path.split("/")
    if len(parts) < 3:
        return None
    job = f"{parts[0]}/{parts[1]}/"
    return job if job in pipeline.processes else None


def primary_input(
    pipeline: Pipeline,
    process_name: str,
    project_dir: str | os.PathLike,
    defs: JobDefs,
    read_jobstar: Callable[[Path], JobStar] = JobStar.read,
    main_inputs: Mapping[str, str] | None = None,
) -> str | None:
    """Pipeline input node on the job's main data path. ``read_jobstar`` lets callers cache job.star reads;
    ``main_inputs`` (job -> node) are the choices made in cryo-savanna."""
    pr = pipeline.processes[process_name]
    if not pr.inputs:
        return None
    chosen = (main_inputs or {}).get(process_name)
    if chosen in pr.inputs:
        return chosen
    own = _own_main_input(process_name, project_dir, read_jobstar)
    if own in pr.inputs:
        return own
    return default_input(pipeline, input_order(pipeline, process_name, project_dir, defs, read_jobstar))


def primary_parent(
    pipeline: Pipeline,
    process_name: str,
    project_dir: str | os.PathLike,
    defs: JobDefs,
    read_jobstar: Callable[[Path], JobStar] = JobStar.read,
    main_inputs: Mapping[str, str] | None = None,
) -> str | None:
    """The process that produced the job's primary input (None for a root)."""
    pr = pipeline.processes[process_name]
    if (main_inputs or {}).get(process_name) not in pr.inputs:  # no choice of cryo-savanna's applies
        own = _own_main_input(process_name, project_dir, read_jobstar)
        if own is not None and own not in pr.inputs:  # a pytom job whose main input RELION did not link
            node = pipeline.nodes.get(own)
            job = node.producer if node is not None and node.producer else _job_holding(pipeline, own)
            if job is not None:
                return job
    main = primary_input(pipeline, process_name, project_dir, defs, read_jobstar, main_inputs)
    return pipeline.nodes[main].producer if main is not None else None


def primary_parents(
    pipeline: Pipeline,
    project_dir: str | os.PathLike,
    defs: JobDefs,
    read_jobstar: Callable[[Path], JobStar] = JobStar.read,
    main_inputs: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """process name -> process that produced its primary input (None for roots)."""
    read_once = _read_once(read_jobstar)
    return cut_loops({name: primary_parent(pipeline, name, project_dir, defs, read_once, main_inputs)
                      for name in pipeline.processes})


def _read_once(read_jobstar: Callable[[Path], JobStar]) -> Callable[[Path], JobStar]:
    """``read_jobstar`` answering each file once, for one pass over a pipeline. A job's job.star is asked
    for up to three times in a pass (its own main input twice, the order of its window once), and a
    reader that caches by the file's stamp still looks at the file every time: three stats per job over
    a network file system, at every change of the pipeline."""
    seen: dict[Path, tuple[JobStar | None, Exception | None]] = {}

    def read(path: Path) -> JobStar:
        hit = seen.get(path)
        if hit is None:
            try:
                hit = (read_jobstar(path), None)
            except (OSError, ValueError, KeyError) as e:  # what the callers take for "no job.star"
                hit = (None, e)
            seen[path] = hit
        js, error = hit
        if error is not None or js is None:
            raise error if error is not None else OSError(f"{path}: not read")
        return js

    return read


def cut_loops(parents: dict[str, str | None]) -> dict[str, str | None]:
    """Parents without loops: a job whose primary input is its own output, or a loop of parents
    (never in a sound pipeline, but an input inside the job's own directory or a hand-edited
    pipeline makes one), becomes a root where the loop closes, so every job stays in the tree."""
    out = dict(parents)
    done: set[str] = set()
    for name in out:
        path: list[str] = []
        on_path: set[str] = set()
        a: str | None = name
        while a is not None and a not in done and a not in on_path:
            path.append(a)
            on_path.add(a)
            a = out.get(a)
        if a is not None and a in on_path:  # the walk came back to itself: the last job closes the loop
            out[path[-1]] = None
        done.update(path)  # jobs below a loop keep their parent
    return out
