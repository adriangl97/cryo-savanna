# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Main data lineage of a pipeline: which parent a job is drawn under in the job tree.

The main input of a job is, in this order:

1. the one chosen for that job in cryo-savanna (``main_inputs``: job -> input node, kept by
   project/main_inputs.py), if it is still one of the job's inputs;
2. its tomography optimisation set (TomoOptimisationSet), else its particle set
   (ParticleGroupMetadata): the data a job works on. With several, the first in the job window's
   order;
3. else the first input-node option of the job's own job.star (window order, with the curated
   LINEAGE_INPUTS overrides) whose value is one of its pipeline inputs: RELION records inputs in the
   order its command builder adds them, which is not the data path (PostProcess lists the mask
   before the half map);
4. else the first pipeline input.

Other inputs are secondary links.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from cryo_savanna.jobdefs.model import JobDefs
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
    try:
        js = read_jobstar(Path(project_dir) / process_name / "job.star")
        variant = defs.variant_for_jobstar(js)
    except (OSError, ValueError, KeyError):
        js = variant = None
    if variant is not None:
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
    return default_input(pipeline, input_order(pipeline, process_name, project_dir, defs, read_jobstar))


def primary_parents(
    pipeline: Pipeline,
    project_dir: str | os.PathLike,
    defs: JobDefs,
    read_jobstar: Callable[[Path], JobStar] = JobStar.read,
    main_inputs: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """process name -> process that produced its primary input (None for roots)."""
    out: dict[str, str | None] = {}
    for name in pipeline.processes:
        node = primary_input(pipeline, name, project_dir, defs, read_jobstar, main_inputs)
        out[name] = pipeline.nodes[node].producer if node is not None else None
    return cut_loops(out)


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
