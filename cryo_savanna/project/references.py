# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Several initial references for a 3D classification.

relion_refine takes them as a STAR file that lists the maps (``--ref references.star``, one row of
_rlnReferenceImage per map). Each map then starts one class, in that order, and ``--K`` is ignored
(MlModel::initialiseFromImages: "ignore nr_classes from the command line, use number of entries in STAR
file"); no other flag is needed. RELION's own job window has no field for this, so the job editor keeps the
list and the file is written into the job's own directory when the job is added
(``Class3D/job012/references.star``, job.star's "Reference map:"). RELION still declares the job's output
maps from "Number of classes:" (getOutputNodesRefine), which the editor sets to the number of maps.

The pipeline records that STAR file as the job's input; the maps' own jobs are linked by
project/snapshot.py, which reads it (``read``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from cryo_savanna.project.star import StarBlock, StarError, read_star, write_star

FILE = "references.star"
KEY = "fn_ref"  # the job option that names the reference(s)
NR_CLASSES = "nr_classes"
LABELS = frozenset({"relion.class3d"})  # job types that take several references this way
BLOCK = "model_classes"  # the block relion_refine looks in first (then any block with the label)
LABEL = "rlnReferenceImage"

_OWN = re.compile(r"(?:\./)?[^/\s]+/job\d{3,}/" + re.escape(FILE))


def supported(label: str) -> bool:
    return label in LABELS


def path_in(job_dir: str) -> str:
    """Where a job's list of references goes: ``Class3D/job012/references.star``."""
    return job_dir.rstrip("/") + "/" + FILE


def is_list_path(value: str) -> bool:
    """Whether a "Reference map:" value names a job's own list of references (as path_in makes it)."""
    return bool(_OWN.fullmatch(value.strip()))


def write(project_dir: str | os.PathLike, job_dir: str, maps: list[str]) -> Path:
    """Write the list into the job's directory; returns the file."""
    path = Path(project_dir) / path_in(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_star(path, [StarBlock.from_records(BLOCK, [LABEL], [[m] for m in maps])])
    return path


def read(project_dir: str | os.PathLike, value: str) -> list[str] | None:
    """The maps a "Reference map:" value lists, if it names a STAR file of references (relative to the
    project, as RELION runs from there); None for a map or a file that cannot be read."""
    value = value.strip()
    if not value.endswith(".star"):
        return None
    path = Path(value) if os.path.isabs(value) else Path(project_dir) / value
    try:
        blocks = read_star(path)
    except (OSError, StarError, ValueError):
        return None
    block = blocks.get(BLOCK)
    if block is None or not block.has(LABEL):  # relion_refine reads any block with the label then
        block = next((b for b in blocks.values() if b.has(LABEL)), None)
    if block is None:
        return None
    maps = [str(m) for m in block.column(LABEL) if str(m).strip()]
    return maps or None
