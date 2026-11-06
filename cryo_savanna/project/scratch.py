# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""A scratch directory of its own for every new job.

RELION's "Copy particles to scratch directory" (scratch_dir) wipes <dir>/relion_volatile/ when a job
starts and when it ends, so two jobs given the same directory destroy each other's particle copies.
cryo-savanna therefore gives each new job a subdirectory savanna_NNN of the chosen directory, with a
3-digit number no other job of the project uses and that is not on disk already. RELION creates the
directories itself (mkdir -p) when the job starts.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

SCRATCH_KEY = "scratch_dir"
ENV_VAR = "RELION_SCRATCH_DIR"  # its default (RELION's), or cryo-savanna's Preferences when the shell leaves it unset
PREFIX = "savanna_"
_SUFFIX = re.compile(r"/+" + PREFIX + r"(?P<num>\d{3})/*$")
_OWN_NAME = re.compile(PREFIX + r"\d{3}")


def base_dir(path: str) -> str:
    """The directory the user chose: ``path`` without a savanna_NNN this module added."""
    m = _SUFFIX.search(path)
    return path if m is None else path[:m.start()] or "/"


def number_of(path: str) -> int | None:
    m = _SUFFIX.search(path)
    return int(m.group("num")) if m else None


def wants_scratch(path: str) -> bool:
    """A directory to copy to (empty: no scratch; $$name: a Schemes variable, filled in later)."""
    p = path.strip()
    return bool(p) and not p.startswith("$$")


def taken_numbers(project_dir: str | os.PathLike, pipeline: Pipeline | None, type_labels: Iterable[str]) -> set[int]:
    """The numbers the project's jobs (of the types with a scratch option) already have."""
    labels = set(type_labels)
    out: set[int] = set()
    if pipeline is None:
        return out
    for name, pr in pipeline.processes.items():
        if pr.type_label not in labels:
            continue
        try:
            js = JobStar.read(Path(project_dir) / name / "job.star")
        except (OSError, ValueError, KeyError):
            continue
        n = number_of(js.options.get(SCRATCH_KEY, ""))
        if n is not None:
            out.add(n)
    return out


def unique_dir(path: str, taken: set[int], exists: Callable[[str], bool] = os.path.exists,
               rng: random.Random | None = None) -> str:
    """``path``/savanna_NNN with a number not in ``taken`` and a directory not on disk yet.

    The number is random (not the next free one), so projects sharing a scratch disk rarely pick
    the same one either.
    """
    base = base_dir(path.strip()).rstrip("/") or "/"
    numbers = list(range(1000))
    (rng or random.SystemRandom()).shuffle(numbers)
    for n in numbers:
        if n in taken:
            continue
        candidate = os.path.join(base, f"{PREFIX}{n:03d}")
        if not exists(candidate):
            return candidate
    raise RuntimeError(f"No free scratch directory left in {base} (all {PREFIX}000 – {PREFIX}999 are used)")


def assign(options: dict[str, str], project_dir: str | os.PathLike, pipeline: Pipeline | None,
           type_labels: Iterable[str], exists: Callable[[str], bool] = os.path.exists) -> str | None:
    """Give a new job's options a scratch directory of its own; returns it (None: no scratch)."""
    path = options.get(SCRATCH_KEY, "")
    if not wants_scratch(path):
        return None
    options[SCRATCH_KEY] = unique_dir(path, taken_numbers(project_dir, pipeline, type_labels), exists)
    return options[SCRATCH_KEY]


def own_dirs(base: str | os.PathLike) -> list[Path]:
    """The savanna_NNN directories directly in ``base`` (not deeper, and no symlinks)."""
    try:
        with os.scandir(Path(base).expanduser()) as it:
            return sorted(Path(e.path) for e in it if _OWN_NAME.fullmatch(e.name) and e.is_dir(follow_symlinks=False))
    except OSError:
        return []


def wipe(dirs: Iterable[str | os.PathLike]) -> list[str]:
    """Delete these directories and everything in them; returns what could not be deleted."""
    errors = []
    for d in dirs:
        try:
            shutil.rmtree(d)
        except OSError as e:
            errors.append(f"{d}: {e.strerror or e}")
    return errors
