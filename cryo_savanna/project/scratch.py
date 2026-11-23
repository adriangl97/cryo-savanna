# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""A scratch directory of its own for a job that would share one with another job that runs.

RELION's "Copy particles to scratch directory" (scratch_dir) wipes <dir>/relion_volatile/ when a job
starts and when it ends, so two jobs that run at the same time with the same directory destroy each
other's particle copies. With Preferences → "A subdirectory of its own when the directory is in use",
cryo-savanna looks when it starts a job (Pipeliner.run: new, overwritten, continued and scheduled jobs
alike) whether another job uses that directory now:

- a process of this machine (of any user and project, also one RELION's own GUI started) has it as its
  ``--scratch_dir``: how RELION gives it to relion_refine(_mpi) and the tomo programs;
- a job of this project that runs here (not in a queue) has it in its job.star: marked Running, or in a
  run of a relion_pipeliner that is still going on this host (waiting for its GPUs, or about to start).

Only then does the job get a subdirectory savanna_NNN of it, with a random 3-digit number no such job has
and that is not on disk already; when ps cannot be read, the directory counts as in use. Otherwise the job
keeps the directory as chosen (a savanna_NNN left in its job.star is taken off again). RELION creates the
directories itself (mkdir -p) when the job starts. A job sent to a queue (Slurm) is left as it is: it gets
a scratch disk of its node. Jobs on other machines that share the directory cannot be seen.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.gpu_wait import process_table

SCRATCH_KEY = "scratch_dir"
ENV_VAR = "RELION_SCRATCH_DIR"  # its default (RELION's), or cryo-savanna's Preferences when the shell leaves it unset
PREFIX = "savanna_"
_SUFFIX = re.compile(r"/+" + PREFIX + r"(?P<num>\d{3})/*$")
_OWN_NAME = re.compile(PREFIX + r"\d{3}")
# a program's --scratch_dir (RELION writes it unquoted; a shell running the job's command may have quotes)
_SCRATCH_ARG = re.compile(r"--scratch_dir(?:=|\s+)(?P<q>['\"]?)(?P<dir>[^\s'\"]+)(?P=q)")


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


def unique_dir(path: str, taken: set[int], exists: Callable[[str], bool] = os.path.exists,
               rng: random.Random | None = None) -> str:
    """``path``/savanna_NNN with a number not in ``taken`` and a directory not on disk yet (left by a job
    before: RELION removes only relion_volatile/ in it).

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


def queued(options: Mapping[str, str]) -> bool:
    """Sent to a queue (Slurm): it runs on a node, with that node's scratch disk."""
    return (options.get("do_queue") or "").strip() == "Yes"


def normalized(path: str, project_dir: str | os.PathLike) -> str:
    """A scratch directory as a path to compare (~ expanded, relative to the project, symlinks followed)."""
    p = os.path.expanduser(path.strip())
    return os.path.realpath(os.path.join(project_dir, p))


Table = Callable[[], Mapping[int, tuple[int, str]]]  # {pid: (parent pid, command line)}, gpu_wait.process_table


def dirs_on_host(project_dir: str | os.PathLike, table: Table | None = None) -> set[str] | None:
    """The scratch directories processes of this machine copy to (their --scratch_dir), normalized; None when
    ps cannot be read. A relative one is taken as relative to this project (where its process runs is not
    looked up)."""
    procs = (table or process_table)()
    if not procs:  # ps lists itself at least
        return None
    return {normalized(m.group("dir"), project_dir) for _, args in procs.values() for m in _SCRATCH_ARG.finditer(args)}


@dataclass
class InUse:
    """What other jobs use now: when a job starts, its scratch directory is checked against it."""
    dirs: set[str] = field(default_factory=set)  # normalized
    numbers: set[int] = field(default_factory=set)  # the savanna_NNN numbers among them
    unknown: bool = False  # ps could not be read: every directory counts as in use


def in_use(project_dir: str | os.PathLike, others: Iterable[Mapping[str, str]], table: Table | None = None) -> InUse:
    """The scratch directories in use: those of this machine's processes, and of ``others``, the options of
    the project's jobs that run, or are about to (those sent to a queue do not count)."""
    host = dirs_on_host(project_dir, table)
    use = InUse(set(host or ()), unknown=host is None)
    for o in others:
        path = o.get(SCRATCH_KEY, "")
        if wants_scratch(path) and not queued(o):
            use.dirs.add(normalized(path, project_dir))
    use.numbers = {n for n in map(number_of, use.dirs) if n is not None}
    return use


def settle(options: dict[str, str], project_dir: str | os.PathLike, use: InUse,
           exists: Callable[[str], bool] = os.path.exists, rng: random.Random | None = None) -> str | None:
    """When a job starts: its scratch directory, in ``options``. The one chosen when no other job uses it, else
    a savanna_NNN of it (the job's own from before, if that is free). Returns the savanna_NNN given, None when it
    has none (no scratch, sent to a queue: left as it is)."""
    path = options.get(SCRATCH_KEY, "")
    if not wants_scratch(path) or queued(options):
        return None
    base = base_dir(path.strip())
    if not use.unknown and normalized(base, project_dir) not in use.dirs:
        options[SCRATCH_KEY] = base
        return None
    n = number_of(path)
    if n is None or n in use.numbers or normalized(path, project_dir) in use.dirs:
        path = unique_dir(base, use.numbers, exists, rng)
    options[SCRATCH_KEY] = path
    use.dirs.add(normalized(path, project_dir))  # the next job of the same start: another one
    use.numbers.add(number_of(path) or 0)
    return path


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
