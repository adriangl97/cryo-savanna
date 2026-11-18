# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# clean_targets() follows PipeLine::cleanupJob in src/pipeliner.cpp. RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
"""How much disk space a project takes, job by job, and what cleaning would give back.

``scan()`` measures every job directory of the pipeline, Trash/ (one entry per Trash/<Type>/<jobNNN>/)
and the rest of the project, as ``du`` does: the blocks allocated, symbolic links not followed (the movies
an Import job links to are not counted) and a file with several hard links counted once.

``clean_targets()`` is the choice RELION makes in PipeLine::cleanupJob (relion_pipeliner --gentle_clean /
--harsh_clean N), so the space a clean frees can be shown before it is done. Cleaning itself is left to
relion_pipeliner. RELION does not delete what it cleans: it moves it to Trash/ (``mv -f <file>
Trash/<dir>/``), where it takes as much space as before until Trash/ is emptied. ``empty_trash()`` does
that, for chosen entries of Trash/: for good, and a deleted job whose copy is removed can no longer be
undeleted.
"""

from __future__ import annotations

import glob
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.ops import TRASH_DIR
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.relion_consts import STATUS_SUCCEEDED, job_type_for_label

DELETED_PIPELINE = "deleted_pipeline.star"  # in Trash/<Type>/<jobNNN>/ of a deleted job: it can be undeleted
NO_HARSH_CLEAN = "NO_HARSH_CLEAN"  # RELION's "clean all" leaves a job with this file out of harsh cleaning

# RELION's job types (pipeline_jobs.h) as cleanupJob tells them apart, by label
_NO_CLEANUP = {"relion.import", "relion.manualpick", "relion.select", "relion.maskcreate", "relion.joinstar",
               "relion.localres"}
_ITERATIONS = {"relion.class2d", "relion.class3d", "relion.refine3d", "relion.initialmodel", "relion.multibody"}
_CTFREFINE_PATTERNS = ("*_wAcc_optics-group*.mrc", "*_xyAcc_optics-group*.mrc", "*_aberr-Axx_optics-group_*.mrc",
                       "*_aberr-Axy_optics-group_*.mrc", "*_aberr-Ayy_optics-group_*.mrc",
                       "*_aberr-bx_optics-group_*.mrc", "*_aberr-by_optics-group_*.mrc", "*_mag_optics-group_*.mrc",
                       "*_fit.star", "*_fit.eps")
_ITERATION_PATTERNS = ("run_it[0-9][0-9][0-9]_data.star", "run_ct[0-9]_it[0-9][0-9][0-9]_data.star",
                       "run_ct[0-9][0-9]_it[0-9][0-9][0-9]_data.star",
                       "run_ct[0-9][0-9][0-9]_it[0-9][0-9][0-9]_data.star")


class Cancelled(Exception):
    """The scan was stopped (its window was closed)."""


@dataclass
class Usage:
    bytes: int = 0  # allocated on disk
    files: int = 0  # files and links (not directories)

    def add(self, other: Usage) -> None:
        self.bytes += other.bytes
        self.files += other.files


@dataclass
class JobUsage:
    name: str  # "Class2D/job006/"
    alias: str
    type_label: str
    status: str  # in the pipeline (what relion_pipeliner goes by)
    usage: Usage
    exists: bool = True  # its directory is there
    gentle: Usage | None = None  # what a gentle clean moves to Trash/ (None: RELION does not clean the job)
    harsh: Usage | None = None
    no_harsh_clean: bool = False  # NO_HARSH_CLEAN: left out of RELION's harsh "clean all"


@dataclass
class TrashEntry:
    path: str  # "Trash/Class2D/job006", relative to the project
    usage: Usage
    job: str | None  # "Class2D/job006/" for a Trash/<Type>/<jobNNN> entry
    deleted_job: bool  # has deleted_pipeline.star: a deleted (or overwritten) job, which can be undeleted
    in_pipeline: bool  # the pipeline has a job of that name (its cleaned files, or its overwritten earlier run)


@dataclass
class ProjectUsage:
    project_dir: Path
    jobs: list[JobUsage]
    trash: list[TrashEntry]
    other: list[tuple[str, Usage]]  # everything else, by top-level path (or Type/<folder> not in the pipeline)
    disk_total: int | None = None  # the file system the project is on
    disk_free: int | None = None
    scanned_at: float = field(default_factory=time.time)

    @property
    def jobs_total(self) -> Usage:
        return _sum(j.usage for j in self.jobs)

    @property
    def trash_total(self) -> Usage:
        return _sum(t.usage for t in self.trash)

    @property
    def other_total(self) -> Usage:
        return _sum(u for _, u in self.other)

    @property
    def total(self) -> Usage:
        return _sum([self.jobs_total, self.trash_total, self.other_total])

    def by_type(self) -> dict[str, tuple[int, Usage]]:
        """Job type directory (Class2D, ...) -> (number of jobs, their usage), largest first."""
        out: dict[str, tuple[int, Usage]] = {}
        for j in self.jobs:
            t = j.name.split("/", 1)[0]
            n, u = out.get(t, (0, Usage()))
            u.add(j.usage)
            out[t] = (n + 1, u)
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1].bytes))


def _sum(usages) -> Usage:
    out = Usage()
    for u in usages:
        out.add(u)
    return out


# ---------------------------------------------------------------------------- measuring

class _Meter:
    """du: allocated blocks, symbolic links not followed, each inode counted once over the whole scan."""

    def __init__(self, cancel: threading.Event | None = None):
        self.seen: set[tuple[int, int]] = set()
        self.cancel = cancel

    def _file(self, st: os.stat_result) -> int:
        if st.st_nlink > 1:
            key = (st.st_dev, st.st_ino)
            if key in self.seen:
                return 0
            self.seen.add(key)
        blocks = getattr(st, "st_blocks", None)
        return blocks * 512 if blocks is not None else st.st_size

    def path(self, path: str | os.PathLike) -> Usage:
        """Usage of a file, link or directory tree (nothing for what is not there)."""
        try:
            st = os.lstat(path)
        except OSError:
            return Usage()
        if not os.path.isdir(path) or os.path.islink(path):
            return Usage(self._file(st), 1)
        out = Usage(self._file(st), 0)  # the directory itself
        stack = [os.fspath(path)]
        while stack:
            if self.cancel is not None and self.cancel.is_set():
                raise Cancelled()
            d = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if e.is_dir(follow_symlinks=False):
                    out.bytes += self._file(st)
                    stack.append(e.path)
                else:
                    out.bytes += self._file(st)
                    out.files += 1
        return out

    def paths(self, paths: list[str]) -> Usage:
        """Usage of several paths, of which some may lie in others (a directory and files in it): counted once."""
        out = Usage()
        for p in _outermost(paths):
            out.add(self.path(p))
        return out


def _outermost(paths: list[str]) -> list[str]:
    """``paths`` without those inside another one of them."""
    norm = sorted({os.path.normpath(p) for p in paths})
    out: list[str] = []
    for p in norm:
        if not any(p.startswith(q + os.sep) for q in out):
            out.append(p)
    return out


# ---------------------------------------------------------------------------- RELION's cleanupJob

def _glob(root: str, pattern: str) -> list[str]:
    """Names relative to the project, as RELION globs them there (FileName::globFiles: glob(3), sorted)."""
    return sorted(glob.glob(pattern, root_dir=root))


def _subdirs(job_dir: str) -> list[str]:
    """Every directory under the job's, relative and ending in "/", breadth first as cleanupJob finds them:
    real directories only (not links to one), none whose name starts with ".", nor what is under one."""
    out: list[str] = []
    level = [""]
    while level:
        nxt = []
        for rel in level:
            try:
                entries = sorted(os.scandir(os.path.join(job_dir, rel)), key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                if e.name[0] != "." and e.is_dir(follow_symlinks=False):
                    nxt.append(rel + e.name + "/")
        out += nxt
        level = nxt
    return out


def clean_targets(project_dir: str | os.PathLike, pipeline: Pipeline, job: str, harsh: bool) -> list[str] | None:
    """The files and directories (relative to the project) that ``relion_pipeliner --gentle_clean`` (or
    ``--harsh_clean``) of ``job`` moves to Trash/, as PipeLine::cleanupJob chooses them; None when RELION
    refuses to clean the job (it has not finished successfully). An empty list: nothing to clean."""
    pr = pipeline.processes.get(job)
    if pr is None or pr.status != STATUS_SUCCEEDED:
        return None
    jt = job_type_for_label(pr.type_label)
    label = jt.label if jt is not None else pr.type_label
    if label in _NO_CLEANUP:
        return []
    return _targets(os.fspath(project_dir), pipeline, pr, label, harsh)


def _targets(root: str, pipeline: Pipeline, pr: Process, label: str, harsh: bool) -> list[str]:
    """cleanupJob's choice, run in the project ``root`` (RELION's names are relative to it)."""
    name = pr.name
    job_dir = os.path.join(root, name)
    subdirs = _subdirs(job_dir) if os.path.isdir(job_dir) else []
    out: list[str] = []

    def g(pattern: str) -> list[str]:
        return _glob(root, pattern)

    if label == "relion.motioncorr":
        for d in subdirs:
            if harsh:
                out.append(name + d)
            else:
                for ext in ("*.com", "*.err", "*.out", "*.log"):
                    out += g(name + d + ext)
    elif label == "relion.ctffind":
        out += g(name + "gctf*.out") + g(name + "gctf*.err")
        out += [name + d for d in subdirs]
    elif label == "relion.autopick":
        for d in subdirs:
            out += g(name + d + "*.spi")
    elif label == "relion.extract":
        for d in subdirs:
            out += [name + d] if harsh else g(name + d + "*_extract.star")
    elif label in _ITERATIONS:
        iters: list[str] = []
        for pattern in _ITERATION_PATTERNS:
            iters += g(name + pattern)
        nodes = list(pipeline.nodes)
        # all but the last one found (as RELION: the last of these globs, not always the highest iteration)
        for data in iters[:-1]:
            prefix = data[:-len("_data.star")]
            if not any(prefix in n for n in nodes):  # iterations that are in the pipeline are kept
                out += g(prefix + "*")
            if harsh and label == "relion.multibody":
                out += g(name + "analyse_component???_bin???.mrc")
    elif label == "relion.ctfrefine":
        for d in subdirs:
            for pattern in _CTFREFINE_PATTERNS:
                out += g(name + d + pattern)
    elif label == "relion.polish":  # RELION's MotionRefine
        for d in subdirs:
            for pattern in ("*_FCC_cc.mrc", "*_FCC_w0.mrc", "*_FCC_w1.mrc") + (
                    ("*_shiny.mrcs", "*_shiny.star") if harsh else ()):
                out += g(name + d + pattern)
    elif label == "relion.subtract":
        if harsh:
            out += g(name + "subtracted.*")
    return list(dict.fromkeys(out))  # each once, in the order found


# ---------------------------------------------------------------------------- the scan

def scan(project_dir: str | os.PathLike, pipeline: Pipeline, cancel: threading.Event | None = None) -> ProjectUsage:
    """Measure the project (slow on a large project: run it in a thread; ``cancel`` stops it)."""
    project = Path(project_dir)
    meter = _Meter(cancel)
    jobs: list[JobUsage] = []
    for name, pr in pipeline.processes.items():
        d = project / name
        ju = JobUsage(name, pr.alias, pr.type_label, pr.status, meter.path(d), exists=d.is_dir())
        ju.no_harsh_clean = (d / NO_HARSH_CLEAN).exists()
        jobs.append(ju)
    # what a clean would move: measured apart (those files were counted with their job already)
    clean_meter = _Meter(cancel)
    for ju in jobs:
        for harsh in (False, True):
            targets = clean_targets(project, pipeline, ju.name, harsh) if ju.exists else None
            if targets is not None:
                u = clean_meter.paths([str(project / p) for p in targets])
                setattr(ju, "harsh" if harsh else "gentle", u)
    trash = _trash_entries(project, pipeline, meter)
    other = _other(project, pipeline, meter)
    total: int | None = None
    free: int | None = None
    try:
        disk = shutil.disk_usage(project)
        total, free = disk.total, disk.free
    except OSError:
        pass
    return ProjectUsage(project, jobs, trash, other, total, free)


def _trash_entries(project: Path, pipeline: Pipeline, meter: _Meter) -> list[TrashEntry]:
    root = project / TRASH_DIR
    out: list[TrashEntry] = []
    try:
        tops = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return out
    for top in tops:
        if top.is_dir(follow_symlinks=False):
            try:
                inner = sorted(os.scandir(top.path), key=lambda e: e.name)
            except OSError:
                inner = []
            for e in inner:
                job = f"{top.name}/{e.name}/" if e.is_dir(follow_symlinks=False) else None
                out.append(TrashEntry(f"{TRASH_DIR}/{top.name}/{e.name}", meter.path(e.path), job,
                                      job is not None and os.path.exists(os.path.join(e.path, DELETED_PIPELINE)),
                                      job is not None and job in pipeline.processes))
            if not inner:
                out.append(TrashEntry(f"{TRASH_DIR}/{top.name}", meter.path(top.path), None, False, False))
        else:
            out.append(TrashEntry(f"{TRASH_DIR}/{top.name}", meter.path(top.path), None, False, False))
    return sorted(out, key=lambda t: -t.usage.bytes)


def _other(project: Path, pipeline: Pipeline, meter: _Meter) -> list[tuple[str, Usage]]:
    """Everything that is neither a job of the pipeline nor in Trash/: by top-level name, and in a job type's
    directory by folder (a job run outside the pipeline, an alias link)."""
    job_dirs: dict[str, set[str]] = {}
    for name in pipeline.processes:
        t, _, rest = name.partition("/")
        job_dirs.setdefault(t, set()).add(rest.rstrip("/"))
    out: list[tuple[str, Usage]] = []
    try:
        tops = sorted(os.scandir(project), key=lambda e: e.name)
    except OSError:
        return out
    for top in tops:
        if top.name == TRASH_DIR:
            continue
        if top.name in job_dirs and top.is_dir(follow_symlinks=False):
            try:
                inner = sorted(os.scandir(top.path), key=lambda e: e.name)
            except OSError:
                continue
            for e in inner:
                if e.name not in job_dirs[top.name]:
                    out.append((f"{top.name}/{e.name}", meter.path(e.path)))
        else:
            out.append((top.name, meter.path(top.path)))
    return sorted(out, key=lambda kv: -kv[1].bytes)


# ---------------------------------------------------------------------------- emptying Trash/

class TrashError(RuntimeError):
    pass


def empty_trash(project_dir: str | os.PathLike, entries: list[str]) -> list[str]:
    """Delete these entries of Trash/ ("Trash/Class2D/job006", as ``scan()`` lists them) for good. Only paths
    inside the project's Trash/ are taken (a link there is removed, not what it points to). Returns what
    could not be removed, with why."""
    project = Path(project_dir).resolve()
    root = project / TRASH_DIR
    errors: list[str] = []
    for rel in entries:
        p = project / rel
        parent = Path(os.path.realpath(p.parent))
        if (parent != root and root not in parent.parents) or p.name in ("", ".", ".."):
            raise TrashError(f"not in {TRASH_DIR}/: {rel}")
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif os.path.lexists(p):
                p.unlink()
        except OSError as e:
            errors.append(f"{rel}: {e.strerror or e}")
    for d in sorted({(project / rel).parent for rel in entries}, reverse=True):  # Trash/<Type>/ left empty
        if d != root and root in d.parents:
            try:
                d.rmdir()
            except OSError:
                pass
    return errors


def human(n: int | None) -> str:
    """1536 -> "1.5 KB" (powers of 1024, as du -h)."""
    if n is None:
        return ""
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} PB"
