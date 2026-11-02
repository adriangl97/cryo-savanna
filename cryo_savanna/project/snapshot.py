# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only view of a project at one moment, and a loader that refreshes it cheaply.

The loader never writes to the project. Each refresh costs one stat of the pipeline file plus a
few stats per running job, so it can poll every couple of seconds on network filesystems.
Finished jobs are only looked at again when the pipeline file changes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.jobdefs.cli_tools import EXTERNAL, external_program
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lineage import primary_parents
from cryo_savanna.project.main_inputs import MainInputs
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.relion_consts import (
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    job_type_for_label,
)

Stamp = tuple[int, int] | None  # (mtime_ns, size), None when the file is missing

# Checked in this order, like PipeLine::checkProcessCompletion
EXIT_MARKERS = ((RELION_JOB_EXIT_SUCCESS, STATUS_SUCCEEDED), (RELION_JOB_EXIT_FAILURE, STATUS_FAILED),
                (RELION_JOB_EXIT_ABORTED, STATUS_ABORTED))


def stamp(path: str | os.PathLike) -> Stamp:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


@dataclass
class JobInfo:
    process: Process
    primary_parent: str | None  # job drawn above this one in the tree
    other_parents: list[str]  # producers of the other inputs (references, masks, ...)
    children: list[str] = field(default_factory=list)  # jobs whose primary parent is this one
    status: str = ""  # effective status, see ProjectLoader._effective_status
    pending_update: bool = False  # finished on disk, pipeline still says Running
    modified: float | None = None  # latest activity in the job directory (epoch seconds)
    program: str = ""  # External jobs: the program they run (shown with the type, found by the filter)

    @property
    def name(self) -> str:
        return self.process.name


@dataclass
class ProjectSnapshot:
    project_dir: Path
    pipeline: Pipeline
    jobs: dict[str, JobInfo]  # pipeline order
    roots: list[str]
    pipeline_stamp: Stamp
    loaded_at: float

    def structure_key(self) -> tuple:
        """Changes when jobs, their parents, their names or what they run change, not when only statuses change."""
        return tuple((j.name, j.process.alias, j.primary_parent, tuple(j.other_parents), j.program)
                     for j in self.jobs.values())

    def type_dir_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for j in self.jobs.values():
            out[j.process.type_dir] = out.get(j.process.type_dir, 0) + 1
        return out


class ProjectLoader:
    """Builds ProjectSnapshots for one project. Not thread-safe: use from one thread at a time."""

    def __init__(self, project_dir: str | os.PathLike, defs: JobDefs | None = None, pipeline_name: str = "default",
                 main_inputs: MainInputs | None = None):
        """``main_inputs``: the main inputs chosen in cryo-savanna (project/main_inputs.py); a change to
        them is picked up by the next refresh, like a change of the pipeline."""
        self.project_dir = Path(project_dir).resolve()
        self.main_inputs = main_inputs
        self._main_stamp = None
        self.defs = defs or JobDefs.load()
        self.pipeline_path = self.project_dir / f"{pipeline_name}_pipeline.star"
        self.pipeline_name = pipeline_name
        self._jobstars: dict[Path, tuple[Stamp, JobStar]] = {}
        self._snapshot: ProjectSnapshot | None = None
        self._live: dict[str, tuple] = {}  # running job -> stamps of its markers and log

    @property
    def snapshot(self) -> ProjectSnapshot | None:
        return self._snapshot

    def read_jobstar(self, path: Path) -> JobStar:
        st = stamp(path)
        hit = self._jobstars.get(path)
        if hit is not None and hit[0] == st:
            return hit[1]
        js = JobStar.read(path)
        self._jobstars[path] = (st, js)
        return js

    def is_project(self) -> bool:
        return self.pipeline_path.is_file()

    def refresh(self, force: bool = False) -> ProjectSnapshot | None:
        """Return a new snapshot if anything changed since the last one, else None."""
        st = stamp(self.pipeline_path)
        old = self._snapshot
        main = self.main_inputs.stamp() if self.main_inputs is not None else None
        if force or old is None or st != old.pipeline_stamp or main != self._main_stamp:
            self._snapshot = self._load(st)
            self._main_stamp = main  # after the load: a failed one is tried again
            return self._snapshot
        live = self._live_stamps(old)
        if live != self._live:
            self._snapshot = self._rebuild(old.pipeline, old.pipeline_stamp, old)
            return self._snapshot
        return None

    # ------------------------------------------------------------------ internals
    def _load(self, st: Stamp) -> ProjectSnapshot:
        pipeline = Pipeline.read(self.project_dir, self.pipeline_name)
        # The stamp taken before reading may be older than what was read; take it again so a
        # write that raced with this read is picked up by the next refresh, not missed.
        return self._rebuild(pipeline, st if st == stamp(self.pipeline_path) else None, self._snapshot)

    def _rebuild(self, pipeline: Pipeline, st: Stamp, old: ProjectSnapshot | None) -> ProjectSnapshot:
        if old is not None and old.pipeline is pipeline:
            parents = {n: j.primary_parent for n, j in old.jobs.items()}  # only live state changed
        else:
            parents = primary_parents(pipeline, self.project_dir, self.defs, self.read_jobstar,
                                      self.main_inputs.choices() if self.main_inputs is not None else None)
        jobs: dict[str, JobInfo] = {}
        for name, pr in pipeline.processes.items():
            primary = parents[name]
            others = [x for x in pipeline.parents(name) if x not in (primary, name)]
            jobs[name] = JobInfo(pr, primary, others, program=self._program(pr))
        roots = []
        for name, info in jobs.items():
            if info.primary_parent is None or info.primary_parent not in jobs:
                roots.append(name)
            else:
                jobs[info.primary_parent].children.append(name)

        for name, info in jobs.items():
            prev = old.jobs.get(name) if old is not None else None
            if prev is not None and prev.process.status == info.process.status and info.process.status != STATUS_RUNNING:
                info.status, info.pending_update, info.modified = prev.status, prev.pending_update, prev.modified
            else:
                info.status, info.pending_update = self._effective_status(info.process)
                info.modified = self._modified(name)
        snap = ProjectSnapshot(self.project_dir, pipeline, jobs, roots, st, time.time())
        self._live = self._live_stamps(snap)
        return snap

    def _program(self, pr: Process) -> str:
        jt = job_type_for_label(pr.type_label)
        if jt is None or jt.label != EXTERNAL:
            return ""
        try:  # cached by the file's stamp: cheap on every refresh, and follows continue / overwrite
            return external_program(self.read_jobstar(self.project_dir / pr.name / "job.star").options.get("fn_exe", ""))
        except (OSError, ValueError, KeyError):
            return ""

    def _effective_status(self, pr: Process) -> tuple[str, bool]:
        """Status as RELION's GUI would show it after its next completion check (without writing)."""
        if pr.status != STATUS_RUNNING:
            return pr.status, False
        jobdir = self.project_dir / pr.name
        for marker, status in EXIT_MARKERS:
            if (jobdir / marker).exists():
                return status, True
        return pr.status, False

    def _modified(self, name: str) -> float | None:
        jobdir = self.project_dir / name
        times = [st[0] for st in (stamp(jobdir), stamp(jobdir / "run.out"), stamp(jobdir / "run.err")) if st]
        return max(times) / 1e9 if times else None

    def _live_stamps(self, snap: ProjectSnapshot) -> dict[str, tuple]:
        out = {}
        for name, info in snap.jobs.items():
            if info.process.status == STATUS_RUNNING:
                jobdir = self.project_dir / name
                out[name] = (stamp(jobdir), stamp(jobdir / "run.out"), stamp(jobdir / "run.err"),
                             *(stamp(jobdir / m) for m, _ in EXIT_MARKERS))
        return out


def job_type_names(defs: JobDefs) -> dict[str, str]:
    """Job type label -> name used in RELION's job menus ("relion.class2d" -> "2D classification")."""
    out: dict[str, str] = {}
    for tomo in (False, True):
        for menu_label, variant in defs.menu(tomo):
            out.setdefault(variant.label, menu_label)
    return out
