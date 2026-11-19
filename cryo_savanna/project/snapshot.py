# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only view of a project at one moment, and a loader that refreshes it cheaply.

The loader never writes to the project. Each refresh costs one stat of the pipeline file plus a
few stats per running job, so it can poll every couple of seconds on network filesystems.
Finished jobs are only looked at again when the pipeline file changes. For a moment after the
pipeline file changed, it is also read again (RACY_S).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.jobdefs.external_jobs import EXTERNAL, program_title
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project import references
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lineage import job_holding, primary_parents
from cryo_savanna.project.main_inputs import MainInputs
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.project.stars import Stars
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
from cryo_savanna.stamps import Stamp, stamp  # noqa: F401 - also imported from here


# Checked in this order, like PipeLine::checkProcessCompletion
EXIT_MARKERS = ((RELION_JOB_EXIT_SUCCESS, STATUS_SUCCEEDED), (RELION_JOB_EXIT_FAILURE, STATUS_FAILED),
                (RELION_JOB_EXIT_ABORTED, STATUS_ABORTED))

# A file system's clock can be coarse (1 s on Lustre, ext3, HFS+ and many NFS servers): a second write of the
# pipeline in the tick of the one just read, of the same size (Running -> Aborted, Scheduled -> Succeeded: one
# job started or stopped right after another), leaves its stamp as it was, and would never be seen. For this
# long after a new stamp was seen, the file's content is compared too (once more after it, then trusted).
RACY_S = 2.0


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


@dataclass
class JobInfo:
    process: Process
    primary_parent: str | None  # job drawn above this one in the tree
    other_parents: list[str]  # producers of the other inputs (references, masks, ...; each of several references)
    children: list[str] = field(default_factory=list)  # jobs whose primary parent is this one
    status: str = ""  # effective status, see ProjectLoader._effective_status
    pending_update: bool = False  # finished on disk, pipeline still says Running
    modified: float | None = None  # latest activity in the job directory (epoch seconds)
    program: str = ""  # External jobs: the program they run (shown with the type, found by the filter)
    starred: bool = False  # starred in cryo-savanna (project/stars.py)
    reference_maps: dict[str, str] = field(default_factory=dict)  # several references: map -> its job (other_parents)

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
        """Changes when jobs, their parents, their names or what they run change, not when only statuses change.

        Worked out once per snapshot (the list, the tree and the job actions each ask at every refresh); a
        snapshot's jobs are not changed once it is made."""
        key = self.__dict__.get("_structure_key")
        if key is None:
            key = self.__dict__["_structure_key"] = tuple(
                (j.name, j.process.alias, j.primary_parent, tuple(j.other_parents), j.program)
                for j in self.jobs.values())
        return key

    def type_dir_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for j in self.jobs.values():
            out[j.process.type_dir] = out.get(j.process.type_dir, 0) + 1
        return out


class ProjectLoader:
    """Builds ProjectSnapshots for one project. Not thread-safe: use from one thread at a time."""

    def __init__(self, project_dir: str | os.PathLike, defs: JobDefs | None = None, pipeline_name: str = "default",
                 main_inputs: MainInputs | None = None, stars: Stars | None = None):
        """``main_inputs``: the main inputs chosen in cryo-savanna (project/main_inputs.py); a change to
        them is picked up by the next refresh, like a change of the pipeline. ``stars``: the starred
        jobs (project/stars.py), also picked up by the next refresh."""
        self.project_dir = Path(project_dir).resolve()
        self.main_inputs = main_inputs
        self._main_stamp: tuple | None = None
        self.stars = stars
        self._stars_stamp: tuple | None = None
        self.defs = defs or JobDefs.load()
        self.pipeline_path = self.project_dir / f"{pipeline_name}_pipeline.star"
        self.pipeline_name = pipeline_name
        self._jobstars: dict[Path, tuple[Stamp, JobStar]] = {}
        self._ref_lists: dict[str, tuple[Stamp, list[str]]] = {}  # list of references -> its maps (by its stamp)
        self._snapshot: ProjectSnapshot | None = None
        self._live: dict[str, tuple] = {}  # running job -> stamps of its markers and log
        self._digest: bytes = b""  # of the pipeline text last read
        self._settle_at: float | None = None  # until then (time.monotonic) an unchanged stamp is not trusted

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
        seen = time.monotonic()
        st = stamp(self.pipeline_path)
        old = self._snapshot
        main = self.main_inputs.stamp() if self.main_inputs is not None else None
        if force or old is None or st != old.pipeline_stamp or main != self._main_stamp or self._rewritten():
            self._snapshot = self._load(st, seen)
            self._main_stamp = main  # after the load: a failed one is tried again
            return self._snapshot
        live = self._live_stamps(old.pipeline)
        starred = self.stars.stamp() if self.stars is not None else None
        if live != self._live or starred != self._stars_stamp:
            self._snapshot = self._rebuild(old.pipeline, old.pipeline_stamp, old, live)
            return self._snapshot
        return None

    # ------------------------------------------------------------------ internals
    def _load(self, st: Stamp, seen: float) -> ProjectSnapshot:
        """``seen``: when ``st`` was taken (time.monotonic)."""
        pipeline, text = Pipeline.read_with_text(self.project_dir, self.pipeline_name)
        self._digest, self._settle_at = _digest(text), seen + RACY_S
        # The stamp taken before reading may be older than what was read; take it again so a
        # write that raced with this read is picked up by the next refresh, not missed.
        return self._rebuild(pipeline, st if st == stamp(self.pipeline_path) else None, self._snapshot)

    def _rewritten(self) -> bool:
        """Whether the pipeline file was written again without a new stamp (RACY_S): its content is compared
        while the stamp is new, and once more after RACY_S (a write in the same tick happened by then)."""
        if self._settle_at is None:
            return False
        now = time.monotonic()
        try:
            text = self.pipeline_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True  # gone or unreadable meanwhile: a full read tells
        if now >= self._settle_at:
            self._settle_at = None
        return _digest(text) != self._digest

    def _rebuild(self, pipeline: Pipeline, st: Stamp, old: ProjectSnapshot | None,
                 live: dict[str, tuple] | None = None) -> ProjectSnapshot:
        """``live``: _live_stamps(pipeline), when the caller took them just now."""
        # Before the exit files are looked at: one written while this runs changes them, so the next refresh
        # looks again. Taken after, it would be in them unseen, and the job would stay Running until the
        # pipeline changes (never, when no relion_pipeliner waits for the job).
        if live is None:
            live = self._live_stamps(pipeline)
        if old is not None and old.pipeline is pipeline:
            parents = {n: j.primary_parent for n, j in old.jobs.items()}  # only live state changed
        else:
            parents = primary_parents(pipeline, self.project_dir, self.defs, self.read_jobstar,
                                      self.main_inputs.choices() if self.main_inputs is not None else None)
        refs = self._reference_jobs(pipeline)
        if self.stars is not None:
            self._stars_stamp = self.stars.stamp()
        starred = self.stars.starred() if self.stars is not None else set()
        jobs: dict[str, JobInfo] = {}
        for name, pr in pipeline.processes.items():
            primary = parents[name]
            maps = refs.get(name, {})
            others = [x for x in dict.fromkeys(pipeline.parents(name) + list(maps.values())) if x not in (primary, name)]
            jobs[name] = JobInfo(pr, primary, others, program=self._program(pr), starred=name in starred,
                                 reference_maps=maps)
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
        self._live = live
        return snap

    def _reference_jobs(self, pipeline: Pipeline) -> dict[str, dict[str, str]]:
        """Jobs whose maps a 3D classification starts from, when it has several (their list, references.star,
        is its input in the pipeline: project/references.py): job -> {map: the job it comes from}."""
        out: dict[str, dict[str, str]] = {}
        for name, pr in pipeline.processes.items():
            for node in pr.inputs:
                if not node.endswith("/" + references.FILE):
                    continue
                for m in self._reference_list(node):
                    n = pipeline.nodes.get(m)
                    job = n.producer if n is not None and n.producer else job_holding(pipeline, m)
                    if job is not None and job != name:
                        out.setdefault(name, {})[m] = job
        return out

    def _reference_list(self, path: str) -> list[str]:
        st = stamp(self.project_dir / path)  # read again when written again (a job overwritten)
        hit = self._ref_lists.get(path)
        if hit is None or hit[0] != st:
            hit = self._ref_lists[path] = (st, (references.read(self.project_dir, path) or []) if st else [])
        return hit[1]

    def _program(self, pr: Process) -> str:
        jt = job_type_for_label(pr.type_label)
        if jt is None or jt.label != EXTERNAL:
            return ""
        try:  # cached by the file's stamp: cheap on every refresh, and follows continue / overwrite
            return program_title(self.read_jobstar(self.project_dir / pr.name / "job.star").options.get("fn_exe", ""))
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

    def _live_stamps(self, pipeline: Pipeline) -> dict[str, tuple]:
        """Stamps of what tells that a job the pipeline has as Running ended (its exit files) or did something."""
        out = {}
        for name, pr in pipeline.processes.items():
            if pr.status == STATUS_RUNNING:
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
