# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Jobs that are out of date: finished jobs whose inputs changed after they ran (no Qt).

A finished job is out of date when

- an input it read was written again after the job finished (its file is newer than the job's
  RELION_JOB_EXIT_SUCCESS): the job that made it was continued, overwritten and run again, or a selection
  was changed and saved again;
- an input is gone while the job that makes it is not finished (overwritten: its old files are in Trash/, the
  new ones are not written yet);
- a job it takes inputs from is out of date itself.

Inputs nobody in the pipeline makes (imported files) and jobs without RELION's exit file (marked finished by
hand) are not looked at: nothing tells when they ran. Looked at once per change of the pipeline (every run,
continue or overwrite changes it): one stat of each job's exit file, and of the inputs of only the jobs whose own
exit file or whose input jobs' exit files changed since the last look.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.ops import decompose
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.relion_consts import RELION_JOB_EXIT_SUCCESS, STATUS_SUCCEEDED
from cryo_savanna.stamps import Stamp, stamp

SLACK_S = 2.0  # a file written this little after the job ended is the job's own last write (coarse clocks)


@dataclass(frozen=True)
class Stale:
    """Why a job is out of date: its inputs written again (``changed``), gone while the job making them runs
    again (``missing``), or a job it depends on out of date (``upstream``)."""
    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    upstream: str | None = None
    because_of: tuple[str, ...] = field(default=(), compare=False)  # the jobs whose outputs those are

    def text(self) -> str:
        if self.changed or self.missing:
            jobs = ", ".join(j.rstrip("/") for j in self.because_of)
            if self.changed and not self.missing:
                return f"its input from {jobs} was written again after it ran"
            return f"{jobs} is being run again (its input is gone)"
        return f"it depends on {self.upstream.rstrip('/') if self.upstream else 'a job'}, which is out of date"


class StalenessChecker:
    """``check(pipeline)`` for one project, remembering what it looked at (use from one thread at a time)."""

    def __init__(self, project_dir: str | os.PathLike):
        self.project_dir = Path(project_dir)
        self._direct: dict[str, tuple[tuple, Stale | None]] = {}  # job -> (what it was worked out from, result)

    def _exit_stamp(self, job: str) -> Stamp:
        return stamp(self.project_dir / job / RELION_JOB_EXIT_SUCCESS)

    def check(self, pipeline: Pipeline) -> dict[str, Stale]:
        """The out-of-date jobs of the pipeline (finished ones only), in pipeline order."""
        procs = pipeline.processes
        exits = {name: self._exit_stamp(name) for name in procs}
        inputs = {name: _input_jobs(pipeline, name) for name in procs}
        direct: dict[str, Stale | None] = {}
        for name, pr in procs.items():
            if pr.status != STATUS_SUCCEEDED or exits[name] is None:
                continue
            producers = inputs[name]
            key = (exits[name], tuple((n, p, procs[p].status, exits[p]) for n, p in producers))
            hit = self._direct.get(name)
            if hit is not None and hit[0] == key:
                direct[name] = hit[1]
                continue
            done_at = exits[name][0] / 1e9  # type: ignore[index]  # (not None: checked above)
            changed, missing, because = [], [], []
            for n, p in producers:
                st = stamp(self.project_dir / n)
                if st is None:
                    if procs[p].status != STATUS_SUCCEEDED:  # a finished job's missing output: cannot tell why
                        missing.append(n)
                        because.append(p)
                elif st[0] / 1e9 > done_at + SLACK_S:
                    changed.append(n)
                    because.append(p)
            result = Stale(tuple(changed), tuple(missing), None, tuple(dict.fromkeys(because))) \
                if changed or missing else None
            self._direct[name] = (key, result)
            direct[name] = result
        for gone in set(self._direct) - set(procs):
            del self._direct[gone]

        out: dict[str, Stale] = {n: s for n, s in direct.items() if s is not None}
        # then what depends on them, as far down as it goes (a pass more than the longest chain changes nothing)
        grew = True
        while grew:
            grew = False
            for name, pr in procs.items():
                if name in out or pr.status != STATUS_SUCCEEDED:  # (also one marked finished by hand)
                    continue
                up = next((p for _, p in inputs[name] if p in out), None)
                if up is not None:
                    out[name] = Stale(upstream=up)
                    grew = True
        return {n: out[n] for n in procs if n in out}


def _input_jobs(pipeline: Pipeline, job: str) -> list[tuple[str, str]]:
    """(input node, the job that makes it) of a job's inputs made by another job of the pipeline. A node with no
    producer in the pipeline but inside a job's directory is that job's (an External job declares its outputs only
    when it finishes, so a job added before that has the node with no producer)."""
    out = []
    for n in pipeline.processes[job].inputs:
        node = pipeline.nodes.get(n)
        producer = node.producer if node is not None else None
        if producer is None:
            parts = decompose(n)
            producer = parts[0] + parts[1] if parts else None
        if producer is not None and producer != job and producer in pipeline.processes:
            out.append((n, producer))
    return out


def dependency_order(pipeline: Pipeline, jobs: list[str]) -> list[str]:
    """``jobs``, each after those of them it takes inputs from, else in pipeline order: one relion_pipeliner runs
    them in this order and waits for each job's input files (a job before the one making its input would wait
    for ever). Jobs in a loop of inputs (not made by RELION) keep pipeline order."""
    chosen = set(jobs)
    needs = {j: {p for _, p in _input_jobs(pipeline, j) if p in chosen} for j in jobs if j in pipeline.processes}
    order = [n for n in pipeline.processes if n in needs]
    out: list[str] = []
    placed: set[str] = set()
    while order:
        ready = next((j for j in order if needs[j] <= placed), order[0])
        out.append(ready)
        placed.add(ready)
        order.remove(ready)
    return out + [j for j in jobs if j not in needs]


def independent_groups(pipeline: Pipeline, jobs: list[str]) -> list[list[str]]:
    """``jobs`` split into groups that do not take inputs from each other, each in dependency_order, in the order
    of their first job: one relion_pipeliner runs its jobs one after the other (each after the last one ended),
    so jobs of different groups can each have a run of their own, and start at once."""
    chosen = [j for j in jobs if j in pipeline.processes]
    group = {j: j for j in chosen}  # union-find over "takes an input from"

    def root(j: str) -> str:
        while group[j] != j:
            group[j] = group[group[j]]
            j = group[j]
        return j

    for j in chosen:
        for _, p in _input_jobs(pipeline, j):
            if p in group:
                group[root(j)] = root(p)
    members: dict[str, list[str]] = {}
    for j in chosen:
        members.setdefault(root(j), []).append(j)
    order = {n: i for i, n in enumerate(pipeline.processes)}
    out = [dependency_order(pipeline, m) for m in members.values()]
    out.sort(key=lambda g: min(order[j] for j in g))
    unknown = [j for j in jobs if j not in pipeline.processes]
    return out + ([unknown] if unknown else [])


def downstream(pipeline: Pipeline, job: str, among: dict[str, Stale]) -> list[str]:
    """``job`` (if it is in ``among``) and the jobs of ``among`` that depend on it, in pipeline order."""
    want = {job}
    grew = True
    while grew:
        grew = False
        for name in among:
            if name not in want and any(p in want for _, p in _input_jobs(pipeline, name)):
                want.add(name)
                grew = True
    return [n for n in pipeline.processes if n in want and n in among]
