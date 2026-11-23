# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The local GPU queue: a run of jobs that use GPUs of this machine waits until those GPUs are free (no Qt).

Without it, jobs started at once (Variations…, several Run now) all start on the same GPUs and run out of their
memory. With it, the relion_pipeliner of such a run is started through ``gpu_wait.py``, which waits for the
locks of the run's GPUs and then becomes that relion_pipeliner, keeping them until the run ends (see there).
The locks are files in a folder of this machine's /tmp that every user can write, so runs of everyone on the
machine (with cryo-savanna) wait for each other; RELION's own GUI takes no locks, but its jobs show in
nvidia-smi, which the waiting runs also look at (only RELION jobs there count, native or External: not any other
program on the GPU; see gpu_wait.py).

Which GPUs: those a job's options give it (RELION's "Which GPUs to use", pytom's and AreTomo3's GPU ids),
mapped through CUDA_VISIBLE_DEVICES; every GPU of the machine when they are left empty. Jobs sent to a queue,
and jobs that use no GPU, take none.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.relion_consts import job_type_for_label

WAITER = Path(__file__).resolve().parent.parent / "gpu_wait.py"
SHARED_DIR = "/tmp/cryo-savanna-gpu-locks"

_REFINE = {"relion.class2d", "relion.class3d", "relion.refine3d", "relion.initialmodel", "relion.multibody"}


def _yes(o: Mapping[str, str], key: str) -> bool:
    return (o.get(key) or "").strip() == "Yes"


def parse_ids(text: str) -> list[int]:
    """GPU ids as RELION takes them ("0:1,2:3", "0 1"), each once, in order."""
    out: list[int] = []
    for tok in re.split(r"[\s,:;]+", text or ""):
        if tok.isdigit() and int(tok) not in out:
            out.append(int(tok))
    return out


def job_gpu_ids(js: JobStar, queued: bool = False) -> list[int] | None:
    """The GPUs (as the job numbers them) a job uses on this machine: [] for every GPU it sees (the field left
    empty), None when it uses none here (no GPU, or sent to a queue). ``queued``: also for a job sent to a queue
    (the GPUs it uses on the node it gets)."""
    o = js.options
    if _yes(o, "do_queue") and not queued:
        return None
    jt = job_type_for_label(js.type_label)
    t = jt.label if jt is not None else js.type_label
    if t == "relion.external":
        from cryo_savanna.jobdefs import aretomo_jobs, external_jobs, membrain_jobs, pytom_jobs
        found = external_jobs.kind_of(js)
        if found is None:
            return None
        family, kind = found
        own, _ = pytom_jobs.decode(o.get("other_args", ""))
        if (family.name, kind) in (("pytom", pytom_jobs.MATCH), ("aretomo", aretomo_jobs.ALIGN),
                                   ("membrain", membrain_jobs.SEGMENT)):
            return parse_ids(own.get("gpu_ids", ""))
        return None
    if t in _REFINE:
        uses = _yes(o, "use_gpu")
    elif t == "relion.autopick":
        uses = _yes(o, "do_topaz") or (_yes(o, "use_gpu") and not _yes(o, "do_log"))
    elif t == "relion.motioncorr":
        uses = not _yes(o, "do_own_motioncor")  # MotionCor2
    elif t == "relion.aligntiltseries":
        uses = _yes(o, "do_aretomo2")
    elif t == "relion.denoisetomo":
        uses = _yes(o, "do_cryocare_train") or _yes(o, "do_cryocare_predict")
    elif "gpu_id" in o:  # DynaMight, ModelAngelo: one GPU
        return parse_ids(o.get("gpu_id", "")) or [0]
    else:
        return None
    return parse_ids(o.get("gpu_ids", "")) if uses else None


def _visible(environ: Mapping[str, str]) -> list[str] | None:
    """CUDA_VISIBLE_DEVICES as a list (None: not set: every GPU, by its number)."""
    vis = (environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not vis:
        return None
    if vis in ("NoDevFiles", "-1"):
        return []
    return [v.strip() for v in vis.split(",") if v.strip()]


def count_gpus() -> int:
    """GPUs nvidia-smi lists on this machine (0 when there is none, or no nvidia-smi)."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, universal_newlines=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.startswith("GPU ")) if r.returncode == 0 else 0


class GpuQueue:
    """Which GPUs a run takes, and the command that waits for them."""

    def __init__(self, lock_dir: str | os.PathLike | None = None, count: Callable[[], int] = count_gpus,
                 python: str = sys.executable):
        self._lock_dir = Path(lock_dir) if lock_dir is not None else None
        self._count = count
        self._counted: int | None = None
        self.python = python

    def lock_dir(self) -> Path:
        """The shared folder in /tmp (made writable by everyone, sticky as /tmp), else one in this user's cache."""
        if self._lock_dir is not None:
            return self._lock_dir
        shared = Path(SHARED_DIR)
        try:
            shared.mkdir(exist_ok=True)
            if shared.stat().st_uid == os.getuid():
                shared.chmod(0o1777)
            if os.access(shared, os.W_OK | os.X_OK):
                self._lock_dir = shared
                return shared
        except OSError:
            pass
        own = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cryo-savanna" / \
            f"gpu-locks-{socket.gethostname()}"
        own.mkdir(parents=True, exist_ok=True)
        self._lock_dir = own
        return own

    def _all(self, environ: Mapping[str, str]) -> list[str]:
        vis = _visible(environ)
        if vis is not None:
            return vis
        if self._counted is None:
            self._counted = self._count()
        return [str(i) for i in range(self._counted or 0)]

    def gpus_of(self, jobstars: Iterable[JobStar], environ: Mapping[str, str]) -> list[str]:
        """The GPUs of this machine (CUDA_VISIBLE_DEVICES' names, else numbers) the jobs take, together."""
        out: list[str] = []
        for js in jobstars:
            ids = job_gpu_ids(js)
            if ids is None:
                continue
            every = self._all(environ)
            vis = _visible(environ)
            names = every if not ids else [vis[i] if vis is not None and i < len(vis) else str(i) for i in ids]
            out += [g for g in names if g not in out]
        return out

    def command(self, cmd: list[str], gpus: list[str], label: str) -> list[str]:
        """``cmd``, started once ``gpus`` are free (gpu_wait.py). Isolated (-I): RELION's environment, which the run
        gets, may set PYTHONPATH or PYTHONHOME for its own Python tools; the relion_pipeliner it becomes still gets
        all of it."""
        return [self.python, "-I", str(WAITER), "--locks", str(self.lock_dir()), "--gpus", ",".join(gpus),
                "--label", label, "--", *cmd]


def is_waiting(args: list[str] | None) -> bool:
    """Whether a run's process (its command line) is still waiting for its GPUs (not yet its relion_pipeliner)."""
    return args is not None and any(a.endswith("gpu_wait.py") for a in args)
