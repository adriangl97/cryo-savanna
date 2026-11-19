# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Force abort: the processes of a job that runs on this machine (not in a queue), found and killed.

RELION's abort (RELION_JOB_ABORT_NOW) waits for the program's next check, which can take long (a big
iteration) or never come (a stuck program, an External script that does not look). Force abort ends it now.
The job's processes are, with everything they started (mpirun's ranks, an External job's programs):

- under a relion_pipeliner that cryo-savanna started on this host (a run record whose process is alive):
  the processes whose command line names the job's directory, i.e. the shell RELION runs the job's command
  in (``cmd >> Job/jobNNN/run.out 2>> Job/jobNNN/run.err``) and what it started;
- anywhere on this host (RELION's GUI and pipeliner run a local job in the background, so it is not under
  them), this user's processes whose command line gives the job's directory as where to write: ``--o``,
  ``--odir``, ``--part_dir`` (what RELION gives its programs and External jobs) or a ``>>``/``>`` redirection
  (the shell of RELION's ``cmd >> Job/jobNNN/run.out``). Where the working directory of a process can be read
  (Linux), it has to be the project's.

RELION_JOB_EXIT_ABORTED is written first, so the pipeliner or the GUI that waits for the job sees it aborted
(and a run stops, as for an abort) however soon it looks; the pipeliner itself is not killed. Then the
processes get SIGTERM, and SIGKILL after a few seconds.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cryo_savanna.relion_consts import (
    RELION_JOB_ABORT_NOW,
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
)

EXITS = (RELION_JOB_EXIT_SUCCESS, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_ABORTED)
TERM_WAIT_S = 5.0  # between SIGTERM and SIGKILL


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    uid: int
    state: str  # ps's STAT: "Z..." is a zombie (ended, not yet reaped)
    args: str

    @property
    def zombie(self) -> bool:
        return self.state.startswith("Z")


class ForceAbortError(RuntimeError):
    pass


def process_table() -> dict[int, Proc]:
    """Every process on this host (ps, as on Linux and macOS)."""
    try:
        r = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,uid=,stat=,args="], capture_output=True, text=True,
                           timeout=20, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        raise ForceAbortError(f"Could not list the processes of this machine (ps): {e}") from e
    out: dict[int, Proc] = {}
    for line in r.stdout.splitlines():
        w = line.split(None, 4)
        if len(w) < 4:
            continue
        try:
            p = Proc(int(w[0]), int(w[1]), int(w[2]), w[3], w[4] if len(w) > 4 else "")
        except ValueError:
            continue
        out[p.pid] = p
    return out


def _names_dir(job: str) -> re.Pattern:
    """The job's directory anywhere in a command line, as a path of its own (not .../Other/Class3D/job009/)."""
    return re.compile(r"(?:^|[\s'\"=<>|;])(?:\./)?" + re.escape(job.rstrip("/") + "/"))


def _output_is(job: str) -> re.Pattern:
    """``--o <job directory>...`` (or --odir, --part_dir, a redirection) in a command line."""
    return re.compile(r"(?:^|\s)(?:--o|--odir|--part_dir|\d?>>?)\s*['\"]?(?:\./)?" + re.escape(job.rstrip("/") + "/"))


def _cwd(pid: int) -> Path | None:
    """The working directory of a process: /proc (Linux), else lsof (macOS, which has no /proc); None when it
    cannot be read. Without it, a process of another project that writes to a job directory of the same name
    (--o Class3D/job009/) would be taken for one of this job's."""
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except OSError:
        if os.path.isdir("/proc/self"):
            return None  # Linux: gone, or not ours to read
    try:
        r = subprocess.run(["lsof", "-a", "-n", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True,
                           timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in r.stdout.splitlines():  # -F: one field per line, "n<path>" for the name
        if line.startswith("n") and len(line) > 1:
            return Path(line[1:]).resolve()
    return None


def _subtrees(roots: list[int], table: dict[int, Proc]) -> list[int]:
    """``roots`` and everything under them, parents first."""
    children: dict[int, list[int]] = {}
    for p in table.values():
        children.setdefault(p.ppid, []).append(p.pid)
    seen: dict[int, None] = {}
    todo = deque(roots)
    while todo:
        pid = todo.popleft()
        if pid in seen:
            continue
        seen[pid] = None
        todo.extend(sorted(children.get(pid, [])))
    return list(seen)


def _topmost(pids: set[int], table: dict[int, Proc]) -> list[int]:
    """The ones whose parent is not one of them."""
    return sorted(pid for pid in pids if table[pid].ppid not in pids)


def job_processes(project_dir: str | os.PathLike, job: str, pipeliner_pids: Iterable[int] = (),
                  table: dict[int, Proc] | None = None) -> list[Proc]:
    """The processes of ``job`` on this host, parents first (see the module's docstring).

    ``pipeliner_pids``: relion_pipeliners started here that run the job (Pipeliner.runs_of)."""
    table = process_table() if table is None else table
    project = Path(project_dir).resolve()
    me = os.getpid()
    pipeliners = {pid for pid in pipeliner_pids if pid in table}
    names, output = _names_dir(job), _output_is(job)
    found: set[int] = set()
    for root in pipeliners:  # under a pipeliner of ours
        found |= {pid for pid in _subtrees([root], table)[1:] if names.search(table[pid].args)}
    uid = os.getuid()
    for p in table.values():  # elsewhere: --o <job>, in this project
        if p.uid == uid and output.search(p.args) and p.pid not in found:
            cwd = _cwd(p.pid)
            if cwd is None or cwd == project:
                found.add(p.pid)
    pids = _subtrees(_topmost(found, table), table) if found else []
    return [table[pid] for pid in pids if pid != me and pid not in pipeliners and not table[pid].zombie]


def _alive(procs: list[Proc]) -> list[Proc]:
    """Those still running: the same process (its command line, so not a new one given its number), not a zombie."""
    table = process_table()
    return [p for p in procs if p.pid in table and table[p.pid].args == p.args and not table[p.pid].zombie]


def _signal(procs: list[Proc], sig: int) -> list[Proc]:
    """Send ``sig``; the processes it could not be sent to (another user's)."""
    denied = []
    for p in procs:
        try:
            os.kill(p.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            denied.append(p)
    return denied


def kill(project_dir: str | os.PathLike, job: str, procs: list[Proc], term_wait_s: float | None = None) -> None:
    """Mark the job as aborted and end ``procs`` (from job_processes, perhaps a while ago: only those still the
    same are signalled). Raises ForceAbortError naming the processes that are still running (the job is then
    not marked)."""
    d = Path(project_dir) / job
    marked = d.is_dir() and not any((d / f).exists() for f in EXITS)  # not if it ended meanwhile
    if marked:
        (d / RELION_JOB_EXIT_ABORTED).touch()
    procs = _alive(procs)
    denied = _signal(procs, signal.SIGTERM)
    deadline = time.monotonic() + (TERM_WAIT_S if term_wait_s is None else term_wait_s)
    left = _alive(procs)
    while left and time.monotonic() < deadline:
        time.sleep(0.2)
        left = _alive(left)
    if left:
        denied += _signal(left, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while left and time.monotonic() < deadline:
            time.sleep(0.1)
            left = _alive(left)
    if left:
        if marked:
            (d / RELION_JOB_EXIT_ABORTED).unlink(missing_ok=True)
        why = " (not yours to stop)" if denied else ""
        raise ForceAbortError(f"Still running{why}:\n" + "\n".join(f"{p.pid}  {p.args}" for p in left))
    (d / RELION_JOB_ABORT_NOW).unlink(missing_ok=True)  # nothing is left to read it; a continue would stop at once
    if marked:
        with open(d / "run.err", "a") as fh:  # where RELION's GUI and this window show why a job ended
            fh.write(f"\n ++++ cryo-savanna: force abort on {socket.gethostname()}, "
                     f"{time.ctime()}: {len(procs)} process(es) killed, "
                     f"{RELION_JOB_EXIT_ABORTED} written\n")
