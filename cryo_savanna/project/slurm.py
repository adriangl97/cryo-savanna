# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""SLURM: the queued, running and finished SLURM jobs of a RELION project's jobs.

cryo-savanna does not submit anything itself: for a job with "Submit to queue?" relion_pipeliner
writes run_submit.script from the queue template and runs "<qsub command> run_submit.script &"
in the project directory. This module finds those SLURM jobs again, in this order:

1. the job name: RELION fills the template's XXXnameXXX with the job directory ("Class2D/job012/"),
   which the usual templates pass to --job-name (only SLURM jobs started in this project count);
2. the output file: XXXoutfileXXX is "<job dir>run.out", and SLURM reports where it writes stdout;
3. the "Submitted batch job N" lines that sbatch printed into cryo-savanna's run logs
   (.cryo-savanna/runs/*.log), matched in order to the queued jobs of that run.

For the history of a job none of these found (it ran while no cryo-savanna looked at the queue),
sacct is asked by job name; and where the template gives no --job-name (SLURM then names every
job "run_submit.script"), by submit line: RELION submits "<qsub command> <job dir>run_submit.script",
so sacct's SubmitLine names the job whatever the template does (only newer SLURM releases have
that field; with an older one this last step is skipped).

Every SLURM job found for a RELION job is remembered in .cryo-savanna/slurm.json, so the queue
history (sacct) and the check for jobs that ended in SLURM without telling RELION (time limit,
out of memory, node failure, scancel) still work after a restart.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.shared_json import update_json
from cryo_savanna.relion_consts import (
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
    STATUS_RUNNING,
)

GUI_DIR = ".cryo-savanna"
KNOWN_FILE = "slurm.json"
SEP = "\t"
# squeue -o: id, state, reason, time used, time limit, nodes, partition, name, working directory
SQUEUE_FORMAT = SEP.join(("%i", "%T", "%r", "%M", "%l", "%N", "%P", "%j", "%Z"))
SACCT_FIELDS = ("JobID", "JobName", "State", "Elapsed", "Start", "End", "NodeList", "ExitCode", "MaxRSS",
                "AllocCPUS", "Partition", "WorkDir")
ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD",
          "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED", "RESV_DEL_HOLD", "SPECIAL_EXIT"}
ENDED_BADLY = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "BOOT_FAIL", "DEADLINE",
               "PREEMPTED", "REVOKED"}
LOST_GRACE_S = 120  # RELION's exit file can come a little after SLURM says the job ended
LOST_RECHECK_S = 120  # sacct (slurmdbd) is asked again about a job that may be lost at most this often
HISTORY_DAYS = 120
_SUBMITTED = re.compile(r"Submitted batch job (\d+)")


class SlurmError(RuntimeError):
    pass


def _first_start(note: Path) -> float | None:
    """When a job first ran: the first " ++++ Executing new job on <ctime>" of its note.txt."""
    from cryo_savanna.project.summary import parse_ctime
    try:
        text = note.read_text(errors="replace")
    except OSError:
        return None
    marker = "Executing new job on"
    i = text.find(marker)
    return parse_ctime(text[i + len(marker):].split("\n", 1)[0]) if i >= 0 else None


def _id_order(job_id: str) -> tuple[int, str]:
    """Sort key for SLURM ids, also array ids ("123_4", "123_[1-5]")."""
    head = job_id.split("_", 1)[0]
    return (int(head) if head.isdigit() else 0, job_id)


# ---------------------------------------------------------------------------- records

@dataclass(frozen=True)
class QueueJob:
    """A job in squeue."""
    id: str
    state: str
    reason: str = ""
    elapsed: str = ""
    limit: str = ""
    nodes: str = ""
    partition: str = ""
    name: str = ""
    workdir: str = ""

    @property
    def pending(self) -> bool:
        return self.state == "PENDING"

    def short(self) -> str:
        """For the tree and the cards: "queued (Priority)", "running 1:02:03 on node12"."""
        if self.pending:
            return f"queued ({self.reason})" if self.reason and self.reason != "None" else "queued"
        if self.state == "RUNNING":
            return f"running {self.elapsed}" + (f" on {self.nodes}" if self.nodes else "")
        return self.state.lower().replace("_", " ")

    def describe(self) -> str:
        bits = [f"SLURM job {self.id}", self.state]
        if self.pending and self.reason and self.reason != "None":
            bits.append(f"waiting for: {self.reason}")
        if self.nodes:
            bits.append(f"on {self.nodes}")
        if self.elapsed and not self.pending:
            bits.append(f"{self.elapsed}" + (f" of {self.limit}" if self.limit else ""))
        elif self.limit:
            bits.append(f"time limit {self.limit}")
        if self.partition:
            bits.append(f"partition {self.partition}")
        return " · ".join(bits)


@dataclass(frozen=True)
class AcctJob:
    """A job in sacct (one allocation; MaxRSS is the largest of its steps)."""
    id: str
    name: str = ""
    state: str = ""
    elapsed: str = ""
    start: str = ""
    end: str = ""
    nodes: str = ""
    exit_code: str = ""
    max_rss: str = ""
    cpus: str = ""
    partition: str = ""
    workdir: str = ""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def ended_badly(self) -> bool:
        return self.state in ENDED_BADLY

    def end_time(self) -> float | None:
        try:
            return time.mktime(time.strptime(self.end, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return None

    def short(self) -> str:
        return f"{self.state.lower().replace('_', ' ')} in SLURM ({self.id})"


def parse_squeue(text: str) -> list[QueueJob]:
    out = []
    for line in text.splitlines():
        parts = line.split(SEP)
        if len(parts) < 9 or not parts[0].strip():
            continue
        out.append(QueueJob(*(p.strip() for p in parts[:8]), workdir=SEP.join(parts[8:]).strip()))
    return out


def _rss_bytes(rss: str) -> float:
    m = re.match(r"^([\d.]+)([KMGTP]?)$", rss.strip())
    if not m:
        return 0.0
    unit = m.group(2)
    return float(m.group(1)) * (1024 ** ("KMGTP".index(unit) + 1) if unit else 1)


def parse_sacct(text: str) -> list[AcctJob]:
    """sacct -P -n -o SACCT_FIELDS (with steps): one AcctJob per allocation."""
    jobs: dict[str, dict] = {}
    order: list[str] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < len(SACCT_FIELDS) or not parts[0]:
            continue
        rec = dict(zip(SACCT_FIELDS, parts[:len(SACCT_FIELDS) - 1] + ["|".join(parts[len(SACCT_FIELDS) - 1:])]))
        job_id, _, step = rec["JobID"].partition(".")
        if not step:
            jobs[job_id] = rec
            order.append(job_id)
        elif job_id in jobs and _rss_bytes(rec["MaxRSS"]) > _rss_bytes(jobs[job_id].get("MaxRSS", "")):
            jobs[job_id]["MaxRSS"] = rec["MaxRSS"]
    return [AcctJob(id=i, name=r["JobName"], state=r["State"].split()[0] if r["State"] else "",
                    elapsed=r["Elapsed"], start=r["Start"], end=r["End"], nodes=r["NodeList"],
                    exit_code=r["ExitCode"], max_rss=r["MaxRSS"], cpus=r["AllocCPUS"], partition=r["Partition"],
                    workdir=r["WorkDir"]) for i, r in ((i, jobs[i]) for i in order)]


# ---------------------------------------------------------------------------- commands

class SlurmClient:
    """squeue, sacct and scancel for the current user."""

    def __init__(self, user: str | None = None, which: Callable[[str], str | None] = shutil.which,
                 timeout: float = 30):
        self.user = user or os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        self.which = which
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.which("squeue") is not None

    def _call(self, program: str, *args: str) -> str:
        exe = self.which(program)
        if exe is None:
            raise SlurmError(f"{program} not found")
        try:
            r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=self.timeout,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            raise SlurmError(f"{program}: {e}") from None
        if r.returncode != 0:
            raise SlurmError(f"{program}: {(r.stderr or r.stdout).strip() or f'exit code {r.returncode}'}")
        return r.stdout

    def queue(self) -> list[QueueJob]:
        args = ["-h", "-o", SQUEUE_FORMAT] + (["-u", self.user] if self.user else [])
        return parse_squeue(self._call("squeue", *args))

    def stdout_paths(self, ids: list[str]) -> dict[str, str]:
        """Where SLURM writes the stdout of these jobs (squeue --Format StdOut)."""
        if not ids:
            return {}
        out = {}
        text = self._call("squeue", "-h", "-j", ",".join(ids), "-O", "JobID:40,StdOut:4096")
        for line in text.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
        return out

    def accounting(self, ids: list[str]) -> list[AcctJob]:
        if not ids or self.which("sacct") is None:
            return []
        since = time.strftime("%Y-%m-%d", time.localtime(time.time() - HISTORY_DAYS * 86400))
        return parse_sacct(self._call("sacct", "-P", "-n", "-S", since, "-j", ",".join(ids),
                                      "-o", ",".join(SACCT_FIELDS)))

    def accounting_by_name(self, names: list[str]) -> list[AcctJob]:
        """sacct of the jobs with these names (RELION names them after the job directory)."""
        if not names or self.which("sacct") is None:
            return []
        since = time.strftime("%Y-%m-%d", time.localtime(time.time() - HISTORY_DAYS * 86400))
        return parse_sacct(self._call("sacct", "-P", "-n", "-S", since, "--name", ",".join(names),
                                      "-o", ",".join(SACCT_FIELDS)))

    def submitted_since(self, since: float) -> list[tuple[str, str, str]]:
        """(id, working directory, submit line) of this user's SLURM jobs submitted since ``since`` (epoch
        seconds; sacct, allocations only). SlurmError also when this sacct has no SubmitLine field."""
        if self.which("sacct") is None:
            return []
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since))
        text = self._call("sacct", "-P", "-n", "-X", "-S", start, "-o", "JobID,WorkDir,SubmitLine",
                          *(["-u", self.user] if self.user else []))
        out = []
        for line in text.splitlines():
            parts = line.split("|", 2)  # the submit line comes last: it may hold anything
            if len(parts) == 3 and parts[0]:
                out.append((parts[0], parts[1], parts[2].strip()))
        return out

    def cancel(self, ids: list[str]) -> None:
        if ids:
            self._call("scancel", *ids)


# ---------------------------------------------------------------------------- mapping

class QueueMapper:
    """Which SLURM jobs belong to which jobs of one RELION project."""

    def __init__(self, project_dir: str | os.PathLike, persist: bool = True):
        self.project_dir = Path(project_dir)
        self.persist = persist  # False (--readonly): the ids are only remembered for this session
        self._real = os.path.realpath(self.project_dir)
        self.known: dict[str, str] = {}  # SLURM id -> RELION job
        self._logs: dict[Path, tuple[tuple, list[str]]] = {}
        self._do_queue: dict[str, tuple[float, bool]] = {}
        self._load()

    # -- remembered ids -----------------------------------------------------------------------
    @property
    def _known_path(self) -> Path:
        return self.project_dir / GUI_DIR / KNOWN_FILE

    def _load(self) -> None:
        try:
            data = json.loads(self._known_path.read_text())
            self.known = {str(k): str(v) for k, v in data.get("jobs", {}).items()}
        except (OSError, ValueError, AttributeError):
            self.known = {}

    def _save(self) -> None:
        if not self.persist:
            return
        def merge(data: dict) -> dict:  # other sessions save it too: keep the ids they found
            theirs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
            known = {str(k): str(v) for k, v in theirs.items()}
            known.update(self.known)
            return {"jobs": dict(sorted(known.items(), key=lambda kv: _id_order(kv[0])))}

        try:
            self.known = dict(update_json(self._known_path, merge)["jobs"])
        except (OSError, ValueError):
            pass  # a read-only project: mapping still works, the history only lasts this session

    def history(self, job: str) -> list[str]:
        """SLURM ids seen for a RELION job (in squeue or in the run logs), oldest first."""
        ids = {i for i, j in self.known.items() if j == job}
        ids |= {i for i, j in self.submitted_in_runs().items() if j == job}
        return sorted(ids, key=_id_order)

    def remember(self, accounting: list[AcctJob], job: str) -> list[AcctJob]:
        """Keep the sacct records of this project's ``job`` (by working directory), remembering their ids."""
        mine = [a for a in accounting if not a.workdir or os.path.realpath(a.workdir) == self._real]
        new = {a.id: job for a in mine if self.known.get(a.id) != job}
        if new:
            self.known.update(new)
            self._save()
        return mine

    def by_submit_line(self, job: str, rows: list[tuple[str, str, str]]) -> list[str]:
        """The SLURM ids among ``rows`` (SlurmClient.submitted_since) that ran this project's ``job``: their
        submit line names its run_submit.script (relative to the job's working directory, or absolute)."""
        script = os.path.realpath(os.path.join(self._real, job, "run_submit.script"))
        out = []
        for job_id, workdir, line in rows:
            for word in line.split():
                word = word.strip("'\"")
                if not word.endswith("run_submit.script"):
                    continue
                path = word if os.path.isabs(word) else os.path.join(workdir or self._real, word)
                if os.path.realpath(path) == script:
                    out.append(job_id)
                    break
        return out

    # -- mapping --------------------------------------------------------------------------------
    def here(self, q: QueueJob) -> bool:
        """Whether a SLURM job was submitted from this project (working directory)."""
        return not q.workdir or os.path.realpath(q.workdir) == self._real

    def map(self, jobs: set[str], queue: list[QueueJob],
            stdout_paths: Callable[[list[str]], dict[str, str]] | None = None) -> dict[str, QueueJob]:
        """RELION job -> its SLURM job in ``queue`` (the newest if several)."""
        found: dict[str, str] = {}
        for q in queue:
            job = self._by_name(q, jobs) or (self.known.get(q.id) if self.known.get(q.id) in jobs else None)
            if job:
                found[q.id] = job
        rest = [q for q in queue if q.id not in found and self.here(q)]
        if rest and stdout_paths is not None:
            try:
                paths = stdout_paths([q.id for q in rest])
            except SlurmError:
                paths = {}
            for q in rest:
                job = self._job_of_path(paths.get(q.id, ""), jobs)
                if job:
                    found[q.id] = job
        rest = [q for q in rest if q.id not in found]
        if rest:
            submitted = self.submitted_in_runs()
            for q in rest:
                if submitted.get(q.id) in jobs:
                    found[q.id] = submitted[q.id]
        changed = False
        for i, job in found.items():
            if self.known.get(i) != job:
                self.known[i] = job
                changed = True
        if changed:
            self._save()
        out: dict[str, QueueJob] = {}
        for q in sorted(queue, key=lambda q: _id_order(q.id)):
            if q.id in found:
                out[found[q.id]] = q
        return out

    def _by_name(self, q: QueueJob, jobs: set[str]) -> str | None:
        if not self.here(q):
            return None
        name = q.name if q.name.endswith("/") else q.name + "/"
        return name if name in jobs else None

    def _job_of_path(self, path: str, jobs: set[str]) -> str | None:
        if not path:
            return None
        real = os.path.realpath(path if os.path.isabs(path) else os.path.join(self._real, path))
        if not real.startswith(self._real + os.sep):
            return None
        parts = os.path.relpath(real, self._real).split(os.sep)
        job = "/".join(parts[:2]) + "/" if len(parts) >= 3 else None
        return job if job in jobs else None

    def submitted_in_runs(self) -> dict[str, str]:
        """SLURM id -> RELION job from the sbatch lines in cryo-savanna's run logs."""
        out: dict[str, str] = {}
        runs = self.project_dir / GUI_DIR / "runs"
        for rec_file in sorted(runs.glob("*.json")) if runs.is_dir() else []:
            try:
                rec = json.loads(rec_file.read_text())
            except (OSError, ValueError):
                continue
            ids = self._submitted_ids(self.project_dir / rec.get("log", ""))
            queued = [j for j in rec.get("jobs", []) if self.goes_to_queue(j)]
            if ids and queued:
                for k, i in enumerate(ids):  # repeated runs submit the same jobs again, in order
                    out[i] = queued[k % len(queued)]
        return out

    def _submitted_ids(self, log: Path) -> list[str]:
        try:
            st = log.stat()
        except OSError:
            return []
        key = (st.st_mtime_ns, st.st_size)
        hit = self._logs.get(log)
        if hit is None or hit[0] != key:
            try:
                ids = _SUBMITTED.findall(log.read_text(errors="replace"))
            except OSError:
                ids = []
            self._logs[log] = hit = (key, ids)
        return hit[1]

    def goes_to_queue(self, job: str) -> bool:
        path = self.project_dir / job / "job.star"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        hit = self._do_queue.get(job)
        if hit is None or hit[0] != mtime:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                return False
            m = re.search(r"^\s*do_queue\s+(\S+)", text, re.M)
            hit = (mtime, bool(m) and m.group(1).strip("\"'") == "Yes")
            self._do_queue[job] = hit
        return hit[1]


# ---------------------------------------------------------------------------- monitor

@dataclass
class QueueState:
    """One look at the queue, for one project."""
    available: bool = False
    error: str = ""
    checked_at: float = 0.0
    live: dict[str, QueueJob] = field(default_factory=dict)  # RELION job -> its SLURM job in squeue
    lost: dict[str, AcctJob] = field(default_factory=dict)  # Running in RELION, but ended in SLURM
    user_jobs: int = 0  # all of the user's jobs in squeue

    def counts(self) -> tuple[int, int]:
        queued = sum(q.pending for q in self.live.values())
        return queued, len(self.live) - queued


class SlurmMonitor:
    """Looks at squeue (and, for jobs that left it without an exit file, sacct)."""

    def __init__(self, project_dir: str | os.PathLike, client: SlurmClient | None = None, persist: bool = True):
        self.project_dir = Path(project_dir)
        self.client = client or SlurmClient()
        self.mapper = QueueMapper(project_dir, persist)
        self._suspects: dict[str, tuple[tuple[str, ...], float, AcctJob | None]] = {}  # job -> (ids, asked at, last)
        self._submit_line = True  # False once sacct turned out not to know SubmitLine (an older SLURM)

    def poll(self, jobs: dict[str, str]) -> QueueState:
        """``jobs``: RELION job -> status, as the GUI shows it."""
        state = QueueState(available=self.client.available, checked_at=time.time())
        if not state.available:
            return state
        try:
            queue = self.client.queue()
        except SlurmError as e:
            state.error = str(e)
            return state
        state.user_jobs = len(queue)
        state.live = self.mapper.map(set(jobs), queue, self.client.stdout_paths)
        # Running in RELION, sent to the queue, but no longer in squeue and no exit file: ask sacct
        suspects = [job for job, status in jobs.items()
                    if status == STATUS_RUNNING and job not in state.live and self.mapper.goes_to_queue(job)
                    and not self._has_exit_file(job)]
        self._suspects = {j: hit for j, hit in self._suspects.items() if j in suspects}  # rerun: ask afresh
        for job in suspects:
            try:
                last = self._last_slurm_job(job, state.checked_at)
            except SlurmError as e:
                state.error = str(e)
                break
            if self._lost(last, state.checked_at) and not self._before_submission(job, last):
                state.lost[job] = last
        return state

    @staticmethod
    def _lost(last: AcctJob | None, now: float) -> bool:
        end = last.end_time() if last is not None else None
        return last is not None and not last.active and (end is None or now - end > LOST_GRACE_S)

    def _before_submission(self, job: str, last: AcctJob) -> bool:
        """Whether that SLURM job ended before the job was last sent to the queue (RELION writes
        run_submit.script on every submission): an earlier run's, not the one RELION waits for."""
        end = last.end_time()
        try:
            submitted = (self.project_dir / job / "run_submit.script").stat().st_mtime
        except OSError:
            return False
        return end is not None and end < submitted

    def _last_slurm_job(self, job: str, now: float) -> AcctJob | None:
        """The newest SLURM job of a job that may be lost. sacct is not asked on every poll: a SLURM job
        that ended stays as it is (until the job gets a new SLURM id), others are asked again after
        LOST_RECHECK_S."""
        hit = self._suspects.get(job)
        if hit is not None and hit[0] == tuple(self.mapper.history(job)):
            ids, asked, last = hit
            if (last is not None and not last.active) or now - asked < LOST_RECHECK_S:
                return last
        last = next(iter(self.history(job)), None)
        self._suspects[job] = (tuple(self.mapper.history(job)), now, last)  # after history(): it may remember ids
        return last

    def _has_exit_file(self, job: str) -> bool:
        d = self.project_dir / job
        return any((d / f).exists() for f in (RELION_JOB_EXIT_SUCCESS, RELION_JOB_EXIT_FAILURE, RELION_JOB_EXIT_ABORTED))

    def history(self, job: str) -> list[AcctJob]:
        """sacct of every SLURM job of a RELION job, newest first: the ids seen before, or else the
        jobs named after its directory, or else those whose submit line ran its run_submit.script
        (submitted while cryo-savanna was not looking)."""
        ids = self.mapper.history(job)
        acct = self.client.accounting(ids) if ids else []
        if not acct:
            acct = self.mapper.remember(self.client.accounting_by_name([job]), job)
        if not acct:
            acct = self.mapper.remember(self._by_submit_line(job), job)
        return sorted(acct, key=lambda a: _id_order(a.id), reverse=True)

    def _by_submit_line(self, job: str) -> list[AcctJob]:
        """sacct of the SLURM jobs that ran this job's run_submit.script, since the job first ran (its note.txt;
        else a day before its run_submit.script was written), for templates without --job-name=XXXnameXXX."""
        if not self._submit_line:
            return []
        d = self.project_dir / job
        try:
            written = (d / "run_submit.script").stat().st_mtime
        except OSError:
            return []  # never sent to the queue
        since = max(time.time() - HISTORY_DAYS * 86400, (_first_start(d / "note.txt") or written - 86400) - 600)
        try:
            rows = self.client.submitted_since(since)
        except SlurmError as e:
            if "submitline" in str(e).lower().replace(" ", ""):  # "Invalid field requested: SubmitLine"
                self._submit_line = False
                return []
            raise
        ids = self.mapper.by_submit_line(job, rows)
        return self.client.accounting(ids) if ids else []
