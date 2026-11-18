# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Adding and running jobs through RELION's own relion_pipeliner.

RELION builds the commands and owns default_pipeline.star; cryo-savanna only writes job.star
files (byte-identical to RELION's: tests/test_against_relion.py) and starts the pipeliner:

- add:      relion_pipeliner --addJobFromStar <job.star> [--setJobAlias A]  -> a Scheduled job
- run:      relion_pipeliner --RunJobs "a b c" --schedule S ...             started detached,
            so jobs keep going when the GUI or the X11 session closes. It waits for each job's
            input files, runs the jobs in order and stops at the first failure or abort.
- continue: the job's own job.star gets the continue flag and the changed options, then run.
- overwrite: as RELION's GUI does it: the job goes to Trash/, and the new job is added with the
            pipeline's job counter set back to its number, so it gets the same directory, and takes the
            old job's place in the pipeline (no job moves: RELION's pipeliners keep running jobs by place).
- abort:    RELION_JOB_ABORT_NOW in the job directory (what RELION's programs check).
- force abort: the job's processes on this host killed, RELION_JOB_EXIT_ABORTED written (project/processes.py).
- clean:    relion_pipeliner --gentle_clean N / --harsh_clean N.

Every run started here is recorded in <project>/.cryo-savanna/runs/<id>.json with its output in
<id>.log, so the GUI can still stop a repeating run after a restart (``repeating_runs``,
``stop_schedule``) and find the SLURM jobs a run submitted (project/slurm.py).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryo_savanna.env.child import external_environ
from cryo_savanna.jobdefs import external_jobs
from cryo_savanna.project import processes
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.ops import TRASH_DIR, PipelineOps, placeholder
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.relion_consts import (
    RELION_JOB_ABORT_NOW,
    RELION_JOB_EXIT_SUCCESS,
    STATUS_RUNNING,
    STATUS_SCHEDULED,
    job_type_for_label,
)

GUI_DIR = ".cryo-savanna"
SCHEDULE = "cryo-savanna"  # one pipeline_cryo-savanna.log for all single runs (RELION writes it in the project)


class PipelinerError(RuntimeError):
    pass


@dataclass
class RunRecord:
    id: str
    jobs: list[str]
    schedule: str
    pid: int
    host: str
    user: str
    started: float
    command: list[str]
    repeat: int = 1
    log: str = ""  # relative to the project

    def is_alive(self) -> bool | None:
        """True/False on the host that started it; None elsewhere (cannot tell)."""
        if self.host != socket.gethostname():
            return None
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return self._is_this_run()  # another user's process: their run, or the number given out again
        # a zombie of our own child counts as finished
        try:
            done, _ = os.waitpid(self.pid, os.WNOHANG)
            return done == 0
        except ChildProcessError:
            return self._is_this_run()  # not started by this window: this run, or the number given out again

    def _is_this_run(self) -> bool:
        """Whether the process with this run's number is its relion_pipeliner (its command line has the
        run's --schedule name), not another program that got the number after the run ended. True when
        that cannot be told (no /proc and no ps)."""
        args = _command_line(self.pid)
        if args is None:
            try:  # gone meanwhile (reaped by another look at it, in another thread), or cannot be told
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
            return True
        return "--RunJobs" in args and self.schedule in args


@dataclass
class Held:
    """A job that a relion_pipeliner started here and still going is to run (or runs): see Pipeliner.held_jobs."""
    run: RunRecord
    current: bool  # the job the run is at now (not one after it)
    waiting_for: list[str] = field(default_factory=list)  # its input files not there yet (RELION waits for them)

    @property
    def waiting(self) -> bool:
        """The run waits for this job's input files: it has not started it, and nothing of it runs."""
        return self.current and bool(self.waiting_for)

    def describe(self) -> str:
        started = time.strftime("%d %b %H:%M", time.localtime(self.run.started))
        text = f"in a run started {started} (relion_pipeliner {self.run.pid} on {self.run.host})"
        if self.waiting:
            text += ", waiting for " + ", ".join(self.waiting_for)
        elif not self.current:
            text += ", after the job it runs now"
        return text


@dataclass
class AddResult:
    job: str
    warnings: list[str] = field(default_factory=list)


class Pipeliner:
    def __init__(self, project_dir: str | os.PathLike, exe: str | None = None, environ: dict | None = None):
        self.project_dir = Path(project_dir)
        self.exe = exe or shutil.which("relion_pipeliner")
        # None: this process's environment (the loaded RELION module), without cryo-savanna's own
        # settings: RELION's GUI tools (relion_display, napari for tomography) need OpenGL
        self.environ = environ if environ is not None else external_environ()
        self._records: dict[str, RunRecord] = {}  # run record file -> its record (see runs)

    @property
    def available(self) -> bool:
        return self.exe is not None

    def _require(self) -> str:
        if not self.exe:
            raise PipelinerError("relion_pipeliner was not found on PATH: load RELION 5.1 first")
        return self.exe

    def _gui_dir(self, sub: str) -> Path:
        d = self.project_dir / GUI_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _call(self, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
        r = subprocess.run([self._require(), *args], cwd=self.project_dir, env=self.environ, capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise PipelinerError((r.stderr or r.stdout).strip() or f"relion_pipeliner exited with {r.returncode}")
        return r

    # ------------------------------------------------------------------ add
    def add_job(self, jobstar: JobStar, alias: str | None = None) -> AddResult:
        """Add a job as Scheduled. Returns its name ("Class2D/job012/")."""
        self._runner_for(jobstar)
        tmp = self._gui_dir("tmp") / f"{uuid.uuid4().hex}.job.star"
        jobstar.write(tmp)
        before = set(Pipeline.read(self.project_dir).processes)
        args = ["--addJobFromStar", str(tmp.relative_to(self.project_dir))]
        if alias:
            args += ["--setJobAlias", alias]
        try:
            r = self._call(*args)
        finally:
            tmp.unlink(missing_ok=True)
        after = Pipeline.read(self.project_dir)
        # RELION labels some jobs by what they do ("relion.select.interactive", "relion.autopick.log")
        new = [n for n in after.processes if n not in before
               and _same_type(after.processes[n].type_label, jobstar.type_label)]
        if len(new) > 1:  # someone else added jobs at the same time: find ours by its options
            new = [n for n in new if self._same_options(n, jobstar)] or new
        if not new:
            raise PipelinerError("relion_pipeliner did not add the job (no new job in the pipeline)")
        warnings = [line.strip() for line in r.stderr.splitlines() if "WARNING" in line]
        return AddResult(new[-1], warnings)

    def overwrite_job(self, job: str, jobstar: JobStar, alias: str | None = None) -> AddResult:
        """Replace ``job`` by a new job with these options in the same directory (as Scheduled).

        The old job and its files go to Trash/ (RELION's GUI does the same). Jobs that used its
        outputs are linked to the new job's outputs of the same name.
        """
        self._require()
        pr = Pipeline.read(self.project_dir).processes.get(job)
        if pr is None or pr.job_number is None:
            raise PipelinerError(f"no such job: {job}")
        if pr.status == STATUS_RUNNING:
            raise PipelinerError(f"{job.rstrip('/')} is running: abort it first.")
        self._check_free([job])  # the run would run the new job, and looks it up by name meanwhile
        ops = PipelineOps(self.project_dir)
        # the old job's place is kept, and the new job (added at the end) is put there: as in RELION's GUI,
        # which runs it where it was, no job moves (RELION's pipeliners keep running jobs by their place)
        ops.delete(job, keep_place=True)
        held = placeholder(job)
        old_counter = ops.set_job_counter(pr.job_number)
        try:
            res = self.add_job(jobstar, alias)
        except Exception as e:  # also a timeout or an unwritable job.star: the old job is in Trash/ either way
            ops.take_place(held, None)
            raise PipelinerError(f"{e}\n\n{_in_trash(job)}") from e
        finally:
            ops.restore_job_counter(old_counter)
        ops.take_place(held, res.job)
        # relion_pipeliner takes the lock itself, so the counter is set back and restored around it (not
        # under one lock, as in RELION's GUI): a job added meanwhile elsewhere can get a number that a
        # job further down already has
        twins = _same_numbers(Pipeline.read(self.project_dir), pr.job_number, old_counter)
        if twins:
            res.warnings.append("Jobs were added elsewhere during the overwrite, and these now share a job number: "
                                + "; ".join(", ".join(n.rstrip("/") for n in names) for names in twins)
                                + ". Commands that take a job number (Clean: relion_pipeliner --gentle_clean / --harsh_clean) "
                                  "may act on the wrong one; delete one of each to be safe.")
        if res.job != job:
            res.warnings.append(f"Another job was added at the same moment: the new job is {res.job}, "
                                f"and {job.rstrip('/')} is in {TRASH_DIR}/.")
            return res
        trashed = self.project_dir / TRASH_DIR / job / "deleted_pipeline.star"
        if trashed.exists():
            ops.relink_consumers(job, trashed)
        return res

    # ------------------------------------------------------------------ jobs done in cryo-savanna
    def add_done_here(self, jobstar: JobStar, write, command, alias: str | None = None) -> AddResult:
        """Add a job whose work was done in cryo-savanna (a selection): add it with relion_pipeliner,
        ``write(job)`` its output files, then finish it as RELION does when a job ends.

        If writing fails, the job goes to Trash/ again and the error is raised.
        """
        res = self.add_job(jobstar, alias)
        try:
            self.finish_done_here(res.job, write, command)
        except Exception:
            PipelineOps(self.project_dir).delete(res.job)
            raise
        return res

    def overwrite_done_here(self, job: str, jobstar: JobStar, write, command, alias: str | None = None) -> AddResult:
        """overwrite_job for a job whose work is done in cryo-savanna (a selection). If writing fails,
        the new job is removed and the old one is put back from Trash/, as it was."""
        res = self.overwrite_job(job, jobstar, alias)
        try:
            self.finish_done_here(res.job, write, command)
        except Exception as e:
            ops = PipelineOps(self.project_dir)
            if res.job != job:  # another job took the number: as in add_done_here
                ops.delete(res.job)
                raise PipelinerError(f"{e}\n\n{_in_trash(job)}") from e
            # deleting the new job would replace the old one in Trash/: keep it aside meanwhile
            old = self.project_dir / TRASH_DIR / job.rstrip("/")
            aside = old.with_name(old.name + ".overwritten")
            shutil.move(str(old), str(aside))
            ops.delete(job, keep_place=True)  # and the old job back in its place (see overwrite_job)
            shutil.rmtree(old, ignore_errors=True)
            shutil.move(str(aside), str(old))
            ops.undelete(old / "deleted_pipeline.star")
            ops.take_place(placeholder(job), job)
            (self.project_dir / job / "deleted_pipeline.star").unlink(missing_ok=True)  # it was never deleted
            raise PipelinerError(f"{e}\n\n{job.rstrip('/')} was not overwritten: it is back as it was.") from e
        return res

    def finish_done_here(self, job: str, write, command) -> None:
        """Write (or rewrite) the outputs of ``job``, note what produced them, mark it finished.

        ``command``: the command RELION would have run (a string, or a function of the job name).
        """
        write(job)
        command = command(job) if callable(command) else command
        d = self.project_dir / job
        with open(d / "note.txt", "a") as fh:
            fh.write(f" ++++ Executing new job on {time.strftime('%a %b %d %H:%M:%S %Y')}\n"
                     f" ++++ with the following command(s): \n{command}\n"
                     f" ++++ (done interactively in cryo-savanna, which wrote the files this command writes)\n ++++ \n")
        (d / RELION_JOB_EXIT_SUCCESS).touch()
        PipelineOps(self.project_dir).mark_finished(job)

    def _runner_for(self, jobstar: JobStar) -> None:
        """A job made from one of cryo-savanna's own forms (a CLI tool, pytom, AreTomo3, tomogram masks) runs
        the project's copy of its runner: put it there (jobdefs/external_jobs.py)."""
        external_jobs.install_runner(jobstar, self.project_dir)

    def _same_options(self, job: str, jobstar: JobStar) -> bool:
        try:
            return JobStar.read(self.project_dir / job / "job.star").options == jobstar.options
        except (OSError, ValueError, KeyError):
            return False

    # ------------------------------------------------------------------ run
    def run(self, jobs: list[str], repeat: int = 1, min_wait: int = 0, min_wait_before: int = 0,
            sec_wait_after: int = 10) -> RunRecord:
        """Run these jobs, in this order, in a detached relion_pipeliner."""
        if not jobs:
            raise PipelinerError("no jobs to run")
        self._check_free(jobs)
        for job in jobs:  # scheduled earlier, perhaps by another cryo-savanna (or the project's copy was removed)
            try:
                self._runner_for(JobStar.read(self.project_dir / job / "job.star"))
            except (OSError, ValueError, KeyError):
                pass
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        schedule = SCHEDULE if repeat <= 1 else f"{SCHEDULE}_{rid}"
        cmd = [self._require(), "--RunJobs", " ".join(jobs), "--schedule", schedule, "--repeat", str(repeat),
               "--min_wait", str(min_wait), "--min_wait_before", str(min_wait_before),
               "--sec_wait_after", str(sec_wait_after)]
        runs = self._gui_dir("runs")
        log = runs / f"{rid}.log"
        with open(log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=self.project_dir, env=self.environ, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        rec = RunRecord(rid, list(jobs), schedule, proc.pid, socket.gethostname(), _user(), time.time(), cmd, repeat,
                        str(log.relative_to(self.project_dir)))
        tmp = runs / f"{rid}.json.tmp"  # complete or absent: runs() reads each record only once
        tmp.write_text(json.dumps(asdict(rec), indent=1))
        os.replace(tmp, runs / f"{rid}.json")
        return rec

    def continue_job(self, job: str, jobstar: JobStar, **run_args) -> RunRecord:
        """Continue an existing job with these (continue-mode) options."""
        self._check_free([job])
        jobstar.is_continue = True
        self._runner_for(jobstar)
        jobstar.write(self.project_dir / job / "job.star")
        return self.run([job], **run_args)

    # ------------------------------------------------------------------ runs still going
    def held_jobs(self, pipeline: Pipeline | None = None) -> dict[str, Held]:
        """The jobs that a relion_pipeliner started on this host, and still going, is to run or runs: those of
        its jobs the pipeline has as Scheduled or Running (it goes through them in order and stops at the first
        that fails, so one that ended is done). Runs on other hosts cannot be told apart from ended ones.

        Such a job must stay in the pipeline under its name (at the end of a run, RELION's pipeliner looks each
        of its jobs up by name, and does not check that it found it: it corrupts its memory, and can leave
        default_pipeline.star empty with the lock taken), and must not be run by a second pipeliner."""
        p = pipeline if pipeline is not None else Pipeline.read(self.project_dir)
        out: dict[str, Held] = {}
        for rec in sorted(self.runs(), key=lambda r: r.started):  # a newer run of a job wins
            ahead = [j for j in rec.jobs if j in p.processes and p.processes[j].status in (STATUS_SCHEDULED,
                                                                                             STATUS_RUNNING)]
            if not ahead or rec.is_alive() is not True:
                continue
            for k, job in enumerate(ahead):
                pr = p.processes[job]
                missing = ([n for n in pr.inputs if not (self.project_dir / n).exists()]
                           if k == 0 and pr.status == STATUS_SCHEDULED else [])
                out[job] = Held(rec, k == 0, missing)
        return out

    def _check_free(self, jobs: list[str]) -> None:
        """Refuse to run jobs that a run still going is to run already (two pipeliners for one job: both
        would run it, in the same directory, and both would set its status)."""
        try:
            held = self.held_jobs()
        except (OSError, ValueError, RuntimeError):
            return  # the pipeline cannot be read now: relion_pipeliner tells
        taken = [j for j in jobs if j in held]
        if taken:
            raise PipelinerError("Already in a run that is still going:\n" + "\n".join(
                f"  {j.rstrip('/')}: {held[j].describe()}" for j in taken)
                + "\n\nWait for it, or stop that run first (Job → Stop runs…).")

    def stoppable_runs(self, pipeline: Pipeline | None = None) -> list[tuple[RunRecord, str]]:
        """Runs that can be stopped, newest first: (run, "repeating") for a repeating run (it stops after its
        current job), (run, "waiting") for a run on this host that waits for the input files of its next job
        (its relion_pipeliner can be ended now: nothing of the job has started)."""
        out: dict[str, tuple[RunRecord, str]] = {r.id: (r, "repeating") for r in self.repeating_runs()}
        for held in self.held_jobs(pipeline).values():
            if held.waiting and held.run.id not in out:
                out[held.run.id] = (held.run, "waiting")
        return sorted(out.values(), key=lambda x: x[0].started, reverse=True)

    def end_run(self, rec: RunRecord) -> None:
        """End the relion_pipeliner of a run that waits for input files (stoppable_runs). It is ended while
        cryo-savanna holds RELION's pipeline lock, so it is not in the middle of writing the pipeline (RELION
        writes the file in place: a pipeliner ended there leaves it empty). The jobs it was to run stay
        Scheduled."""
        if rec.host != socket.gethostname():
            raise PipelinerError(f"That run was started on {rec.host}: stop it from a cryo-savanna there.")
        args = _command_line(rec.pid)
        if rec.is_alive() is not True or (args is not None and "--RunJobs" not in args):
            return  # it ended meanwhile
        with PipelineLock(self.project_dir, message="cryo-savanna: stopping a run that waits for its inputs"):
            try:
                os.kill(rec.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 10
            while rec.is_alive() and time.monotonic() < deadline:
                time.sleep(0.2)
        if rec.is_alive():
            raise PipelinerError(f"relion_pipeliner {rec.pid} did not stop.")

    def stop_schedule(self, rec: RunRecord) -> bool:
        """Stop a repeating run after the current job (RELION checks RUNNING_PIPELINER_<schedule>)."""
        flag = self.project_dir / f"RUNNING_PIPELINER_{rec.schedule}"
        if flag.exists():
            flag.unlink()
            return True
        return False

    def repeating_runs(self) -> list[RunRecord]:
        """Repeating runs that are still going. Found from RUNNING_PIPELINER_<schedule>, so also for runs
        started on another host; but RELION removes that file only when every repeat succeeded, so a run
        that stopped at a failed or aborted job (its pipeline_<schedule>.log says so), or whose
        relion_pipeliner is gone on this host, does not count."""
        return [r for r in self.runs()
                if r.repeat > 1 and (self.project_dir / f"RUNNING_PIPELINER_{r.schedule}").exists()
                and r.is_alive() is not False and not self._stopped(r)]

    def _stopped(self, rec: RunRecord) -> bool:
        """Whether RELION's scheduler log says the run stopped (runScheduledJobs writes this when it ends)."""
        try:
            with open(self.project_dir / f"pipeline_{rec.schedule}.log", "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - 4096))
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return False
        return " + Stopping pipeliner" in tail or "Stopping pipeliner now" in tail

    def runs(self) -> list[RunRecord]:
        """Every run recorded in the project. A record never changes once written, so each file is read
        once: this runs on every refresh of the window, and the records pile up over a project's life."""
        out = []
        for f in sorted((self.project_dir / GUI_DIR / "runs").glob("*.json")):
            rec = self._records.get(f.name)
            if rec is None:
                try:
                    rec = self._records[f.name] = RunRecord(**json.loads(f.read_text()))
                except (OSError, ValueError, TypeError):
                    continue
            out.append(rec)
        return out

    def runs_of(self, job: str) -> list[RunRecord]:
        """The recorded runs that ran (or run) ``job``, newest first."""
        return sorted((r for r in self.runs() if job in r.jobs), key=lambda r: r.started, reverse=True)

    # ------------------------------------------------------------------ abort / clean
    def abort(self, job: str) -> None:
        (self.project_dir / job / RELION_JOB_ABORT_NOW).touch()

    def force_abort_targets(self, job: str) -> tuple[list[processes.Proc], str]:
        """The processes of ``job`` on this host (project/processes.py), and, when there are none, the host the
        last run of it was started on if that is another one ("" otherwise)."""
        here = socket.gethostname()
        runs = self.runs_of(job)
        procs = processes.job_processes(self.project_dir, job, [r.pid for r in runs if r.host == here and r.is_alive()])
        return procs, (runs[0].host if not procs and runs and runs[0].host != here else "")

    def force_abort(self, job: str, procs: list[processes.Proc]) -> None:
        """Kill ``procs`` (force_abort_targets) and mark the job as aborted."""
        processes.kill(self.project_dir, job, procs)

    def clean(self, job: str, harsh: bool = False) -> None:
        number = Pipeline.read(self.project_dir).processes[job].job_number
        if number is None:
            raise PipelinerError(f"not a job: {job}")
        self._call("--harsh_clean" if harsh else "--gentle_clean", str(number))


def _command_line(pid: int) -> list[str] | None:
    """The arguments of a running process (/proc, else ps); None when they cannot be read."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            data = fh.read()
        if data:
            return [a.decode("utf-8", "replace") for a in data.split(b"\0") if a]
    except OSError:
        pass
    try:
        r = subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=10,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.split() if r.returncode == 0 and r.stdout.strip() else None


def _in_trash(job: str) -> str:
    return f"{job.rstrip('/')} was moved to {TRASH_DIR}/: use Project → Undelete to put it back."


def _same_numbers(p: Pipeline, lo: int, hi: int) -> list[list[str]]:
    """Jobs with numbers in [lo, hi) that share their number with another job (pipeline order)."""
    by_number: dict[int, list[str]] = {}
    for name, pr in p.processes.items():
        n = pr.job_number
        if n is not None and lo <= n < hi:
            by_number.setdefault(n, []).append(name)
    return [names for _, names in sorted(by_number.items()) if len(names) > 1]


def _same_type(a: str, b: str) -> bool:
    ta, tb = job_type_for_label(a), job_type_for_label(b)
    return ta is not None and ta is tb or a == b


def _user() -> str:
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get("USER", "")
