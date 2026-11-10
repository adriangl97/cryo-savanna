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
            pipeline's job counter set back to its number, so it gets the same directory.
- abort:    RELION_JOB_ABORT_NOW in the job directory (what RELION's programs check).
- clean:    relion_pipeliner --gentle_clean N / --harsh_clean N.

Every run started here is recorded in <project>/.cryo-savanna/runs/<id>.json with its output in
<id>.log, so the GUI can still stop a repeating run after a restart (``repeating_runs``,
``stop_schedule``) and find the SLURM jobs a run submitted (project/slurm.py).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryo_savanna.env.child import external_environ
from cryo_savanna.jobdefs.cli_tools import cli_program_of, install_runner
from cryo_savanna.jobdefs import pytom_jobs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.ops import TRASH_DIR, PipelineOps
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.relion_consts import RELION_JOB_ABORT_NOW, RELION_JOB_EXIT_SUCCESS, STATUS_RUNNING, job_type_for_label

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
            return True
        # a zombie of our own child counts as finished
        try:
            done, _ = os.waitpid(self.pid, os.WNOHANG)
            return done == 0
        except ChildProcessError:
            return True


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
        ops = PipelineOps(self.project_dir)
        ops.delete(job)
        old_counter = ops.set_job_counter(pr.job_number)
        try:
            res = self.add_job(jobstar, alias)
        except Exception as e:  # also a timeout or an unwritable job.star: the old job is in Trash/ either way
            raise PipelinerError(f"{e}\n\n{_in_trash(job)}") from e
        finally:
            ops.restore_job_counter(old_counter)
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
            ops.delete(job)
            shutil.rmtree(old, ignore_errors=True)
            shutil.move(str(aside), str(old))
            ops.undelete(old / "deleted_pipeline.star")
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
        """A CLI tool job runs the project's copy of cli_runner.py, a pytom job that of pytom_runner.py: put
        it there (jobdefs/cli_tools.py, jobdefs/pytom_jobs.py)."""
        if cli_program_of(jobstar) is not None:
            install_runner(self.project_dir)
        elif pytom_jobs.pytom_kind_of(jobstar) is not None:
            pytom_jobs.install_runner(self.project_dir)

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
        jobstar.is_continue = True
        self._runner_for(jobstar)
        jobstar.write(self.project_dir / job / "job.star")
        return self.run([job], **run_args)

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

    # ------------------------------------------------------------------ abort / clean
    def abort(self, job: str) -> None:
        (self.project_dir / job / RELION_JOB_ABORT_NOW).touch()

    def clean(self, job: str, harsh: bool = False) -> None:
        number = Pipeline.read(self.project_dir).processes[job].job_number
        if number is None:
            raise PipelinerError(f"not a job: {job}")
        self._call("--harsh_clean" if harsh else "--gentle_clean", str(number))


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
