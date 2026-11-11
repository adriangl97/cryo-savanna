# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Adding and running jobs through a real relion_pipeliner 5.1 (skipped without one on PATH)."""

import stat
import subprocess
import time
from pathlib import Path

import pytest

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import GUI_DIR, Pipeliner, PipelinerError
from cryo_savanna.project.star import write_star
from cryo_savanna.tools.validate import EXTERNAL_SCRIPT, _clean_env, pipeliner_version

_version = pipeliner_version()
needs_relion = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                  reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")


@pytest.fixture
def project(tmp_path) -> Path:
    write_star(tmp_path / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    script = tmp_path / "ext.sh"
    script.write_text(EXTERNAL_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return tmp_path


def external(project: Path, in_mic: str = "", **params: str) -> JobStar:
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": str(project / "ext.sh"), "do_queue": "No", "nr_threads": "1", "in_mic": in_mic})
    for i, (label, value) in enumerate(params.items(), start=1):
        js.options[f"param{i}_label"], js.options[f"param{i}_value"] = label, value
    return js


def wait_status(project: Path, job: str, statuses: set[str], timeout: float = 90) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = Pipeline.read(project).processes[job].status
        if status in statuses:
            return status
        time.sleep(0.5)
    raise AssertionError(f"{job} still {Pipeline.read(project).processes[job].status}")


def pipeliner(project: Path) -> Pipeliner:
    return Pipeliner(project, environ=_clean_env())


@needs_relion
def test_add_with_alias_and_run(project):
    pl = pipeliner(project)
    res = pl.add_job(external(project, sleep="1"), alias="first")
    assert res.job == "External/job001/" and not res.warnings
    pr = Pipeline.read(project).processes[res.job]
    assert (pr.status, pr.alias) == ("Scheduled", "External/first/")
    assert not list((project / GUI_DIR / "tmp").iterdir())  # temporary job.star removed

    rec = pl.run([res.job], sec_wait_after=1)
    assert wait_status(project, res.job, {"Succeeded", "Failed"}) == "Succeeded"
    assert [r.id for r in pl.runs()] == [rec.id]
    deadline = time.time() + 30
    while rec.is_alive() and time.time() < deadline:
        time.sleep(0.5)
    assert rec.is_alive() is False
    assert (project / rec.log).exists()


@needs_relion
def test_chain_waits_for_inputs(project):
    pl = pipeliner(project)
    a = pl.add_job(external(project, sleep="2")).job
    b = pl.add_job(external(project, in_mic=f"{a}micrographs.star", sleep="1")).job
    assert Pipeline.read(project).parents(b) == [a]
    pl.run([a, b], sec_wait_after=1)
    assert wait_status(project, b, {"Succeeded", "Failed"}) == "Succeeded"


@needs_relion
def test_continue_and_abort(project):
    pl = pipeliner(project)
    job = pl.add_job(external(project, sleep="1")).job
    pl.run([job], sec_wait_after=1)
    wait_status(project, job, {"Succeeded"})
    js = JobStar.read(project / job / "job.star")
    js.options["param1_value"] = "60"  # sleep long enough to abort
    pl.continue_job(job, js, sec_wait_after=1)
    assert JobStar.read(project / job / "job.star").is_continue
    wait_status(project, job, {"Running"})
    pl.abort(job)
    assert wait_status(project, job, {"Aborted", "Succeeded"}) == "Aborted"


@needs_relion
def test_clean(project):
    pl = pipeliner(project)
    job = pl.add_job(external(project, sleep="0")).job
    pl.run([job], sec_wait_after=1)
    wait_status(project, job, {"Succeeded"})
    pl.clean(job)  # gentle clean of a finished job succeeds


@needs_relion
def test_overwrite_keeps_the_job_number_and_its_users(project):
    pl = pipeliner(project)
    a = pl.add_job(external(project, sleep="0"), alias="first").job
    b = pl.add_job(external(project, in_mic=f"{a}micrographs.star", sleep="0")).job
    pl.run([a, b], sec_wait_after=1)
    wait_status(project, b, {"Succeeded"})
    counter = Pipeline.read(project).job_counter

    res = pl.overwrite_job(a, external(project, sleep="2"), alias="first")
    assert res.job == a and not res.warnings
    p = Pipeline.read(project)
    assert p.job_counter == counter  # back where it was
    assert (p.processes[a].status, p.processes[a].alias) == ("Scheduled", "External/first/")
    assert JobStar.read(project / a / "job.star").options["param1_value"] == "2"
    assert (project / "Trash" / a / "job.star").exists()  # the old job, as RELION's GUI does
    assert (project / "External/first").is_symlink()
    assert p.parents(b) == [a]  # still linked to the new job's output of the same name
    pl.run([a], sec_wait_after=1)
    assert wait_status(project, a, {"Succeeded", "Failed"}) == "Succeeded"


def test_jobs_sharing_a_number_after_an_overwrite_are_found():
    """A job added elsewhere while an overwrite has the counter set back can take a number already in use."""
    from cryo_savanna.project.pipeline import Process
    from cryo_savanna.project.pipeliner import _same_numbers
    p = Pipeline(job_counter=9, processes={n: Process(n, "None", "relion.external", "Succeeded") for n in (
        "Import/job001/", "External/job005/", "Select/job006/", "External/job006/", "Class2D/job008/")})
    assert _same_numbers(p, 5, 9) == [["Select/job006/", "External/job006/"]]
    assert _same_numbers(p, 7, 9) == []


def test_missing_pipeliner_is_reported(project):
    pl = Pipeliner(project, exe=None)
    pl.exe = None
    assert not pl.available
    with pytest.raises(PipelinerError, match="load RELION"):
        pl.add_job(external(project))


def test_repeating_runs_and_stopping_them(project):
    """Found again from their records (also after a restart) while RELION's RUNNING_PIPELINER_ flag is there."""
    import json
    from dataclasses import asdict

    from cryo_savanna.project.pipeliner import RunRecord
    pl = Pipeliner(project, exe="/bin/false", environ={})
    runs = project / GUI_DIR / "runs"
    runs.mkdir(parents=True)
    for rid, repeat in (("a", 1), ("b", 5), ("c", 3)):
        rec = RunRecord(rid, ["Class2D/job002/"], f"cryo-savanna_{rid}", 1, "host", "me", 0.0, [], repeat)
        (runs / f"{rid}.json").write_text(json.dumps(asdict(rec)))
    (project / "RUNNING_PIPELINER_cryo-savanna_b").touch()  # c ended already, a does not repeat
    (project / "RUNNING_PIPELINER_cryo-savanna_a").touch()
    [rec] = pl.repeating_runs()
    assert rec.id == "b"
    assert pl.stop_schedule(rec) and pl.repeating_runs() == []


def test_run_records_are_read_once(project, monkeypatch):
    """The window asks for the repeating runs on every refresh: each record file is parsed only once."""
    import json
    from dataclasses import asdict
    from pathlib import Path as P

    from cryo_savanna.project.pipeliner import RunRecord
    pl = Pipeliner(project, exe="/bin/false", environ={})
    runs = project / GUI_DIR / "runs"
    runs.mkdir(parents=True)
    rec = RunRecord("a", ["Class2D/job002/"], "cryo-savanna_a", 1, "host", "me", 0.0, [], 3)
    (runs / "a.json").write_text(json.dumps(asdict(rec)))
    (runs / "b.json").write_text("{half a rec")  # still being written: tried again next time
    assert [r.id for r in pl.runs()] == ["a"]
    reads = []
    read_text = P.read_text
    monkeypatch.setattr(P, "read_text", lambda self, *a, **k: reads.append(self.name) or read_text(self, *a, **k))
    (runs / "b.json").write_text(json.dumps(asdict(RunRecord("b", [], "s", 1, "h", "u", 0.0, []))))
    assert [r.id for r in pl.runs()] == ["a", "b"] and reads == ["b.json"]


def test_repeating_runs_that_stopped_are_not_listed(project):
    """RELION leaves RUNNING_PIPELINER_<schedule> behind when a repeating run stops at a failed or
    aborted job: its scheduler log, or a dead relion_pipeliner on this host, tells it ended."""
    import json
    import socket
    from dataclasses import asdict

    from cryo_savanna.project.pipeliner import RunRecord
    pl = Pipeliner(project, exe="/bin/false", environ={})
    runs = project / GUI_DIR / "runs"
    runs.mkdir(parents=True)
    dead = subprocess.Popen(["true"])
    dead.wait()
    for rid, host, pid in (("going", "elsewhere", 1), ("failed", "elsewhere", 1),
                           ("gone", socket.gethostname(), dead.pid)):
        rec = RunRecord(rid, ["Class2D/job002/"], f"cryo-savanna_{rid}", pid, host, "me", 0.0, [], 3)
        (runs / f"{rid}.json").write_text(json.dumps(asdict(rec)))
        (project / f"RUNNING_PIPELINER_cryo-savanna_{rid}").touch()
    (project / "pipeline_cryo-savanna_failed.log").write_text(
        " + Stopping pipeliner due to a job that failed with an error ...\n")
    assert [r.id for r in pl.repeating_runs()] == ["going"]


def test_a_run_whose_process_number_was_given_out_again_is_not_alive(tmp_path):
    """After a restart a run is found by its process number; another program that got that number since
    (here: this test) is not the run's relion_pipeliner."""
    import os
    import socket
    import subprocess
    import sys

    from cryo_savanna.project.pipeliner import RunRecord, _command_line
    rec = RunRecord("a", ["Class2D/job002/"], "cryo-savanna_a", os.getpid(), socket.gethostname(), "me", 0.0, [], 3)
    assert rec.is_alive() is False
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "--RunJobs", "Class2D/job002/",
                             "--schedule", "cryo-savanna_a"])
    try:
        deadline = time.time() + 10
        while "--RunJobs" not in (_command_line(proc.pid) or []) and time.time() < deadline:
            time.sleep(0.05)
        mine = RunRecord("a", ["Class2D/job002/"], "cryo-savanna_a", proc.pid, socket.gethostname(), "me", 0.0, [], 3)
        other = RunRecord("b", ["Class2D/job002/"], "cryo-savanna_b", proc.pid, socket.gethostname(), "me", 0.0, [], 3)
        assert mine._is_this_run() and not other._is_this_run()
        assert mine.is_alive() is True
    finally:
        proc.kill()
        proc.wait()
    assert mine.is_alive() is False
