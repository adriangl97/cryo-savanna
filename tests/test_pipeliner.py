# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Adding and running jobs through a real relion_pipeliner 5.1 (skipped without one on PATH)."""

import os
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


# ---------------------------------------------------------------------------- runs still going

def _live_run(project: Path, rid: str, jobs: list[str]):
    """A run record whose relion_pipeliner is a process of this test (alive until killed)."""
    import json
    import socket
    import sys
    from dataclasses import asdict

    from cryo_savanna.project.pipeliner import RunRecord
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "--RunJobs", " ".join(jobs),
                             "--schedule", "cryo-savanna"])
    rec = RunRecord(rid, jobs, "cryo-savanna", proc.pid, socket.gethostname(), "me", time.time(), [])
    runs = project / GUI_DIR / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{rid}.json").write_text(json.dumps(asdict(rec)))
    return proc


def test_the_jobs_of_runs_still_going_are_known(spa_copy):
    waiting = _live_run(spa_copy, "a", ["Refine3D/job016/"])
    going = _live_run(spa_copy, "b", ["Polish/job015/", "Refine3D/job016/", "Class2D/job006/"])
    try:
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        held = pl.held_jobs()
        # the newer run wins; Class2D/job006/ succeeded: that run is past it
        assert sorted(held) == ["Polish/job015/", "Refine3D/job016/"]
        assert held["Polish/job015/"].current and not held["Polish/job015/"].waiting  # it runs
        assert not held["Refine3D/job016/"].current  # after the job that runs now
        going.kill()
        going.wait()
        held = pl.held_jobs()
        assert list(held) == ["Refine3D/job016/"] and held["Refine3D/job016/"].waiting
        assert held["Refine3D/job016/"].waiting_for == ["Polish/job015/shiny.star"]
        assert [(r.id, kind) for r, kind in pl.stoppable_runs()] == [("a", "waiting")]
    finally:
        for p in (waiting, going):
            p.kill()
            p.wait()


def test_a_job_in_a_run_still_going_is_not_run_overwritten_or_continued_again(spa_copy):
    """Two pipeliners for one job would both run it in its directory and both set its status; an overwrite would
    take its name away while the run looks it up."""
    proc = _live_run(spa_copy, "a", ["Refine3D/job016/"])
    try:
        pl = Pipeliner(spa_copy, exe="/bin/false", environ={})
        js = JobStar(type_label="relion.refine3d", is_continue=False, is_tomo=False, options={})
        before = (spa_copy / "default_pipeline.star").read_bytes()
        for attempt in (lambda: pl.run(["Refine3D/job016/"]), lambda: pl.continue_job("Refine3D/job016/", js),
                        lambda: pl.overwrite_job("Refine3D/job016/", js)):
            with pytest.raises(PipelinerError, match="Already in a run that is still going"):
                attempt()
        assert (spa_copy / "default_pipeline.star").read_bytes() == before
        assert not (spa_copy / "Trash").exists() or not (spa_copy / "Trash/Refine3D/job016").exists()
    finally:
        proc.kill()
        proc.wait()


def test_a_run_waiting_for_its_inputs_is_ended_under_the_lock(spa_copy, monkeypatch):
    proc = _live_run(spa_copy, "a", ["Refine3D/job016/"])
    try:
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        [(rec, kind)] = pl.stoppable_runs()
        assert kind == "waiting"
        seen = []
        kill = os.kill

        def kill_seeing_the_lock(pid, sig):
            seen.append((spa_copy / ".relion_lock").is_dir())
            kill(pid, sig)
        monkeypatch.setattr(os, "kill", kill_seeing_the_lock)
        pl.end_run(rec)
        assert True in seen  # the signal went while cryo-savanna held RELION's lock
        assert proc.poll() is not None
        assert not (spa_copy / ".relion_lock").exists()
        assert Pipeline.read(spa_copy).processes["Refine3D/job016/"].status == "Scheduled"
        assert pl.held_jobs() == {}
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture
def non_english_dates():
    """LC_TIME of a non-English locale, as Qt sets it from the user's (setlocale(LC_ALL, "") on start)."""
    import locale
    old = locale.setlocale(locale.LC_TIME)
    for name in ("sv_SE.UTF-8", "de_DE.UTF-8", "es_ES.UTF-8", "fr_FR.UTF-8"):
        try:
            locale.setlocale(locale.LC_TIME, name)
        except locale.Error:
            continue
        if time.strftime("%a %b", (2026, 3, 5, 0, 0, 0, 3, 64, -1)) != "Thu Mar":
            break
    else:
        locale.setlocale(locale.LC_TIME, old)
        pytest.skip("no non-English locale installed")
    yield
    locale.setlocale(locale.LC_TIME, old)


def test_the_note_of_a_job_done_here_has_relions_dates_whatever_the_locale(spa_copy, non_english_dates):
    from cryo_savanna.project.summary import last_start
    job = next(n for n, pr in Pipeline.read(spa_copy).processes.items() if pr.status == "Succeeded")
    before = time.time()
    Pipeliner(spa_copy, exe="/nonexistent").finish_done_here(job, lambda _job: None, "relion_display ...")
    started = last_start(spa_copy / job / "note.txt")  # " ++++ Executing new job on Thu Mar  5 10:00:00 2026"
    assert started is not None and before - 2 <= started <= time.time() + 1


def test_a_run_that_has_not_started_its_job_can_be_stopped(spa_copy):
    """At a Scheduled job with its inputs there and no GPU or thread awaited ("Scheduled · in a run"), but not
    started: its relion_pipeliner is stalled, or starts it at any moment. Stop runs ends it; the job stays
    Scheduled."""
    job = "Refine3D/job016/"
    (spa_copy / "Polish/job015/shiny.star").write_text("")  # its input: nothing to wait for
    proc = _live_run(spa_copy, "a", [job])
    try:
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        h = pl.held_jobs()[job]
        assert h.current and not h.waiting and h.not_started and h.stoppable and "not started yet" in h.describe()
        [(rec, kind)] = pl.stoppable_runs()
        assert kind == "waiting" and pl.current_job(rec) == job
        pl.end_run(rec)
        assert proc.poll() is not None and pl.held_jobs() == {}
        assert Pipeline.read(spa_copy).processes[job].status == "Scheduled"
    finally:
        proc.kill()
        proc.wait()


def test_a_run_ended_just_as_it_started_its_job_says_so(spa_copy):
    from tests.test_processes import start, wait_found
    job = "Refine3D/job016/"
    (spa_copy / "Polish/job015/shiny.star").write_text("")
    proc = _live_run(spa_copy, "a", [job])
    sh = start(spa_copy, job)  # started by the run just then (the pipeline does not say Running yet)
    try:
        wait_found(spa_copy, job, 1)
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        [(rec, _)] = pl.stoppable_runs()
        with pytest.raises(PipelinerError, match="had just started Refine3D/job016"):
            pl.end_run(rec)
        assert proc.poll() is not None
    finally:
        sh.kill()
        proc.kill()
        proc.wait()


def test_a_run_busy_with_a_job_before_scheduled_ones_is_offered(spa_copy):
    """Its later jobs show "Scheduled · in a run"; it stops once the job it runs is aborted."""
    proc = _live_run(spa_copy, "b", ["Polish/job015/", "Refine3D/job016/"])  # Polish/job015/ is Running
    try:
        pl = Pipeliner(spa_copy, exe="relion_pipeliner", environ={})
        assert not pl.held_jobs()["Refine3D/job016/"].current
        [(rec, kind)] = pl.stoppable_runs()
        assert (rec.id, kind) == ("b", "busy") and pl.current_job(rec) == "Polish/job015/"
    finally:
        proc.kill()
        proc.wait()
