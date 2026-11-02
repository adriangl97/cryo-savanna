# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""SLURM monitor: parsing, mapping SLURM jobs to RELION jobs, lost jobs, cancelling (fake SLURM commands)."""

import json
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.slurm import QueueMapper, SlurmClient, SlurmMonitor, parse_sacct, parse_squeue
from tests.fixtures.fake_slurm import State, install


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "slurm_bin"
    state_file = tmp_path / "slurm_state.json"
    install(bin_dir, state_file)
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_file))
    which = lambda p: str(bin_dir / p) if (bin_dir / p).exists() else None  # noqa: E731
    return State(state_file), SlurmClient(user="me", which=which), bin_dir


def queue_job(project: Path, job: str) -> None:
    """Set "Submit to queue?" in a fixture job's job.star."""
    star = project / job / "job.star"
    star.write_text(re.sub(r"(do_queue\s+)No", r"\1Yes", star.read_text()))


def test_parse_squeue_and_sacct():
    q = parse_squeue("123\tPENDING\tPriority\t0:00\t1-00:00:00\t\tgpu\tClass3D/job009/\t/data/proj\n"
                     "124\tRUNNING\tNone\t1:02:03\t2:00:00\tnode12\tgpu\tRefine3D/job011/\t/data/my proj\n")
    assert [x.short() for x in q] == ["queued (Priority)", "running 1:02:03 on node12"]
    assert q[1].workdir == "/data/my proj" and "of 2:00:00" in q[1].describe()
    acct = parse_sacct("200|Refine3D/job011/|TIMEOUT|2:00:01|2026-10-29T10:00:00|2026-10-29T12:00:01|node3|0:15||"
                       "16|gpu|/data/proj\n"
                       "200.batch|batch|CANCELLED|2:00:01|||node3|0:15|12.5G|16||\n"
                       "200.0|relion_refine_mpi|CANCELLED by 1234|2:00:00|||node3|0:15|31.2G|16||\n"
                       "201|Class3D/job009/|CANCELLED by 1234|0:00:00|Unknown|2026-10-29T09:00:00||0:0||4|gpu|/data/proj\n")
    assert [(a.id, a.state, a.max_rss) for a in acct] == [("200", "TIMEOUT", "31.2G"), ("201", "CANCELLED", "")]
    assert acct[0].ended_badly and not acct[0].active and acct[0].end_time() is not None


def test_maps_many_queued_jobs_and_survives_a_restart(spa_copy, slurm, tmp_path):
    state, client, _ = slurm
    proj = str(spa_copy)
    other = str(tmp_path / "another_project")
    jobs = set(Pipeline.read(spa_copy).processes)
    # 1. named after the job directory (RELION's templates), submitted from this project
    by_name = ["MotionCorr/job002/", "CtfFind/job003/", "AutoPick/job004/", "Extract/job005/", "Class2D/job006/",
               "InitialModel/job008/", "Class3D/job009/", "Refine3D/job011/"]
    ids = {}
    for k, job in enumerate(by_name):
        ids[job] = state.add(name=job, workdir=proj, state="RUNNING" if k % 2 else "PENDING",
                             nodes="node0%d" % k if k % 2 else "")["id"]
    # the same names in another project must not count
    for job in by_name[:3]:
        state.add(name=job, workdir=other)
    # 2. a site template with its own job name: found by the output file
    for job in ("PostProcess/job013/", "CtfRefine/job014/"):
        ids[job] = state.add(name="relion", workdir=proj, stdout=f"{proj}/{job}run.out")["id"]
    # 3. neither: found by sbatch's line in cryo-savanna's run log
    for job in ("Polish/job015/", "Refine3D/job016/"):
        queue_job(spa_copy, job)
    a = state.add(name="relion", workdir=proj)["id"]
    b = state.add(name="relion", workdir=proj)["id"]
    runs = spa_copy / ".cryo-savanna" / "runs"
    runs.mkdir(parents=True)
    (runs / "r1.json").write_text(json.dumps({"jobs": ["MaskCreate/job012/", "Polish/job015/", "Refine3D/job016/"],
                                              "log": ".cryo-savanna/runs/r1.log"}))
    (runs / "r1.log").write_text(f"Submitted batch job {a}\nsome output\nSubmitted batch job {b}\n")
    ids["Polish/job015/"], ids["Refine3D/job016/"] = a, b

    queue = client.queue()
    assert len(queue) == 15
    live = QueueMapper(spa_copy).map(jobs, queue, client.stdout_paths)
    assert {job: q.id for job, q in live.items()} == ids  # 12 jobs, each to its own SLURM job
    assert live["Class2D/job006/"].pending and live["CtfFind/job003/"].nodes == "node01"

    # restart: the ids are remembered, also once the jobs left the queue
    for i in ids.values():
        state.set(i, state="COMPLETED")
    again = QueueMapper(spa_copy)
    assert again.map(jobs, client.queue()) == {}
    assert again.history("PostProcess/job013/") == [ids["PostProcess/job013/"]]


def test_symlinked_project_path_counts_as_the_same_project(spa_copy, slurm, tmp_path):
    state, client, _ = slurm
    link = tmp_path / "link_to_project"
    link.symlink_to(spa_copy)
    i = state.add(name="Class3D/job009/", workdir=str(link))["id"]
    assert QueueMapper(spa_copy).map(set(Pipeline.read(spa_copy).processes), client.queue())["Class3D/job009/"].id == i


def test_finds_jobs_that_ended_in_slurm_without_telling_relion(spa_copy, slurm):
    state, client, _ = slurm
    job = "Polish/job015/"  # Running in the pipeline
    queue_job(spa_copy, job)
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    monitor = SlurmMonitor(spa_copy, client)
    i = state.add(name=job, workdir=str(spa_copy), state="RUNNING", nodes="node7")["id"]
    statuses = {n: pr.status for n, pr in Pipeline.read(spa_copy).processes.items()}
    s = monitor.poll(statuses)
    assert s.live[job].id == i and not s.lost and s.counts() == (0, 1)
    state.set(i, state="TIMEOUT", end=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 600)),
              max_rss="40G")
    s = monitor.poll(statuses)
    assert job not in s.live and s.lost[job].state == "TIMEOUT"
    # just ended: give RELION's exit file a moment before calling it lost
    state.set(i, end=time.strftime("%Y-%m-%dT%H:%M:%S"))
    assert job not in SlurmMonitor(spa_copy, client).poll(statuses).lost
    assert [a.id for a in monitor.history(job)] == [i]


def test_sacct_is_not_asked_on_every_poll(spa_copy, slurm, monkeypatch):
    """A lost job stays lost without asking slurmdbd again; a job still active in sacct is asked again later."""
    state, client, _ = slurm
    job = "Polish/job015/"
    queue_job(spa_copy, job)
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    i = state.add(name=job, workdir=str(spa_copy), state="TIMEOUT",
                  end=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 600)))["id"]
    asked = []
    for name in ("accounting", "accounting_by_name"):
        real = getattr(client, name)
        monkeypatch.setattr(client, name, lambda ids, real=real: asked.append(ids) or real(ids))
    monitor = SlurmMonitor(spa_copy, client)
    statuses = {n: pr.status for n, pr in Pipeline.read(spa_copy).processes.items()}
    for _ in range(3):
        assert monitor.poll(statuses).lost[job].id == i
    assert len(asked) == 1
    # still active in sacct, not in squeue (requeued): asked again only after LOST_RECHECK_S
    state.set(i, state="REQUEUED", end="Unknown")
    statuses[job] = "Succeeded"
    monitor.poll(statuses)  # not a suspect any more: forgotten
    statuses[job] = "Running"
    asked.clear()
    for _ in range(3):
        assert job not in monitor.poll(statuses).lost
    assert len(asked) == 1
    later = time.time() + 1000
    monkeypatch.setattr("cryo_savanna.project.slurm.time.time", lambda: later)
    monitor.poll(statuses)
    assert len(asked) == 2


def test_history_by_name_for_jobs_never_seen(spa_copy, slurm):
    state, client, _ = slurm
    i = state.add(name="Refine3D/job011/", workdir=str(spa_copy), state="COMPLETED", elapsed="5:00:00",
                  end="2026-10-01T10:00:00")["id"]
    state.add(name="Refine3D/job011/", workdir="/elsewhere", state="COMPLETED")
    monitor = SlurmMonitor(spa_copy, client)
    assert [a.id for a in monitor.history("Refine3D/job011/")] == [i]
    assert SlurmMonitor(spa_copy, client).mapper.history("Refine3D/job011/") == [i]  # remembered


def test_cancel_and_no_slurm(slurm, spa_copy):
    state, client, _ = slurm
    i = state.add(name="Class3D/job009/", workdir=str(spa_copy))["id"]
    client.cancel([i])
    assert state.cancelled() == [i] and state.job(i)["state"] == "CANCELLED"
    none = SlurmClient(which=lambda p: None)
    s = SlurmMonitor(spa_copy, none).poll({})
    assert not s.available and not s.live


# ---------------------------------------------------------------------------- with relion_pipeliner

from cryo_savanna.tools.validate import EXTERNAL_SCRIPT, pipeliner_version  # noqa: E402

_version = pipeliner_version()
needs_relion = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                  reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=XXXnameXXX
#SBATCH --output=XXXoutfileXXX
#SBATCH --error=XXXerrfileXXX
#SBATCH --ntasks=XXXmpinodesXXX
XXXcommandXXX
"""


@needs_relion
def test_job_submitted_by_relion_pipeliner_is_found(tmp_path, slurm, monkeypatch):
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.pipeliner import Pipeliner
    from cryo_savanna.project.star import write_star
    state, client, bin_dir = slurm
    project = tmp_path / "proj"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    script = project / "ext.sh"
    script.write_text(EXTERNAL_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (project / "template.sh").write_text(TEMPLATE)
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": str(script), "param1_label": "sleep", "param1_value": "3", "do_queue": "Yes",
                       "qsub": str(bin_dir / "sbatch"), "qsubscript": "template.sh", "queuename": "gpu"})
    monkeypatch.setenv("FAKE_SBATCH_HOLD", "1")  # stays queued until the test lets it run
    pl = Pipeliner(project, environ={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    job = pl.add_job(js).job
    rec = pl.run([job], sec_wait_after=1)
    monitor = SlurmMonitor(project, client)
    deadline = time.time() + 60
    s = monitor.poll({job: "Running"})
    while job not in s.live and time.time() < deadline:
        time.sleep(0.5)
        s = monitor.poll({job: "Running"})
    assert s.live[job].pending and s.live[job].name == job
    assert "Submitted batch job" in (project / rec.log).read_text()
    # cancel it like the Abort action does for a queued job
    client.cancel([s.live[job].id])
    (project / job / "RELION_JOB_EXIT_ABORTED").touch()
    while Pipeline.read(project).processes[job].status != "Aborted" and time.time() < deadline:
        time.sleep(0.5)
    assert Pipeline.read(project).processes[job].status == "Aborted"
    assert job not in monitor.poll({job: "Aborted"}).live
    shutil.rmtree(project / ".cryo-savanna", ignore_errors=True)


def test_readonly_remembers_nothing_in_the_project(spa_copy, slurm):
    """--readonly: the SLURM ids are only kept for the session, no .cryo-savanna/slurm.json."""
    state, client, _ = slurm
    i = state.add(name="Refine3D/job011/", workdir=str(spa_copy), state="COMPLETED", end="2026-10-01T10:00:00")["id"]
    monitor = SlurmMonitor(spa_copy, client, persist=False)
    assert [a.id for a in monitor.history("Refine3D/job011/")] == [i]
    assert monitor.mapper.history("Refine3D/job011/") == [i]
    assert not (spa_copy / ".cryo-savanna" / "slurm.json").exists()


def test_array_job_ids_are_remembered(tmp_path):
    m = QueueMapper(tmp_path)
    m.known.update({"124_3": "Class2D/job005/", "123": "Class2D/job004/", "124_[4-6]": "Class2D/job005/"})
    m._save()
    saved = json.loads((tmp_path / ".cryo-savanna" / "slurm.json").read_text())["jobs"]
    assert list(saved) == ["123", "124_3", "124_[4-6]"]
    assert m.history("Class2D/job005/") == ["124_3", "124_[4-6]"]


def test_ids_found_by_another_session_are_kept(tmp_path):
    """Two windows on one project both save slurm.json: neither loses the ids the other found."""
    a, b = QueueMapper(tmp_path), QueueMapper(tmp_path)
    a.known["101"] = "Class2D/job004/"
    a._save()
    b.known["102"] = "Class3D/job009/"  # b read the file before a saved
    b._save()
    saved = json.loads((tmp_path / ".cryo-savanna" / "slurm.json").read_text())["jobs"]
    assert saved == {"101": "Class2D/job004/", "102": "Class3D/job009/"}
    assert b.history("Class2D/job004/") == ["101"]


def _submit_like_relion(project: Path, job: str, bin_dir: Path, monkeypatch) -> str:
    """What relion_pipeliner does for a queued job: <job>run_submit.script from a template without --job-name
    (SLURM then names the job after the script), "sbatch <job>run_submit.script" in the project, note.txt."""
    queue_job(project, job)
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (project / job / f).unlink(missing_ok=True)
    script = project / job / "run_submit.script"
    script.write_text(f"#!/bin/bash\n#SBATCH -o {job}run.out\nsleep 1\n")
    an_hour_ago = time.time() - 3600
    os.utime(script, (an_hour_ago, an_hour_ago))
    with open(project / job / "note.txt", "a") as fh:
        fh.write(f" ++++ Executing new job on {time.strftime('%a %b %d %H:%M:%S %Y', time.localtime(an_hour_ago))}\n")
    monkeypatch.setenv("FAKE_SBATCH_HOLD", "1")
    r = subprocess.run([str(bin_dir / "sbatch"), f"{job}run_submit.script"], cwd=project, capture_output=True, text=True)
    return re.search(r"Submitted batch job (\d+)", r.stdout).group(1)


def test_history_by_submit_line_when_the_template_names_no_job(spa_copy, slurm, tmp_path, monkeypatch):
    """A template without --job-name=XXXnameXXX: every SLURM job is "run_submit.script". A job that SLURM ended
    while no cryo-savanna looked is still found, by sacct's SubmitLine, and flagged; others' jobs are not taken."""
    state, client, bin_dir = slurm
    job, other = "Polish/job015/", "Refine3D/job016/"
    mine = _submit_like_relion(spa_copy, job, bin_dir, monkeypatch)
    _submit_like_relion(spa_copy, other, bin_dir, monkeypatch)  # another job of this project
    elsewhere = tmp_path / "another_project"
    (elsewhere / job).mkdir(parents=True)
    state.add(name="run_submit.script", workdir=str(elsewhere), state="TIMEOUT",
              submit_line=f"sbatch {job}run_submit.script")  # the same job directory in another project
    assert state.job(mine)["name"] == "run_submit.script"
    ended = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 600))
    for i in [j["id"] for j in state.load()["jobs"]]:
        state.set(i, state="TIMEOUT", end=ended)
    monitor = SlurmMonitor(spa_copy, client)  # opened after the job left the queue
    s = monitor.poll({job: "Running"})
    assert s.lost[job].id == mine and s.lost[job].state == "TIMEOUT"
    assert [a.id for a in monitor.history(job)] == [mine]
    assert SlurmMonitor(spa_copy, client).mapper.history(job) == [mine]  # remembered: sacct asked by id next time


def test_older_sacct_without_submit_line(spa_copy, slurm, monkeypatch):
    """An sacct that does not know SubmitLine: no history for such a job, no error, and it is not asked again."""
    state, client, bin_dir = slurm
    job = "Polish/job015/"
    _submit_like_relion(spa_copy, job, bin_dir, monkeypatch)
    for j in state.load()["jobs"]:
        state.set(j["id"], state="TIMEOUT", end="2026-10-01T10:00:00")
    monkeypatch.setenv("FAKE_SACCT_OLD", "1")
    asked = []
    real = client.submitted_since
    monkeypatch.setattr(client, "submitted_since", lambda since: asked.append(since) or real(since))
    monitor = SlurmMonitor(spa_copy, client)
    assert monitor.history(job) == [] and monitor.history(job) == []
    s = monitor.poll({job: "Running"})
    assert not s.error and job not in s.lost and len(asked) == 1
