# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Force abort of a job run on this machine (project/processes.py): which processes are the job's, and killing
them, through a stand-in relion_pipeliner (tests/fixtures/fake_pipeliner.py) and from the window."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project import processes as P
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import Pipeliner, PipelinerError
from cryo_savanna.project.star import write_star

UID = os.getuid()
JOB = "Class3D/job009/"


def table(*rows):
    return {pid: P.Proc(pid, ppid, uid, state, args) for pid, ppid, uid, state, args in rows}


def test_the_job_processes_are_found():
    t = table(
        (100, 1, UID, "S", "relion_pipeliner --RunJobs Class3D/job009/ --schedule cryo-savanna"),
        (101, 100, UID, "S", "sh -c mpirun -n 3 relion_refine_mpi --o Class3D/job009/run --j 4 "
                             ">> Class3D/job009/run.out 2>> Class3D/job009/run.err"),
        (102, 101, UID, "S", "mpirun -n 3 relion_refine_mpi --o Class3D/job009/run --j 4"),
        (103, 102, UID, "R", "relion_refine_mpi --o Class3D/job009/run --j 4"),
        (104, 102, UID, "R", "relion_refine_mpi --o Class3D/job009/run --j 4"),
        (105, 102, UID, "Z", "[relion_refine_mp] <defunct>"),  # ended already
        (200, 1, UID, "S", "tail -f Class3D/job009/run.out"),  # only looks at it
        (300, 1, UID, "S", "relion_refine --o Class3D/job0090/run"),  # another job
        (301, 1, UID, "S", "relion_refine --o Other/Class3D/job009/run"),  # another project's
        (400, 1, UID + 1, "S", "relion_refine --o Class3D/job009/run"),  # another user's
        (500, 1, UID, "S", "relion_display --i Class3D/job009/run_it025_class001.mrc"),
    )
    found = P.job_processes("/nonexistent", JOB, [100], table=t)
    assert [p.pid for p in found] == [101, 102, 103, 104]  # not the pipeliner, not the zombie
    # started elsewhere (RELION's GUI): by --o, with everything under it
    t[101] = P.Proc(101, 7, UID, "S", t[101].args)
    t[7] = P.Proc(7, 1, UID, "S", "relion")
    assert sorted(p.pid for p in P.job_processes("/nonexistent", JOB, [], table=t)) == [101, 102, 103, 104]
    # a pipeliner that runs another job: nothing of it
    t[101] = P.Proc(101, 100, UID, "S", "sh -c ext.sh --o External/job010/ >> External/job010/run.out")
    for pid in (102, 103, 104):
        t[pid] = P.Proc(pid, t[pid].ppid, UID, "S", t[pid].args.replace("Class3D/job009/", "External/job010/"))
    assert P.job_processes("/nonexistent", JOB, [100], table=t) == []


def test_ps_lists_this_process():
    me = P.process_table()[os.getpid()]
    assert me.uid == UID and me.ppid == os.getppid() and "python" in me.args.lower()


IGNORES_TERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"


def start(cwd: Path, job: str, ignore_term: bool = False) -> subprocess.Popen:
    """A program of ``job`` as RELION starts it, in a shell (``--o <job>``)."""
    code = IGNORES_TERM if ignore_term else "import time; time.sleep(120)"
    return subprocess.Popen(f"{sys.executable} -c '{code}' --o {job} >> {job}run.out", shell=True, cwd=cwd,
                            start_new_session=True)


def wait_found(project: Path, job: str, n: int, timeout: float = 10) -> list[P.Proc]:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        found = P.job_processes(project, job)
        # the same twice: not the shell caught just before it exec'd the program (kill() takes a process whose
        # command line changed for another one, given that number since, and leaves it alone)
        if len(found) >= n and found == last:
            return found
        last = found
        time.sleep(0.1)
    raise AssertionError(f"{job}: {P.job_processes(project, job)}")


def test_kill_ends_them_and_marks_the_job_aborted(tmp_path):
    job = "External/job001/"
    (tmp_path / job).mkdir(parents=True)
    (tmp_path / job / "RELION_JOB_ABORT_NOW").touch()  # asked to stop before, did not
    sh = start(tmp_path, job, ignore_term=True)
    try:
        found = wait_found(tmp_path, job, 1)
        P.kill(tmp_path, job, found, term_wait_s=0.3)  # SIGTERM ignored: SIGKILL
        assert sh.wait(10) is not None
        assert P.job_processes(tmp_path, job) == []
    finally:
        sh.kill()
    d = tmp_path / job
    assert (d / "RELION_JOB_EXIT_ABORTED").exists() and not (d / "RELION_JOB_ABORT_NOW").exists()
    assert "force abort" in (d / "run.err").read_text()


def test_kill_leaves_a_job_that_ended_meanwhile_as_it_ended(tmp_path):
    job = "External/job001/"
    (tmp_path / job).mkdir(parents=True)
    (tmp_path / job / "RELION_JOB_EXIT_SUCCESS").touch()
    P.kill(tmp_path, job, [P.Proc(2 ** 22 + 7, 1, UID, "S", "gone")])
    assert not (tmp_path / job / "RELION_JOB_EXIT_ABORTED").exists()


def ext_script(project: Path) -> str:
    f = project / "ext.py"  # an External job that does not look at RELION_JOB_ABORT_NOW
    f.write_text(f"#!{sys.executable}\n{IGNORES_TERM.replace('; ', chr(10))}\n")
    f.chmod(0o755)
    return str(f)


def test_force_abort_of_a_run_started_here(tmp_path, monkeypatch):
    from tests.fixtures.fake_pipeliner import install
    install(tmp_path / "bin")
    project = tmp_path / "p"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    environ = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    pl = Pipeliner(project, exe=str(tmp_path / "bin/relion_pipeliner"), environ=environ)
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": ext_script(project), "do_queue": "No", "nr_threads": "1"})
    job = pl.add_job(js).job
    rec = pl.run([job])
    monkeypatch.setattr(P, "TERM_WAIT_S", 0.3)
    try:
        deadline = time.time() + 20
        procs, host = pl.force_abort_targets(job)
        while len(procs) < 2 and time.time() < deadline:  # the shell and the script
            time.sleep(0.1)
            procs, host = pl.force_abort_targets(job)
        assert host == "" and rec.pid not in [p.pid for p in procs] and len(procs) >= 2
        pl.force_abort(job, procs)
        deadline = time.time() + 20
        while rec.is_alive() and time.time() < deadline:  # the pipeliner sees the job aborted and ends
            time.sleep(0.1)
        assert rec.is_alive() is False
        assert Pipeline.read(project).processes[job].status == "Aborted"
    finally:
        for p in P.job_processes(project, job, [rec.pid]):
            os.kill(p.pid, 9)


def test_force_abort_from_the_window(qtbot, tmp_path, spa_copy, monkeypatch):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.ui.main_window import MainWindow
    job = "Polish/job015/"  # Running in the pipeline
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    star = spa_copy / job / "job.star"
    queued = star.read_text()
    star.write_text(re.sub(r"(do_queue\s+)Yes", r"\1No", queued))  # run on a machine
    for box in ("question", "warning", "critical"):  # a dialog not answered here would wait for ever
        monkeypatch.setattr(QMessageBox, box, lambda *args, box=box, **k: pytest.fail(f"{box}: {args[1:]}"))
    w = MainWindow(spa_copy, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.select_job(job)
    a = w.actions
    assert a.act_abort.isEnabled()
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **k: told.append(args[2]))
    monkeypatch.setattr(a, "_choose_local_abort", lambda job: "force")
    a.abort_job()  # nothing of it runs here
    assert "No process" in told[0] and not (spa_copy / job / "RELION_JOB_EXIT_ABORTED").exists()

    sh = start(spa_copy, job)
    try:
        wait_found(spa_copy, job, 2)
        shown = []
        monkeypatch.setattr(a, "_confirm_kill", lambda label, procs: shown.append(procs) or False)
        a.abort_job()  # shown, not killed
        assert sh.poll() is None and {p.pid for p in shown[0]} >= {sh.pid}
        monkeypatch.setattr(a, "_confirm_kill", lambda label, procs: True)
        a.abort_job()
        a.wait()
        assert sh.wait(10) is not None and (spa_copy / job / "RELION_JOB_EXIT_ABORTED").exists()
    finally:
        sh.kill()

    monkeypatch.setattr(a, "_choose_local_abort", lambda job: "stop")
    a.abort_job()
    a.wait()
    assert (spa_copy / job / "RELION_JOB_ABORT_NOW").exists()

    star.write_text(queued)  # in a queue: only asked to stop (its processes are not here)
    (spa_copy / job / "RELION_JOB_ABORT_NOW").unlink()
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **k: asked.append(args[2]) or QMessageBox.Yes)
    monkeypatch.setattr(a, "_choose_local_abort", lambda job: pytest.fail("offered a force abort"))
    a.abort_job()
    a.wait()
    assert asked and (spa_copy / job / "RELION_JOB_ABORT_NOW").exists()


def test_a_process_of_another_project_with_a_job_of_that_name_is_not_the_jobs(tmp_path):
    """Its working directory tells (/proc on Linux, lsof on macOS)."""
    job = "External/job001/"
    here, other = tmp_path / "here", tmp_path / "other"
    for d in (here, other):
        (d / job).mkdir(parents=True)
    sh = start(other, job)
    try:
        found = wait_found(other, job, 1)
        assert P.job_processes(here, job) == []
        P.kill(other, job, found, term_wait_s=0.3)
        assert sh.wait(10) is not None
    finally:
        sh.kill()


LEFT_BEHIND = "sleep 3017"  # what the job leaves in the background: no job directory in its command line


def test_force_abort_frees_the_gpus_of_its_run(tmp_path, monkeypatch):
    """A program the job left in the background keeps the run's GPU locks (gpu_wait.py: inherited) without being
    one of the job's processes (its command line names no job directory, and it is under nothing of the job's):
    force abort kills it too, and the next run gets the GPUs."""
    from cryo_savanna import gpu_wait as W
    from cryo_savanna.project.gpu_queue import GpuQueue
    from tests.fixtures.fake_pipeliner import install

    class OneGpu(GpuQueue):
        def gpus_of(self, jobstars, environ):
            return ["0"]

    install(tmp_path / "bin")
    project = tmp_path / "p"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    environ = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    pl = Pipeliner(project, exe=str(tmp_path / "bin/relion_pipeliner"), environ=environ)
    locks = tmp_path / "locks"
    pl.gpu_queue = OneGpu(locks)
    script = project / "ext.sh"
    script.write_text(f"#!/bin/sh\n({LEFT_BEHIND} > /dev/null 2>&1 &)\n"
                      f"exec {sys.executable} -c '{IGNORES_TERM}'\n")
    script.chmod(0o755)
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": str(script), "do_queue": "No", "nr_threads": "1"})
    job = pl.add_job(js).job
    rec = pl.run([job])
    monkeypatch.setattr(P, "TERM_WAIT_S", 0.3)

    def left_behind():
        return [p for p in P.process_table().values() if p.args == LEFT_BEHIND and p.uid == UID]

    try:
        deadline = time.time() + 20
        while not left_behind() and time.time() < deadline:
            time.sleep(0.1)
        assert left_behind() and W.try_lock(str(locks), ["0"]) is None  # the run has the GPU
        procs, _ = pl.force_abort_targets(job)
        assert {p.pid for p in left_behind()} <= {p.pid for p in procs}  # shown before it is killed
        pl.force_abort(job, procs)
        deadline = time.time() + 20
        held = None
        while held is None and time.time() < deadline:
            held = W.try_lock(str(locks), ["0"])
            time.sleep(0.1)
        assert held is not None, f"GPU 0 still locked by {left_behind()}"
        for fd in held:
            os.close(fd)
        assert not left_behind() and Pipeline.read(project).processes[job].status == "Aborted"
    finally:
        for p in left_behind() + P.job_processes(project, job, [rec.pid]):
            os.kill(p.pid, 9)


def test_a_job_whose_killed_processes_are_slow_to_end_stays_aborted(tmp_path, monkeypatch):
    """A program on a GPU can take long to end after SIGKILL: the job stays marked as aborted (else the
    relion_pipeliner waiting for it, which keeps the run's GPU locks, would wait for ever). Only processes that
    could not be signalled (another user's) leave it unmarked."""
    job = "Class3D/job009/"
    d = tmp_path / job
    d.mkdir(parents=True)
    stuck = P.Proc(2 ** 22 + 7, 1, UID, "D", "relion_refine --o Class3D/job009/run --gpu 0")
    monkeypatch.setattr(P, "_alive", lambda procs: list(procs))
    monkeypatch.setattr(P, "KILL_WAIT_S", 0.2)
    monkeypatch.setattr(P, "_signal", lambda procs, sig: [])
    P.kill(tmp_path, job, [stuck], term_wait_s=0.1)
    assert (d / "RELION_JOB_EXIT_ABORTED").exists() and "still ending" in (d / "run.err").read_text()

    (d / "RELION_JOB_EXIT_ABORTED").unlink()
    monkeypatch.setattr(P, "_signal", lambda procs, sig: list(procs))  # another user's
    with pytest.raises(P.ForceAbortError, match="not yours"):
        P.kill(tmp_path, job, [stuck], term_wait_s=0.1)
    assert not (d / "RELION_JOB_EXIT_ABORTED").exists()


def test_lock_leftovers_are_only_those_of_the_run():
    """Holders of a run's GPU locks other than its pipeliner; none while the pipeliner still waits for them (they
    are another run's then), nor, once it is gone, while another relion_pipeliner has them."""
    t = table(
        (100, 1, UID, "S", "relion_pipeliner --RunJobs Class3D/job009/ --schedule cryo-savanna"),
        (200, 1, UID, "S", "sleep 3017"),  # left in the background by a job of the run
        (201, 200, UID, "S", "child of it"),
        (300, 1, UID, "S", "python3 -I /x/gpu_wait.py --locks /tmp/l --gpus 0 -- relion_pipeliner"),  # waits
    )
    holders = {100, 200, 300}
    import unittest.mock as m
    with m.patch.object(P, "_lock_holders", lambda paths, tbl: set(holders)), \
            m.patch.object(P, "_cwd", lambda pid: None):
        assert [p.pid for p in P.lock_leftovers("/p", ["/l"], 100, t)] == [200, 201]
        holders.discard(100)  # the pipeliner does not have them: it still waits, the holders are another run's
        assert P.lock_leftovers("/p", ["/l"], 100, t) == []
        del t[100]  # the pipeliner is gone
        assert [p.pid for p in P.lock_leftovers("/p", ["/l"], 100, t)] == [200, 201]
        t[400] = P.Proc(400, 1, UID, "S", "relion_pipeliner --RunJobs External/job020/ --schedule cryo-savanna")
        holders.add(400)  # another run got the GPUs since
        assert P.lock_leftovers("/p", ["/l"], 100, t) == []


def test_a_killed_job_of_a_run_started_here_can_then_be_overwritten(tmp_path, monkeypatch):
    """Overwrite of a running job (ui/actions/running.py _kill_to_overwrite): kill it, wait until its
    relion_pipeliner has set it Aborted and ended (wait_released), then overwrite it in its place."""
    from tests.fixtures.fake_pipeliner import install
    install(tmp_path / "bin")
    project = tmp_path / "p"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    environ = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    pl = Pipeliner(project, exe=str(tmp_path / "bin/relion_pipeliner"), environ=environ)
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": ext_script(project), "do_queue": "No", "nr_threads": "1"})
    job = pl.add_job(js).job
    rec = pl.run([job])
    monkeypatch.setattr(P, "TERM_WAIT_S", 0.3)
    try:
        deadline = time.time() + 20
        procs, _ = pl.force_abort_targets(job)
        while len(procs) < 2 and time.time() < deadline:
            time.sleep(0.1)
            procs, _ = pl.force_abort_targets(job)
        assert Pipeline.read(project).processes[job].status == "Running"
        pl.force_abort(job, procs)
        pl.wait_released(job, poll_s=0.1, settle_s=0, release_s=20)  # settle 0: only its run's own end counts
        assert rec.is_alive() is False and Pipeline.read(project).processes[job].status == "Aborted"
        res = pl.overwrite_job(job, js)
        assert res.job == job and Pipeline.read(project).processes[job].status == "Scheduled"
        assert (project / "Trash" / job).is_dir()
    finally:
        for p in P.job_processes(project, job, [rec.pid]):
            os.kill(p.pid, 9)


def test_a_killed_job_no_run_here_runs_is_set_aborted(spa_copy):
    """Started elsewhere (RELION's GUI): nothing here sets it Aborted, so wait_released does after settle_s."""
    job = "Polish/job015/"  # Running in the pipeline, no run recorded
    pl = Pipeliner(spa_copy, exe="/bin/false", environ={})
    pl.wait_released(job, poll_s=0.05, settle_s=0.2)
    assert Pipeline.read(spa_copy).processes[job].status == "Aborted"
    pl.wait_released(job, poll_s=0.05, settle_s=0)  # not Running any more: at once


def test_overwrite_of_a_running_job_from_the_window_kills_it_first(qtbot, tmp_path, spa_copy, monkeypatch):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.project.jobstar import JobStar
    from cryo_savanna.ui.main_window import MainWindow
    job = "Polish/job015/"  # Running in the pipeline
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    star = spa_copy / job / "job.star"
    star.write_text(re.sub(r"(do_queue\s+)Yes", r"\1No", star.read_text()))  # run on a machine
    for box in ("question", "warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, box, lambda *args, box=box, **k: pytest.fail(f"{box}: {args[1:]}"))
    w = MainWindow(spa_copy, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.select_job(job)
    a = w.actions
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **k: told.append(args[2]))
    assert a._kill_to_overwrite(job) is None and "no process of it" in told[0]  # nothing of it runs here

    sh = start(spa_copy, job)
    try:
        wait_found(spa_copy, job, 2)
        asked = []
        monkeypatch.setattr(a, "_confirm_kill", lambda label, procs, **k: asked.append(k) or False)
        assert a._kill_to_overwrite(job) is None and sh.poll() is None  # said no: left running
        assert asked[0]["button"] == "Kill and overwrite"
        monkeypatch.setattr(a, "_confirm_kill", lambda label, procs, **k: True)
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **k: QMessageBox.Yes)
        overwritten = []
        monkeypatch.setattr(a.pipeliner, "wait_released", lambda j: overwritten.append(("released", j)))
        def overwrite_job(j, *rest):
            overwritten.append(("overwrite", j))
            raise PipelinerError("stop here")  # (what an overwrite does is tested in test_pipeliner.py)
        monkeypatch.setattr(a.pipeliner, "overwrite_job", overwrite_job)
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *args, **k: warned.append(args[2]))
        a.editor.job = job
        a._submit_overwrite(JobStar.read(star), "", schedule_only=True)
        a.wait()
        assert sh.wait(10) is not None and (spa_copy / job / "RELION_JOB_EXIT_ABORTED").exists()
        assert overwritten == [("released", job), ("overwrite", job)]  # in that order, after the kill
        qtbot.waitUntil(lambda: bool(warned), timeout=5000)  # (signalled to the window)
        assert warned[0].startswith("stop here")
    finally:
        sh.kill()


def _stuck_run(tmp_path, waits: str):
    """A run whose job ended without RELION's exit file (a crash, the out-of-memory killer) and whose
    relion_pipeliner waits for it (FAKE_PIPELINER_WAITS, as RELION's): what Stop runs could not see."""
    from tests.fixtures.fake_pipeliner import install
    install(tmp_path / "bin")
    project = tmp_path / "p"
    project.mkdir()
    write_star(project / "default_pipeline.star", Pipeline(job_counter=1).to_blocks(), atomic=False)
    environ = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", FAKE_PIPELINER_WAITS=waits)
    pl = Pipeliner(project, exe=str(tmp_path / "bin/relion_pipeliner"), environ=environ)
    crash = project / "crash.sh"
    crash.write_text("#!/bin/sh\nexit 139\n")
    crash.chmod(0o755)
    js = JobDefs.load().variant("relion.external").new_jobstar(RelionEnv({}))
    js.options.update({"fn_exe": str(crash), "do_queue": "No", "nr_threads": "1"})
    job = pl.add_job(js).job
    rec = pl.run([job])
    deadline = time.time() + 20
    while not (project / job / "run.err").exists() and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)  # the job has ended: the pipeliner waits
    return project, pl, job, rec


def _age(project: Path, job: str, s: float = 3600) -> None:
    """Its directory as if nothing changed in it for ``s`` seconds (stuck_runs waits STUCK_GRACE_S)."""
    then = time.time() - s
    for f in (project / job, project / job / "run.out", project / job / "run.err"):
        if f.exists():
            os.utime(f, (then, then))


def test_a_run_waiting_for_a_job_that_ended_without_exit_file_can_be_stopped(tmp_path):
    project, pl, job, rec = _stuck_run(tmp_path, "1")
    try:
        assert rec.is_alive() is True and Pipeline.read(project).processes[job].status == "Running"
        assert pl.stuck_runs() == []  # its directory just changed: perhaps a job being started
        _age(project, job)
        assert pl.stuck_runs() == [(rec, job)]
        assert (rec.id, "stuck") in [(r.id, k) for r, k in pl.stoppable_runs()]
        pl.end_stuck(rec, job, wait_s=20)  # the exit file it waits for: it ends by itself
        assert rec.is_alive() is False and Pipeline.read(project).processes[job].status == "Aborted"
        assert "no process of this job was left" in (project / job / "run.err").read_text()
        assert pl.stuck_runs() == [] and pl.stoppable_runs() == []
    finally:
        if rec.is_alive():
            os.kill(rec.pid, 9)


def test_a_stuck_run_that_does_not_see_the_exit_file_is_ended(tmp_path):
    project, pl, job, rec = _stuck_run(tmp_path, "forever")
    try:
        _age(project, job)
        assert pl.stuck_runs() == [(rec, job)]
        pl.end_stuck(rec, job, wait_s=0.5)
        assert rec.is_alive() is False and Pipeline.read(project).processes[job].status == "Aborted"
        js = JobStar.read(project / job / "job.star")
        assert pl.overwrite_job(job, js).job == job  # and it can be overwritten
    finally:
        if rec.is_alive():
            os.kill(rec.pid, 9)


def test_a_job_with_processes_left_is_not_stuck(tmp_path):
    project, pl, job, rec = _stuck_run(tmp_path, "1")
    sh = start(project, job)  # a program of it still runs (as one started in the background)
    try:
        wait_found(project, job, 1)
        _age(project, job)
        assert pl.stuck_runs() == []
        with pytest.raises(PipelinerError, match="running again"):
            pl.end_stuck(rec, job)
        assert not (project / job / "RELION_JOB_EXIT_ABORTED").exists()
    finally:
        sh.kill()
        if rec.is_alive():
            os.kill(rec.pid, 9)


def test_a_stuck_run_from_the_window(qtbot, tmp_path, monkeypatch):
    """Stop runs lists it (and counts it), and Force abort of its job, which finds no process, offers to end it."""
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QInputDialog, QMessageBox

    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.ui.main_window import MainWindow
    for box in ("question", "warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, box, lambda *args, box=box, **k: pytest.fail(f"{box}: {args[1:]}"))
    runs = []
    for i in (1, 2):
        project, pl, job, rec = _stuck_run(tmp_path / str(i), "1")
        _age(project, job)
        runs.append((project, job, rec))
    try:
        project, job, rec = runs[0]
        w = MainWindow(project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
        qtbot.addWidget(w)
        w.load_now()
        a = w.actions
        a.look_at_project_now()
        assert a.act_stop_runs.isEnabled() and "(1)" in a.act_stop_runs.text()
        shown = []
        monkeypatch.setattr(QInputDialog, "getItem", lambda parent, title, text, items, *r: shown.append(items)
                            or (items[0], True))
        a.stop_runs()
        a.wait()
        assert "is not running any more" in shown[0][0]
        assert rec.is_alive() is False and Pipeline.read(project).processes[job].status == "Aborted"

        project, job, rec = runs[1]  # Force abort: no process of it, its run waits
        w.close()
        w = MainWindow(project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s2.ini"), QSettings.IniFormat))
        qtbot.addWidget(w)
        w.load_now()
        w.select_job(job)
        a = w.actions
        monkeypatch.setattr(a, "_confirm_end_run", lambda label, r, title, button, then="": r.pid == rec.pid)
        a.force_abort(job)
        a.wait()
        assert rec.is_alive() is False and Pipeline.read(project).processes[job].status == "Aborted"
    finally:
        for _, _, r in runs:
            if r.is_alive():
                os.kill(r.pid, 9)
