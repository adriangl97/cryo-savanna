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
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.pipeliner import Pipeliner
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
    while time.time() < deadline:
        found = P.job_processes(project, job)
        if len(found) >= n:
            return found
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
