# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The SLURM monitor in the window: Queue column, cards, overview, status bar, abort (fake SLURM)."""

import re
import time

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings  # noqa: E402
from qtpy.QtWidgets import QMessageBox  # noqa: E402

from cryo_savanna.env.relion_env import RelionEnv  # noqa: E402
from cryo_savanna.jobdefs.flags import FlagSource  # noqa: E402
from cryo_savanna.project.slurm import SlurmClient  # noqa: E402
from cryo_savanna.ui.job_tree import COL_QUEUE  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402
from tests.fixtures.fake_slurm import State, install  # noqa: E402


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "slurm_bin"
    state_file = tmp_path / "slurm_state.json"
    install(bin_dir, state_file)
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_file))
    return State(state_file), SlurmClient(user="me", which=lambda p: str(bin_dir / p) if (bin_dir / p).exists() else None)


@pytest.fixture
def window(qtbot, tmp_path, slurm):
    def make(project):
        w = MainWindow(project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat), queue_poll_s=1000,
                       slurm=slurm[1])
        qtbot.addWidget(w)
        w.load_now()
        return w
    return make


def queue_text(w, job):
    return w.tree._items[job][COL_QUEUE].text()


def test_queue_shows_in_tree_cards_overview_and_status_bar(window, spa_copy, slurm):
    state, _ = slurm
    proj = str(spa_copy)
    jobs = ["MotionCorr/job002/", "CtfFind/job003/", "AutoPick/job004/", "Extract/job005/", "Class2D/job006/",
            "Select/job007/", "InitialModel/job008/", "Class3D/job009/", "Select/job010/", "Refine3D/job011/",
            "MaskCreate/job012/", "PostProcess/job013/"]
    for k, job in enumerate(jobs):
        state.add(name=job, workdir=proj, **({"state": "RUNNING", "reason": "None", "nodes": f"node{k:02d}",
                                             "elapsed": "1:02:03"} if k % 3 == 0 else {"reason": "Resources"}))
    w = window(spa_copy)
    assert w.tree.isColumnHidden(COL_QUEUE)  # not shown before SLURM was asked
    s = w.queue_now()
    assert len(s.live) == 12 and s.counts() == (8, 4)
    assert not w.tree.isColumnHidden(COL_QUEUE)
    assert queue_text(w, "CtfFind/job003/") == "queued (Resources)"
    assert queue_text(w, "Refine3D/job011/") == "running 1:02:03 on node09"
    assert queue_text(w, "Import/job001/") == ""
    assert "queued (Resources)" in w.graph.nodes["Class2D/job006/"].job_line()
    w.select_job("Refine3D/job011/")
    assert "running 1:02:03 on node09" in w.info.overview.toPlainText()
    assert "SLURM: 8 queued, 4 running" in w.status_state.text()
    w.search.setText("node09")  # the filter also looks at the queue column
    assert "Refine3D/job011/" in w.tree.visible_jobs() and "Class2D/job018/" not in w.tree.visible_jobs()


def test_abort_cancels_queued_and_running_jobs(window, spa_copy, slurm, monkeypatch):
    state, _ = slurm
    job = "Polish/job015/"  # Running in the pipeline
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    i = state.add(name=job, workdir=str(spa_copy), reason="Priority")["id"]
    w = window(spa_copy)
    w.queue_now()
    w.select_job(job)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    w.actions.abort_job()
    w.actions.wait()
    assert f"scancel {i}" in asked[0]
    assert state.cancelled() == [i] and (spa_copy / job / "RELION_JOB_EXIT_ABORTED").exists()

    # running in SLURM: a choice between RELION's abort file and scancel
    (spa_copy / job / "RELION_JOB_EXIT_ABORTED").unlink()
    j = state.add(name=job, workdir=str(spa_copy), state="RUNNING", nodes="node3")["id"]
    w.queue_now()
    monkeypatch.setattr(w.actions, "_choose_abort", lambda job, q: "stop")
    w.actions.abort_job()
    w.actions.wait()
    assert (spa_copy / job / "RELION_JOB_ABORT_NOW").exists() and state.cancelled() == [i]
    monkeypatch.setattr(w.actions, "_choose_abort", lambda job, q: "cancel")
    w.actions.abort_job()
    w.actions.wait()
    assert state.cancelled() == [i, j] and (spa_copy / job / "RELION_JOB_EXIT_ABORTED").exists()


def test_job_that_ended_in_slurm_without_telling_relion(window, spa_copy, slurm):
    state, client = slurm
    job = "Polish/job015/"
    star = spa_copy / job / "job.star"
    star.write_text(re.sub(r"(do_queue\s+)No", r"\1Yes", star.read_text()))
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (spa_copy / job / f).unlink(missing_ok=True)
    state.add(name=job, workdir=str(spa_copy), state="OUT_OF_MEMORY", elapsed="3:10:00", exit_code="0:125",
              end=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600)), max_rss="250G",
              start="2026-10-29T08:00:00")
    w = window(spa_copy)
    s = w.queue_now()
    assert s.lost[job].state == "OUT_OF_MEMORY"
    assert queue_text(w, job) == "out of memory in SLURM (1000)"
    w.select_job(job)
    w.history_now(job)
    text = w.info.overview.toPlainText()
    assert "Mark as failed" in text and "250G" in text and "OUT_OF_MEMORY" in text
    assert "ended without telling RELION" in w.status_state.text()

    # a new window (restart) still knows the job's SLURM history
    w2 = window(spa_copy)
    w2.queue_now()
    w2.select_job(job)
    w2.history_now(job)
    assert "250G" in w2.info.overview.toPlainText()


def test_no_slurm_no_queue_column(qtbot, tmp_path, spa_project):
    w = MainWindow(spa_project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat),
                   slurm=SlurmClient(which=lambda p: None))
    qtbot.addWidget(w)
    w.load_now()
    assert w.tree.isColumnHidden(COL_QUEUE) and not w.queue_timer.isActive()
    assert "SLURM" not in w.status_state.text()


def test_background_polling_and_history(window, spa_copy, slurm, qtbot):
    state, _ = slurm
    i = state.add(name="Refine3D/job011/", workdir=str(spa_copy), state="RUNNING", nodes="node5")["id"]
    w = window(spa_copy)
    w._poll_queue()  # what the timer does
    qtbot.waitUntil(lambda: w.queue_state.checked_at > 0, timeout=10000)
    assert queue_text(w, "Refine3D/job011/").startswith("running")
    w.select_job("Refine3D/job011/")  # asks sacct in the background
    qtbot.waitUntil(lambda: isinstance(w.info.history.get("Refine3D/job011/"), list), timeout=10000)
    assert [a.id for a in w.info.history["Refine3D/job011/"]] == [i]
    assert "SLURM jobs (1)" in w.info.overview.toPlainText()


def _ended_in_slurm(project, state, job, slurm_state, submitted_ago=7200, ended_ago=3600):
    """A queued job that SLURM ended (``slurm_state``) without RELION's exit file; sent to the queue
    ``submitted_ago`` seconds ago (run_submit.script), ended ``ended_ago`` seconds ago."""
    import os

    from cryo_savanna.project.lock import PipelineLock
    from cryo_savanna.project.pipeline import Pipeline
    with PipelineLock(project) as lock:  # Running in the pipeline, as RELION leaves it
        pipeline = Pipeline.read(project)
        pipeline.processes[job].status = "Running"
        pipeline.write(project, lock=lock)
    star = project / job / "job.star"
    star.write_text(re.sub(r"(do_queue\s+)No", r"\1Yes", star.read_text()))
    for f in ("RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"):
        (project / job / f).unlink(missing_ok=True)
    script = project / job / "run_submit.script"
    script.write_text("#!/bin/bash\n")
    os.utime(script, (time.time() - submitted_ago,) * 2)
    return state.add(name=job, workdir=str(project), state=slurm_state, exit_code="0:125",
                     end=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - ended_ago)))["id"]


def test_a_job_slurm_ended_is_marked_failed(window, spa_copy, slurm):
    """SLURM ended it (out of memory, cancelled) and RELION was not told: the exit file its program would have
    written is written (FAILURE, ABORTED when cancelled), with a line in run.err, and the job shows as such."""
    state, client = slurm
    oom, cancelled = "Polish/job015/", "CtfRefine/job014/"
    i = _ended_in_slurm(spa_copy, state, oom, "OUT_OF_MEMORY")
    _ended_in_slurm(spa_copy, state, cancelled, "CANCELLED")
    w = window(spa_copy)
    assert set(w.queue_now().lost) == {oom, cancelled}
    assert w.actions.wait(10000)
    assert (spa_copy / oom / "RELION_JOB_EXIT_FAILURE").exists()
    assert (spa_copy / cancelled / "RELION_JOB_EXIT_ABORTED").exists()
    assert f"SLURM job {i} ended OUT_OF_MEMORY" in (spa_copy / oom / "run.err").read_text()
    w.load_now()
    assert (w.snapshot.jobs[oom].status, w.snapshot.jobs[cancelled].status) == ("Failed", "Aborted")
    assert not w.queue_now().lost  # finished now, as far as RELION is concerned


def test_read_only_and_earlier_runs_are_not_marked(window, spa_copy, slurm, qtbot, tmp_path):
    """Read-only: nothing is written (the Queue column still says it). A SLURM job that ended before the job was
    last sent to the queue (an earlier run's) is not taken for the run RELION waits for."""
    state, client = slurm
    job, rerun = "Polish/job015/", "CtfRefine/job014/"
    _ended_in_slurm(spa_copy, state, job, "TIMEOUT")
    _ended_in_slurm(spa_copy, state, rerun, "TIMEOUT", submitted_ago=60, ended_ago=3600)  # sent again since
    w = MainWindow(spa_copy, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "ro.ini"), QSettings.IniFormat), queue_poll_s=1000,
                   slurm=client, readonly=True)
    qtbot.addWidget(w)
    w.load_now()
    s = w.queue_now()
    assert job in s.lost and rerun not in s.lost
    assert w.actions.wait(10000)
    assert not any((spa_copy / j / f).exists() for j in (job, rerun)
                   for f in ("RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"))
    assert queue_text(w, job) == "timeout in SLURM (1000)"


def test_a_refine_job_that_never_started_can_be_marked_failed_and_overwritten(window, spa_copy, slurm, monkeypatch):
    """RELION refuses to mark a Refine3D job without any iteration (and leaves it Running, so only Delete is left).
    Here Job → Mark as failed works, the question says it is still queued, and Overwrite is offered after it."""
    state, _ = slurm
    job = "Refine3D/job016/"
    _ended_in_slurm(spa_copy, state, job, "PENDING")  # Running in the pipeline, waiting in the queue
    state.set(next(j["id"] for j in state.load()["jobs"] if j["name"] == job), end="Unknown")
    w = window(spa_copy)
    w.queue_now()
    w.select_job(job)
    assert not w.actions.act_overwrite.isEnabled()
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: asked.append(("warning", a[2])))
    w.actions.mark(failed=True)
    assert w.actions.wait(10000)
    assert "still in the SLURM queue" in asked[0] and not any(isinstance(x, tuple) for x in asked)
    w.load_now()
    w.select_job(job)
    assert w.snapshot.jobs[job].status == "Failed" and w.actions.act_overwrite.isEnabled()
