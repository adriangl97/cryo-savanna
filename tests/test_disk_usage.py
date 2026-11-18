# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""project/disk_usage.py: what a project takes on disk, what RELION's clean moves to Trash/, emptying Trash/."""

import os
import threading

import pytest

from cryo_savanna.project import disk_usage as D
from cryo_savanna.project.pipeline import Node, Pipeline, Process


def _write(path, n=10_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)


def _pipeline(*procs, nodes=()):
    p = Pipeline()
    for name, label, status in procs:
        p.processes[name] = Process(name, "None", label, status)
    for n in nodes:
        p.nodes[n] = Node(n, "ParticleGroupMetadata.star.relion")
    return p


# ---------------------------------------------------------------------------- RELION's cleanupJob

def test_iterations_but_the_last_and_those_in_the_pipeline_are_cleaned(tmp_path):
    job = "Class2D/job006/"
    for it in ("run_it010", "run_it020", "run_it025"):
        for suffix in ("_data.star", "_model.star", "_optimiser.star", "_classes.mrcs"):
            _write(tmp_path / job / (it + suffix), 100)
    _write(tmp_path / job / "run.out", 10)
    p = _pipeline((job, "relion.class2d", "Succeeded"), nodes=[job + "run_it020_optimiser.star"])
    gentle = D.clean_targets(tmp_path, p, job, False)
    assert gentle == [job + f"run_it010{s}" for s in ("_classes.mrcs", "_data.star", "_model.star", "_optimiser.star")]
    assert D.clean_targets(tmp_path, p, job, True) == gentle  # the same for harsh


def test_the_last_iteration_found_is_the_last_glob_as_in_relion(tmp_path):
    """RELION keeps the last _data.star of its four globs: with a continued run, that is run_ctN_..., and the
    run_it... of the first run go (unless the pipeline has them)."""
    job = "Refine3D/job011/"
    for f in ("run_it005_data.star", "run_it010_data.star", "run_ct10_it012_data.star", "run_it012_data.star"):
        _write(tmp_path / job / f, 10)
    p = _pipeline((job, "relion.refine3d", "Succeeded"))
    out = D.clean_targets(tmp_path, p, job, False)
    assert job + "run_it012_data.star" in out and job + "run_ct10_it012_data.star" not in out


def test_motioncorr_gentle_keeps_micrographs_harsh_moves_the_folders(tmp_path):
    job = "MotionCorr/job002/"
    for f in ("Movies/a.mrc", "Movies/a.log", "Movies/a.err", "Movies/sub/b.out", "Movies/sub/b.mrc",
              ".hidden/c.log"):
        _write(tmp_path / job / f, 10)
    os.symlink(tmp_path / job / "Movies", tmp_path / job / "Link")  # a link to a directory is not one of them
    p = _pipeline((job, "relion.motioncorr.own", "Succeeded"))
    assert D.clean_targets(tmp_path, p, job, False) == [job + "Movies/a.err", job + "Movies/a.log",
                                                         job + "Movies/sub/b.out"]
    assert D.clean_targets(tmp_path, p, job, True) == [job + "Movies/", job + "Movies/sub/"]


def test_what_relion_does_not_clean(tmp_path):
    p = _pipeline(("Import/job001/", "relion.import.movies", "Succeeded"),
                  ("Class2D/job003/", "relion.class2d", "Running"),
                  ("PostProcess/job004/", "relion.postprocess", "Succeeded"))
    assert D.clean_targets(tmp_path, p, "Import/job001/", True) == []  # never cleaned
    assert D.clean_targets(tmp_path, p, "Class2D/job003/", False) is None  # only finished jobs
    assert D.clean_targets(tmp_path, p, "PostProcess/job004/", True) == []  # no rule: nothing


def test_extract_and_subtract(tmp_path):
    _write(tmp_path / "Extract/job005/Movies/a.mrcs")
    _write(tmp_path / "Extract/job005/Movies/a_extract.star")
    _write(tmp_path / "Subtract/job007/subtracted.star")
    _write(tmp_path / "Subtract/job007/subtracted.mrcs")
    p = _pipeline(("Extract/job005/", "relion.extract", "Succeeded"), ("Subtract/job007/", "relion.subtract", "Succeeded"))
    assert D.clean_targets(tmp_path, p, "Extract/job005/", False) == ["Extract/job005/Movies/a_extract.star"]
    assert D.clean_targets(tmp_path, p, "Extract/job005/", True) == ["Extract/job005/Movies/"]
    assert D.clean_targets(tmp_path, p, "Subtract/job007/", False) == []
    assert D.clean_targets(tmp_path, p, "Subtract/job007/", True) == ["Subtract/job007/subtracted.mrcs",
                                                                       "Subtract/job007/subtracted.star"]


# ---------------------------------------------------------------------------- the scan

def test_scan_counts_jobs_trash_and_the_rest_once(tmp_path):
    job = "Class2D/job006/"
    _write(tmp_path / job / "run_it010_data.star", 50_000)
    _write(tmp_path / job / "run_it025_data.star", 50_000)
    os.link(tmp_path / job / "run_it025_data.star", tmp_path / job / "hardlink.star")  # counted once
    _write(tmp_path / "raw/movie.tiff", 200_000)
    os.symlink(tmp_path / "raw", tmp_path / "Movies")  # not followed
    _write(tmp_path / "Trash/Class2D/job004/deleted_pipeline.star", 100)  # a deleted job
    _write(tmp_path / "Trash/Class2D/job004/run_it025_classes.mrcs", 100_000)
    _write(tmp_path / f"Trash/{job}run_it005_data.star", 30_000)  # cleaned files of a job in the pipeline
    _write(tmp_path / "Class2D/job099/run.out", 10_000)  # a folder that is not a job of the pipeline
    p = _pipeline((job, "relion.class2d", "Succeeded"))
    u = D.scan(tmp_path, p)
    (j,) = u.jobs
    assert j.usage.files == 3 and 100_000 <= j.usage.bytes < 120_000
    assert j.gentle is not None and 50_000 <= j.gentle.bytes < 60_000 and j.gentle.files == 1
    by_path = {t.path: t for t in u.trash}
    deleted, cleaned = by_path["Trash/Class2D/job004"], by_path["Trash/Class2D/job006"]
    assert deleted.deleted_job and not deleted.in_pipeline and deleted.job == "Class2D/job004/"
    assert not cleaned.deleted_job and cleaned.in_pipeline
    assert u.trash_total.bytes >= 130_000
    other = dict(u.other)
    assert set(other) == {"Class2D/job099", "Movies", "raw"} and other["Movies"].bytes < 10_000
    assert u.total.bytes == u.jobs_total.bytes + u.trash_total.bytes + u.other_total.bytes
    assert u.disk_free is not None and u.by_type() == {"Class2D": (1, j.usage)}


def test_scan_can_be_stopped(tmp_path):
    _write(tmp_path / "Class2D/job001/a")
    stop = threading.Event()
    stop.set()
    with pytest.raises(D.Cancelled):
        D.scan(tmp_path, _pipeline(("Class2D/job001/", "relion.class2d", "Succeeded")), stop)


def test_scan_of_the_synthetic_project(spa_copy):
    u = D.scan(spa_copy, Pipeline.read(spa_copy))
    assert len(u.jobs) == len(Pipeline.read(spa_copy).processes) and u.jobs_total.bytes > 0
    assert all(j.gentle is None or j.gentle.bytes <= j.usage.bytes for j in u.jobs)


# ---------------------------------------------------------------------------- emptying Trash/

def test_empty_trash_removes_the_entries_and_nothing_else(tmp_path):
    _write(tmp_path / "Trash/Class2D/job004/run_it025_classes.mrcs")
    _write(tmp_path / "Trash/Class2D/job005/x")
    _write(tmp_path / "outside/keep")
    os.symlink(tmp_path / "outside", tmp_path / "Trash/Class2D/link")
    assert D.empty_trash(tmp_path, ["Trash/Class2D/job004", "Trash/Class2D/link"]) == []
    assert not (tmp_path / "Trash/Class2D/job004").exists() and not os.path.lexists(tmp_path / "Trash/Class2D/link")
    assert (tmp_path / "outside/keep").exists() and (tmp_path / "Trash/Class2D/job005/x").exists()
    assert D.empty_trash(tmp_path, ["Trash/Class2D/job005"]) == []
    assert not (tmp_path / "Trash/Class2D").exists() and (tmp_path / "Trash").is_dir()  # Trash/ itself stays


@pytest.mark.parametrize("rel", ["Class2D/job004", "Trash", "Trash/../Class2D", "Trash/x/../../outside"])
def test_empty_trash_refuses_what_is_not_in_trash(tmp_path, rel):
    _write(tmp_path / "Class2D/job004/a")
    _write(tmp_path / "outside/a")
    (tmp_path / "Trash/x").mkdir(parents=True)
    with pytest.raises(D.TrashError):
        D.empty_trash(tmp_path, [rel])
    assert (tmp_path / "Class2D/job004/a").exists() and (tmp_path / "outside/a").exists()


def test_human_sizes():
    assert [D.human(n) for n in (0, 1023, 1536, 5 * 1024 ** 3)] == ["0 B", "1023 B", "1.5 KB", "5.0 GB"]


# ---------------------------------------------------------------------------- the window

def test_disk_usage_window_cleans_and_empties_trash(qtbot, spa_copy, tmp_path, monkeypatch):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings, Qt
    from qtpy.QtWidgets import QMessageBox

    from cryo_savanna.ui.main_window import MainWindow

    job = "Class2D/job006/"
    _write(spa_copy / job / "run_it010_data.star", 40_000)  # an iteration a gentle clean moves
    _write(spa_copy / "Trash/Class2D/job004/deleted_pipeline.star", 100)
    _write(spa_copy / "Trash/Class2D/job004/big.mrcs", 80_000)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: asked.append(self.text()) or QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning shown: {a[1:]}"))
    w = MainWindow(spa_copy, poll_s=1000, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    cleaned = []

    class Pipeliner:  # relion_pipeliner --gentle_clean, as far as the window can tell
        available = True

        def clean(self, name, harsh):
            cleaned.append((name, harsh))
    w.actions.pipeliner = Pipeliner()
    assert w.actions.act_disk.isEnabled()
    w.actions.act_disk.trigger()
    dlg = w.actions._disk_window
    qtbot.waitUntil(lambda: dlg.usage is not None, timeout=10_000)
    assert "Trash/" in dlg.summary.text() and dlg.jobs.rowCount() == len(dlg.usage.jobs)
    assert dlg.tabs.count() == 4 and dlg.trash.rowCount() == 1
    assert "deleted job Class2D/job004 (can be undeleted)" == dlg.trash.item(0, 3).text()

    row = next(r for r in range(dlg.jobs.rowCount()) if dlg.jobs.item(r, 0).data(Qt.UserRole) == job)
    dlg.jobs.selectRow(row)
    assert dlg.btn_gentle.isEnabled()
    dlg.btn_gentle.click()
    w.actions.wait()
    assert cleaned == [(job, False)] and "RELION moves 40.0 KB to Trash/" in asked[-1]  # 40 000 bytes: ten blocks of 4 KB

    dlg.trash.selectRow(0)
    first = dlg.usage
    dlg.btn_empty.click()
    assert "can then no longer be undeleted: Class2D/job004" in asked[-1]
    w.actions.wait()
    qtbot.waitUntil(lambda: dlg.usage is not first and dlg.trash.rowCount() == 0, timeout=10_000)
    assert not (spa_copy / "Trash/Class2D/job004").exists()
    dlg.close()


def test_disk_usage_window_only_looks_when_read_only(qtbot, spa_project, tmp_path):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(spa_project, poll_s=1000, readonly=True,
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.actions.act_disk.trigger()
    dlg = w.actions._disk_window
    qtbot.waitUntil(lambda: dlg.usage is not None, timeout=10_000)
    dlg.jobs.selectAll()
    assert not dlg.btn_gentle.isEnabled() and not dlg.btn_harsh.isEnabled()
    assert not dlg.btn_empty.isEnabled() and not dlg.btn_empty_all.isEnabled()
    assert "read-only" in dlg.btn_gentle.toolTip()
    dlg.close()
