# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomography: tilt-series and tomogram readers, viewers, picks in their tomograms, napari, the tomo job menu."""

import stat

import numpy as np
import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.results import registry as R
from cryo_savanna.results.tomo import (
    columns_of,
    has_tomograms,
    is_tomo_particles,
    read_tilt_series_set,
    read_tomo_picks,
    series_summary,
    tilt_figures,
    tomograms_for_particles,
)
from cryo_savanna.viewers.external import napari_command

# ---------------------------------------------------------------------------- no Qt


def test_tilt_series_set_and_tilts(tomo_project):
    ts = read_tilt_series_set(tomo_project, "CtfFind/job003/tilt_series_ctf.star")
    assert ts.names == ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"] and ts.pixel_size(0) == 0.675
    tilts = ts.tilts(2)
    assert len(tilts) == 41 and tilts.numeric("rlnTomoNominalStageTiltAngle")[0] == -60
    assert ts.tilt_image(tilts, 0).path.endswith("MotionCorr/job002/tilt_images/TS_43_001.mrc")
    assert ts.ctf_image(tilts, 0).path.endswith("TS_43_001.ctf")
    assert ts.tilts(2) is tilts  # cached while the file is unchanged
    s = series_summary(tilts)
    assert s["Tilts"] == "41" and s["Range (°)"] == "-60 … 60" and "Max. res. (Å)" in s
    keys = [f.key for f in tilt_figures(tilts)]
    assert keys == ["defocus", "ctf", "dose"]
    aligned = read_tilt_series_set(tomo_project, "AlignTiltSeries/job005/aligned_tilt_series.star")
    assert "shifts" in [f.key for f in tilt_figures(aligned.tilts(0))]
    raw = read_tilt_series_set(tomo_project, "Import/job001/tilt_series.star")
    assert raw.tilt_image(raw.tilts(0), 0).path.endswith(".mrc") and raw.ctf_image(raw.tilts(0), 0) is None
    with pytest.raises(ValueError):
        read_tilt_series_set(tomo_project, "Picks/job008/particles.star")


def test_tomograms_and_picks(tomo_project):
    assert has_tomograms(tomo_project / "Denoise/job007/tomograms.star")
    assert not has_tomograms(tomo_project / "AlignTiltSeries/job005/aligned_tilt_series.star")
    ts = read_tilt_series_set(tomo_project, "Denoise/job007/tomograms.star")
    assert [c for c, _ in ts.tomogram_kinds()] == ["rlnTomoReconstructedTomogramDenoised",
                                                   "rlnTomoReconstructedTomogram"]
    assert ts.tomogram(0).path.endswith("Denoise/job007/tomograms/rec_TS_01.mrc")
    assert ts.tomogram(0, "rlnTomoReconstructedTomogram").path.endswith("Tomograms/job006/tomograms/rec_TS_01.mrc")
    assert ts.binning(0) == 10.0
    assert is_tomo_particles(tomo_project / "Picks/job008/particles.star")
    picks = read_tomo_picks(tomo_project, "Picks/job008/particles.star")
    assert sum(picks.count(n) for n in ts.names) == sum(len(v) for v in picks.by_tomo.values()) > 0
    vox = picks.voxels("TS_01", 6.75, (16, 64, 64))
    assert ((vox[:, :2] > 0) & (vox[:, :2] < 64)).all() and ((vox[:, 2] > 0) & (vox[:, 2] < 16)).all()
    assert tomograms_for_particles(tomo_project, "Extract/job009/particles.star", {}) == \
        "Tomograms/job006/tomograms.star"
    assert tomograms_for_particles(tomo_project, "Picks/job008/particles.star",
                                   {"in_tomoset": "Denoise/job007/tomograms.star"}) == "Denoise/job007/tomograms.star"
    assert columns_of(tomo_project / "Picks/job008/particles.star", "particles")[0] == "rlnTomoName"


def test_old_style_pixel_coordinates(tmp_path):
    (tmp_path / "p.star").write_text("data_particles\nloop_\n_rlnTomoName\n_rlnCoordinateX\n_rlnCoordinateY\n"
                                     "_rlnCoordinateZ\nTS_1 400 800 100\n")
    p = read_tomo_picks(tmp_path, "p.star")
    assert not p.centered_angst and np.allclose(p.voxels("TS_1", 6.75, (50, 100, 100), binning=10), [[40, 80, 10]])


def views(project, job):
    snap = ProjectLoader(project).refresh(True)
    opts = JobStar.read(project / job / "job.star").options
    return R.views_for(R.context_for(project, snap.jobs[job], snap.pipeline, opts))


def test_tomo_views_per_job(tomo_project):
    def titles(job):
        return [v.title for v in views(tomo_project, job)]
    for job in ("Import/job001/", "MotionCorr/job002/", "CtfFind/job003/", "AlignTiltSeries/job005/"):
        assert titles(job) == ["Tilt series", "Files"], job
    assert titles("ExcludeTiltImages/job004/") == ["Selection", "Tilt series", "Files"]
    assert titles("Tomograms/job006/") == ["Tomograms", "Files"]
    assert titles("Denoise/job007/") == ["Tomograms", "Files"]
    picks = views(tomo_project, "Picks/job008/")[0]
    assert picks.kind == R.TOMOGRAMS and picks.args["picks"] == "Picks/job008/particles.star"
    assert picks.args["star"] == "Denoise/job007/tomograms.star"
    ext = views(tomo_project, "Extract/job009/")[0]
    assert ext.title == "Picks" and ext.args["diameter"] == 256.0  # half the box
    assert titles("Refine3D/job011/")[:2] == ["Maps", "Picks"]
    assert titles("Reconstruct/job010/") == ["Map", "Files"]


def test_napari_command(tmp_path):
    exe = tmp_path / "relion_python_tomo_view"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    env = {"PATH": str(tmp_path)}
    assert napari_command("tilt-series", "a.star", environ=env) == [str(exe), "tilt-series",
                                                                    "--tilt-series-star-file", "a.star"]
    assert napari_command("particles", "p.star", "t.star", environ=env)[1:] == [
        "particles", "--particle-star-file", "p.star", "--tilt-series-star-file", "t.star"]
    assert napari_command("tomograms", "t.star", environ={"PATH": ""}) is None


# ---------------------------------------------------------------------------- viewers


@pytest.fixture
def settle(qtbot):
    pytest.importorskip("pytestqt")
    from qtpy.QtWidgets import QApplication
    from cryo_savanna.viewers.tasks import pool

    def run(rounds: int = 6):
        for _ in range(rounds):
            pool().waitForDone(20000)
            QApplication.processEvents()
            qtbot.wait(40)
    return run


def show(qtbot, w):
    qtbot.addWidget(w)
    w.resize(1300, 850)
    w.show()
    qtbot.waitExposed(w)
    return w


def test_tilt_series_viewer(qtbot, tomo_project, settle):
    from qtpy.QtCore import Qt
    from cryo_savanna.viewers.tomo_viewer import TiltSeriesViewer
    v = show(qtbot, TiltSeriesViewer(tomo_project, "CtfFind/job003/tilt_series_ctf.star"))
    settle()
    assert v.series_model.rowCount() == 5 and v.tilt_model.rowCount() == 41
    headers = [v.series_model.headerData(c, Qt.Horizontal) for c in range(v.series_model.columnCount())]
    assert "Tilts" in headers and "Max. res. (Å)" in headers
    # starts at the untilted image, with its CTF spectrum
    row = v.tilt_proxy.mapToSource(v.tilt_view.currentIndex()).row()
    assert v.tilts.numeric("rlnTomoNominalStageTiltAngle")[row] == 0
    assert v.canvas.full_shape == (64, 64) and v.ctf.isVisible() and v.ctf.full_shape == (48, 48)
    assert [f.key for f in v.figures.figures] == ["defocus", "ctf", "dose"]
    v.select_series(3)
    settle()
    assert "TS_45" in v.figures.figures[0].title
    assert "5 tilt series, 205 tilt images" in v.info.text()


def test_tomogram_viewer_slab_side_views_and_picks(qtbot, tomo_project, settle):
    from qtpy.QtCore import Qt
    from cryo_savanna.viewers.tomo_viewer import TomogramViewer
    v = show(qtbot, TomogramViewer(tomo_project, "Denoise/job007/tomograms.star",
                                   picks="Picks/job008/particles.star", diameter=60))
    settle()
    assert v.shape == (16, 64, 64) and v.voxel == 6.75 and v.z == 8
    assert v.canvas.preview.shape == (64, 64)
    assert v.model.headerData(2, Qt.Horizontal) == "Picks"  # right after the name
    xyz = v.picks_voxels()
    r = 30 / 6.75
    assert v.cut_count == int((np.abs(xyz[:, 2] - 8.5) < r).sum())
    assert len(v.xz.dots.data) == len(xyz)
    # the slab averages sections around z
    from cryo_savanna.images.mrc import MrcReader
    with MrcReader(tomo_project / "Denoise/job007/tomograms/rec_TS_01.mrc") as rd:
        expected = rd.planes(6, 11).mean(axis=0)
    assert np.allclose(v.canvas.preview, expected, atol=1e-5)
    v.zslider.setValue(2)
    qtbot.wait(60)
    settle()
    assert v.z == 2 and v.xz.line.value() == 2.5
    # denoised and raw side by side, linked
    assert v.compare.isVisible()
    v.compare.setChecked(True)
    settle()
    assert v.raw.isVisible() and v.raw.preview is not None
    assert not np.allclose(v.raw.preview, v.canvas.preview)
    v.show_picks.setChecked(False)
    assert len(v.canvas.points.data) == 0
    assert "particles" in v.info.text()


def test_napari_button_reports_a_failed_start(qtbot, tmp_path):
    from cryo_savanna.viewers.tomo_viewer import NapariButton
    exe = tmp_path / "relion_python_tomo_view"
    exe.write_text("#!/bin/sh\necho 'ModuleNotFoundError: napari' >&2\nexit 3\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    seen = []
    b = NapariButton(tmp_path, seen.append)
    qtbot.addWidget(b)
    b.set_command([str(exe), "tomograms", "--tilt-series-star-file", "t.star"])
    b._watch.setInterval(300)
    b.click()
    qtbot.waitUntil(lambda: any("stopped" in s for s in seen), timeout=5000)
    assert "exit 3" in seen[-1] and "ModuleNotFoundError" in seen[-1]
    b.set_command(None)
    assert not b.isEnabled()


def test_window_picks_the_tomo_job_menu(qtbot, tmp_path, tomo_project, spa_project):
    from qtpy.QtCore import QSettings
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.project.slurm import SlurmClient
    from cryo_savanna.ui.main_window import MainWindow

    def window(project, tomo=None):
        w = MainWindow(project, tomo=tomo, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat),
                       slurm=SlurmClient(which=lambda p: None))
        qtbot.addWidget(w)
        w.load_now()
        return w

    def menu(w):
        return [a.text() for a in w.actions.new_menu.actions() if a.text()]

    t = window(tomo_project)
    assert t.tomo and t.act_tomo.isChecked() and "[tomo]" in t.windowTitle()
    assert any("tilt" in m.lower() for m in menu(t))
    s = window(spa_project)
    assert not s.tomo and not any("tilt" in m.lower() for m in menu(s))
    s.act_tomo.setChecked(True)  # by hand
    assert s.tomo and any("tilt" in m.lower() for m in menu(s))
    forced = window(spa_project, tomo=True)  # cryo-savanna --tomo
    assert forced.tomo
    t.select_job("Picks/job008/")
    assert t.info.view_titles()[0] == "Picks"
    v = t.info.show_view("Picks")
    from cryo_savanna.viewers.tasks import pool
    for _ in range(6):
        pool().waitForDone(20000)
        qtbot.wait(40)
    assert v.shape == (16, 64, 64)
