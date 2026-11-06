# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomography: tilt-series and tomogram readers, viewers, picks in their tomograms, napari, the tomo job menu."""

import shutil
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


def test_one_picks_tab_for_several_particle_files(tomo_project, tmp_path):
    """A continued job lists its earlier particle files too: one Picks tab, for the newest."""
    import os
    import shutil
    project = tmp_path / "tomo"
    shutil.copytree(tomo_project, project, symlinks=True)
    job = "Refine3D/job011/"
    older = job + "run_it005_data.star"
    shutil.copy(project / job / "run_data.star", project / older)
    os.utime(project / older, ns=(1, 1))
    snap = ProjectLoader(project).refresh(force=True)
    ctx = R.context_for(project, snap.jobs[job], snap.pipeline, JobStar.read(project / job / "job.star").options)
    ctx.outputs = [(older, "ParticleGroupMetadata.star.relion.refine3d")] + ctx.outputs
    picks = [v for v in R.views_for(ctx) if v.title == "Picks"]
    assert len(picks) == 1 and picks[0].args["picks"] == job + "run_data.star"


def test_tomogram_picks_of_chosen_classes(qtbot, tomo_project, settle, tmp_path):
    from cryo_savanna.project.star import StarBlock, read_star, write_star
    from cryo_savanna.viewers.tomo_viewer import TomogramViewer
    blocks = read_star(tomo_project / "Picks/job008/particles.star")
    parts = blocks["particles"]
    n = len(parts.rows)
    classes = [i % 3 + 1 for i in range(n)]
    cols = list(parts.columns) + ["rlnClassNumber"]
    blocks["particles"] = StarBlock.from_records("particles", cols, [list(r) + [c] for r, c in zip(parts.rows, classes)])
    star = tmp_path / "run_it010_data.star"  # as a tomography Class3D writes them
    write_star(star, list(blocks.values()))
    picks = read_tomo_picks(tomo_project, str(star))
    assert picks.classes == {c: classes.count(c) for c in (1, 2, 3)}
    v = show(qtbot, TomogramViewer(tomo_project, "Denoise/job007/tomograms.star", picks=str(star), diameter=60))
    settle()
    assert v.classes.isVisible() and v.classes.list.count() == 3
    every = len(v.picks_voxels())
    v.classes.set_selected({2})
    name = v.set.names[v.current]
    assert len(v.picks_voxels()) == picks.count(name, {2}) < every
    assert len(v.xz.dots.data) == picks.count(name, {2})
    assert "(class 2)" in v.info.text()
    v.classes.set_selected({1, 3})
    assert len(v.picks_voxels()) == every - picks.count(name, {2})


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


# ---------------------------------------------------------------------------- Warp/M particles


def make_warp(root):
    """A Warp project (tomostar/, warp_tiltseries/ with tomograms and subtomograms) with the RELION
    project in relion/: the particles of a Class3D of Warp's subtomograms (named by their .tomostar,
    coordinates in pixels of the exported 10 Å, from the corner)."""
    import mrcfile
    from cryo_savanna.project.star import StarBlock, write_star
    rng = np.random.default_rng(3)
    rec = root / "warp_tiltseries" / "reconstruction"
    (rec / "denoised").mkdir(parents=True)
    vol = rng.normal(size=(20, 40, 50)).astype(np.float32)
    for name, data, px in (("TS_1_20.00Apx.mrc", vol, 20.0), ("TS_1_10.00Apx.mrc", np.zeros((40, 80, 100), np.float32), 10.0),
                           ("TS_10_20.00Apx.mrc", vol, 20.0), ("denoised/TS_1_20.00Apx.mrc", vol + 1, 20.0)):
        with mrcfile.new(rec / name) as m:
            m.set_data(data)
            m.voxel_size = px
    (root / "tomostar").mkdir()
    for ts in ("TS_1", "TS_2"):
        (root / "tomostar" / f"{ts}.tomostar").write_text("data_\nloop_\n_wrpMovieName\n_wrpAngleTilt\nx.tif 0\n")
    project = root / "relion"
    job = project / "Class3D" / "job003"
    job.mkdir(parents=True)
    rows = []
    for ts, n in (("TS_1.tomostar", 30), ("TS_2.tomostar", 10)):
        for k in range(n):
            x, y, z = rng.uniform(0, 100), rng.uniform(0, 80), rng.uniform(0, 40)  # 10 Å pixels
            rows.append([x, y, z, ts, f"../warp_tiltseries/subtomo/{ts[:-9]}/{ts[:-9]}_{k:06d}_10.00A.mrc",
                         1, 1 + k % 2])
    optics = StarBlock.from_records("optics", ["rlnOpticsGroup", "rlnOpticsGroupName", "rlnImagePixelSize",
                                               "rlnImageSize", "rlnImageDimensionality"], [[1, "opticsGroup1", 10.0, 32, 3]])
    parts = StarBlock.from_records("particles", ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
                                                 "rlnMicrographName", "rlnImageName", "rlnOpticsGroup",
                                                 "rlnClassNumber"], rows)
    write_star(job / "run_it025_data.star", [optics, parts])
    return project, "Class3D/job003/run_it025_data.star"


def test_warp_particles_and_their_tomograms(tmp_path):
    from cryo_savanna.results.tomo import is_warp_particles, warp_tomogram_set
    project, rel = make_warp(tmp_path)
    assert is_warp_particles(project / rel) and not is_tomo_particles(project / rel)
    picks = read_tomo_picks(project, rel)
    assert picks.pixel_size == 10.0 and set(picks.by_tomo) == {"TS_1.tomostar", "TS_2.tomostar"}
    assert picks.classes == {1: 20, 2: 20}
    # 10 Å coordinates in a 20 Å tomogram: half
    assert np.allclose(picks.voxels("TS_1.tomostar", 20.0, (20, 40, 50)), picks.by_tomo["TS_1.tomostar"] / 2)
    ts, searched = warp_tomogram_set(project, picks, rel)
    assert searched[0] == tmp_path / "warp_tiltseries" / "reconstruction"  # from the subtomograms' path
    assert ts.names == ["TS_1.tomostar", "TS_2.tomostar"]
    # the coarsest tomogram (not TS_10's), and its denoised copy first
    assert [c for c, _ in ts.tomogram_kinds()] == ["rlnTomoReconstructedTomogramDenoised", "rlnTomoReconstructedTomogram"]
    assert ts.tomogram(0, "rlnTomoReconstructedTomogram").path == str(searched[0] / "TS_1_20.00Apx.mrc")
    assert ts.tomogram(0).path == str(searched[0] / "denoised" / "TS_1_20.00Apx.mrc")
    assert ts.tomogram(1) is None  # TS_2 was not reconstructed
    # a folder the user names comes first
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copy(searched[0] / "TS_1_20.00Apx.mrc", other / "TS_1_20.00Apx.mrc")
    ts2, searched2 = warp_tomogram_set(project, picks, rel, [str(other)])
    assert searched2[0] == other and ts2.tomogram(0, "rlnTomoReconstructedTomogram").path == str(other / "TS_1_20.00Apx.mrc")


def test_warp_export_without_optics(tmp_path):
    """As ts_export_particles writes them: one unnamed table, the pixel size per particle."""
    from cryo_savanna.results.tomo import is_warp_particles
    (tmp_path / "p.star").write_text(
        "data_\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n_rlnCoordinateZ\n_rlnMicrographName\n_rlnMagnification\n"
        "_rlnDetectorPixelSize\n_rlnImageName\n"
        "10 20 30 TS_1.tomostar 10000.0 7.50000 sub/TS_1_000001_7.50A.mrc\n")
    assert is_warp_particles(tmp_path / "p.star")
    p = read_tomo_picks(tmp_path, "p.star")
    assert p.pixel_size == 7.5 and np.allclose(p.voxels("TS_1.tomostar", 15.0, (40, 40, 40)), [[5, 10, 15]])
    # SPA particles are not Warp's
    (tmp_path / "spa.star").write_text("data_particles\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n_rlnMicrographName\n"
                                       "1 2 mics/a.mrc\n")
    assert not is_warp_particles(tmp_path / "spa.star")


def test_warp_class3d_gets_the_tomogram_picks_tab(tmp_path):
    project, rel = make_warp(tmp_path)
    ctx = R.JobContext(project, "Class3D/job003/", "relion.class3d.tomo", "Succeeded",
                       [(rel, "ParticleGroupMetadata.star.relion.class3d")],
                       {"nr_classes": "2", "particle_diameter": "200"})
    picks = [v for v in R.views_for(ctx) if v.title == "Picks"]
    assert len(picks) == 1 and picks[0].kind == R.TOMOGRAMS and picks[0].args["picks"] == rel
    assert picks[0].args["star"] == "" and picks[0].args["diameter"] == 200.0
    # any job with Warp particles among its outputs
    ctx = R.JobContext(project, "Select/job004/", "relion.select", "Succeeded",
                       [(rel, "ParticleGroupMetadata.star.relion")])
    assert [v.kind for v in R.views_for(ctx) if v.title == "Picks"] == [R.TOMOGRAMS]


def test_warp_particles_on_their_tomograms(qtbot, tmp_path, settle):
    from cryo_savanna.viewers.tomo_viewer import _WARP_FOLDERS, TomogramViewer
    project, rel = make_warp(tmp_path)
    v = show(qtbot, TomogramViewer(project, "", picks=rel, diameter=200))
    settle()
    assert v.warp and v.folder_btn.isVisible() and not v.napari.isEnabled()
    assert v.set.names == ["TS_1.tomostar", "TS_2.tomostar"] and v.shape == (20, 40, 50) and v.voxel == 20.0
    xyz = v.picks_voxels()
    assert len(xyz) == 30 and np.allclose(xyz, v.picks.by_tomo["TS_1.tomostar"] / 2)
    assert len(v.xz.dots.data) == 30 and len(v.canvas.points.data) == v.cut_count > 0
    # TS_2 has no tomogram: nothing left on screen, and where it looked
    v.select(1)
    settle()
    assert v.shape is None and len(v.canvas.points.data) == 0 and len(v.xz.dots.data) == 0
    assert v.canvas.item.image is None and "No Warp tomogram for TS_2.tomostar" in v.info.text()
    assert "warp_tiltseries" in v.info.text()
    # back to one with a tomogram: the totals again
    v.select(0)
    settle()
    assert v.shape == (20, 40, 50) and "40 particles" in v.info.text()
    # a folder chosen for the missing one
    other = tmp_path / "more" / "reconstruction"
    other.mkdir(parents=True)
    import mrcfile
    with mrcfile.new(other / "TS_2_20.00Apx.mrc") as m:
        m.set_data(np.ones((20, 40, 50), np.float32))
        m.voxel_size = 20.0
    try:
        v.use_folder(str(tmp_path / "more"))
        settle()
        v.select(1)
        settle()
        assert v.shape == (20, 40, 50) and len(v.picks_voxels()) == 10
    finally:
        _WARP_FOLDERS.clear()
