# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Views and card metadata of Polish, MultiBody's flexibility analysis, Subtract, LocalRes, MaskCreate,
CtfRefine and the tomography refinement jobs (CtfRefine, Polish, Reconstruct particle), on files laid
out as RELION 5.1.1 writes them."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.star import StarBlock, read_star
from cryo_savanna.project.summary import SummaryReader
from cryo_savanna.results import flex, mapstats, polish
from cryo_savanna.results import registry as R
from cryo_savanna.results.tables import Table
from cryo_savanna.results.tomo import change_figures, input_tomograms, read_tilt_series_set, tilt_changes
from tests.fixtures.make_fixtures import TILT_SERIES, JobSpec, add_job, blob_volume, mrc, star

SHINY = "Polish/job020/shiny.star"


def views(project, job):
    snap = ProjectLoader(project).refresh(True)
    opts = JobStar.read(project / job / "job.star").options
    return R.views_for(R.context_for(project, snap.jobs[job], snap.pipeline, opts))


def summary(project, job):
    snap = ProjectLoader(project).refresh(True)
    return SummaryReader(project).summarize(snap.jobs[job], snap.pipeline)


# ---------------------------------------------------------------------------- SPA jobs

def _w_training(d: Path, ctx) -> None:
    (d / "opt_params_all_groups.txt").write_text("0.4485 1230.5 2.1\n")
    (d / "run.out").write_text("good parameters: --s_vel 0.4485 --s_div 1230.5 --s_acc 2.1\n")


def _w_polish(d: Path, ctx) -> None:
    root = ctx.root
    blocks = read_star(root / "CtfRefine/job014/particles_ctf_refine.star")
    star(d / "shiny.star", *blocks.values())
    frames = 24
    star(d / "bfactors.star", StarBlock.from_records(
        "perframe_bfactors", ["rlnMovieFrameNumber", "rlnBfactorUsedForSharpening", "rlnFittedInterceptGuinierPlot"],
        [[f, -40.0 - 4 * abs(f - 3) ** 1.2, -0.02 * f] for f in range(frames)]))
    parts = blocks["particles"]
    per_mic: dict[str, int] = {}
    for m in parts.column("rlnMicrographName"):
        per_mic[m] = per_mic.get(m, 0) + 1
    for m, n in list(per_mic.items())[:-1]:  # the last micrograph has no tracks
        tracks = [StarBlock.from_dict("general", {"rlnParticleNumber": n})]
        for p in range(n):
            steps = ctx.rng.normal(0, 0.4, (frames, 2)).cumsum(axis=0)
            tracks.append(StarBlock.from_records(str(p), ["rlnOriginXAngst", "rlnOriginYAngst"], steps.tolist()))
        path = root / polish.tracks_file("Polish/job020/", m)
        path.parent.mkdir(parents=True, exist_ok=True)
        star(path, *tracks)
    (d / "logfile.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")


def _w_multibody(d: Path, ctx) -> None:
    vol = blob_volume(32, ctx.rng)
    mrc(d / "run_body001.mrc", vol, 3.54)
    mrc(d / "run_body002.mrc", vol, 3.54)
    for c in range(1, 4):
        for b in range(1, 3):
            mrc(d / f"analyse_component{c:03d}_bin{b:03d}.mrc", vol * (1 + 0.1 * c * b), 3.54)
    proj = ctx.rng.normal(0, [5, 3, 1, 0.5], (200, 4))
    (d / flex.PROJECTIONS).write_text("".join(
        f"{i + 1:06d}@Extract/job005/Movies/m.mrcs " + "".join(f"{v:15.6f}" for v in row) + " \n"
        for i, row in enumerate(proj)))
    (d / "run.out").write_text(" Reading in data.star file ...\n"
                               "  + Component 1 explains 64.5% of variance.\n  + Component 2 explains 23.1% of variance.\n"
                               "  + Component 3 explains 8.4% of variance.\n  + Component 4 explains 4% of variance.\n")
    (d / "analyse_logfile.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")


def _w_subtract(d: Path, ctx) -> None:
    star(d / "particles_subtracted.star", *read_star(ctx.root / "Refine3D/job011/run_data.star").values())


def _w_locres(d: Path, ctx) -> None:
    z, y, x = np.mgrid[:32, :32, :32]
    r = np.sqrt((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2)
    locres = (3 + r / 6).astype(np.float32)  # best in the middle
    mrc(d / "relion_locres.mrc", locres, 3.54)
    mrc(d / "relion_locres_filtered.mrc", blob_volume(32, ctx.rng), 3.54)
    (d / "histogram.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")


@pytest.fixture(scope="module")
def spa_refine(tmp_path_factory, spa_project) -> Path:
    root = tmp_path_factory.mktemp("refine") / "spa"
    shutil.copytree(spa_project, root, symlinks=True)
    mics = "MotionCorr/job002/corrected_micrographs.star"
    data, post = "CtfRefine/job014/particles_ctf_refine.star", "PostProcess/job013/postprocess.star"
    add_job(root, JobSpec(19, "relion.polish", [("opt_params_all_groups.txt", "ParamsData.txt.relion.polish")],
                          inputs=[data, post, mics], write=_w_training,
                          options={"fn_data": data, "fn_post": post, "fn_mic": mics, "do_param_optim": "Yes",
                                   "do_polish": "No"}))
    add_job(root, JobSpec(20, "relion.polish",
                          [("logfile.pdf", "LogFile.pdf.relion.polish"),
                           ("shiny.star", "ParticleGroupMetadata.star.relion.polished")],
                          inputs=[data, post, mics, "Polish/job019/opt_params_all_groups.txt"], write=_w_polish,
                          options={"fn_data": data, "fn_post": post, "fn_mic": mics, "do_polish": "Yes",
                                   "opt_params": "Polish/job019/opt_params_all_groups.txt", "first_frame": "2"}))
    add_job(root, JobSpec(21, "relion.multibody",
                          [("run_body001.mrc", "DensityMap.mrc.relion.multibody"),
                           ("run_body002.mrc", "DensityMap.mrc.relion.multibody"),
                           ("analyse_logfile.pdf", "LogFile.pdf.relion.flexanalysis")],
                          inputs=["Refine3D/job011/run_optimiser.star"], write=_w_multibody,
                          options={"fn_in": "Refine3D/job011/run_optimiser.star", "do_analyse": "Yes"}))
    add_job(root, JobSpec(22, "relion.subtract",
                          [("particles_subtracted.star", "ParticleGroupMetadata.star.relion.subtracted")],
                          inputs=["Refine3D/job011/run_optimiser.star", "MaskCreate/job012/mask.mrc"], write=_w_subtract,
                          options={"fn_opt": "Refine3D/job011/run_optimiser.star", "fn_mask": "MaskCreate/job012/mask.mrc",
                                   "new_box": "48"}))
    add_job(root, JobSpec(23, "relion.localres",
                          [("histogram.pdf", "LogFile.pdf.relion.localres"),
                           ("relion_locres_filtered.mrc", "DensityMap.mrc.relion.localresfiltered"),
                           ("relion_locres.mrc", "Image3D.mrc.relion.localresmap")],
                          inputs=["Refine3D/job011/run_half1_class001_unfil.mrc"], write=_w_locres,
                          options={"fn_in": "Refine3D/job011/run_half1_class001_unfil.mrc", "do_relion_locres": "Yes",
                                   "do_resmap_locres": "No"}))
    return root


def test_polish_training_shows_its_parameters(spa_refine):
    assert [v.title for v in views(spa_refine, "Polish/job019/")] == ["Files"]
    s = summary(spa_refine, "Polish/job019/")
    assert s.result == ["σ vel 0.4485 · σ div 1230.5 · σ acc 2.1"] and s.settings == ["training"]
    p = polish.trained_params(spa_refine / "Polish/job019")[0]
    assert p.flags() == "--s_vel 0.4485 --s_div 1230.5 --s_acc 2.1"


def test_polish_views_and_summary(spa_refine):
    vs = views(spa_refine, "Polish/job020/")
    assert [v.title for v in vs] == ["Frame weights", "Picks", "Particles", "Files"]
    picks = vs[1]
    assert picks.kind == R.MICROGRAPHS and picks.args == {
        "star": "MotionCorr/job002/corrected_micrographs.star", "picks": SHINY, "tracks": "Polish/job020/"}
    s = summary(spa_refine, "Polish/job020/")
    assert s.result == ["288 particles"]
    assert s.settings == ["σ vel 0.4485 · σ div 1230.5 · σ acc 2.1", "frames 2–last"]
    figs = polish.polish_figures(spa_refine / "Polish/job020")
    assert [f.key for f in figs] == ["bfactors", "scale"]
    assert figs[0].curves[0].x[0] == 1 and len(figs[0].curves[0].x) == 24  # frames counted from 1
    assert "at frame 4" in figs[0].notes[0]


def test_polish_tracks(spa_refine):
    blocks = read_star(spa_refine / SHINY)
    m = blocks["particles"].column("rlnMicrographName")[0]
    rel = polish.tracks_file("Polish/job020/", m)
    assert rel.startswith("Polish/job020/Movies/") and rel.endswith("_tracks.star")
    tracks = polish.read_tracks(spa_refine / rel)
    n = blocks["particles"].column("rlnMicrographName").count(m)
    assert len(tracks) == n and tracks[0].shape == (24, 2)
    x, y = polish.track_lines(np.array([100.0] * n), np.array([200.0] * n), tracks, pixel_size=0.885, scale=10)
    assert len(x) == n * 25 and np.isnan(x[24])  # NaN between particles
    assert x[0] == pytest.approx(100 + tracks[0][0, 0] * 10 / 0.885)
    assert polish.read_tracks(spa_refine / "Polish/job020/none_tracks.star") == []


def test_polish_viewer_draws_tracks(qtbot, spa_refine):
    from tests.test_tomo_viewers import show
    from tests.test_selectors import settle
    from cryo_savanna.viewers.micrograph_viewer import MicrographViewer
    v = show(qtbot, MicrographViewer(spa_refine, "MotionCorr/job002/corrected_micrographs.star", picks=SHINY,
                                     tracks="Polish/job020/"))
    settle(qtbot)
    assert v.show_tracks.isVisible()
    has = [(spa_refine / polish.tracks_file("Polish/job020/", n)).is_file() for n in v.mics.names]
    v.select(has.index(True))
    settle(qtbot)
    x, _ = v.canvas.lines.getData()
    assert x is not None and len(x) > 0 and "tracks" in v.pick_info.text() and "long on average" in v.pick_info.text()
    v.show_tracks.setChecked(False)
    x, _ = v.canvas.lines.getData()
    assert x is None or len(x) == 0
    v.show_tracks.setChecked(True)
    v.select(has.index(False))  # the micrograph without tracks
    settle(qtbot)
    assert "no tracks" in v.pick_info.text()


def test_multibody_flexibility(spa_refine):
    titles = [v.title for v in views(spa_refine, "MultiBody/job021/")]
    assert titles[:3] == ["Maps", "Flexibility", "Components"]
    comps = next(v for v in views(spa_refine, "MultiBody/job021/") if v.title == "Components")
    assert comps.args["maps"][:3] == ["MultiBody/job021/analyse_component001_bin001.mrc",
                                     "MultiBody/job021/analyse_component001_bin002.mrc",
                                     "MultiBody/job021/analyse_component002_bin001.mrc"]
    fa = flex.read_flex(spa_refine / "MultiBody/job021")
    assert fa.from_log and fa.variance.tolist() == [64.5, 23.1, 8.4, 4.0] and fa.projections.shape == (200, 4)
    figs = flex.flex_figures(spa_refine / "MultiBody/job021")
    assert [f.key for f in figs] == ["variance", "hist1", "hist2", "hist3", "hist4"]
    assert summary(spa_refine, "MultiBody/job021/").quality == ["component 1: 64.5% of the variance"]
    # no variance in run.out: that of the projections
    (spa_refine / "MultiBody/job021/run.out").write_text("")
    fa = flex.read_flex(spa_refine / "MultiBody/job021")
    assert not fa.from_log and fa.variance[0] > fa.variance[1] > fa.variance[3] and fa.variance.sum() == pytest.approx(100)


def test_subtract_shows_its_mask_on_the_map(spa_refine):
    vs = views(spa_refine, "Subtract/job022/")
    assert vs[0].title == "Maps" and vs[0].args["maps"] == ["Refine3D/job011/run_class001.mrc", "MaskCreate/job012/mask.mrc"]
    assert vs[0].args["mask"] == "MaskCreate/job012/mask.mrc"
    assert "Particles" in [v.title for v in vs] and "Picks" in [v.title for v in vs]
    assert summary(spa_refine, "Subtract/job022/").settings == ["recentred on the mask", "box 48 px"]


def test_localres_histogram_and_summary(spa_refine):
    vs = views(spa_refine, "LocalRes/job023/")
    assert [v.title for v in vs][:2] == ["Map", "Local resolution"]
    assert vs[1].args == {"figures": "locres", "star": "LocalRes/job023/relion_locres.mrc", "mask": ""}
    st = mapstats.locres_stats(spa_refine / "LocalRes/job023/relion_locres.mrc",
                               spa_refine / "MaskCreate/job012/mask.mrc")
    assert st.masked and 3 <= st.low < st.median < st.high
    fig = mapstats.locres_figure(st)
    assert fig.vlines[0][0] == pytest.approx(st.median)
    s = summary(spa_refine, "LocalRes/job023/")
    assert s.settings == ["RELION"] and s.quality[0].startswith("local res ") and "median" in s.quality[0]


def test_resmap_outside_mark_is_left_out(tmp_path):
    v = np.full((8, 8, 8), mapstats.RESMAP_OUTSIDE, np.float32)
    v[2:6, 2:6, 2:6] = 4.0
    mrc(tmp_path / "half1_resmap.mrc", v, 1.0)
    st = mapstats.locres_stats(tmp_path / "half1_resmap.mrc")
    assert st.median == 4.0 and len(st.values) == 64


def test_maskcreate_and_ctfrefine_settings(spa_project):
    assert summary(spa_project, "MaskCreate/job012/").settings == [
        "threshold 0.01", "extend 3 px", "soft edge 3 px", "lowpass 15 Å"]
    assert summary(spa_project, "CtfRefine/job014/").settings == [
        "defocus (per-particle)", "astigmatism (per-micrograph)", "beam tilt", "trefoil", "4th-order aberrations"]


# ---------------------------------------------------------------------------- tomography jobs

def _shifted_set(src_set: str, job: str, changes: dict):
    """A tilt-series set as tomography CtfRefine / Polish write it: the input's, with some values changed."""
    def w(d: Path, ctx) -> None:
        root = ctx.root
        src_job = src_set.rsplit("/", 1)[0] + "/"
        glob = read_star(root / src_set)["global"]
        col = glob.col_index("rlnTomoTiltSeriesStarFile")
        for r in glob.rows:
            r[col] = f"{job}tilt_series/{r[0]}.star"
        star(d / "tomograms.star", glob)
        for ts in TILT_SERIES:
            b = next(iter(read_star(root / src_job / "tilt_series" / f"{ts}.star").values()))
            for label, fn in changes.items():
                if label not in b.columns:
                    b.columns.append(label)
                    for r in b.rows:
                        r.append(0.0)
                i = b.col_index(label)
                for k, r in enumerate(b.rows):
                    r[i] = fn(float(r[i]), k)
            star(d / "tilt_series" / f"{ts}.star", b)
        star(d / "optimisation_set.star", StarBlock.from_dict("", {"rlnTomoTomogramsFile": job + "tomograms.star"}))
    return w


@pytest.fixture(scope="module")
def tomo_refine(tmp_path_factory, tomo_project) -> Path:
    root = tmp_path_factory.mktemp("refine") / "tomo"
    shutil.copytree(tomo_project, root, symlinks=True)
    ctf_in = "CtfFind/job003/tilt_series_ctf.star"
    add_job(root, JobSpec(15, "relion.ctfrefinetomo",
                          [("optimisation_set.star", "TomoOptimisationSet.star.relion.tomo.ctfrefine"),
                           ("tomograms.star", "TomogramGroupMetadata.star.relion.tomo.ctfrefine")],
                          inputs=[ctf_in], tomo=True,
                          write=_shifted_set(ctf_in, "CtfRefine/job015/", {
                              "rlnDefocusU": lambda v, k: v + 100 + 10 * k, "rlnDefocusV": lambda v, k: v - 50,
                              "rlnCtfScalefactor": lambda v, k: 1 - 0.01 * k}),
                          options={"use_direct_entries": "Yes", "in_tomograms": ctf_in, "do_defocus": "Yes",
                                   "do_scale": "Yes", "do_frame_scale": "Yes"}))
    al_in = "AlignTiltSeries/job005/aligned_tilt_series.star"
    add_job(root, JobSpec(16, "relion.framealigntomo",
                          [("tomograms.star", "TomogramGroupMetadata.star.relion.tomo.polish")],
                          inputs=[al_in], tomo=True,
                          write=_shifted_set(al_in, "Polish/job016/", {
                              "rlnTomoXShiftAngst": lambda v, k: v + 3, "rlnTomoZRot": lambda v, k: v + 0.5}),
                          options={"use_direct_entries": "Yes", "in_tomograms": al_in, "do_shift_align": "Yes"}))

    def w_recon(d: Path, ctx) -> None:
        vol = blob_volume(32, ctx.rng)
        mrc(d / "merged.mrc", vol, 2.7)
        mrc(d / "half1.mrc", vol + ctx.rng.normal(0, 0.05, vol.shape), 2.7)
        mrc(d / "half2.mrc", vol + ctx.rng.normal(0, 0.05, vol.shape), 2.7)
    add_job(root, JobSpec(17, "relion.reconstructparticletomo",
                          [("merged.mrc", "DensityMap.mrc.relion.tomo.map.reconstruct"),
                           ("half1.mrc", "DensityMap.mrc.relion.tomo.halfmap.reconstruct")],
                          inputs=["Extract/job009/optimisation_set.star"], write=w_recon, tomo=True,
                          options={"in_optimisation": "Extract/job009/optimisation_set.star", "box_size": "64"}))
    return root


def test_tomo_ctfrefine_shows_the_changes(tomo_refine):
    vs = views(tomo_refine, "CtfRefine/job015/")
    assert [v.title for v in vs] == ["Tilt series", "Files"]
    assert vs[0].args["compare"] == "CtfFind/job003/tilt_series_ctf.star"
    after = read_tilt_series_set(tomo_refine, "CtfRefine/job015/tomograms.star")
    before = read_tilt_series_set(tomo_refine, "CtfFind/job003/tilt_series_ctf.star")
    ch = tilt_changes(after.tilts(0), before.tilts(0))
    assert set(ch) == {"rlnDefocusU", "rlnDefocusV"}
    assert ch["rlnDefocusU"][:2].tolist() == pytest.approx([100, 110]) and np.all(ch["rlnDefocusV"] == pytest.approx(-50))
    figs = change_figures(after.tilts(0), ch, "TS_01")
    assert [f.key for f in figs] == ["d_defocus"] and len(figs[0].curves) == 2
    assert summary(tomo_refine, "CtfRefine/job015/").settings == ["defocus", "scale (per frame)"]


def test_tomo_polish_shows_the_changes(tomo_refine):
    vs = views(tomo_refine, "Polish/job016/")
    assert vs[0].title == "Tilt series" and vs[0].args["compare"] == "AlignTiltSeries/job005/aligned_tilt_series.star"
    after = read_tilt_series_set(tomo_refine, "Polish/job016/tomograms.star")
    before = read_tilt_series_set(tomo_refine, vs[0].args["compare"])
    ch = tilt_changes(after.tilts(2), before.tilts(2))
    assert set(ch) == {"rlnTomoXShiftAngst", "rlnTomoZRot"} and ch["rlnTomoZRot"] == pytest.approx(0.5)
    assert summary(tomo_refine, "Polish/job016/").settings == ["shifts (entire micrographs)"]


def test_tomo_input_from_the_optimisation_set(tomo_refine):
    assert input_tomograms(tomo_refine, {"in_optimisation": "Extract/job009/optimisation_set.star"}) == \
        "Tomograms/job006/tomograms.star"
    assert input_tomograms(tomo_refine, {"in_optimisation": "none.star", "in_tomograms": ""}) is None


def test_tomo_tilt_series_viewer_with_changes(qtbot, tomo_refine):
    from qtpy.QtCore import Qt
    from tests.test_tomo_viewers import show
    from tests.test_selectors import settle
    from cryo_savanna.viewers.tomo_viewer import TiltSeriesViewer
    v = show(qtbot, TiltSeriesViewer(tomo_refine, "CtfRefine/job015/tomograms.star",
                                     compare="CtfFind/job003/tilt_series_ctf.star"))
    settle(qtbot)
    series = [v.series_model.headerData(c, Qt.Horizontal) for c in range(v.series_model.columnCount())]
    tilts = [v.tilt_model.headerData(c, Qt.Horizontal) for c in range(v.tilt_model.columnCount())]
    assert "|ΔDefocus| (Å)" in series and "ΔDefocus U (Å)" in tilts and "CTF scale" in tilts
    keys = [f.key for f in v.figures.figures]
    assert keys[0] == "d_defocus" and "scale" in keys
    assert "changes from CtfFind/job003/tilt_series_ctf.star" in v.info.text()


def test_reconstruct_particle_fsc(tomo_refine):
    vs = views(tomo_refine, "Reconstruct/job017/")
    assert [v.title for v in vs] == ["Maps", "FSC", "Files"]
    assert vs[0].args["maps"] == ["Reconstruct/job017/merged.mrc", "Reconstruct/job017/half1.mrc",
                                  "Reconstruct/job017/half2.mrc"]
    fig = mapstats.halfmap_fsc(tomo_refine / vs[1].args["half1"], tomo_refine / vs[1].args["half2"])
    y = fig.curves[0].y
    assert y[0] > 0.9 and y[-1] < y[0] and fig.x_resolution


def test_fsc_of_identical_and_unrelated_maps():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16, 16))
    _, same = mapstats.fsc(a, a)
    assert np.allclose(same, 1)
    _, other = mapstats.fsc(a, rng.normal(size=(16, 16, 16)))
    assert np.abs(other[2:]).max() < 0.5
    assert mapstats.crossing(np.array([0, 1, 2, 3.0]), np.array([1, 0.8, 0.1, 0]), 0.143) == pytest.approx(1 + 0.657 / 0.7)


def test_figures_view_kinds(qtbot, spa_refine, tomo_refine):
    from tests.test_selectors import settle
    from cryo_savanna.viewers.views import FiguresView
    for project, kind, kw, keys in (
            (spa_refine, "polish", {"job": "Polish/job020/"}, ["bfactors", "scale"]),
            (spa_refine, "flex", {"job": "MultiBody/job021/"}, ["variance", "hist1"]),
            (spa_refine, "locres", {"star": "LocalRes/job023/relion_locres.mrc",
                                    "args": {"mask": "MaskCreate/job012/mask.mrc"}}, ["locres"]),
            (tomo_refine, "halfmap_fsc", {"star": "Reconstruct/job017/half2.mrc",
                                          "args": {"half1": "Reconstruct/job017/half1.mrc",
                                                   "half2": "Reconstruct/job017/half2.mrc"}}, ["fsc"])):
        w = FiguresView(project, kind, **kw)
        qtbot.addWidget(w)
        settle(qtbot)
        assert [f.key for f in w.grid.figures][:len(keys)] == keys, kind
        assert w.info.text().startswith("from "), w.info.text()


def test_tables_of_changes_keep_unmatched_tilts_empty():
    after = Table.from_block(StarBlock.from_records("t", ["rlnMicrographName", "rlnDefocusU"],
                                                    [["a", 10.0], ["b", 20.0], ["c", 30.0]]))
    before = Table.from_block(StarBlock.from_records("t", ["rlnMicrographName", "rlnDefocusU"],
                                                     [["b", 15.0], ["a", 10.0]]))
    ch = tilt_changes(after, before)
    assert ch["rlnDefocusU"][:2].tolist() == [0.0, 5.0] and np.isnan(ch["rlnDefocusU"][2])
