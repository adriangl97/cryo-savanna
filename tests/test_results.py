# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Reading results for the viewers (no Qt): images, cache, micrographs, picks, classes, curves, registry."""

import os
import time

import mrcfile
import numpy as np
import pytest

from cryo_savanna.images.cache import ThumbCache
from cryo_savanna.images.mrc import (
    ImageRef,
    contrast_limits,
    downsample,
    fourier_crop,
    header,
    read_image,
    read_images,
    read_region,
)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.star import loop_columns
from cryo_savanna.results import registry as R
from cryo_savanna.results.classes import iterations, read_class_averages, read_model_classes, read_particles
from cryo_savanna.results.curves import angular_distribution, convergence, model_fsc, motion_figure, postprocess_figures
from cryo_savanna.results.micrographs import motion_track, read_micrographs, read_picks
from cryo_savanna.results.tables import Table, blocks_of


def write_mrc(path, data, voxel=1.0):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(data, np.float32))
        m.voxel_size = voxel


# ---------------------------------------------------------------------------- images

def test_image_names():
    assert ImageRef.parse("000012@Extract/job005/Movies/a.mrcs") == ImageRef("Extract/job005/Movies/a.mrcs", 11)
    assert ImageRef.parse("CtfFind/job003/Movies/a.ctf:mrc") == ImageRef("CtfFind/job003/Movies/a.ctf", None)
    assert str(ImageRef("s.mrcs", 0)) == "000001@s.mrcs"
    assert ImageRef.parse("a.mrc").resolve("/p") == ImageRef("/p/a.mrc", None)


def test_slices_regions_and_volumes(tmp_path):
    stack = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    write_mrc(tmp_path / "s.mrcs", stack, 2.0)
    h = header(tmp_path / "s.mrcs")
    assert (h.nx, h.ny, h.nz, h.is_stack, h.pixel_size) == (7, 6, 5, True, 2.0)
    assert np.array_equal(read_image(ImageRef(str(tmp_path / "s.mrcs"), 3)), stack[3])
    assert set(read_images(str(tmp_path / "s.mrcs"), [0, 4, 9])) == {0, 4}  # 9 does not exist
    assert np.array_equal(read_region(ImageRef(str(tmp_path / "s.mrcs"), 1), 2, 5, 1, 4), stack[1, 2:5, 1:4])
    write_mrc(tmp_path / "v.mrc", stack)  # a volume: its central section
    assert np.array_equal(read_image(ImageRef(str(tmp_path / "v.mrc"))), stack[2])


def test_a_file_rewritten_while_read_is_an_error_not_a_crash(tmp_path):
    """RELION truncates and rewrites maps and micrographs of running jobs; a memory-mapped read of
    such a file would kill the GUI with SIGBUS, a plain read only fails."""
    from cryo_savanna.images.mrc import ImageError, MrcReader
    write_mrc(tmp_path / "m.mrc", np.ones((4, 64, 64)))
    with MrcReader(tmp_path / "m.mrc") as r:
        with open(tmp_path / "m.mrc", "r+b") as fh:
            fh.truncate(1024 + 64 * 64 * 4)  # the header and one section left
        assert r.section(0).shape == (64, 64)
        with pytest.raises(ImageError, match="shorter than its header"):
            r.section(2)
    with pytest.raises(ImageError):
        read_image(ImageRef(str(tmp_path / "m.mrc"), 3))


def test_downsampling_keeps_the_picture():
    y, x = np.mgrid[:1000, :1200]
    img = (np.sin(x / 40.0) + np.cos(y / 55.0)).astype(np.float32)
    small, factor = downsample(img, 300)
    assert max(small.shape) == 300 and factor == pytest.approx(4.0)
    ref = img[::4, ::4][: small.shape[0], : small.shape[1]]
    assert np.corrcoef(small.ravel(), ref.ravel())[0, 1] > 0.99
    assert abs(small.mean() - img.mean()) < 0.01
    assert fourier_crop(img, 1000, 1200) is not None
    same, f = downsample(img[:100, :100], 300)
    assert f == 1.0 and same.shape == (100, 100)
    lo, hi = contrast_limits(img, 0)
    assert (lo, hi) == pytest.approx((img.min(), img.max()))


def test_4k_micrograph_preview_is_fast(tmp_path):
    img = np.random.default_rng(0).normal(0, 1, (4096, 4096)).astype(np.float32)
    write_mrc(tmp_path / "big.mrc", img)
    t = time.perf_counter()
    small, factor = downsample(read_image(ImageRef(str(tmp_path / "big.mrc"))), 1024)
    took = time.perf_counter() - t
    assert small.shape == (1024, 1024) and factor == 4.0
    assert took < 3.0, f"{took:.2f} s"


def test_thumb_cache_memory_disk_and_changes(tmp_path):
    f = tmp_path / "a.mrc"
    write_mrc(f, np.full((8, 8), 5.0) + np.eye(8))
    ref = ImageRef(str(f))
    calls = []

    def make():
        calls.append(1)
        return read_image(ref)

    c = ThumbCache(tmp_path / "proj", root=tmp_path / "cache")
    a = c.get(ref, 64, make, disk=True)
    assert c.get(ref, 64, make, disk=True) is a and len(calls) == 1
    c2 = ThumbCache(tmp_path / "proj", root=tmp_path / "cache")  # a new session reads it from disk
    b = c2.get(ref, 64, make, disk=True)
    assert len(calls) == 1 and np.allclose(a, b, atol=0.01)
    time.sleep(0.01)
    write_mrc(f, np.zeros((8, 8)))  # rewritten: read again
    assert c2.get(ref, 64, make, disk=True).max() == 0 and len(calls) == 2
    c3 = ThumbCache(tmp_path / "proj", root=tmp_path / "cache", max_disk_mb=0)
    c3.prune()
    assert not list((c3.dir).glob("*.npz"))


# ---------------------------------------------------------------------------- STAR

def test_loop_columns_and_table(spa_project):
    cols, data = loop_columns(spa_project / "Extract/job005/particles.star", "particles",
                              ["rlnImageName", "rlnMicrographName", "rlnNope"])
    assert "rlnCoordinateX" in cols and set(data) == {"rlnImageName", "rlnMicrographName"}
    assert len(data["rlnImageName"]) == 24 * 40
    t = Table.read(spa_project / "CtfFind/job003/micrographs_ctf.star", "micrographs")
    assert t.numeric("rlnMicrographName") is None and len(t.numeric("rlnDefocusU")) == 24
    assert "rlnCtfFigureOfMerit" in t.numeric_columns()
    assert blocks_of(spa_project / "CtfFind/job003/micrographs_ctf.star") == ["optics", "micrographs"]


# ---------------------------------------------------------------------------- micrographs and picks

def test_micrographs_ctf_and_motion(spa_project):
    ms = read_micrographs(spa_project, "CtfFind/job003/micrographs_ctf.star")
    assert len(ms) == 24 and ms.kind == "micrographs" and ms.pixel_size == 0.885
    assert [t for _, t in ms.metric_columns()][:2] == ["Defocus U (Å)", "Defocus V (Å)"]
    assert read_image(ms.ctf_image(0)).shape == (64, 64)
    mc = read_micrographs(spa_project, "MotionCorr/job002/corrected_micrographs.star")
    frames, sx, sy = motion_track(mc.motion_file(0))
    assert len(frames) == 24 and sx[0] == 0
    fig = motion_figure(mc.motion_file(0), 0.885)
    assert fig.xlabel == "Shift X (Å)" and "24 frames" in fig.notes[0]
    movies = read_micrographs(spa_project, "Import/job001/movies.star")
    assert movies.kind == "movies" and movies.names[0].endswith(".tiff")


def test_picks_from_coordinate_lists_and_particles(spa_project):
    ps = read_picks(spa_project, "AutoPick/job004/autopick.star")
    mic = ps.micrographs[0]
    p = ps.picks(mic)
    assert len(p) == ps.count(mic) > 0 and p.fom is not None
    parts = read_picks(spa_project, "Extract/job005/particles.star")
    assert len(parts.micrographs) == 24 and len(parts.picks(parts.micrographs[0])) == 40
    assert parts.count("not a micrograph") == 0
    with pytest.raises(ValueError):
        read_picks(spa_project, "CtfFind/job003/micrographs_ctf.star")


# ---------------------------------------------------------------------------- classes and particles

def test_classes_per_iteration(spa_project):
    its = iterations(spa_project / "Class2D/job006")
    assert its[-1][0] == 200 and len(its) > 1
    cs = read_model_classes(spa_project, "Class2D/job006/run_it200_model.star",
                            spa_project / "Class2D/job006/run_it200_data.star")
    assert len(cs.classes) == 50 and cs.dim == 2 and cs.iteration == 200
    assert sum(c.particles or 0 for c in cs.classes) == 24 * 40
    assert abs(sum(c.distribution for c in cs.classes) - 1) < 1e-3
    assert read_image(cs.classes[4].ref).shape == (48, 48)
    c3 = read_model_classes(spa_project, "Class3D/job009/run_it025_model.star")
    assert c3.dim == 3 and read_image(c3.classes[0].ref).shape == (32, 32)  # central section
    sel = read_class_averages(spa_project, "Select/job007/class_averages.star")
    assert [c.number for c in sel.classes] == [3, 7, 12, 18, 22] and sel.classes[0].ref.index == 2
    # Refine3D: the half1 models
    assert [it for it, _ in iterations(spa_project / "Refine3D/job011")] == [0, 3, 6, 9, 12, 15]


def test_particles(spa_project):
    ps = read_particles(spa_project, "Select/job007/particles.star")
    assert len(ps) == 24 * 25 and ps.box == 64 and ps.pixel_size == 3.54
    assert read_image(ps.image(3)).shape == (32, 32)
    assert "micrograph" in ps.tooltip(0)


# ---------------------------------------------------------------------------- curves

def test_postprocess_fsc_and_guinier(spa_project):
    fsc, gui = postprocess_figures(spa_project / "PostProcess/job013/postprocess.star")
    assert [c.label for c in fsc.curves][0] == "Corrected" and len(fsc.curves) == 4
    assert fsc.vlines[0][1] == "3.20 Å" and fsc.x_resolution and fsc.hlines == [(0.143, "0.143")]
    assert gui.key == "guinier" and len(gui.curves) == 2


def test_convergence_and_model_fsc(spa_project):
    figs = {f.key: f for f in convergence(spa_project / "Class3D/job009")}
    assert set(figs) == {"resolution", "occupancy", "accuracy", "changes"}
    assert list(figs["resolution"].curves[0].x) == [5, 10, 15, 20, 25]
    assert len(figs["occupancy"].curves) == 4
    assert figs["changes"].curves[0].y[0] == pytest.approx(8.0)
    big = {f.key for f in convergence(spa_project / "Class2D/job006")}
    assert "accuracy" not in big  # 50 classes: no per-class accuracy curves
    fsc = model_fsc(spa_project / "Refine3D/job011/run_it015_half1_model.star")
    assert fsc.curves[0].label == "Class 1" and fsc.curves[0].y[0] > 0.9
    h, n = angular_distribution(spa_project / "Refine3D/job011/run_data.star")
    assert n == 24 * 12 and h.shape == (36, 72) and h.sum() == n


def test_curves_of_incomplete_files(tmp_path):
    """Files with a column missing or no usable values give fewer plots, not an error."""
    parts = tmp_path / "data.star"
    parts.write_text("data_particles\nloop_\n_rlnAngleTilt #1\n10\n20\n")  # no rlnAngleRot
    assert angular_distribution(parts) is None
    pp = tmp_path / "postprocess.star"
    pp.write_text("data_general\n_rlnFittedSlopeGuinierPlot -10\n_rlnFittedInterceptGuinierPlot 1\n\n"
                  "data_guinier\nloop_\n_rlnResolutionSquared #1\n_rlnLogAmplitudesOriginal #2\nnan 1\ninf 2\n")
    (gui,) = postprocess_figures(pp)
    assert [c.label for c in gui.curves] == ["Original"]  # no fit line without a finite resolution


# ---------------------------------------------------------------------------- registry

def views(project, job):
    snap = ProjectLoader(project).refresh(True)
    opts = JobStar.read(project / job / "job.star").options
    return R.views_for(R.context_for(project, snap.jobs[job], snap.pipeline, opts))


def test_viewers_per_job(spa_project):
    def titles(job):
        return [v.title for v in views(spa_project, job)]
    assert titles("Import/job001/") == ["Movies", "Files"]
    assert titles("MotionCorr/job002/") == ["Micrographs", "Motion plots", "Files"]
    assert titles("CtfFind/job003/")[:2] == ["Micrographs", "CTF plots"]
    picks = views(spa_project, "AutoPick/job004/")[0]
    assert picks.title == "Picks" and picks.args["star"] == "CtfFind/job003/micrographs_ctf.star"
    assert picks.args["diameter"] > 0
    assert titles("Extract/job005/") == ["Particles", "On micrographs", "Plots", "Files"]
    assert titles("Class2D/job006/")[:2] == ["Classes", "Convergence"]
    assert titles("Class3D/job009/")[:4] == ["Classes", "Maps", "Convergence", "Angular distribution"]
    assert titles("Refine3D/job011/")[:4] == ["Maps", "FSC", "Convergence", "Angular distribution"]
    assert titles("PostProcess/job013/") == ["FSC", "Maps", "Files"]
    assert titles("Select/job007/") == ["Selection", "Selected classes", "Particles", "Files"]
    assert titles("CtfRefine/job014/") == ["CTF refinement", "Per-particle CTF", "Aberrations", "Particles", "Files"]
    mask = views(spa_project, "MaskCreate/job012/")[0]
    assert mask.title == "Maps" and mask.args == {
        "maps": ["Refine3D/job011/run_class001.mrc", "MaskCreate/job012/mask.mrc"],
        "mask": "MaskCreate/job012/mask.mrc", "color": "", "masks": ["MaskCreate/job012/mask.mrc"]}
    pp = views(spa_project, "PostProcess/job013/")[1]
    assert pp.args["maps"][0].endswith("postprocess_masked.mrc") and pp.args["mask"] == "MaskCreate/job012/mask.mrc"
    assert titles("Polish/job015/") == ["Files"]  # running, nothing written yet


def test_classification_jobs_get_picks_of_their_particles(spa_project):
    c2 = views(spa_project, "Class2D/job006/")
    picks = next(v for v in c2 if v.title == "Picks")
    assert picks.kind == R.MICROGRAPHS and picks.args == {"picks": "Class2D/job006/run_it200_data.star",
                                                          "diameter": 200.0}
    assert [v.title for v in c2][2:] == ["Picks", "Particles", "Files"]  # the particles keep their tab
    assert "Picks" in [v.title for v in views(spa_project, "Class3D/job009/")]
    assert "Picks" not in [v.title for v in views(spa_project, "Refine3D/job011/")]  # one class


def test_picks_from_particles_carry_their_classes(spa_project):
    ps = read_picks(spa_project, "Class3D/job009/run_it025_data.star")
    assert set(ps.classes) <= {1, 2, 3, 4} and sum(ps.classes.values()) == 24 * 25
    assert ps.pixel_size == 0.885  # the micrographs' (rlnMicrographOriginalPixelSize)
    mic = ps.micrographs[0]
    p = ps.picks(mic)
    assert p.cls is not None and len(p.cls) == len(p) == 25
    assert ps.count(mic, {1, 2}) == int(np.isin(p.cls, [1, 2]).sum())
    assert ps.count(mic, set()) == 0 and ps.count(mic) == 25
    coords = read_picks(spa_project, "AutoPick/job004/autopick.star")  # coordinate files: no classes
    assert coords.classes == {} and coords.picks(coords.micrographs[0]).in_classes({1}).all()


def test_cache_limit_covers_all_projects_and_emptying_keeps_other_files(tmp_path):
    """$CRYO_SAVANNA_CACHE may be a directory with other things in it: only cryo-savanna's own files go."""
    import os

    from cryo_savanna.images.cache import cache_files, cache_size_mb, clear_cache
    root = tmp_path / "scratch"
    (root / "results").mkdir(parents=True)
    (root / "results" / "keep.txt").write_text("mine")
    (root / "notes.npz").write_bytes(b"mine too")
    f = tmp_path / "a.mrc"
    write_mrc(f, np.ones((64, 64)))
    ref = ImageRef(str(f))
    old, new = ThumbCache(tmp_path / "p1", root=root), ThumbCache(tmp_path / "p2", root=root)
    old.get(ref, 32, lambda: read_image(ref), disk=True)
    new.get(ref, 32, lambda: read_image(ref), disk=True)
    files = cache_files(root)
    assert len(files) == 2 and cache_size_mb(root) > 0
    old_file = next(p for _, _, p in files if p.startswith(str(old.dir)))
    os.utime(old_file, (1, 1))
    new.max_disk = max(size for _, size, _ in files)  # room for one: the other project's older file goes
    new.prune()
    assert [p for _, _, p in cache_files(root)] == [p for _, _, p in files if p != old_file]
    clear_cache(root)
    assert cache_files(root) == [] and not old.dir.exists() and not new.dir.exists()
    assert (root / "results" / "keep.txt").read_text() == "mine" and (root / "notes.npz").exists()


def test_iterations_of_a_continued_run(tmp_path):
    """MultiBody (and every refine job before RELION 3.1) names a continued run run_ct<iteration>_..."""
    from cryo_savanna.project.summary import _ITER_RE
    from cryo_savanna.results.classes import iterations
    for name in ("run_it005_half1_model.star", "run_ct5_it006_half1_model.star", "run_ct5_it007_half1_model.star",
                 "other_it009_model.star"):
        (tmp_path / name).touch()
    assert [(it, p.name) for it, p in iterations(tmp_path)] == [
        (5, "run_it005_half1_model.star"), (6, "run_ct5_it006_half1_model.star"), (7, "run_ct5_it007_half1_model.star")]
    assert _ITER_RE.match("run_ct5_it007_half1_model.star").group(1) == "007"


def test_a_continued_run_wins_for_the_iterations_both_runs_have(tmp_path):
    """Continued from iteration 5 of a run that went on to 10: iterations 6-10 are the continuation's."""
    from cryo_savanna.project.pipeline import Process
    from cryo_savanna.project.snapshot import JobInfo
    from cryo_savanna.project.summary import SummaryReader
    job = tmp_path / "MultiBody" / "job020"
    job.mkdir(parents=True)
    for it in range(1, 11):
        (job / f"run_it{it:03d}_half1_model.star").touch()
    for it in range(6, 9):
        (job / f"run_ct5_it{it:03d}_half1_model.star").touch()
    (job / "run_ct7_it008_half1_model.star").touch()
    found = dict(iterations(job))
    assert found[5].name == "run_it005_half1_model.star"
    assert found[6].name == "run_ct5_it006_half1_model.star" and found[8].name == "run_ct7_it008_half1_model.star"
    assert max(found) == 8  # iterations 9 and 10 belong to the run that was replaced
    pr = Process("MultiBody/job020/", "None", "relion.multibody", "Running")
    s = SummaryReader(tmp_path).summarize(JobInfo(pr, None, [], status="Running"), None)
    assert s.iteration == 8


def test_a_failed_cache_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    (tmp_path / "img.mrc").write_bytes(b"")
    cache = ThumbCache(tmp_path, root=tmp_path / "cache")

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(np, "savez", full_disk)
    img = cache.get(ImageRef(str(tmp_path / "img.mrc")), 64, lambda: np.ones((4, 4), np.float32), disk=True)
    assert img.shape == (4, 4)
    assert not [f for f in (tmp_path / "cache").rglob("*") if f.is_file()]


def test_stale_cache_temp_files_are_removed(tmp_path):
    from cryo_savanna.images.cache import clear_cache, stale_temp_files
    c = ThumbCache(tmp_path / "proj", root=tmp_path / "cache")
    c.dir.mkdir(parents=True)
    old, new = c.dir / "a.1.2.tmp", c.dir / "b.1.2.tmp"
    old.write_bytes(b"half")
    new.write_bytes(b"being written")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    assert stale_temp_files(c.root) == [str(old)]
    c.prune()
    assert not old.exists() and new.exists()
    os.utime(new, (time.time() - 120, time.time() - 120))
    clear_cache(c.root)
    assert not new.exists() and not c.dir.exists()


def test_read_compact_keeps_numbers_as_arrays(spa_project, monkeypatch):
    from cryo_savanna.results import tables
    path = spa_project / "Extract/job005/particles.star"
    full = Table.read(path, "particles")
    monkeypatch.setattr(tables, "COMPACT_CHUNK", 7)  # several chunks
    t = tables.read_compact(path, "particles")
    assert t.columns == full.columns and len(t) == len(full)
    for c in t.columns:
        if full.numeric(c) is not None and np.isfinite(full.numeric(c)).all():
            assert isinstance(t.column(c), np.ndarray) and np.allclose(t.numeric(c), full.numeric(c))
        else:
            assert t.column(c) == full.column(c)
    assert t.numeric("rlnCoordinateX") is t.column("rlnCoordinateX")  # no copy
    # a column that turns out to hold text after some rows of numbers becomes text
    f = spa_project.parent / "mixed.star"
    f.write_text("data_x\nloop_\n_a #1\n_b #2\n" + "".join(f"{i} 1.5\n" for i in range(10)) + "x 2\n")
    t = tables.read_compact(f, "x")
    assert t.column("a")[-1] == "x" and t.column("a")[3] == "3" and isinstance(t.column("b"), np.ndarray)


# ---------------------------------------------------------------------------- CtfRefine

def test_ctf_refine_optics_before_and_after(spa_project):
    from cryo_savanna.results import ctfrefine as C
    out = spa_project / "CtfRefine/job014/particles_ctf_refine.star"
    g, = C.read_optics(out, spa_project / "Refine3D/job011/run_data.star")
    assert (g.number, g.name, g.pixel_size, g.box) == (1, "opticsGroup1", 3.54, 64)
    assert g.tilt_before == (0.0, 0.0) or g.tilt_before is None
    assert g.tilt_after == (0.42, -0.27) and g.tilt_changed
    assert [(z.n, z.m, z.name) for z in g.odd][2] == (3, -3, "trefoil")
    assert [z.after for z in g.odd] == [0.0, 0.0, 0.05, -0.31, 0.19, 0.08] and g.odd[0].before is None
    assert next(z for z in g.even if (z.n, z.m) == (4, 0)).after == 0.35
    assert g.mag_after is None


def test_zernike_order_is_relions():
    from cryo_savanna.results.ctfrefine import even_index, odd_index, parse_vector
    assert [odd_index(i) for i in range(6)] == [(1, -1), (1, 1), (3, -3), (3, -1), (3, 1), (3, 3)]
    assert [even_index(i) for i in range(9)] == [(0, 0), (2, -2), (2, 0), (2, 2), (4, -4), (4, -2), (4, 0),
                                                 (4, 2), (4, 4)]
    assert parse_vector("[0.1,-2e-3,3]") == [0.1, -0.002, 3.0] and parse_vector("[]") == []


def test_magnification_anisotropy_and_scale():
    from cryo_savanna.results.ctfrefine import Magnification
    m = Magnification(np.array([[1.01, 0.0], [0.0, 0.99]]))
    assert m.anisotropy == pytest.approx(2.0) and m.mean == pytest.approx(1.0)
    assert Magnification(np.eye(2)).is_identity


def test_ctf_refine_particle_changes_match_by_image_name(spa_project):
    from cryo_savanna.results import ctfrefine as C
    out = spa_project / "CtfRefine/job014/particles_ctf_refine.star"
    inp = spa_project / "Refine3D/job011/run_data.star"
    ch = C.read_particle_changes(out, inp)
    t = ch.table
    assert ch.matched == len(t) == 288 and ch.has_changes  # the output is in another order than the input
    before = Table.read(inp, "particles")
    pos = {n: i for i, n in enumerate(before.column("rlnImageName"))}
    k = 5
    j = pos[t.column("rlnImageName")[k]]
    u0, v0 = before.numeric("rlnDefocusU")[j], before.numeric("rlnDefocusV")[j]
    u, v = t.numeric("rlnDefocusU")[k], t.numeric("rlnDefocusV")[k]
    assert t.numeric(C.D_DEFOCUS)[k] == pytest.approx((u + v) / 2 - (u0 + v0) / 2)
    assert t.numeric(C.D_ASTIG)[k] == pytest.approx(abs(u - v) - abs(u0 - v0))
    assert np.all(np.abs(t.numeric(C.D_ANGLE)) <= 90)
    s = C.stat(t, C.D_DEFOCUS)
    assert s.n == 288 and 50 < s.sd < 500
    mics = C.per_micrograph(t, [C.D_DEFOCUS])
    assert len(mics.names) == 24 and mics.particles.sum() == 288
    # without the input, only the new values
    alone = C.read_particle_changes(out)
    assert alone.matched == 0 and not alone.has_changes and alone.table.has(C.DEFOCUS)


def test_ctf_refine_aberration_images(tmp_path, spa_project):
    from cryo_savanna.results import ctfrefine as C
    sets = C.aberration_images(spa_project, "CtfRefine/job014/")
    assert [s.group for s in sets] == [1]
    rows = sets[0].rows
    assert [r.title.split(" (")[0] for r in rows] == ["Antisymmetric aberrations", "Symmetric aberrations"]
    assert rows[0].fit.path.endswith("iter-fit_optics-group_1_N-3.mrc") and rows[0].order == 3
    assert rows[1].observed.path == "CtfRefine/job014/aberr_delta-phase_per-pixel_optics-group_1.mrc"
    # magnification, and group 1 not mistaken for group 11; of two fits the higher order
    job = tmp_path / "CtfRefine/job020"
    job.mkdir(parents=True)
    for name in ("mag_disp_x_optics-group_1.mrc", "mag_disp_x_fit_optics-group_1.mrc",
                 "mag_disp_y_optics-group_1.mrc", "mag_disp_y_fit_optics-group_1.mrc",
                 "beamtilt_delta-phase_per-pixel_optics-group_11.mrc",
                 "beamtilt_delta-phase_iter-fit_optics-group_11.mrc",
                 "beamtilt_delta-phase_iter-fit_optics-group_11_N-3.mrc"):
        write_mrc(job / name, np.zeros((4, 4)))
    sets = C.aberration_images(tmp_path, "CtfRefine/job020/")
    assert [s.group for s in sets] == [1, 11]
    assert [r.title for r in sets[0].rows] == ["Magnification: X displacement", "Magnification: Y displacement"]
    assert sets[0].rows[0].observed.kind == "displacement"
    assert sets[1].rows[0].fit.path.endswith("_N-3.mrc")
    res = C.residual(np.array([3.0]), np.array([-3.0]), "phase")
    assert res[0] == pytest.approx(6.0 - 2 * np.pi)


def test_ctf_refine_what_was_fitted():
    from cryo_savanna.results.ctfrefine import fits_ctf, fitted
    opts = {"do_ctf": "Yes", "do_defocus": "Per-particle", "do_astig": "No", "do_bfactor": "Per-micrograph",
            "do_phase": "No", "do_tilt": "Yes", "do_trefoil": "No", "do_4thorder": "No", "do_aniso_mag": "No"}
    assert fitted(opts) == [("Defocus", "per-particle"), ("B-factor", "per-micrograph"),
                            ("Beam tilt", "per optics group")]
    assert fits_ctf(opts) and fits_ctf({})
    aniso = dict(opts, do_aniso_mag="Yes")  # RELION then fits the magnification only
    assert fitted(aniso) == [("Anisotropic magnification", "per optics group")] and not fits_ctf(aniso)
    assert fitted(dict(opts, do_ctf="No", do_tilt="No")) == []


def test_ctf_refine_running_job_counts_finished_micrographs(spa_project):
    from cryo_savanna.results.ctfrefine import micrographs_done
    assert micrographs_done(spa_project / "CtfRefine/job014") == 3
