# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""3D maps: binning, sections, levels, the software isosurface, the volume viewer, ChimeraX scripts."""

import os
import stat
import time

import mrcfile
import numpy as np
import pytest

from cryo_savanna.images.mrc import ImageError
from cryo_savanna.images.render import framing, render, rotation
from cryo_savanna.images.volume import colormap, load_volume, outline, projection, resolution_range, section
from cryo_savanna.viewers.external import chimerax_script, find_chimerax


def write_map(path, data, voxel=1.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(data, np.float32))
        m.voxel_size = voxel


def ball(n: int, r: float, centre=(0, 0, 0)) -> np.ndarray:
    z, y, x = np.mgrid[:n, :n, :n] - n / 2
    return np.exp(-((x - centre[2]) ** 2 + (y - centre[1]) ** 2 + (z - centre[0]) ** 2) / (2 * r * r)).astype(np.float32)


# ---------------------------------------------------------------------------- no Qt

def test_load_bin_sections_and_levels(tmp_path):
    vol = np.arange(40 * 30 * 20, dtype=np.float32).reshape(20, 30, 40)
    write_map(tmp_path / "m.mrc", vol, 2.0)
    v = load_volume(tmp_path / "m.mrc", max_size=20)
    assert v.shape == (20, 30, 40) and v.factor == 2 and v.data.shape == (10, 15, 20)
    assert v.binned_voxel == 4.0 and v.data[0, 0, 0] == pytest.approx(vol[:2, :2, :2].mean())
    assert np.array_equal(section(tmp_path / "m.mrc", 0, 5), vol[5])
    assert np.array_equal(section(tmp_path / "m.mrc", 1, 7), vol[:, 7, :])
    assert np.array_equal(section(tmp_path / "m.mrc", 2, 99), vol[:, :, 39])  # clamped
    assert projection(v.data, 0).shape == (15, 20)
    full = load_volume(tmp_path / "m.mrc")
    level = full.level_for_fraction(0.01)
    assert (full.data > level).mean() == pytest.approx(0.01, abs=0.002)
    write_map(tmp_path / "flat.mrc", vol[0])
    with pytest.raises(ImageError):
        load_volume(tmp_path / "flat.mrc")


def test_mask_outline_and_colours():
    m = np.zeros((9, 9), np.float32)
    m[2:7, 2:7] = 1
    e = outline(m)
    assert e.sum() == 16 and e[2, 2] and not e[4, 4] and not e[0, 0]
    rgb = colormap(np.array([3.0, 6.0, 9.0]), 3.0, 9.0)
    assert rgb[0][2] > rgb[0][0] and rgb[2][0] > rgb[2][2]  # blue best, red worst
    loc = np.full((10, 10, 10), 20.0)
    loc[3:7, 3:7, 3:7] = np.linspace(3, 5, 64).reshape(4, 4, 4)
    mask = (loc < 10).astype(np.float32)
    lo, hi = resolution_range(loc, mask)
    assert 3 <= lo < 3.2 and 4.8 < hi <= 5  # the 20 Å outside the mask does not stretch the colours


def test_isosurface_render():
    vol = ball(64, 8) + 0.8 * ball(64, 5, (0, 0, 18))
    r = render(vol, 0.5, rotation(0, 0), 96)
    assert r.rgb.shape == (96, 96, 3) and 0.02 < r.hit.mean() < 0.5
    framed = render(vol, 0.5, rotation(0, 0), 96, frame=framing(vol, 0.5))
    assert framed.hit.mean() > 2 * r.hit.mean()  # framing fills the picture
    # looking along z, the second ball (offset in x) is to the right of the first
    cols = np.nonzero(framed.hit.any(axis=0))[0]
    rows = np.nonzero(framed.hit.any(axis=1))[0]
    assert (cols.max() - cols.min()) > (rows.max() - rows.min())
    # turned 90° about y it is in front or behind: the picture is narrower
    side = render(vol, 0.5, rotation(90, 0), 96, frame=framing(vol, 0.5))
    assert side.hit.any(axis=0).sum() < framed.hit.any(axis=0).sum()
    # lit from the upper left: the upper-left of the ball is brighter than the lower-right
    g = framed.rgb.astype(float).sum(axis=2)
    hit = np.argwhere(framed.hit)
    c = hit.mean(axis=0).astype(int)
    assert g[c[0] - 6, c[1] - 6] > g[c[0] + 6, c[1] + 6]
    # nothing above the level: an empty picture
    assert not render(vol, 10.0, rotation(0, 0), 32).hit.any()
    # coloured by a second map
    col = render(vol, 0.5, rotation(0, 0), 64, color=np.full(vol.shape, 9.0), color_range=(3, 9))
    px = col.rgb[col.hit]
    assert (px[:, 0] > px[:, 2]).all()  # all red: the worst resolution


def test_render_speed():
    vol = ball(160, 30)
    t = time.perf_counter()
    render(vol, 0.5, rotation(30, -20), 112, frame=framing(vol, 0.5))
    fast = time.perf_counter() - t
    t = time.perf_counter()
    render(vol, 0.5, rotation(30, -20), 320, frame=framing(vol, 0.5))
    full = time.perf_counter() - t
    assert fast < 0.5 and full < 3.0, (fast, full)


def test_chimerax_script_and_lookup(tmp_path):
    s = chimerax_script("/p/map.mrc", 0.0123, mask="/p/mask.mrc", color="/p/locres.mrc", color_range=(3.1, 6.25))
    assert s.splitlines()[0] == 'open "/p/map.mrc"' and "volume #1 style surface level 0.0123" in s
    assert "volume #2 style mesh level 0.5" in s and "volume #3 hide" in s
    assert "color sample #1 map #3 palette blue:cyan:green:yellow:red range 3.1,6.25" in s
    assert chimerax_script("/p/m.mrc").splitlines() == ['open "/p/m.mrc"', "volume #1 style surface", "view"]
    exe = tmp_path / "bin" / "chimerax"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    assert find_chimerax({"PATH": str(exe.parent)}) == str(exe)
    assert find_chimerax({"PATH": "", "CRYO_SAVANNA_CHIMERAX": str(exe)}) == str(exe)
    if not os.path.exists("/Applications"):
        assert find_chimerax({"PATH": ""}) is None


# ---------------------------------------------------------------------------- the viewer

@pytest.fixture
def settle(qtbot):
    pytest.importorskip("pytestqt")
    from qtpy.QtWidgets import QApplication
    from cryo_savanna.viewers.tasks import pool

    def run(rounds: int = 6):
        for _ in range(rounds):
            pool().waitForDone(20000)
            QApplication.processEvents()
            qtbot.wait(15)
    return run


def make_viewer(qtbot, project, maps, **kw):
    from cryo_savanna.viewers.volume_viewer import VolumeViewer
    v = VolumeViewer(project, maps, **kw)
    qtbot.addWidget(v)
    v.resize(900, 700)
    v.show()
    qtbot.waitExposed(v)
    return v


def test_sections_cross_and_surface(qtbot, tmp_path, settle):
    write_map(tmp_path / "Refine3D/job001/run_class001.mrc", ball(48, 6) + 0.7 * ball(48, 4, (10, 0, 0)))
    v = make_viewer(qtbot, tmp_path, ["Refine3D/job001/run_class001.mrc"])
    settle()
    assert v.vol is not None and v.pos == [24, 24, 24]
    assert [s.img.image.shape for s in v.slices] == [(48, 48)] * 3
    assert v.surface.image is not None and v.surface.hit_fraction > 0.02
    assert "+" in v.level_info.text() and "σ" in v.level_info.text()
    # clicking in the XY section moves the XZ and YZ sections
    v._on_click(0, 10.0, 30.0)
    settle()
    assert v.pos == [24, 10, 30] and v.slices[1].slider.value() == 10 and v.slices[2].slider.value() == 30
    assert np.allclose(v.slices[1].img.image, v.vol.data[:, 10, :])
    # projections
    v.mode.setCurrentIndex(1)
    assert np.allclose(v.slices[0].img.image, v.vol.data.sum(axis=0)) and not v.slices[0].hline.isVisible()
    # a higher level: a smaller surface
    before = v.surface.hit_fraction
    v.level.setValue(0.9)
    settle()
    assert v.surface.hit_fraction < before
    # rotating with the mouse renders at reduced size first, then in full
    v.surface.rot = rotation(70, -20)
    v.surface._request(fast=True)
    settle()
    assert v.surface.image.width() == v.surface.FAST
    qtbot.wait(300)
    settle()
    assert v.surface.image.width() > v.surface.FAST


def test_dragging_rotates_like_chimerax():
    """The side facing you follows the mouse, about the screen's axes, whatever the current view."""
    from cryo_savanna.images.render import rotation
    from cryo_savanna.viewers.volume_viewer import drag_rotation
    front = np.array([0.0, 0.0, -1.0])  # screen: x right, y down, z into the screen
    for start in (np.eye(3), rotation(30, -20), rotation(120, 75)):
        point = start.T @ front  # the map point facing the viewer now
        right = drag_rotation(start, 20, 0) @ point
        assert right[0] > 0.1 and abs(right[1]) < 1e-9  # dragged right: it moved right, not up or down
        down = drag_rotation(start, 0, 20) @ point
        assert down[1] > 0.1 and abs(down[0]) < 1e-9  # dragged down: it moved down
        r = start
        for _ in range(500):
            r = drag_rotation(r, 3, -2)
        assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1)


def test_renders_do_not_pile_up_on_a_slow_machine(qtbot, tmp_path, settle, monkeypatch):
    """Mouse moves faster than renders: one render at a time, the latest view, then the full one."""
    import cryo_savanna.viewers.volume_viewer as vv
    write_map(tmp_path / "m.mrc", ball(48, 6) + 0.7 * ball(48, 4, (10, 0, 0)))
    v = make_viewer(qtbot, tmp_path, ["m.mrc"])
    settle()
    calls = []
    real = vv.render

    def slow(*a, **kw):
        calls.append(a[2].copy())
        time.sleep(0.1)
        return real(*a, **kw)

    monkeypatch.setattr(vv, "render", slow)
    shown = []
    v.surface.rendered.connect(lambda: shown.append(v.surface.image.width()))
    for i in range(30):  # 30 moves in about 0.3 s
        v.surface.rot = vv.drag_rotation(v.surface.rot, 6, 0)
        v.surface._request(fast=True)
        qtbot.wait(10)
    last = v.surface.rot.copy()
    v.surface._request(fast=False)  # mouse released
    qtbot.waitUntil(lambda: shown and shown[-1] > v.surface.FAST and not v.surface._busy, timeout=5000)
    assert len(calls) <= 6, len(calls)  # not 31
    assert 1 <= sum(1 for w in shown if w == v.surface.FAST)  # frames were shown while rotating
    assert np.allclose(calls[-1], last)  # the latest view


def test_slider_reads_only_its_own_section(qtbot, tmp_path, settle, monkeypatch):
    import cryo_savanna.viewers.volume_viewer as vv
    write_map(tmp_path / "big.mrc", ball(400, 60))
    v = make_viewer(qtbot, tmp_path, ["big.mrc"])
    settle()
    reads = []
    real = vv.section
    monkeypatch.setattr(vv, "section", lambda p, a, i: reads.append(a) or real(p, a, i))
    v.slices[1].slider.setValue(150)
    settle()
    assert reads == [1] and v.slices[0].hline.value() == 150.5  # the XY crosshair moved


def test_mask_outline_and_local_resolution(qtbot, tmp_path, settle):
    n = 40
    write_map(tmp_path / "LocalRes/job017/relion_locres_filtered.mrc", ball(n, 6))
    z, y, x = np.mgrid[:n, :n, :n] - n / 2
    locres = (3 + np.sqrt(x * x + y * y + z * z) / 5).astype(np.float32)
    write_map(tmp_path / "LocalRes/job017/relion_locres.mrc", locres)
    write_map(tmp_path / "MaskCreate/job012/mask.mrc", (ball(n, 9) > 0.3).astype(np.float32))
    v = make_viewer(qtbot, tmp_path, ["LocalRes/job017/relion_locres_filtered.mrc"],
                    mask="MaskCreate/job012/mask.mrc", color="LocalRes/job017/relion_locres.mrc")
    settle()
    assert v.mask is not None and v.locres is not None
    lo, hi = v.color_range
    assert 3 <= lo < hi < 6 and v.bar_lo.text().endswith("Å")
    ov = v.slices[0].overlay.image
    assert ov is not None and (ov[..., 3] == 255).any() and (ov[..., 3] == 110).any()
    assert v.surface.color is not None
    v.show_color.setChecked(False)
    v.show_mask.setChecked(False)
    settle()
    assert v.slices[0].overlay.image is None and v.surface.color is None
    v.show_mask.setChecked(True)
    v.show_color.setChecked(True)
    s = v.chimerax_command()
    assert "relion_locres.mrc" in s and "mask.mrc" in s and f"level {v.level.value():.6g}" in s


def test_mismatched_mask_is_left_out(qtbot, tmp_path, settle):
    write_map(tmp_path / "a/map.mrc", ball(32, 5))
    write_map(tmp_path / "a/mask.mrc", np.ones((24, 24, 24)))
    v = make_viewer(qtbot, tmp_path, ["a/map.mrc"], mask="a/mask.mrc")
    settle()
    assert v.mask is None and "differs" in v.info.text() and not v.show_mask.isEnabled()


def test_big_map_sections_from_the_file(qtbot, tmp_path, settle):
    write_map(tmp_path / "big.mrc", ball(400, 60))
    v = make_viewer(qtbot, tmp_path, ["big.mrc"])
    settle()
    assert v.vol.factor == 3 and v.vol.data.shape == (133, 133, 133)
    assert v.slices[0].img.image.shape == (400, 400)  # the section at full resolution
    assert "binned by 3" in v.info.text()


def test_map_changes_while_open_and_switching_maps(qtbot, tmp_path, settle):
    write_map(tmp_path / "j/class1.mrc", ball(32, 5))
    write_map(tmp_path / "j/class2.mrc", ball(32, 8))
    v = make_viewer(qtbot, tmp_path, ["j/class1.mrc", "j/class2.mrc"])
    settle()
    small = v.surface.hit_fraction
    v.map_combo.setCurrentIndex(1)
    settle()
    assert v.vol.path.endswith("class2.mrc")
    v.pos = [5, 5, 5]
    time.sleep(0.01)
    write_map(tmp_path / "j/class2.mrc", ball(32, 8) * 2)
    v.refresh()
    settle()
    assert v.pos == [5, 5, 5] and v.vol.maximum == pytest.approx(2.0, abs=0.05)
    assert small > 0


def test_maps_in_the_window(qtbot, tmp_path, spa_copy, settle):
    from qtpy.QtCore import QSettings
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.project.slurm import SlurmClient
    from cryo_savanna.ui.main_window import MainWindow
    from cryo_savanna.viewers.volume_viewer import VolumeViewer
    w = MainWindow(spa_copy, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat), slurm=SlurmClient(which=lambda p: None))
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.load_now()
    w.select_job("PostProcess/job013/")
    assert w.info.view_titles() == ["FSC", "Maps", "Files"]
    v = w.info.show_view("Maps")
    settle()
    assert isinstance(v, VolumeViewer) and v.map_combo.currentText() == "postprocess_masked.mrc"
    assert v.mask is not None and v.show_mask.isChecked()
    w.select_job("MaskCreate/job012/")
    v = w.info.show_view("Maps")
    settle()
    assert v.map_combo.count() == 2 and v.mask is not None
    v.map_combo.setCurrentIndex(1)  # the mask itself: shown at level 0.5
    settle()
    assert v.level.value() == 0.5
    f = w.info.show_view("Files")
    assert f.select_file("mask.mrc")
    settle()
    assert isinstance(f.stack.currentWidget(), VolumeViewer) and f.volume.vol is not None


# ---------------------------------------------------------------------------- angular distribution in 3D

def test_bild_files_and_their_maps(spa_project):
    from cryo_savanna.results.bild import angdist_pairs, read_bild
    b = read_bild(spa_project / "Refine3D/job011/run_class001_angdist.bild")
    r = 32 * 3.54 / 2  # RELION: R = box × pixel size / 2, around (R, R, R)
    assert 0 < len(b) <= 192 and b.start.shape == b.end.shape == (len(b), 3)
    assert np.allclose(np.linalg.norm(b.start - r, axis=1), r, atol=1e-3)
    assert (np.linalg.norm(b.end - r, axis=1) <= 1.3 * r + 1e-3).all() and np.allclose(b.radius, b.radius[0])
    assert (b.color >= 0).all() and (b.color <= 1).all() and np.allclose(b.color[:, 1], 0)  # blue … red
    assert [(p.label, p.map) for p in angdist_pairs(spa_project, "Refine3D/job011/")] == [
        ("Final, class 1", "Refine3D/job011/run_class001.mrc"),
        ("Iteration 15 (half 1), class 1", "Refine3D/job011/run_it015_half1_class001.mrc")]
    assert [p.label for p in angdist_pairs(spa_project, "Class3D/job009/")] == [
        f"Iteration 25, class {c}" for c in range(1, 5)]
    assert angdist_pairs(spa_project, "Class2D/job006/") == []


def test_bild_cylinders_on_the_map(qtbot, tmp_path, settle):
    """Cylinders in front of the map cover it, those behind are hidden by it, those beside it show."""
    from cryo_savanna.images.volume import load_volume
    from cryo_savanna.results.bild import Bild
    from cryo_savanna.viewers.angdist_view import AngdistSurface, bild_to_voxels
    write_map(tmp_path / "m.mrc", ball(48, 6), 1.5)
    vol = load_volume(tmp_path / "m.mrc")
    r = 48 * 1.5 / 2

    def bild(*cyls):  # (direction, rgb)
        d = np.array([c[0] for c in cyls], float)
        return Bild(r * d + r, 1.3 * r * d + r, np.full(len(cyls), 2.0), np.array([c[1] for c in cyls], float))

    start, end, radius, centre = bild_to_voxels(bild(((1, 0, 0), (1, 0, 0))), vol)
    assert np.allclose(centre, 24) and np.allclose(start, [[48, 24, 24]]) and np.allclose(radius, 2 / 1.5)
    s = AngdistSurface()
    qtbot.addWidget(s)
    s.resize(420, 420)
    s.show()
    qtbot.waitExposed(s)
    red, green, blue = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    # seen along z (into the screen): +x beside the map, -z towards you (in front), +z away (behind)
    s.set_bild(bild(((1, 0, 0), red), ((0, 0, -1), green)), vol)
    s.set_volume(vol.data, 0.5)
    s.view_along("z")
    settle()
    img = s.grab().toImage()
    side = min(s.width(), s.height())
    from qtpy.QtCore import QRectF
    target = QRectF((s.width() - side) / 2, (s.height() - side) / 2, side, side)
    (mid_x,), _, _ = s.project(((start + end) / 2), target)
    (c_xy,), _, _ = s.project(centre[None], target)

    def rgb(xy):
        c = img.pixelColor(int(round(xy[0] * img.devicePixelRatio())), int(round(xy[1] * img.devicePixelRatio())))
        return c.red(), c.green(), c.blue()
    rr, gg, bb = rgb(mid_x)
    assert rr > 120 and gg < 60 and bb < 60  # the cylinder beside the map
    rr, gg, bb = rgb(c_xy)
    assert gg > rr + 100 and gg > bb + 100  # the one in front (pointing at you: a disc) covers the map's centre
    s.set_bild(bild(((1, 0, 0), red), ((0, 0, 1), blue)), vol)  # now pointing away: behind the map
    s.set_volume(vol.data, 0.5)
    s.view_along("z")
    settle()
    img = s.grab().toImage()
    rr, gg, bb = rgb(c_xy)
    assert abs(rr - bb) < 40 and abs(gg - bb) < 40 and rr > 80  # the grey map, not the blue cylinder


def test_angular_distribution_tab_in_3d(qtbot, spa_copy, settle):
    from cryo_savanna.viewers.views import AnglesView
    v = AnglesView(spa_copy, "Refine3D/job011/")
    qtbot.addWidget(v)
    v.resize(1100, 600)
    v.show()
    settle()
    p = v.panel3d
    assert [p.combo.itemText(i) for i in range(p.combo.count())] == ["Final, class 1", "Iteration 15 (half 1), class 1"]
    assert p.surface.cylinders is not None and p.surface.image is not None and "directions" in p.info.text()
    assert p.surface.message == ""  # the map's surface is there
    # a new iteration while the tab is open: listed; the final one stays shown
    job = spa_copy / "Refine3D/job011"
    (job / "run_it018_half1_class001.mrc").write_bytes((job / "run_it015_half1_class001.mrc").read_bytes())
    (job / "run_it018_half1_class001_angdist.bild").write_text(
        (job / "run_it015_half1_class001_angdist.bild").read_text())
    os.utime(job, ns=(time.time_ns(), time.time_ns() + 10**9))
    v.refresh()
    settle()
    assert p.combo.itemText(1) == "Iteration 18 (half 1), class 1" and p.combo.currentText() == "Final, class 1"
    p.combo.setCurrentIndex(2)
    settle()
    assert "run_it015_half1_class001_angdist.bild" in p.info.text()
    # a job without .bild files says so
    from cryo_savanna.viewers.angdist_view import AngdistPanel
    none = AngdistPanel(spa_copy, "Class2D/job006/")
    qtbot.addWidget(none)
    assert none.combo.count() == 0 and "No .bild" in none.surface.message


# ---------------------------------------------------------------------------- a running job's maps

def test_latest_iteration_maps(tmp_path):
    """The maps of the newest iteration RELION has finished writing (its optimiser.star comes last)."""
    from cryo_savanna.results import registry as R
    from cryo_savanna.results.classes import latest_iteration_maps
    job = tmp_path / "Class3D/job009"
    job.mkdir(parents=True)
    for it in (1, 2, 3):
        (job / f"run_it{it:03d}_model.star").touch()
        for c in (2, 1):
            write_map(job / f"run_it{it:03d}_class{c:03d}.mrc", ball(8, 2))
    for it in (1, 2):
        (job / f"run_it{it:03d}_optimiser.star").touch()  # iteration 3 is still being written
    assert latest_iteration_maps(tmp_path, "Class3D/job009/") == [
        "Class3D/job009/run_it002_class001.mrc", "Class3D/job009/run_it002_class002.mrc"]
    refine = tmp_path / "Refine3D/job011"
    refine.mkdir(parents=True)
    (refine / "run_it004_half1_model.star").touch()
    (refine / "run_it004_optimiser.star").touch()
    for half in (2, 1):
        write_map(refine / f"run_it004_half{half}_class001.mrc", ball(8, 2))
    assert latest_iteration_maps(tmp_path, "Refine3D/job011/") == [
        "Refine3D/job011/run_it004_half1_class001.mrc", "Refine3D/job011/run_it004_half2_class001.mrc"]

    final = [("Class3D/job009/run_it025_class001.mrc", "DensityMap.mrc.relion.class3d")]
    for status in ("Running", "Failed"):  # running, or stopped before its final maps
        ctx = R.JobContext(tmp_path, "Class3D/job009/", "relion.class3d", status, final)
        [maps] = [v for v in R.views_for(ctx) if v.kind == R.VOLUME]
        assert maps.title == "Maps" and maps.args["latest"] and maps.args["job"] == "Class3D/job009/"
    write_map(job / "run_it025_class001.mrc", ball(8, 2))
    ctx = R.JobContext(tmp_path, "Class3D/job009/", "relion.class3d", "Succeeded", final)
    [maps] = [v for v in R.views_for(ctx) if v.kind == R.VOLUME]
    assert maps.args["maps"] == ["Class3D/job009/run_it025_class001.mrc"] and not maps.args.get("latest")


def test_maps_of_a_running_job_follow_its_iterations(qtbot, tmp_path, settle):
    """A new iteration replaces the maps shown, keeping the class, the position and the level."""
    from cryo_savanna.results import registry as R
    from cryo_savanna.viewers.views import build_view
    job = tmp_path / "Class3D/job009"

    def iteration(it: int, scale: float) -> None:
        for c in (1, 2):
            write_map(job / f"run_it{it:03d}_class{c:03d}.mrc", ball(32, 4 + 2 * c) * scale)
        (job / f"run_it{it:03d}_model.star").touch()
        (job / f"run_it{it:03d}_optimiser.star").touch()

    iteration(1, 1.0)
    ctx = R.JobContext(tmp_path, "Class3D/job009/", "relion.class3d", "Running")
    [spec] = [v for v in R.views_for(ctx) if v.kind == R.VOLUME]
    specs = [(s.key, s.args) for s in R.views_for(ctx)]
    v = build_view(spec, ctx, None)
    qtbot.addWidget(v)
    v.resize(900, 700)
    v.show()
    settle()
    assert v.map_combo.currentText() == "run_it001_class001.mrc" and "latest iteration" in v.info.text()
    v.map_combo.setCurrentIndex(1)
    settle()
    v.pos = [10, 11, 12]
    v.level.setValue(0.3)
    v.refresh()  # nothing new: nothing read again
    settle()
    assert v.vol.path.endswith("run_it001_class002.mrc")
    iteration(2, 2.0)
    v.refresh()
    settle()
    assert v.map_combo.currentText() == "run_it002_class002.mrc" and v.vol.path.endswith("run_it002_class002.mrc")
    assert v.pos == [10, 11, 12] and v.level.value() == pytest.approx(0.3)
    assert v.vol.maximum == pytest.approx(2.0, abs=0.05)
    assert [(s.key, s.args) for s in R.views_for(ctx)] == specs  # the same tabs: the window keeps the viewer


def test_maps_of_earlier_iterations(qtbot, tmp_path, settle):
    """Every iteration's maps can be shown: after the final maps of a finished job, or after the latest
    iteration of a running one (which stops following while an earlier iteration is shown)."""
    from cryo_savanna.results import registry as R
    from cryo_savanna.results.classes import iteration_maps
    from cryo_savanna.viewers.views import build_view
    job = tmp_path / "Class3D/job009"
    job.mkdir(parents=True)

    def iteration(it: int) -> None:
        for c in (1, 2):
            write_map(job / f"run_it{it:03d}_class{c:03d}.mrc", ball(16, 3) * it)
        (job / f"run_it{it:03d}_model.star").touch()
        (job / f"run_it{it:03d}_optimiser.star").touch()

    for it in (1, 2):
        iteration(it)
    assert iteration_maps(tmp_path, "Class3D/job009/") == [
        (it, [f"Class3D/job009/run_it{it:03d}_class001.mrc", f"Class3D/job009/run_it{it:03d}_class002.mrc"])
        for it in (1, 2)]
    ctx = R.JobContext(tmp_path, "Class3D/job009/", "relion.class3d", "Running")
    [spec] = [v for v in R.views_for(ctx) if v.kind == R.VOLUME]
    v = build_view(spec, ctx, None)
    qtbot.addWidget(v)
    v.show()
    settle()
    assert v.iterations_offered() == ["Latest iteration", "Iteration 2", "Iteration 1"]
    assert v.show_iteration("Iteration 1")
    settle()
    assert v.vol.path.endswith("run_it001_class001.mrc") and "latest iteration" not in v.info.text()
    iteration(3)
    v.refresh()
    settle()
    assert v.iterations_offered() == ["Latest iteration", "Iteration 3", "Iteration 2", "Iteration 1"]
    assert v.vol.path.endswith("run_it001_class001.mrc")  # not following while iteration 1 is shown
    v.show_iteration("Latest iteration")
    settle()
    assert v.vol.path.endswith("run_it003_class001.mrc")

    write_map(job / "run_class001.mrc", ball(16, 3))
    final = [("Class3D/job009/run_class001.mrc", "DensityMap.mrc.relion.class3d")]
    ctx = R.JobContext(tmp_path, "Class3D/job009/", "relion.class3d", "Succeeded", final)
    [spec] = [s for s in R.views_for(ctx) if s.kind == R.VOLUME]
    v = build_view(spec, ctx, None)
    qtbot.addWidget(v)
    settle()
    assert v.iterations_offered()[:2] == ["Final", "Iteration 3"] and v.vol.path.endswith("run_class001.mrc")
    v.show_iteration("Iteration 2")
    settle()
    assert v.vol.path.endswith("run_it002_class001.mrc") and v.map_combo.count() == 2


def test_yz_sections_of_a_big_map_keep_a_slab(tmp_path, monkeypatch):
    """A YZ section needs every XY plane: the columns around it are kept, so the next steps read nothing."""
    from cryo_savanna.images import volume
    from cryo_savanna.images.mrc import MrcReader
    a = np.random.default_rng(1).random((20, 12, 70), dtype=np.float32)
    f = tmp_path / "big.mrc"
    with mrcfile.new(f) as m:
        m.set_data(a)
    volume._slabs.clear()
    reads = []
    original = MrcReader.section
    monkeypatch.setattr(MrcReader, "section", lambda self, z: (reads.append(z), original(self, z))[1])
    for x in range(70):
        assert np.array_equal(section(f, 2, x), a[:, :, x])
    assert len(reads) == 20 * 3  # one read of the map per slab of 32 columns
    os.utime(f, ns=(time.time_ns() + 10**9,) * 2)  # rewritten: read again
    section(f, 2, 0)
    assert len(reads) == 20 * 4
