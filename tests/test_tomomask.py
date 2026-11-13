# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomogram mask jobs: the masks (raster, MRC, links between identical ones), the form, the runner, the Masks
view, and pytom's tomogram masks folder."""

import copy
import json
import os
import shutil
import subprocess

import mrcfile
import numpy as np
import pytest

from cryo_savanna import mask_runner as M
from cryo_savanna import pytom_runner as P
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.cli_tools import external_program
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import PytomCatalog
from cryo_savanna.jobdefs.tomomask_jobs import (
    TomoMaskCatalog,
    install_runner,
    mask_kind,
    mask_kind_of,
    mask_main_input,
    masks_job_of,
    runner_command,
)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import read_star
from cryo_savanna.results import registry as R

from tests.test_pytom import _fake_pytom, _project, match_values, run_runner as run_pytom

ENV = RelionEnv({})
DEFS = JobDefs.load()
CATALOG = TomoMaskCatalog(DEFS, ENV)
PYTOM = PytomCatalog(DEFS, ENV)
TOMOS = "Tomograms/job006/tomograms.star"
JOB = "External/job020/"
NAMES = ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"]


def reference(shape, g):
    """The mask the numpy way: voxel centres inside the polygon (even-odd) and between the planes."""
    nx, ny, nz = shape
    cx, cy, cz = nx / 2, ny / 2, nz / 2
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx] + 0.5
    pts = np.array(g["polygon"], float) + [cx, cy]
    X, Y = x[0], y[0]
    inside = np.zeros_like(X, bool)
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        cross = ((y1 <= Y) & (Y < y2)) | ((y2 <= Y) & (Y < y1))
        with np.errstate(all="ignore"):
            xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (X < xi)
    out = np.broadcast_to(inside, (nz, ny, nx)).copy()
    if g["z_mode"] == M.Z_SLAB:
        b = cz + g["bottom"][0] + g["bottom"][1] * (x - cx) + g["bottom"][2] * (y - cy)
        t = cz + g["top"][0] + g["top"][1] * (x - cx) + g["top"][2] * (y - cy)
        out &= (b <= z + 1e-6) & (z <= t + 1e-6)
    return out.astype(np.uint8)


def raster(shape, g):
    r = M.Raster(shape, g)
    return np.stack([np.frombuffer(r.plane(k), np.uint8).reshape(shape[1], shape[0]) for k in range(shape[2])])


POLY = [[-20, -18], [25, -10], [10, 22], [-28, 15], [-5, 2]]  # not convex


@pytest.mark.parametrize("g", [
    {"polygon": POLY, "z_mode": M.Z_SLAB, "bottom": [-6, 0.1, -0.05], "top": [5, -0.08, 0.12]},
    {"polygon": POLY, "z_mode": M.Z_ALL, "bottom": [0, 0, 0], "top": [0, 0, 0]},
    M.default_geometry((60, 50, 30), M.Z_SLAB, 10, 4),
    M.default_geometry((60, 50, 30)),
])
def test_raster_matches_numpy(g):
    shape = (60, 50, 30)
    assert np.array_equal(raster(shape, g), reference(shape, g))


def test_default_geometry_is_the_whole_tomogram():
    assert raster((20, 10, 6), M.default_geometry((20, 10, 6))).all()
    slab = raster((20, 10, 8), M.default_geometry((20, 10, 8), M.Z_SLAB, 4))
    assert slab[2:6].all() and not slab[:2].any() and not slab[6:].any()


def test_all_z_does_not_depend_on_the_planes():
    shape = (30, 20, 10)
    a = M.default_geometry(shape)
    b = dict(a, bottom=[-1, 0.5, 0], top=[1, 0, 0.2])
    assert M.geometry_key(shape, a) == M.geometry_key(shape, b)
    assert M.geometry_key(shape, a) != M.geometry_key(shape, dict(b, z_mode=M.Z_SLAB))


def test_written_mask_is_an_int8_mrc(tmp_path):
    shape, g = (40, 30, 12), {"polygon": POLY, "z_mode": M.Z_SLAB, "bottom": [-3, 0.1, 0], "top": [4, 0, 0.1]}
    sha = M.write_mask(str(tmp_path / "m.mrc"), shape, 6.75, g)
    with mrcfile.open(tmp_path / "m.mrc") as m:
        assert m.data.dtype == np.int8 and m.data.shape == (12, 30, 40)
        assert m.voxel_size.x == pytest.approx(6.75)
        assert np.array_equal(m.data, reference(shape, g))
        assert m.header.dmean == pytest.approx(m.data.mean(), abs=1e-6)
    assert P.mrc_header(str(tmp_path / "m.mrc")) == (40, 30, 12, pytest.approx(6.75))
    assert len(sha) == 40


# ---------------------------------------------------------------------------- the form

def test_form_round_trip(tomo_project):
    v = CATALOG.variant()
    assert [t.label for t in v.tabs] == ["Input", "Running"]
    values = v.default_values(ENV)
    values.update(in_tomograms=TOMOS, start_z=M.START_SLAB, start_thickness="40", tomo_names="TS_01 TS_03")
    pipeline = Pipeline.read(tomo_project)
    js = v.make_jobstar(values, False, JOB, pipeline)
    assert js.options["fn_exe"] == runner_command() == "python3 .cryo-savanna/mask_runner.py masks"
    assert TOMOS in js.options.values()  # in an External input field: the job hangs under its tomograms
    assert mask_kind(js.options["fn_exe"]) == "masks" and mask_kind_of(js) == "masks"
    assert mask_main_input(js) == TOMOS
    assert external_program(js.options["fn_exe"]) == "Tomogram masks"
    back, ignored = v.values_from_jobstar(js, ENV)
    assert not ignored and all(back[k] == values[k] for k in v.own_keys)


# ---------------------------------------------------------------------------- the runner

@pytest.fixture
def project(tomo_project, tmp_path):
    return _project(tomo_project, tmp_path)


def run_masks(project, job=JOB, **kw):
    v = CATALOG.variant()
    values = v.default_values(ENV)
    values.update(dict(in_tomograms=TOMOS, **kw) if "in_tomograms" not in kw else kw)
    js = v.make_jobstar(values, False, job, None)
    (project / job).mkdir(parents=True, exist_ok=True)
    js.write(project / job / "job.star")
    install_runner(project)
    cmd = f"{js.options['fn_exe']} --o {job} --j 1 {js.options['other_args']}"
    return subprocess.run(cmd, shell=True, cwd=project, capture_output=True, text=True, timeout=120)


def mask_file(project, name, job=JOB):
    return project / job / M.MASK_DIR / (name + P.TOMO_MASK_SUFFIX)


def tomogram_shape(project, name):
    rows = {r[P.TOMO_NAME]: r for r in P.loop_block(str(project / TOMOS), "global").dicts()}
    with mrcfile.open(project / rows[name]["rlnTomoReconstructedTomogram"], permissive=True,
                      header_only=True) as m:
        return int(m.header.nx), int(m.header.ny), int(m.header.nz)


def test_runner_writes_one_file_and_links_identical_masks(project):
    res = run_masks(project)
    assert res.returncode == 0, res.stderr + res.stdout
    d = project / JOB
    assert (d / P.SUCCESS).exists()
    files = [mask_file(project, n) for n in NAMES]
    assert all(f.is_file() for f in files)
    shapes = {n: tomogram_shape(project, n) for n in NAMES}
    real = [f for f in files if not f.is_symlink()]
    assert len(real) == len(set(shapes.values()))  # one file per tomogram size: the rest are links to it
    for n, f in zip(NAMES, files):
        if f.is_symlink():
            assert "/" not in os.readlink(f)  # relative, next to it
        with mrcfile.open(f) as m:
            assert m.data.shape == shapes[n][::-1] and m.data.all()
    nodes = read_star(d / P.OUTPUT_NODES)["output_nodes"].rows
    assert nodes == [[JOB + M.LIST, M.NODE_LIST]]
    listed = read_star(d / M.LIST)["global"]
    assert [r[0] for r in listed.rows] == NAMES
    assert listed.rows[0][1] == JOB + "masks/TS_01_mask.mrc"
    data = json.loads((d / M.GEOMETRY).read_text())
    assert data["in_tomograms"] == TOMOS and set(data["tomograms"]) == set(NAMES)
    assert "linked to an identical one" in res.stdout or len(real) == len(NAMES)


def test_save_rewrites_only_the_changed_masks(project):
    assert run_masks(project).returncode == 0
    d = str(project / JOB)
    data = M.read_geometry(d)
    shape = tuple(data["tomograms"]["TS_01"]["shape"])
    same_size = [n for n in NAMES if tuple(data["tomograms"][n]["shape"]) == shape]
    assert len(same_size) >= 3, "the fixture's tomograms should share a size"
    first, second, third = same_size[:3]
    real = [n for n in same_size if not mask_file(project, n).is_symlink()]
    assert len(real) == 1
    # the masks linked to a file whose own mask changes keep their mask
    owner = real[0]
    others = [n for n in same_size if n != owner]
    slab = M.default_geometry(shape, M.Z_SLAB, 4)
    data["tomograms"][owner]["geometry"] = slab
    done = M.save_masks(d, data)
    assert done[owner] == "written" and all(done[n] == "kept" for n in others)
    with mrcfile.open(mask_file(project, owner)) as m:
        assert not m.data.all() and m.data.any()
    for n in others:
        with mrcfile.open(mask_file(project, n)) as m:
            assert m.data.all()
    assert sum(not mask_file(project, n).is_symlink() for n in others) == 1
    # another with the same new mask: a link to the first; one with other numbers but the same voxels too
    a, b = others[0], others[1]
    data["tomograms"][a]["geometry"] = slab
    big = M.default_geometry(shape, M.Z_SLAB, 4, margin=-50)  # a larger rectangle: the same voxels
    data["tomograms"][b]["geometry"] = big
    assert M.geometry_key(shape, big) != M.geometry_key(shape, slab)
    done = M.save_masks(d, data)
    assert done[a] == "linked" and done[b] == "linked" and done[owner] == "kept"
    for n in (a, b):
        assert os.path.realpath(mask_file(project, n)) == os.path.realpath(mask_file(project, owner))
    # nothing changed: nothing written
    assert set(M.save_masks(d, M.read_geometry(d)).values()) == {"kept"}


def test_saving_from_elsewhere_names_the_masks_from_the_project(project, tmp_path, monkeypatch):
    """The Masks view saves from wherever cryo-savanna was started (--project): tomogram_masks.star, the
    job's output, still names the masks from the project, as the job itself wrote it."""
    assert run_masks(project).returncode == 0
    d = project / JOB
    before = (d / M.LIST).read_text()
    data = M.read_geometry(str(d))
    shape = tuple(data["tomograms"]["TS_01"]["shape"])
    for n in NAMES:
        if tuple(data["tomograms"][n]["shape"]) == shape:
            data["tomograms"][n]["geometry"] = M.default_geometry(shape, M.Z_SLAB, 4)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    done = M.save_masks(str(d), data, project_dir=str(project))
    assert "written" in done.values() and "linked" in done.values()
    assert (d / M.LIST).read_text() == before
    listed = read_star(d / M.LIST)["global"]
    assert listed.rows[0][1] == JOB + "masks/TS_01_mask.mrc" and all((project / r[1]).is_file() for r in listed.rows)
    # a mask linked to an identical one has that one's SHA-1 (the file's, taken when it was written)
    files = M.read_geometry(str(d))["files"]
    same = [n for n in NAMES if tuple(data["tomograms"][n]["shape"]) == shape]
    assert len({files[n]["sha1"] for n in same}) == 1 and files[same[0]]["sha1"]


def test_continue_keeps_the_drawn_masks(project):
    assert run_masks(project, tomo_names="TS_01").returncode == 0
    d = str(project / JOB)
    data = M.read_geometry(d)
    shape = tuple(data["tomograms"]["TS_01"]["shape"])
    data["tomograms"]["TS_01"]["geometry"] = M.default_geometry(shape, M.Z_SLAB, 6)
    M.save_masks(d, data)
    res = run_masks(project, start_z=M.START_SLAB, start_thickness="2")
    assert res.returncode == 0, res.stderr
    data = M.read_geometry(d)
    assert data["tomograms"]["TS_01"]["geometry"]["top"][0] == 3  # kept
    assert data["tomograms"]["TS_03"]["geometry"]["top"][0] == 1  # the new starting slab
    assert "1 masks kept" in res.stdout


def test_runner_fails_without_tomograms(project):
    res = run_masks(project, in_tomograms="nothing.star")
    assert res.returncode != 0 and "tomograms.star not found" in res.stderr
    assert (project / JOB / P.FAILURE).exists()


# ---------------------------------------------------------------------------- pytom's masks folder

def test_tomogram_mask_file(tmp_path):
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks/TS_01_mask.mrc").write_bytes(b"x")
    (tmp_path / "TS_01.mrc").write_bytes(b"x")
    folder = str(tmp_path / "masks")
    assert P.tomogram_mask_file({"tomogram_mask": folder}, "TS_01") == os.path.join(folder, "TS_01_mask.mrc")
    assert P.tomogram_mask_file({"tomogram_mask": str(tmp_path / "{name}.mrc")}, "TS_01") == str(tmp_path / "TS_01.mrc")
    assert P.tomogram_mask_file({"tomogram_mask": ""}, "TS_01") == ""
    with pytest.raises(P.Usage, match="no tomogram mask TS_03_mask.mrc in the masks folder"):
        P.tomogram_mask_file({"tomogram_mask": folder}, "TS_03")


def test_pytom_match_reads_the_masks_folder(project, tmp_path, monkeypatch):
    log = _fake_pytom(tmp_path, monkeypatch)
    assert run_masks(project).returncode == 0
    folder = JOB + M.MASK_DIR
    res = run_pytom(project, "match", "External/job021/", match_values(tomogram_mask=folder))
    assert res.returncode == 0, res.stderr + res.stdout
    match = [c.split() for c in log.read_text().splitlines() if c.startswith("match")]
    masks = sorted(c[c.index("--tomogram-mask") + 1] for c in match)
    assert masks == sorted(f"{folder}/{n}_mask.mrc" for n in NAMES)
    # a tomogram without its mask stops the job, naming it
    os.remove(mask_file(project, "TS_45"))
    res = run_pytom(project, "match", "External/job022/", match_values(tomogram_mask=folder))
    assert res.returncode != 0 and "TS_45: no tomogram mask TS_45_mask.mrc" in res.stderr


def test_pytom_match_needs_masks_of_the_tomogram_size(project, tmp_path, monkeypatch):
    _fake_pytom(tmp_path, monkeypatch)
    (project / "small").mkdir()
    for n in NAMES:
        M.write_mask(str(project / "small" / (n + P.TOMO_MASK_SUFFIX)), (8, 8, 4), 1.0, M.default_geometry((8, 8, 4)))
    res = run_pytom(project, "match", "External/job021/", match_values(tomogram_mask="small"))
    assert res.returncode != 0 and "need the same size" in res.stderr


def test_pytom_hangs_under_the_mask_job(tomo_project):
    p = Pipeline.read(tomo_project)
    node = JOB + M.LIST
    p.processes[JOB] = Process(JOB, "None", "relion.external", "Succeeded", outputs=[node])
    p.nodes[node] = Node(node, M.NODE_LIST, producer=JOB)
    assert masks_job_of(JOB + "masks/", p) == node and masks_job_of("./" + JOB + "masks", p) == node
    assert masks_job_of("elsewhere/masks", p) is None
    js = PYTOM.variant("match").make_jobstar(match_values(tomogram_mask=JOB + "masks"), False, "External/job021/", p)
    assert node in js.options.values()
    extract = PYTOM.variant("extract")
    vals = extract.default_values(ENV)
    vals.update(in_match="External/job015/pytom_match.star", tomogram_mask=JOB + "masks")
    assert node not in extract.make_jobstar(vals, False, "External/job022/", p).options.values()  # not used
    vals["tomogram_mask_mode"] = P.TMASK_OTHER
    assert node in extract.make_jobstar(vals, False, "External/job022/", p).options.values()


def test_folder_widget_chooses_a_folder(qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QFileDialog

    from cryo_savanna.ui.job_editor import FolderWidget, JobEditor, make_widget
    ed = JobEditor(ENV, {})
    qtbot.addWidget(ed)
    ed.project_dir = tmp_path
    (tmp_path / "External/job020/masks").mkdir(parents=True)
    opt = PYTOM.variant("match").options["tomogram_mask"]
    assert opt.kind == "folder"
    w = make_widget(opt, ed)
    assert isinstance(w, FolderWidget)
    asked = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a: asked.append(a) or str(tmp_path / "External/job020/masks"))
    w._browse()
    assert w.value() == "External/job020/masks" and asked


# ---------------------------------------------------------------------------- the Masks view

@pytest.fixture
def masks_done(project):
    assert run_masks(project).returncode == 0
    return project


def test_registry_gives_the_masks_view(masks_done):
    js = JobStar.read(masks_done / JOB / "job.star")
    ctx = R.JobContext(masks_done, JOB, "relion.external", "Succeeded", [], js.options)
    assert [v.title for v in R.views_for(ctx)] == ["Masks", "Files"]


def test_masks_view_draws_copies_and_saves(qtbot, masks_done):
    from cryo_savanna.viewers.tomo_mask_view import TomoMaskView
    v = TomoMaskView(masks_done, JOB)
    qtbot.addWidget(v)
    assert v.tasks.wait()
    assert v.viewer is not None and v.viewer.tasks.wait()
    assert v.viewer.tasks.wait()
    assert v.table.rowCount() == 5 and v.current == "TS_01"
    nx, ny, nz = v.shape()
    assert len(v.poly.getHandles()) == 4
    assert v._read_polygon(v.shape()) == [[-nx / 2, -ny / 2], [nx / 2, -ny / 2], [nx / 2, ny / 2], [-nx / 2, ny / 2]]
    assert not any(r.isVisible() for r in v.lines.values())  # all Z: no planes
    # a slab: the plane lines come; moving the top line in XZ tilts the plane along x
    v.zmode.setCurrentIndex(v.zmode.findData(M.Z_SLAB))
    assert v.geometry()["z_mode"] == M.Z_SLAB and all(r.isVisible() for r in v.lines.values())
    assert v.table.item(0, 4).text() == "edited" and v.save_btn.isEnabled()
    top = v.lines[("xz", "top")]
    v._set_line(top, (0, nz / 2 + 1), (nx, nz / 2 + 3))
    v._on_line("xz", "top")
    v.flush()
    z0, ax, ay = v.geometry()["top"]
    assert z0 == pytest.approx(2) and ax == pytest.approx(2 / nx) and ay == 0
    assert v.geometry("TS_03")["z_mode"] == M.Z_ALL  # only the tomogram shown
    # a smaller polygon
    v.poly.setPoints([(2, 2), (nx - 2, 2), (nx / 2, ny - 2)], closed=True)  # reported as finished: no _on_poly()
    v.flush()
    assert len(v.geometry()["polygon"]) == 3 and 0 < v.inside_fraction() < 1
    # the same for all: every tomogram gets it, and later edits go to all of them
    v.same.setChecked(True)
    assert all(v.geometry(n) == v.geometry("TS_01") for n in v.names())
    v.zmode.setCurrentIndex(v.zmode.findData(M.Z_ALL))
    assert all(v.geometry(n)["z_mode"] == M.Z_ALL for n in v.names())
    v.same.setChecked(False)  # from here, one by one
    v.table.selectRow(1)
    assert v.current == "TS_03"
    v._reset()
    assert v.geometry("TS_03")["polygon"] != v.geometry("TS_01")["polygon"]
    assert v.save(ask=False)
    assert v.tasks.wait(30)
    assert not v.dirty and "Saved" in v.info.text()
    with mrcfile.open(mask_file(masks_done, "TS_01")) as m:
        assert 0 < m.data.mean() < 1
    with mrcfile.open(mask_file(masks_done, "TS_03")) as m:
        assert m.data.all()
    data = M.read_geometry(str(masks_done / JOB))
    assert len(data["tomograms"]["TS_01"]["geometry"]["polygon"]) == 3
    listed = read_star(masks_done / JOB / M.LIST)["global"]  # named from the project, not from where this runs
    assert listed.rows[0][1] == JOB + "masks/TS_01_mask.mrc"
    v.refresh()  # its own save: not read again
    assert not v.tasks.busy()


def test_masks_view_takes_edits_back(qtbot, masks_done):
    from cryo_savanna.viewers.tomo_mask_view import TomoMaskView
    v = TomoMaskView(masks_done, JOB)
    qtbot.addWidget(v)
    assert v.tasks.wait() and v.viewer.tasks.wait()
    start = {n: copy.deepcopy(v.geometry(n)) for n in v.names()}
    assert not v.undo_btn.isEnabled() and not v.redo_btn.isEnabled()
    v.set_geometry(v.geometry())  # nothing changed: nothing to take back
    assert not v.undo_btn.isEnabled()
    v.zmode.setCurrentIndex(v.zmode.findData(M.Z_SLAB))  # edit 1: this tomogram
    slab = copy.deepcopy(v.geometry())
    v.same.setChecked(True)  # edit 2: the others get it
    assert all(v.geometry(n) == slab for n in v.names())
    nx, ny, _ = v.shape()
    v.poly.setPoints([(2, 2), (nx - 2, 2), (nx / 2, ny - 2)], closed=True)  # edit 3: all of them
    v.flush()
    assert all(len(v.geometry(n)["polygon"]) == 3 for n in v.names()) and v.dirty
    v.undo()
    assert all(v.geometry(n) == slab for n in v.names()) and len(v.poly.getHandles()) == 4
    v.undo()
    assert v.geometry("TS_01") == slab and v.geometry("TS_03") == start["TS_03"]
    assert v.table.item(1, 4).text() == "" and v.table.item(0, 4).text() == "edited"
    v.undo()
    assert {n: v.geometry(n) for n in v.names()} == start and not v.dirty and not v.save_btn.isEnabled()
    assert not v.undo_btn.isEnabled() and v.redo_btn.isEnabled()
    v.undo()  # nothing left: nothing happens
    v.redo()
    v.redo()
    assert all(v.geometry(n) == slab for n in v.names()) and v.dirty
    v._reset()  # the masks are as Reset makes them already: no edit, the way forward stays
    assert v.redo_btn.isEnabled()
    v.zmode.setCurrentIndex(v.zmode.findData(M.Z_ALL))  # a new edit: what was taken back cannot be made again
    assert not v.redo_btn.isEnabled() and v.undo_btn.isEnabled()
    v.load()  # read again (the job ran again): no way back to masks that are gone
    assert v.tasks.wait() and not v.undo_btn.isEnabled()


def test_masks_view_before_the_job_ran(qtbot, tmp_path):
    from cryo_savanna.viewers.tomo_mask_view import TomoMaskView
    (tmp_path / JOB).mkdir(parents=True)
    v = TomoMaskView(tmp_path, JOB)
    qtbot.addWidget(v)
    assert "when the job has run" in v.info.text() and not v.save()


def test_new_mask_job_in_window(qtbot, masks_done, tmp_path):
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(masks_done, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    assert "Tomogram masks (for pytom)" in [a.text() for a in w.actions.external_menu.actions()]
    w.actions.new_tomomask_job()
    ed = w.actions.editor
    assert ed.variant.kind == "masks" and "Tomogram masks" in ed.title.text()
    assert ed.values()["in_tomograms"].endswith("tomograms.star")
    shutil.rmtree(masks_done / ".cryo-savanna", ignore_errors=True)
