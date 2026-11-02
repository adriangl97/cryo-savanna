# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Selections made in cryo-savanna write the files RELION's interactive tools write (Select, Exclude tilt images)."""

import shutil

import numpy as np
import pytest

from cryo_savanna.project.selection import (
    SELECT_FILES,
    SelectionError,
    center_of_mass_to_center,
    read_backup_selection,
    read_class_input,
    read_tilt_exclusion,
    read_tilt_input,
    regroup_particles,
    write_class_selection,
    write_tilt_exclusion,
)
from cryo_savanna.project.star import StarBlock, format_star, read_star, write_star

OPT = "Class2D/job006/run_it200_optimiser.star"


def test_class_input_from_optimiser_and_model(spa_project):
    inp = read_class_input(spa_project, OPT)
    assert inp.model == "Class2D/job006/run_it200_model.star" and inp.data == "Class2D/job006/run_it200_data.star"
    assert inp.numbers == list(range(1, 51)) and inp.classes.columns[-1] == "rlnClassNumber"
    assert inp.writes_class_averages
    m = read_class_input(spa_project, "Class3D/job009/run_it025_model.star")
    assert m.data == "Class3D/job009/run_it025_data.star" and not m.writes_class_averages
    with pytest.raises(SelectionError):
        read_class_input(spa_project, "Class2D/job006/run_it025_model.star")  # no data.star for that iteration


def test_class_selection_files(spa_copy):
    inp = read_class_input(spa_copy, OPT)
    res = write_class_selection(inp, "Select/job099/", [12, 3, 7])
    parts = read_star(spa_copy / "Select/job099/particles.star")
    assert list(parts) == ["optics", "particles"]
    data = read_star(spa_copy / inp.data)
    assert format_star([parts["optics"]]) == format_star([data["optics"]])  # optics as they were
    p = parts["particles"]
    assert p.columns == data["particles"].columns
    classes = [int(c) for c in p.column("rlnClassNumber")]
    # class by class in class order (3, 7, 12), each class's particles in file order
    assert classes == sorted(classes) and set(classes) == {3, 7, 12}
    src = [r for r in data["particles"].rows if int(r[data["particles"].col_index("rlnClassNumber")]) == 7]
    assert [r for r in p.rows if int(r[p.col_index("rlnClassNumber")]) == 7] == src
    assert res.particles == len(p) and res.classes == 3
    avg = read_star(spa_copy / "Select/job099/class_averages.star")[""]
    model = read_star(spa_copy / inp.model, ["model_classes"])["model_classes"]
    assert avg.columns == model.columns + ["rlnClassNumber"]
    assert avg.column("rlnClassNumber") == ["3", "7", "12"]
    assert avg.rows[0][:-1] == model.rows[2]
    backup = read_star(spa_copy / "Select/job099/backup_selection.star")[""]
    assert backup.columns == ["rlnSelected"] and len(backup) == 50
    assert [i + 1 for i, v in enumerate(backup.column("rlnSelected")) if v == "1"] == [3, 7, 12]
    assert read_backup_selection(spa_copy, "Select/job099/", 50) == [n in (3, 7, 12) for n in range(1, 51)]
    assert read_backup_selection(spa_copy, "Select/job099/", 49) is None
    # the fixture's own Select job (made like RELION's) reads back too
    assert sum(read_backup_selection(spa_copy, "Select/job007/", 50)) == 5


def test_3d_selection_has_no_class_averages(spa_copy):
    inp = read_class_input(spa_copy, "Class3D/job009/run_it025_optimiser.star".replace("_optimiser", "_model"))
    res = write_class_selection(inp, "Select/job098/", [2])
    assert not (spa_copy / "Select/job098/class_averages.star").exists()
    assert res.files == ["Select/job098/particles.star", "Select/job098/backup_selection.star"]


def test_nothing_selected(spa_copy):
    with pytest.raises(SelectionError, match="No classes selected"):
        write_class_selection(read_class_input(spa_copy, OPT), "Select/job097/", [])
    assert not (spa_copy / "Select/job097").exists()


def test_rejected_selection_keeps_the_saved_one(spa_copy):
    """Classes without particles: nothing is written, so a job saved earlier stays consistent."""
    inp = read_class_input(spa_copy, OPT)
    write_class_selection(inp, "Select/job097/", [1, 2])
    files = {f: (spa_copy / "Select/job097" / f).read_bytes() for f in SELECT_FILES
             if (spa_copy / "Select/job097" / f).exists()}
    ni = inp.classes.col_index("rlnClassNumber")
    inp.classes.rows.append([*inp.classes.rows[0][:ni], "999", *inp.classes.rows[0][ni + 1:]])  # an empty class
    with pytest.raises(SelectionError, match="no particles"):
        write_class_selection(inp, "Select/job097/", [999])
    assert files == {f: (spa_copy / "Select/job097" / f).read_bytes() for f in files}


def test_recenter(spa_copy):
    img = np.zeros((32, 32), np.float32)
    img[20:24, 5:9] = 1  # mass around (21.5, 6.5); centre is (16, 16)
    out = center_of_mass_to_center(img)
    y, x = np.mgrid[:32, :32]
    assert abs((out * y).sum() / out.sum() - 16) < 0.1 and abs((out * x).sum() / out.sum() - 16) < 0.1
    assert out.sum() == pytest.approx(img.sum())
    # as RELION's MultidimArray::centerOfMass, negative pixels do not count
    neg = img.copy()
    neg[0:4, 28:32] = -5  # far away, negative
    assert np.allclose(center_of_mass_to_center(neg)[16:20, 14:18], out[16:20, 14:18])
    write_class_selection(read_class_input(spa_copy, OPT), "Select/job096/", [1, 2], recenter=True)
    avg = read_star(spa_copy / "Select/job096/class_averages.star")[""]
    assert avg.column("rlnReferenceImage") == ["000001@Select/job096/class_averages.mrcs",
                                               "000002@Select/job096/class_averages.mrcs"]
    import mrcfile
    with mrcfile.open(spa_copy / "Select/job096/class_averages.mrcs") as m:
        assert m.data.shape == (2, 48, 48)


def test_regroup(spa_copy):
    # give the data file groups and the model its group scales
    data = read_star(spa_copy / "Class2D/job006/run_it200_data.star")
    p = data["particles"]
    p.columns.append("rlnGroupNumber")
    for i, r in enumerate(p.rows):
        r.append(str(i % 8 + 1))
    write_star(spa_copy / "Class2D/job006/run_it200_data.star", data.values(), atomic=False)
    model = read_star(spa_copy / "Class2D/job006/run_it200_model.star")
    model["model_groups"] = StarBlock.from_records(
        "model_groups", ["rlnGroupNumber", "rlnGroupName", "rlnGroupNrParticles", "rlnGroupScaleCorrection"],
        [[g, f"group_{g}", 120, 1.0 + 0.1 * ((g * 5) % 8)] for g in range(1, 9)])
    write_star(spa_copy / "Class2D/job006/run_it200_model.star", model.values(), atomic=False)
    inp = read_class_input(spa_copy, OPT)
    res = write_class_selection(inp, "Select/job095/", list(range(1, 51)), regroup=4)
    out = read_star(spa_copy / "Select/job095/particles.star")["particles"]
    assert "rlnGroupNumber" not in out.columns and "rlnGroupName" in out.columns
    names = sorted(set(out.column("rlnGroupName")))
    # RELION counts the group that overflows into the next one: 8 equal groups "into 4" make 3
    assert names == ["group_1", "group_2", "group_3"] and res.messages[0] == "Regrouped particles into 3 groups"
    with pytest.raises(SelectionError, match="at least 10"):
        write_class_selection(inp, "Select/job094/", [1], regroup=40)


def test_regroup_refuses_groups_the_model_does_not_have():
    parts = StarBlock.from_records("particles", ["rlnImageName", "rlnGroupNumber", "rlnOpticsGroup"],
                                   [[f"{i + 1:06d}@x.mrcs", 1 + i % 2, 1] for i in range(30)])
    groups = StarBlock.from_records("model_groups", ["rlnGroupNumber", "rlnGroupScaleCorrection"], [[1, 1.0]])
    with pytest.raises(SelectionError, match="group.s. 2 of the particles"):
        regroup_particles(parts, groups, 1)
    assert parts.columns == ["rlnImageName", "rlnGroupNumber", "rlnOpticsGroup"]  # left as it was


# ---------------------------------------------------------------------------- tilt images

TILTS_IN = "CtfFind/job003/tilt_series_ctf.star"


def test_tilt_exclusion_files(tomo_project, tmp_path):
    proj = tmp_path / "tomo"
    shutil.copytree(tomo_project, proj, symlinks=True)
    inp = read_tilt_input(proj, TILTS_IN)
    assert inp.names[0] == "TS_01" and len(inp.tilts["TS_01"]) == 41
    kept = {n: [True] * 41 for n in inp.names}
    kept["TS_01"][0] = kept["TS_01"][40] = False  # the ±60° images
    kept["TS_43"] = [False] * 41  # the whole series
    counts = write_tilt_exclusion(inp, "ExcludeTiltImages/job099/", kept)
    assert counts["TS_01"] == 39 and counts["TS_43"] == 0
    g = read_star(proj / "ExcludeTiltImages/job099/selected_tilt_series.star")["global"]
    assert g.column("rlnTomoName") == ["TS_01", "TS_03", "TS_45", "TS_54"]  # TS_43 dropped
    assert g.columns[-1] == "rlnTomoTiltSeriesStarFile"
    assert g.column("rlnTomoTiltSeriesStarFile")[0] == "ExcludeTiltImages/job099/tilt_series/TS_01.star"
    ts = read_star(proj / "ExcludeTiltImages/job099/tilt_series/TS_01.star")
    assert list(ts) == ["TS_01"] and len(ts["TS_01"]) == 39
    angles = [float(a) for a in ts["TS_01"].column("rlnTomoNominalStageTiltAngle")]
    assert angles == sorted(angles) and min(angles) == -57 and max(angles) == 57
    assert not (proj / "ExcludeTiltImages/job099/tilt_series/TS_43.star").exists()
    # read back, matched by image name
    back = read_tilt_exclusion(inp, "ExcludeTiltImages/job099/")
    assert back == kept
    # the tables still read as tilt series in the viewers
    from cryo_savanna.results.tomo import read_tilt_series_set
    s = read_tilt_series_set(proj, "ExcludeTiltImages/job099/selected_tilt_series.star")
    assert len(s.tilts(0)) == 39
    with pytest.raises(SelectionError, match="every tilt"):
        write_tilt_exclusion(inp, "ExcludeTiltImages/job098/", {n: [False] * 41 for n in inp.names})
    assert read_tilt_exclusion(inp, "ExcludeTiltImages/job097/") is None


# ---------------------------------------------------------------------------- with relion_pipeliner

from cryo_savanna.tools.validate import pipeliner_version  # noqa: E402

_version = pipeliner_version()
needs_relion = pytest.mark.skipif(not (_version or "").startswith("5.1"),
                                  reason=f"needs relion_pipeliner 5.1 on PATH (found: {_version})")


@needs_relion
def test_select_job_done_here_is_a_finished_relion_job(spa_copy):
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.pipeliner import Pipeliner
    js = JobDefs.load().variant("relion.select").new_jobstar(RelionEnv({}))
    js.options["fn_model"] = OPT
    inp = read_class_input(spa_copy, OPT)
    pl = Pipeliner(spa_copy)
    res = pl.add_done_here(js, lambda job: write_class_selection(inp, job, [3, 7]),
                           f"relion_display --gui --i {OPT} --allow_save ...", alias="good")
    p = Pipeline.read(spa_copy)
    pr = p.processes[res.job]
    assert pr.status == "Succeeded" and pr.alias.endswith("good/")
    assert set(pr.outputs) == {res.job + "particles.star", res.job + "class_averages.star"}
    assert OPT in pr.inputs
    assert (spa_copy / res.job / "RELION_JOB_EXIT_SUCCESS").exists()
    assert "cryo-savanna" in (spa_copy / res.job / "note.txt").read_text()
    # changing the selection later rewrites the files in place
    pl.finish_done_here(res.job, lambda job: write_class_selection(inp, job, [3]), "relion_display ...")
    assert read_backup_selection(spa_copy, res.job, 50).count(True) == 1
    assert Pipeline.read(spa_copy).processes[res.job].status == "Succeeded"


@needs_relion
def test_failed_write_leaves_no_job(spa_copy):
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.pipeliner import Pipeliner
    js = JobDefs.load().variant("relion.select").new_jobstar(RelionEnv({}))
    js.options["fn_model"] = OPT
    before = set(Pipeline.read(spa_copy).processes)

    def fail(job):
        raise SelectionError("nope")

    with pytest.raises(SelectionError):
        Pipeliner(spa_copy).add_done_here(js, fail, "x")
    assert set(Pipeline.read(spa_copy).processes) == before


@needs_relion
def test_add_job_finds_jobs_relion_labels_by_mode(spa_copy):
    """RELION labels some jobs by what they do (relion.autopick.log, relion.select.interactive):
    add_job used to compare the label exactly and report "did not add the job"."""
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.pipeliner import Pipeliner
    js = JobDefs.load().variant("relion.autopick").new_jobstar(RelionEnv({}))
    js.options.update({"fn_input_autopick": "CtfFind/job003/micrographs_ctf.star", "do_log": "Yes"})
    res = Pipeliner(spa_copy).add_job(js)
    assert Pipeline.read(spa_copy).processes[res.job].type_label == "relion.autopick.log"


def test_classes_numbered_by_rlnClassNumber_are_saved_as_shown(tmp_path):
    """A model_classes table with its own class numbers: the gallery shows them, and a selection of what
    is shown writes those classes' particles (and reads back as the same classes)."""
    from cryo_savanna.project.selection import selected_numbers
    from cryo_savanna.results.classes import read_model_classes
    job = tmp_path / "Class2D/job004"
    job.mkdir(parents=True)
    write_star(job / "run_it025_model.star", [
        StarBlock.from_dict("model_general", {"rlnReferenceDimensionality": 2}),
        StarBlock.from_records("model_classes", ["rlnReferenceImage", "rlnClassDistribution", "rlnClassNumber"],
                               [["000001@Class2D/job004/run_it025_classes.mrcs", 0.5, 3],
                                ["000002@Class2D/job004/run_it025_classes.mrcs", 0.5, 7]])])
    write_star(job / "run_it025_data.star", [StarBlock.from_records(
        "particles", ["rlnImageName", "rlnClassNumber"], [["1@a.mrcs", 3], ["2@a.mrcs", 7], ["3@a.mrcs", 7]])])
    shown = read_model_classes(tmp_path, "Class2D/job004/run_it025_model.star", job / "run_it025_data.star")
    assert [(c.number, c.particles) for c in shown.classes] == [(3, 1), (7, 2)]
    inp = read_class_input(tmp_path, "Class2D/job004/run_it025_model.star")
    res = write_class_selection(inp, "Select/job005/", [shown.classes[1].number])
    assert res.particles == 2
    flags = read_backup_selection(tmp_path, "Select/job005/", 2)
    assert selected_numbers(inp, flags) == [7]


def test_saving_a_selection_as_a_job_and_a_failed_overwrite(spa_copy, tmp_path, monkeypatch):
    """With a stand-in relion_pipeliner (tests/fixtures/fake_pipeliner.py): a selection saved as a new job is
    a finished Select job with its outputs, linked to its classes; an overwrite whose files cannot be written
    puts the job back as it was (pipeline, alias, files), and a failed new job leaves nothing behind."""
    import os

    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.model import JobDefs
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.pipeliner import Pipeliner, PipelinerError
    from cryo_savanna.project.selection import relion_command
    from tests.fixtures.fake_pipeliner import install
    install(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    src = str(sorted((spa_copy / "Class2D/job006").glob("run_it*_optimiser.star"))[-1].relative_to(spa_copy))
    js = JobDefs.load().variant("relion.select").new_jobstar(RelionEnv({}))
    js.options["fn_model"] = src
    picked = read_class_input(spa_copy, src).numbers[:3]

    def write(job):
        return write_class_selection(read_class_input(spa_copy, src), job, picked)

    def command(job):
        return relion_command("classes", js.options, job)

    pl = Pipeliner(spa_copy)
    job = pl.add_done_here(js, write, command, alias="picked").job
    p = Pipeline.read(spa_copy)
    pr = p.processes[job]
    assert (pr.status, pr.alias) == ("Succeeded", "Select/picked/") and p.parents(job) == ["Class2D/job006/"]
    assert {f"{job}particles.star", f"{job}class_averages.star"} <= set(pr.outputs)
    files = {f: (spa_copy / job / f).read_bytes() for f in os.listdir(spa_copy / job)}

    def fail(job):
        raise OSError("No space left on device")
    with pytest.raises(PipelinerError, match="back as it was"):
        pl.overwrite_done_here(job, js, fail, command, alias="picked")
    back = Pipeline.read(spa_copy).processes[job]
    assert (back.status, back.alias, sorted(back.outputs)) == (pr.status, pr.alias, sorted(pr.outputs))
    assert {f: (spa_copy / job / f).read_bytes() for f in os.listdir(spa_copy / job)} == files
    assert os.readlink(spa_copy / "Select/picked") == f"../{job}"
    before = set(Pipeline.read(spa_copy).processes)
    with pytest.raises(OSError):
        pl.add_done_here(js, fail, command)
    assert set(Pipeline.read(spa_copy).processes) == before
