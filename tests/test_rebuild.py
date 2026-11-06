# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Rebuilding a pipeline from job directories (project/rebuild.py) and its review dialog."""
import os
import time
from pathlib import Path

import pytest

from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.rebuild import (
    MINI_PIPELINE,
    RebuildError,
    adopt_folder,
    find_job_dirs,
    needs_pipeline,
    scan,
    unregistered_job_dirs,
    write_pipeline,
)
from cryo_savanna.project.star import write_star

DEFS = JobDefs.load()
OLD = time.time() - 3600


def _lose_pipeline(project: Path) -> Pipeline:
    orig = Pipeline.read(project)
    (project / "default_pipeline.star").unlink()
    return orig


def _write_mini_pipelines(project: Path, orig: Pipeline) -> None:
    """job_pipeline.star in every job directory, as RELION writes it when it adds a job."""
    for name, pr in orig.processes.items():
        mini = Pipeline(job_counter=orig.job_counter)
        mini.processes[name] = Process(name, pr.alias, pr.type_label, "Scheduled", list(pr.inputs), list(pr.outputs))
        for n in pr.inputs + pr.outputs:
            mini.nodes[n] = Node(n, orig.nodes[n].type_label, orig.nodes[n].type_depth)
        write_star(project / name / MINI_PIPELINE, mini.to_blocks())


def _age(d: Path, t: float = OLD) -> None:
    for f in d.iterdir():
        os.utime(f, (t, t), follow_symlinks=False)


def _cli_refine(project: Path, name: str = "Refine3D/job019") -> Path:
    """A relion_refine run by hand: no job.star, no note.txt, no exit file."""
    d = project / name
    d.mkdir(parents=True)
    (d / "run_it005_data.star").write_text("\ndata_particles\nloop_\n_rlnImageName\n1@x.mrcs\n")
    (d / "run_it010_data.star").write_text("\ndata_particles\nloop_\n_rlnImageName\n1@x.mrcs\n")
    (d / "run_it010_optimiser.star").write_text(
        "\ndata_optimiser_general\n_rlnOutputRootName Refine3D/job019/run\n"
        "_rlnExperimentalDataStarFile Select/job010/particles.star\n_rlnSolventMaskName MaskCreate/job012/mask.mrc\n")
    (d / "run_it010_half1_class001.mrc").write_bytes(b"\0" * 1024)
    # a log naming files in another job's subfolder: not inputs
    (d / "run.out").write_text("Reading MotionCorr/job002/Movies/mic001.mrc\n")
    _age(d)
    return d


# -- finding job directories ---------------------------------------------------------------

def test_job_dirs_and_aliases_are_found(spa_copy):
    dirs = find_job_dirs(spa_copy)
    assert len(dirs) == 18 and dirs[0] == "Import/job001/" and dirs[-1] == "Class2D/job018/"
    assert "AutoPick/LoG_based/" not in dirs  # an alias symlink
    assert not needs_pipeline(spa_copy) and unregistered_job_dirs(spa_copy) == []
    _lose_pipeline(spa_copy)
    assert needs_pipeline(spa_copy)


def test_an_empty_folder_needs_no_pipeline(tmp_path):
    (tmp_path / "Movies").mkdir()
    (tmp_path / "Class2D" / "job001").mkdir(parents=True)  # empty: nothing ran there
    assert find_job_dirs(tmp_path) == [] and not needs_pipeline(tmp_path)


# -- a lost pipeline -----------------------------------------------------------------------

def test_lost_pipeline_is_rebuilt_from_job_star(spa_copy):
    orig = _lose_pipeline(spa_copy)
    res = scan(spa_copy, defs=DEFS)
    assert [j.name for j in res.jobs] == list(orig.processes)
    for j in res.jobs:
        o = orig.processes[j.name]
        assert (j.type_label, j.status, j.alias) == (o.type_label, o.status, o.alias), j.name
        assert sorted(j.inputs) == sorted(o.inputs), j.name
        assert "job.star" in j.sources
        if o.status == "Succeeded":
            assert set(j.output_names()) == set(o.outputs), j.name
    # a scheduled job declares the files it will write, so jobs after it can be linked to it
    assert "Refine3D/job016/run_data.star" in res.job("Refine3D/job016/").output_names()

    added = write_pipeline(spa_copy, res, new=True)
    p = Pipeline.read(spa_copy)
    assert added == list(orig.processes) and list(p.processes) == list(orig.processes)
    assert p.job_counter == 19
    assert p.parents("PostProcess/job013/") == ["Refine3D/job011/", "MaskCreate/job012/"]
    assert p.processes["Refine3D/job011/"].alias == "Refine3D/first_refine/"
    assert (spa_copy / ".Nodes" / "DensityMap" / "Refine3D/first_refine/run_class001.mrc").exists()
    with pytest.raises(RebuildError):
        write_pipeline(spa_copy, res, new=True)  # never over an existing pipeline


def test_lost_pipeline_is_rebuilt_exactly_from_job_pipeline_star(spa_copy):
    orig = _lose_pipeline(spa_copy)
    _write_mini_pipelines(spa_copy, orig)
    res = scan(spa_copy, defs=DEFS)
    for j in res.jobs:
        o = orig.processes[j.name]
        assert j.sources[0] == MINI_PIPELINE
        assert (j.type_label, j.status, j.alias, j.inputs) == (o.type_label, o.status, o.alias, o.inputs), j.name
        assert j.output_names() == o.outputs, j.name


def test_tomo_pipeline_is_rebuilt(tomo_project, tmp_path):
    import shutil
    project = tmp_path / "tomo"
    shutil.copytree(tomo_project, project, symlinks=True)
    orig = _lose_pipeline(project)
    res = scan(project, defs=DEFS)
    for j in res.jobs:
        o = orig.processes[j.name]
        assert (j.type_label, j.status, j.alias) == (o.type_label, o.status, o.alias), j.name
        assert set(j.inputs) <= set(o.inputs) and bool(j.inputs) == bool(o.inputs), j.name
    assert res.job("Import/job001/").type_label == "relion.importtomo"


def test_only_the_chosen_jobs_are_written(spa_copy):
    _lose_pipeline(spa_copy)
    res = scan(spa_copy, defs=DEFS)
    keep = [j.name for j in res.jobs if j.name != "MaskCreate/job012/"]
    write_pipeline(spa_copy, res, keep, statuses={"Polish/job015/": "Failed"}, new=True)
    p = Pipeline.read(spa_copy)
    assert "MaskCreate/job012/" not in p.processes
    assert p.parents("PostProcess/job013/") == ["Refine3D/job011/"]  # the link to the left-out job is dropped
    assert p.processes["Polish/job015/"].status == "Failed"
    assert p.job_counter == 19  # job012 still exists on disk: its number is not given out again


# -- jobs run from the command line --------------------------------------------------------

def test_cli_job_is_added_to_the_pipeline(spa_copy):
    _cli_refine(spa_copy)
    assert unregistered_job_dirs(spa_copy) == ["Refine3D/job019/"]
    res = scan(spa_copy, Pipeline.read(spa_copy), defs=DEFS)
    (j,) = res.jobs
    assert j.type_label == "relion.refine3d" and j.status == "Succeeded"
    assert j.inputs == ["Select/job010/particles.star", "MaskCreate/job012/mask.mrc"]
    assert j.output_names() == ["Refine3D/job019/run_it010_data.star", "Refine3D/job019/run_it010_optimiser.star",
                                "Refine3D/job019/run_it010_half1_class001.mrc"]
    assert any("folder name" in n for n in j.notes)
    assert write_pipeline(spa_copy, res) == ["Refine3D/job019/"]
    p = Pipeline.read(spa_copy)
    assert p.parents("Refine3D/job019/") == ["Select/job010/", "MaskCreate/job012/"]
    assert p.job_counter == 20 and len(p.processes) == 19
    assert unregistered_job_dirs(spa_copy) == []


def test_cli_job_still_writing_is_running(spa_copy):
    d = _cli_refine(spa_copy)
    _age(d, time.time())
    (j,) = scan(spa_copy, Pipeline.read(spa_copy), defs=DEFS).jobs
    assert j.status == "Running" and any("running" in n for n in j.notes)


def test_a_command_in_note_txt_gives_the_inputs(spa_copy):
    d = spa_copy / "PostProcess/job019"
    d.mkdir()
    (d / "note.txt").write_text(
        " ++++ Executing new job on Tue Oct 28 10:00:00 2026\n ++++ with the following command(s):\n"
        "relion_postprocess --mask MaskCreate/job012/mask.mrc --i Refine3D/first_refine/run_half1_class001_unfil.mrc "
        "--o PostProcess/job019/postprocess --angpix -1\n ++++ \n")
    (d / "postprocess.mrc").write_bytes(b"\0" * 1024)
    (d / "postprocess.star").write_text("\ndata_general\n_rlnFinalResolution 3.1\n")
    _age(d)
    (j,) = scan(spa_copy, Pipeline.read(spa_copy), defs=DEFS).jobs
    assert j.inputs == ["MaskCreate/job012/mask.mrc", "Refine3D/job011/run_half1_class001_unfil.mrc"]  # alias resolved
    assert {"PostProcess/job019/postprocess.mrc", "PostProcess/job019/postprocess.star"} <= set(j.output_names())


def test_a_file_another_job_uses_becomes_an_output(spa_copy):
    """A CLI job that takes a file of a pipeline job that the pipeline does not declare."""
    (spa_copy / "Class2D/job006/run_it150_data.star").write_text("\ndata_particles\nloop_\n_rlnImageName\n1@x\n")
    d = spa_copy / "Select/job019"
    d.mkdir()
    (d / "run_submit.script").write_text("#!/bin/bash\n#SBATCH -n 1\nrelion_star_handler --i "
                                         "Class2D/job006/run_it150_data.star --o Select/job019/particles.star\n")
    (d / "particles.star").write_text("\ndata_particles\nloop_\n_rlnImageName\n1@x\n")
    _age(d)
    res = scan(spa_copy, Pipeline.read(spa_copy), defs=DEFS)
    assert res.jobs[0].inputs == ["Class2D/job006/run_it150_data.star"]
    assert res.extra_outputs == {"Class2D/job006/": [("Class2D/job006/run_it150_data.star",
                                                      "ParticleGroupMetadata.star.relion", 1)]}
    write_pipeline(spa_copy, res)
    p = Pipeline.read(spa_copy)
    assert p.nodes["Class2D/job006/run_it150_data.star"].producer == "Class2D/job006/"
    assert p.parents("Select/job019/") == ["Class2D/job006/"]


def test_a_folder_is_adopted_as_a_job(spa_copy):
    free = spa_copy / "Class3D/bigbox"
    free.mkdir()
    (free / "run_it025_optimiser.star").write_text(
        "\ndata_optimiser_general\n_rlnExperimentalDataStarFile Refine3D/first_refine/run_data.star\n")
    (free / "run_it025_data.star").write_text("\ndata_particles\nloop_\n_rlnImageName\n1@x.mrcs\n")
    _age(free)
    name = adopt_folder(spa_copy, "Class3D/bigbox", "relion.class3d")
    assert name == "Class3D/job019/"
    assert os.readlink(spa_copy / "Class3D/bigbox") == "../Class3D/job019"  # RELION's alias link
    assert (spa_copy / "Class3D/bigbox/run_it025_data.star").exists()
    assert Pipeline.read(spa_copy).job_counter == 20  # the number is taken
    res = scan(spa_copy, Pipeline.read(spa_copy), defs=DEFS)
    (j,) = res.jobs
    assert (j.name, j.alias, j.inputs) == (name, "Class3D/bigbox/", ["Refine3D/job011/run_data.star"])
    write_pipeline(spa_copy, res)
    assert Pipeline.read(spa_copy).processes[name].alias == "Class3D/bigbox/"


def test_adopting_refuses_job_dirs_and_outside_folders(spa_copy, tmp_path):
    with pytest.raises(RebuildError):
        adopt_folder(spa_copy, "Class2D/job006", "relion.class2d")
    with pytest.raises(RebuildError):
        adopt_folder(spa_copy, tmp_path, "relion.class2d")


# -- the startup question and the review dialog --------------------------------------------

qt = pytest.importorskip("qtpy.QtWidgets")


def test_startup_offers_to_generate_the_pipeline(qapp, spa_copy):
    from cryo_savanna.ui import start_project as sp
    _lose_pipeline(spa_copy)
    asked = []

    def generate(folder):
        write_pipeline(folder, scan(folder, defs=DEFS), new=True)
        return True

    got = sp.choose_project(spa_copy, False, ask=lambda f, ro: pytest.fail("new project offered"),
                            ask_generate=lambda f, n, ro: asked.append(n) or sp.GENERATE, generate=generate)
    assert got == spa_copy and asked == [18] and len(Pipeline.read(spa_copy).processes) == 18


def test_startup_readonly_never_generates(qapp, spa_copy):
    from cryo_savanna.ui import start_project as sp
    _lose_pipeline(spa_copy)
    got = sp.choose_project(spa_copy, True, ask_generate=lambda f, n, ro: sp.GENERATE if not ro else sp.QUIT,
                            generate=lambda f: pytest.fail("generated read-only"))
    assert got is None and not (spa_copy / "default_pipeline.star").exists()


def test_review_dialog_leaves_out_and_changes_status(qtbot, spa_copy):
    from qtpy.QtCore import Qt

    from cryo_savanna.ui.rebuild_dialog import RebuildDialog
    _lose_pipeline(spa_copy)
    res = scan(spa_copy, defs=DEFS)
    dlg = RebuildDialog(spa_copy, res, lambda: res, new=True)
    qtbot.addWidget(dlg)
    assert dlg.table.rowCount() == 18 and len(dlg.chosen()) == 18
    dlg.table.item(0, 0).setCheckState(Qt.Unchecked)
    assert "Import/job001/" not in dlg.chosen()
    dlg.table.cellWidget(14, 2).setCurrentText("Succeeded")
    assert dlg.statuses()["Polish/job015/"] == "Succeeded"
    assert "first_refine" in dlg.table.item(10, 0).text()


def test_window_adds_jobs_run_outside_the_pipeline(qtbot, spa_copy, tmp_path, monkeypatch):
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QDialog

    from cryo_savanna.ui.main_window import MainWindow
    from cryo_savanna.ui.rebuild_dialog import RebuildDialog
    _cli_refine(spa_copy)
    w = MainWindow(spa_copy, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    assert w.actions.act_import in w.menuBar().actions()[0].menu().actions()
    assert w.actions.act_import.isEnabled()
    shown = []
    monkeypatch.setattr(RebuildDialog, "exec", lambda dlg: shown.append(dlg.chosen()) or QDialog.Accepted)
    w.actions.act_import.trigger()
    assert shown == [["Refine3D/job019/"]]
    assert w.actions.wait(10000)
    qtbot.waitUntil(lambda: "Refine3D/job019/" in w.graph.nodes, timeout=5000)
    assert Pipeline.read(spa_copy).parents("Refine3D/job019/") == ["Select/job010/", "MaskCreate/job012/"]


def test_window_offers_to_generate_a_lost_pipeline(qtbot, spa_copy, spa_project, tmp_path, monkeypatch):
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QDialog

    from cryo_savanna.ui import main_window
    from cryo_savanna.ui.rebuild_dialog import RebuildDialog
    _lose_pipeline(spa_copy)
    w = main_window.MainWindow(spa_project, poll_s=1000.0,
                               settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    monkeypatch.setattr(main_window, "ask_generate", lambda *a, **k: "generate")
    monkeypatch.setattr(RebuildDialog, "exec", lambda dlg: QDialog.Accepted)
    w.open_project(spa_copy)
    assert len(Pipeline.read(spa_copy).processes) == 18
    qtbot.waitUntil(lambda: w.snapshot is not None and w.snapshot.project_dir == spa_copy.resolve(), timeout=5000)
