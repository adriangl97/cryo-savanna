# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Several reference maps of a 3D classification: their list, written into the job's directory when the job is
added, and the jobs of the maps linked in the job tree (project/references.py)."""

import os

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project import references
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Node, Pipeline
from cryo_savanna.project.pipeliner import Pipeliner
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.star import StarBlock, read_star, write_star
from tests.fixtures.fake_pipeliner import install as install_fake_pipeliner
from tests.fixtures.make_fixtures import JobSpec, add_job

MAPS = ["Refine3D/job011/run_class001.mrc", "InitialModel/job008/initial_model.mrc"]


def test_list_is_written_as_relion_refine_reads_it(tmp_path):
    path = references.write(tmp_path, "Class3D/job020/", MAPS)
    assert path == tmp_path / "Class3D/job020/references.star"
    block = read_star(path)["model_classes"]  # the block relion_refine looks in first
    assert block.column("rlnReferenceImage") == MAPS
    assert references.read(tmp_path, "Class3D/job020/references.star") == MAPS
    assert references.read(tmp_path, "Refine3D/job011/run_class001.mrc") is None  # a map
    assert references.read(tmp_path, "Class3D/job021/references.star") is None  # not there
    write_star(tmp_path / "other.star", [StarBlock.from_records("", ["rlnReferenceImage"], [["a.mrc"], ["b.mrc"]])])
    assert references.read(tmp_path, str(tmp_path / "other.star")) == ["a.mrc", "b.mrc"]  # any block, as RELION
    write_star(tmp_path / "parts.star", [StarBlock.from_records("particles", ["rlnImageName"], [["1@a.mrcs"]])])
    assert references.read(tmp_path, "parts.star") is None
    assert references.is_list_path("Class3D/job020/references.star") and not references.is_list_path("refs.star")


def _class3d(project) -> JobStar:
    js = JobDefs.load().variant("relion.class3d").new_jobstar(RelionEnv({}))
    js.options.update(fn_img="Select/job010/particles.star", nr_classes="2",
                      fn_ref="Class3D/job001/references.star")  # the editor's guess, made a while ago
    return js


def _pipeliner(project, tmp_path, monkeypatch) -> Pipeliner:
    install_fake_pipeliner(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return Pipeliner(project)


def test_the_list_goes_in_the_directory_the_job_gets(spa_copy, tmp_path, monkeypatch):
    pl = _pipeliner(spa_copy, tmp_path, monkeypatch)
    counter = Pipeline.read(spa_copy).job_counter
    res = pl.add_job(_class3d(spa_copy), None, MAPS)
    assert res.job == f"Class3D/job{counter:03d}/" and not res.warnings
    assert JobStar.read(spa_copy / res.job / "job.star").options["fn_ref"] == f"{res.job}references.star"
    assert references.read(spa_copy, f"{res.job}references.star") == MAPS
    # as RELION records it (the stand-in pipeliner only links nodes it knows): the list is an input
    with PipelineLock(spa_copy) as lock:
        p = Pipeline.read(spa_copy)
        node = f"{res.job}references.star"
        p.nodes[node] = Node(node, "DensityMap.star", consumers=[res.job])
        p.processes[res.job].inputs.append(node)
        p.write(spa_copy, lock=lock)
    info = ProjectLoader(spa_copy).refresh().jobs[res.job]
    assert info.primary_parent == "Select/job010/"
    assert info.other_parents == ["Refine3D/job011/", "InitialModel/job008/"]
    assert info.reference_maps == dict(zip(MAPS, info.other_parents))


def test_another_job_taking_the_directory_meanwhile(spa_copy, tmp_path, monkeypatch):
    pl = _pipeliner(spa_copy, tmp_path, monkeypatch)
    counter = Pipeline.read(spa_copy).job_counter
    call = pl._call

    def racing(*args, **kwargs):  # a job added elsewhere just before relion_pipeliner adds ours
        if "--addJobFromStar" in args:
            add_job(spa_copy, JobSpec(counter, "relion.maskcreate", []))
        return call(*args, **kwargs)
    monkeypatch.setattr(pl, "_call", racing)
    res = pl.add_job(_class3d(spa_copy), None, MAPS)
    assert res.job == f"Class3D/job{counter + 1:03d}/"
    assert JobStar.read(spa_copy / res.job / "job.star").options["fn_ref"] == f"{res.job}references.star"
    assert references.read(spa_copy, f"{res.job}references.star") == MAPS
    assert res.warnings and "at the same moment" in res.warnings[0]


def test_a_job_with_one_map_is_added_as_it_is(spa_copy, tmp_path, monkeypatch):
    pl = _pipeliner(spa_copy, tmp_path, monkeypatch)
    js = _class3d(spa_copy)
    js.options["fn_ref"] = MAPS[0]
    res = pl.add_job(js)
    assert JobStar.read(spa_copy / res.job / "job.star").options["fn_ref"] == MAPS[0]
    assert not (spa_copy / res.job / references.FILE).exists()


def test_the_job_overview_names_the_maps(qtbot, spa_copy, tmp_path, monkeypatch):
    from cryo_savanna.ui.job_info import JobInfoPanel
    pl = _pipeliner(spa_copy, tmp_path, monkeypatch)
    res = pl.add_job(_class3d(spa_copy), None, MAPS)
    with PipelineLock(spa_copy) as lock:
        p = Pipeline.read(spa_copy)
        node = f"{res.job}references.star"
        p.nodes[node] = Node(node, "DensityMap.star", consumers=[res.job])
        p.processes[res.job].inputs.append(node)
        p.write(spa_copy, lock=lock)
    snap = ProjectLoader(spa_copy).refresh()
    panel = JobInfoPanel(JobDefs.load(), RelionEnv({}), {})
    qtbot.addWidget(panel)
    panel.show_job(snap, res.job)
    text = panel.overview.toPlainText()
    assert "the list of the reference maps, one class each" in text
    assert "run_class001.mrc from Refine3D/first_refine (job011)" in text  # (by its alias)
    assert "initial_model.mrc from InitialModel/job008" in text
