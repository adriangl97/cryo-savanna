# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Select jobs on tomography particles get an optimisation set (project/tomo_select.py)."""

import shutil

import pytest

from cryo_savanna.project import tomo_select
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.pipeliner import Pipeliner
from cryo_savanna.project.star import StarBlock, read_star, write_star

PARTS = "ParticleGroupMetadata.star.relion.select"


@pytest.fixture
def tomo_copy(tomo_project, tmp_path):
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    return dst


def add_select(project, number, options, outputs=("particles.star",), status="Succeeded", tomo=True,
               source="Extract/job009/particles.star"):
    """A Select job as relion_pipeliner leaves it: job.star, the particle files, and the pipeline."""
    job = f"Select/job{number:03d}/"
    d = project / job
    d.mkdir(parents=True)
    JobStar(type_label="relion.select", is_continue=False, is_tomo=tomo, options=dict(options)).write(d / "job.star")
    src = read_star(project / source)
    for out in outputs:
        write_star(d / out, list(src.values()))
    with PipelineLock(project, "default", message="test") as lock:
        p = Pipeline.read(project)
        src_in = options.get("fn_data") or options.get("fn_model")
        p.processes[job] = Process(job, "None", "relion.select", status, inputs=[src_in],
                                   outputs=[job + o for o in outputs])
        for o in outputs:
            p.nodes[job + o] = Node(job + o, PARTS, producer=job)
        if src_in in p.nodes:
            p.nodes[src_in].consumers.append(job)
        p.write(project, "default", lock)
    return job


def values(path):
    return next(iter(read_star(path).values())).as_dict()


def test_names_of_the_sets():
    assert tomo_select.set_for("Select/job020/particles.star") == "Select/job020/optimisation_set.star"
    assert tomo_select.set_for("Select/job020/particles_split12.star") == "Select/job020/optimisation_set_split12.star"


def test_the_input_is_the_one_relion_selects_from():
    assert tomo_select.select_input({"fn_model": "Class3D/job010/run_it025_optimiser.star", "fn_data": ""}) == \
        "Class3D/job010/run_it025_optimiser.star"
    assert tomo_select.select_input({"fn_data": "Extract/job009/particles.star", "do_split": "Yes"}) == \
        "Extract/job009/particles.star"
    assert tomo_select.select_input({"fn_mic": "x.star", "fn_data": "p.star", "do_select_values": "Yes"}) == ""


def test_select_on_extracted_particles_gets_their_tomograms(tomo_copy):
    job = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_remove_duplicates": "Yes"})
    assert job in tomo_select.candidates(Pipeline.read(tomo_copy))
    assert tomo_select.complete(tomo_copy, job) == [job + "optimisation_set.star"]
    assert values(tomo_copy / job / "optimisation_set.star") == {
        "rlnTomoParticlesFile": job + "particles.star", "rlnTomoTomogramsFile": "Tomograms/job006/tomograms.star"}
    p = Pipeline.read(tomo_copy)
    node = p.nodes[job + "optimisation_set.star"]
    assert node.kind == "TomoOptimisationSet" and node.producer == job
    assert job + "optimisation_set.star" in p.processes[job].outputs
    assert (tomo_copy / ".Nodes" / "TomoOptimisationSet" / job / "optimisation_set.star").exists()
    assert tomo_select.complete(tomo_copy, job) == []  # once


def test_select_on_a_refinement_takes_the_tomograms_its_job_took(tomo_copy):
    """Refine3D's run_optimisation_set.star of the test project names no tomograms: those of its own input."""
    job = add_select(tomo_copy, 20, {"fn_data": "Refine3D/job011/run_data.star", "do_select_values": "Yes"},
                     source="Refine3D/job011/run_data.star")
    assert tomo_select.complete(tomo_copy, job) == [job + "optimisation_set.star"]
    assert values(tomo_copy / job / "optimisation_set.star")["rlnTomoTomogramsFile"] == \
        "Tomograms/job006/tomograms.star"


def test_subsets_each_get_their_own_set_with_the_trajectories(tomo_copy):
    write_star(tomo_copy / "Extract/job009/optimisation_set.star", [StarBlock.from_dict("", {
        "rlnTomoParticlesFile": "Extract/job009/particles.star", "rlnTomoTomogramsFile": "Tomograms/job006/tomograms.star",
        "rlnTomoTrajectoriesFile": "FrameAlign/job030/motion.star",
        "rlnTomoReferenceMap1File": "Refine3D/job011/run_half1_class001_unfil.mrc"})])
    job = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_split": "Yes"},
                     outputs=("particles_split1.star", "particles_split2.star"))
    assert tomo_select.complete(tomo_copy, job) == [job + "optimisation_set_split1.star",
                                                    job + "optimisation_set_split2.star"]
    v = values(tomo_copy / job / "optimisation_set_split2.star")
    assert v == {"rlnTomoParticlesFile": job + "particles_split2.star",
                 "rlnTomoTomogramsFile": "Tomograms/job006/tomograms.star",
                 "rlnTomoTrajectoriesFile": "FrameAlign/job030/motion.star"}  # not the refinement's maps


def test_a_selection_of_a_selection_takes_the_first_ones_set(tomo_copy):
    first = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_remove_duplicates": "Yes"})
    tomo_select.complete(tomo_copy, first)
    second = add_select(tomo_copy, 21, {"fn_data": first + "particles.star", "do_select_values": "Yes"},
                        source=first + "particles.star")
    assert tomo_select.complete(tomo_copy, second) == [second + "optimisation_set.star"]
    assert values(tomo_copy / second / "optimisation_set.star")["rlnTomoTomogramsFile"] == \
        "Tomograms/job006/tomograms.star"


def test_nothing_for_spa_particles_or_unfinished_jobs(tomo_copy, spa_copy):
    spa = next(n for n in tomo_select.candidates(Pipeline.read(spa_copy)))
    before = Pipeline.read(spa_copy)
    assert tomo_select.complete(spa_copy, spa) == []
    assert Pipeline.read(spa_copy).nodes.keys() == before.nodes.keys()
    assert not list((spa_copy / spa).glob("optimisation_set*.star"))
    running = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_discard": "Yes"},
                         status="Running")
    assert running not in tomo_select.candidates(Pipeline.read(tomo_copy))
    assert tomo_select.complete(tomo_copy, running) == []


def test_a_set_gone_from_disk_is_written_again(tomo_copy):
    job = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_remove_duplicates": "Yes"})
    tomo_select.complete(tomo_copy, job)
    (tomo_copy / job / "optimisation_set.star").unlink()
    assert tomo_select.missing(tomo_copy, Pipeline.read(tomo_copy), job) == [job + "particles.star"]
    tomo_select.complete(tomo_copy, job)
    assert (tomo_copy / job / "optimisation_set.star").is_file()


def test_a_selection_saved_in_cryo_savanna_gets_its_set(tomo_copy):
    """As the class selection window saves it (Pipeliner.finish_done_here)."""
    job = add_select(tomo_copy, 20, {"fn_model": "Refine3D/job011/run_optimiser.star"}, status="Running",
                     source="Refine3D/job011/run_data.star")

    def write(j):  # what write_class_selection writes
        shutil.copy(tomo_copy / "Refine3D/job011/run_data.star", tomo_copy / j / "particles.star")
    Pipeliner(tomo_copy, exe="/nonexistent").finish_done_here(job, write, "relion_display ...")
    p = Pipeline.read(tomo_copy)
    assert p.processes[job].status == "Succeeded"
    assert p.nodes[job + "optimisation_set.star"].kind == "TomoOptimisationSet"
    assert values(tomo_copy / job / "optimisation_set.star") == {
        "rlnTomoParticlesFile": job + "particles.star", "rlnTomoTomogramsFile": "Tomograms/job006/tomograms.star"}


def test_a_rebuilt_pipeline_keeps_the_sets(tomo_copy):
    from cryo_savanna.project.rebuild import OUTPUTS
    labels = {pat: label for pats, label, _ in OUTPUTS["relion.select"] for pat in pats}
    assert labels["optimisation_set.star"] == tomo_select.LABEL
    assert labels["optimisation_set_split*.star"] == tomo_select.LABEL


@pytest.mark.parametrize("readonly", [False, True])
def test_the_window_writes_the_set_of_a_select_relion_ran(qtbot, tmp_path, tomo_copy, readonly):
    pytest.importorskip("qtpy")
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    job = add_select(tomo_copy, 20, {"fn_data": "Extract/job009/particles.star", "do_remove_duplicates": "Yes"})
    w = MainWindow(tomo_copy, poll_s=1000, readonly=readonly,
                   settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    if readonly:
        qtbot.wait(300)
        assert w._optsets_call is None and not (tomo_copy / job / "optimisation_set.star").exists()
        return
    qtbot.waitUntil(lambda: w._optsets_call is None, timeout=5000)
    assert (tomo_copy / job / "optimisation_set.star").is_file()
    w.load_now()
    assert job + "optimisation_set.star" in w.snapshot.pipeline.processes[job].outputs
