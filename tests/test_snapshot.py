# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import os
import time
from pathlib import Path

from cryo_savanna.project.lock import is_locked
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.ui.graph_layout import layout_graph
from tests.fixtures.make_fixtures import JobSpec, add_job, make_large_project

NEW_JOB = JobSpec(19, "relion.class2d", [("run_it025_optimiser.star", "OptimiserData.star.relion.class2d")],
                  inputs=["Select/job007/particles.star"], options={"fn_img": "Select/job007/particles.star"},
                  status="Running")


def tree_files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.lstat().st_mtime_ns for p in root.rglob("*")}


def test_snapshot_lineage(spa_project):
    snap = ProjectLoader(spa_project).refresh()
    assert snap.roots == ["Import/job001/"]
    assert snap.jobs["Extract/job005/"].children == ["Class2D/job006/", "Class2D/job018/"]
    pp = snap.jobs["PostProcess/job013/"]
    assert pp.primary_parent == "Refine3D/job011/"
    assert pp.other_parents == ["MaskCreate/job012/"]
    assert snap.jobs["Polish/job015/"].status == "Running"
    assert not snap.jobs["Polish/job015/"].pending_update


def test_refresh_returns_none_when_nothing_changed(spa_project):
    loader = ProjectLoader(spa_project)
    assert loader.refresh() is not None
    assert loader.refresh() is None


def test_new_job_is_picked_up(spa_copy):
    loader = ProjectLoader(spa_copy)
    old = loader.refresh()
    add_job(spa_copy, NEW_JOB)
    snap = loader.refresh()
    assert snap is not None and "Class2D/job019/" in snap.jobs
    assert snap.jobs["Class2D/job019/"].primary_parent == "Select/job007/"
    assert "Class2D/job019/" in snap.jobs["Select/job007/"].children
    assert old.structure_key() != snap.structure_key()


def test_finished_running_job_shows_without_writing(spa_copy):
    loader = ProjectLoader(spa_copy)
    loader.refresh()
    before = (spa_copy / "default_pipeline.star").read_bytes()
    (spa_copy / "Polish/job015/RELION_JOB_EXIT_FAILURE").touch()
    snap = loader.refresh()
    assert snap is not None
    polish = snap.jobs["Polish/job015/"]
    assert polish.status == "Failed" and polish.pending_update
    assert polish.process.status == "Running"  # pipeline unchanged: only RELION writes it
    assert (spa_copy / "default_pipeline.star").read_bytes() == before


def test_running_job_activity_updates_modified_time(spa_copy):
    loader = ProjectLoader(spa_copy)
    first = loader.refresh().jobs["Polish/job015/"].modified
    log = spa_copy / "Polish/job015/run.out"
    with open(log, "a") as f:
        f.write(" + more\n")
    later = first + 5
    os.utime(log, (later, later))
    snap = loader.refresh()
    assert snap is not None and snap.jobs["Polish/job015/"].modified == later


def test_loader_never_writes(spa_copy):
    before = tree_files(spa_copy)
    loader = ProjectLoader(spa_copy)
    loader.refresh()
    loader.refresh(force=True)
    assert tree_files(spa_copy) == before
    assert not is_locked(spa_copy)


def test_large_project_loads_fast(tmp_path):
    make_large_project(tmp_path / "big", rounds=35)
    t = time.perf_counter()
    snap = ProjectLoader(tmp_path / "big").refresh()
    lay = layout_graph(snap)
    elapsed = time.perf_counter() - t
    assert len(snap.jobs) == 253
    assert len(lay.pos) == 253
    assert elapsed < 2.0


def make_looped_project(root: Path) -> Path:
    """Import, then an External job whose input is its own output, then two jobs that use each other's outputs."""
    from cryo_savanna.project.pipeline import Node, Pipeline, Process
    from cryo_savanna.project.star import write_star
    p = Pipeline(job_counter=5)

    def job(name: str, type_label: str, status: str, inputs: list[str], output: str) -> None:
        p.processes[name] = Process(name, "None", type_label, status, inputs=list(inputs), outputs=[output])
        p.nodes[output] = Node(output, "ProcessData.star", producer=name)

    job("Import/job001/", "relion.import", "Running", [], "Import/job001/movies.star")
    job("External/job002/", "relion.external", "Running", ["External/job002/out.star"], "External/job002/out.star")
    job("External/job003/", "relion.external", "Succeeded", ["External/job004/out.star"], "External/job003/out.star")
    job("External/job004/", "relion.external", "Succeeded", ["External/job003/out.star"], "External/job004/out.star")
    for pr in p.processes.values():
        for n in pr.inputs:
            p.nodes[n].consumers.append(pr.name)
        (root / pr.name).mkdir(parents=True)
    write_star(root / "default_pipeline.star", p.to_blocks(), atomic=False)
    return root


def test_a_loop_of_parents_keeps_every_job_in_the_tree(tmp_path):
    """Never in a sound pipeline, but a job whose input is its own output must not drop out of the tree."""
    snap = ProjectLoader(make_looped_project(tmp_path)).refresh(force=True)
    seen = []

    def walk(name: str) -> None:
        seen.append(name)
        for c in snap.jobs[name].children:
            walk(c)
    for r in snap.roots:
        walk(r)
    assert sorted(seen) == sorted(snap.jobs)
    assert snap.jobs["External/job002/"].primary_parent is None
    assert "External/job002/" not in snap.jobs["External/job002/"].other_parents
