# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.lineage import primary_input, primary_parents
from cryo_savanna.project.pipeline import Pipeline


def test_spa_lineage(spa_project):
    p = Pipeline.read(spa_project)
    parents = primary_parents(p, spa_project, JobDefs.load())
    assert parents["Import/job001/"] is None
    assert parents["Extract/job005/"] == "AutoPick/job004/"  # curated: picks, not micrographs
    assert parents["PostProcess/job013/"] == "Refine3D/job011/"  # half-map, although RELION lists the mask first
    assert parents["Polish/job015/"] == "CtfRefine/job014/"  # curated: particles, not micrographs
    assert parents["Refine3D/job016/"] == "Polish/job015/"
    assert parents["Class3D/job009/"] == "Select/job007/"


def test_tomo_lineage(tomo_project):
    p = Pipeline.read(tomo_project)
    parents = primary_parents(p, tomo_project, JobDefs.load())
    assert parents["Extract/job009/"] == "Picks/job008/"
    assert parents["Refine3D/job011/"] == "Extract/job009/"
    assert parents["CtfRefine/job014/"] == "Refine3D/job011/"


def test_missing_jobstar_falls_back_to_first_input(spa_project, tmp_path):
    p = Pipeline.read(spa_project)
    # tmp_path has no job directories at all
    assert primary_input(p, "PostProcess/job013/", tmp_path, JobDefs.load()) == p.processes["PostProcess/job013/"].inputs[0]


def test_a_loop_is_cut_inside_the_loop_only():
    from cryo_savanna.project.lineage import cut_loops
    for order in (["A", "B", "C"], ["B", "C", "A"], ["C", "B", "A"]):
        parents = {"A": "B", "B": "C", "C": "B"}  # A hangs below the loop B <-> C
        out = cut_loops({k: parents[k] for k in order})
        assert out["A"] == "B" and [out["B"], out["C"]].count(None) == 1
    assert cut_loops({"X": "X", "Y": "X"}) == {"X": None, "Y": "X"}


def _job_with_inputs(*inputs: tuple[str, str]) -> Pipeline:
    """A job (no job.star: inputs in pipeline order) whose inputs come from other jobs."""
    from cryo_savanna.project.pipeline import Node, Process
    p = Pipeline()
    job = Process("Refine3D/job009/", "None", "relion.refine3d", "Succeeded", inputs=[n for n, _ in inputs])
    p.processes[job.name] = job
    for n, label in inputs:
        producer = n.rsplit("/", 1)[0] + "/"
        p.processes.setdefault(producer, Process(producer, "None", "relion.external", "Succeeded", outputs=[n]))
        p.nodes[n] = Node(n, label, producer=producer, consumers=[job.name])
    return p


def test_main_input_is_the_optimisation_set_else_the_particles(tmp_path):
    ref = ("InitialModel/job004/ref.mrc", "DensityMap.mrc")
    parts = ("Extract/job003/particles.star", "ParticleGroupMetadata.star.relion")
    opt = ("Extract/job003/optimisation_set.star", "TomoOptimisationSet.star.relion")
    defs = JobDefs.load()
    for inputs, want in (((ref, parts, opt), opt), ((ref, parts), parts), ((ref,), ref)):
        p = _job_with_inputs(*inputs)
        assert primary_input(p, "Refine3D/job009/", tmp_path, defs) == want[0]


def test_a_main_input_chosen_here_wins_and_is_kept(spa_copy):
    from cryo_savanna.project.main_inputs import MainInputs
    p = Pipeline.read(spa_copy)
    mask = next(n for n in p.processes["PostProcess/job013/"].inputs if n.startswith("MaskCreate/"))
    mi = MainInputs(spa_copy)
    mi.set("PostProcess/job013/", mask)
    parents = primary_parents(p, spa_copy, JobDefs.load(), main_inputs=MainInputs(spa_copy).choices())
    assert parents["PostProcess/job013/"] == "MaskCreate/job012/"
    # a choice that is no longer one of the job's inputs is ignored; None goes back to the default
    other = primary_parents(p, spa_copy, JobDefs.load(), main_inputs={"PostProcess/job013/": "Gone/job099/x.star"})
    assert other["PostProcess/job013/"] == "Refine3D/job011/"
    mi.set("PostProcess/job013/", None)
    assert MainInputs(spa_copy).choices() == {}
    ro = MainInputs(spa_copy, persist=False)  # --readonly: only for the session
    ro.set("PostProcess/job013/", mask)
    assert ro.choices() == {"PostProcess/job013/": mask} and MainInputs(spa_copy).choices() == {}
