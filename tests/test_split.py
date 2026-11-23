# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Split off zero-weight particles: the runner (run.err, the failed job's inputs, the split, the optimisation
set) run as RELION runs it, and the form (its inputs, where the job hangs in the tree, which job it starts with)."""

import shlex
import subprocess
import sys

import pytest

from cryo_savanna import runner_common as RC
from cryo_savanna import split_runner as S
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import external_jobs as X
from cryo_savanna.jobdefs.flags import FlagSource
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.split_jobs import (
    SplitCatalog,
    failed_job,
    install_runner,
    normalise_job,
    split_kind,
    split_main_input,
)
from cryo_savanna.project.lineage import primary_parent
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import read_star

ENV = RelionEnv({})
DEFS = JobDefs.load()
CATALOG = SplitCatalog(DEFS, ENV)
JOB = "External/job020/"
FAILED = "Refine3D/job012/"
PARTS = "Extract/job005/particles.star"

# RELION 5.1's error, as relion_refine's GPU code writes it to run.err (src/acc/acc_ml_optimiser_impl.h)
GPU_ERROR = """
 fn_img= {name}
 adaptive_fraction= 0.999
 min_diff2= 1.79769e+308
Dumped data: error_dump_pdf_orientation, error_dump_pdf_orientation and error_dump_unsorted.
in: /opt/relion/src/acc/acc_ml_optimiser_impl.h, line 2255
ERROR:
No orientation was found as better than any other.

A particle image was compared to the reference and resulted in all-zero
weights (for all orientations). This should not happen, unless your data
has very special characteristics. This has historically happened for some
lower-precision calculations, but multiple fallbacks have since been
implemented. Please report this error to the relion developers at
github.com/3dem/relion/issues

=== Backtrace  ===
--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 3 in communicator MPI_COMM_WORLD
"""

OPTICS = """
# version 50001

data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnImagePixelSize #3
_rlnImageSize #4
_rlnImageDimensionality #5
opticsGroup1            1     1.060000    256     2
opticsGroup2            2     1.060000    256     2

"""
# file order is not RELION's: it sorts on rlnMicrographName (mic_a before mic_b), stably
SPA_ROWS = [  # (image, micrograph, optics group)
    ("000001@Extract/job005/Movies/mic_b.mrcs", "MotionCorr/job002/Movies/mic_b.mrc", 2),
    ("000002@Extract/job005/Movies/mic_b.mrcs", "MotionCorr/job002/Movies/mic_b.mrc", 1),
    ("000001@Extract/job005/Movies/mic_a.mrcs", "MotionCorr/job002/Movies/mic_a.mrc", 1),
    ("000002@Extract/job005/Movies/mic_a.mrcs", "MotionCorr/job002/Movies/mic_a.mrc", 2),
    ("000003@Extract/job005/Movies/mic_a.mrcs", "MotionCorr/job002/Movies/mic_a.mrc", 2),
    ("000003@Extract/job005/Movies/mic_b.mrcs", "MotionCorr/job002/Movies/mic_b.mrc", 2),
]
# RELION's order: mic_a 1 (og1), mic_a 2 (og2), mic_a 3 (og2), mic_b 1 (og2), mic_b 2 (og1), mic_b 3 (og2)


def spa_particles(rows=SPA_ROWS):
    text = OPTICS + "\n# version 50001\n\ndata_particles\n\nloop_ \n_rlnImageName #1 \n_rlnMicrographName #2 \n" \
                    "_rlnOpticsGroup #3 \n_rlnAngleRot #4 \n"
    text += "".join(f"{img} {mic} {og} {10.5 * i:.6f} \n" for i, (img, mic, og) in enumerate(rows))
    return text + " \n"


def project(tmp_path, run_err, particles=None, tomo=False, options=None, jobstar=True):
    """A project with the failed job (its job.star and run.err) and its input particles."""
    (tmp_path / "Extract/job005").mkdir(parents=True)
    (tmp_path / PARTS).write_text(spa_particles() if particles is None else particles)
    d = tmp_path / FAILED
    d.mkdir(parents=True)
    (d / "run.err").write_text(run_err)
    (d / "run.out").write_text(" Expectation iteration 3 of 25\n")
    if jobstar:
        js = DEFS.variant("relion.refine3d", tomo).new_jobstar(ENV)
        js.options.update(options if options is not None else {"fn_img": PARTS})
        js.write(d / "job.star")
    return tmp_path


def run(proj, in_job=FAILED, tomo=False, **values):
    """The job as RELION runs it: the project's copy of the runner (python -I), from the project, with the
    form's options as the job.star's "Additional arguments"."""
    v = CATALOG.variant(tomo=tomo)
    js = v.make_jobstar({**v.default_values(ENV), "in_job": in_job, **values}, False, JOB, None)
    install_runner(proj)
    (proj / JOB).mkdir(parents=True, exist_ok=True)
    words = js.options["fn_exe"].split()
    cmd = [sys.executable, "-I"] + words[1:] + ["--o", JOB, "--j", "1"] + shlex.split(js.options["other_args"])
    assert shlex.split(js.options["other_args"])[:2] == ["--in_job", normalise_job(in_job)]
    return subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=60)


def images(path, block="particles"):
    return [r["rlnImageName"] for r in read_star(path)[block].records()] if block in read_star(path) else []


def nodes(proj):
    return [(r["rlnPipeLineNodeName"], r["rlnPipeLineNodeTypeLabel"])
            for r in read_star(proj / JOB / RC.OUTPUT_NODES)["output_nodes"].records()]


def test_the_particles_run_err_names_are_split_off(tmp_path):
    """Two attempts of the job (RELION appends to run.err): both particles go; every other line is copied."""
    err = GPU_ERROR.format(name=SPA_ROWS[4][0]) + GPU_ERROR.format(name=SPA_ROWS[1][0])
    proj = project(tmp_path, err)
    r = run(proj)
    assert r.returncode == 0, r.stderr
    acc, rej = proj / JOB / S.ACCEPTED, proj / JOB / S.REJECTED
    assert images(rej) == [SPA_ROWS[1][0], SPA_ROWS[4][0]]
    assert images(acc) == [x[0] for i, x in enumerate(SPA_ROWS) if i not in (1, 4)]
    src = (proj / PARTS).read_text().splitlines()  # line by line as it was, each row in one of the two
    gone = (SPA_ROWS[1][0], SPA_ROWS[4][0])
    assert acc.read_text().splitlines() == [x for x in src if not x.startswith(gone)]
    assert rej.read_text().splitlines() == [x for x in src if "@" not in x or x.startswith(gone)]
    assert list(read_star(acc)["optics"].records()) == list(read_star(proj / PARTS)["optics"].records())
    assert nodes(proj) == [(JOB + S.ACCEPTED, S.NODE_PARTICLES), (JOB + S.REJECTED, S.NODE_PARTICLES)]
    assert (proj / JOB / RC.SUCCESS).exists() and not (proj / JOB / S.OPTIMISATION_SET).exists()
    assert "2 of 6 particles rejected" in r.stdout


@pytest.mark.parametrize("folder", ["Refine3D/job012", "./Refine3D/job012/", "Refine3D/job012/"])
def test_the_failed_job_folder_can_be_written_any_way(tmp_path, folder):
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[0][0]))
    assert run(proj, folder).returncode == 0
    assert images(proj / JOB / S.REJECTED) == [SPA_ROWS[0][0]]


@pytest.mark.parametrize("name, row", [
    ("000002@/ssd/relion_volatile/opticsgroup2_particles.mrcs", 4),  # og2 in RELION's order: rows 3, 4, 0, 5
    ("000001@/ssd/relion_volatile/opticsgroup1_particles.mrcs", 2),  # og1: rows 2, 1
    ("000002@/ssd/relion_volatile/opticsgroup1_particles.mrcs", 1),
    ("000004@relion_volatile/opticsgroup2_particles.mrcs", 5),
])
def test_a_particle_on_the_scratch_disk_is_found_by_its_place(tmp_path, name, row):
    """--scratch_dir: RELION names the copy, the n-th particle of its optics group in RELION's own order."""
    proj = project(tmp_path, GPU_ERROR.format(name=name))
    r = run(proj)
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == [SPA_ROWS[row][0]]


def test_the_cpu_code_names_the_particle_by_its_index(tmp_path):
    """relion_refine without GPUs: " part_id= N", N in RELION's order (0-based)."""
    err = (" exp_thisimage_sumweight= 0\n part_id= 3\n group_id= 0 mymodel.scale_correction[group_id]= 1\n"
           " exp_ipass= 0\nin: /opt/relion/src/ml_optimiser.cpp, line 8123\nERROR: \nERROR!!! zero sum of "
           "weights....\n")
    proj = project(tmp_path, err)
    assert run(proj).returncode == 0
    assert images(proj / JOB / S.REJECTED) == [SPA_ROWS[0][0]]  # 4th of: rows 2, 3, 4, 0, 1, 5


def test_without_job_star_the_command_in_note_txt_names_the_input(tmp_path):
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[2][0]), jobstar=False)
    (proj / FAILED / "note.txt").write_text(
        " ++++ Executing new job on Mon Nov 16 10:00:00 2026\n ++++ with the following command(s): \n"
        f"`which relion_refine_mpi` --o {FAILED}run --auto_refine --split_random_halves --i {PARTS} --ref "
        "InitialModel/job010/initial_model.mrc --gpu \"\" --pipeline_control Refine3D/job012/\n ++++\n")
    r = run(proj)
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == [SPA_ROWS[2][0]]


@pytest.mark.parametrize("err, message", [
    (" Expectation iteration 3 of 25\nERROR: \nmask and map have different sizes\n", "does not show RELION's"),
    ("ERROR: \nNo orientation was found as better than any other.\n", "names no particle"),
    (GPU_ERROR.format(name="000009@Extract/job005/Movies/mic_z.mrcs"), "none of the particles"),
])
def test_it_fails_and_says_why(tmp_path, err, message):
    proj = project(tmp_path, err)
    r = run(proj)
    assert r.returncode == 2 and message in r.stderr, r.stderr
    assert (proj / JOB / RC.FAILURE).exists() and not (proj / JOB / S.ACCEPTED).exists()


def test_a_missing_failed_job_or_input(tmp_path):
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[0][0]))
    r = run(proj, "Refine3D/job099/")
    assert r.returncode == 2 and "not there: Refine3D/job099/" in r.stderr
    (proj / PARTS).unlink()
    r = run(proj)
    assert r.returncode == 2 and "input particles of Refine3D/job012/ are not there" in r.stderr


# ---------------------------------------------------------------------------- tomography

TOMO_PARTS = "Extract/job010/particles.star"
OPTSET = "Extract/job010/optimisation_set.star"
TOMO_ROWS = [f"Extract/job010/Subtomograms/TS_0{t}/{n}_stack2d.mrcs" for t in (1, 2) for n in (1, 2, 3)]


def tomo_particles():
    text = """
# version 50001

data_general

_rlnTomoSubTomosAre2DStacks                       1
""" + OPTICS + """
# version 50001

data_particles

loop_
_rlnTomoName #1
_rlnCenteredCoordinateXAngst #2
_rlnOpticsGroup #3
_rlnImageName #4
"""
    text += "".join(f"TS_0{1 + i // 3}   {100.0 * i:.6f}  1  {img} \n" for i, img in enumerate(TOMO_ROWS))
    return text + " \n"


def tomo_project(tmp_path, err, options):
    proj = project(tmp_path, err, particles=spa_particles(), tomo=True, options=options)
    (proj / "Extract/job010").mkdir(parents=True)
    (proj / TOMO_PARTS).write_text(tomo_particles())
    (proj / OPTSET).write_text(f"\n# version 50001\n\ndata_\n\n_rlnTomoParticlesFile   {TOMO_PARTS}\n"
                               "_rlnTomoTomogramsFile   Tomograms/job008/tomograms.star\n\n")
    return proj


def test_tomography_gets_an_optimisation_set_with_the_accepted_particles(tmp_path):
    proj = tomo_project(tmp_path, GPU_ERROR.format(name=TOMO_ROWS[4]),
                        {"in_optimisation": OPTSET, "use_direct_entries": "No", "in_particles": ""})
    r = run(proj, tomo=True)
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == [TOMO_ROWS[4]]
    assert images(proj / JOB / S.ACCEPTED) == TOMO_ROWS[:4] + TOMO_ROWS[5:]
    acc = read_star(proj / JOB / S.ACCEPTED)
    assert acc["general"].as_dict() == {"rlnTomoSubTomosAre2DStacks": "1"}
    opt = read_star(proj / JOB / S.OPTIMISATION_SET)[""].as_dict()
    assert opt == {"rlnTomoParticlesFile": JOB + S.ACCEPTED, "rlnTomoTomogramsFile": "Tomograms/job008/tomograms.star"}
    assert nodes(proj)[-1] == (JOB + S.OPTIMISATION_SET, S.NODE_OPTIMISATION_SET)


def test_tomography_can_reject_whole_tomograms(tmp_path):
    """--reject_tomograms Yes: every particle of TS_02 goes, the optimisation set keeps TS_01's."""
    proj = tomo_project(tmp_path, GPU_ERROR.format(name=TOMO_ROWS[4]),
                        {"in_optimisation": OPTSET, "use_direct_entries": "No", "in_particles": ""})
    r = run(proj, tomo=True, reject_tomograms="Yes")
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == TOMO_ROWS[3:]
    assert images(proj / JOB / S.ACCEPTED) == TOMO_ROWS[:3]
    assert "every particle of 1 tomogram rejected: TS_02" in r.stdout and "3 of 6 particles rejected" in r.stdout
    opt = read_star(proj / JOB / S.OPTIMISATION_SET)[""].as_dict()
    assert opt["rlnTomoParticlesFile"] == JOB + S.ACCEPTED
    # both tomograms named: nothing would be left
    (proj / FAILED / "run.err").write_text(GPU_ERROR.format(name=TOMO_ROWS[0]) + GPU_ERROR.format(name=TOMO_ROWS[5]))
    r = run(proj, tomo=True, reject_tomograms="Yes")
    assert r.returncode == 2 and "every particle of Extract/job010/particles.star would be rejected" in r.stderr


def test_whole_tomograms_need_tomogram_names(tmp_path):
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[0][0]))
    r = run(proj, tomo=True, reject_tomograms="Yes")
    assert r.returncode == 2 and "no rlnTomoName" in r.stderr


def test_each_form_offers_its_own_whole_groups():
    """Tomography: whole tomograms; SPA: whole micrographs. Both off to start with."""
    tomo, spa = CATALOG.variant(tomo=True), CATALOG.variant(tomo=False)
    assert S.WHOLE_TOMOGRAMS in tomo.options and S.WHOLE_MICROGRAPHS not in tomo.options
    assert S.WHOLE_MICROGRAPHS in spa.options and S.WHOLE_TOMOGRAMS not in spa.options
    assert tomo.default_values(ENV)[S.WHOLE_TOMOGRAMS] == spa.default_values(ENV)[S.WHOLE_MICROGRAPHS] == "No"


@pytest.mark.parametrize("name", [SPA_ROWS[4][0], "000002@/ssd/relion_volatile/opticsgroup2_particles.mrcs"])
def test_spa_can_reject_whole_micrographs(tmp_path, name):
    """--reject_micrographs Yes: every particle of mic_a goes (also when run.err names the scratch copy)."""
    proj = project(tmp_path, GPU_ERROR.format(name=name))
    r = run(proj, reject_micrographs="Yes")
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == [x[0] for x in SPA_ROWS[2:5]]
    assert images(proj / JOB / S.ACCEPTED) == [x[0] for x in SPA_ROWS[:2] + SPA_ROWS[5:]]
    assert "every particle of 1 micrograph rejected: MotionCorr/job002/Movies/mic_a.mrc" in r.stdout
    assert "3 of 6 particles rejected" in r.stdout
    # particles on both micrographs: nothing would be left
    (proj / FAILED / "run.err").write_text(GPU_ERROR.format(name=SPA_ROWS[0][0]) + GPU_ERROR.format(name=name))
    r = run(proj, reject_micrographs="Yes")
    assert r.returncode == 2 and "would be rejected" in r.stderr


def test_whole_micrographs_need_micrograph_names(tmp_path):
    proj = project(tmp_path, GPU_ERROR.format(name=TOMO_ROWS[0]))
    (proj / PARTS).write_text(tomo_particles())  # no rlnMicrographName
    r = run(proj, reject_micrographs="Yes")
    assert r.returncode == 2 and "no rlnMicrographName" in r.stderr


def test_whole_tomograms_and_micrographs_are_not_both_rejected(tmp_path):
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[0][0]))
    install_runner(proj)
    (proj / JOB).mkdir(parents=True)
    r = subprocess.run([sys.executable, "-I", ".cryo-savanna/split_runner.py", S.ZERO_WEIGHTS, "--o", JOB, "--j", "1",
                        "--in_job", FAILED, "--reject_tomograms", "Yes", "--reject_micrographs", "Yes"],
                       cwd=proj, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "not both" in r.stderr


def test_tomography_with_direct_entries(tmp_path):
    proj = tomo_project(tmp_path, GPU_ERROR.format(name="/scratch/relion_volatile/opticsgroup1_particle2.mrcs"),
                        {"use_direct_entries": "Yes", "in_optimisation": "", "in_particles": TOMO_PARTS,
                         "in_tomograms": "Tomograms/job008/tomograms.star",
                         "in_trajectories": "Polish/job030/motion.star"})
    r = run(proj, tomo=True)
    assert r.returncode == 0, r.stderr
    assert images(proj / JOB / S.REJECTED) == [TOMO_ROWS[1]]  # 2D stacks: no sort; the 2nd of optics group 1
    opt = read_star(proj / JOB / S.OPTIMISATION_SET)[""].as_dict()
    assert opt == {"rlnTomoParticlesFile": JOB + S.ACCEPTED, "rlnTomoTomogramsFile": "Tomograms/job008/tomograms.star",
                   "rlnTomoTrajectoriesFile": "Polish/job030/motion.star"}


def test_the_split_runs_in_process_too(tmp_path, monkeypatch):
    """run() itself, as cryo-savanna imports it (the module's own runner_common)."""
    proj = project(tmp_path, GPU_ERROR.format(name=SPA_ROWS[3][0]))
    (proj / JOB).mkdir(parents=True)
    monkeypatch.chdir(proj)
    assert S.run(["zero_weights", "--o", JOB, "--j", "1", "--in_job", FAILED]) == 0
    assert images(proj / JOB / S.REJECTED) == [SPA_ROWS[3][0]]


# ---------------------------------------------------------------------------- the form

def pipeline(tomo=False):
    p = Pipeline()
    parts, optset = (TOMO_PARTS, OPTSET) if tomo else (PARTS, None)
    p.processes["Extract/job005/"] = Process("Extract/job005/", "None", "relion.extract", "Succeeded",
                                             outputs=[parts] + ([optset] if optset else []))
    p.nodes[parts] = Node(parts, "ParticleGroupMetadata.star.relion", producer="Extract/job005/")
    inputs = [parts, "InitialModel/job010/initial_model.mrc"]
    p.nodes["InitialModel/job010/initial_model.mrc"] = Node("InitialModel/job010/initial_model.mrc",
                                                            "DensityMap.mrc", producer="InitialModel/job010/")
    p.processes["InitialModel/job010/"] = Process("InitialModel/job010/", "None", "relion.initialmodel", "Succeeded")
    if optset:
        p.nodes[optset] = Node(optset, "TomoOptimisationSet.star.relion", producer="Extract/job005/")
        inputs = [optset] + inputs[1:]
    p.processes[FAILED] = Process(FAILED, "None", "relion.refine3d" + (".tomo" if tomo else ""), "Failed",
                                  inputs=inputs, outputs=[FAILED + "run_data.star"])
    p.processes["Class3D/job013/"] = Process("Class3D/job013/", "None", "relion.class3d", "Succeeded")
    return p


@pytest.mark.parametrize("tomo", [False, True])
def test_the_form_links_the_failed_jobs_particles_and_hangs_under_it(tmp_path, tomo):
    p = pipeline(tomo)
    forms = X.Forms(DEFS, ENV, FlagSource(which=lambda program: None))
    v = forms.variant(X.SPLIT, S.ZERO_WEIGHTS, tomo)
    js = v.make_jobstar({**v.default_values(ENV), "in_job": "./Refine3D/job012"}, False, JOB, p)
    assert js.is_tomo == tomo and js.type_label == "relion.external"
    assert js.options["other_args"] == "--in_job Refine3D/job012/" + (" --reject_tomograms No" if tomo else " --reject_micrographs No")
    linked = [js.options[k] for k in ("in_mov", "in_mic", "in_part", "in_coords", "in_3dref", "in_mask") if js.options.get(k)]
    assert linked == [OPTSET if tomo else PARTS]
    assert X.kind_of(js)[0].name == X.SPLIT and split_kind(js.options["fn_exe"]) == S.ZERO_WEIGHTS
    assert X.main_input(js) == split_main_input(js) == FAILED
    assert forms.variant_for(js) is v
    # in the tree: under the failed job (its folder is no node), not under Extract
    p.processes[JOB] = Process(JOB, "None", "relion.external", "Succeeded", inputs=linked)
    (tmp_path / JOB).mkdir(parents=True)
    js.write(tmp_path / JOB / "job.star")
    assert primary_parent(p, JOB, tmp_path, DEFS) == FAILED


def test_the_form_starts_with_the_selected_or_newest_failed_refinement():
    p = pipeline()
    assert failed_job(p, FAILED) == {"in_job": FAILED}
    assert failed_job(p, "Class3D/job013/") == {"in_job": FAILED}  # not failed: the newest failed one
    p.processes["Class2D/job014/"] = Process("Class2D/job014/", "None", "relion.class2d", "Failed")
    p.processes["PostProcess/job015/"] = Process("PostProcess/job015/", "None", "relion.postprocess", "Failed")
    assert failed_job(p, None) == {"in_job": "Class2D/job014/"}
    assert failed_job(Pipeline(), None) == {}


def test_normalise_job():
    assert normalise_job(" ./Refine3D/job012 ") == FAILED
    assert normalise_job("Refine3D/job012//") == FAILED
    assert normalise_job("") == normalise_job("./") == ""


def test_new_split_job_in_window(qtbot, spa_copy, tmp_path):
    """The menu entry, and the form started with the failed refinement; run as RELION would on the fixture's
    particles."""
    import re

    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    pipe = spa_copy / "default_pipeline.star"
    text = pipe.read_text()
    text, n = re.subn(r"^(Refine3D/job011/\s+\S+\s+relion\.refine3d\s+)Succeeded", r"\1Failed", text, flags=re.M)
    assert n == 1
    pipe.write_text(text)
    w = MainWindow(spa_copy, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    assert "Split off zero-weight particles (of a failed refinement)" in \
        [a.text() for a in w.actions.external_menu.actions()]
    w.actions.new_split_job()
    ed = w.actions.editor
    assert ed.variant.kind == S.ZERO_WEIGHTS and not ed.variant.is_tomo
    assert "Split off zero-weight particles" in ed.title.text()
    assert ed.values()["in_job"] == "Refine3D/job011/"
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())] == ["Input", "Running"]
    assert S.WHOLE_TOMOGRAMS not in ed.values()  # SPA: no tomograms to reject, but micrographs
    assert ed.values()[S.WHOLE_MICROGRAPHS] == "No"
    js = ed.variant.make_jobstar(ed.values(), False, JOB, w.actions.snapshot.pipeline)
    linked = [js.options[k] for k in ("in_mov", "in_mic", "in_part", "in_coords", "in_3dref", "in_mask")
              if js.options.get(k)]
    failed = w.actions.snapshot.pipeline.processes["Refine3D/job011/"]
    assert len(linked) == 1 and linked[0] in failed.inputs
    # the fixture's job stopped on one of its particles: the job splits it off
    first = next(read_star(spa_copy / linked[0])["particles"].records())["rlnImageName"]
    (spa_copy / "Refine3D/job011/run.err").write_text(GPU_ERROR.format(name=first))
    r = run(spa_copy, "Refine3D/job011/")
    assert r.returncode == 0, r.stderr
    assert images(spa_copy / JOB / S.REJECTED) == [first]


def test_new_split_job_in_window_for_tomography(qtbot, tomo_project, tmp_path):
    """The tomography form: its whole-tomograms option, off to start with; the failed refinement filled in."""
    import re
    import shutil

    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    proj = tmp_path / "tomo"
    shutil.copytree(tomo_project, proj, symlinks=True)
    pipe = proj / "default_pipeline.star"
    text, n = re.subn(r"^(Refine3D/job011/\s+\S+\s+relion\.refine3d\S*\s+)Succeeded", r"\1Failed",
                      pipe.read_text(), flags=re.M)
    assert n == 1
    pipe.write_text(text)
    w = MainWindow(proj, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.actions.new_split_job()
    ed = w.actions.editor
    assert ed.variant.is_tomo
    assert ed.values()["in_job"] == "Refine3D/job011/" and ed.values()[S.WHOLE_TOMOGRAMS] == "No"
    ed.widgets[S.WHOLE_TOMOGRAMS].set_value("Yes")
    js = ed.variant.make_jobstar(ed.values(), False, JOB, w.actions.snapshot.pipeline)
    assert js.is_tomo and "--reject_tomograms Yes" in js.options["other_args"]
