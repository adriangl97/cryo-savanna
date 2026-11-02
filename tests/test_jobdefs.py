# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import pytest

from cryo_savanna.env.relion_env import RelionEnv, RelionEnvError, float_to_string
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.relion_consts import JOB_TYPES

RUNNING_KEYS = ["do_queue", "queuename", "qsub", "qsubscript", "min_dedicated", "other_args"]


@pytest.fixture(scope="module")
def defs() -> JobDefs:
    return JobDefs.load()


def test_every_job_type_has_spa_and_tomo_variant(defs):
    assert defs.version == "5.1.1"
    for jt in JOB_TYPES:
        for tomo in (False, True):
            v = defs.variant(jt.label, tomo)
            assert v.proc_id == jt.proc_id and v.dirname == jt.dirname and v.is_tomo == tomo
            for k in RUNNING_KEYS:
                assert k in v.options, (v.key, k)


def test_menus(defs):
    spa = [label for label, _ in defs.menu(tomo=False)]
    tomo = [(label, v.key) for label, v in defs.menu(tomo=True)]
    assert spa[:3] == ["Import", "Motion correction", "CTF estimation"]
    assert "External" == spa[-1]
    assert ("Import", "relion.importtomo") in tomo
    assert ("3D auto-refine", "relion.refine3d:tomo") in tomo
    assert ("Subset selection", "relion.select") in tomo  # RELION does not pass is_tomo for Select
    assert all(label != "Particle subtraction" for label, _ in tomo)


def test_layout_is_consistent(defs):
    for v in defs.variants.values():
        placed = set()
        for t in v.tabs:
            for r in t.rows:
                for k in r.keys:
                    assert k in v.options, (v.key, k)
                    placed.add(k)
                if r.group is not None:
                    assert r.group in v.groups, (v.key, r)
        for g in v.groups.values():
            if g.controller is not None:
                assert v.options[g.controller].kind == "boolean", (v.key, g)
            if g.parent is not None:
                assert g.parent in v.groups
        assert v.tabs[-1].label == "Running"
        # Every option is shown somewhere, except a few that RELION defines without a widget
        assert placed <= set(v.options)


def test_class2d_details(defs):
    v = defs.variant("relion.class2d")
    assert [t.label for t in v.tabs] == ["I/O", "CTF", "Optimisation", "Sampling", "Helix", "Compute", "Running"]
    nr = v.options["nr_classes"]
    assert (nr.kind, nr.default, nr.min, nr.max, nr.step, nr.toggle) == ("slider", "1", 1, 50, 1, "deactivate")
    assert v.options["tau_fudge"].step == 0.1
    assert v.options["fn_img"].node_type == "ParticleGroupMetadata.star.relion"
    assert v.options["fn_cont"].toggle == "reactivate"
    assert v.input_options() == ["fn_img"]
    # do_ctf_correction switches the group holding ctf_intact_first_peak
    ctf_tab = v.tabs[1]
    gid = ctf_tab.rows[1].group
    assert v.groups[gid].controller == "do_ctf_correction"
    assert v.group_active(gid, {"do_ctf_correction": "Yes"})
    assert not v.group_active(gid, {"do_ctf_correction": "No"})
    assert v.group_active(gid, {"do_ctf_correction": "$$ctf"})  # Schemes variables keep groups active


def test_inverted_and_nested_groups(defs):
    v = defs.variant("relion.class2d")
    preread = [g for g in v.groups.values() if g.controller == "do_preread_images"][0]
    assert preread.active_when == "No"  # RELION's actually_activate=true
    xoff = [g for g in v.groups.values() if g.controller == "do_restrict_xoff"][0]
    helix = [g for g in v.groups.values() if g.controller == "do_helix"][0]
    assert xoff.parent == helix.id
    vals = {"do_helix": "No", "do_restrict_xoff": "Yes"}
    assert not v.group_active(xoff.id, vals)


def test_tomo_variants_differ(defs):
    spa, tomo = defs.variant("relion.motioncorr"), defs.variant("relion.motioncorr", True)
    assert "dose_per_frame" in spa.options and "dose_per_frame" not in tomo.options
    assert "do_even_odd_split" in tomo.options
    assert "in_optimisation" in defs.variant("relion.refine3d", True).options
    assert "in_optimisation" not in defs.variant("relion.refine3d").options


def test_modes(defs):
    v = defs.variant("relion.class2d")
    assert v.enabled_for_mode(v.options["fn_img"], is_continue=False)
    assert not v.enabled_for_mode(v.options["fn_img"], is_continue=True)
    assert v.enabled_for_mode(v.options["fn_cont"], is_continue=True)
    assert v.enabled_for_mode(v.options["nr_mpi"], is_continue=True)


# ------------------------------------------------------------------ environment

def test_float_to_string():
    assert [float_to_string(x) for x in (1, 0.5, 2.0, 0.1, 1e6, -1, 64)] == ["1", "0.5", "2", "0.1", "1e+06", "-1", "64"]


def test_compiled_defaults(defs):
    o = defs.variant("relion.class2d").resolved_options(RelionEnv({}))
    assert o["qsubscript"].default == "/public/EM/RELION/relion/bin/relion_qsub.csh"
    assert o["qsub"].default == "sbatch"
    assert o["queuename"].default == "openmpi"
    assert o["do_queue"].default == "No"
    assert (o["nr_mpi"].default, o["nr_mpi"].max) == ("1", 64)
    assert (o["nr_threads"].default, o["nr_threads"].max) == ("1", 16)
    assert not o["min_dedicated"].forced_inactive
    assert not any(k.startswith("qsub_extra") for k in o)


def test_env_resolution(defs):
    env = RelionEnv({
        "RELION_QSUB_TEMPLATE": "/opt/slurm/relion_submit.sh", "RELION_QSUB_COMMAND": "sbatch --parsable",
        "RELION_QUEUE_NAME": "gpu", "RELION_QUEUE_USE": "TRUE", "RELION_MPI_MAX": "512", "RELION_THREAD_MAX": "48",
        "RELION_QSUB_NRMPI": "5", "RELION_QSUB_NRTHREADS": "300",  # 300 wraps like RELION's char
        "RELION_MINIMUM_DEDICATED": "32", "RELION_ALLOW_CHANGE_MINIMUM_DEDICATED": "0",
        "RELION_QSUB_EXTRA_COUNT": "3", "RELION_QSUB_EXTRA1": "Time limit", "RELION_QSUB_EXTRA1_DEFAULT": "24:00:00",
        "RELION_QSUB_EXTRA3": "Account", "RELION_QSUB_EXTRA3_HELP": "Project account",
        "RELION_CTFFIND_EXECUTABLE": "/opt/ctffind/ctffind5",
    })
    v = defs.variant("relion.class2d")
    o = v.resolved_options(env)
    assert o["qsubscript"].default == "/opt/slurm/relion_submit.sh"
    assert o["qsub"].default == "sbatch --parsable"
    assert o["queuename"].default == "gpu"
    assert o["do_queue"].default == "Yes"
    assert (o["nr_mpi"].default, o["nr_mpi"].max) == ("5", 512)
    assert (o["nr_threads"].default, o["nr_threads"].max) == ("44", 48)
    assert o["min_dedicated"].default == "32" and o["min_dedicated"].forced_inactive
    assert [k for k in o if k.startswith("qsub_extra")] == ["qsub_extra1", "qsub_extra3"]
    assert (o["qsub_extra1"].label, o["qsub_extra1"].default) == ("Time limit", "24:00:00")
    assert o["qsub_extra3"].help == "Project account"
    assert "XXXextra1XXX" in o["qsub_extra1"].help
    running = [r.keys[0] for r in v.resolved_tabs(env)[-1].rows]
    assert running.index("qsub_extra1") == running.index("qsub") + 1
    assert running.index("qsubscript") == running.index("qsub_extra3") + 1
    ctf = defs.variant("relion.ctffind").resolved_options(env)
    assert ctf["fn_ctffind_exe"].default == "/opt/ctffind/ctffind5"


def test_env_errors_like_relion(defs):
    with pytest.raises(RelionEnvError):
        defs.variant("relion.class2d").resolved_options(RelionEnv({"RELION_QUEUE_USE": "maybe"}))
    with pytest.raises(RelionEnvError):
        defs.variant("relion.class2d").resolved_options(RelionEnv({"RELION_MPI_MAX": "lots"}))


def test_new_jobstar_and_merge(defs):
    env = RelionEnv({})
    v = defs.variant("relion.refine3d", True)
    js = v.new_jobstar(env)
    assert js.type_label == "relion.refine3d" and js.is_tomo and not js.is_continue
    assert set(js.options) == set(v.options)
    old = JobStar("relion.refine3d", False, True, {"fn_ref": "x.mrc", "removed_in_5_1": "1"})
    values, ignored = v.values_from_jobstar(old, env)
    assert values["fn_ref"] == "x.mrc" and values["sym_name"] == v.options["sym_name"].default
    assert ignored == ["removed_in_5_1"]


def test_ccpem_style_labels(defs):
    js = JobStar("relion.class2d.em", False, False, {})
    assert defs.variant_for_jobstar(js).key == "relion.class2d"


def test_fixture_jobstars_match_definitions(spa_project, tomo_project, defs):
    env = RelionEnv({})
    for root in (spa_project, tomo_project):
        for js_path in root.glob("*/job*/job.star"):
            js = JobStar.read(js_path)
            v = defs.variant_for_jobstar(js)
            _, ignored = v.values_from_jobstar(js, env)
            assert ignored == [], js_path
            assert JobStar.read(js_path).to_text() == js_path.read_text()
