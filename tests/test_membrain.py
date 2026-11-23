# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""MemBrain-seg jobs: the form, the runner (with a fake membrain on the PATH), the views."""

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import mrcfile
import numpy as np
import pytest

from cryo_savanna import membrain_runner as M
from cryo_savanna import runner_common as C
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import external_jobs as X
from cryo_savanna.jobdefs.cli_tools import external_program
from cryo_savanna.jobdefs.membrain_jobs import (
    MembrainCatalog,
    delete_score_maps,
    read_thresholds,
    save_thresholds,
    install_runner,
    last_settings,
    membrain_kind,
    membrain_kind_of,
    membrain_main_input,
    runner_command,
)
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import decode
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.project.star import read_star
from cryo_savanna.results import registry as R

ENV = RelionEnv({})
CATALOG = MembrainCatalog(JobDefs.load(), ENV)
NAMES = ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"]
CKPT = "MemBrain_seg_v10_alpha.ckpt"

# ---------------------------------------------------------------------------- a fake membrain

FAKE = """
    import os, sys, time, numpy as np, mrcfile
    a = sys.argv[1:]
    def arg(f):
        return a[a.index(f) + 1]
    log = os.environ["FAKE_LOG"]
    open(log, "a").write(" ".join(a) + "\\n")
    open(log + ".gpus", "a").write("%s %s\\n" % (a[0], os.environ.get("CUDA_VISIBLE_DEVICES", "-")))
    if os.environ.get("FAKE_SLEEP"):
        time.sleep(float(os.environ["FAKE_SLEEP"]))
    out = arg("--out-folder")
    os.makedirs(out, exist_ok=True)
    if a[0] == "segment":
        tomo = arg("--tomogram-path")
        if os.environ.get("FAKE_FAIL") and os.environ["FAKE_FAIL"] in tomo:
            print("Traceback (most recent call last):\\nRuntimeError: CUDA out of memory")
            sys.exit(3)
        stem = os.path.splitext(os.path.basename(tomo))[0]
        token = os.path.basename(arg("--ckpt-path")).split("-val_loss")[0]
        with mrcfile.open(tomo, permissive=True) as m:
            data, voxel = np.asarray(m.data), m.voxel_size
        seg = (data > float(arg("--segmentation-threshold"))).astype(np.int8)
        def write(name, d):
            with mrcfile.new(os.path.join(out, name), overwrite=True) as m:
                m.set_data(d)
                m.voxel_size = voxel
        write("%s_%s_segmented.mrc" % (stem, token), seg)
        if "--store-probabilities" in a:
            write(stem + "_scores.mrc", data.astype(np.float32))
        if "--store-uncertainty-map" in a:
            write("%s_%s_uncertainty_map.mrc" % (stem, token), np.zeros_like(seg))
    elif a[0] == "skeletonize":
        label = arg("--label-path")
        stem = os.path.splitext(os.path.basename(label))[0]
        with mrcfile.open(label, permissive=True) as m:
            d = np.asarray(m.data)
        with mrcfile.new(os.path.join(out, stem + "_skel.mrc"), overwrite=True) as m:
            m.set_data((d > 0).astype(np.float32))
    elif a[0] == "thresholds":
        scores = arg("--scoremap-path")
        stem = os.path.splitext(os.path.basename(scores))[0]
        t = float(arg("--thresholds"))
        with mrcfile.open(scores, permissive=True) as m:
            d, voxel = np.asarray(m.data), m.voxel_size
        with mrcfile.new(os.path.join(out, stem + f"_threshold_{t}.mrc"), overwrite=True) as m:
            m.set_data((d > t).astype(np.int8))
            m.voxel_size = voxel
    elif a[0] == "components":
        seg = arg("--segmentation-path")
        stem = os.path.splitext(os.path.basename(seg))[0]
        with mrcfile.open(seg, permissive=True) as m:
            d = np.asarray(m.data)
        with mrcfile.new(os.path.join(out, stem + "_components.mrc"), overwrite=True) as m:
            m.set_data((d > 0).astype(np.int8) * 7)  # "labels"
    else:
        sys.exit(2)
"""


@pytest.fixture
def fake_membrain(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    f = bin_dir / "membrain"
    f.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE))
    f.chmod(0o755)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CRYO_SAVANNA_POLL_S", "0.05")
    return log


@pytest.fixture
def project(tomo_project, tmp_path):
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    (dst / CKPT).write_text("a model")
    return dst


def values(**kw):
    v = CATALOG.variant().default_values(ENV)
    v.update({"in_tomograms": "Tomograms/job006/tomograms.star", "ckpt_path": CKPT, "membrain_env": "",
              "gpu_ids": "0:1", "cuda_module": ""})
    v.update(kw)
    return v


def run_runner(project, job, vals, env=None, wait=True):
    js = CATALOG.variant().make_jobstar(vals, False, job, None)
    (project / job).mkdir(parents=True, exist_ok=True)
    js.write(project / job / "job.star")  # as RELION writes it
    install_runner(project)
    cmd = f"{js.options['fn_exe']} --o {job} --j 1 {js.options['other_args']}"
    e = dict(os.environ, **(env or {}))
    if not wait:
        return subprocess.Popen(cmd, shell=True, cwd=project, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
    return subprocess.run(cmd, shell=True, cwd=project, env=e, capture_output=True, text=True, timeout=120)


def calls(log, what):
    return [c.split() for c in (log.read_text().splitlines() if log.exists() else []) if c.split()[0] == what]


def summary(project, job):
    b = read_star(project / job / M.SUMMARY)["global"]
    return {r[b.col_index(C.TOMO_NAME)]: dict(zip(b.columns, r)) for r in b.rows}


# ---------------------------------------------------------------------------- the form


def test_form_round_trip():
    v = CATALOG.variant()
    assert v.label == "relion.external" and v.is_tomo and [t.label for t in v.tabs] == [
        "Input", "Segmentation", "Skeletons", "Compute", "Running"]
    for t in v.tabs:
        for r in t.rows:
            assert all(k in v.options for k in r.keys)
            assert r.group is None or r.group in v.groups
    js = v.new_jobstar(ENV)
    assert membrain_kind_of(js) == "segment" and js.options["fn_exe"] == runner_command()
    assert v.values_from_jobstar(js, ENV)[0] == v.default_values(ENV)
    vals = values(tomo_names="TS_01 TS_03", segment_args="--x 'a b'", do_queue="Yes", nr_threads="4")
    js = v.make_jobstar(vals, False, "External/job020/", None)
    assert js.options["nr_threads"] == "4" and '"' not in js.options["other_args"]
    back, ignored = v.values_from_jobstar(js, ENV)
    assert back == vals and not ignored
    assert membrain_main_input(js) == "Tomograms/job006/tomograms.star"


def test_defaults_from_the_environment(monkeypatch):
    monkeypatch.setenv("CRYO_SAVANNA_MEMBRAIN_ENV", "my_membrain")
    monkeypatch.setenv("CRYO_SAVANNA_MEMBRAIN_CKPT", "/models/best.ckpt")
    v = MembrainCatalog(JobDefs.load(), ENV).variant()
    assert v.options["membrain_env"].default == "my_membrain" and v.options["ckpt_path"].default == "/models/best.ckpt"
    monkeypatch.delenv("CRYO_SAVANNA_MEMBRAIN_ENV")
    assert MembrainCatalog(JobDefs.load(), ENV).variant().options["membrain_env"].default == "membrain"


def test_groups_switch_rows():
    v = CATALOG.variant()
    vals = v.default_values(ENV)
    group_of = {r.keys[0]: r.group for t in v.tabs for r in t.rows}
    assert not v.group_active(group_of["skel_method"], vals) and not v.group_active(group_of["connected_component_thres"], vals)
    vals.update(do_skeleton="Yes", store_connected_components="Yes")
    assert v.group_active(group_of["skel_method"], vals) and v.group_active(group_of["connected_component_thres"], vals)
    assert v.group_active(group_of["store_uncertainty"], vals)
    vals["test_time_augmentation"] = "No"
    assert not v.group_active(group_of["store_uncertainty"], vals)


def test_names_and_family():
    assert membrain_kind("python3 .cryo-savanna/membrain_runner.py segment") == "segment"
    assert membrain_kind("python3 .cryo-savanna/membrain_runner.py other") is None
    assert membrain_kind("python3 .cryo-savanna/pytom_runner.py match") is None
    assert external_program(runner_command()) == "MemBrain-seg membrane segmentation"
    js = CATALOG.variant().new_jobstar(ENV)
    assert X.family_of(js) == X.MEMBRAIN


def test_inputs_go_in_external_fields(tomo_project):
    pipeline = Pipeline.read(tomo_project)
    js = CATALOG.variant().make_jobstar(values(), False, "External/job015/", pipeline)
    o = js.options
    assert "Tomograms/job006/tomograms.star" in (o["in_mov"], o["in_mic"], o["in_part"], o["in_coords"])


def test_gpu_queue_knows_its_gpus():
    from cryo_savanna.project.gpu_queue import job_gpu_ids
    js = CATALOG.variant().make_jobstar(values(gpu_ids="1:2"), False, "External/job015/", None)
    assert job_gpu_ids(js) == [1, 2]


def test_last_settings(project):
    p = Pipeline(job_counter=17)
    p.processes["External/job015/"] = Process("External/job015/", "", "relion.external", "Succeeded")
    (project / "External/job015").mkdir(parents=True)
    CATALOG.variant().make_jobstar(values(membrain_env="mb", ckpt_path="/m/x.ckpt"), False, "External/job015/",
                                   None).write(project / "External/job015/job.star")
    assert last_settings(project, p) == {"cuda_module": "", "membrain_env": "mb", "conda": "", "ckpt_path": "/m/x.ckpt"}


# ---------------------------------------------------------------------------- the commands


def test_segment_command():
    opts = values(rescale_patches="Yes", test_time_augmentation="No", store_probabilities="Yes",
                  store_connected_components="Yes", connected_component_thres="500", segmentation_threshold="0.5",
                  segment_args="--foo 1")
    cmd = M.segment_command(opts, "t.mrc", "m.ckpt", 13.5, "out")
    assert cmd[:2] == ["membrain", "segment"]
    for flag, value in (("--tomogram-path", "t.mrc"), ("--ckpt-path", "m.ckpt"), ("--out-folder", "out"),
                        ("--in-pixel-size", "13.5"), ("--out-pixel-size", "10"), ("--segmentation-threshold", "0.5"),
                        ("--sliding-window-size", "160"), ("--connected-component-thres", "500"), ("--foo", "1")):
        assert cmd[cmd.index(flag) + 1] == value, flag
    for flag in ("--rescale-patches", "--no-test-time-augmentation", "--store-probabilities",
                 "--store-connected-components"):
        assert flag in cmd
    assert "--store-uncertainty-map" not in cmd
    cmd = M.segment_command(values(), "t.mrc", "m.ckpt", 10, "out")
    assert "--no-test-time-augmentation" not in cmd and "--connected-component-thres" not in cmd
    assert M.conda_prefix({"membrain_env": "membrain", "conda": "mamba"}) == [
        "mamba", "run", "--no-capture-output", "-n", "membrain"]


@pytest.mark.parametrize("kw, error", [({"sliding_window_size": "150"}, "multiple of 32"),
                                       ({"store_uncertainty": "Yes", "test_time_augmentation": "No"},
                                        "test-time augmentation"),
                                       ({"segmentation_threshold": "x"}, "Segmentation threshold")])
def test_segment_command_errors(kw, error):
    with pytest.raises(M.Usage, match=error):
        M.segment_command(values(**kw), "t.mrc", "m.ckpt", 10, "out")


def test_skeleton_command():
    cmd = M.skeleton_command(values(skel_method="2D-skimage", skel_batch_size="1000000"), "s.mrc", "out")
    assert cmd == ["membrain", "skeletonize", "--label-path", "s.mrc", "--out-folder", "out", "--method",
                   "2D-skimage", "--batch-size", "1000000"]
    with pytest.raises(M.Usage, match="method"):
        M.skeleton_command(values(skel_method="other"), "s.mrc", "out")


def test_ckpt_token():
    assert M.ckpt_token("/a/MemBrain_seg_v10_alpha.ckpt") == "MemBrain_seg_v10_alpha.ckpt"
    assert M.ckpt_token("/a/epoch=999-val_loss=0.5.ckpt") == "epoch=999"


# ---------------------------------------------------------------------------- the runner


def test_segment_and_skeletonize(project, fake_membrain):
    job = "External/job015/"
    res = run_runner(project, job, values(store_probabilities="Yes", store_uncertainty="Yes", do_skeleton="Yes"))
    assert res.returncode == 0, res.stderr
    d = project / job
    assert (d / C.SUCCESS).exists()
    segs = calls(fake_membrain, "segment")
    assert len(segs) == 5 and len(calls(fake_membrain, "skeletonize")) == 5
    paths = sorted(c[c.index("--tomogram-path") + 1] for c in segs)
    assert paths == [job + f"inputs/{n}.mrc" for n in NAMES]  # the links, named by rlnTomoName
    first = segs[0]
    assert first[first.index("--in-pixel-size") + 1] == "6.75"  # from the header: no tilt-series pixel size
    for name in NAMES:
        for suffix in (M.SEGMENTED, M.SCORES, M.UNCERTAINTY, M.SKELETON):
            assert (d / M.SEG_DIR / (name + suffix)).is_file(), name + suffix
        with mrcfile.open(d / M.SEG_DIR / (name + M.SEGMENTED)) as m:
            assert m.data.shape == (16, 64, 64)
    # nothing left under membrain-seg's own names
    assert not list((d / M.SEG_DIR).glob("*ckpt*")) and not list((d / M.SEG_DIR).glob("*_skel.mrc"))
    rows = summary(project, job)
    assert list(rows) == NAMES and all(r[M.COL_STATUS] == "done" for r in rows.values())
    assert rows["TS_01"][M.COL_SEGMENTATION] == job + "segmentations/TS_01_segmented.mrc"
    assert rows["TS_01"][M.COL_SKELETON] == job + "segmentations/TS_01_skeleton.mrc"
    tomos = read_star(d / M.TOMOGRAMS)["global"]
    assert M.COL_STATUS not in tomos.columns and len(tomos.rows) == 5
    assert "rlnTomoReconstructedTomogram" in tomos.columns and M.COL_SEGMENTATION in tomos.columns
    nodes = read_star(d / C.OUTPUT_NODES)["output_nodes"]
    assert [r[0] for r in nodes.rows] == [job + M.TOMOGRAMS, job + M.SUMMARY]
    # one tomogram per GPU, each program shown only its GPU
    gpus = [line.split() for line in (fake_membrain.parent / "fake.log.gpus").read_text().splitlines()]
    assert {g for _, g in gpus} == {"0", "1"}


def test_optional_columns_only_when_asked(project, fake_membrain):
    job = "External/job015/"
    res = run_runner(project, job, values(tomo_names="TS_01"))
    assert res.returncode == 0, res.stderr
    cols = read_star(project / job / M.SUMMARY)["global"].columns
    assert M.COL_SEGMENTATION in cols and M.COL_SCORES not in cols and M.COL_SKELETON not in cols
    assert list(summary(project, job)) == ["TS_01"]


def test_continue_skips_done_and_only_skeletonizes(project, fake_membrain):
    job = "External/job015/"
    assert run_runner(project, job, values(tomo_names="TS_01 TS_03")).returncode == 0
    fake_membrain.unlink()
    res = run_runner(project, job, values(tomo_names="TS_01 TS_03"))
    assert res.returncode == 0 and not fake_membrain.exists()  # all done: nothing runs
    res = run_runner(project, job, values(tomo_names="TS_01 TS_03", do_skeleton="Yes"))
    assert res.returncode == 0, res.stderr
    assert calls(fake_membrain, "segment") == [] and len(calls(fake_membrain, "skeletonize")) == 2
    res = run_runner(project, job, values(tomo_names="TS_01", skip_done="No"))
    assert res.returncode == 0 and len(calls(fake_membrain, "segment")) == 1


def test_failed_tomogram_and_continue(project, fake_membrain):
    job = "External/job015/"
    res = run_runner(project, job, values(), env={"FAKE_FAIL": "TS_43"})
    assert res.returncode == 1 and "1 tomogram(s) failed: TS_43" in res.stderr
    assert "CUDA out of memory" in res.stderr  # the end of its log
    assert (project / job / C.FAILURE).exists() and not (project / job / C.OUTPUT_NODES).exists()
    rows = summary(project, job)
    assert rows["TS_43"][M.COL_STATUS] == "failed" and rows["TS_01"][M.COL_STATUS] == "done"
    fake_membrain.unlink()
    res = run_runner(project, job, values())
    assert res.returncode == 0, res.stderr
    assert [c[c.index("--tomogram-path") + 1] for c in calls(fake_membrain, "segment")] == [job + "inputs/TS_43.mrc"]


def test_inputs_are_checked_before_anything_runs(project, fake_membrain):
    res = run_runner(project, "External/job015/", values(ckpt_path="missing.ckpt"))
    assert res.returncode == 1 and "model checkpoint not found" in res.stderr
    res = run_runner(project, "External/job015/", values(ckpt_path=""))
    assert res.returncode == 1 and "give the model checkpoint" in res.stderr
    res = run_runner(project, "External/job015/", values(sliding_window_size="100"))
    assert res.returncode == 1 and "multiple of 32" in res.stderr
    assert not fake_membrain.exists()


def test_abort(project, fake_membrain):
    job = "External/job015/"
    proc = run_runner(project, job, values(gpu_ids="0"), env={"FAKE_SLEEP": "60"}, wait=False)
    d = project / job
    end = time.time() + 30
    while time.time() < end and not (d / "logs/TS_01.log").exists():
        time.sleep(0.2)
    time.sleep(0.3)
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=40)
    assert (d / C.ABORTED).exists()
    assert summary(project, job)["TS_01"][M.COL_STATUS] == "aborted"


# ---------------------------------------------------------------------------- views


def _ctx(project, job):
    js = JobStar.read(project / job / "job.star")
    return R.JobContext(project, job, "relion.external", "Succeeded",
                        [(job + M.TOMOGRAMS, M.NODE_TOMOGRAMS), (job + M.SUMMARY, M.NODE_SUMMARY)], js.options)


def test_views_and_tomogram_viewer(qtbot, project, fake_membrain):
    from cryo_savanna.viewers.tomo_viewer import TomogramViewer
    job = "External/job015/"
    vals = values(do_skeleton="Yes")
    assert run_runner(project, job, vals).returncode == 0
    CATALOG.variant().make_jobstar(vals, False, job, None).write(project / job / "job.star")
    views = R.views_for(_ctx(project, job))
    titles = [v.title for v in views]
    assert titles[0] == "Membranes" and views[0].args["star"] == job + M.SUMMARY
    assert titles.count("Tomograms") == 0  # its tomograms.star: the same, no tab of its own
    v = TomogramViewer(project, job + M.SUMMARY)
    qtbot.addWidget(v)
    assert v.tasks.wait()
    kinds = [v.kind.itemData(i) for i in range(v.kind.count())]
    assert kinds == ["rlnTomoReconstructedTomogram", M.COL_SEGMENTATION, M.COL_SKELETON]
    assert v.compare.isVisibleTo(v) and v.compare.text() == "Membranes beside it"
    v.kind.setCurrentIndex(1)
    assert v.compare.text() == "Raw beside it"
    v.compare.setChecked(True)
    assert v.tasks.wait()
    assert v.raw.isVisibleTo(v)


def test_own_args_are_the_forms():
    js = CATALOG.variant().make_jobstar(values(), False, "External/job015/", None)
    own, other = decode(js.options["other_args"])
    assert not other and set(own) == set(CATALOG.variant().own_keys)


def test_new_membrain_job_in_window(qtbot, project, tmp_path):
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(project, poll_s=1000.0, settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    titles = [a.text() for a in w.actions.external_menu.actions()]
    assert "MemBrain-seg: segment membranes" in titles
    w.actions.new_membrain_job()
    ed = w.actions.editor
    assert ed.variant.kind == "segment" and "MemBrain-seg" in ed.title.text()
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())] == \
        ["Input", "Segmentation", "Skeletons", "Compute", "Running"]
    assert ed.values()["in_tomograms"].endswith("tomograms.star")
    assert not ed.widgets["skel_method"].isEnabled()
    ed.widgets["do_skeleton"].set_value("Yes")
    ed._on_changed("do_skeleton")
    assert ed.widgets["skel_method"].isEnabled()
    js = ed.jobstar()
    assert membrain_kind_of(js) == "segment" and "--do_skeleton Yes" in js.options["other_args"]


# ---------------------------------------------------------------------------- thresholds and score maps


def _scores_of(project, job, name):
    with mrcfile.open(project / job / M.SEG_DIR / (name + M.SCORES)) as m:
        return np.asarray(m.data)


def _seg_of(project, job, name):
    with mrcfile.open(project / job / M.SEG_DIR / (name + M.SEGMENTED)) as m:
        return np.asarray(m.data)


def test_tomo_thresholds_option():
    assert M.tomo_thresholds({"tomo_thresholds": "TS_01=-0.5 TS_03=1"}) == {"TS_01": -0.5, "TS_03": 1.0}
    assert M.threshold_of({"tomo_thresholds": "TS_01=-0.5", "segmentation_threshold": "0.2"}, "TS_03") == 0.2
    assert M.threshold_of({}, "TS_01") == 0.0
    with pytest.raises(M.Usage, match="TS_01:x"):
        M.tomo_thresholds({"tomo_thresholds": "TS_01:x"})
    assert M.threshold_name(-1.5) == "-1.5" and M.threshold_name(0) == "0.0"
    assert M.thresholds_command({}, "s.mrc", -0.5, "out") == [
        "membrain", "thresholds", "--scoremap-path", "s.mrc", "--thresholds", "-0.5", "--out-folder", "out"]


def test_continue_thresholds_again_from_the_score_maps(project, fake_membrain):
    job = "External/job015/"
    vals = values(tomo_names="TS_01 TS_03", store_probabilities="Yes", store_connected_components="Yes",
                  do_skeleton="Yes")
    assert run_runner(project, job, vals).returncode == 0
    rows = summary(project, job)
    assert rows["TS_01"][M.COL_THRESHOLD] == "0"
    st = read_thresholds(project, job)
    assert st.used == {"TS_01": 0.0, "TS_03": 0.0} and set(st.scores) == {"TS_01", "TS_03"} and st.to_redo(NAMES) == []
    t = float(np.median(_scores_of(project, job, "TS_01")))
    save_thresholds(project, job, {"TS_01": t, "TS_03": 0.0})  # TS_03: the job's own, left out
    js = JobStar.read(project / job / "job.star")
    own = decode(js.options["other_args"])[0]
    assert own["tomo_thresholds"] == f"TS_01={M.num(t)}"
    st = read_thresholds(project, job)
    assert st.to_redo(NAMES) == ["TS_01"]
    fake_membrain.unlink()
    res = run_runner(project, job, {**vals, "tomo_thresholds": own["tomo_thresholds"]})
    assert res.returncode == 0, res.stderr
    assert calls(fake_membrain, "segment") == []  # no GPU segmentation
    th = calls(fake_membrain, "thresholds")
    assert len(th) == 1 and th[0][th[0].index("--scoremap-path") + 1] == job + "segmentations/TS_01_scores.mrc"
    assert len(calls(fake_membrain, "components")) == 1 and len(calls(fake_membrain, "skeletonize")) == 1
    seg, scores = _seg_of(project, job, "TS_01"), _scores_of(project, job, "TS_01")
    assert np.array_equal(seg > 0, scores > float(M.num(t)))  # at the new threshold (as components: labels 7)
    assert set(np.unique(seg)) <= {0, 7}
    assert summary(project, job)["TS_01"][M.COL_THRESHOLD] == M.num(t)
    assert not list((project / job / M.SEG_DIR).glob("*_threshold_*")) and \
        not list((project / job / M.SEG_DIR).glob("*_components.mrc"))
    assert read_thresholds(project, job).to_redo(NAMES) == []


def test_a_new_threshold_without_its_score_map_segments_again(project, fake_membrain):
    job = "External/job015/"
    vals = values(tomo_names="TS_01", store_probabilities="Yes")
    assert run_runner(project, job, vals).returncode == 0
    freed = delete_score_maps(project, job)
    assert freed > 0 and not (project / job / M.SEG_DIR / ("TS_01" + M.SCORES)).exists()
    own = decode(JobStar.read(project / job / "job.star").options["other_args"])[0]
    assert own["store_probabilities"] == "No"
    for f in (M.SUMMARY, M.TOMOGRAMS):
        assert M.COL_SCORES not in read_star(project / job / f)["global"].columns
    assert M.COL_THRESHOLD in read_star(project / job / M.SUMMARY)["global"].columns
    fake_membrain.unlink()
    res = run_runner(project, job, {**vals, "store_probabilities": "No"})  # as the job.star says now
    assert res.returncode == 0 and not fake_membrain.exists()  # nothing to do: no re-segmentation for the maps
    res = run_runner(project, job, {**vals, "store_probabilities": "No", "tomo_thresholds": "TS_01=0.25"})
    assert res.returncode == 0, res.stderr
    seg = calls(fake_membrain, "segment")
    assert len(seg) == 1 and seg[0][seg[0].index("--segmentation-threshold") + 1] == "0.25"
    assert calls(fake_membrain, "thresholds") == []


def test_a_summary_of_an_earlier_cryo_savanna_has_no_thresholds(project, fake_membrain):
    job = "External/job015/"
    assert run_runner(project, job, values(tomo_names="TS_01", segmentation_threshold="0.1")).returncode == 0
    path = project / job / M.SUMMARY
    blocks = read_star(path)
    g = blocks["global"]
    i = g.col_index(M.COL_THRESHOLD)
    g.columns.pop(i)
    for r in g.rows:
        r.pop(i)
    from cryo_savanna.project.star import write_star
    write_star(path, blocks.values(), atomic=False)
    assert M.previous_thresholds(str(project / job), 0.1) == {"TS_01": 0.1}  # made with the job's threshold


def test_scores_tab_only_with_score_maps(project, fake_membrain):
    job = "External/job015/"
    vals = values(tomo_names="TS_01", store_probabilities="Yes")
    assert run_runner(project, job, vals).returncode == 0
    CATALOG.variant().make_jobstar(vals, False, job, None).write(project / job / "job.star")
    titles = [v.title for v in R.views_for(_ctx(project, job))]
    assert titles[:2] == ["Membranes", "Scores"]
    delete_score_maps(project, job)
    assert "Scores" not in [v.title for v in R.views_for(_ctx(project, job))]


def test_scores_view(qtbot, project, fake_membrain):
    from cryo_savanna.viewers.membrain_view import ABOVE, HEATMAP, NO_OVERLAY, MembrainScoresView
    job = "External/job015/"
    vals = values(tomo_names="TS_01 TS_03", store_probabilities="Yes")
    assert run_runner(project, job, vals).returncode == 0
    CATALOG.variant().make_jobstar(vals, False, job, None).write(project / job / "job.star")
    v = MembrainScoresView(project, job)
    qtbot.addWidget(v)
    qtbot.waitUntil(lambda: v.score is not None and v._planes is not None, timeout=10000)
    assert v.table.rowCount() == 2 and v.current == "TS_01" and v.table.item(0, 1).text() == "done"
    assert not v.used_line.isVisible() and v.line.value() == 0.0  # segmented at the threshold shown
    assert v.overlay_item.isVisible()
    above = v.overlay_rgba()
    assert above is not None and above.shape == (64, 64, 4)
    n0 = int((above[..., 3] > 0).sum())
    v.set_threshold(float(np.percentile(v._planes[4], 90)))
    n1 = int((v.overlay_rgba()[..., 3] > 0).sum())
    assert 0 < n1 < n0
    assert "→" in v.table.item(0, 2).text() and v.table.item(1, 2).text() == "0"  # only the tomogram shown
    assert v.used_line.isVisible() and v.used_line.value() == 0.0
    assert "TS_01" in v.redo_label.text() and v.save_btn.isEnabled()
    v.overlay.setCurrentText(HEATMAP)
    heat = v.overlay_rgba()
    assert heat is not None and heat[..., 3].max() > 0
    v.overlay.setCurrentText(NO_OVERLAY)
    assert v.overlay_rgba() is None and not v.overlay_item.isVisible()
    v.overlay.setCurrentText(ABOVE)
    v.same.setChecked(True)  # every tomogram gets the one shown
    assert v.wanted["TS_03"] == v.wanted["TS_01"]
    # a threshold not applied yet: the score maps are kept
    assert not v.delete_scores(ask=False)
    assert v.save()
    own = decode(JobStar.read(project / job / "job.star").options["other_args"])[0]
    assert set(M.tomo_thresholds(own)) == {"TS_01", "TS_03"}
    assert not v.dirty and not v.save_btn.isEnabled()
    # continued: thresholded again, then the maps can go
    res = run_runner(project, job, {**vals, "tomo_thresholds": own["tomo_thresholds"]})
    assert res.returncode == 0, res.stderr
    v.load()
    qtbot.waitUntil(lambda: v.settings is not None and not v.settings.to_redo(v.names), timeout=10000)
    assert v.delete_scores(ask=False)
    assert not list((project / job / M.SEG_DIR).glob("*" + M.SCORES))
    qtbot.waitUntil(lambda: v.settings is not None and not v.settings.scores, timeout=10000)
    assert not v.delete_btn.isEnabled() and "no score maps" in v.info.text()
