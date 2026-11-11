# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""pytom jobs: the forms, the runner (with a fake pytom on the PATH), the views, the threshold viewer."""

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryo_savanna import pytom_runner as P
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.cli_tools import external_program
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import (
    PytomCatalog,
    decode,
    encode,
    install_runner,
    last_settings,
    pytom_kind,
    pytom_kind_of,
    quoted_values,
    runner_command,
)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.star import read_star, write_star
from cryo_savanna.results import pytom as RP
from cryo_savanna.results import registry as R

ENV = RelionEnv({})
CATALOG = PytomCatalog(JobDefs.load(), ENV)
NAMES = ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"]

# ---------------------------------------------------------------------------- a fake pytom

FAKE = {
    "pytom_match_template.py": """
        import json, os, sys, time, numpy as np, mrcfile
        a = sys.argv[1:]
        def arg(f, n=1):
            i = a.index(f)
            return a[i + 1] if n == 1 else a[i + 1:i + 1 + n]
        open(os.environ["FAKE_LOG"], "a").write(" ".join(["match"] + a) + "\\n")
        if os.environ.get("CUDA_HOME"):
            open(os.environ["FAKE_LOG"] + ".cuda", "a").write(os.environ["CUDA_HOME"] + "\\n")
        if os.environ.get("FAKE_SLEEP"):
            time.sleep(float(os.environ["FAKE_SLEEP"]))
        if os.environ.get("FAKE_FAIL") and os.environ["FAKE_FAIL"] in arg("-v"):
            sys.exit(3)
        tomo, dest = arg("-v"), arg("-d")
        stem = os.path.splitext(os.path.basename(tomo))[0]
        with mrcfile.open(tomo, permissive=True) as m:
            shape = m.data.shape
        rng = np.random.default_rng(len(stem))
        scores = rng.normal(0, 0.02, shape).astype(np.float32)
        scores[8, 32, 32] = 0.5
        for name, data in (("_scores.mrc", scores), ("_angles.mrc", np.zeros(shape, np.float32))):
            with mrcfile.new(os.path.join(dest, stem + name), overwrite=True) as m:
                m.set_data(data)
                m.voxel_size = 6.75
        json.dump({"output_dir": os.path.abspath(dest), "tomo_id": stem, "tomogram": os.path.abspath(tomo),
                   "particle_diameter": 120.0, "n_rotations": 1000, "voxel_size": 6.75,
                   "job_stats": {"search_space": 6.5e7, "std": 0.02, "variance": 0.0004}},
                  open(os.path.join(dest, stem + "_job.json"), "w"))
    """,
    "pytom_extract_candidates.py": """
        import json, os, sys
        a = sys.argv[1:]
        open(os.environ["FAKE_LOG"], "a").write(" ".join(["extract"] + a) + "\\n")
        job = json.load(open(a[a.index("-j") + 1]))
        n = int(a[a.index("-n") + 1])
        cut = float(a[a.index("--cut-off") + 1]) if "--cut-off" in a else 0.3
        out = job["output_dir"]
        assert os.path.exists(os.path.join(out, job["tomo_id"] + "_scores.mrc"))  # found where the json says
        rows = []
        for i in range(n):
            lcc = 0.5 - 0.02 * i
            if lcc >= cut:
                rows.append("%d %d %d 10 20 30 %.4f %.4f 0.02 6.75 %s" % (32 + i % 8, 32, 8, lcc, cut, job["tomo_id"]))
        cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt",
                "rlnAnglePsi", "rlnLCCmax", "rlnCutOff", "rlnSearchStd", "rlnDetectorPixelSize", "rlnMicrographName"]
        with open(os.path.join(out, job["tomo_id"] + "_particles.star"), "w") as fh:
            fh.write("data_particles\\n\\nloop_\\n" + "".join("_%s #%d\\n" % (c, k + 1) for k, c in enumerate(cols)))
            fh.write("\\n".join(rows) + "\\n")
        open(os.path.join(out, job["tomo_id"] + "_extraction_graph.svg"), "w").write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>')
    """,
    "pytom_create_template.py": """
        import os, sys, numpy as np, mrcfile
        a = sys.argv[1:]
        open(os.environ["FAKE_LOG"], "a").write(" ".join(["template"] + a) + "\\n")
        box = int(a[a.index("-b") + 1]) if "-b" in a else 24
        with mrcfile.new(a[a.index("-o") + 1], overwrite=True) as m:
            m.set_data(np.ones((box, box, box), np.float32) * (-1 if "--invert" in a else 1))
            m.voxel_size = float(a[a.index("--output-voxel-size-angstrom") + 1])
    """,
    "pytom_create_mask.py": """
        import os, sys, numpy as np, mrcfile
        a = sys.argv[1:]
        open(os.environ["FAKE_LOG"], "a").write(" ".join(["mask"] + a) + "\\n")
        box = int(a[a.index("-b") + 1])
        with mrcfile.new(a[a.index("-o") + 1], overwrite=True) as m:
            m.set_data(np.ones((box, box, box), np.float32))
            m.voxel_size = float(a[a.index("--voxel-size") + 1])
    """,
}


@pytest.fixture
def fake_pytom(tmp_path, monkeypatch):
    return _fake_pytom(tmp_path, monkeypatch)


def _fake_pytom(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    for name, code in FAKE.items():
        f = bin_dir / name
        f.write_text(f"#!{sys.executable}\n" + textwrap.dedent(code))
        f.chmod(0o755)
    (bin_dir / "python").symlink_to(sys.executable)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    return log


@pytest.fixture
def project(tomo_project, tmp_path):
    """A copy of the tomo project whose tomograms have CTF values (the per-tilt files of CtfFind)."""
    return _project(tomo_project, tmp_path)


def _project(tomo_project, tmp_path):
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    star = dst / "Tomograms/job006/tomograms.star"
    blocks = read_star(star)
    g = blocks["global"]
    i = g.col_index("rlnTomoTiltSeriesStarFile")
    for r in g.rows:
        r[i] = f"CtfFind/job003/tilt_series/{r[g.col_index('rlnTomoName')]}.star"
    g.columns.append("rlnTomoTiltSeriesPixelSize")
    for r in g.rows:
        r.append("0.675")
    write_star(star, blocks.values(), atomic=False)
    with mrcfile.new(dst / "ref.mrc") as m:
        m.set_data(np.zeros((40, 40, 40), np.float32))
        m.voxel_size = 1.35
    return dst


def run_runner(project, mode, job, values, threads=1, env=None, wait=True):
    v = CATALOG.variant(mode)
    js = v.make_jobstar(values, False, job, None)
    (project / job).mkdir(parents=True, exist_ok=True)
    install_runner(project)
    cmd = f"{js.options['fn_exe']} --o {job} --j {threads} {js.options['other_args']}"
    e = dict(os.environ, **(env or {}))
    if not wait:
        return subprocess.Popen(cmd, shell=True, cwd=project, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
    return subprocess.run(cmd, shell=True, cwd=project, env=e, capture_output=True, text=True, timeout=120)


def match_values(**kw):
    v = CATALOG.variant("match").default_values(ENV)
    v.update({"in_tomograms": "Tomograms/job006/tomograms.star", "fn_template": "ref.mrc", "pytom_env": "",
              "particle_diameter": "120", "gpu_ids": "0:1", "cuda_module": ""})
    v.update(kw)
    return v


def extract_values(**kw):
    v = CATALOG.variant("extract").default_values(ENV)
    v.update({"in_match": "External/job015/pytom_match.star", "pytom_env": "", "n_particles": "20", "cuda_module": ""})
    v.update(kw)
    return v


# ---------------------------------------------------------------------------- forms


def test_forms_round_trip():
    for kind in ("match", "extract"):
        v = CATALOG.variant(kind)
        assert v.label == "relion.external" and v.is_tomo and v.tabs[-1].label == "Running"
        for t in v.tabs:
            for r in t.rows:
                assert all(k in v.options for k in r.keys)
                assert r.group is None or r.group in v.groups
        js = v.new_jobstar(ENV)
        assert pytom_kind_of(js) == kind and js.options["fn_exe"] == runner_command(kind)
        assert v.values_from_jobstar(js, ENV)[0] == v.default_values(ENV)
    v = CATALOG.variant("match")
    vals = match_values(tomo_names="TS_01 TS_03", match_args="--x 'a b'", do_queue="Yes", nr_threads="4")
    js = v.make_jobstar(vals, False, "External/job020/", None)
    assert js.options["nr_threads"] == "4" and js.options["do_queue"] == "Yes" and '"' not in js.options["other_args"]
    back, ignored = v.values_from_jobstar(js, ENV)
    assert back == vals and not ignored
    assert quoted_values({"a": 'x"y', "b": "z"}, ["a", "b"]) == ["a"]
    assert decode(encode({"a": "it's", "b": ""}, ["a", "b"]))[0] == {"a": "it's", "b": ""}


def test_groups_switch_rows():
    v = CATALOG.variant("match")
    vals = v.default_values(ENV)
    group_of = {r.keys[0]: r.group for t in v.tabs for r in t.rows}
    assert v.group_active(group_of["template_box"], vals)
    vals["do_prep_template"] = "No"
    assert not v.group_active(group_of["template_box"], vals)
    assert v.group_active(group_of["mask_radius"], vals) and not v.group_active(group_of["fn_mask"], vals)
    vals["mask_source"] = P.MASK_FILE
    assert v.group_active(group_of["fn_mask"], vals) and not v.group_active(group_of["mask_box"], vals)
    vals["do_prep_mask"] = "Yes"
    assert v.group_active(group_of["mask_box"], vals)


def test_inputs_go_in_external_fields(tomo_project):
    pipeline = Pipeline.read(tomo_project)
    v = CATALOG.variant("match")
    js = v.make_jobstar(match_values(fn_template="Reconstruct/job010/merged.mrc"), False, "External/job015/", pipeline)
    o = js.options
    assert o["in_3dref"] == "Reconstruct/job010/merged.mrc"
    assert "Tomograms/job006/tomograms.star" in (o["in_mov"], o["in_mic"], o["in_part"], o["in_coords"])


def test_tree_follows_the_main_input(tomo_project, tmp_path):
    """A pytom job sits under the job of its form's main input: the tomograms (not the template, whatever
    order the pipeline lists them in, or when only the template was linked), and the matching job of an
    extraction added while the matching ran (its pytom_match.star was no pipeline node: nothing linked)."""
    from cryo_savanna.project.lineage import primary_parents
    from cryo_savanna.project.pipeline import Node, Process
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    p = Pipeline.read(dst)
    tomos, ref = "Tomograms/job006/tomograms.star", "Reconstruct/job010/merged.mrc"

    def add(name, kind, values, inputs):
        js = CATALOG.variant(kind).make_jobstar(values, False, name, p)
        (dst / name).mkdir(parents=True)
        js.write(dst / name / "job.star")
        p.processes[name] = Process(name, "None", "relion.external", "Running", inputs=list(inputs))
        for n in inputs:
            p.nodes[n].consumers.append(name)

    add("External/job015/", "match", match_values(fn_template=ref), [ref, tomos])  # pipeline order: the template first
    add("External/job016/", "match", match_values(fn_template=ref), [ref])  # the tomograms not linked
    add("External/job017/", "extract", extract_values(), [])  # the matching job had no output nodes yet
    add("External/job018/", "extract", extract_values(in_match="./External/job015/pytom_match.star"), [])
    parents = primary_parents(p, dst, JobDefs.load())
    assert parents["External/job015/"] == parents["External/job016/"] == "Tomograms/job006/"
    assert parents["External/job017/"] == parents["External/job018/"] == "External/job015/"
    # once the matching job's output is a node and linked, the extraction stays there
    out = "External/job015/pytom_match.star"
    p.nodes[out] = Node(out, P.NODE_SUMMARY, producer="External/job015/", consumers=["External/job017/"])
    p.processes["External/job017/"].inputs.append(out)
    assert primary_parents(p, dst, JobDefs.load())["External/job017/"] == "External/job015/"
    # a main input chosen in cryo-savanna still wins
    chosen = primary_parents(p, dst, JobDefs.load(), main_inputs={"External/job015/": ref})
    assert chosen["External/job015/"] == "Reconstruct/job010/"


def test_names():
    assert pytom_kind("python3 .cryo-savanna/pytom_runner.py match") == "match"
    assert pytom_kind("python3 .cryo-savanna/cli_runner.py relion_image_handler") is None
    assert external_program(runner_command("extract")) == "pytom candidate extraction"


# ---------------------------------------------------------------------------- the runner


def test_runner_parse():
    mode, job, threads, opts = P.parse(["match", "--o", "External/job001", "--in_movies", "x.star", "--j", "3",
                                        "--gpu_ids", "0:1", "--tomo_names", "", "--pipeline_control", "External/job001/"])
    assert (mode, job, threads) == ("match", "External/job001/", 3)
    assert opts == {"gpu_ids": "0:1", "tomo_names": ""}
    with pytest.raises(P.Usage):
        P.parse(["other", "--o", "x", "--j", "1"])


@pytest.mark.parametrize("value", ["40 160", "40,160", "40:160", "40-160", " 40  160 ", "40.0 160"])
def test_search_range_forms(value):
    assert P._pair({"search_z": value}, "search_z", "Search range z") == ["40", "160"]


@pytest.mark.parametrize("value", ["160", "0 100 200", "a b", "200 100"])
def test_search_range_errors_name_the_field(value):
    with pytest.raises(P.Usage, match="Search range z") as e:
        P._pair({"search_z": value}, "search_z", "Search range z")
    assert repr(value) in str(e.value)
    assert P._pair({"search_z": ""}, "search_z", "Search range z") == []


@pytest.mark.parametrize("value,ids", [("0:1:2:3", [0, 1, 2, 3]), ("2", [2]), ("0,1", [0, 1]), ("0 1", [0, 1])])
def test_gpu_ids_colons(value, ids):
    assert P.gpu_ids({"gpu_ids": value}) == ids
    with pytest.raises(P.Usage, match="colons"):
        P.gpu_ids({"gpu_ids": "0;x"})


def test_match_needs_ctf_values(project, fake_pytom):
    star = project / "Tomograms/job006/tomograms.star"
    blocks = read_star(star)
    g = blocks["global"]
    i = g.col_index("rlnTomoTiltSeriesStarFile")
    for r in g.rows:
        r[i] = r[i].replace("CtfFind/job003", "Tomograms/job006")  # the reconstruction's tilt files: no defocus
    write_star(star, blocks.values(), atomic=False)
    res = run_runner(project, "match", "External/job015/", match_values())
    assert res.returncode == 1 and "no rlnDefocusU" in res.stderr
    assert (project / "External/job015" / P.FAILURE).exists()
    assert not fake_pytom.exists()  # nothing ran: the inputs are checked first
    res = run_runner(project, "match", "External/job015/", match_values(use_ctf="No", tomo_names="TS_01"))
    assert res.returncode == 0, res.stderr
    assert "--defocus" not in fake_pytom.read_text()


def test_match_and_extract(project, fake_pytom):
    job = "External/job015/"
    res = run_runner(project, "match", job, match_values(phase_shift="0", search_z="2 14", volume_split="2 2 1"))
    assert res.returncode == 0, res.stderr + res.stdout
    d = project / job
    assert (d / P.SUCCESS).exists()
    # pytom's inputs, from the tilt-series STAR files
    tlt = [float(x) for x in (d / "inputs/TS_01/TS_01.tlt").read_text().split()]
    assert tlt == sorted(tlt) and len(tlt) == 41 and tlt[0] == -60
    assert len((d / "inputs/TS_01/TS_01_dose.txt").read_text().split()) == 41
    defocus = [float(x) for x in (d / "inputs/TS_01/TS_01_defocus.txt").read_text().split()]
    assert all(3 <= x <= 4.5 for x in defocus)  # µm
    assert os.path.islink(d / "inputs/TS_01/TS_01.mrc")
    calls = fake_pytom.read_text().splitlines()
    assert calls[0].startswith("template -i ref.mrc -o External/job015/template.mrc --output-voxel-size-angstrom 6.75")
    assert "--invert" in calls[0]
    assert calls[1].startswith("mask -b 24 -o External/job015/mask.mrc --voxel-size 6.75 -r 8.9")  # 120 Å / 2 / 6.75
    match = [c for c in calls if c.startswith("match")]
    assert len(match) == 5
    m = match[0].split()
    assert m[m.index("-g") + 1] in ("0", "1") and m[m.index("-g") + 2] == "-a"  # one GPU each
    for flag in ("--per-tilt-weighting", "--dose-accumulation", "--defocus", "--voltage", "--particle-diameter"):
        assert flag in m
    assert m[m.index("--voltage") + 1] == "300" and m[m.index("--spherical-aberration") + 1] == "2.7"
    assert m[m.index("-s") + 1:m.index("-s") + 4] == ["2", "2", "1"]
    assert m[m.index("--search-z") + 1:m.index("--search-z") + 3] == ["2", "14"]
    assert "--phase-shift" not in m and "--search-x" not in m
    nodes = read_star(d / P.OUTPUT_NODES)["output_nodes"].rows
    assert [r[0] for r in nodes] == [job + P.SUMMARY, job + "template.mrc", job + "mask.mrc"]
    summary = RP.read_match_summary(project, job + P.SUMMARY)
    assert summary.names == NAMES and set(summary.table.column(P.COL_STATUS)) == {"done"}
    assert summary.general[P.P + "TomogramsFile"] == "Tomograms/job006/tomograms.star"
    assert RP.job_stats(project / summary.value(0, P.COL_JOB))["std"] == 0.02

    # again (continue): every tomogram is done
    fake_pytom.unlink()
    res = run_runner(project, "match", job, match_values())
    assert res.returncode == 0 and "already done" in res.stdout
    assert not any(c.startswith("match") for c in fake_pytom.read_text().splitlines())

    # extraction, with candidates below the automatic cut-off (0.3 in the fake)
    ex = "External/job016/"
    res = run_runner(project, "extract", ex, extract_values(), threads=2)
    assert res.returncode == 0, res.stderr + res.stdout
    e = project / ex
    calls = [c.split() for c in fake_pytom.read_text().splitlines() if c.startswith("extract")]
    assert len(calls) == 10 and sum("--cut-off" in c for c in calls) == 5  # auto, then candidates
    low = next(c for c in calls if "--cut-off" in c)
    assert float(low[low.index("--cut-off") + 1]) == pytest.approx(0.21)
    assert not list((d / "tm").glob("*_particles.star"))  # nothing written in the match job
    cands = read_star(e / P.CANDIDATES)["particles"]
    assert cands.columns[:4] == ["rlnTomoName"] + list(P.CENTERED) and "rlnAngleRot" in cands.columns
    assert len(cands.rows) == 5 * 15  # 0.5 down to 0.22
    x = float(cands.rows[0][1])
    assert x == pytest.approx((32 * 10 - 4000 / 2) * 0.675)  # RELION 5: (voxel × binning − size / 2) × pixel
    parts = read_star(e / P.PARTICLES)["particles"]
    assert len(parts.rows) == 5 * 11  # 0.5 down to 0.3
    sel = RP.read_selection(project, ex)
    assert sel.names == NAMES and sel.auto["TS_01"] == pytest.approx(0.3) and sel.counts("TS_01") == (15, 11)
    opt = read_star(e / P.OPTIMISATION_SET)[""]
    assert opt.get("rlnTomoParticlesFile") == ex + P.PARTICLES
    assert opt.get("rlnTomoTomogramsFile") == "Tomograms/job006/tomograms.star"
    assert sorted(f.name for f in (e / "plots").iterdir())[:2] == ["TS_01_candidates_extraction.svg",
                                                                   "TS_01_extraction.svg"]
    nodes = dict(read_star(e / P.OUTPUT_NODES)["output_nodes"].rows)
    assert nodes[ex + P.PARTICLES] == P.NODE_PARTICLES

    # new thresholds, as the threshold viewer saves them
    cwd = os.getcwd()
    counts = RP.save_thresholds(project, ex, {"TS_01": 0.45, "TS_03": None})
    assert os.getcwd() == cwd
    assert counts["TS_01"] == (15, 3) and counts["TS_03"] == (15, 0) and counts["TS_43"] == (15, 11)
    sel = RP.read_selection(project, ex)
    assert sel.threshold["TS_01"] == pytest.approx(0.45) and sel.threshold["TS_03"] is None
    assert sel.auto["TS_01"] == pytest.approx(0.3)
    assert len(read_star(e / P.PARTICLES)["particles"].rows) == 3 + 0 + 3 * 11


def test_failed_tomogram_and_continue(project, fake_pytom):
    job = "External/job015/"
    res = run_runner(project, "match", job, match_values(gpu_ids="0"), env={"FAKE_FAIL": "TS_43"})
    assert res.returncode == 1 and "TS_43" in res.stderr and "Continue the job" in res.stderr
    s = RP.read_match_summary(project, job + P.SUMMARY)
    status = dict(zip(s.names, s.table.column(P.COL_STATUS)))
    assert status["TS_43"] == "failed" and status["TS_54"] == "done"  # the others went on
    fake_pytom.unlink()
    res = run_runner(project, "match", job, match_values(gpu_ids="0"))
    assert res.returncode == 0
    assert [c.split()[c.split().index("-v") + 1] for c in fake_pytom.read_text().splitlines()
            if c.startswith("match")] == [job + "inputs/TS_43/TS_43.mrc"]


def test_abort(project, fake_pytom):
    job = "External/job015/"
    proc = run_runner(project, "match", job, match_values(gpu_ids="0"), env={"FAKE_SLEEP": "60"}, wait=False)
    d = project / job
    end = time.time() + 30
    while time.time() < end and not (d / "logs/TS_01.log").exists():
        time.sleep(0.2)
    (d / P.ABORT_NOW).touch()
    proc.wait(timeout=40)
    assert (d / P.ABORTED).exists() and not (d / P.SUCCESS).exists()
    s = RP.read_match_summary(project, job + P.SUMMARY)
    assert s.table.column(P.COL_STATUS)[0] == "aborted"


def test_sigterm_stops_pytom(project, fake_pytom):
    job = "External/job015/"
    proc = run_runner(project, "match", job, match_values(gpu_ids="0"), env={"FAKE_SLEEP": "60"}, wait=False)
    d = project / job
    end = time.time() + 30
    while time.time() < end and not (d / "logs/TS_01.log").exists():
        time.sleep(0.2)
    time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=40)
    assert (d / P.ABORTED).exists()


def test_prepared_mask_file(project, fake_pytom):
    job = "External/job015/"
    with mrcfile.new(project / "mask_in.mrc") as m:
        m.set_data(np.ones((30, 30, 30), np.float32))
    res = run_runner(project, "match", job, match_values(mask_source=P.MASK_FILE, fn_mask="mask_in.mrc",
                                                         do_prep_mask="Yes", mask_invert="Yes", tomo_names="TS_01"))
    assert res.returncode == 0, res.stderr
    with mrcfile.open(project / job / "mask.mrc") as m:
        assert m.data.shape == (24, 24, 24) and float(m.data.max()) == 0.0  # resampled to the template, inverted
    res = run_runner(project, "match", "External/job017/", match_values(
        mask_source=P.MASK_FILE, fn_mask="mask_in.mrc", tomo_names="TS_01"))
    assert res.returncode == 1 and "same box" in res.stderr


def test_conda_prefix():
    assert P.conda_prefix({"pytom_env": "pytom_tm", "conda": "/opt/conda/bin/conda"}) == [
        "/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "pytom_tm"]
    assert P.conda_prefix({"pytom_env": "/envs/pytom", "conda": "mamba"})[-2:] == ["-p", "/envs/pytom"]
    assert P.conda_prefix({"pytom_env": ""}) == []


def test_gaussian_cut_off():
    from math import erfc, sqrt
    cut = RP.gaussian_cut_off(0.02, 6.5e7)
    assert erfc(cut / 0.02 / sqrt(2)) == pytest.approx(2 / 6.5e7, rel=1e-6)


# ---------------------------------------------------------------------------- views


@pytest.fixture(scope="module")
def _done_jobs_once(tomo_project, tmp_path_factory):
    """A project with a finished match and a finished extract job, run once for the tests that only look
    at their results (each gets a copy: done_jobs)."""
    tmp = tmp_path_factory.mktemp("pytom_done")
    project = _project(tomo_project, tmp)
    with pytest.MonkeyPatch.context() as mp:
        _fake_pytom(tmp, mp)
        assert run_runner(project, "match", "External/job015/", match_values()).returncode == 0
        assert run_runner(project, "extract", "External/job016/", extract_values()).returncode == 0
    for job, kind in (("External/job015/", "match"), ("External/job016/", "extract")):
        vals = match_values() if kind == "match" else extract_values()
        CATALOG.variant(kind).make_jobstar(vals, False, job, None).write(project / job / "job.star")
    return project


@pytest.fixture
def done_jobs(_done_jobs_once, fake_pytom, tmp_path):
    dst = tmp_path / "tomo"
    shutil.copytree(_done_jobs_once, dst, symlinks=True)
    return dst


def _ctx(project, job):
    js = JobStar.read(project / job / "job.star")
    return R.JobContext(project, job, "relion.external", "Succeeded", [], js.options)


def test_registry(done_jobs):
    views = R.views_for(_ctx(done_jobs, "External/job015/"))
    assert [v.title for v in views][:3] == ["Scores", "Tomograms", "Maps"]
    views = R.views_for(_ctx(done_jobs, "External/job016/"))
    titles = [v.title for v in views]
    assert titles[:4] == ["Thresholds", "Picks", "pytom plots", "Particle plots"]
    picks = views[1]
    assert picks.args["star"] == "External/job015/pytom_match.star"
    assert picks.args["picks"] == "External/job016/particles.star"


def test_last_settings(done_jobs):
    p = Pipeline(job_counter=17)
    from cryo_savanna.project.pipeline import Process
    p.processes["External/job015/"] = Process("External/job015/", "", "relion.external", "Succeeded")
    vals = match_values(pytom_env="my_pytom")
    CATALOG.variant("match").make_jobstar(vals, False, "External/job015/", None).write(
        done_jobs / "External/job015/job.star")
    assert last_settings(done_jobs, p, "extract") == {"cuda_module": "", "pytom_env": "my_pytom", "conda": ""}


def test_scores_view(qtbot, done_jobs):
    from cryo_savanna.viewers.pytom_view import PytomScoresView
    v = PytomScoresView(done_jobs, "External/job015/")
    qtbot.addWidget(v)
    assert v.tasks.wait()
    assert v.table.rowCount() == 5 and v.table.item(0, 1).text() == "done"
    assert v.tasks.wait()
    assert v._score is not None and v._score.peak == pytest.approx(0.5)
    assert len(v.figures.figures) == 2 and v.figures.figures[0].vlines
    assert "5 done" in v.info.text()


def test_tomogram_viewer_scores_beside(qtbot, done_jobs):
    from cryo_savanna.viewers.tomo_viewer import TomogramViewer
    v = TomogramViewer(done_jobs, "External/job015/pytom_match.star")
    qtbot.addWidget(v)
    assert v.tasks.wait()
    kinds = [v.kind.itemData(i) for i in range(v.kind.count())]
    assert kinds == ["rlnTomoReconstructedTomogram", "savannaPytomScores"]
    assert v.compare.isVisibleTo(v) and v.compare.text() == "pytom scores beside it"
    v.kind.setCurrentIndex(1)
    assert v.compare.text() == "Raw beside it"
    v.compare.setChecked(True)
    assert v.tasks.wait()
    assert v.raw.isVisibleTo(v)


def test_threshold_view(qtbot, done_jobs, monkeypatch):
    from cryo_savanna.viewers.pytom_view import PytomThresholdView
    v = PytomThresholdView(done_jobs, "External/job016/", "External/job015/pytom_match.star")
    qtbot.addWidget(v)
    assert v.tasks.wait()
    assert v.viewer is not None and v.diameter == 120.0  # the matching job's
    assert v.viewer.tasks.wait()
    assert v.table.rowCount() == 5 and v.table.item(0, 3).text() == "15" and v.table.item(0, 4).text() == "11"
    assert v.viewer.picks is not None and v.viewer.picks.count("TS_01") == 11
    v.set_threshold(0.45)
    assert v.table.item(0, 4).text() == "3" and v.viewer.picks.count("TS_01") == 3 and v.dirty
    assert v.save_btn.isEnabled()
    v.same.setChecked(True)  # all tomograms get the threshold shown
    assert all(v.table.item(i, 4).text() == "3" for i in range(5))
    v._reset()
    assert v.table.item(1, 4).text() == "11"
    v.set_threshold(0.41)  # only the tomogram shown, now
    assert v.save(ask=False)
    assert len(read_star(done_jobs / "External/job016/particles.star")["particles"].rows) == 5 + 4 * 11
    v.refresh()  # its own save: not read again
    assert not v.tasks.busy()


def test_svg_view(qtbot, done_jobs):
    from cryo_savanna.viewers.pytom_view import SvgPlotsView
    v = SvgPlotsView(done_jobs, "External/job016/")
    qtbot.addWidget(v)
    assert v.list.count() == 10 and v.info.text().endswith(".svg")


def test_new_pytom_job_in_window(qtbot, done_jobs, tmp_path):
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(done_jobs, poll_s=1000.0, settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    titles = [a.text() for a in w.actions.external_menu.actions()]
    assert titles == ["pytom: template matching", "pytom: extract candidates"]
    w.actions.new_pytom_job("match")
    ed = w.actions.editor
    assert ed.variant.kind == "match" and "pytom template matching" in ed.title.text()
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())] == \
        ["Inputs and template", "Weighting and CTF", "Picking", "Compute", "Running"]
    assert ed.values()["in_tomograms"].endswith("tomograms.star")
    assert not ed.widgets["template_box"].isEnabled() or ed.values()["do_prep_template"] == "Yes"
    ed.widgets["do_prep_template"].set_value("No")
    ed._on_changed("do_prep_template")
    assert not ed.widgets["template_box"].isEnabled()
    js = ed.jobstar()
    assert pytom_kind_of(js) == "match" and "--do_prep_template No" in js.options["other_args"]


def test_options_tab_shows_the_pytom_form(qtbot):
    from cryo_savanna.ui.job_info import JobInfoPanel

    v = CATALOG.variant("match")
    values = v.default_values(ENV)
    values.update(gpu_ids="0:1", search_z="40 160", particle_diameter="120")
    js = v.make_jobstar(values, False, "External/job015/", None)
    js.options["other_args"] += " --not_a_field 3"
    panel = JobInfoPanel(JobDefs.load(), ENV, {})
    qtbot.addWidget(panel)
    panel._jobstar = js
    panel._fill_options()
    tree = panel.options
    tabs = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert tabs[:4] == ["Inputs and template", "Weighting and CTF", "Picking", "Compute"]
    assert "Running" in tabs and tabs[-1] == "Not options of this form"
    rows = {tree.topLevelItem(i).child(j).text(0): tree.topLevelItem(i).child(j)
            for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())}
    assert rows["GPUs to use"].text(1) == "0:1" and rows["GPUs to use"].font(1).bold()
    assert rows["Search range z (px)"].text(1) == "40 160"
    assert rows["not_a_field"].text(1) == "3"
    assert "Additional arguments" not in rows  # External's own fields are not shown
    panel.only_changed.setChecked(True)
    rows = [tree.topLevelItem(i).child(j).text(0)
            for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())]
    assert set(rows) >= {"GPUs to use", "Search range z (px)", "Particle diameter (Å)"} and "Volume split (x y z)" not in rows


def test_failed_log_shows_the_worker_error(tmp_path, capsys):
    log = tmp_path / "TS_01.log"
    worker = ["Process Process-1:\n", "Traceback (most recent call last):\n", '  File "x.py", line 1\n',
              "cupy.cuda.compiler.CompileException: cannot open cuda_fp16.h\n"]
    main = ["Traceback (most recent call last):\n", '  File "y.py", line 2\n',
            "RuntimeError: One or more of the processes stopped unexpectedly.\n"]
    log.write_text("".join(["info\n"] + worker + [f"line {i}\n" for i in range(40)] + main))
    P._show_tail(str(log))
    err = capsys.readouterr().err
    assert "first error in the log (line 2)" in err and "CompileException" in err
    assert err.index("CompileException") < err.index("stopped unexpectedly")
    log.write_text("".join(main))  # short: the error is in the tail already, shown once
    P._show_tail(str(log))
    err = capsys.readouterr().err
    assert "first error" not in err and err.count("stopped unexpectedly") == 1


FAKE_MODULES = """
module() {
  case "$1" in
    load) shift
      for m in "$@"; do
        case "$m" in
          cuda/12.4) export CUDA_HOME=/fake/cuda-12.4 PATH="/fake/cuda-12.4/bin:$PATH"; unset GONE_AFTER_LOAD ;;
          *) echo "ERROR: Unable to locate a modulefile for '$m'" >&2 ;;
        esac
      done ;;
    -t) printf '/fake/modules:\\ncuda/11.8\\ncuda/12.4(default)\\ncudnn/9.1\\n' >&2 ;;
  esac
}
"""


@pytest.fixture
def fake_modules(tmp_path, monkeypatch):
    home = tmp_path / "modules"
    (home / "init").mkdir(parents=True)
    (home / "init/bash").write_text(FAKE_MODULES)
    monkeypatch.setenv("MODULESHOME", str(home))
    return home


@pytest.fixture
def saved_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_load_modules(fake_modules, monkeypatch, saved_environ):
    monkeypatch.setenv("GONE_AFTER_LOAD", "1")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    path = os.environ["PATH"]
    P.load_modules({"cuda_module": "cuda/12.4"})
    assert os.environ["CUDA_HOME"] == "/fake/cuda-12.4"
    assert os.environ["PATH"] == "/fake/cuda-12.4/bin:" + path and "GONE_AFTER_LOAD" not in os.environ
    with pytest.raises(P.Usage, match="(?s)module load cuda/99.*Unable to locate"):  # exit code 0, but ERROR
        P.load_modules({"cuda_module": "cuda/99"})
    P.load_modules({"cuda_module": ""})  # nothing to load
    monkeypatch.setenv("MODULESHOME", "/nonexistent")
    if not any(os.path.isfile(f) for f in ("/etc/profile.d/lmod.sh", "/etc/profile.d/modules.sh",
                                           "/usr/share/lmod/lmod/init/bash")):
        with pytest.raises(P.Usage, match="no module command"):
            P.load_modules({"cuda_module": "cuda/12.4"})


def test_match_loads_the_cuda_module(project, fake_pytom, fake_modules):
    res = run_runner(project, "match", "External/job015/", match_values(cuda_module="cuda/12.4", tomo_names="TS_01"))
    assert res.returncode == 0, res.stderr
    assert "module load cuda/12.4" in res.stdout and "CUDA in /fake/cuda-12.4" in res.stdout
    assert Path(str(fake_pytom) + ".cuda").read_text().split() == ["/fake/cuda-12.4"]
    res = run_runner(project, "match", "External/job016/", match_values(cuda_module="cuda/99"))
    assert res.returncode == 1 and "Unable to locate a modulefile" in res.stderr
    assert (project / "External/job016" / P.FAILURE).exists()


def test_default_cuda_module(fake_modules, monkeypatch):
    from cryo_savanna.jobdefs import pytom_jobs as J
    monkeypatch.delenv(J.CUDA_ENV_VAR, raising=False)
    monkeypatch.setenv("LOADEDMODULES", "gcc/12:cuda/12.1:cudnn/9")
    J.default_cuda_module.cache_clear()
    assert J.default_cuda_module() == "cuda/12.1"
    monkeypatch.setenv("LOADEDMODULES", "gcc/12")
    J.default_cuda_module.cache_clear()
    assert J.default_cuda_module() == "cuda/12.4"  # module avail: the one marked default
    monkeypatch.setenv(J.CUDA_ENV_VAR, "CUDA/12.6.0")
    J.default_cuda_module.cache_clear()
    assert J.default_cuda_module() == "CUDA/12.6.0"
    J.default_cuda_module.cache_clear()
    assert J.pick_cuda_module("/a:\ncuda/11.8\ncuda/12.10\ncuda/12.9\n") == "cuda/12.10"
    assert J.pick_cuda_module("/a:\ncuda/12.2\ncuda/12.6 (D)\n") == "cuda/12.6"
    assert J.pick_cuda_module("No module(s) or extension(s) found!\n") == ""
    assert J.pick_cuda_module("/apps:\ngcc/12\nopenmpi/4.1\ncudnn/9\n") == ""  # no CUDA module: empty


def test_empty_cuda_module_uses_the_path(saved_environ, monkeypatch):
    monkeypatch.setenv("MODULESHOME", "/nonexistent")  # no module command at all: still fine
    before = dict(os.environ)
    P.load_modules({"cuda_module": ""})
    assert dict(os.environ) == before


def test_options_tab_reads_a_continued_job_star_again(qtbot, tmp_path):
    """A Continue writes job.star again in the same directory: the Options tab shows the new settings."""
    from cryo_savanna.project.pipeline import Process
    from cryo_savanna.project.snapshot import ProjectLoader
    from cryo_savanna.ui.job_info import JobInfoPanel

    job = "External/job007/"
    (tmp_path / job).mkdir(parents=True)
    p = Pipeline(job_counter=8)
    p.processes[job] = Process(job, "", "relion.external", "Running")
    from cryo_savanna.project.lock import PipelineLock
    with PipelineLock(tmp_path) as lock:
        p.write(tmp_path, lock=lock)
    v = CATALOG.variant("match")
    v.make_jobstar(match_values(cuda_module=""), False, job, None).write(tmp_path / job / "job.star")
    loader = ProjectLoader(tmp_path, JobDefs.load())
    panel = JobInfoPanel(JobDefs.load(), ENV, {})
    qtbot.addWidget(panel)

    def cuda():
        tree = panel.options
        for i in range(tree.topLevelItemCount()):
            t = tree.topLevelItem(i)
            for j in range(t.childCount()):
                if t.child(j).text(0) == "CUDA module":
                    return t.child(j).text(1)

    panel.show_job(loader.refresh(force=True), job)
    assert cuda() == ""
    path = tmp_path / job / "job.star"
    v.make_jobstar(match_values(cuda_module="cuda/12.4"), True, job, None).write(path)
    os.utime(path, ns=(time.time_ns() + 10**9,) * 2)  # a coarse clock: make the change visible
    panel.show_job(loader.refresh(force=True), job)
    assert cuda() == "cuda/12.4"


def test_a_tilt_without_an_angle_is_named(tmp_path):
    """Not a TypeError from sorting the tilts: the tomogram and the file are named."""
    star = tmp_path / "TS_1.star"
    star.write_text("data_TS_1\n\nloop_\n_rlnTomoNominalStageTiltAngle #1\n_rlnMicrographPreExposure #2\n"
                    "-3.0 0\nnan 3\n3.0 6\n")
    with pytest.raises(P.Usage, match="TS_1: a tilt without a rlnTomoNominalStageTiltAngle"):
        P.tilt_table({"rlnTomoName": "TS_1", "rlnTomoTiltSeriesStarFile": str(star)}, {})
    star.write_text(star.read_text().replace("nan", "0.0"))
    assert [t[0] for t in P.tilt_table({"rlnTomoName": "TS_1", "rlnTomoTiltSeriesStarFile": str(star)}, {})[1]] \
        == [-3.0, 0.0, 3.0]


# pytom_match_template.py's options (pytom-match-pick 0.14, its --help)
PYTOM_MATCH_FLAGS = {
    "-t", "-v", "-d", "-m", "--non-spherical-mask", "--particle-diameter", "--angular-search",
    "--z-axis-rotational-symmetry", "-s", "--search-x", "--search-y", "--search-z", "--tomogram-mask", "-a",
    "--tilt-angles-first-column", "--per-tilt-weighting", "--voxel-size-angstrom", "--low-pass", "--high-pass",
    "--dose-accumulation", "--defocus", "--amplitude-contrast", "--spherical-aberration", "--voltage",
    "--phase-shift", "--tomogram-ctf-model", "--defocus-handedness", "--spectral-whitening", "-r",
    "--half-precision", "--rng-seed", "--relion5-tomograms-star", "--warp-xml-file", "-g", "--log"}


def test_match_command_uses_only_pytom_flags(tmp_path):
    opts = {k: "Yes" for k in ("per_tilt_weighting", "spectral_whitening", "non_spherical", "random_phase",
                                "half_precision")}
    opts.update(phase_shift="5", tomogram_ctf_model="phase-flip", defocus_handedness="1", low_pass="20",
                high_pass="400", particle_diameter="230", z_symmetry="2", rng_seed="7", volume_split="2 2 1",
                search_x="0 100", search_y="0 100", search_z="10 50", mirror_template="Yes",
                invert_template_contrast="Yes")
    files = {"tomogram": "t.mrc", "tilts": "t.tlt", "voxel": 10.0, "dose": "d.txt", "defocus": "f.txt",
             "ctf": (300, 2.7, 0.1), "tomogram_mask": "m.mrc"}
    cmd = P.match_command(opts, files, "template.mrc", "mask.mrc", [0, 1], "tm")
    flags = {w for w in cmd if w.startswith("-") and not w.lstrip("-").replace(".", "").isdigit()}
    assert flags <= PYTOM_MATCH_FLAGS, flags - PYTOM_MATCH_FLAGS
    assert "-r" in cmd and "--mirror-template" not in cmd and "--invert-template-contrast" not in cmd
    assert not {"mirror_template", "invert_template_contrast"} & set(CATALOG.variant("match").options)


def test_mask_mirror(project, fake_pytom):
    opts = {"mask_mirror": "Yes", "template_mirror": "No"}
    assert "--mirror" in P.template_command(opts, "m.mrc", "out.mrc", 1.0, 24, "mask")
    assert "--mirror" not in P.template_command({"template_mirror": "Yes"}, "m.mrc", "out.mrc", 1.0, 24, "mask")
    with mrcfile.new(project / "mask_in.mrc") as m:
        m.set_data(np.ones((30, 30, 30), np.float32))
    base = dict(mask_source=P.MASK_FILE, fn_mask="mask_in.mrc", do_prep_mask="Yes", tomo_names="TS_01",
                template_mirror="Yes")
    res = run_runner(project, "match", "External/job015/", match_values(**base))
    assert res.returncode == 0, res.stderr
    assert "template is mirrored but the mask file is not" in res.stdout
    res = run_runner(project, "match", "External/job016/", match_values(mask_mirror="Yes", **base))
    assert res.returncode == 0, res.stderr
    assert "may not match" not in res.stdout
    assert "--mirror" in [line for line in fake_pytom.read_text().splitlines() if "mask_in.mrc" in line][-1]
    v = CATALOG.variant("match")
    group_of = {k: r.group for t in v.tabs for r in t.rows for k in r.keys}
    assert group_of["mask_mirror"] == group_of["mask_invert"]
