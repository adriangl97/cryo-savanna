# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""AreTomo3 jobs: the form, the runner (with a fake AreTomo3 on the PATH), the stacks, the views."""

import os
import shutil
import subprocess
import sys
import textwrap
import time

import mrcfile
import numpy as np
import pytest

from cryo_savanna import aretomo_runner as A
from cryo_savanna import pytom_runner as P
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.aretomo_jobs import (
    AretomoCatalog,
    aretomo_kind,
    aretomo_kind_of,
    aretomo_main_input,
    install_runner,
    last_settings,
    newest_input,
    runner_command,
)
from cryo_savanna.jobdefs.cli_tools import external_program
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import decode
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline
from cryo_savanna.project.star import read_star, write_star
from cryo_savanna.results import registry as R

ENV = RelionEnv({})
CATALOG = AretomoCatalog(JobDefs.load(), ENV)
NAMES = ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"]
CTF_SET = "CtfFind/job003/tilt_series_ctf.star"
ALIGNED_SET = "AlignTiltSeries/job005/aligned_tilt_series.star"

# ---------------------------------------------------------------------------- a fake AreTomo3

FAKE = """
    import os, sys, time, numpy as np, mrcfile
    a = sys.argv[1:]
    def arg(f, n=1):
        i = a.index(f)
        return a[i + 1] if n == 1 else a[i + 1:i + 1 + n]
    open(os.environ["FAKE_LOG"], "a").write(" ".join(["aretomo"] + a) + "\\n")
    if os.environ.get("FAKE_SLEEP"):
        time.sleep(float(os.environ["FAKE_SLEEP"]))
    stack, out, cmd = arg("-InPrefix"), arg("-OutDir"), arg("-Cmd")
    name = os.path.splitext(os.path.basename(stack))[0]
    if os.environ.get("FAKE_FAIL") == name:
        sys.exit(3)
    with mrcfile.open(stack, permissive=True) as m:
        n, ny, nx = m.data.shape
        px = float(m.voxel_size.x)
    tilts = [float(line.split()[0]) for line in open(stack[:-4] + "_TLT.txt")]
    assert tilts == sorted(tilts) and len(tilts) == n, (len(tilts), n)
    b = float(arg("-AtBin"))
    volz = int(arg("-VolZ"))
    volz = 320 if volz < 0 else volz
    if cmd == "1":
        patches = int(arg("-AtPatch", 2)[0]) * int(arg("-AtPatch", 2)[1]) if "-AtPatch" in a else 0
        axis = float(arg("-TiltAxis")) if "-TiltAxis" in a else 0.0
        lines = ["# AreTomo Alignment / Priims bprmMn", "# RawSize = %d %d %d" % (nx, ny, n),
                 "# NumPatches = %d" % patches, "# DarkFrame =    %4d %4d %8.2f" % (n - 1, n, tilts[-1]),
                 "# AlphaOffset =     0.50", "# BetaOffset =     0.00", "# Thickness = 300",
                 "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT"]
        for s in range(1, n):  # the last (highest tilt) is dark
            lines.append("%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  %7.2f  %7.2f  %7.2f  %8.2f"
                         % (s, axis + 0.3, 1, s * 0.5, -s * 0.25, 1, 1, 1, 0, tilts[s - 1] + 0.5))
        if patches:
            lines.append("# Local Alignment")
            lines += ["%4d %3d %8.2f  %8.2f  %8.2f  %8.2f  %4.1f" % (t, p, 10, 20, 0.1, 0.2, 1.0)
                      for t in range(n - 1) for p in range(patches)]
        open(os.path.join(out, name + ".aln"), "w").write("\\n".join(lines) + "\\n")
        with open(os.path.join(out, name + "_CTF.txt"), "w") as fh:
            fh.write("# Columns: #1 micrograph number; ...\\n")
            for i in range(n):
                fh.write("%4d   %8.2f  %8.2f  %8.2f  %9.4f  %8.4f  %8.4f  %3d\\n"
                         % (i + 1, 20000 + i, 19000 + i, 45, 0, 0.1, 8.5, 1))
    else:
        aln = stack[:-4] + ".aln"
        assert os.path.exists(aln)
        open(os.environ["FAKE_LOG"] + "." + name + ".aln", "w").write(open(aln).read())
        if os.path.exists(stack[:-4] + "_CTF.txt"):
            open(os.environ["FAKE_LOG"] + "." + name + ".ctf", "w").write(open(stack[:-4] + "_CTF.txt").read())
    vols = ["_Vol"]
    if arg("-SplitSum") == "1":
        assert os.path.exists(stack[:-4] + "_EVN.mrc") and os.path.exists(stack[:-4] + "_ODD.mrc")
        vols += ["_EVN_Vol", "_ODD_Vol"]
    for v in vols:
        with mrcfile.new(os.path.join(out, name + v + ".mrc"), overwrite=True) as m:
            m.set_data(np.zeros((int(volz / b) // 2 * 2, int(ny / b), int(nx / b)), np.float32))
            m.voxel_size = px * b
"""


@pytest.fixture
def fake_aretomo(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    f = bin_dir / "AreTomo3"
    f.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE))
    f.chmod(0o755)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    return log


@pytest.fixture
def project(tomo_project, tmp_path):
    dst = tmp_path / "tomo"
    shutil.copytree(tomo_project, dst, symlinks=True)
    return dst


def values(**kw):
    v = CATALOG.variant("align").default_values(ENV)
    v.update({"in_tiltseries": CTF_SET, "gpu_ids": "0:1", "cuda_module": "", "aretomo_exe": "AreTomo3"})
    v.update(kw)
    return v


def run_runner(project, job, vals, env=None, wait=True):
    js = CATALOG.variant("align").make_jobstar(vals, False, job, None)
    (project / job).mkdir(parents=True, exist_ok=True)
    install_runner(project)
    cmd = f"{js.options['fn_exe']} --o {job} --j 1 {js.options['other_args']}"
    e = dict(os.environ, **(env or {}))
    if not wait:
        return subprocess.Popen(cmd, shell=True, cwd=project, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
    return subprocess.run(cmd, shell=True, cwd=project, env=e, capture_output=True, text=True, timeout=120)


def calls(log):
    return [c.split() for c in log.read_text().splitlines()]


def flag(call, name, n=1):
    i = call.index(name)
    return call[i + 1] if n == 1 else call[i + 1:i + 1 + n]


def table(path, block):
    b = read_star(path)[block]
    return [dict(zip(b.columns, r)) for r in b.rows], b.columns


# ---------------------------------------------------------------------------- forms


def test_form_round_trip():
    v = CATALOG.variant("align")
    assert v.label == "relion.external" and v.is_tomo and v.tabs[-1].label == "Running"
    for t in v.tabs:
        for r in t.rows:
            assert all(k in v.options for k in r.keys)
            assert r.group is None or r.group in v.groups
    js = v.new_jobstar(ENV)
    assert aretomo_kind_of(js) == "align" and js.options["fn_exe"] == runner_command("align")
    assert v.values_from_jobstar(js, ENV)[0] == v.default_values(ENV)
    vals = values(tomo_names="TS_01 TS_03", aretomo_args="-OutImod 1", alignment=A.ALIGN_INPUT, do_queue="Yes")
    js = v.make_jobstar(vals, False, "External/job020/", None)
    assert '"' not in js.options["other_args"] and "\\" not in js.options["other_args"]  # no escaped quotes
    back, ignored = v.values_from_jobstar(js, ENV)
    assert back == vals and not ignored
    group_of = {r.keys[0]: r.group for t in v.tabs for r in t.rows}
    assert v.group_active(group_of["at_patch"], vals) is False  # alignment options: only when AreTomo3 aligns
    assert v.group_active(group_of["at_patch"], values())


def test_names_and_main_input(tomo_project):
    assert aretomo_kind("python3 .cryo-savanna/aretomo_runner.py align") == "align"
    assert aretomo_kind("python3 .cryo-savanna/pytom_runner.py match") is None
    assert external_program(runner_command("align")) == "AreTomo3 alignment and reconstruction"
    js = CATALOG.variant("align").make_jobstar(values(in_tiltseries="./" + CTF_SET), False, "External/job020/", None)
    assert aretomo_main_input(js) == CTF_SET
    pipeline = Pipeline.read(tomo_project)
    assert newest_input(pipeline) == {"in_tiltseries": ALIGNED_SET}  # the newest tilt-series set, not tomograms
    js = CATALOG.variant("align").make_jobstar(values(), False, "External/job020/", pipeline)
    o = js.options
    assert CTF_SET in (o["in_mov"], o["in_mic"], o["in_part"], o["in_coords"], o["in_3dref"], o["in_mask"])


def test_install_puts_both_runners(tmp_path):
    install_runner(tmp_path)
    assert (tmp_path / ".cryo-savanna/aretomo_runner.py").read_bytes() == open(A.__file__, "rb").read()
    assert (tmp_path / ".cryo-savanna/pytom_runner.py").read_bytes() == open(P.__file__, "rb").read()


# ---------------------------------------------------------------------------- files


def test_doses():
    tilts = [{"rlnMicrographPreExposure": str(x)} for x in (6, 3, 0, 9)]
    acq, dose = A.doses(tilts)
    assert acq == [3, 2, 1, 4] and dose == [3.0, 3.0, 3.0, 3.0]
    assert A.doses([{}, {}]) == ([1, 2], [0.0, 0.0])


def test_aln_round_trip(tmp_path):
    f = tmp_path / "x.aln"
    A.write_aln(f, (64, 64, 3), [(1, 85.0, 1.5, -2.0, -3.0), (2, 85.1, 0, 0, 0), (3, 85.2, 1, 1, 3)],
                patches=1, local=["   0   0    10.00    20.00     0.10     0.20   1.0"] * 3, thickness=250, alpha=0.5)
    aln = A.read_aln(f)
    assert [r["sec"] for r in aln["rows"]] == [1, 2, 3] and aln["rows"][0]["tx"] == 1.5 and aln["rows"][2]["tilt"] == 3
    assert aln["patches"] == 1 and len(aln["local"]) == 3
    assert A._header_number(aln, "Thickness") == 250 and A._header_number(aln, "AlphaOffset") == 0.5
    # 0-based sections (older AreTomo) become 1-based
    f.write_text(f.read_text().replace("    1   85.0000", "    0   85.0000").replace("    2   85.1", "    1   85.1")
                 .replace("    3   85.2", "    2   85.2"))
    assert [r["sec"] for r in A.read_aln(f)["rows"]] == [1, 2, 3]


@pytest.mark.parametrize("numpy", [True, False])
@pytest.mark.parametrize("dtype,mode", [(np.float16, 2), (np.int16, 1), (np.float32, 2), (np.int8, 2)])
def test_build_stack(tmp_path, monkeypatch, numpy, dtype, mode):
    rng = np.random.default_rng(1)
    images = []
    for i in range(3):
        f = tmp_path / f"{i}.mrc"
        with mrcfile.new(f) as m:
            m.set_data((rng.normal(0, 20, (5, 7))).astype(dtype))
            m.voxel_size = 2.5
        images.append(str(f))
    out = tmp_path / "stack.mrc"
    with monkeypatch.context() as m:
        if not numpy:
            m.setitem(sys.modules, "numpy", None)  # the stdlib path, as with a python without numpy
        A.build_stack(images, str(out), 2.5)
    with mrcfile.open(out) as m:
        assert m.header.mode == mode and m.data.shape == (3, 5, 7)
        assert abs(float(m.voxel_size.x) - 2.5) < 1e-6 and abs(float(m.voxel_size.z) - 2.5) < 1e-6
        for k, f in enumerate(images):
            with mrcfile.open(f) as src:
                np.testing.assert_array_equal(m.data[k], src.data.astype(m.data.dtype))


def test_build_stack_refuses_other_sizes(tmp_path):
    for i, shape in enumerate(((5, 7), (5, 8))):
        with mrcfile.new(tmp_path / f"{i}.mrc") as m:
            m.set_data(np.zeros(shape, np.float32))
    with pytest.raises(A.Usage, match="not a 7×5 image"):
        A.build_stack([str(tmp_path / "0.mrc"), str(tmp_path / "1.mrc")], str(tmp_path / "s.mrc"), 1.0)


# ---------------------------------------------------------------------------- the runner


def test_align_and_reconstruct(project, fake_aretomo):
    job = "External/job015/"
    res = run_runner(project, job, values(at_patch="2 2"))
    assert res.returncode == 0, res.stderr + res.stdout
    d = project / job
    assert (d / P.SUCCESS).exists()
    runs = calls(fake_aretomo)
    assert len(runs) == 5 and sorted(flag(c, "-Gpu") for c in runs) == ["0", "0", "0", "1", "1"]
    c = runs[0]
    assert flag(c, "-Cmd") == "1" and flag(c, "-InSuffix") == ".mrc" and flag(c, "-PixSize") == "5.4"
    assert flag(c, "-kV") == "300" and flag(c, "-Cs") == "2.7" and flag(c, "-AmpContrast") == "0.1"
    assert flag(c, "-TiltAxis") == "85" and flag(c, "-AtBin") == "4" and flag(c, "-VolZ") == "-1"
    assert flag(c, "-AtPatch", 2) == ["2", "2"] and flag(c, "-FlipVol") == "1" and flag(c, "-SplitSum") == "0"
    assert flag(c, "-CorrCTF") == "1" and flag(c, "-TiltCor") == "0" and "-Wbp" not in c
    # the inputs: the tilts sorted, with acquisition order and dose; the stacks gone afterwards
    tlt = [line.split() for line in (d / "inputs/TS_01/TS_01_TLT.txt").read_text().splitlines()]
    assert len(tlt) == 41 and float(tlt[0][0]) == -60 and tlt[20][1] == "1" and float(tlt[20][2]) == 3.0
    assert not (d / "inputs/TS_01/TS_01.mrc").exists()
    # the alignment in RELION's convention, the dark tilt left out
    rows, cols = table(d / "tilt_series/TS_01.star", "TS_01")
    assert len(rows) == 40 and set(A.ALIGN_COLS) <= set(cols)
    r = rows[1]  # section 2
    assert float(r["rlnTomoXTilt"]) == 0 and float(r["rlnTomoZRot"]) == pytest.approx(85.3)
    assert float(r["rlnTomoYTilt"]) == pytest.approx(float(r["rlnTomoNominalStageTiltAngle"]) + 0.5)
    assert float(r["rlnTomoXShiftAngst"]) == pytest.approx(1.0 * 5.4) and float(r["rlnTomoYShiftAngst"]) == pytest.approx(-0.5 * 5.4)
    assert float(rows[-1]["rlnTomoNominalStageTiltAngle"]) == 57  # 60 was dark
    # CtfFind's defocus stays (the input has some)
    src, _ = table(project / "CtfFind/job003/tilt_series/TS_01.star", "TS_01")
    assert rows[0]["rlnDefocusU"] == sorted(src, key=lambda t: float(t["rlnTomoNominalStageTiltAngle"]))[0]["rlnDefocusU"]
    # aligned_tilt_series.star and tomograms.star for RELION
    g, gcols = table(d / A.ALIGNED, "global")
    assert [x["rlnTomoName"] for x in g] == NAMES and g[0]["rlnTomoTiltSeriesStarFile"] == job + "tilt_series/TS_01.star"
    assert g[0][A.COL_ALN] == job + "aretomo/TS_01/TS_01.aln" and g[0]["rlnTomoTiltSeriesPixelSize"] == "5.4"
    t, tcols = table(d / A.TOMOGRAMS, "global")
    assert t[0]["rlnTomoReconstructedTomogram"] == job + "aretomo/TS_01/TS_01_Vol.mrc"
    assert (t[0]["rlnTomoSizeX"], t[0]["rlnTomoSizeY"], t[0]["rlnTomoSizeZ"]) == ("64", "64", "320")
    assert float(t[0]["rlnTomoTomogramBinning"]) == 4 and "rlnTomoReconstructedTomogramHalf1" not in tcols
    s, _ = table(d / A.SUMMARY, "global")
    assert [x[A.COL_STATUS] for x in s] == ["done"] * 5
    assert s[0][A.COL_DARK] == "1" and s[0][A.COL_TILTS] == "40" and float(s[0][A.COL_THICKNESS]) == pytest.approx(1620)
    assert float(s[0][A.COL_AXIS]) == pytest.approx(85.3) and float(s[0][A.COL_ALPHA]) == 0.5
    nodes = (d / P.OUTPUT_NODES).read_text()
    assert A.NODE_ALIGNED in nodes and A.NODE_TOMOGRAMS in nodes and A.NODE_SUMMARY in nodes
    # run again (continue): everything was done
    fake_aretomo.unlink()
    assert run_runner(project, job, values(at_patch="2 2")).returncode == 0
    assert not fake_aretomo.exists()


def test_ctf_into_the_tilt_series(project, fake_aretomo):
    job = "External/job015/"
    res = run_runner(project, job, values(ctf_to_star=A.CTF_ALWAYS, tomo_names="TS_01", gpu_ids="0"))
    assert res.returncode == 0, res.stderr
    rows, cols = table(project / job / "tilt_series/TS_01.star", "TS_01")
    assert float(rows[0]["rlnDefocusU"]) == 20000 and float(rows[1]["rlnDefocusV"]) == 19001
    assert float(rows[0]["rlnDefocusAngle"]) == 45 and "rlnPhaseShift" not in cols
    s, _ = table(project / job / A.SUMMARY, "global")
    assert float(s[0][A.COL_CTF_RES]) == 8.5


def test_odd_even_halves(project, fake_aretomo):
    for name in NAMES[:1]:
        f = project / "CtfFind/job003/tilt_series" / f"{name}.star"
        blocks = read_star(f)
        b = blocks[name]
        mi = b.col_index("rlnMicrographName")
        b.columns += [A.EVEN, A.ODD]
        for r in b.rows:
            r += [r[mi], r[mi]]
        write_star(f, blocks.values(), atomic=False)
    job = "External/job015/"
    res = run_runner(project, job, values(do_halves="Yes", tomo_names="TS_01", gpu_ids="0", recon_method=A.RECON_WBP))
    assert res.returncode == 0, res.stderr
    c = calls(fake_aretomo)[0]
    assert flag(c, "-SplitSum") == "1" and flag(c, "-Wbp") == "1"
    t, _ = table(project / job / A.TOMOGRAMS, "global")
    assert t[0]["rlnTomoReconstructedTomogramHalf1"] == job + "aretomo/TS_01/TS_01_EVN_Vol.mrc"
    assert t[0]["rlnTomoReconstructedTomogramHalf2"] == job + "aretomo/TS_01/TS_01_ODD_Vol.mrc"
    res = run_runner(project, "External/job016/", values(do_halves="Yes", tomo_names="TS_03"))
    assert res.returncode == 1 and "rlnMicrographNameEven" in res.stderr


def test_reconstruct_with_relions_alignment(project, fake_aretomo):
    f = project / "AlignTiltSeries/job005/tilt_series/TS_01.star"  # with CTF values, as after CtfFind
    blocks = read_star(f)
    b = blocks["TS_01"]
    b.columns += ["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]
    for i, r in enumerate(b.rows):
        r += [str(30000 + i), str(29000 + i), "12.5"]
    write_star(f, blocks.values(), atomic=False)
    job = "External/job015/"
    vals = values(in_tiltseries=ALIGNED_SET, alignment=A.ALIGN_INPUT, tomo_names="TS_01", gpu_ids="0")
    res = run_runner(project, job, vals)
    assert res.returncode == 1 and "give the tomogram thickness" in res.stderr  # not AreTomo3's: no thickness
    res = run_runner(project, job, dict(vals, vol_z="1200"))
    assert res.returncode == 0, res.stderr
    c = calls(fake_aretomo)[0]
    assert flag(c, "-Cmd") == "2" and flag(c, "-VolZ") == "1200" and "-TiltAxis" not in c and "-AtPatch" not in c
    aln = A.read_aln(str(fake_aretomo) + ".TS_01.aln")
    src, _ = table(project / "AlignTiltSeries/job005/tilt_series/TS_01.star", "TS_01")
    src.sort(key=lambda t: float(t["rlnTomoYTilt"]))
    assert len(aln["rows"]) == 41 and aln["patches"] == 0
    assert aln["rows"][3]["rot"] == pytest.approx(float(src[3]["rlnTomoZRot"]), abs=1e-3)
    assert aln["rows"][3]["tx"] == pytest.approx(float(src[3]["rlnTomoXShiftAngst"]) / 5.4, abs=1e-3)
    ctf = A.read_ctf(str(fake_aretomo) + ".TS_01.ctf")
    assert len(ctf) == 41 and ctf[3][0] == pytest.approx(float(src[3]["rlnDefocusU"]), abs=0.01)
    assert not (project / job / A.ALIGNED).exists()  # only tomograms: the alignment is the input's
    t, _ = table(project / job / A.TOMOGRAMS, "global")
    assert t[0]["rlnTomoSizeZ"] == "1200"
    # without CTF values: no _CTF.txt, said in the log
    res = run_runner(project, "External/job016/", dict(vals, vol_z="1200", tomo_names="TS_03"))
    assert res.returncode == 0 and "no CTF values" in res.stdout
    assert not (project / "External/job016/inputs/TS_03/TS_03_CTF.txt").exists()
    assert t[0]["rlnTomoTiltSeriesStarFile"] == job + "tilt_series/TS_01.star"


def test_reconstruct_again_keeps_the_local_alignment(project, fake_aretomo):
    first = "External/job015/"
    assert run_runner(project, first, values(at_patch="2 2", tomo_names="TS_01", gpu_ids="0")).returncode == 0
    job = "External/job016/"
    vals = values(in_tiltseries=first + A.ALIGNED, alignment=A.ALIGN_INPUT, at_bin="8", gpu_ids="0")
    res = run_runner(project, job, vals)
    assert res.returncode == 0, res.stderr
    aln = A.read_aln(str(fake_aretomo) + ".TS_01.aln")
    assert aln["patches"] == 4 and len(aln["local"]) == 40 * 4 and len(aln["rows"]) == 40
    assert A._header_number(aln, "Thickness") == 300  # so -VolZ -1 works
    assert "local alignment (4 patches)" in res.stdout


def test_failed_tilt_series_and_continue(project, fake_aretomo):
    job = "External/job015/"
    res = run_runner(project, job, values(gpu_ids="0"), env={"FAKE_FAIL": "TS_43"})
    assert res.returncode == 1 and "TS_43" in res.stderr and "Continue the job" in res.stderr
    s, _ = table(project / job / A.SUMMARY, "global")
    status = {x["rlnTomoName"]: x[A.COL_STATUS] for x in s}
    assert status["TS_43"] == "failed" and status["TS_54"] == "done"
    t, _ = table(project / job / A.TOMOGRAMS, "global")
    assert "TS_43" not in [x["rlnTomoName"] for x in t] and len(t) == 4  # the others are usable already
    fake_aretomo.unlink()
    res = run_runner(project, job, values(gpu_ids="0"))
    assert res.returncode == 0, res.stderr
    assert [flag(c, "-InPrefix") for c in calls(fake_aretomo)] == [job + "inputs/TS_43/TS_43.mrc"]


def test_missing_executable(project, fake_aretomo):
    res = run_runner(project, "External/job015/", values(aretomo_exe="NoSuchAreTomo3"))
    assert res.returncode == 1 and "AreTomo3 not found" in res.stderr
    assert (project / "External/job015" / P.FAILURE).exists()


def test_abort(project, fake_aretomo):
    job = "External/job015/"
    proc = run_runner(project, job, values(gpu_ids="0"), env={"FAKE_SLEEP": "60"}, wait=False)
    d = project / job
    end = time.time() + 30
    while time.time() < end and not fake_aretomo.exists():
        time.sleep(0.2)
    (d / P.ABORT_NOW).touch()
    proc.wait(timeout=40)
    assert (d / P.ABORTED).exists() and not (d / P.SUCCESS).exists()
    s, _ = table(d / A.SUMMARY, "global")
    assert s[0][A.COL_STATUS] == "aborted"


# ---------------------------------------------------------------------------- the app


@pytest.fixture
def done_job(project, fake_aretomo):
    job = "External/job015/"
    assert run_runner(project, job, values(tomo_names="TS_01 TS_03")).returncode == 0
    return project, job


def test_registry(done_job):
    project, job = done_job
    js = CATALOG.variant("align").make_jobstar(values(tomo_names="TS_01 TS_03"), False, job, None)
    js.write(project / job / "job.star")
    assert JobStar.read(project / job / "job.star").options["fn_exe"] == runner_command("align")
    ctx = R.JobContext(project, job, "relion.external", "Succeeded", [], js.options)
    views = R.views_for(ctx)
    assert [v.title for v in views][:3] == ["Tomograms", "Tilt series", "AreTomo3"]
    assert views[0].args["star"] == job + A.TOMOGRAMS and views[1].args["star"] == job + A.ALIGNED


def test_last_settings(done_job):
    project, job = done_job
    js = CATALOG.variant("align").make_jobstar(values(cuda_module="cuda/12.4", aretomo_exe="/opt/AreTomo3"), False,
                                               job, None)
    js.write(project / job / "job.star")
    p = Pipeline(job_counter=16)
    from cryo_savanna.project.pipeline import Process
    p.processes[job] = Process(job, "", "relion.external", "Succeeded")
    assert last_settings(project, p) == {"cuda_module": "cuda/12.4", "aretomo_exe": "/opt/AreTomo3"}
    assert decode(js.options["other_args"])[0]["aretomo_exe"] == "/opt/AreTomo3"


def test_new_aretomo_job_in_window(qtbot, tomo_project, tmp_path):
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(tomo_project, poll_s=1000.0, settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    titles = [a.text() for a in w.actions.external_menu.actions()]
    assert titles[-1] == "AreTomo3: align and reconstruct tilt series"
    w.actions.new_aretomo_job("align")
    ed = w.actions.editor
    assert ed.variant.kind == "align" and "AreTomo3" in ed.title.text()
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())] == \
        ["Input", "Alignment", "Reconstruction", "Compute", "Running"]
    assert ed.values()["in_tiltseries"] == ALIGNED_SET
    assert ed.widgets["at_patch"].isEnabled()
    ed.widgets["alignment"].set_value(A.ALIGN_INPUT)
    ed._on_changed("alignment")
    assert not ed.widgets["at_patch"].isEnabled()
    js = ed.jobstar()
    assert aretomo_kind_of(js) == "align" and "--alignment 'Use the alignment of the input'" in js.options["other_args"]


def test_options_tab_shows_the_aretomo_form(qtbot):
    from cryo_savanna.ui.job_info import JobInfoPanel

    js = CATALOG.variant("align").make_jobstar(values(at_bin="6"), False, "External/job015/", None)
    panel = JobInfoPanel(JobDefs.load(), ENV, {})
    qtbot.addWidget(panel)
    panel._jobstar = js
    panel._fill_options()
    tree = panel.options
    tabs = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert tabs[:4] == ["Input", "Alignment", "Reconstruction", "Compute"]
    rows = {tree.topLevelItem(i).child(j).text(0): tree.topLevelItem(i).child(j)
            for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())}
    assert rows["Binning"].text(1) == "6" and rows["Binning"].font(1).bold()
