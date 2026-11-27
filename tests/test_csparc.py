# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""CryoSPARC particles to RELION (pyem csparc2star): the form, and the runner run as RELION runs it, on a
made-up CryoSPARC project with a fake csparc2star.py (on the PATH, following pyem's rules for coordinates,
paths and flags) and the test's Python (numpy) as the environment's python."""

import os
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cryo_savanna import csparc_runner as CS
from cryo_savanna import runner_common as C
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import external_jobs as X
from cryo_savanna.jobdefs.csparc_jobs import (
    ENV_VAR,
    CsparcCatalog,
    csparc_kind,
    csparc_main_input,
    install_runner,
    last_settings,
    newest_micrographs,
)
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.jobdefs.pytom_jobs import decode
from cryo_savanna.project.pipeline import Node, Pipeline, Process

ENV = RelionEnv({})
DEFS = JobDefs.load()
CATALOG = CsparcCatalog(DEFS, ENV)
JOB = "External/job020/"
MICS = "CtfFind/job003/micrographs_ctf.star"
UID_A, UID_B, UID_S = "012345678901234567890", "112345678901234567891", "212345678901234567892"
CS_SHAPE = (2880, 4096)  # CryoSPARC's micrographs (y, x), 0.5 Å/px; RELION's are binned by 2: 2048 x 1440 at 1 Å

# ---------------------------------------------------------------------------- a fake csparc2star.py

FAKE = r"""
    import os, sys, numpy as np
    a = sys.argv[1:]
    open(os.environ["FAKE_LOG"], "a").write(" ".join(a) + "\n")
    takes = ("--class", "--boxsize", "--minphic", "--transform", "--copy-micrograph-coordinates",
             "--micrograph-path", "--max-header-size", "--loglevel")
    pos, flags, i = [], {}, 0
    while i < len(a):
        if a[i] in takes:
            flags.setdefault(a[i], []).append(a[i + 1]); i += 2
        elif a[i].startswith("--"):
            flags[a[i]] = True; i += 1
        else:
            pos.append(a[i]); i += 1
    if os.environ.get("FAKE_FAIL"):
        print("Traceback (most recent call last):\nKeyError: 'blob/path'"); sys.exit(1)
    inputs, out = pos[:-1], pos[-1]
    cs = np.load(inputs[0])
    data = {n: cs[n] for n in cs.dtype.names}
    for p in inputs[1:]:  # passthroughs, by uid (in the order of the particles)
        pt = np.load(p)
        order = {u: k for k, u in enumerate(pt["uid"])}
        idx = [order[u] for u in cs["uid"]]
        for n in pt.dtype.names:
            data.setdefault(n, pt[n][idx])
    s = lambda v: (v.decode() if isinstance(v, bytes) else str(v)).lstrip(">")
    n = len(cs)
    xf, yf = data["location/center_x_frac"].astype(float), data["location/center_y_frac"].astype(float)
    if "--invertx" in flags: xf = 1 - xf
    if "--inverty" not in flags: yf = 1 - yf  # pyem flips Y unless --inverty
    shape = data["location/micrograph_shape"]
    if "--noswapxy" in flags:
        x, y = np.round(xf * shape[:, 0]), np.round(yf * shape[:, 1])
    else:
        x, y = np.round(xf * shape[:, 1]), np.round(yf * shape[:, 0])
    psize = data["blob/psize_A"].astype(float)
    optics = data["ctf/exp_group_id"].astype(int) + 1
    cols = {"rlnImageName": ["%06d@%s" % (k + 1, s(p)) for k, p in zip(data["blob/idx"], data["blob/path"])],
            "rlnMicrographName": [s(m) for m in data["location/micrograph_path"]],
            "rlnCoordinateX": x.astype(int), "rlnCoordinateY": y.astype(int),
            "rlnDefocusU": data["ctf/df1_A"], "rlnDefocusV": data["ctf/df2_A"],
            "rlnDefocusAngle": np.rad2deg(data["ctf/df_angle_rad"]), "rlnPhaseShift": np.zeros(n),
            "rlnOpticsGroup": optics}
    if "alignments3D/pose" in data:
        pose = np.rad2deg(data["alignments3D/pose"])
        cols["rlnAngleRot"], cols["rlnAngleTilt"], cols["rlnAnglePsi"] = pose[:, 0], pose[:, 1], pose[:, 2]
        cols["rlnOriginXAngst"] = data["alignments3D/shift"][:, 0] * psize
        cols["rlnOriginYAngst"] = data["alignments3D/shift"][:, 1] * psize
        if "--flipy-pose" in flags or "--flipy" in flags:
            cols["rlnOriginYAngst"] = -cols["rlnOriginYAngst"]
            cols["rlnAngleTilt"] = 180 - cols["rlnAngleTilt"]  # (a stand-in for pyem's transform)
    elif "alignments2D/pose" in data:
        cols["rlnAnglePsi"] = np.rad2deg(data["alignments2D/pose"])
        cols["rlnOriginXAngst"] = data["alignments2D/shift"][:, 0] * psize
        cols["rlnOriginYAngst"] = data["alignments2D/shift"][:, 1] * psize
    if "--flipy-ctf" in flags or "--flipy" in flags:
        cols["rlnDefocusAngle"] = -cols["rlnDefocusAngle"]
    with open(out, "w") as fh:
        fh.write("\ndata_optics\n\nloop_\n")
        oc = ["rlnAmplitudeContrast", "rlnSphericalAberration", "rlnVoltage", "rlnImagePixelSize",
              "rlnImageSize", "rlnImageDimensionality", "rlnOpticsGroup", "rlnOpticsGroupName"]
        fh.write("".join("_%s #%d \n" % (c, k + 1) for k, c in enumerate(oc)))
        for g in sorted(set(optics)):
            k = list(optics).index(g)
            fh.write("0.100000 2.700000 300.000000 %.6f %d 2 %d opticsGroup%d\n"
                     % (psize[k], data["blob/shape"][k][0], g, g))
        fh.write("\ndata_particles\n\nloop_\n")
        fh.write("".join("_%s #%d \n" % (c, k + 1) for k, c in enumerate(cols)))
        for k in range(n):
            fh.write(" ".join(("%.6f" % v[k]) if isinstance(v[k], (float, np.floating)) else str(v[k])
                              for v in cols.values()) + "\n")
"""


@pytest.fixture
def fake_pyem(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    f = bin_dir / "csparc2star.py"
    f.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE))
    f.chmod(0o755)
    (bin_dir / "python").symlink_to(sys.executable)  # pyem's environment's python, with numpy
    log = tmp_path / "fake.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CRYO_SAVANNA_POLL_S", "0.05")
    return log


def calls(log):
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


# ---------------------------------------------------------------------------- a CryoSPARC and a RELION project

def mrc_stub(path, nx, ny, nz, apix=1.0):
    """An MRC header without data (enough for what the runner reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = bytearray(1024)
    struct.pack_into("<4i", h, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", h, 28, nx, ny, nz)
    struct.pack_into("<3f", h, 40, nx * apix, ny * apix, nz * apix)
    h[208:212] = b"MAP "
    path.write_bytes(bytes(h))


def save_cs(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:  # (np.save of a name would add .npy)
        np.save(fh, arr)


def cs_mic(uid, name, job="J3", folder="motioncorrected", suffix="_patch_aligned_doseweighted"):
    return f"{job}/{folder}/{uid}_{name}{suffix}.mrc"


# the particles: (uid, stack, index in it, micrograph, x frac, y frac, exposure group)
PARTICLES = [
    (11, f"J12/extract/{UID_A}_mic_a_patch_aligned_doseweighted_particles.mrc", 0, cs_mic(UID_A, "mic_a"), 0.25, 0.25, 0),
    (12, f"J12/extract/{UID_A}_mic_a_patch_aligned_doseweighted_particles.mrc", 2, cs_mic(UID_A, "mic_a"), 0.5, 0.75, 0),
    (13, f"J12/extract/{UID_B}_mic_b_patch_aligned_doseweighted_particles.mrc", 1, cs_mic(UID_B, "mic_b"), 0.75, 0.5, 1),
    (14, f"J20/downsample/{UID_S}_mic_c_patch_aligned_doseweighted_particles_downsampled.mrc", 0,
     cs_mic(UID_S, "mic_c"), 0.1, 0.9, 1),
]


def passthrough_array(parts=PARTICLES, folder="motioncorrected", suffix="_patch_aligned_doseweighted", psize=1.2):
    dt = [("uid", "<u8"), ("blob/path", "S120"), ("blob/idx", "<u4"), ("blob/shape", "<u4", (2,)),
          ("blob/psize_A", "<f4"), ("location/micrograph_path", "S120"), ("location/micrograph_shape", "<u4", (2,)),
          ("location/micrograph_psize_A", "<f4"), ("location/center_x_frac", "<f4"),
          ("location/center_y_frac", "<f4"), ("ctf/df1_A", "<f4"), ("ctf/df2_A", "<f4"),
          ("ctf/df_angle_rad", "<f4"), ("ctf/exp_group_id", "<u4")]
    a = np.zeros(len(parts), dtype=dt)
    for k, (uid, stack, idx, mic, x, y, g) in enumerate(parts):
        mic = mic.replace("motioncorrected", folder).replace("_patch_aligned_doseweighted", suffix)
        a[k] = (uid, stack, idx, (256, 256), psize, mic, CS_SHAPE, 0.5, x, y, 12000 + k, 11800 + k, 0.5, g)
    return a


def particles_array(parts=PARTICLES, two_d=False):
    kind = "alignments2D" if two_d else "alignments3D"
    dt = [("uid", "<u8"), (f"{kind}/pose", "<f4") if two_d else (f"{kind}/pose", "<f4", (3,)),
          (f"{kind}/shift", "<f4", (2,)), (f"{kind}/class_posterior", "<f4")]
    a = np.zeros(len(parts), dtype=dt)
    for k, p in enumerate(parts):
        a[k] = (p[0], 0.3 if two_d else (0.1, 0.5, 0.3), (1.0 + k, 2.0 + k), 1.0)
    return a


@pytest.fixture
def cs_project(tmp_path):
    """CS-proj: Extract J12, Downsample J20, the refinement J45 (particles + passthrough)."""
    root = tmp_path / "cryosparc" / "CS-proj"
    for _, stack, *_ in PARTICLES:
        if not (root / stack).exists():
            mrc_stub(root / stack, 256, 256, 3, 1.2)
    save_cs(root / "J45" / "J45_008_particles.cs", particles_array())
    save_cs(root / "J45" / "J45_passthrough_particles.cs", passthrough_array())
    return root


RELION_OPTICS = """
# version 50001

data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographOriginalPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
_rlnMicrographPixelSize #7
opticsGroup1 1 0.500000 300.000000 2.700000 0.100000 1.000000
opticsGroup2 2 0.500000 300.000000 2.700000 0.100000 1.000000

"""


@pytest.fixture
def relion(tmp_path):
    """The RELION project: three micrographs (mic_c in optics group 2), binned by 2 against CryoSPARC's."""
    root = tmp_path / "relion"
    rows = [("mic_a", 1), ("mic_b", 1), ("mic_c", 2), ("mic_d", 2)]
    for name, _ in rows:
        mrc_stub(root / "MotionCorr/job002/Movies" / f"{name}.mrc", CS_SHAPE[1] // 2, CS_SHAPE[0] // 2, 1)
    text = RELION_OPTICS + "data_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnOpticsGroup #2\n_rlnDefocusU #3\n"
    text += "".join(f"MotionCorr/job002/Movies/{n}.mrc {g} 15000.0\n" for n, g in rows)
    (root / MICS).parent.mkdir(parents=True)
    (root / MICS).write_text(text)
    (root / JOB).mkdir(parents=True)
    return root


def values(**kw):
    v = CATALOG.variant().default_values(ENV)
    v.update({"pyem_env": ""})
    v.update(kw)
    return v


def run(project, vals):
    js = CATALOG.variant().make_jobstar(vals, False, JOB, None)
    (project / JOB).mkdir(parents=True, exist_ok=True)
    install_runner(project)
    cmd = f"{js.options['fn_exe']} --o {JOB} --j 1 {js.options['other_args']}"
    return subprocess.run(cmd, shell=True, cwd=project, capture_output=True, text=True, timeout=120)


def table(path, block):
    b = C.read_star(str(path))[block]
    return b.dicts()


def snapshot(root):
    """Every file under the CryoSPARC project: its size and time (nothing is to change)."""
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


# ---------------------------------------------------------------------------- direct use of the stacks

def test_the_stacks_are_linked_as_mrcs_and_the_cryosparc_files_left_alone(cs_project, relion, fake_pyem):
    before = snapshot(cs_project.parent)
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs")))
    assert r.returncode == 0, r.stdout + r.stderr
    assert snapshot(cs_project.parent) == before  # nothing copied, changed or added there
    (args,) = calls(fake_pyem)
    assert args[:3] == [str(cs_project / "J45/J45_008_particles.cs"), str(cs_project / "J45/J45_passthrough_particles.cs"),
                        JOB + CS.PYEM_STAR]  # the job's passthrough file, found beside it
    assert "--inverty" not in args and "--flipy-pose" not in args  # no RELION micrographs: pyem's default
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    names = [p["rlnImageName"] for p in parts]
    assert names == [f"000001@{JOB}Stacks/J12/extract/{UID_A}_mic_a_patch_aligned_doseweighted_particles.mrcs",
                     f"000003@{JOB}Stacks/J12/extract/{UID_A}_mic_a_patch_aligned_doseweighted_particles.mrcs",
                     f"000002@{JOB}Stacks/J12/extract/{UID_B}_mic_b_patch_aligned_doseweighted_particles.mrcs",
                     f"000001@{JOB}Stacks/J20/downsample/{UID_S}_mic_c_patch_aligned_doseweighted_particles_downsampled.mrcs"]
    for name, (_, stack, *_) in zip(names, PARTICLES):
        link = relion / name.split("@")[1]
        assert link.is_symlink() and os.readlink(link) == str(cs_project / stack)  # absolute, to the original
    # CryoSPARC's micrographs and coordinates, pyem's optics (its exposure groups, the stacks' pixel size)
    assert parts[0]["rlnMicrographName"] == cs_mic(UID_A, "mic_a")
    assert (parts[0]["rlnCoordinateX"], parts[0]["rlnCoordinateY"]) == ("1024", "2160")
    optics = table(relion / JOB / CS.PARTICLES, "optics")
    assert [(o["rlnOpticsGroup"], o["rlnImagePixelSize"], o["rlnImageSize"]) for o in optics] == \
        [("1", "1.200000", "256"), ("2", "1.200000", "256")]
    assert C.read_star(str(relion / JOB / C.OUTPUT_NODES))["output_nodes"].rows == \
        [[JOB + CS.PARTICLES, CS.NODE_PARTICLES]]
    assert (relion / JOB / C.SUCCESS).exists()
    assert "3 particle stacks linked" in r.stdout and "4 particles:" in r.stdout


def test_again_the_links_are_reused(cs_project, relion, fake_pyem):
    vals = values(in_cs=str(cs_project / "J45/J45_008_particles.cs"))
    assert run(relion, vals).returncode == 0
    assert run(relion, vals).returncode == 0
    assert len(list((relion / JOB / "Stacks").rglob("*.mrcs"))) == 3


def test_an_export_elsewhere_finds_its_stacks_in_the_project_folder_given(cs_project, relion, tmp_path, fake_pyem):
    """An exported .cs (paths with '>') copied away from the CryoSPARC project: the project folder on the form."""
    a = passthrough_array()
    a["blob/path"] = [b">" + p for p in a["blob/path"]]
    save_cs(tmp_path / "elsewhere" / "exported_particles.cs", a)
    vals = values(in_cs=str(tmp_path / "elsewhere" / "exported_particles.cs"))
    r = run(relion, vals)
    assert r.returncode != 0 and "3 of 3 particle stacks not found" in r.stderr and "CryoSPARC project" in r.stderr
    assert (relion / JOB / C.FAILURE).exists()
    r = run(relion, {**vals, "cs_project": str(cs_project)})
    assert r.returncode == 0, r.stderr
    assert calls(fake_pyem)[-1][:2] == [str(tmp_path / "elsewhere" / "exported_particles.cs"), JOB + CS.PYEM_STAR]


def test_an_export_in_the_project_finds_its_stacks_above_it(cs_project, relion, fake_pyem):
    save_cs(cs_project / "exports/groups/J45_particles/J45_particles_exported.cs", passthrough_array())
    r = run(relion, values(in_cs=str(cs_project / "exports/groups/J45_particles/J45_particles_exported.cs")))
    assert r.returncode == 0, r.stderr
    assert len(table(relion / JOB / CS.PARTICLES, "particles")) == 4


def test_uids_stripped_from_the_links(cs_project, relion, fake_pyem):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), strip_uid="Yes"))
    assert r.returncode == 0, r.stderr
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    assert parts[0]["rlnImageName"] == f"000001@{JOB}Stacks/J12/extract/mic_a_patch_aligned_doseweighted_particles.mrcs"
    assert parts[0]["rlnMicrographName"] == "J3/motioncorrected/mic_a_patch_aligned_doseweighted.mrc"


def test_a_stack_too_short_for_its_particles(cs_project, relion, fake_pyem):
    mrc_stub(cs_project / PARTICLES[0][1], 256, 256, 2)  # image 3 is asked for
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs")))
    assert r.returncode != 0 and "has 2 images, but the particles name image 3" in r.stderr


def test_csparc2star_failing_shows_its_log(cs_project, relion, fake_pyem, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs")))
    assert r.returncode != 0 and "csparc2star.py failed" in r.stderr and "KeyError: 'blob/path'" in r.stderr
    assert (relion / JOB / C.FAILURE).exists()


def test_the_flags_of_the_form(cs_project, relion, fake_pyem):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), passthrough_cs=str(
        cs_project / "J45/J45_passthrough_particles.cs"), classes="1 3", minphic="0.9", boxsize="300",
        invertx="Yes", transform="[[1,0,0],[0,1,0],[0,0,1]]", first10k="Yes", more_args="--loglevel DEBUG"))
    assert r.returncode == 0, r.stderr
    args = calls(fake_pyem)[0]
    assert " ".join(args[3:]) == ("--boxsize 300 --class 1 --class 3 --minphic 0.9 --invertx --transform "
                                  "[[1,0,0],[0,1,0],[0,0,1]] --10k --loglevel INFO --loglevel DEBUG")


# ---------------------------------------------------------------------------- re-extraction

def test_reextraction_on_the_relion_micrographs(cs_project, relion, fake_pyem):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS))
    assert r.returncode == 0, r.stdout + r.stderr
    # pyem's default: its coordinates are RELION's; the micrographs the same way up, so poses and CTF stay
    assert not {"--inverty", "--flipy", "--flipy-pose", "--flipy-ctf"} & set(calls(fake_pyem)[0])
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    assert [p["rlnMicrographName"] for p in parts] == ["MotionCorr/job002/Movies/mic_a.mrc"] * 2 + \
        ["MotionCorr/job002/Movies/mic_b.mrc", "MotionCorr/job002/Movies/mic_c.mrc"]
    # in RELION's pixels (half CryoSPARC's), Y flipped by pyem
    assert [(p["rlnCoordinateX"], p["rlnCoordinateY"]) for p in parts] == \
        [("512", "1080"), ("1024", "360"), ("1536", "720"), ("205", "144")]
    assert [p["rlnOpticsGroup"] for p in parts] == ["1", "1", "1", "2"]  # the micrographs'
    assert float(parts[0]["rlnOriginYAngst"]) > 0 and float(parts[0]["rlnDefocusAngle"]) > 0  # as CryoSPARC's
    assert parts[0]["rlnImageName"] == "000001@" + str(cs_project / PARTICLES[0][1])  # (no links)
    assert not (relion / JOB / "Stacks").exists()
    optics = table(relion / JOB / CS.PARTICLES, "optics")
    assert [o["rlnOpticsGroupName"] for o in optics] == ["opticsGroup1", "opticsGroup2"]
    assert [(o["rlnMicrographPixelSize"], o["rlnImagePixelSize"], o["rlnImageSize"]) for o in optics] == \
        [("1.000000", "1.200000", "256")] * 2
    assert "particle box size 308 px" in r.stdout and "Re-center refined coordinates?' Yes" in r.stdout
    assert "3 of 3 micrographs matched" in r.stdout and "upside-down" not in r.stdout


def test_imported_micrographs_are_matched_too(cs_project, relion, fake_pyem):
    save_cs(cs_project / "J45/J45_passthrough_particles.cs", passthrough_array(folder="imported", suffix=""))
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS))
    assert r.returncode == 0, r.stderr
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    assert parts[0]["rlnMicrographName"] == "MotionCorr/job002/Movies/mic_a.mrc"
    assert (parts[0]["rlnCoordinateX"], parts[0]["rlnCoordinateY"]) == ("512", "1080")


@pytest.mark.parametrize("mode, flip, keep_ctf, flags", [
    (CS.REEXTRACT, CS.FLIP_DEFAULT, "Yes", set()),
    (CS.REEXTRACT, CS.FLIP_CRYOSPARC, "Yes", {"--inverty"}),
    (CS.REEXTRACT, CS.FLIP_UPSIDE_DOWN, "Yes", {"--inverty", "--flipy-pose", "--flipy-ctf"}),
    (CS.REEXTRACT, CS.FLIP_UPSIDE_DOWN, "No", {"--inverty", "--flipy-pose"}),  # (the CTF is left out)
    (CS.DIRECT, CS.FLIP_DEFAULT, "Yes", set()),
    (CS.DIRECT, CS.FLIP_CRYOSPARC, "Yes", {"--inverty"}),
    (CS.DIRECT, CS.FLIP_UPSIDE_DOWN, "Yes", {"--inverty"}),  # CryoSPARC's own stacks: their poses stay
])
def test_the_y_axis_set_on_the_form(cs_project, relion, fake_pyem, mode, flip, keep_ctf, flags):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=mode, in_micrographs=MICS,
                           flip_y=flip, keep_ctf=keep_ctf))
    assert r.returncode == 0, r.stderr
    assert {"--inverty", "--flipy", "--flipy-pose", "--flipy-ctf"} & set(calls(fake_pyem)[0]) == flags
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    assert parts[0]["rlnCoordinateY"] == ("1080" if flip == CS.FLIP_DEFAULT else "360")
    mirrored = mode == CS.REEXTRACT and flip == CS.FLIP_UPSIDE_DOWN
    assert float(parts[0]["rlnOriginYAngst"]) < 0 if mirrored else float(parts[0]["rlnOriginYAngst"]) > 0


def test_cryosparcs_ctf_left_out(cs_project, relion, fake_pyem):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS, keep_ctf="No"))
    assert r.returncode == 0, r.stderr
    assert "--flipy-ctf" not in calls(fake_pyem)[0]
    b = C.read_star(str(relion / JOB / CS.PARTICLES))["particles"]
    assert not set(CS.CTF_PARTICLE) & set(b.columns)
    assert "Extract takes the micrographs'" in r.stdout


def test_2d_poses_of_upside_down_micrographs_are_mirrored_here(cs_project, relion, fake_pyem):
    save_cs(cs_project / "J45/J45_008_particles.cs", particles_array(two_d=True))
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS, flip_y=CS.FLIP_UPSIDE_DOWN))
    assert r.returncode == 0, r.stderr
    args = calls(fake_pyem)[0]
    assert "--flipy-pose" not in args and "--flipy-ctf" in args  # (pyem's pose flip needs 3D poses)
    p = table(relion / JOB / CS.PARTICLES, "particles")[0]
    assert float(p["rlnAnglePsi"]) == pytest.approx(-np.rad2deg(0.3), abs=1e-4)
    assert float(p["rlnOriginYAngst"]) == pytest.approx(-2.4, abs=1e-4)
    assert "Re-center" not in r.stdout


def test_particles_of_micrographs_relion_has_not_are_left_out(cs_project, relion, fake_pyem):
    text = (relion / MICS).read_text().replace("MotionCorr/job002/Movies/mic_b.mrc 1 15000.0\n", "")
    (relion / MICS).write_text(text)
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS))
    assert r.returncode == 0, r.stderr
    assert len(table(relion / JOB / CS.PARTICLES, "particles")) == 3
    assert "WARNING: 1 particle(s) of 1 CryoSPARC micrograph(s) have no micrograph" in r.stdout
    assert "3 particles (of 4)" in r.stdout


def test_direct_use_on_the_relion_micrographs(cs_project, relion, fake_pyem):
    """The stacks linked, the micrographs RELION's (Picks, CTF refinement); a particle whose micrograph RELION
    has not keeps CryoSPARC's."""
    text = (relion / MICS).read_text().replace("MotionCorr/job002/Movies/mic_b.mrc 1 15000.0\n", "")
    (relion / MICS).write_text(text)
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), in_micrographs=MICS))
    assert r.returncode == 0, r.stderr
    args = calls(fake_pyem)[0]
    assert "--inverty" not in args  # pyem's default: RELION's coordinates
    parts = table(relion / JOB / CS.PARTICLES, "particles")
    assert [p["rlnMicrographName"] for p in parts] == ["MotionCorr/job002/Movies/mic_a.mrc"] * 2 + \
        [cs_mic(UID_B, "mic_b"), "MotionCorr/job002/Movies/mic_c.mrc"]
    assert parts[2]["rlnCoordinateX"] == "3072"  # CryoSPARC's pixels
    assert parts[0]["rlnImageName"].startswith(f"000001@{JOB}Stacks/")
    assert [p["rlnOpticsGroup"] for p in parts] == ["1", "1", "2", "2"]  # pyem's (CryoSPARC's exposure groups)


def test_reextraction_needs_micrographs(cs_project, relion, fake_pyem):
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT))
    assert r.returncode != 0 and "re-extraction needs the RELION micrographs" in r.stderr
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), in_micrographs=MICS,
                           copy_coords="other.star"))
    assert r.returncode != 0 and "--copy-micrograph-coordinates" in r.stderr
    assert calls(fake_pyem) == []


def test_unknown_micrograph_sizes_scale_by_pixel_size(cs_project, relion, fake_pyem):
    for p in (relion / "MotionCorr/job002/Movies").iterdir():
        p.write_bytes(b"")  # not readable: the pixel sizes (0.5 / 1.0)
    r = run(relion, values(in_cs=str(cs_project / "J45/J45_008_particles.cs"), mode=CS.REEXTRACT,
                           in_micrographs=MICS))
    assert r.returncode == 0, r.stderr
    assert table(relion / JOB / CS.PARTICLES, "particles")[0]["rlnCoordinateX"] == "512"


# ---------------------------------------------------------------------------- names, without a job

@pytest.mark.parametrize("cs, passthrough", [
    ("J45/J45_008_particles.cs", "J45/J45_passthrough_particles.cs"),
    ("J45/J45_particles.cs", "J45/J45_passthrough_particles.cs"),
    ("J31/J31_particles_selected.cs", "J31/J31_passthrough_particles_selected.cs"),
    ("J31/J31_passthrough_particles_selected.cs", None),
    ("J31/J31_particles_excluded.cs", None),  # (not there)
])
def test_the_passthrough_file_of_a_job(tmp_path, cs, passthrough):
    for f in ("J45/J45_passthrough_particles.cs", "J31/J31_passthrough_particles_selected.cs"):
        (tmp_path / f).parent.mkdir(exist_ok=True)
        (tmp_path / f).write_text("")
    found = CS.passthrough_for(str(tmp_path / cs))
    assert found == (str(tmp_path / passthrough) if passthrough else None)


@pytest.mark.parametrize("cs, relion_name", [
    (f"J3/motioncorrected/{UID_A}_FoilHole_1_Data_2_3_20230101_fractions_patch_aligned_doseweighted.mrc",
     "MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101_fractions.mrc"),
    (f"J3/motioncorrected/{UID_A}_FoilHole_1_Data_2_3_20230101_fractions_patch_aligned.mrc",
     "MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101_fractions.mrc"),
    (f">J1/imported/{UID_A}_{UID_B}_FoilHole_1_Data_2_3_20230101_fractions.mrc",  # imported RELION micrographs
     "MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101_fractions.mrc"),
    (f"J3/motioncorrected/{UID_A}_FoilHole_1_Data_2_3_20230101_EER_patch_aligned_doseweighted.mrc",
     "MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101.mrc"),  # contained
    (f"J3/motioncorrected/{UID_A}_FoilHole_9_patch_aligned_doseweighted.mrc", None),  # none
])
def test_micrographs_are_matched_by_name(tmp_path, cs, relion_name):
    names = ["MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101_fractions.mrc",
             "MotionCorr/job002/Movies/FoilHole_1_Data_2_3_20230101.mrc",
             "MotionCorr/job002/Movies/FoilHole_1_Data_2_4_20230101.mrc"]
    if relion_name and relion_name not in names[:1]:
        names = names[1:]  # (the _fractions one would be contained as well)
    star = tmp_path / "mics.star"
    star.write_text("data_micrographs\nloop_\n_rlnMicrographName #1\n" + "\n".join(names) + "\n")
    assert CS.Micrographs(str(star)).match(cs) == relion_name


def test_two_relion_micrographs_of_the_same_name_match_neither(tmp_path):
    star = tmp_path / "mics.star"
    star.write_text("data_micrographs\nloop_\n_rlnMicrographName #1\nMotionCorr/job002/A/mic_a.mrc\n"
                    "MotionCorr/job004/B/mic_a.mrc\n")
    assert CS.Micrographs(str(star)).match(f"J3/motioncorrected/{UID_A}_mic_a_patch_aligned.mrc") is None


# ---------------------------------------------------------------------------- the form

def test_the_form_round_trip_and_family():
    v = CATALOG.variant()
    vals = values(in_cs="/data/CS-proj/J45/J45_008_particles.cs", mode=CS.REEXTRACT, in_micrographs="./" + MICS)
    js = v.make_jobstar(vals, False, JOB, None)
    assert js.type_label == "relion.external" and not js.is_tomo
    assert js.options["fn_exe"] == "python3 .cryo-savanna/csparc_runner.py csparc2star"
    assert csparc_kind(js.options["fn_exe"]) == CS.CSPARC2STAR and X.family_of(js) == X.CSPARC
    assert csparc_main_input(js) == X.main_input(js) == MICS
    back, ignored = v.values_from_jobstar(js, ENV)
    assert ignored == [] and all(back[k] == vals[k] for k in v.own_keys)
    assert [t.label for t in v.tabs] == ["Input", "RELION", "csparc2star", "Running"]
    groups = {r.keys[0]: r.group for t in v.tabs for r in t.rows}
    assert v.groups[groups["keep_ctf"]].active_when == CS.REEXTRACT
    assert v.groups[groups["strip_uid"]].active_when == CS.DIRECT


def test_the_micrographs_go_in_an_external_field():
    p = Pipeline()
    p.processes["CtfFind/job003/"] = Process("CtfFind/job003/", "None", "relion.ctffind", "Succeeded", outputs=[MICS])
    p.nodes[MICS] = Node(MICS, "MicrographGroupMetadata.star.relion.ctf", producer="CtfFind/job003/")
    js = CATALOG.variant().make_jobstar(values(in_micrographs=MICS), False, JOB, p)
    assert js.options["in_mic"] == MICS
    assert newest_micrographs(p) == {"in_micrographs": MICS}
    assert newest_micrographs(Pipeline()) == {}


def test_default_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "my_pyem")
    assert CsparcCatalog(DEFS, ENV).variant().default_values(ENV)["pyem_env"] == "my_pyem"
    monkeypatch.delenv(ENV_VAR)
    assert CsparcCatalog(DEFS, ENV).variant().default_values(ENV)["pyem_env"] == "pyem"


def test_last_settings(tmp_path):
    p = Pipeline()
    p.processes[JOB] = Process(JOB, "None", "relion.external", "Succeeded")
    js = CATALOG.variant().make_jobstar(values(pyem_env="/opt/envs/pyem", cs_project="/data/CS-proj"), False, JOB,
                                        None)
    (tmp_path / JOB).mkdir(parents=True)
    js.write(tmp_path / JOB / "job.star")
    assert last_settings(tmp_path, p) == {"pyem_env": "/opt/envs/pyem", "conda": "", "cs_project": "/data/CS-proj"}
    assert decode(js.options["other_args"])[0]["pyem_env"] == "/opt/envs/pyem"
    assert last_settings(tmp_path, None) == {}


def test_new_csparc_job_in_window(qtbot, spa_copy, tmp_path):
    from qtpy.QtCore import QSettings

    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(spa_copy, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    assert "CryoSPARC particles to RELION (pyem csparc2star)" in [a.text() for a in w.actions.external_menu.actions()]
    w.actions.new_csparc_job()
    ed = w.actions.editor
    assert ed.variant.kind == CS.CSPARC2STAR and not ed.variant.is_tomo
    assert ed.values()["in_micrographs"] == MICS and ed.values()["mode"] == CS.DIRECT
    assert [ed.tabs.tabText(i) for i in range(ed.tabs.count())] == ["Input", "RELION", "csparc2star", "Running"]
