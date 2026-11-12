#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its AreTomo3 jobs, and travels with it
# (with pytom_runner.py, whose helpers it uses).
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Run AreTomo3 (tilt-series alignment and tomogram reconstruction) as a RELION External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/aretomo_runner.py align

(a copy of this file in the project, beside a copy of pytom_runner.py, whose STAR, MRC, process and
module helpers it uses). RELION runs it as

    ... --o <job dir> [--in_movies|--in_mics|... F]... --j <threads> --<option> <value> ...

where the options after ``--j`` are the job's settings, written by cryo-savanna into "Additional
arguments". AreTomo3 is run as a program: this file needs nothing but the standard library (Python
3.6+: RELION may start it on a compute node with the system python3).

For each tilt series of a RELION 5 tilt-series set (from MotionCorr, CtfFind, Exclude tilt images or
an alignment job), the job stacks its motion-corrected images (sorted by tilt angle, odd/even halves
too when asked) into inputs/<name>/<name>.mrc with <name>_TLT.txt (tilt angle, acquisition order,
dose of each image), and runs AreTomo3 on it, one tilt series per GPU at a time:

- "Align with AreTomo3" (-Cmd 1): CTF estimation, alignment (global, optionally local), CTF
  correction and reconstruction. The alignment (AreTomo3's .aln) goes back into RELION 5's per-tilt
  STAR files as RELION's own AreTomo wrapper writes it (rlnTomoXTilt 0, rlnTomoYTilt the tilt,
  rlnTomoZRot the tilt axis, rlnTomoX/YShiftAngst the shifts × pixel size); tilts AreTomo3 found dark
  are left out. aligned_tilt_series.star can go into RELION's Reconstruct tomograms, and AreTomo3's
  CTF fit can go into the per-tilt rlnDefocusU/V/Angle.
- "Use the alignment of the input" (-Cmd 2): reconstruction only, from the alignment in the input STAR
  files (written into an .aln for AreTomo3; the local alignment of an earlier AreTomo3 job is kept
  when the tilts are still the same), with their CTF values (as a _CTF.txt).

AreTomo3's tomograms (aretomo/<name>/<name>_Vol.mrc, _EVN_Vol / _ODD_Vol for the halves) are listed
in tomograms.star (rlnTomoReconstructedTomogram, Half1/2, rlnTomoSize[XYZ], rlnTomoTomogramBinning)
for cryo-savanna's tomogram viewer and template matching. aretomo3.star has one row per tilt series
(status, run time, tilt axis, thickness, ...); it and the two STAR files are rewritten as tilt series
finish. Tilt series already done are skipped when the job runs again (continue).

Like every External job it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED and, before success,
RELION_OUTPUT_NODES.star. RELION_JOB_ABORT_NOW and SIGTERM/SIGINT/SIGHUP (scancel) stop AreTomo3.
"""

import array
import json
import os
import shlex
import shutil
import statistics
import struct
import sys
import time
import traceback

if __package__:  # imported from cryo-savanna
    from cryo_savanna import pytom_runner as B
else:  # run in a project: the copy of pytom_runner.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pytom_runner as B

Usage, Aborted = B.Usage, B.Aborted
yes, text, number, words, num = B.yes, B.text, B.number, B.words, B.num
_f, _rel, _touch = B._f, B._rel, B._touch

ALIGN, STACK = "align", "_stack"  # _stack: internal, builds the stacks of one tilt series
MODES = (ALIGN,)

SUMMARY = "aretomo3.star"  # one row per tilt series
ALIGNED = "aligned_tilt_series.star"
TOMOGRAMS = "tomograms.star"
INPUTS_DIR, OUT_DIR, TS_DIR, LOG_DIR = "inputs", "aretomo", "tilt_series", "logs"

NODE_SUMMARY = "ProcessData.star.aretomo3.summary"
NODE_ALIGNED = "TomogramGroupMetadata.star.relion.tomo.aretomo3.aligned"
NODE_TOMOGRAMS = "TomogramGroupMetadata.star.relion.tomo.aretomo3.tomograms"

# columns cryo-savanna adds (RELION keeps labels it does not know)
P = "savannaAreTomo"
COL_STATUS, COL_SECONDS, COL_LOG = P + "Status", P + "Seconds", P + "Log"
COL_ALN, COL_TOMOGRAM, COL_TILTS, COL_DARK = P + "Alignment", P + "Tomogram", P + "Tilts", P + "DarkTilts"
COL_AXIS, COL_THICKNESS, COL_ALPHA, COL_BETA = P + "TiltAxis", P + "ThicknessAngst", P + "AlphaOffset", P + "BetaOffset"
COL_CTF_RES = P + "CtfMaxResolution"
TOMO_NAME, TS_STAR = "rlnTomoName", "rlnTomoTiltSeriesStarFile"
ALIGN_COLS = ("rlnTomoXTilt", "rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst")
CTF_COLS = ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnCtfMaxResolution", "rlnCtfFigureOfMerit")
TOMOGRAM_COLS = ("rlnTomoReconstructedTomogram", "rlnTomoReconstructedTomogramDenoised",
                 "rlnTomoReconstructedTomogramHalf1", "rlnTomoReconstructedTomogramHalf2", "rlnTomoSizeX",
                 "rlnTomoSizeY", "rlnTomoSizeZ", "rlnTomoTomogramBinning", "rlnTomoReconstructedTomogramFile")
EVEN, ODD = "rlnMicrographNameEven", "rlnMicrographNameOdd"

# choices of the form (jobdefs/aretomo_jobs.py)
ALIGN_ARETOMO, ALIGN_INPUT = "Align with AreTomo3", "Use the alignment of the input"
AXIS_REFINE, AXIS_FIXED, AXIS_SEARCH = "Refine the nominal angle", "Keep the nominal angle", "Search the full range"
TILTCOR = {"Measure, use it for the alignment only (0)": "0", "Measure, also level the tomogram (1)": "1",
           "Off (-1)": "-1"}
CTF_IF_NONE, CTF_ALWAYS, CTF_NEVER = "When the input has none", "Always (replace the input values)", "Never"
CORR_CTF = {"Local CTF correction (1)": "1", "Phase flip (2)": "2", "None (0)": "0",
            "CTF multiplication, for template matching (3)": "3"}
RECON_SART, RECON_WBP = "SART", "WBP"
FLIP_VOL = {"xyz, flipped (-FlipVol 1)": "1", "xyz, rotated (-FlipVol 2)": "2"}

MRC_MODES = {0: "b", 1: "h", 2: "f", 6: "H", 12: "e"}  # MRC mode -> struct/array type code
_ITEM = {"b": 1, "h": 2, "f": 4, "H": 2, "e": 2}


def say(msg):
    print(" ++++ AreTomo3: " + msg, flush=True)


# ---------------------------------------------------------------------------- tilt series from RELION

def tilt_series_rows(opts):
    """The rows of the input tilt-series set to process (filtered by tomo_names)."""
    star = text(opts, "in_tiltseries")
    if not star or not os.path.isfile(star):
        raise Usage("input tilt-series STAR file not found: %r" % star)
    b = B.loop_block(star, "global")
    for col in (TOMO_NAME, TS_STAR):
        if col not in b.columns:
            raise Usage("%s: no %s (not a RELION 5 tilt-series set)" % (star, col))
    rows = b.dicts()
    names = set(words(opts, "tomo_names"))
    if names:
        missing = names - set(r[TOMO_NAME] for r in rows)
        if missing:
            raise Usage("not in %s: %s" % (star, " ".join(sorted(missing))))
        rows = [r for r in rows if r[TOMO_NAME] in names]
    if not rows:
        raise Usage("%s: no tilt series" % star)
    return star, b.columns, rows


def _image(path):
    """A file name from a STAR file (RELION's ``name:mrc`` suffix dropped)."""
    return path[:-4] if path.endswith(":mrc") else path


def per_tilt(row, sort_col):
    """(columns, tilt rows sorted by ``sort_col``) of a tilt series' own STAR file."""
    star = row[TS_STAR]
    if not os.path.isfile(star):
        raise Usage("%s: tilt-series STAR file not found: %s" % (row[TOMO_NAME], star))
    b = B.loop_block(star, row[TOMO_NAME])
    if "rlnMicrographName" not in b.columns:
        raise Usage("%s: no rlnMicrographName in %s (a tilt series before motion correction? Use the "
                    "output of MotionCorr, CtfFind or Exclude tilt images)" % (row[TOMO_NAME], star))
    if sort_col not in b.columns:
        raise Usage("%s: no %s in %s" % (row[TOMO_NAME], sort_col, star))
    tilts = b.dicts()
    for t in tilts:
        if _f(t, sort_col) is None:
            raise Usage("%s: a tilt without a %s in %s" % (row[TOMO_NAME], sort_col, star))
    tilts.sort(key=lambda t: _f(t, sort_col))
    return b.columns, tilts


def doses(tilts):
    """(acquisition order from 1, dose of each image in e/Å²) from rlnMicrographPreExposure: the
    exposure until the next image (the last one gets the median of the others); 0 without it."""
    pre = [_f(t, "rlnMicrographPreExposure") for t in tilts]
    n = len(tilts)
    if any(p is None for p in pre):
        return list(range(1, n + 1)), [0.0] * n
    order = sorted(range(n), key=lambda i: (pre[i], i))
    acq, dose = [0] * n, [0.0] * n
    steps = [pre[order[k + 1]] - pre[order[k]] for k in range(n - 1)]
    last = statistics.median(steps) if steps else 0.0
    for k, i in enumerate(order):
        acq[i] = k + 1
        dose[i] = max(0.0, steps[k] if k < len(steps) else last)
    return acq, dose


def pixel_size(row, opts, first_image):
    v = number(opts, "angpix", what="Tilt-series pixel size")
    if v:
        return v
    v = _f(row, "rlnTomoTiltSeriesPixelSize")
    if v:
        return v
    v = mrc_info(first_image)["angpix"]
    if v > 0:
        return v
    raise Usage("%s: no pixel size (rlnTomoTiltSeriesPixelSize, MRC header): give it in the job" % row[TOMO_NAME])


def optics(row, tilts):
    """(voltage kV, Cs mm, amplitude contrast) of a tilt series, from its row or its first tilt."""
    out = []
    for label, default in (("rlnVoltage", None), ("rlnSphericalAberration", None), ("rlnAmplitudeContrast", 0.07)):
        v = _f(row, label)
        if v is None and tilts:
            v = _f(tilts[0], label)
        if v is None:
            v = default
        if v is None:
            raise Usage("%s: no %s in the tilt-series set" % (row[TOMO_NAME], label))
        out.append(v)
    return out


# ---------------------------------------------------------------------------- MRC stacks

def mrc_info(path):
    """nx, ny, nz, mode, data offset, voxel size of a little-endian MRC file."""
    with open(path, "rb") as fh:
        h = fh.read(1024)
    if len(h) < 1024:
        raise Usage("%s: not an MRC file" % path)
    nx, ny, nz, mode = struct.unpack("<4i", h[:16])
    if not (0 < nx < 1 << 20 and 0 < ny < 1 << 20 and 0 < nz < 1 << 20):
        raise Usage("%s: not a little-endian MRC file" % path)
    mx = struct.unpack("<i", h[28:32])[0]
    cx = struct.unpack("<f", h[40:44])[0]
    nsymbt = struct.unpack("<i", h[92:96])[0]
    return {"nx": nx, "ny": ny, "nz": nz, "mode": mode, "offset": 1024 + max(0, nsymbt),
            "angpix": cx / mx if mx else 0.0}


def _header(nx, ny, nz, mode, angpix):
    h = bytearray(1024)
    struct.pack_into("<10i", h, 0, nx, ny, nz, mode, 0, 0, 0, nx, ny, nz)
    struct.pack_into("<3f", h, 40, nx * angpix, ny * angpix, nz * angpix)
    struct.pack_into("<3f", h, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3i", h, 64, 1, 2, 3)
    struct.pack_into("<i", h, 104, 20140)  # nversion
    h[208:212] = b"MAP "
    h[212:216] = b"\x44\x44\x00\x00"  # little-endian machine stamp
    return bytes(h)


def _convert(data, code, out_code):
    """Raw image bytes of one MRC mode as those of another (float32 for float16 / int8 input)."""
    if code == out_code:
        return data
    try:
        import numpy as np  # fast, when the python running the job has it
        dt = {"b": "<i1", "h": "<i2", "f": "<f4", "H": "<u2", "e": "<f2"}
        return np.frombuffer(data, dtype=dt[code]).astype(dt[out_code]).tobytes()
    except ImportError:
        pass
    out = array.array(out_code)
    step = 1 << 20
    size = _ITEM[code]
    for start in range(0, len(data), step * size):
        chunk = data[start:start + step * size]
        out.extend(struct.unpack("<%d%s" % (len(chunk) // size, code), chunk))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def build_stack(images, out, angpix):
    """One MRC stack of 2D images (in this order), float32 unless all of them are int16 / uint16 /
    float32 (the modes AreTomo3 reads)."""
    infos = [mrc_info(p) for p in images]
    nx, ny = infos[0]["nx"], infos[0]["ny"]
    for p, i in zip(images, infos):
        if i["nz"] != 1 or (i["nx"], i["ny"]) != (nx, ny):
            raise Usage("%s: %d×%d×%d, not a %d×%d image like the first of the tilt series"
                        % (p, i["nx"], i["ny"], i["nz"], nx, ny))
        if i["mode"] not in MRC_MODES:
            raise Usage("%s: MRC mode %d cannot be read" % (p, i["mode"]))
    modes = set(i["mode"] for i in infos)
    mode = infos[0]["mode"] if len(modes) == 1 and infos[0]["mode"] in (1, 2, 6) else 2
    tmp = "%s.%d.tmp" % (out, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(_header(nx, ny, len(images), mode, angpix))
        for p, i in zip(images, infos):
            code = MRC_MODES[i["mode"]]
            with open(p, "rb") as src:
                src.seek(i["offset"])
                data = src.read(nx * ny * _ITEM[code])
            if len(data) < nx * ny * _ITEM[code]:
                raise Usage("%s: the file is shorter than its header says" % p)
            fh.write(_convert(data, code, MRC_MODES[mode]))
    os.replace(tmp, out)


def run_stack(spec_file):
    """The internal _stack command: the stacks of one tilt series, from a JSON spec."""
    with open(spec_file) as fh:
        spec = json.load(fh)
    for s in spec["stacks"]:
        t0 = time.time()
        build_stack(s["images"], s["out"], spec["angpix"])
        say("%s: %d images in %.0f s" % (s["out"], len(s["images"]), time.time() - t0))
    return 0


# ---------------------------------------------------------------------------- AreTomo3's files

def read_aln(path):
    """AreTomo3's .aln: {"header": {key: value}, "dark": [(index, section, tilt)], "rows": [{sec, rot,
    tx, ty, tilt}], "local": [lines], "patches": n}. Section indices are made 1-based (older files
    have 0-based ones)."""
    header, dark, rows, local, in_local = {}, [], [], [], False
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("Local Alignment"):
                    in_local = True
                elif "=" in body:
                    k, v = body.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "DarkFrame":
                        w = v.split()
                        dark.append((int(w[0]), int(w[1]), float(w[2])))
                    else:
                        header[k] = v
                continue
            w = line.split()
            if in_local:
                local.append(raw.rstrip("\n"))
            elif len(w) >= 10:
                rows.append({"sec": int(w[0]), "rot": float(w[1]), "tx": float(w[3]), "ty": float(w[4]),
                             "tilt": float(w[9])})
    if rows and min(r["sec"] for r in rows) == 0 or dark and min(d[1] for d in dark) == 0:
        for r in rows:
            r["sec"] += 1
        dark = [(i, s + 1, t) for i, s, t in dark]
    try:
        patches = int(header.get("NumPatches", "0").split()[0])
    except (ValueError, IndexError):
        patches = 0
    return {"header": header, "dark": dark, "rows": rows, "local": local, "patches": patches}


def _header_number(aln, key):
    try:
        return float(aln["header"].get(key, "").split()[0])
    except (ValueError, IndexError):
        return None


def write_aln(path, size, rows, patches=0, local=(), thickness=0, alpha=0.0, beta=0.0):
    """An .aln for AreTomo3 -Cmd 2. rows: (section, tilt axis, x shift px, y shift px, tilt)."""
    out = ["# AreTomo Alignment / Priims bprmMn", "# RawSize = %d %d %d" % tuple(size),
           "# NumPatches = %d" % patches, "# AlphaOffset = %8.2f" % alpha, "# BetaOffset = %8.2f" % beta,
           "# Thickness = %d" % thickness,
           "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT"]
    for sec, rot, tx, ty, tilt in rows:
        out.append("%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  %7.2f  %7.2f  %7.2f  %8.2f"
                   % (sec, rot, 1.0, tx, ty, 1.0, 1.0, 1.0, 0.0, tilt))
    if patches:
        out.append("# Local Alignment")
        out += list(local)
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def read_ctf(path):
    """AreTomo3's _CTF.txt: [(defocus 1 Å, defocus 2 Å, azimuth °, phase shift rad, score, resolution Å)]."""
    out = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            w = line.split()
            if not w or w[0].startswith("#") or len(w) < 7:
                continue
            try:
                out.append(tuple(float(x) for x in w[1:7]))
            except ValueError:
                continue
    return out


def write_ctf(path, tilts):
    """A _CTF.txt for AreTomo3 -Cmd 2 from RELION's per-tilt CTF (None when a tilt has none)."""
    lines = ["# Columns: #1 micrograph number; #2 - defocus 1 [A]; #3 - defocus 2; #4 - azimuth of "
             "astigmatism; #5 - additional phase shift [radian]; #6 - cross correlation; #7 - spacing (in "
             "Angstroms) up to which CTF rings were fit successfully; #8 - dfHand"]
    for i, t in enumerate(tilts):
        u, v, a = _f(t, "rlnDefocusU"), _f(t, "rlnDefocusV"), _f(t, "rlnDefocusAngle")
        if u is None:
            return False
        phase = (_f(t, "rlnPhaseShift") or 0.0) * 0.017453293
        lines.append("%4d   %8.2f  %8.2f  %8.2f  %9.4f  %8.4f  %8.4f  %3d"
                     % (i + 1, u, v if v is not None else u, a or 0.0, phase, _f(t, "rlnCtfFigureOfMerit") or 0.0,
                        _f(t, "rlnCtfMaxResolution") or 0.0, 1))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return True


def write_tlt(path, angles, acq, dose):
    with open(path, "w") as fh:
        fh.write("".join("%8.2f  %4d  %8.2f\n" % (a, q, d) for a, q, d in zip(angles, acq, dose)))


# ---------------------------------------------------------------------------- one tilt series

def _paths(job_dir, name):
    inp, out = os.path.join(job_dir, INPUTS_DIR, name), os.path.join(job_dir, OUT_DIR, name)
    return {"in": inp, "out": out, "stack": os.path.join(inp, name + ".mrc"),
            "even": os.path.join(inp, name + "_EVN.mrc"), "odd": os.path.join(inp, name + "_ODD.mrc"),
            "tlt": os.path.join(inp, name + "_TLT.txt"), "in_aln": os.path.join(inp, name + ".aln"),
            "in_ctf": os.path.join(inp, name + "_CTF.txt"), "aln": os.path.join(out, name + ".aln"),
            "ctf": os.path.join(out, name + "_CTF.txt"), "vol": os.path.join(out, name + "_Vol.mrc"),
            "vol_even": os.path.join(out, name + "_EVN_Vol.mrc"), "vol_odd": os.path.join(out, name + "_ODD_Vol.mrc"),
            "star": os.path.join(job_dir, TS_DIR, name + ".star"), "spec": os.path.join(inp, "stacks.json"),
            "log": os.path.join(job_dir, LOG_DIR, name + ".log")}


def prepare(row, opts, job_dir):
    """What one tilt series needs: its tilts (sorted), pixel size, optics, halves; checks the inputs."""
    name = row[TOMO_NAME]
    use_input = text(opts, "alignment") == ALIGN_INPUT
    columns, tilts = per_tilt(row, "rlnTomoYTilt" if use_input else "rlnTomoNominalStageTiltAngle")
    if len(tilts) < 7:
        raise Usage("%s: %d tilts; AreTomo3 needs at least 7" % (name, len(tilts)))
    if use_input:
        missing = [c for c in ALIGN_COLS if c not in columns]
        if missing:
            raise Usage("%s: the input has no alignment (%s): use 'Align with AreTomo3', or a tilt-series set "
                        "of an alignment job" % (name, ", ".join(missing)))
    images = [_image(t["rlnMicrographName"]) for t in tilts]
    for p in images:
        if not os.path.isfile(p):
            raise Usage("%s: tilt image not found: %s" % (name, p))
    halves = None
    if yes(opts, "do_halves"):
        if EVEN not in columns or ODD not in columns:
            raise Usage("%s: no %s / %s in its STAR file (MotionCorr's 'Save images for denoising?'), or "
                        "switch off 'Odd/even tomograms'" % (name, EVEN, ODD))
        halves = ([_image(t[EVEN]) for t in tilts], [_image(t[ODD]) for t in tilts])
        for p in halves[0] + halves[1]:
            if not os.path.isfile(p):
                raise Usage("%s: half image not found: %s" % (name, p))
    px = pixel_size(row, opts, images[0])
    kv, cs, ac = optics(row, tilts)
    axis = number(opts, "tilt_axis", what="Tilt axis angle")
    if axis is None:
        axis = _f(tilts[0], "rlnTomoNominalTiltAxisAngle")
    if axis is None:
        axis = _f(row, "rlnTomoNominalTiltAxisAngle")
    return {"name": name, "row": row, "columns": columns, "tilts": tilts, "images": images, "halves": halves,
            "angpix": px, "kv": kv, "cs": cs, "ac": ac, "axis": axis, "paths": _paths(job_dir, name)}


def stack_spec(ts):
    """The JSON spec of the _stack command for a tilt series (written next to its stacks)."""
    p = ts["paths"]
    stacks = [{"images": ts["images"], "out": p["stack"]}]
    if ts["halves"]:
        stacks += [{"images": ts["halves"][0], "out": p["even"]}, {"images": ts["halves"][1], "out": p["odd"]}]
    return {"angpix": ts["angpix"], "stacks": stacks}


def write_inputs(ts, opts):
    """inputs/<name>/: the _TLT.txt, the stacks' spec, and for -Cmd 2 the .aln and _CTF.txt."""
    p = ts["paths"]
    for d in (p["in"], p["out"]):
        if not os.path.isdir(d):
            os.makedirs(d)
    tilts = ts["tilts"]
    acq, dose = doses(tilts)
    use_input = text(opts, "alignment") == ALIGN_INPUT
    angles = [_f(t, "rlnTomoYTilt" if use_input else "rlnTomoNominalStageTiltAngle") for t in tilts]
    write_tlt(p["tlt"], angles, acq, dose)
    with open(p["spec"], "w") as fh:
        json.dump(stack_spec(ts), fh, indent=1)
    ts["notes"] = []
    if not use_input:
        return
    # the input's alignment as an .aln (the local alignment of the AreTomo3 job it came from, if the
    # tilts are still those it aligned)
    info = mrc_info(ts["images"][0])
    px = ts["angpix"]
    rows = [(i + 1, _f(t, "rlnTomoZRot"), _f(t, "rlnTomoXShiftAngst") / px, _f(t, "rlnTomoYShiftAngst") / px,
             _f(t, "rlnTomoYTilt")) for i, t in enumerate(tilts)]
    if any(abs(_f(t, "rlnTomoXTilt") or 0.0) > 0.5 for t in tilts):
        ts["notes"].append("rlnTomoXTilt is not 0: AreTomo3 has no x tilt, it is left out")
    patches, local, thickness, alpha, beta = 0, [], 0, 0.0, 0.0
    old = ts["row"].get(COL_ALN, "")
    if old and os.path.isfile(old):
        aln = read_aln(old)
        same = len(aln["rows"]) == len(rows) and all(abs(a["tilt"] - r[4]) < 0.05 for a, r in zip(aln["rows"], rows))
        thickness = int(_header_number(aln, "Thickness") or 0)
        alpha, beta = _header_number(aln, "AlphaOffset") or 0.0, _header_number(aln, "BetaOffset") or 0.0
        if same and aln["patches"]:
            patches, local = aln["patches"], aln["local"]
            ts["notes"].append("local alignment (%d patches) from %s" % (patches, old))
        elif aln["patches"]:
            ts["notes"].append("the tilts differ from those of %s: its local alignment is not used" % old)
    if not thickness and number(opts, "vol_z", int) is None:
        raise Usage("%s: no sample thickness from an AreTomo3 alignment: give the tomogram thickness" % ts["name"])
    write_aln(p["in_aln"], (info["nx"], info["ny"], len(tilts)), rows, patches, local, thickness, alpha, beta)
    if not write_ctf(p["in_ctf"], tilts):
        if os.path.exists(p["in_ctf"]):
            os.remove(p["in_ctf"])
        ts["notes"].append("no CTF values (rlnDefocusU) in the input: the tomogram is not CTF-corrected")


def _pair_words(opts, key, what, count=2, kind=float):
    w = words(opts, key)
    if not w:
        return []
    if len(w) > count:
        raise Usage("%s: at most %d numbers (got %r)" % (what, count, text(opts, key)))
    try:
        return [num(kind(x)) for x in w]
    except ValueError:
        raise Usage("%s: numbers, e.g. %s (got %r)" % (what, " ".join(["4"] * count), text(opts, key)))


def binnings(opts):
    b = _pair_words(opts, "at_bin", "Binning", 3) or ["4"]
    if float(b[0]) < 1:
        raise Usage("Binning: 1 or more (got %s)" % b[0])
    return b


def aretomo_command(opts, ts, gpu):
    """The AreTomo3 command line for one tilt series."""
    p = ts["paths"]
    use_input = text(opts, "alignment") == ALIGN_INPUT
    cmd = words(opts, "aretomo_exe") or ["AreTomo3"]
    cmd += ["-InPrefix", p["stack"], "-InSuffix", ".mrc", "-OutDir", p["out"] + "/", "-Cmd", "2" if use_input else "1",
            "-Gpu", str(gpu), "-PixSize", num(round(ts["angpix"], 5)), "-kV", str(int(round(ts["kv"]))),
            "-Cs", num(ts["cs"]), "-AmpContrast", num(ts["ac"]), "-SplitSum", "1" if ts["halves"] else "0"]
    if not use_input:
        mode = text(opts, "tilt_axis_mode") or AXIS_REFINE
        if mode != AXIS_SEARCH:
            if ts["axis"] is None:
                raise Usage("%s: no rlnTomoNominalTiltAxisAngle: give the tilt axis angle, or search the full "
                            "range" % ts["name"])
            cmd += ["-TiltAxis", num(round(ts["axis"], 3))] + (["-1"] if mode == AXIS_FIXED else [])
        z = number(opts, "align_z", int, what="Alignment thickness")
        if z:
            cmd += ["-AlignZ", str(z)]
        patch = _pair_words(opts, "at_patch", "Local alignment patches", 2, int)
        if patch and patch != ["0", "0"] and patch != ["0"]:
            if len(patch) != 2:
                raise Usage("Local alignment patches: two numbers (x y), e.g. 4 4")
            cmd += ["-AtPatch"] + patch
        tc = TILTCOR.get(text(opts, "tilt_cor"), "0")
        offset = number(opts, "tilt_offset", what="Tilt angle offset")
        cmd += ["-TiltCor", tc] + ([num(offset)] if offset is not None and tc != "-1" else [])
        dark = number(opts, "dark_tol", what="Dark image tolerance")
        if dark is not None:
            cmd += ["-DarkTol", num(dark)]
        phase = _pair_words(opts, "ext_phase", "Phase shift and search range", 2)
        if phase:
            cmd += ["-ExtPhase"] + phase
    vol_z = number(opts, "vol_z", int, what="Tomogram thickness")
    cmd += ["-VolZ", str(vol_z if vol_z else -1)]
    ext = number(opts, "ext_z", int, what="Extra thickness")
    if not vol_z and ext is not None:
        cmd += ["-ExtZ", str(ext)]
    cmd += ["-AtBin"] + binnings(opts)
    cmd += ["-CorrCTF", CORR_CTF.get(text(opts, "corr_ctf"), "1")]
    if text(opts, "recon_method") == RECON_WBP:
        cmd += ["-Wbp", "1"]
    else:
        sart = _pair_words(opts, "sart", "SART iterations and projections per subset", 2, int)
        if sart:
            cmd += ["-Sart"] + sart
    rng = _pair_words(opts, "recon_range", "Tilt range for the reconstruction", 2)
    if rng:
        if len(rng) != 2:
            raise Usage("Tilt range for the reconstruction: two angles, e.g. -60 60")
        cmd += ["-ReconRange"] + rng
    cmd += ["-FlipVol", FLIP_VOL.get(text(opts, "flip_vol"), "1")]
    if yes(opts, "flip_int"):
        cmd += ["-FlipInt", "1"]
    for path in (p["stack"], p["out"]):
        if len(path) > 200:
            raise Usage("%s: AreTomo3 keeps file names in 256 characters: %s is too long" % (ts["name"], path))
    return cmd + words(opts, "aretomo_args")


def collect(ts, opts):
    """After AreTomo3: the tilt series' STAR file with the alignment (and CTF), and its row values."""
    p, tilts, px = ts["paths"], ts["tilts"], ts["angpix"]
    use_input = text(opts, "alignment") == ALIGN_INPUT
    cols = list(ts["columns"])
    info = {}
    if use_input:
        kept = [dict(t) for t in tilts]
        info.update({"aln": _rel(p["in_aln"]), "dark": 0})
        aln = read_aln(p["in_aln"])
    else:
        if not os.path.isfile(p["aln"]):
            raise Usage("%s: AreTomo3 wrote no alignment (%s)" % (ts["name"], p["aln"]))
        aln = read_aln(p["aln"])
        by_sec = dict((r["sec"], r) for r in aln["rows"])
        for c in ALIGN_COLS:
            if c not in cols:
                cols.append(c)
        kept, sections = [], []
        for i, t in enumerate(tilts):
            r = by_sec.get(i + 1)
            if r is None:
                continue  # dark: AreTomo3 left it out
            t = dict(t)
            t.update({"rlnTomoXTilt": 0.0, "rlnTomoYTilt": r["tilt"], "rlnTomoZRot": r["rot"],
                      "rlnTomoXShiftAngst": r["tx"] * px, "rlnTomoYShiftAngst": r["ty"] * px})
            kept.append(t)
            sections.append(i)
        if not kept:
            raise Usage("%s: no tilt in AreTomo3's alignment %s" % (ts["name"], p["aln"]))
        info.update({"aln": _rel(p["aln"]), "dark": len(tilts) - len(kept)})
        if info["dark"]:
            say("%s: %d dark tilt(s) left out: %s" % (ts["name"], info["dark"], " ".join(
                "%.1f" % _f(t, "rlnTomoNominalStageTiltAngle") for i, t in enumerate(tilts) if i not in sections)))
        _ctf_into(ts, opts, kept, sections, cols, aln, info)
    info["axis"] = statistics.mean(r["rot"] for r in aln["rows"]) if aln["rows"] else None
    thick = _header_number(aln, "Thickness")
    info["thickness"] = thick * px if thick else None
    info["alpha"], info["beta"] = _header_number(aln, "AlphaOffset"), _header_number(aln, "BetaOffset")
    info["tilts"] = len(kept)
    d = os.path.dirname(p["star"])
    if not os.path.isdir(d):
        os.makedirs(d)
    B.write_star(p["star"], [(ts["name"], None, cols, [[t.get(c, "") for c in cols] for t in kept])])
    return info


def _ctf_into(ts, opts, kept, sections, cols, aln, info):
    """AreTomo3's CTF fit in the kept tilts, as the form asks (When the input has none / Always / Never)."""
    policy = text(opts, "ctf_to_star") or CTF_IF_NONE
    p = ts["paths"]
    if not os.path.isfile(p["ctf"]):
        return
    fits = read_ctf(p["ctf"])
    if len(fits) == len(ts["tilts"]):  # one per input image
        mine = [fits[i] for i in sections]
    elif len(fits) == len(aln["rows"]) == len(kept):  # one per image after dark ones went
        mine = fits
    else:
        say("%s: %d CTF fits for %d tilts: AreTomo3's CTF is not used" % (ts["name"], len(fits), len(ts["tilts"])))
        return
    res = sorted(f[5] for f in mine if f[5] > 0)
    info["ctf_res"] = res[len(res) // 2] if res else None
    has = "rlnDefocusU" in ts["columns"] and all(_f(t, "rlnDefocusU") for t in kept)
    if policy == CTF_NEVER or (policy == CTF_IF_NONE and has):
        return
    phase = any(abs(f[3]) > 1e-6 for f in mine)
    for c in CTF_COLS + (("rlnPhaseShift",) if phase else ()):
        if c not in cols:
            cols.append(c)
    for t, (d1, d2, az, ph, score, r) in zip(kept, mine):
        t.update({"rlnDefocusU": d1, "rlnDefocusV": d2, "rlnDefocusAngle": az, "rlnCtfMaxResolution": r,
                  "rlnCtfFigureOfMerit": score})
        if phase:
            t["rlnPhaseShift"] = ph / 0.017453293


def tomogram_values(ts, opts):
    """The tomograms.star values of a tilt series: tomograms, sizes in tilt-series pixels, binning."""
    p = ts["paths"]
    if not os.path.isfile(p["vol"]):
        raise Usage("%s: AreTomo3 wrote no tomogram (%s)" % (ts["name"], p["vol"]))
    v = mrc_info(p["vol"])
    binning = float(binnings(opts)[0])
    if v["angpix"] > 0 and ts["angpix"]:
        binning = round(v["angpix"] / ts["angpix"], 3) or binning
    out = {"rlnTomoReconstructedTomogram": _rel(p["vol"]), "rlnTomoTomogramBinning": num(binning),
           "rlnTomoSizeX": str(int(round(v["nx"] * binning))), "rlnTomoSizeY": str(int(round(v["ny"] * binning))),
           "rlnTomoSizeZ": str(int(round(v["nz"] * binning))), "rlnTomoTiltSeriesPixelSize": num(round(ts["angpix"], 5))}
    if ts["halves"] and os.path.isfile(p["vol_even"]) and os.path.isfile(p["vol_odd"]):
        out["rlnTomoReconstructedTomogramHalf1"] = _rel(p["vol_even"])
        out["rlnTomoReconstructedTomogramHalf2"] = _rel(p["vol_odd"])
    return out


def remove_stacks(ts):
    for k in ("stack", "even", "odd"):
        f = ts["paths"][k]
        if os.path.exists(f):
            os.remove(f)


# ---------------------------------------------------------------------------- the job's tables

def write_tables(job_dir, opts, columns, rows, state):
    """aretomo3.star (every tilt series), aligned_tilt_series.star and tomograms.star (those done)."""
    extra = [COL_STATUS, COL_SECONDS, COL_LOG, COL_ALN, COL_TOMOGRAM, COL_TILTS, COL_DARK, COL_AXIS,
             COL_THICKNESS, COL_ALPHA, COL_BETA, COL_CTF_RES]
    summary_cols = [TOMO_NAME] + extra
    table = []
    for r in rows:
        s = state[r[TOMO_NAME]]
        i = s.get("info", {})

        def v(x, fmt="%.2f"):
            return "" if x is None else fmt % x
        table.append([r[TOMO_NAME], s["status"], "%.0f" % s.get("seconds", 0), s["log"], i.get("aln", ""),
                      s.get("tomo", {}).get("rlnTomoReconstructedTomogram", ""), i.get("tilts", ""), i.get("dark", ""),
                      v(i.get("axis")), v(i.get("thickness"), "%.1f"), v(i.get("alpha")), v(i.get("beta")),
                      v(i.get("ctf_res"))])
    general = {P + "TiltSeriesFile": text(opts, "in_tiltseries"), P + "Alignment": text(opts, "alignment") or ALIGN_ARETOMO}
    B.write_star(os.path.join(job_dir, SUMMARY), [("general", general, None, None), ("global", None, summary_cols, table)])

    done = [r for r in rows if state[r[TOMO_NAME]]["status"] == "done"]
    base = [c for c in columns if c not in TOMOGRAM_COLS and not c.startswith("savannaPytom") and c != COL_ALN]
    for c in ("rlnTomoTiltSeriesPixelSize",):
        if c not in base:
            base.append(c)
    use_input = text(opts, "alignment") == ALIGN_INPUT

    def rows_of(cols, add):
        out = []
        for r in done:
            s = state[r[TOMO_NAME]]
            vals = dict(r)
            vals[TS_STAR] = _rel(s["star"])
            vals["rlnTomoTiltSeriesPixelSize"] = s.get("tomo", {}).get("rlnTomoTiltSeriesPixelSize",
                                                                       vals.get("rlnTomoTiltSeriesPixelSize", ""))
            vals[COL_ALN] = s.get("info", {}).get("aln", "")
            if add:
                vals.update(s.get("tomo", {}))
            out.append([vals.get(c, "") for c in cols])
        return out
    if not use_input:
        cols = base + [COL_ALN]
        B.write_star(os.path.join(job_dir, ALIGNED), [("global", None, cols, rows_of(cols, False))])
    halves = any("rlnTomoReconstructedTomogramHalf1" in state[r[TOMO_NAME]].get("tomo", {}) for r in done)
    cols = base + [COL_ALN, "rlnTomoReconstructedTomogram", "rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ",
                   "rlnTomoTomogramBinning"]
    if halves:
        cols += ["rlnTomoReconstructedTomogramHalf1", "rlnTomoReconstructedTomogramHalf2"]
    B.write_star(os.path.join(job_dir, TOMOGRAMS), [("global", None, cols, rows_of(cols, True))])


# ---------------------------------------------------------------------------- the job

def check_executable(opts):
    exe = (words(opts, "aretomo_exe") or ["AreTomo3"])[0]
    if os.sep in exe:
        ok = os.path.isfile(exe) and os.access(exe, os.X_OK)
    else:
        ok = shutil.which(exe) is not None
    if not ok:
        raise Usage("AreTomo3 not found: %r (set 'AreTomo3 executable' on the Compute tab, or load its module "
                    "in 'Modules')" % exe)


def run_align(sup, job_dir, threads, opts):
    star, columns, rows = tilt_series_rows(opts)
    for d in (INPUTS_DIR, OUT_DIR, TS_DIR, LOG_DIR):
        p = os.path.join(job_dir, d)
        if not os.path.isdir(p):
            os.makedirs(p)
    check_executable(opts)
    use_input = text(opts, "alignment") == ALIGN_INPUT
    series, state = {}, {}
    for r in rows:  # every tilt series' inputs first: a wrong setting shows before hours of processing
        ts = prepare(r, opts, job_dir)
        name = ts["name"]
        series[name] = ts
        p = ts["paths"]
        aretomo_command(opts, ts, 0)  # checks the options
        state[name] = {"status": "pending", "log": _rel(p["log"]), "star": p["star"]}
        if yes(opts, "skip_done") and os.path.isfile(p["vol"]) and (use_input or os.path.isfile(p["aln"])):
            try:
                state[name].update({"info": collect(ts, opts), "tomo": tomogram_values(ts, opts), "status": "done"})
                say("%s: already done, skipped" % name)
            except Usage as e:
                say("%s: done before, but %s: running it again" % (name, e))
    write_tables(job_dir, opts, columns, rows, state)

    ids = B.gpu_ids(opts)
    per_gpu = max(1, number(opts, "per_gpu", int, what="Tilt series per GPU") or 1)
    slots = [g for g in ids for _ in range(per_gpu)]
    todo = [r[TOMO_NAME] for r in rows if state[r[TOMO_NAME]]["status"] != "done"]
    say("%d tilt series to process, GPU(s) %s, %d at a time" % (len(todo), " ".join(map(str, ids)), len(slots)))

    def task(name):
        def factory(gpu):
            def gen():
                ts, s = series[name], state[name]
                s["status"], t0 = "running", time.time()
                write_tables(job_dir, opts, columns, rows, state)
                write_inputs(ts, opts)
                for note in ts["notes"]:
                    say("%s: %s" % (name, note))
                rc = yield [sys.executable, os.path.abspath(__file__), STACK, ts["paths"]["spec"]], s["log"]
                if rc != 0:
                    B._show_tail(s["log"])
                    raise Usage("%s: making the tilt-series stack failed (log: %s)" % (name, s["log"]))
                say("%s: AreTomo3 on GPU %s" % (name, gpu))
                rc = yield aretomo_command(opts, ts, gpu), s["log"]
                s["seconds"] = time.time() - t0
                if rc != 0:
                    B._show_tail(s["log"])
                    raise Usage("%s: AreTomo3 exited with %s (log: %s)" % (name, rc, s["log"]))
                try:
                    s["info"] = collect(ts, opts)
                    s["tomo"] = tomogram_values(ts, opts)
                except Usage:
                    B._show_tail(s["log"])
                    raise
                if not yes(opts, "keep_stacks"):
                    remove_stacks(ts)
                return True
            return gen()
        return name, factory

    def changed(name, result):
        s = state[name]
        s["status"] = "done" if result else "failed"
        say("%s: %s in %.1f min" % (name, s["status"], s.get("seconds", 0) / 60))
        write_tables(job_dir, opts, columns, rows, state)

    try:
        B.run_parallel(sup, [task(n) for n in todo], slots, changed)
    except Aborted:
        for s in state.values():
            if s["status"] == "running":
                s["status"] = "aborted"
        write_tables(job_dir, opts, columns, rows, state)
        raise
    failed = [n for n, s in state.items() if s["status"] != "done"]
    if failed:
        raise Usage("%d tilt series failed: %s. Continue the job to run only those again."
                    % (len(failed), " ".join(failed)))
    nodes = [(os.path.join(job_dir, TOMOGRAMS), NODE_TOMOGRAMS), (os.path.join(job_dir, SUMMARY), NODE_SUMMARY)]
    if not use_input:
        nodes.insert(0, (os.path.join(job_dir, ALIGNED), NODE_ALIGNED))
    return nodes


def run(argv):
    mode, job_dir, threads, opts = B.parse(argv, MODES)
    B.say = say  # in this process, the helpers' messages (running: ..., module load ...) are AreTomo3's
    for f in (B.SUCCESS, B.FAILURE, B.ABORTED):
        if os.path.exists(os.path.join(job_dir, f)):
            os.remove(os.path.join(job_dir, f))
    sup = B.Supervisor(job_dir)
    say("cryo-savanna AreTomo3 %s, job %s" % (mode, job_dir))
    try:
        B.load_modules(opts)
        nodes = run_align(sup, job_dir, threads, opts)
    except Aborted:
        _touch(os.path.join(job_dir, B.ABORTED))
        return 128 + sup.stopped if sup.stopped else 1
    except Usage as e:
        print("ERROR: " + str(e), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, B.FAILURE))
        return 1
    B.write_output_nodes(job_dir, [(_rel(p), t) for p, t in nodes])
    say("done")
    _touch(os.path.join(job_dir, B.SUCCESS))
    return 0


def main():
    argv = sys.argv[1:]
    if argv[:1] == [STACK]:
        try:
            sys.exit(run_stack(argv[1]))
        except (Usage, OSError, ValueError, KeyError, IndexError) as e:
            print("ERROR: %s" % e, file=sys.stderr, flush=True)
            sys.exit(1)
    try:
        sys.exit(run(argv))
    except Usage as e:
        print("cryo-savanna aretomo_runner: " + str(e), file=sys.stderr, flush=True)
        B._fail(argv)
        sys.exit(2)
    except Exception:  # anything else: still an exit file, or RELION shows the job running for ever
        traceback.print_exc()
        sys.stderr.flush()
        B._fail(argv)
        sys.exit(1)


if __name__ == "__main__":
    main()
