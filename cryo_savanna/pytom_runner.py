#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its pytom jobs, and travels with it
# (with runner_common.py, whose helpers it uses).
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Run pytom-match-pick (template matching in tomograms) as a RELION External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/pytom_runner.py match|extract

(a copy of this file in the project, beside a copy of runner_common.py, the helpers it shares with the other
runners). RELION runs it as

    ... --o <job dir> [--in_movies|--in_3dref|--in_mask|... F]... --j <threads> --<option> <value> ...

where the options after ``--j`` are the job's settings, written by cryo-savanna into "Additional
arguments" (all of them, ``--key value``, shell-quoted). The ``--in_*`` inputs are only there so RELION
links the job to its inputs. pytom is not imported: its programs are run (``conda run -n <env>`` when an
environment is given), so this file is only a wrapper and needs nothing but the standard library
(Python 3.6+: RELION may start it on a compute node with the system python3).

match: for each tomogram of a RELION 5 tomograms.star, writes pytom's inputs from the tilt-series STAR
files (tilt angles, accumulated dose, defocus per tilt, voltage, Cs, amplitude contrast and the
tomogram's voxel size) in inputs/<tomogram>/, optionally prepares the template
(pytom_create_template.py) and the mask (pytom_create_template.py on a mask file, or
pytom_create_mask.py), and runs pytom_match_template.py. Several GPUs: either one tomogram per GPU at
a time (no MPI is needed: each pytom process gets one GPU), or every tomogram split over all of them
(pytom's own -g). pytom_match.star (one row per tomogram, the tomograms.star columns and the score
maps, with the status of each tomogram) is rewritten as tomograms finish, so the results can be looked
at while the job runs. Tomograms already done are skipped when the job is run again (continue).

extract: runs pytom_extract_candidates.py on the tomograms of a match job (in this job's own directory:
the match job's score maps are linked, its job.json copied with output_dir changed), with pytom's
automatic cut-off (number of false positives), or a fixed one; optionally also further candidates below
it (to choose thresholds interactively in cryo-savanna) and pytom_estimate_roc.py. The candidates go into
candidates.star (RELION 5: rlnTomoName, rlnCenteredCoordinate[XYZ]Angst, angles, rlnLCCmax, ...), the
thresholds per tomogram into thresholds.star, the particles above them into particles.star (with
optimisation_set.star) for RELION 5's Extract subtomos. write_selection() rewrites particles.star for
new thresholds (cryo-savanna's threshold viewer calls it). thresholds.star also keeps, per tomogram, the
maximum number of particles it was extracted with and a key of the other settings: when the job runs
again (continue), only the tomograms whose settings changed are extracted again (e.g. a larger maximum
for one tomogram, tomo_n_particles, set in the threshold viewer); the others keep their candidates and
the thresholds chosen for them.

Like every External job it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED and, before success,
RELION_OUTPUT_NODES.star. RELION_JOB_ABORT_NOW and SIGTERM/SIGINT/SIGHUP (scancel) stop the pytom
processes (their process groups: SIGTERM, then SIGKILL).
"""

import hashlib
import json
import os
import re
import shlex
import sys
import time

if __package__:  # imported from cryo-savanna
    from cryo_savanna import runner_common as C
else:  # run in a project: the copy of runner_common.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import runner_common as C  # type: ignore[no-redef]

# The helpers shared with the other runners (runner_common.py), under the names they had here before:
# cryo-savanna's viewers use them from this module, and so do the copies of aretomo_runner.py and
# mask_runner.py that projects kept from before runner_common.py.
EXTERNAL_INPUTS, SUCCESS, FAILURE, ABORTED = C.EXTERNAL_INPUTS, C.SUCCESS, C.FAILURE, C.ABORTED
ABORT_NOW, OUTPUT_NODES, KILL_GRACE_S, POLL_S = C.ABORT_NOW, C.OUTPUT_NODES, C.KILL_GRACE_S, C.POLL_S
TOMO_NAME, TOMO_MASK_SUFFIX, TOMOGRAM_KINDS = C.TOMO_NAME, C.TOMO_MASK_SUFFIX, C.TOMOGRAM_KINDS
Usage, Aborted, say = C.Usage, C.Aborted, C.say
yes, text, number, words, num, selected_names = C.yes, C.text, C.number, C.words, C.num, C.selected_names
MODULE_INIT, load_modules = C.MODULE_INIT, C.load_modules
Block, read_star, loop_block, write_star, _fmt = C.Block, C.read_star, C.loop_block, C.write_star, C._fmt
mrc_header, Supervisor, run_parallel = C.mrc_header, C.Supervisor, C.run_parallel
_touch, _rel, _link, write_output_nodes = C._touch, C._rel, C._link, C.write_output_nodes
_f, tomogram_rows, tomogram_of, gpu_ids = C._f, C.tomogram_rows, C.tomogram_of, C.gpu_ids
_first_error, _show_tail, _fail = C._first_error, C._show_tail, C._fail

WHO = "pytom"  # say(): " ++++ pytom: ..."
MATCH, EXTRACT = "match", "extract"


# files of the jobs
SUMMARY = "pytom_match.star"  # match: one row per tomogram
CANDIDATES = "candidates.star"  # extract: every candidate, RELION 5 coordinates
THRESHOLDS = "thresholds.star"  # extract: threshold per tomogram
PARTICLES = "particles.star"  # extract: the candidates above the thresholds
OPTIMISATION_SET = "optimisation_set.star"
TM_DIR, INPUTS_DIR, LOG_DIR, PLOTS_DIR = "tm", "inputs", "logs", "plots"

# node types of the outputs (RELION_OUTPUT_NODES.star)
NODE_SUMMARY = "ProcessData.star.pytom.match"
NODE_TEMPLATE = "DensityMap.mrc.pytom.template"
NODE_MASK = "Mask3D.mrc.pytom.mask"
NODE_PARTICLES = "ParticleGroupMetadata.star.relion.tomo.pytom"
NODE_OPTIMISATION = "TomoOptimisationSet.star.relion.tomo.pytom"
NODE_CANDIDATES = "ProcessData.star.pytom.candidates"
NODE_THRESHOLDS = "ProcessData.star.pytom.thresholds"

# columns cryo-savanna adds (not RELION's: RELION keeps labels it does not know)
P = "savannaPytom"
COL_TOMOGRAM, COL_VOXEL = P + "Tomogram", P + "VoxelSize"
COL_SCORES, COL_ANGLES, COL_JOB = P + "Scores", P + "Angles", P + "JobFile"
COL_STATUS, COL_SECONDS, COL_LOG = P + "Status", P + "Seconds", P + "Log"
COL_AUTO, COL_THRESHOLD = P + "AutoCutOff", P + "Threshold"
COL_CANDIDATES, COL_SELECTED = P + "Candidates", P + "Selected"
COL_MAX, COL_SETTINGS = P + "MaxParticles", P + "Settings"
CENTERED = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst")
LCC = "rlnLCCmax"
THRESHOLD_COLUMNS = [TOMO_NAME, COL_AUTO, COL_THRESHOLD, COL_CANDIDATES, COL_SELECTED, COL_MAX, COL_SETTINGS]

ORIENTATIONS = {  # how the candidates' angles are written
    "rlnAngleRot/Tilt/Psi": ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"),
    "rlnTomoSubtomogramRot/Tilt/Psi": ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"),
    "Not written": (),
}
GPU_PER_TOMOGRAM, GPU_SPLIT = "One tomogram per GPU", "Each tomogram over all GPUs"
MASK_FILE, MASK_CREATE = "Input file", "Create sphere or ellipsoid"
CUT_AUTO, CUT_FIXED = "Automatic (false positives)", "Fixed cut-off"
TMASK_JOB, TMASK_OTHER, TMASK_IGNORE = "The match job's", "Other mask(s)", "None (ignore masks)"

# clips a resampled mask to [0, 1] (run with the python of the pytom environment). The mask is never
# inverted: only the template's contrast is (pytom's mask is 1 on the particle, whatever the contrast)
MASK_FIX = ("import sys, numpy as np, mrcfile\n"
            "with mrcfile.open(sys.argv[1], mode='r+') as m:\n"
            "    m.set_data(np.clip(np.asarray(m.data, dtype=np.float32), 0, 1))\n")


# ---------------------------------------------------------------------------- arguments

def parse(argv, modes=(MATCH, EXTRACT)):
    return C.parse(argv, modes)


def conda_prefix(opts):
    """The start of every pytom command: ``conda run`` in the job's environment (nothing: on PATH)."""
    env = text(opts, "pytom_env")
    if not env:
        return []
    conda = text(opts, "conda") or os.environ.get("CONDA_EXE") or "conda"
    return shlex.split(conda) + ["run", "--no-capture-output", "-p" if "/" in env else "-n", env]


def per_tomogram(opts, key, kind=int, what=""):
    """{tomogram: value} of an option written as "name=value name=value ..."."""
    out = {}
    for w in words(opts, key):
        name, sep, v = w.rpartition("=")
        try:
            val = kind(v)
        except ValueError:
            val = None
        if not sep or not name or val is None or val <= 0:
            raise Usage("%s: %r is not <tomogram>=<%s>" % (what or key, w, "whole number" if kind is int
                                                                         else "number"))
        out[name] = val
    return out


# ---------------------------------------------------------------------------- match: inputs from RELION


def voxel_size(row, opts, tomogram):
    v = number(opts, "voxel_size", what="Tomogram voxel size")
    if v:
        return v
    px, binning = _f(row, "rlnTomoTiltSeriesPixelSize"), _f(row, "rlnTomoTomogramBinning")
    if px and binning:
        return px * binning
    v = mrc_header(tomogram)[3]
    if v > 0:
        return v
    raise Usage("%s: no voxel size (rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning, MRC header): "
                "give it in the job" % row[TOMO_NAME])


def tilt_table(row, opts):
    """Per tilt (sorted by tilt angle): (tilt angle, accumulated dose, defocus in µm or None)."""
    star = row.get("rlnTomoTiltSeriesStarFile", "")
    if not star or not os.path.isfile(star):
        raise Usage("%s: tilt-series STAR file not found: %r" % (row[TOMO_NAME], star))
    b = loop_block(star, row[TOMO_NAME])
    aligned = text(opts, "tilt_source").startswith("Aligned")
    angle_col = "rlnTomoYTilt" if aligned else "rlnTomoNominalStageTiltAngle"
    if angle_col not in b.columns:
        raise Usage("%s: no %s in %s" % (row[TOMO_NAME], angle_col, star))
    out = []
    for t in b.dicts():
        dose = _f(t, "rlnMicrographPreExposure")
        u, v = _f(t, "rlnDefocusU"), _f(t, "rlnDefocusV")
        defocus = None if u is None else ((u + (v if v is not None else u)) / 2) / 1e4
        angle = _f(t, angle_col)
        if angle is None:
            raise Usage("%s: a tilt without a %s (%r) in %s" % (row[TOMO_NAME], angle_col, t.get(angle_col, ""), star))
        out.append((angle, dose, defocus))
    out.sort(key=lambda r: r[0])
    return star, out


def ctf_values(row, star):
    """(voltage kV, Cs mm, amplitude contrast) from the tomogram set (else the tilt table's first row)."""
    first = {}
    vals = []
    for label in ("rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast"):
        v = _f(row, label)
        if v is None:
            if not first:
                first = (loop_block(star, row[TOMO_NAME]).dicts() or [{}])[0]
            v = _f(first, label)
        if v is None:
            raise Usage("%s: no %s for the CTF (switch off 'Use the CTF')" % (row[TOMO_NAME], label))
        vals.append(v)
    return vals


def write_lines(path, values):
    with open(path, "w") as fh:
        fh.write("\n".join(num(v) if isinstance(v, (int, float)) else str(v) for v in values) + "\n")


def prepare_tomogram(row, opts, job_dir):
    """pytom's per-tomogram inputs in inputs/<name>/; returns what the command needs."""
    name = row[TOMO_NAME]
    d = os.path.join(job_dir, INPUTS_DIR, name)
    if not os.path.isdir(d):
        os.makedirs(d)
    tomogram = tomogram_of(row, opts)
    link = os.path.join(d, name + ".mrc")  # pytom names its outputs after the tomogram's file
    tomo_in = link if _link(tomogram, link) else tomogram
    stem = os.path.splitext(os.path.basename(tomo_in))[0]
    voxel = voxel_size(row, opts, tomogram)
    star, tilts = tilt_table(row, opts)
    files = {"tomogram": tomo_in, "source": tomogram, "stem": stem, "voxel": voxel}
    files["tilts"] = os.path.join(d, name + ".tlt")
    write_lines(files["tilts"], [t[0] for t in tilts])
    if yes(opts, "use_dose"):
        if any(t[1] is None for t in tilts):
            raise Usage("%s: no rlnMicrographPreExposure in %s (switch off 'Use the accumulated dose')" % (name, star))
        files["dose"] = os.path.join(d, name + "_dose.txt")
        write_lines(files["dose"], [t[1] for t in tilts])
    if yes(opts, "use_ctf"):
        if any(t[2] is None for t in tilts):
            raise Usage("%s: no rlnDefocusU in %s (run CTF estimation, or switch off 'Use the CTF')" % (name, star))
        files["defocus"] = os.path.join(d, name + "_defocus.txt")
        write_lines(files["defocus"], ["%.4f" % t[2] for t in tilts])
        files["ctf"] = ctf_values(row, star)
    mask = tomogram_mask_file(opts, name)
    if mask:
        tx, ty, tz = mrc_header(tomogram)[:3]
        mx, my, mz = mrc_header(mask)[:3]
        if (mx, my, mz) != (tx, ty, tz):
            raise Usage("%s: the tomogram mask %s is %d×%d×%d, the tomogram %d×%d×%d: they need the same size"
                        % (name, mask, mx, my, mz, tx, ty, tz))
        files["tomogram_mask"] = mask
    return files


def tomogram_mask_file(opts, name):
    """The tomogram mask of tomogram ``name`` (option tomogram_mask), else "": in a folder,
    <name>_mask.mrc (TOMO_MASK_SUFFIX, as cryo-savanna's Tomogram masks job writes them); a path with
    {name}, that path for the tomogram; else one file for every tomogram."""
    value = text(opts, "tomogram_mask")
    if not value:
        return ""
    if "{name}" in value:
        mask = value.replace("{name}", name)
    elif os.path.isdir(value):
        mask = os.path.join(value, name + TOMO_MASK_SUFFIX)
        if not os.path.isfile(mask):
            raise Usage("%s: no tomogram mask %s in the masks folder %s" % (name, name + TOMO_MASK_SUFFIX, value))
    else:
        mask = value
    if not os.path.isfile(mask):
        raise Usage("%s: tomogram mask not found: %s" % (name, mask))
    return mask


def _numbers(value):
    """Numbers separated by spaces, commas, colons or a dash between them (0 200, 0,200, 0:200, 0-200)."""
    return [w for w in re.split(r"[\s,:;]+|(?<=\d)-(?=\d)", value.strip()) if w]


def _pair(opts, key, what):
    value = text(opts, key)
    w = _numbers(value)
    if not w:
        return []
    try:
        if len(w) != 2:
            raise ValueError
        start, end = (int(round(float(x))) for x in w)
    except ValueError:
        raise Usage("%s: give a start and an end voxel, e.g. 0 200 (got %r)" % (what, value))
    if start < 0 or end <= start:
        raise Usage("%s: the end must be larger than the start, and both 0 or more (got %r)" % (what, value))
    return [str(start), str(end)]


def match_command(opts, files, template, mask, gpus, dest):
    cmd = conda_prefix(opts) + ["pytom_match_template.py", "-t", template, "-m", mask, "-v", files["tomogram"],
                                "-d", dest, "-g"] + [str(g) for g in gpus]
    cmd += ["-a", files["tilts"], "--voxel-size-angstrom", num(files["voxel"])]
    if yes(opts, "per_tilt_weighting"):
        cmd.append("--per-tilt-weighting")
    if "dose" in files:
        cmd += ["--dose-accumulation", files["dose"]]
    if "defocus" in files:
        kv, cs, ac = files["ctf"]
        cmd += ["--defocus", files["defocus"], "--amplitude-contrast", num(ac), "--spherical-aberration", num(cs),
                "--voltage", num(kv)]
        phase = number(opts, "phase_shift", what="Phase shift")
        if phase:
            cmd += ["--phase-shift", num(phase)]
    if text(opts, "tomogram_ctf_model") == "phase-flip":
        cmd += ["--tomogram-ctf-model", "phase-flip"]
    hand = text(opts, "defocus_handedness") or "0"
    if hand != "0":
        cmd += ["--defocus-handedness", hand]
    if yes(opts, "spectral_whitening"):
        cmd.append("--spectral-whitening")
    for key, flag in (("low_pass", "--low-pass"), ("high_pass", "--high-pass")):
        v = number(opts, key)
        if v:
            cmd += [flag, num(v)]
    diameter = number(opts, "particle_diameter", what="Particle diameter")
    angles = text(opts, "angular_search")
    if not diameter and not angles:
        raise Usage("give the particle diameter or the angular search")
    if diameter:
        cmd += ["--particle-diameter", num(diameter)]
    if angles:
        cmd += ["--angular-search", angles]
    sym = number(opts, "z_symmetry", int, what="Z-axis rotational symmetry")
    if sym and sym > 1:
        cmd += ["--z-axis-rotational-symmetry", str(sym)]
    # pytom_match_template.py has no mirror or invert option: that is done when preparing the template
    for key, flag in (("non_spherical", "--non-spherical-mask"), ("random_phase", "-r"),
                      ("half_precision", "--half-precision")):
        if yes(opts, key):
            cmd.append(flag)
    seed = number(opts, "rng_seed", int, what="Random seed")
    if seed:
        cmd += ["--rng-seed", str(seed)]
    split = words(opts, "volume_split")
    if split and split != ["1", "1", "1"]:
        if len(split) != 3:
            raise Usage("Volume split: three numbers (x y z)")
        cmd += ["-s"] + split
    for axis in "xyz":
        pair = _pair(opts, "search_" + axis, "Search range " + axis)
        if pair:
            cmd += ["--search-" + axis] + pair
    if "tomogram_mask" in files:
        cmd += ["--tomogram-mask", files["tomogram_mask"]]
    cmd += ["--log", (text(opts, "log_level") or "info").lower()]
    return cmd + words(opts, "match_args")


def template_command(opts, source, out, voxel, box, prefix):
    """pytom_create_template.py for the template (prefix "template") or a mask file ("mask")."""
    cmd = conda_prefix(opts) + ["pytom_create_template.py", "-i", source, "-o", out,
                                "--output-voxel-size-angstrom", num(voxel)]
    v = number(opts, prefix + "_angpix_in", what="Input pixel size")
    if v:
        cmd += ["--input-voxel-size-angstrom", num(v)]
    if box:
        cmd += ["-b", str(box)]
    if prefix == "template":
        for key, flag in (("template_invert", "--invert"), ("template_mirror", "--mirror"),
                          ("template_center", "--center")):
            if yes(opts, key):
                cmd.append(flag)
        lp = number(opts, "template_lowpass", what="Template low-pass")
        if lp:
            cmd += ["--low-pass", num(lp)]
    elif yes(opts, "mask_mirror"):
        cmd.append("--mirror")
    return cmd + ["--log", (text(opts, "log_level") or "info").lower()]


def mask_create_command(opts, out, box, voxel):
    r = number(opts, "mask_radius", what="Mask radius")
    if not r:
        d = number(opts, "particle_diameter")
        r = round(d / 2 / voxel, 1) if d else round(box * 0.4, 1)
    cmd = conda_prefix(opts) + ["pytom_create_mask.py", "-b", str(box), "-o", out, "--voxel-size", num(voxel),
                                "-r", num(r)]
    minor = words(opts, "mask_minor")
    if minor:
        if len(minor) != 2:
            raise Usage("Minor radii: two numbers (y and z), or empty for a sphere")
        cmd += ["--radius-minor1", minor[0], "--radius-minor2", minor[1]]
    sigma = number(opts, "mask_sigma", what="Mask edge")
    if sigma:
        cmd += ["-s", num(sigma)]
    return cmd


def prepare_template_and_mask(sup, opts, job_dir, voxel):
    """The template and mask the matching uses (prepared ones in the job directory); output nodes."""
    logs = os.path.join(job_dir, LOG_DIR, "template_and_mask.log")
    nodes = []
    template = text(opts, "fn_template")
    if not template or not os.path.isfile(template):
        raise Usage("template not found: %r" % template)
    if yes(opts, "do_prep_template"):
        out = os.path.join(job_dir, "template.mrc")
        final = number(opts, "template_angpix", what="Template final pixel size") or voxel
        if sup.run(template_command(opts, template, out, final, number(opts, "template_box", int), "template"),
                   logs) != 0 or not os.path.isfile(out):
            raise Usage("pytom_create_template.py failed (see %s)" % logs)
        template = out
        nodes.append((out, NODE_TEMPLATE))
    tx, ty, tz, tvoxel = mrc_header(template)
    if tvoxel and voxel and abs(tvoxel - voxel) > 0.01 * voxel:
        say("WARNING: template voxel %.3f Å, tomogram voxel %.3f Å: pytom needs them equal" % (tvoxel, voxel))
    if text(opts, "mask_source") == MASK_CREATE:
        out = os.path.join(job_dir, "mask.mrc")
        box = number(opts, "mask_box_create", int) or tx
        if sup.run(mask_create_command(opts, out, box, tvoxel or voxel), logs) != 0 or not os.path.isfile(out):
            raise Usage("pytom_create_mask.py failed (see %s)" % logs)
        mask = out
        nodes.append((out, NODE_MASK))
    else:
        mask = text(opts, "fn_mask")
        if not mask or not os.path.isfile(mask):
            raise Usage("mask not found: %r" % mask)
        # either is mirrored only when it is prepared: a template used as it is stays as it is
        mirrored = yes(opts, "do_prep_mask") and yes(opts, "mask_mirror")
        template_mirrored = yes(opts, "do_prep_template") and yes(opts, "template_mirror")
        if template_mirrored != mirrored:
            say("WARNING: the template is %smirrored but the mask file is %s: they may not match"
                % ("" if template_mirrored else "not ", "mirrored" if mirrored else "not"))
        if yes(opts, "do_prep_mask"):
            out = os.path.join(job_dir, "mask.mrc")
            final = number(opts, "mask_angpix", what="Mask final pixel size") or tvoxel or voxel
            box = number(opts, "mask_box", int) or tx
            if sup.run(template_command(opts, mask, out, final, box, "mask"), logs) != 0 or not os.path.isfile(out):
                raise Usage("pytom_create_template.py failed on the mask (see %s)" % logs)
            fix = conda_prefix(opts) + ["python", "-c", MASK_FIX, out]
            if sup.run(fix, logs) != 0:
                raise Usage("could not clip the mask to 0–1 (see %s)" % logs)
            mask = out
            nodes.append((out, NODE_MASK))
    mx, my, mz, _ = mrc_header(mask)
    if (mx, my, mz) != (tx, ty, tz):
        raise Usage("the mask (%d×%d×%d) and the template (%d×%d×%d) need the same box" % (mx, my, mz, tx, ty, tz))
    return template, mask, nodes


# ---------------------------------------------------------------------------- match

def match_outputs(job_dir, stem):
    tm = os.path.join(job_dir, TM_DIR)
    return (os.path.join(tm, stem + "_scores.mrc"), os.path.join(tm, stem + "_angles.mrc"),
            os.path.join(tm, stem + "_job.json"))


def write_summary(job_dir, opts, columns, rows, state, template="", mask=""):
    extra = [COL_TOMOGRAM, COL_VOXEL, COL_SCORES, COL_ANGLES, COL_JOB, COL_STATUS, COL_SECONDS, COL_LOG]
    cols = [c for c in columns if c not in extra] + extra
    table = []
    for r in rows:
        s = state[r[TOMO_NAME]]
        vals = dict(r)
        vals.update({COL_TOMOGRAM: s.get("tomogram", ""), COL_VOXEL: s.get("voxel", ""),
                     COL_SCORES: s.get("scores", ""), COL_ANGLES: s.get("angles", ""), COL_JOB: s.get("json", ""),
                     COL_STATUS: s["status"], COL_SECONDS: "%.0f" % s.get("seconds", 0), COL_LOG: s.get("log", "")})
        table.append([vals.get(c, "") for c in cols])
    general = {P + "TomogramsFile": text(opts, "in_tomograms"), P + "Template": template, P + "Mask": mask,
               P + "TomogramKind": text(opts, "tomo_kind") or "Reconstructed"}
    write_star(os.path.join(job_dir, SUMMARY), [("general", general, None, None), ("global", None, cols, table)])


def run_match(sup, job_dir, threads, opts):
    star, columns, rows = tomogram_rows(opts)
    for d in (TM_DIR, INPUTS_DIR, LOG_DIR):
        p = os.path.join(job_dir, d)
        if not os.path.isdir(p):
            os.makedirs(p)
    state, files = {}, {}
    for r in rows:  # every tomogram's inputs first: a wrong setting shows before hours of matching
        name = r[TOMO_NAME]
        files[name] = prepare_tomogram(r, opts, job_dir)
        scores, angles, js = match_outputs(job_dir, files[name]["stem"])
        done = yes(opts, "skip_done") and all(os.path.isfile(f) for f in (scores, angles, js))
        state[name] = {"status": "done" if done else "pending", "tomogram": _rel(files[name]["tomogram"]),
                       "voxel": "%.4f" % files[name]["voxel"], "scores": _rel(scores), "angles": _rel(angles),
                       "json": _rel(js), "log": _rel(os.path.join(job_dir, LOG_DIR, name + ".log"))}
        if done:
            say("%s: already done, skipped" % name)
    voxels = sorted(set(round(f["voxel"], 3) for f in files.values()))
    if len(voxels) > 1 and not (yes(opts, "do_prep_template") and number(opts, "template_angpix")):
        say("WARNING: the tomograms have different voxel sizes (%s Å): one template fits only one of them"
            % ", ".join(str(v) for v in voxels))
    match_command(opts, files[rows[0][TOMO_NAME]], "template.mrc", "mask.mrc", [0], TM_DIR)  # checks the options
    template, mask, nodes = prepare_template_and_mask(sup, opts, job_dir, voxels[0])
    write_summary(job_dir, opts, columns, rows, state, _rel(template), _rel(mask))

    ids = gpu_ids(opts)
    slots = [ids] if text(opts, "gpu_mode") == GPU_SPLIT else [[i] for i in ids]
    say("%d tomogram(s) to search, GPU(s) %s, %d at a time"
        % (sum(s["status"] != "done" for s in state.values()), " ".join(map(str, ids)), len(slots)))

    def task(name):
        def factory(gpus):
            def gen():
                s = state[name]
                s["status"], t0 = "running", time.time()
                s["gpus"] = gpus
                write_summary(job_dir, opts, columns, rows, state, _rel(template), _rel(mask))
                say("%s: matching on GPU %s" % (name, ":".join(map(str, gpus))))
                cmd = match_command(opts, files[name], template, mask, gpus, os.path.join(job_dir, TM_DIR))
                rc = yield cmd, s["log"]
                s["seconds"] = time.time() - t0
                if rc != 0 or not os.path.isfile(s["scores"]):
                    _show_tail(s["log"])
                    raise Usage("pytom_match_template.py exited with %s (log: %s)" % (rc, s["log"]))
                return True
            return gen()
        return name, factory

    def changed(name, result):
        s = state[name]
        s["status"] = "done" if result else "failed"
        say("%s: %s in %.0f min" % (name, s["status"], s.get("seconds", 0) / 60))
        write_summary(job_dir, opts, columns, rows, state, _rel(template), _rel(mask))

    todo = [task(r[TOMO_NAME]) for r in rows if state[r[TOMO_NAME]]["status"] != "done"]
    try:
        run_parallel(sup, todo, slots, changed)
    except Aborted:
        for s in state.values():
            if s["status"] == "running":
                s["status"] = "aborted"
        write_summary(job_dir, opts, columns, rows, state, _rel(template), _rel(mask))
        raise
    failed = [n for n, s in state.items() if s["status"] != "done"]
    if failed:
        raise Usage("%d tomogram(s) failed: %s. Continue the job to run only those again."
                    % (len(failed), " ".join(failed)))
    return [(os.path.join(job_dir, SUMMARY), NODE_SUMMARY)] + nodes


# ---------------------------------------------------------------------------- extract

def read_summary(path):
    blocks = read_star(path)
    if "global" not in blocks:
        raise Usage("%s: not a pytom match summary" % path)
    return blocks.get("general", Block("general")).pairs, blocks["global"].dicts()


def extract_job_file(row, job_dir):
    """This job's copy of the match job's job.json, with output_dir here and the score maps linked,
    so that pytom writes its files in this job's directory."""
    src = row[COL_JOB]
    with open(src) as fh:
        data = json.load(fh)
    tm = os.path.join(job_dir, TM_DIR)
    stem = os.path.basename(src)[:-len("_job.json")]
    for suffix in ("_scores.mrc", "_angles.mrc", "_whitening_filter.npy"):
        f = os.path.join(os.path.dirname(src), stem + suffix)
        if os.path.exists(f) and not _link(f, os.path.join(tm, stem + suffix)):
            raise Usage("cannot link %s into %s (a file system without symbolic links?)" % (f, tm))
    data["output_dir"] = os.path.abspath(tm)
    dst = os.path.join(tm, stem + "_job.json")
    with open(dst + ".tmp", "w") as fh:
        json.dump(data, fh, indent=4)
    os.replace(dst + ".tmp", dst)
    return dst, stem


def max_particles(opts, name=None):
    """-n of a tomogram: its own (tomo_n_particles) or the job's."""
    own = per_tomogram(opts, "tomo_n_particles", what="Max. particles of some tomograms")
    if name in own:
        return own[name]
    return number(opts, "n_particles", int, True, "Number of particles")


def extract_command(opts, job_file, cut_off=None, n=None):
    n = n or number(opts, "n_particles", int, True, "Number of particles")
    cmd = conda_prefix(opts) + ["pytom_extract_candidates.py", "-j", job_file, "-n", str(n)]
    fp = number(opts, "n_false_positives", what="Number of false positives")
    if fp:
        cmd += ["--number-of-false-positives", num(fp)]
    d = number(opts, "particle_diameter", what="Particle diameter")
    if d:
        cmd += ["--particle-diameter", num(d)]
    if cut_off is not None:
        cmd += ["--cut-off", "%.6g" % max(0.0, cut_off)]
    if yes(opts, "tophat"):
        cmd += ["--tophat-filter", "--tophat-connectivity", str(number(opts, "tophat_connectivity", int) or 1)]
        bins = number(opts, "tophat_bins", int)
        if bins:
            cmd += ["--tophat-bins", str(bins)]
    bins = number(opts, "plot_bins", int)
    if bins:
        cmd += ["--plot-bins", str(bins)]
    cmd += _tomogram_mask_args(opts, job_file)
    cmd += ["--log", (text(opts, "log_level") or "info").lower()]
    return cmd + words(opts, "extract_args")


def _tomogram_mask_args(opts, job_file):
    mode = text(opts, "tomogram_mask_mode") or TMASK_JOB
    if mode == TMASK_IGNORE:
        return ["--ignore_tomogram_mask"]
    if mode == TMASK_OTHER:
        name = os.path.basename(job_file)[:-len("_job.json")]
        mask = tomogram_mask_file(opts, name)
        if not mask:
            raise Usage("no tomogram masks given (Tomogram masks folder)")
        return ["--tomogram-mask", mask]
    return []


def roc_command(opts, job_file, n=None):
    n = number(opts, "roc_particles", int) or 3 * (n or number(opts, "n_particles", int, True, "Number of particles"))
    cmd = conda_prefix(opts) + ["pytom_estimate_roc.py", "-j", job_file, "-n", str(n)]
    d = number(opts, "particle_diameter")
    if d:
        cmd += ["--particle-diameter", num(d)]
    bins = number(opts, "roc_bins", int)
    if bins:
        cmd += ["--bins", str(bins)]
    if yes(opts, "roc_crop"):
        cmd.append("--crop-plot")
    if (text(opts, "tomogram_mask_mode") or TMASK_JOB) == TMASK_IGNORE:
        cmd.append("--ignore_tomogram_mask")
    return cmd


def _move(src, dst):
    if os.path.exists(src):
        os.replace(src, dst)
        return True
    return False


def _cut_off_of(rows):
    for r in rows:
        v = _f(r, "rlnCutOff")
        if v is not None:
            return v
    return None


def read_pytom_particles(path):
    if not os.path.isfile(path):
        return []
    try:
        return loop_block(path, "particles").dicts()
    except Usage:
        return []  # no rows: pytom found nothing above the cut-off


def to_relion5(rows, info, angles):
    """pytom's particles (voxels of the tomogram, from its corner) as RELION 5 rows: rlnTomoName and
    rlnCenteredCoordinate[XYZ]Angst (Å from the tomogram's centre, as RELION 5's picker writes them)."""
    name = info[TOMO_NAME]
    px, binning = _f(info, "rlnTomoTiltSeriesPixelSize"), _f(info, "rlnTomoTomogramBinning")
    sizes = [_f(info, "rlnTomoSize" + a) for a in "XYZ"]
    voxel = _f(info, COL_VOXEL)
    dims = None
    if not (px and binning and all(sizes)):
        tomo = info.get(COL_TOMOGRAM, "")
        nx, ny, nz, hv = mrc_header(tomo)
        dims, voxel = (nx, ny, nz), voxel or hv
    out = []
    for r in rows:
        xyz = [_f(r, "rlnCoordinate" + a) or 0.0 for a in "XYZ"]
        if dims is None:
            c = [(x * binning - s / 2.0) * px for x, s in zip(xyz, sizes)]
        else:
            v = _f(r, "rlnDetectorPixelSize") or voxel
            c = [(x - n / 2.0) * v for x, n in zip(xyz, dims)]
        row = {TOMO_NAME: name, CENTERED[0]: c[0], CENTERED[1]: c[1], CENTERED[2]: c[2]}
        for src, dst in zip(("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"), angles):
            row[dst] = _f(r, src) or 0.0
        lcc = _f(r, LCC) or 0.0
        row.update({LCC: lcc, "rlnCutOff": _f(r, "rlnCutOff") or 0.0, "rlnSearchStd": _f(r, "rlnSearchStd") or 0.0,
                    "rlnAutopickFigureOfMerit": lcc})
        out.append(row)
    return out


def candidate_columns(angles):
    return [TOMO_NAME] + list(CENTERED) + list(angles) + [LCC, "rlnCutOff", "rlnSearchStd", "rlnAutopickFigureOfMerit"]


def read_thresholds(job_dir):
    """{tomogram: (automatic cut-off or None, threshold or None)}, in the file's order."""
    out = {}
    for r in loop_block(os.path.join(job_dir, THRESHOLDS), "thresholds").dicts():
        out[r[TOMO_NAME]] = (_f(r, COL_AUTO), _f(r, COL_THRESHOLD))
    return out


def write_thresholds(job_dir, rows):
    """thresholds.star from {column: value} rows (THRESHOLD_COLUMNS; missing ones are empty)."""
    table = [[r.get(c, "") for c in THRESHOLD_COLUMNS] for r in rows]
    write_star(os.path.join(job_dir, THRESHOLDS), [("thresholds", None, THRESHOLD_COLUMNS, table)])


def write_selection(job_dir, thresholds=None):
    """particles.star: the candidates with rlnLCCmax at or above their tomogram's threshold (rlnCutOff
    becomes that threshold), and thresholds.star with ``thresholds`` ({tomogram: value or None}; None
    selects nothing) put in. Returns {tomogram: (candidates, selected)}."""
    cand = loop_block(os.path.join(job_dir, CANDIDATES), "particles")
    trows = loop_block(os.path.join(job_dir, THRESHOLDS), "thresholds").dicts()
    current = dict((r[TOMO_NAME], _f(r, COL_THRESHOLD)) for r in trows)
    if thresholds:
        for name, thr in thresholds.items():
            if name in current:
                current[name] = thr
    cols = cand.columns
    li, ni = cols.index(LCC), cols.index(TOMO_NAME)
    ci = cols.index("rlnCutOff") if "rlnCutOff" in cols else None
    counts = dict((n, [0, 0]) for n in current)
    rows = []
    for r in cand.rows:
        name = r[ni]
        thr = current.get(name)
        counts.setdefault(name, [0, 0])[0] += 1
        try:
            lcc = float(r[li])
        except ValueError:
            continue
        if thr is not None and lcc >= thr:
            r = list(r)
            if ci is not None:
                r[ci] = "%.6f" % thr
            rows.append(r)
            counts[name][1] += 1
    write_star(os.path.join(job_dir, PARTICLES), [("particles", None, cols, rows)])
    for r in trows:
        n, t = r[TOMO_NAME], current[r[TOMO_NAME]]
        r.update({COL_THRESHOLD: "" if t is None else "%.6f" % t, COL_CANDIDATES: counts[n][0],
                  COL_SELECTED: counts[n][1]})
    write_thresholds(job_dir, trows)
    return dict((n, tuple(c)) for n, c in counts.items())


# options that do not change what a tomogram's extraction gives (or are compared on their own)
NOT_SETTINGS = ("tomo_names", "n_particles", "tomo_n_particles", "pytom_env", "conda", "cuda_module", "log_level")


def settings_key(opts, row):
    """A key of everything else a tomogram's extraction depends on: the job's other options and the
    match job's result for it (its job.json: matched again, it changes)."""
    kept = sorted((k, v) for k, v in opts.items() if k not in NOT_SETTINGS)
    src = row.get(COL_JOB, "")
    try:
        stamp = "%.0f" % os.path.getmtime(src)
    except OSError:
        stamp = ""
    return hashlib.sha1(json.dumps([kept, src, stamp]).encode("utf-8")).hexdigest()[:12]


def previous_extraction(job_dir):
    """What an earlier run of this job left (continue): ({tomogram: thresholds.star row}, {tomogram:
    [candidate rows]}, candidate columns). Empty when it did not finish."""
    th, ca = os.path.join(job_dir, THRESHOLDS), os.path.join(job_dir, CANDIDATES)
    if not (os.path.isfile(th) and os.path.isfile(ca)):
        return {}, {}, []
    try:
        trows = loop_block(th, "thresholds").dicts()
        cand = loop_block(ca, "particles")
    except (Usage, OSError):
        return {}, {}, []
    by = {}
    for r in cand.dicts():
        by.setdefault(r.get(TOMO_NAME, ""), []).append(r)
    return dict((r[TOMO_NAME], r) for r in trows), by, cand.columns


def run_extract(sup, job_dir, threads, opts):
    summary = text(opts, "in_match")
    if not summary or not os.path.isfile(summary):
        raise Usage("pytom match summary not found: %r (the pytom_match.star of a template matching job)" % summary)
    general, rows = read_summary(summary)
    names = selected_names(opts)
    rows = [r for r in rows if not names or r[TOMO_NAME] in names]
    ready = [r for r in rows if r.get(COL_STATUS) == "done" and os.path.isfile(r.get(COL_JOB, ""))]
    for r in rows:
        if r not in ready:
            say("%s: not matched yet (%s), left out" % (r[TOMO_NAME], r.get(COL_STATUS, "?")))
    if not ready:
        raise Usage("no tomograms of %s are done" % summary)
    for d in (TM_DIR, LOG_DIR, PLOTS_DIR):
        p = os.path.join(job_dir, d)
        if not os.path.isdir(p):
            os.makedirs(p)
    fixed = number(opts, "cut_off", what="Cut-off") if text(opts, "cut_off_mode") == CUT_FIXED else None
    if text(opts, "cut_off_mode") == CUT_FIXED and fixed is None:
        raise Usage("give the fixed cut-off")
    interactive = yes(opts, "do_candidates")
    factor = number(opts, "candidate_factor", what="Candidates down to") or 0.7
    if factor <= 0:
        raise Usage("Candidates down to: must be above 0")
    angles = ORIENTATIONS.get(text(opts, "orientations"), ORIENTATIONS["rlnAngleRot/Tilt/Psi"])
    cols = candidate_columns(angles)
    tm, plots = os.path.join(job_dir, TM_DIR), os.path.join(job_dir, PLOTS_DIR)

    # what each tomogram is to be extracted with, and what an earlier run (continue) did
    want = {}
    for r in ready:
        want[r[TOMO_NAME]] = {COL_MAX: max_particles(opts, r[TOMO_NAME]), COL_SETTINGS: settings_key(opts, r)}
    before, before_cands, before_cols = previous_extraction(job_dir)
    found, kept = {}, {}  # name -> (cut-off, candidate rows); name -> threshold kept
    for r in ready:
        name, w = r[TOMO_NAME], want[r[TOMO_NAME]]
        b = before.get(name)
        if b is None or b.get(COL_SETTINGS, "") != w[COL_SETTINGS] or before_cols != cols:
            continue  # extracted anew, with the new cut-off as threshold
        kept[name] = _f(b, COL_THRESHOLD)  # the threshold chosen for it stays
        if _f(b, COL_MAX) == w[COL_MAX]:
            found[name] = (_f(b, COL_AUTO), before_cands.get(name, []))
            say("%s: already extracted with these settings, kept" % name)

    def task(row):
        name = row[TOMO_NAME]
        n = want[name][COL_MAX]

        def factory(_slot):
            def gen():
                log = os.path.join(job_dir, LOG_DIR, name + ".log")
                job_file, stem = extract_job_file(row, job_dir)
                out = os.path.join(tm, stem + "_particles.star")
                rc = yield extract_command(opts, job_file, fixed, n), log
                if rc != 0:
                    _show_tail(log)
                    raise Usage("pytom_extract_candidates.py exited with %s (log: %s)" % (rc, log))
                selected = os.path.join(tm, stem + "_selected.star")
                if not _move(out, selected) and os.path.exists(selected):
                    os.remove(selected)  # from an earlier run
                _move(os.path.join(tm, stem + "_extraction_graph.svg"), os.path.join(plots, name + "_extraction.svg"))
                _move(os.path.join(tm, stem + "_tophat_filter.svg"), os.path.join(plots, name + "_tophat.svg"))
                sel = read_pytom_particles(selected)
                cut = fixed if fixed is not None else _cut_off_of(sel)
                cands = sel
                if interactive:
                    low = factor * cut if cut is not None else 0.0
                    rc = yield extract_command(opts, job_file, low, n), log
                    if rc != 0:
                        _show_tail(log)
                        raise Usage("pytom_extract_candidates.py (candidates) exited with %s (log: %s)" % (rc, log))
                    c_file = os.path.join(tm, stem + "_candidates.star")
                    _move(out, c_file)
                    _move(os.path.join(tm, stem + "_extraction_graph.svg"),
                          os.path.join(plots, name + "_candidates_extraction.svg"))
                    cands = read_pytom_particles(c_file)
                    for r in cands:  # the automatic cut-off, not the one used for the candidates
                        r["rlnCutOff"] = "" if cut is None else str(cut)
                if yes(opts, "do_roc"):
                    rc = yield roc_command(opts, job_file, n), log
                    if rc != 0:
                        say("%s: pytom_estimate_roc.py failed (see %s); going on without it" % (name, log))
                    _move(os.path.join(tm, stem + "_roc.svg"), os.path.join(plots, name + "_roc.svg"))
                found[name] = (cut, to_relion5(cands, row, angles))
                say("%s: %d particle(s) above %s%s (at most %d)"
                    % (name, len(sel), "the cut-off" if cut is None else "%.4f" % cut,
                       ", %d candidate(s) down to %.4g x it" % (len(cands), factor) if interactive else "", n))
                return True
            return gen()
        return name, factory

    todo = [r for r in ready if r[TOMO_NAME] not in found]
    if len(todo) < len(ready):
        say("%d tomogram(s) to extract, %d kept from before" % (len(todo), len(ready) - len(todo)))
    run_parallel(sup, [task(r) for r in todo], [None] * max(1, threads))
    failed = [r[TOMO_NAME] for r in ready if r[TOMO_NAME] not in found]
    if failed:
        raise Usage("extraction failed for: %s" % " ".join(failed))
    table = []
    for r in ready:
        table += [[row.get(c, "") for c in cols] for row in found[r[TOMO_NAME]][1]]
    write_star(os.path.join(job_dir, CANDIDATES), [("particles", None, cols, table)])
    thr = []
    for r in ready:
        name = r[TOMO_NAME]
        cut, w = found[name][0], want[name]
        t = kept[name] if name in kept else cut
        thr.append({TOMO_NAME: name, COL_AUTO: "" if cut is None else "%.6f" % cut,
                    COL_THRESHOLD: "" if t is None else "%.6f" % t, COL_MAX: w[COL_MAX],
                    COL_SETTINGS: w[COL_SETTINGS]})
    write_thresholds(job_dir, thr)
    counts = write_selection(job_dir)
    say("particles.star: %d particle(s) in %d tomogram(s)" % (sum(c[1] for c in counts.values()), len(counts)))
    tomograms = general.get(P + "TomogramsFile", "")
    write_star(os.path.join(job_dir, OPTIMISATION_SET),
               [("", {"rlnTomoParticlesFile": os.path.join(job_dir, PARTICLES), "rlnTomoTomogramsFile": tomograms},
                 None, None)])
    return [(os.path.join(job_dir, PARTICLES), NODE_PARTICLES),
            (os.path.join(job_dir, OPTIMISATION_SET), NODE_OPTIMISATION),
            (os.path.join(job_dir, CANDIDATES), NODE_CANDIDATES),
            (os.path.join(job_dir, THRESHOLDS), NODE_THRESHOLDS)]


# ---------------------------------------------------------------------------- main

def run(argv):
    mode, job_dir, threads, opts = parse(argv)
    C.WHO = WHO
    for f in (SUCCESS, FAILURE, ABORTED):
        if os.path.exists(os.path.join(job_dir, f)):
            os.remove(os.path.join(job_dir, f))
    sup = Supervisor(job_dir, hint=" (is pytom's conda environment right?)")
    say("cryo-savanna pytom %s, job %s" % (mode, job_dir))
    try:
        load_modules(opts)
        nodes = (run_match if mode == MATCH else run_extract)(sup, job_dir, threads, opts)
    except Aborted:
        _touch(os.path.join(job_dir, ABORTED))
        return 128 + sup.stopped if sup.stopped else 1
    except Usage as e:
        print("ERROR: " + str(e), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, FAILURE))
        return 1
    write_output_nodes(job_dir, [(_rel(p), t) for p, t in nodes])
    say("done")
    _touch(os.path.join(job_dir, SUCCESS))
    return 0


def main():
    C.main(run)


if __name__ == "__main__":
    main()
