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
"""Run pytom-match-pick (template matching in tomograms, and the extraction of its candidates) as a RELION
External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/pytom_runner.py match

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
maps, with the status of each tomogram's matching and extraction) is rewritten as tomograms finish, so the
results can be looked at while the job runs. Tomograms already done are skipped when the job is run again
(continue).

With do_extract, each tomogram is extracted (pytom_extract_candidates.py, on a copy of its job.json with
output_dir this job's tm/ as an absolute path) as soon as it is matched, beside the matching of the others,
as many at once as the job's threads (--j) leave beside the matching (one per GPU slot matching), at least
one: those matched before, on a continue, as many as that at the start. pytom's programs
need CUDA even to extract, which is mostly CPU work: each extraction sees only the GPU that matched its
tomogram (CUDA_VISIBLE_DEVICES; those matched before take the job's GPUs in turn), so it runs on the job's
GPUs, beside their matching, and on no other GPU of the machine. pytom's
automatic cut-off (number of false positives), or a fixed one; optionally also further candidates below it
(to choose thresholds interactively in cryo-savanna) and pytom_estimate_roc.py. The candidates go into
candidates.star (RELION 5: rlnTomoName, rlnCenteredCoordinate[XYZ]Angst, angles, rlnLCCmax, ...), the
thresholds per tomogram into thresholds.star, the particles above them into particles.star (with
optimisation_set.star) for RELION 5's Extract subtomos; the four are rewritten as each tomogram is
extracted, keeping the thresholds chosen meanwhile in cryo-savanna's Thresholds view. write_selection()
rewrites particles.star for new thresholds (the Thresholds view calls it). thresholds.star also keeps, per
tomogram, the maximum number of particles it was extracted with and a key of the extraction's other
settings and of its matching: when the job runs again (continue), only the tomograms matched again or whose
extraction settings changed are extracted again (e.g. a larger maximum for one tomogram, tomo_n_particles,
set in the Thresholds view); the others keep their candidates and the thresholds chosen for them.

Like every External job it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED and, before success,
RELION_OUTPUT_NODES.star. RELION_JOB_ABORT_NOW and SIGTERM/SIGINT/SIGHUP (scancel) stop the pytom
processes (their process groups: SIGTERM, then SIGKILL).
"""

import hashlib
import json
import os
import re
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
MATCH = "match"


# files of the jobs
SUMMARY = "pytom_match.star"  # one row per tomogram
CANDIDATES = "candidates.star"  # every candidate, RELION 5 coordinates
THRESHOLDS = "thresholds.star"  # threshold per tomogram
PARTICLES = "particles.star"  # the candidates above the thresholds
OPTIMISATION_SET = "optimisation_set.star"
TM_DIR, INPUTS_DIR, LOG_DIR, PLOTS_DIR = "tm", "inputs", "logs", "plots"
EXTRACT_JOB = "_extract_job.json"  # tm/<stem>_extract_job.json: the copy of pytom's job.json extraction reads
EXTRACT_LOG = "_extract.log"  # logs/<tomogram>_extract.log (logs/<tomogram>.log: its matching)

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
COL_EXTRACTION = P + "Extraction"  # pytom_match.star: the status of the tomogram's extraction ("": none)
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
TMASK_JOB, TMASK_OTHER, TMASK_IGNORE = "As for matching", "Other mask(s)", "None (ignore masks)"

# clips a resampled mask to [0, 1] (run with the python of the pytom environment). The mask is never
# inverted: only the template's contrast is (pytom's mask is 1 on the particle, whatever the contrast)
MASK_FIX = ("import sys, numpy as np, mrcfile\n"
            "with mrcfile.open(sys.argv[1], mode='r+') as m:\n"
            "    m.set_data(np.clip(np.asarray(m.data, dtype=np.float32), 0, 1))\n")


# ---------------------------------------------------------------------------- arguments

def parse(argv, modes=(MATCH,)):
    if argv[:1] == ["extract"] and "extract" not in modes:
        raise Usage("this is a candidate extraction job of an earlier cryo-savanna: it cannot run any more. The "
                    "candidates are now extracted by the template matching job itself: continue that job with "
                    "'Extract candidates?' set to Yes (Extraction tab). This job's results can still be viewed.")
    return C.parse(argv, modes)


def conda_prefix(opts):
    """The start of every pytom command: ``conda run`` in the job's environment (nothing: on PATH)."""
    return C.conda_run(opts, "pytom_env")


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
    first = {}  # type: dict[str, str]
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


def tomogram_mask_file(opts, name, key="tomogram_mask"):
    """The tomogram mask of tomogram ``name`` (option ``key``: the matching's tomogram_mask, or the
    extraction's extract_mask), else "": in a folder, <name>_mask.mrc (TOMO_MASK_SUFFIX, as cryo-savanna's
    Tomogram masks job writes them); a path with {name}, that path for the tomogram; else one file for every
    tomogram."""
    value = text(opts, key)
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


# ---------------------------------------------------------------------------- match: the summary

def match_outputs(job_dir, stem):
    tm = os.path.join(job_dir, TM_DIR)
    return (os.path.join(tm, stem + "_scores.mrc"), os.path.join(tm, stem + "_angles.mrc"),
            os.path.join(tm, stem + "_job.json"))


def write_summary(job_dir, opts, columns, rows, state, template="", mask=""):
    extra = [COL_TOMOGRAM, COL_VOXEL, COL_SCORES, COL_ANGLES, COL_JOB, COL_STATUS, COL_SECONDS, COL_LOG,
             COL_EXTRACTION]
    cols = [c for c in columns if c not in extra] + extra
    table = []
    for r in rows:
        s = state[r[TOMO_NAME]]
        vals = dict(r)
        vals.update({COL_TOMOGRAM: s.get("tomogram", ""), COL_VOXEL: s.get("voxel", ""),
                     COL_SCORES: s.get("scores", ""), COL_ANGLES: s.get("angles", ""), COL_JOB: s.get("json", ""),
                     COL_STATUS: s["status"], COL_SECONDS: "%.0f" % s.get("seconds", 0), COL_LOG: s.get("log", ""),
                     COL_EXTRACTION: s.get("extraction", "")})
        table.append([vals.get(c, "") for c in cols])
    general = {P + "TomogramsFile": text(opts, "in_tomograms"), P + "Template": template, P + "Mask": mask,
               P + "TomogramKind": text(opts, "tomo_kind") or "Reconstructed"}
    write_star(os.path.join(job_dir, SUMMARY), [("general", general, None, None), ("global", None, cols, table)])


# ---------------------------------------------------------------------------- extraction: pytom's programs

def extraction_job_file(job_json, job_dir):
    """A copy of pytom's job.json of a matched tomogram (tm/<stem>_extract_job.json) with output_dir this
    job's tm/ as an absolute path: pytom's extraction reads the score maps from there and writes its particles
    there, also when the project was moved or copied since the tomogram was matched."""
    with open(job_json) as fh:
        data = json.load(fh)
    tm = os.path.join(job_dir, TM_DIR)
    data["output_dir"] = os.path.abspath(tm)
    stem = os.path.basename(job_json)[:-len("_job.json")]
    dst = os.path.join(tm, stem + EXTRACT_JOB)
    tmp = "%s.%d.tmp" % (dst, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=4)
    os.replace(tmp, dst)
    return dst, stem


def max_particles(opts, name=None):
    """-n of a tomogram: its own (tomo_n_particles) or the job's."""
    own = per_tomogram(opts, "tomo_n_particles", what="Max. particles of some tomograms")
    if name in own:
        return own[name]
    return number(opts, "n_particles", int, True, "Max. particles per tomogram")


def extract_command(opts, job_file, name, cut_off=None, n=None):
    n = n or max_particles(opts, name)
    cmd = conda_prefix(opts) + ["pytom_extract_candidates.py", "-j", job_file, "-n", str(n)]
    fp = number(opts, "n_false_positives", what="False positives per tomogram")
    if fp:
        cmd += ["--number-of-false-positives", num(fp)]
    d = number(opts, "extract_diameter", what="Particle diameter for extraction")
    if d:
        cmd += ["--particle-diameter", num(d)]
    if cut_off is not None:
        cmd += ["--cut-off", "%.6g" % max(0.0, cut_off)]
    if yes(opts, "tophat"):
        cmd += ["--tophat-filter", "--tophat-connectivity",
                str(number(opts, "tophat_connectivity", int, what="Tophat connectivity") or 1)]
        bins = number(opts, "tophat_bins", int, what="Tophat histogram bins")
        if bins:
            cmd += ["--tophat-bins", str(bins)]
    bins = number(opts, "plot_bins", int, what="Plot histogram bins")
    if bins:
        cmd += ["--plot-bins", str(bins)]
    cmd += _tomogram_mask_args(opts, name)
    cmd += ["--log", (text(opts, "log_level") or "info").lower()]
    return cmd + words(opts, "extract_args")


def _tomogram_mask_args(opts, name):
    mode = text(opts, "tomogram_mask_mode") or TMASK_JOB
    if mode == TMASK_IGNORE:
        return ["--ignore_tomogram_mask"]
    if mode == TMASK_OTHER:
        mask = tomogram_mask_file(opts, name, "extract_mask")
        if not mask:
            raise Usage("no tomogram masks given for the extraction (Extraction masks folder)")
        return ["--tomogram-mask", mask]
    return []  # pytom's: the tomogram mask of the matching (in its job.json), if any


def roc_command(opts, job_file, n):
    n = number(opts, "roc_particles", int, what="ROC: number of peaks") or 3 * n
    cmd = conda_prefix(opts) + ["pytom_estimate_roc.py", "-j", job_file, "-n", str(n)]
    d = number(opts, "extract_diameter")
    if d:
        cmd += ["--particle-diameter", num(d)]
    bins = number(opts, "roc_bins", int, what="ROC: histogram bins")
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


# ---------------------------------------------------------------------------- extraction: RELION 5 coordinates

def tomogram_geometry(info):
    """How a matched tomogram's voxels map to RELION 5's frame: (tilt-series pixel size, binning,
    rlnTomoSize[XYZ], voxel size) when they can be trusted, else (None, None, the tomogram's (nx, ny, nz),
    voxel size): centred on the tomogram.

    The binning is the tomogram's: rlnTomoTomogramBinning, or the voxel size it was matched with over the
    pixel size (a tomogram of another binning than the tomograms.star says, with the voxel size given in the
    job), whichever fits the tomogram's size (n × binning = rlnTomoSize, give or take a voxel)."""
    px = _f(info, "rlnTomoTiltSeriesPixelSize")
    sizes = [_f(info, "rlnTomoSize" + a) for a in "XYZ"]
    voxel = _f(info, COL_VOXEL)  # what the tomogram was matched with
    try:
        nx, ny, nz, hv = mrc_header(info.get(COL_TOMOGRAM, ""))
        dims = (nx, ny, nz)
    except (OSError, Usage):
        dims, hv = None, 0.0
    voxel = voxel or hv
    if px and all(sizes):
        bins = [b for b in (_f(info, "rlnTomoTomogramBinning"), voxel / px if voxel else None) if b]
        for b in bins:
            if dims is None or all(abs(n * b - s) <= b for n, s in zip(dims, sizes)):
                return px, b, sizes, voxel
        if bins and dims:
            say("WARNING: %s: the tomogram is %s voxels of %.3f Å, rlnTomoSize %s pixels of %.3f Å: the "
                "coordinates are centred on the tomogram instead"
                % (info[TOMO_NAME], "×".join(map(str, dims)), voxel, "×".join(num(s) for s in sizes), px))
    if dims is None:
        raise Usage("%s: no tomogram geometry (rlnTomoTiltSeriesPixelSize, rlnTomoSize[XYZ] or the tomogram's "
                    "MRC header) for RELION 5 coordinates" % info[TOMO_NAME])
    return None, None, dims, voxel


def to_relion5(rows, info, angles):
    """pytom's particles (voxels of the tomogram, from its corner) as RELION 5 rows: rlnTomoName and
    rlnCenteredCoordinate[XYZ]Angst (Å from the tomogram's centre, as RELION 5's picker writes them).

    RELION 5 takes a centred coordinate to tilt-series pixels as Å / rlnTomoTiltSeriesPixelSize +
    rlnTomoSize / 2, and its tomograms have voxel i at tilt-series pixel i × binning (origin 0), so voxel i
    is (i × binning − rlnTomoSize / 2) × pixel size. This is done here rather than with pytom's own
    --relion5-compat, which puts the centre one pixel off (size × binning / 2 − 1) and, unless the matching
    was given --relion5-tomograms-star, centres on the tomogram at its voxel size. If pytom was given
    --relion5-compat anyway (More arguments), its centred coordinates are kept."""
    name = info[TOMO_NAME]
    px, binning, sizes, voxel = tomogram_geometry(info) if rows else (None, None, (), None)
    out = []
    for r in rows:
        if all(_f(r, c) is not None for c in CENTERED):  # pytom's --relion5-compat
            c = [_f(r, k) for k in CENTERED]
        else:
            xyz = [_f(r, "rlnCoordinate" + a) for a in "XYZ"]
            if None in xyz:
                raise Usage("%s: a particle of pytom without coordinates (rlnCoordinate[XYZ] or "
                            "rlnCenteredCoordinate[XYZ]Angst)" % name)
            if binning:
                c = [(x * binning - s / 2.0) * px for x, s in zip(xyz, sizes)]
            else:
                v = _f(r, "rlnDetectorPixelSize") or voxel
                if not v:
                    raise Usage("%s: no voxel size for RELION 5 coordinates" % name)
                c = [(x - n / 2.0) * v for x, n in zip(xyz, sizes)]
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


# ---------------------------------------------------------------------------- extraction: thresholds

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


# the options a tomogram's extraction depends on, besides its matching (n_particles and tomo_n_particles are
# compared on their own: a new maximum keeps the threshold chosen)
EXTRACT_SETTINGS = ("extract_diameter", "cut_off_mode", "n_false_positives", "cut_off", "do_candidates",
                    "candidate_factor", "tophat", "tophat_connectivity", "tophat_bins", "plot_bins",
                    "tomogram_mask_mode", "extract_mask", "do_roc", "roc_particles", "roc_bins", "roc_crop",
                    "orientations", "extract_args")


def settings_key(opts, job_json):
    """A key of everything else a tomogram's extraction depends on: the extraction's options and the
    tomogram's matching (its job.json: matched again, it changes)."""
    kept = sorted((k, text(opts, k)) for k in EXTRACT_SETTINGS)
    try:
        stamp = "%.0f" % os.path.getmtime(job_json)
    except OSError:
        stamp = ""
    return hashlib.sha1(json.dumps([kept, _rel(job_json), stamp]).encode("utf-8")).hexdigest()[:12]


def previous_extraction(job_dir):
    """What an earlier run of this job left (continue): ({tomogram: thresholds.star row}, {tomogram:
    [candidate rows]}, candidate columns). Empty when it extracted nothing."""
    th, ca = os.path.join(job_dir, THRESHOLDS), os.path.join(job_dir, CANDIDATES)
    if not (os.path.isfile(th) and os.path.isfile(ca)):
        return {}, {}, []
    try:
        trows = loop_block(th, "thresholds").dicts()
        cand = loop_block(ca, "particles")
    except (Usage, OSError):
        return {}, {}, []
    by = {}  # type: dict[str, list[dict[str, str]]]
    for r in cand.dicts():
        by.setdefault(r.get(TOMO_NAME, ""), []).append(r)
    return dict((r[TOMO_NAME], r) for r in trows), by, cand.columns


# ---------------------------------------------------------------------------- extraction, beside the matching

class Extraction(object):
    """The candidates of the job's tomograms: pytom_extract_candidates.py on each one as soon as it is matched,
    beside the matching of the others (run_match gives the tasks to a pool capped by the threads the matching
    leaves), and the tables of those extracted so far.

    ``rows`` and ``cands`` hold, per tomogram, its thresholds.star row and its candidates: from an earlier run
    of the job (continue) until it is extracted again, then this run's. write() writes them out after each
    extraction, so the thresholds can be chosen while the rest is matched."""

    def __init__(self, job_dir, opts, names):
        self.job_dir, self.opts, self.names = job_dir, opts, list(names)
        self.fixed = None
        if text(opts, "cut_off_mode") == CUT_FIXED:
            self.fixed = number(opts, "cut_off", what="Fixed cut-off")
            if self.fixed is None:
                raise Usage("give the fixed cut-off (Extraction tab)")
        self.interactive = yes(opts, "do_candidates")
        self.factor = number(opts, "candidate_factor", what="Candidates down to") or 0.7
        if self.factor <= 0:
            raise Usage("Candidates down to: must be above 0")
        self.angles = ORIENTATIONS.get(text(opts, "orientations"), ORIENTATIONS["rlnAngleRot/Tilt/Psi"])
        self.cols = candidate_columns(self.angles)
        for name in self.names:  # every option (and extraction mask) checked before hours of matching
            n = max_particles(opts, name)
            extract_command(opts, "job.json", name, self.fixed, n)
            if yes(opts, "do_roc"):
                roc_command(opts, "job.json", n)
        if not number(opts, "extract_diameter") and not number(opts, "particle_diameter", what="Particle diameter"):
            raise Usage("give the particle diameter for the extraction (Extraction tab): the matching has none")
        p = os.path.join(job_dir, PLOTS_DIR)
        if not os.path.isdir(p):
            os.makedirs(p)
        before, cands, cols = previous_extraction(job_dir)
        same = cols == self.cols  # else (other angle columns) every tomogram is extracted again anyway
        self.rows = dict((n, before[n]) for n in self.names if n in before and same)
        self.cands = dict((n, cands.get(n, [])) for n in self.rows)

    def up_to_date(self, name, job_json):
        """Extracted in an earlier run with these settings and this maximum, from this matching: kept."""
        b = self.rows.get(name)
        return b is not None and b.get(COL_SETTINGS, "") == settings_key(self.opts, job_json) and \
            _f(b, COL_MAX) == max_particles(self.opts, name)

    def task(self, name, info, job_json, gpu):
        """(name, factory) of run_pools: the extraction of one matched tomogram, its programs shown only GPU
        ``gpu`` of the job (pytom's need CUDA, though extracting is mostly CPU work). ``info``: its
        tomograms.star row with COL_TOMOGRAM and COL_VOXEL (the tomogram matched and its voxel size)."""
        opts, job_dir, fixed, factor = self.opts, self.job_dir, self.fixed, self.factor
        tm, plots = os.path.join(job_dir, TM_DIR), os.path.join(job_dir, PLOTS_DIR)
        env = {"CUDA_VISIBLE_DEVICES": C.cuda_device(gpu)}

        def factory(_slot):
            def gen():
                n, key = max_particles(opts, name), settings_key(opts, job_json)
                log = os.path.join(job_dir, LOG_DIR, name + EXTRACT_LOG)
                say("%s: extracting candidates on GPU %d (at most %d)" % (name, gpu, n))
                job_file, stem = extraction_job_file(job_json, job_dir)
                out = os.path.join(tm, stem + "_particles.star")
                rc = yield extract_command(opts, job_file, name, fixed, n), log, env
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
                if self.interactive:
                    low = factor * cut if cut is not None else 0.0
                    rc = yield extract_command(opts, job_file, name, low, n), log, env
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
                    rc = yield roc_command(opts, job_file, n), log, env
                    if rc != 0:
                        say("%s: pytom_estimate_roc.py failed (see %s); going on without it" % (name, log))
                    _move(os.path.join(tm, stem + "_roc.svg"), os.path.join(plots, name + "_roc.svg"))
                relion = to_relion5(cands, info, self.angles)
                cut_text = "" if cut is None else "%.6f" % cut
                # the threshold: the cut-off, unless write() finds one chosen for the same settings
                self.rows[name] = {TOMO_NAME: name, COL_AUTO: cut_text, COL_THRESHOLD: cut_text, COL_MAX: n,
                                   COL_SETTINGS: key}
                self.cands[name] = relion
                say("%s: %d particle(s) above %s%s (at most %d)"
                    % (name, len(sel), "the cut-off" if cut is None else "%.4f" % cut,
                       ", %d candidate(s) down to %.4g x it" % (len(cands), factor) if self.interactive else "", n))
                return True
            return gen()
        return name, factory

    def write(self):
        """candidates.star, thresholds.star, particles.star and optimisation_set.star of the tomograms extracted
        so far (and those kept from an earlier run), in the tomograms' order. A tomogram whose row in
        thresholds.star has the same settings keeps the threshold there: chosen in the Thresholds view, maybe
        while the job runs, or kept from before for a new maximum. Returns {tomogram: (candidates, selected)};
        {} when there is nothing to write yet."""
        names = [n for n in self.names if n in self.rows]
        if not names:
            return {}
        try:
            disk = dict((r[TOMO_NAME], r) for r in loop_block(os.path.join(self.job_dir, THRESHOLDS),
                                                               "thresholds").dicts())
        except (OSError, Usage, KeyError):
            disk = {}
        for n in names:
            d = disk.get(n)
            if d is not None and d.get(COL_SETTINGS, "") == self.rows[n].get(COL_SETTINGS, ""):
                self.rows[n][COL_THRESHOLD] = d.get(COL_THRESHOLD, "")
        table = [[r.get(c, "") for c in self.cols] for n in names for r in self.cands[n]]
        write_star(os.path.join(self.job_dir, CANDIDATES), [("particles", None, self.cols, table)])
        write_thresholds(self.job_dir, [self.rows[n] for n in names])
        counts = write_selection(self.job_dir)
        write_star(os.path.join(self.job_dir, OPTIMISATION_SET),
                   [("", {"rlnTomoParticlesFile": os.path.join(self.job_dir, PARTICLES),
                          "rlnTomoTomogramsFile": text(self.opts, "in_tomograms")}, None, None)])
        return counts

    @staticmethod
    def nodes(job_dir):
        return [(os.path.join(job_dir, PARTICLES), NODE_PARTICLES),
                (os.path.join(job_dir, OPTIMISATION_SET), NODE_OPTIMISATION),
                (os.path.join(job_dir, CANDIDATES), NODE_CANDIDATES),
                (os.path.join(job_dir, THRESHOLDS), NODE_THRESHOLDS)]


# ---------------------------------------------------------------------------- match (and extract)

def run_match(sup, job_dir, threads, opts):
    star, columns, rows = tomogram_rows(opts)
    for d in (TM_DIR, INPUTS_DIR, LOG_DIR):
        p = os.path.join(job_dir, d)
        if not os.path.isdir(p):
            os.makedirs(p)
    extracting = yes(opts, "do_extract")
    state, files = {}, {}
    for r in rows:  # every tomogram's inputs first: a wrong setting shows before hours of matching
        name = r[TOMO_NAME]
        files[name] = prepare_tomogram(r, opts, job_dir)
        scores, angles, js = match_outputs(job_dir, files[name]["stem"])
        done = yes(opts, "skip_done") and all(os.path.isfile(f) for f in (scores, angles, js))
        state[name] = {"status": "done" if done else "pending", "tomogram": _rel(files[name]["tomogram"]),
                       "voxel": "%.4f" % files[name]["voxel"], "scores": _rel(scores), "angles": _rel(angles),
                       "json": _rel(js), "log": _rel(os.path.join(job_dir, LOG_DIR, name + ".log")),
                       "extraction": "pending" if extracting else ""}
        if done:
            say("%s: already matched, skipped" % name)
    voxels = sorted(set(round(f["voxel"], 3) for f in files.values()))
    if len(voxels) > 1 and not (yes(opts, "do_prep_template") and number(opts, "template_angpix")):
        say("WARNING: the tomograms have different voxel sizes (%s Å): one template fits only one of them"
            % ", ".join(str(v) for v in voxels))
    match_command(opts, files[rows[0][TOMO_NAME]], "template.mrc", "mask.mrc", [0], TM_DIR)  # checks the options
    ex = Extraction(job_dir, opts, [r[TOMO_NAME] for r in rows]) if extracting else None  # and these
    template, mask, nodes = prepare_template_and_mask(sup, opts, job_dir, voxels[0])

    def summary():
        write_summary(job_dir, opts, columns, rows, state, _rel(template), _rel(mask))

    ids = gpu_ids(opts)
    slots = [ids] if text(opts, "gpu_mode") == GPU_SPLIT else [[i] for i in ids]
    gpu_pool = C.Pool("match", slots)
    # every tomogram as soon as it is matched, in the job's threads the matching leaves (at least one)
    ex_pool = C.Pool("extract", most=lambda running: max(1, threads - running["match"]))
    by_name = dict((r[TOMO_NAME], r) for r in rows)

    def match_task(name):
        def factory(gpus):
            def gen():
                s = state[name]
                s["status"], t0 = "running", time.time()
                s["gpus"] = gpus
                summary()
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

    def extract_task(name, gpu):
        assert ex is not None  # only queued when the job extracts
        s = state[name]
        info = dict(by_name[name])
        info.update({COL_TOMOGRAM: s["tomogram"], COL_VOXEL: s["voxel"]})
        key, factory = ex.task(name, info, s["json"], gpu)

        def started(slot):
            s["extraction"] = "running"
            summary()
            return factory(slot)
        return key, started

    def to_extract(name, gpu, matched_now):
        """Extracted now, on GPU ``gpu`` of the job, unless extracted before with these settings (a tomogram
        matched in this run always is: its scores are new)."""
        assert ex is not None
        if not matched_now and ex.up_to_date(name, state[name]["json"]):
            state[name]["extraction"] = "done"
            say("%s: already extracted with these settings, kept" % name)
            return False
        state[name]["extraction"] = "pending"
        ex_pool.add(*extract_task(name, gpu))
        return True

    def changed(pool, name, result):
        s = state[name]
        if pool is gpu_pool:
            s["status"] = "done" if result else "failed"
            say("%s: %s in %.0f min" % (name, "matched" if result else "matching failed", s.get("seconds", 0) / 60))
            if result and ex is not None:
                to_extract(name, s["gpus"][0], True)  # on the GPU that matched it
        else:
            assert ex is not None
            s["extraction"] = "done" if result else "failed"
            if result:
                try:
                    counts = ex.write()
                    say("%s: extracted; particles.star: %d particle(s) in %d tomogram(s) so far"
                        % (name, sum(c[1] for c in counts.values()), len(counts)))
                except (Usage, OSError, ValueError, KeyError) as e:
                    say("WARNING: could not write the particles of the tomograms extracted so far: %s" % e)
        summary()

    before = 0  # tomograms matched in an earlier run, to extract now: on the job's GPUs in turn
    for r in rows:
        name = r[TOMO_NAME]
        if state[name]["status"] != "done":
            gpu_pool.add(*match_task(name))
        elif ex is not None and to_extract(name, ids[before % len(ids)], False):
            before += 1
    summary()
    say("%d tomogram(s) to search, GPU(s) %s, %d at a time" % (len(gpu_pool.pending), " ".join(map(str, ids)),
                                                                len(slots)))
    if ex is not None:
        say("candidate extraction: %d tomogram(s) matched before, now; then each one as soon as it is matched, "
            "on the GPU that matched it; at most %d at once (the job's %d thread(s), less one per GPU slot matching, "
            "at least one)" % (before, max(1, threads), threads))
    try:
        C.run_pools(sup, [gpu_pool, ex_pool], changed)
    except Aborted:
        for s in state.values():
            for k in ("status", "extraction"):
                if s.get(k) == "running":
                    s[k] = "aborted"
        summary()
        raise
    names = [r[TOMO_NAME] for r in rows]
    failed = [n for n in names if state[n]["status"] != "done"]
    not_extracted = [n for n in names if ex is not None and state[n]["status"] == "done"
                     and state[n]["extraction"] != "done"]
    if ex is not None:
        try:  # every tomogram's, also when some failed: a continue starts from there
            counts = ex.write()
        except (OSError, ValueError, KeyError) as e:
            raise Usage("could not write the particles (%s, %s): %s" % (CANDIDATES, PARTICLES, e))
        if counts:
            say("particles.star: %d particle(s) in %d tomogram(s)" % (sum(c[1] for c in counts.values()), len(counts)))
    if failed or not_extracted:
        what = ["%s (matching)" % " ".join(failed)] if failed else []
        what += ["%s (extraction)" % " ".join(not_extracted)] if not_extracted else []
        raise Usage("%d tomogram(s) failed: %s. Continue the job to run only those again."
                    % (len(failed) + len(not_extracted), ", ".join(what)))
    out = [(os.path.join(job_dir, SUMMARY), NODE_SUMMARY)] + nodes
    return out + (Extraction.nodes(job_dir) if ex is not None else [])


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
        nodes = run_match(sup, job_dir, threads, opts)
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
