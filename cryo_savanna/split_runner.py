#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its "Split off zero-weight particles"
# jobs, and travels with it (with runner_common.py, whose helpers it uses).
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Split off the particles a refinement stopped on (all-zero weights), as a RELION External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/split_runner.py zero_weights

(a copy of this file in the project, beside a copy of runner_common.py). RELION runs it as
``... --o <job dir> [--in_* F]... --j <threads> --in_job <failed job> ``. It needs nothing but the standard
library (Python 3.6+: RELION may start it with the system python3).

relion_refine (3D auto-refine, 3D and 2D classification, initial model, multi-body) stops when a particle
image gets no weight in any orientation: RELION 5.1's ERRFILTEREDZERO ("A particle image was compared to the
reference and resulted in all-zero weights"), ERRNOSIGNIFS and ERRSUMWEIGHTZERO (src/error.h, raised in
src/acc/acc_ml_optimiser_impl.h), and the CPU code's "zero sum of weights". Before the error, run.err names
the particle:

- the GPU code: `` fn_img= <rlnImageName>``, or the particle's copy on the scratch disk
  (``NNNNNN@<scratch>/opticsgroupG_particles.mrcs``, ``<scratch>/opticsgroupG_particleN.mrcs`` for 2D-stack
  subtomograms, ``.mrc`` for 3D ones: Experiment::getImageNameOnScratch, the N-th particle of optics group G);
- the CPU code: `` part_id= N``, the particle's index in RELION's particle list (0-based).

RELION reads the particles in the order of the STAR file, stable-sorted on rlnMicrographName (rlnTomoName
for 3D subtomograms) when the column is there (Experiment::read), and numbers optics groups by their row in
data_optics (ObservationModel::sortOpticsGroups): the particle list and the scratch copies follow that order.

The job reads the failed job's input particles (job.star: the particles, or for tomography the optimisation
set or the direct entries; else the command in note.txt) and writes, in its own directory,

- accepted_particles.star: every other particle,
- rejected_particles.star: the particles run.err names (with ``--reject_tomograms Yes``, every particle of
  their tomograms, rlnTomoName; with ``--reject_micrographs Yes``, of their micrographs, rlnMicrographName),
- for tomography, optimisation_set.star: the failed job's optimisation set (or its tomograms and trajectories)
  with accepted_particles.star as its particles,

each line of the input copied as it is (the optics and general blocks into both files). They are the job's
output nodes. run.err keeps what every attempt of the job printed (RELION appends to it), so all the particles
of all the attempts are split off at once.
"""

import os
import re
import shlex
import sys

if __package__:  # imported from cryo-savanna
    from cryo_savanna import runner_common as C
else:  # run in a project: the copy of runner_common.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import runner_common as C  # type: ignore[no-redef]

Usage = C.Usage

ZERO_WEIGHTS = "zero_weights"
MODES = (ZERO_WEIGHTS,)
ACCEPTED = "accepted_particles.star"
REJECTED = "rejected_particles.star"
OPTIMISATION_SET = "optimisation_set.star"
NODE_PARTICLES = "ParticleGroupMetadata.star.relion"
NODE_OPTIMISATION_SET = "TomoOptimisationSet.star.relion"
WHOLE_TOMOGRAMS = "reject_tomograms"  # option: Yes rejects every particle of the tomograms of those named
WHOLE_MICROGRAPHS = "reject_micrographs"  # option (SPA): Yes rejects every particle of their micrographs
# what each option rejects whole: (option, the column, what one is called)
WHOLE = ((WHOLE_TOMOGRAMS, "rlnTomoName", "tomogram"), (WHOLE_MICROGRAPHS, "rlnMicrographName", "micrograph"))

WHO = "split particles"  # say(): " ++++ split particles: ..."
say = C.say

# relion_refine's messages for a particle with no weight in any orientation (src/error.h, ml_optimiser.cpp)
GPU_ERRORS = (
    "No orientation was found as better than any other",  # ERRFILTEREDZERO
    "resulted in all-zero",  # ERRFILTEREDZERO, its second line
    "The number of contributing orientations for an image",  # ERRNOSIGNIFS
    "The sum of weights for all orientations was",  # ERRSUMWEIGHTZERO
)
CPU_ERROR = "zero sum of weights"
FN_IMG = re.compile(r"^\s*fn_img=\s*(\S+)\s*$")  # the GPU code's line (not " fn_img= X bg_avg= ...")
PART_ID = re.compile(r"^\s*part_id=\s*(\d+)\s*$")  # the CPU code's (not the GPU code's " op.part_id= ")
SCRATCH_STACK = re.compile(r"^(\d+)@(?:.*/)?opticsgroup(\d+)_particles\.mrcs$")
SCRATCH_SUBTOMO = re.compile(r"^(?:.*/)?opticsgroup(\d+)_particle(\d+)\.mrcs?$")

PARTICLES_FILE = "rlnTomoParticlesFile"
TOMOGRAMS_FILE = "rlnTomoTomogramsFile"
TRAJECTORIES_FILE = "rlnTomoTrajectoriesFile"
IMAGE_NAME = "rlnImageName"


def job_folder(value):
    """A job's folder as RELION names jobs: relative to the project, with a slash at the end."""
    v = C.text({"v": value}, "v")
    if not v:
        raise Usage("no failed job given (--in_job)")
    v = _rel(v)
    return v if v.endswith("/") else v + "/"


def _rel(path):
    p = C._rel(path)
    while p.startswith("./"):
        p = p[2:]
    return p


# ---------------------------------------------------------------------------- run.err

def read_errors(job):
    """What run.err of the job says: (the error was found, [fn_img names], [part_id numbers]), in the order
    RELION printed them, each once."""
    path = os.path.join(job, "run.err")
    found, names, ids = False, [], []
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if not found and (CPU_ERROR in line or any(e in line for e in GPU_ERRORS)):
                    found = True
                m = FN_IMG.match(line)
                if m and m.group(1) not in names:
                    names.append(m.group(1))
                    continue
                m = PART_ID.match(line)
                if m and int(m.group(1)) not in ids:
                    ids.append(int(m.group(1)))
    except OSError as e:
        raise Usage("cannot read %s: %s" % (path, e))
    return found, names, ids


# ---------------------------------------------------------------------------- the failed job's inputs

def _job_star(job):
    """(is tomo, {option: value}) of the job's job.star; (None, {}) without one."""
    path = os.path.join(job, "job.star")
    if not os.path.isfile(path):
        return None, {}
    blocks = C.read_star(path)
    head = blocks.get("job")
    is_tomo = head is not None and head.pairs.get("rlnJobIsTomo", "0").lower() in ("1", "true", "yes")
    values = blocks.get("joboptions_values")
    opts = {}
    if values is not None and values.is_loop:
        for row in values.dicts():
            opts[row.get("rlnJobOptionVariable", "")] = row.get("rlnJobOptionValue", "")
    return is_tomo, opts


def _command(job):
    """The words of the last relion_refine command in the job's note.txt (empty without one)."""
    try:
        with open(os.path.join(job, "note.txt"), "r", errors="replace") as fh:
            lines = [ln for ln in fh if "relion_refine" in ln]
    except OSError:
        return []
    if not lines:
        return []
    try:
        return shlex.split(lines[-1])
    except ValueError:
        return lines[-1].split()


def _flag(words, flag):
    return words[words.index(flag) + 1] if flag in words[:-1] else ""


def _star_labels(path):
    """The labels of a STAR file's first block with any (its first 200 lines)."""
    labels = set()
    try:
        with open(path, "r", errors="replace") as fh:
            for n, line in enumerate(fh):
                if n > 200:
                    break
                s = line.strip()
                if s.startswith("_rln"):
                    labels.add(s.split()[0][1:])
    except OSError:
        pass
    return labels


def _is_optimisation_set(path):
    return PARTICLES_FILE in _star_labels(path)


def optimisation_set_values(path):
    """(block name, {label: value}) of an optimisation set (a list, or a loop with one row)."""
    blocks = C.read_star(path)
    for b in blocks.values():
        if b.pairs:
            return b.name, dict(b.pairs)
        if b.is_loop and b.rows:
            return b.name, dict(zip(b.columns, b.rows[0]))
    return "", {}


def failed_job_inputs(job):
    """The input of the failed job: {"particles", "optimisation_set", "tomograms", "trajectories",
    "is_tomo"} (paths relative to the project; "" when not given). From job.star, else note.txt."""
    is_tomo, opts = _job_star(job)
    out = {"particles": "", "optimisation_set": "", "tomograms": "", "trajectories": "", "is_tomo": bool(is_tomo)}
    if is_tomo and opts.get("use_direct_entries", "No") != "Yes" and opts.get("in_optimisation", "").strip():
        out["optimisation_set"] = opts["in_optimisation"].strip()
    elif is_tomo and opts.get("in_particles", "").strip():
        out["particles"] = opts["in_particles"].strip()
        out["tomograms"] = opts.get("in_tomograms", "").strip()
        out["trajectories"] = opts.get("in_trajectories", "").strip()
    elif opts.get("fn_img", "").strip():
        out["particles"] = opts["fn_img"].strip()
    if not out["particles"] and not out["optimisation_set"]:  # no job.star, or a job type without these
        words = _command(job)
        out["optimisation_set"] = _flag(words, "--ios")
        out["particles"] = _flag(words, "--i")
        out["tomograms"] = _flag(words, "--tomograms")
        out["trajectories"] = _flag(words, "--trajectories")
        if not out["particles"] and not out["optimisation_set"] and _flag(words, "--continue"):
            # a continued run: the particles its optimiser file names
            optimiser = _flag(words, "--continue")
            try:
                general = C.read_star(optimiser).get("optimiser_general")
            except OSError:
                general = None
            if general is not None:
                out["particles"] = general.pairs.get("rlnExperimentalDataStarFile", "")
    if out["particles"] and not out["optimisation_set"] and _is_optimisation_set(out["particles"]):
        out["optimisation_set"], out["particles"] = out["particles"], ""  # --i of a tomo job: an optimisation set
    if out["optimisation_set"]:
        out["is_tomo"] = True
        if not os.path.isfile(out["optimisation_set"]):
            raise Usage("the optimisation set of %s is not there: %s" % (job, out["optimisation_set"]))
        _, values = optimisation_set_values(out["optimisation_set"])
        if not out["particles"]:
            out["particles"] = values.get(PARTICLES_FILE, "")
    if out["tomograms"] or out["trajectories"]:
        out["is_tomo"] = True
    for k in ("particles", "optimisation_set", "tomograms", "trajectories"):
        out[k] = _rel(out[k]) if out[k] else ""
    if not out["particles"]:
        raise Usage("cannot tell which particles %s refined: no input particles in its job.star or note.txt" % job)
    if not os.path.isfile(out["particles"]):
        raise Usage("the input particles of %s are not there: %s" % (job, out["particles"]))
    return out


# ---------------------------------------------------------------------------- the particles STAR file

def _is_row(s):
    return bool(s) and s[0] not in "#_" and not s.startswith("data_") and not s.startswith("loop_")


def _particle_table(path):
    """Which loop holds the particles: (block name, its columns). data_particles, else the first loop with
    rlnImageName (a STAR file of RELION 3.0)."""
    block, columns, in_header, found = None, [], False, []
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("data_"):
                if found and block is not None:
                    break
                block, columns, in_header = s[5:].split()[0] if len(s) > 5 else "", [], False
            elif s.startswith("loop_"):
                in_header, columns = True, []
            elif s.startswith("_") and in_header:
                columns.append(s[1:].split()[0])
            elif _is_row(s) and in_header:
                in_header = False
                if block == "particles":
                    return block, columns
                if IMAGE_NAME in columns and not found:
                    found = [block, list(columns)]
    if found:
        return found[0], found[1]
    raise Usage("%s has no table of particles (no data_particles, no rlnImageName)" % path)


def read_particles(path, labels):
    """(table name, its columns, the values of ``labels`` for every particle in file order; optics rows as
    {label: value})."""
    table, columns = _particle_table(path)
    labels = [x for x in labels if x == IMAGE_NAME or x in columns]
    if IMAGE_NAME not in columns:
        raise Usage("%s: the particles have no rlnImageName (were they extracted?)" % path)
    at = [columns.index(x) for x in labels]
    rows, optics = [], []
    block, in_loop, optics_cols = None, False, []
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("data_"):
                block, in_loop = s[5:].split()[0] if len(s) > 5 else "", False
                optics_cols = []
            elif s.startswith("loop_"):
                in_loop = True
            elif s.startswith("_") and block == "optics":
                optics_cols.append(s[1:].split()[0])
            elif in_loop and _is_row(s):
                if block == table:
                    toks = C._tokens(s)
                    rows.append(tuple(toks[i] if i < len(toks) else "" for i in at))
                elif block == "optics":
                    optics.append(dict(zip(optics_cols, C._tokens(s))))
    return table, labels, rows, optics


def relion_order(rows, labels, optics):
    """The particles in RELION's order (Experiment::read): the indices of ``rows``, stable-sorted on
    rlnMicrographName (rlnTomoName for 3D subtomograms) when the particles have it."""
    dims = {o.get("rlnImageDimensionality", "2") for o in optics}
    key = "rlnTomoName" if dims == {"3"} else "rlnMicrographName"
    if key not in labels:
        return list(range(len(rows)))
    at = labels.index(key)
    return sorted(range(len(rows)), key=lambda i: rows[i][at])


def rejected_rows(path, names, ids, whole=None):
    """(indices of the particles rows that run.err names, in file order; what it named that is not in the file;
    the number of particles; the groups rejected whole). With ``whole`` (a column: rlnTomoName,
    rlnMicrographName), every particle that has the value of one of those particles there."""
    labels = [IMAGE_NAME, "rlnOpticsGroup", "rlnMicrographName", "rlnTomoName"]
    table, labels, rows, optics = read_particles(path, labels)
    by_name = {}
    for i, r in enumerate(rows):
        by_name.setdefault(r[0], []).append(i)
    rejected, missing = set(), []
    cache = {}

    def ordered():
        if "order" not in cache:
            cache["order"] = relion_order(rows, labels, optics)
        return cache["order"]

    def nth_of_group(group, n):
        """The n-th particle (1-based) of optics group ``group`` (1-based: its row in data_optics)."""
        if "groups" not in cache:
            number = dict((o.get("rlnOpticsGroup", str(i + 1)), i + 1) for i, o in enumerate(optics))
            at = labels.index("rlnOpticsGroup") if "rlnOpticsGroup" in labels else None
            groups = {}
            for i in ordered():
                groups.setdefault(number.get(rows[i][at], 0) if at is not None else 1, []).append(i)
            cache["groups"] = groups
        members = cache["groups"].get(group, [])
        return members[n - 1] if 1 <= n <= len(members) else None

    for name in names:
        if name in by_name:
            rejected.update(by_name[name])
            continue
        m = SCRATCH_STACK.match(name)
        hit = None
        if m:
            hit = nth_of_group(int(m.group(2)), int(m.group(1)))
        else:
            m = SCRATCH_SUBTOMO.match(name)
            if m:
                hit = nth_of_group(int(m.group(1)), int(m.group(2)))
        if hit is None:
            missing.append(name)
        else:
            rejected.add(hit)
    if ids:
        order = ordered()
        for n in ids:  # (the CPU code: the particle's index in RELION's list)
            if 0 <= n < len(order):
                rejected.add(order[n])
            else:
                missing.append("part_id= %d" % n)
    groups = []
    if whole and rejected:
        if whole not in labels:
            raise Usage("%s has no %s: its particles cannot be rejected by %s" % (path, whole, whole[3:]))
        at = labels.index(whole)
        values = set(rows[i][at] for i in rejected)
        rejected = set(i for i, r in enumerate(rows) if r[at] in values)
        groups = sorted(values)
    return sorted(rejected), missing, len(rows), groups


def write_split(path, rejected, accepted_path, rejected_path):
    """Copy the STAR file into the two, its particle rows by ``rejected`` (indices, in file order), every
    other line into both."""
    table, _ = _particle_table(path)
    out = [accepted_path, rejected_path]
    tmps = ["%s.%d.tmp" % (p, os.getpid()) for p in out]
    bad = set(rejected)
    try:
        with open(path, "r", errors="replace") as src, open(tmps[0], "w") as acc, open(tmps[1], "w") as rej:
            block, in_loop, row = None, False, 0
            for raw in src:
                s = raw.strip()
                if s.startswith("data_"):
                    block, in_loop = s[5:].split()[0] if len(s) > 5 else "", False
                elif s.startswith("loop_"):
                    in_loop = True
                elif in_loop and block == table and _is_row(s):
                    (rej if row in bad else acc).write(raw if raw.endswith("\n") else raw + "\n")
                    row += 1
                    continue
                acc.write(raw)
                rej.write(raw)
        for tmp, p in zip(tmps, out):
            os.replace(tmp, p)
    finally:
        for tmp in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)


def write_optimisation_set(path, inputs, particles):
    """The failed job's optimisation set (or its tomograms and trajectories) with these particles."""
    name, values = "", {}
    if inputs["optimisation_set"]:
        name, values = optimisation_set_values(inputs["optimisation_set"])
    if inputs["tomograms"]:
        values[TOMOGRAMS_FILE] = inputs["tomograms"]
    if inputs["trajectories"]:
        values[TRAJECTORIES_FILE] = inputs["trajectories"]
    values[PARTICLES_FILE] = particles
    first = [k for k in (PARTICLES_FILE, TOMOGRAMS_FILE, TRAJECTORIES_FILE) if k in values]
    ordered = dict((k, values[k]) for k in first + [k for k in values if k not in first])
    C.write_star(path, [(name, ordered, None, [])])
    return ordered


# ---------------------------------------------------------------------------- the job

def run(argv):
    C.WHO = WHO
    mode, job_dir, _threads, opts = C.parse(argv, MODES)
    failed = job_folder(opts.get("in_job", ""))
    if not os.path.isdir(failed):
        raise Usage("the failed job's folder is not there: %s" % failed)
    if os.path.realpath(failed) == os.path.realpath(job_dir):
        raise Usage("--in_job is this job itself: give the job that failed")
    found, names, ids = read_errors(failed)
    if not found:
        raise Usage("%srun.err does not show RELION's all-zero weights error (\"A particle image was compared to "
                    "the reference and resulted in all-zero weights\", or \"zero sum of weights\")" % failed)
    if not names and not ids:
        raise Usage("%srun.err has the all-zero weights error but names no particle (no 'fn_img=' or 'part_id=' "
                    "line)" % failed)
    inputs = failed_job_inputs(failed)
    particles = inputs["particles"]
    say("%s: %d particle%s named in %srun.err; input particles %s"
        % (failed, len(names) + len(ids), "" if len(names) + len(ids) == 1 else "s", failed, particles))
    whole = [(column, what) for option, column, what in WHOLE if C.yes(opts, option)]
    if len(whole) > 1:
        raise Usage("reject whole tomograms or whole micrographs, not both")
    column, what = whole[0] if whole else (None, "")
    rejected, missing, total, groups = rejected_rows(particles, names, ids, column)
    for m in missing:
        say("not in %s: %s" % (particles, m))
    if groups:
        say("every particle of %d %s%s rejected: %s"
            % (len(groups), what, "" if len(groups) == 1 else "s", " ".join(groups)))
    if not rejected:
        raise Usage("none of the particles %srun.err names is in %s (was the job's input changed since?)"
                    % (failed, particles))
    if len(rejected) == total:
        raise Usage("every particle of %s would be rejected" % particles)
    accepted_path, rejected_path = job_dir + ACCEPTED, job_dir + REJECTED
    write_split(particles, rejected, accepted_path, rejected_path)
    nodes = [(accepted_path, NODE_PARTICLES), (rejected_path, NODE_PARTICLES)]
    say("%d of %d particles rejected (%s), %d accepted (%s)"
        % (len(rejected), total, rejected_path, total - len(rejected), accepted_path))
    if inputs["is_tomo"]:
        if inputs["optimisation_set"] or inputs["tomograms"]:
            path = job_dir + OPTIMISATION_SET
            values = write_optimisation_set(path, inputs, accepted_path)
            nodes.append((path, NODE_OPTIMISATION_SET))
            say("optimisation set %s: %s" % (path, ", ".join("%s %s" % kv for kv in values.items())))
        else:
            say("no optimisation set written: the failed job names no tomograms")
    C.write_output_nodes(job_dir, nodes)
    C._touch(os.path.join(job_dir, C.SUCCESS))
    return 0


if __name__ == "__main__":
    C.main(run)
