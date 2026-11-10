#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its pytom jobs, and travels with it.
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

(a copy of this file in the project). RELION runs it as

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
new thresholds (cryo-savanna's threshold viewer calls it).

Like every External job it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED and, before success,
RELION_OUTPUT_NODES.star. RELION_JOB_ABORT_NOW and SIGTERM/SIGINT/SIGHUP (scancel) stop the pytom
processes (their process groups: SIGTERM, then SIGKILL).
"""

import json
import re
import os
import shlex
import signal
import struct
import subprocess
import sys
import time
import traceback

MATCH, EXTRACT = "match", "extract"
EXTERNAL_INPUTS = ("--in_movies", "--in_mics", "--in_parts", "--in_coords", "--in_3dref", "--in_mask")

SUCCESS, FAILURE, ABORTED = "RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"
ABORT_NOW = "RELION_JOB_ABORT_NOW"
OUTPUT_NODES = "RELION_OUTPUT_NODES.star"
KILL_GRACE_S = 30
POLL_S = 1.0

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
TOMO_NAME = "rlnTomoName"
CENTERED = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst")
LCC = "rlnLCCmax"

TOMOGRAM_KINDS = {
    "Reconstructed": "rlnTomoReconstructedTomogram",
    "Denoised": "rlnTomoReconstructedTomogramDenoised",
    "Half 1": "rlnTomoReconstructedTomogramHalf1",
    "Half 2": "rlnTomoReconstructedTomogramHalf2",
}
ORIENTATIONS = {  # how the candidates' angles are written
    "rlnAngleRot/Tilt/Psi": ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"),
    "rlnTomoSubtomogramRot/Tilt/Psi": ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"),
    "Not written": (),
}
GPU_PER_TOMOGRAM, GPU_SPLIT = "One tomogram per GPU", "Each tomogram over all GPUs"
MASK_FILE, MASK_CREATE = "Input file", "Create sphere or ellipsoid"
CUT_AUTO, CUT_FIXED = "Automatic (false positives)", "Fixed cut-off"
TMASK_JOB, TMASK_OTHER, TMASK_IGNORE = "The match job's", "Other mask(s)", "None (ignore masks)"

# clips a resampled mask to [0, 1] and inverts it on request (run with the python of the pytom environment)
MASK_FIX = ("import sys, numpy as np, mrcfile\n"
            "with mrcfile.open(sys.argv[1], mode='r+') as m:\n"
            "    d = np.clip(np.asarray(m.data, dtype=np.float32), 0, 1)\n"
            "    m.set_data(1 - d if sys.argv[2] == '1' else d)\n")


class Usage(Exception):
    pass


class Aborted(Exception):
    pass


def say(text):
    print(" ++++ pytom: " + text, flush=True)


# ---------------------------------------------------------------------------- arguments

def parse(argv):
    """argv[1:] -> (mode, job dir, number of threads, {option: value})."""
    if not argv or argv[0] not in (MATCH, EXTRACT):
        raise Usage("the first argument must be 'match' or 'extract'")
    mode, rest = argv[0], list(argv[1:])
    if len(rest) >= 2 and rest[-2] == "--pipeline_control":
        rest = rest[:-2]
    params, opts = {}, {}
    i, threads = 0, 1
    while i < len(rest):
        flag = rest[i]
        if flag == "--j":
            try:
                threads = max(1, int(rest[i + 1]))
            except (IndexError, ValueError):
                raise Usage("--j needs a whole number")
            i += 2
            break
        if flag not in ("--o",) + EXTERNAL_INPUTS or i + 1 >= len(rest):
            raise Usage("unexpected argument before --j: %s" % flag)
        params[flag] = rest[i + 1]
        i += 2
    else:
        raise Usage("no --j in the arguments (not started by RELION?)")
    while i < len(rest):  # the job's options: --key value
        flag = rest[i]
        if not flag.startswith("--") or i + 1 >= len(rest):
            raise Usage("job options are --key value pairs, got %r" % flag)
        opts[flag[2:]] = rest[i + 1]
        i += 2
    job_dir = params.get("--o", "")
    if not job_dir:
        raise Usage("no --o <job directory>")
    if not job_dir.endswith("/"):
        job_dir += "/"
    return mode, job_dir, threads, opts


def yes(opts, key):
    return opts.get(key, "No") == "Yes"


def text(opts, key):
    return opts.get(key, "").strip()


def number(opts, key, kind=float, required=False, what=""):
    v = text(opts, key)
    if not v:
        if required:
            raise Usage("%s is required" % (what or key))
        return None
    try:
        return kind(v)
    except ValueError:
        raise Usage("%s: %r is not a %s" % (what or key, v, "whole number" if kind is int else "number"))


def words(opts, key):
    try:
        return shlex.split(text(opts, key))
    except ValueError as e:
        raise Usage("%s: %s" % (key, e))


def num(v):
    """A number as pytom gets it (no trailing .0 for whole numbers)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def conda_prefix(opts):
    """The start of every pytom command: ``conda run`` in the job's environment (nothing: on PATH)."""
    env = text(opts, "pytom_env")
    if not env:
        return []
    conda = text(opts, "conda") or os.environ.get("CONDA_EXE") or "conda"
    return shlex.split(conda) + ["run", "--no-capture-output", "-p" if "/" in env else "-n", env]


def selected_names(opts):
    return set(words(opts, "tomo_names"))


# ---------------------------------------------------------------------------- environment modules

# makes "module" available in a plain bash: a job's shell often has it already (exported by the
# login shell); else the init script of Lmod or Environment Modules
MODULE_INIT = """type module >/dev/null 2>&1 || for f in "${MODULESHOME:-/nonexistent}/init/bash" \\
    /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
  [ -f "$f" ] && . "$f" >/dev/null 2>&1 && type module >/dev/null 2>&1 && break
done
"""
_BEFORE, _AFTER = b"\0__savanna_env_before__\0", b"\0__savanna_env_after__\0"
_SHELL_VARS = ("_", "SHLVL", "PWD", "OLDPWD", "BASH_EXECUTION_STRING")


def _env_block(data):
    out = {}
    for item in data.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep and k:
            out[k.decode("utf-8", "surrogateescape")] = v.decode("utf-8", "surrogateescape")
    return out


def load_modules(opts):
    """module load the job's CUDA module(s) in a bash, and take over what that changed in the
    environment, so every pytom process (and conda run) sees that CUDA."""
    mods = words(opts, "cuda_module")
    if not mods:
        return
    script = (MODULE_INIT + 'type module >/dev/null 2>&1 || { echo "no module command here (Lmod or '
              'Environment Modules)" >&2; exit 127; }\n'
              "printf '%s'; env -0; module load \"$@\" >&2 || exit 3; printf '%s'; env -0\n"
              % (_BEFORE.decode().replace("\0", "\\0"), _AFTER.decode().replace("\0", "\\0")))
    what = "module load " + " ".join(mods)
    try:
        r = subprocess.run(["bash", "-c", script, "bash"] + mods, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL, timeout=600)
    except OSError as e:
        raise Usage("%s: cannot run bash: %s" % (what, e))
    except subprocess.TimeoutExpired:
        raise Usage("%s: no answer in 10 minutes" % what)
    err = r.stderr.decode("utf-8", "replace").strip()
    i, j = r.stdout.find(_BEFORE), r.stdout.find(_AFTER)
    # Environment Modules can exit with 0 after an error: its message says ERROR
    if r.returncode != 0 or i < 0 or j < i or re.search(r"(?i)\berror\b", err):
        raise Usage("%s failed%s (set 'CUDA module' on the Compute tab, or leave it empty when CUDA needs no "
                    "module)" % (what, ":\n" + err if err else " (exit code %d)" % r.returncode))
    before, after = _env_block(r.stdout[i + len(_BEFORE):j]), _env_block(r.stdout[j + len(_AFTER):])
    changed = 0
    for k, v in after.items():
        if k not in _SHELL_VARS and before.get(k) != v:
            os.environ[k] = v
            changed += 1
    for k in before:
        if k not in after and k not in _SHELL_VARS:
            os.environ.pop(k, None)
            changed += 1
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or os.environ.get("CUDA_ROOT") or ""
    say("%s: %d environment variables set%s" % (what, changed, ", CUDA in " + cuda if cuda else ""))
    if err:
        say(err)


# ---------------------------------------------------------------------------- STAR files

class Block(object):
    def __init__(self, name):
        self.name, self.columns, self.rows, self.pairs, self.is_loop = name, [], [], {}, False

    def dicts(self):
        return [dict(zip(self.columns, r)) for r in self.rows]


def _tokens(line):
    if '"' in line or "'" in line:
        try:
            return shlex.split(line, posix=True)
        except ValueError:
            pass
    return line.split()


def read_star(path):
    """{block name: Block} of a STAR file (loops and label/value lists)."""
    blocks = {}
    cur, in_loop, in_header = None, False, False
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#":
                continue
            if line.startswith("data_"):
                cur = Block(line[5:].split()[0] if len(line) > 5 else "")
                blocks[cur.name] = cur
                in_loop = in_header = False
                continue
            if cur is None:
                continue
            if line.startswith("loop_"):
                cur.is_loop, in_loop, in_header = True, True, True
                continue
            if line[0] == "_":
                parts = line[1:].split(None, 1)
                if in_loop and in_header:
                    cur.columns.append(parts[0])
                else:
                    toks = _tokens(parts[1]) if len(parts) > 1 else []
                    cur.pairs[parts[0]] = toks[0] if toks else ""
                    in_loop = False
                continue
            if in_loop:
                in_header = False
                cur.rows.append(_tokens(line))
    return blocks


def loop_block(path, name=None):
    blocks = read_star(path)
    b = blocks.get(name) if name is not None else None
    if b is None or not b.is_loop:
        b = next((b for b in blocks.values() if b.is_loop), None)
    if b is None:
        raise Usage("%s: no table" % path)
    return b


def _fmt(v):
    if isinstance(v, float):
        s = "%.6f" % v
    else:
        s = str(v)
    if s == "":
        return '""'
    if any(c.isspace() for c in s):
        return '"%s"' % s
    return s


def write_star(path, blocks):
    """blocks: (name, {label: value} or None, columns or None, rows). Written to a temp file, then renamed."""
    out = ["", "# version 50001", ""]
    for name, pairs, columns, rows in blocks:
        out += ["data_" + name, ""]
        if pairs:
            width = max(len(k) for k in pairs) + 2
            out += ["_%s %s" % (k.ljust(width), _fmt(v)) for k, v in pairs.items()]
            out.append("")
        if columns is not None:
            out.append("loop_")
            out += ["_%s #%d" % (c, i + 1) for i, c in enumerate(columns)]
            out += [" ".join(_fmt(v) for v in r) for r in rows]
            out.append("")
        out.append("")
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as fh:
        fh.write("\n".join(out))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- MRC headers

def mrc_header(path):
    """(nx, ny, nz, voxel size in Å) from an MRC header."""
    with open(path, "rb") as fh:
        h = fh.read(56)
    if len(h) < 52:
        raise Usage("%s: not an MRC file" % path)
    nx, ny, nz = struct.unpack("<3i", h[:12])
    if not (0 < nx < 1 << 20 and 0 < ny < 1 << 20 and 0 < nz < 1 << 20):
        nx, ny, nz = struct.unpack(">3i", h[:12])
        mx = struct.unpack(">i", h[28:32])[0]
        cx = struct.unpack(">f", h[40:44])[0]
    else:
        mx = struct.unpack("<i", h[28:32])[0]
        cx = struct.unpack("<f", h[40:44])[0]
    return nx, ny, nz, (cx / mx if mx else 0.0)


# ---------------------------------------------------------------------------- processes

class Supervisor(object):
    """Starts commands in process groups of their own, stops them all on abort or a signal."""

    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.running = {}  # pid -> Popen
        self.stopped = None  # signal number, once the runner was told to stop
        self.aborted = False
        for name in ("SIGTERM", "SIGINT", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, lambda signum, frame: self._signal(signum))

    def _signal(self, signum):
        if self.stopped is None:
            self.stopped = signum
            self.kill_all(signal.SIGTERM)
            try:  # now: the next signal may be a SIGKILL that leaves no time for it
                _touch(os.path.join(self.job_dir, ABORTED))
            except OSError:
                pass

    def start(self, cmd, log_path):
        line = " ".join(shlex.quote(c) for c in cmd)
        say("running: " + line)
        fh = open(log_path, "a")
        fh.write("\n ++++ %s\n ++++ %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
        fh.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            fh.write("cannot start %s: %s\n" % (cmd[0], e))
            fh.close()
            raise Usage("cannot start %s: %s (is pytom's conda environment right?)" % (cmd[0], e))
        proc.log = fh
        self.running[proc.pid] = proc
        return proc

    def finished(self, proc):
        rc = proc.poll()
        if rc is None:
            return None
        proc.log.close()
        self.running.pop(proc.pid, None)
        return rc

    def kill_all(self, sig):
        for proc in list(self.running.values()):
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                pass

    def check(self):
        """Raise Aborted when RELION or a signal asked to stop (after stopping the processes)."""
        if self.stopped is None and not self.aborted and os.path.exists(os.path.join(self.job_dir, ABORT_NOW)):
            self.aborted = True
            say("abort requested: stopping pytom")
            self.kill_all(signal.SIGTERM)
        if self.stopped is not None or self.aborted:
            end = time.time() + KILL_GRACE_S
            while self.running and time.time() < end:
                for proc in list(self.running.values()):
                    self.finished(proc)
                time.sleep(0.2)
            self.kill_all(signal.SIGKILL)
            raise Aborted()

    def run(self, cmd, log_path):
        """Run one command to its end (abortable); its exit code."""
        proc = self.start(cmd, log_path)
        while True:
            rc = self.finished(proc)
            if rc is not None:
                return rc
            self.check()
            time.sleep(POLL_S)


def run_parallel(sup, tasks, slots, on_change=None):
    """Run tasks in parallel slots. A task is (key, factory); factory(slot) returns a generator that
    yields (command, log file), gets each command's exit code back, and returns its result. Returns
    {key: result}, None for a task that failed (an exception in it is printed)."""
    pending, running, results = list(tasks), {}, {}

    def advance(si, key, gen, rc):
        try:
            cmd, log = gen.send(rc) if rc is not None else next(gen)
            running[si] = (key, gen, sup.start(cmd, log))
        except StopIteration as e:
            results[key] = e.value if e.value is not None else True
            running.pop(si, None)
            if on_change:
                on_change(key, results[key])
        except (Usage, OSError, ValueError, KeyError) as e:
            print("ERROR: %s: %s" % (key, e), file=sys.stderr, flush=True)
            results[key] = None
            running.pop(si, None)
            if on_change:
                on_change(key, None)
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            results[key] = None
            running.pop(si, None)
            if on_change:
                on_change(key, None)

    while pending or running:
        for si, slot in enumerate(slots):
            if si not in running and pending:
                key, factory = pending.pop(0)
                advance(si, key, factory(slot), None)
        for si in list(running):
            key, gen, proc = running[si]
            rc = sup.finished(proc)
            if rc is not None:
                advance(si, key, gen, rc)
        sup.check()
        if running:
            time.sleep(POLL_S)
    return results


def _touch(path):
    with open(path, "a"):
        os.utime(path, None)


def _rel(path):
    """A path relative to the project (the working directory), as RELION writes them."""
    return os.path.relpath(path) if os.path.isabs(path) else os.path.normpath(path)


def _link(target, link):
    """A relative symbolic link (replaced if it points elsewhere); False if the file system has none."""
    rel = os.path.relpath(os.path.abspath(target), os.path.dirname(os.path.abspath(link)))
    try:
        if os.path.islink(link):
            if os.readlink(link) == rel:
                return True
            os.remove(link)
        elif os.path.exists(link):
            os.remove(link)
        os.symlink(rel, link)
        return True
    except OSError:
        return False


def write_output_nodes(job_dir, nodes):
    lines = ["", "# version 50001", "", "data_output_nodes", "", "loop_",
             "_rlnPipeLineNodeName #1", "_rlnPipeLineNodeTypeLabel #2"]
    lines += ["%s %s" % (_fmt(p), t) for p, t in nodes]
    with open(os.path.join(job_dir, OUTPUT_NODES), "w") as fh:
        fh.write("\n".join(lines) + "\n\n")


# ---------------------------------------------------------------------------- match: inputs from RELION

def _f(row, label):
    try:
        v = float(row.get(label, ""))
    except ValueError:
        return None
    return v if v == v else None  # not NaN


def tomogram_rows(opts):
    """The rows of the input tomograms.star to search (filtered by tomo_names)."""
    star = text(opts, "in_tomograms")
    if not star or not os.path.isfile(star):
        raise Usage("input tomograms.star not found: %r" % star)
    b = loop_block(star, "global")
    if TOMO_NAME not in b.columns:
        raise Usage("%s: no rlnTomoName (not a RELION 5 tomogram set)" % star)
    rows = b.dicts()
    names = selected_names(opts)
    if names:
        missing = names - set(r[TOMO_NAME] for r in rows)
        if missing:
            raise Usage("not in %s: %s" % (star, " ".join(sorted(missing))))
        rows = [r for r in rows if r[TOMO_NAME] in names]
    if not rows:
        raise Usage("%s: no tomograms" % star)
    return star, b.columns, rows


def tomogram_of(row, opts):
    kind = text(opts, "tomo_kind") or "Reconstructed"
    col = TOMOGRAM_KINDS.get(kind, kind)
    path = row.get(col, "")
    if not path or path == "None":
        raise Usage("%s has no %s tomogram (%s)" % (row[TOMO_NAME], kind.lower(), col))
    path = path.split(":")[0] if path.endswith(":mrc") else path
    if not os.path.isfile(path):
        raise Usage("%s: tomogram not found: %s" % (row[TOMO_NAME], path))
    return path


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
        out.append((_f(t, angle_col), dose, defocus))
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
    pattern = text(opts, "tomogram_mask")
    if pattern:
        mask = pattern.replace("{name}", name)
        if not os.path.isfile(mask):
            raise Usage("%s: tomogram mask not found: %s" % (name, mask))
        files["tomogram_mask"] = mask
    return files


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
    for key, flag in (("non_spherical", "--non-spherical-mask"), ("mirror_template", "--mirror-template"),
                      ("invert_template_contrast", "--invert-template-contrast"),
                      ("random_phase", "--random-phase-correction"), ("half_precision", "--half-precision")):
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
        if yes(opts, "do_prep_mask"):
            out = os.path.join(job_dir, "mask.mrc")
            final = number(opts, "mask_angpix", what="Mask final pixel size") or tvoxel or voxel
            box = number(opts, "mask_box", int) or tx
            if sup.run(template_command(opts, mask, out, final, box, "mask"), logs) != 0 or not os.path.isfile(out):
                raise Usage("pytom_create_template.py failed on the mask (see %s)" % logs)
            fix = conda_prefix(opts) + ["python", "-c", MASK_FIX, out, "1" if yes(opts, "mask_invert") else "0"]
            if sup.run(fix, logs) != 0:
                raise Usage("could not clip%s the mask (see %s)" % (" and invert" if yes(opts, "mask_invert") else "", logs))
            mask = out
            nodes.append((out, NODE_MASK))
    mx, my, mz, _ = mrc_header(mask)
    if (mx, my, mz) != (tx, ty, tz):
        raise Usage("the mask (%d×%d×%d) and the template (%d×%d×%d) need the same box" % (mx, my, mz, tx, ty, tz))
    return template, mask, nodes


def gpu_ids(opts):
    """The GPU ids, separated by colons as in RELION (0:1:2:3); commas and spaces work too."""
    ids = [w for w in re.split(r"[\s,:;]+", text(opts, "gpu_ids")) if w]
    if not ids:  # every GPU this job sees
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis and vis not in ("NoDevFiles", "-1"):
            n = len([v for v in vis.split(",") if v.strip()])
        else:
            try:
                out = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, universal_newlines=True,
                                     timeout=60).stdout
            except (OSError, subprocess.SubprocessError):
                out = ""
            n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        if n == 0:
            raise Usage("no GPU found (CUDA_VISIBLE_DEVICES, nvidia-smi): give the GPU ids")
        ids = [str(i) for i in range(n)]
    try:
        return [int(i) for i in ids]
    except ValueError:
        raise Usage("GPU ids: whole numbers separated by colons, e.g. 0:1:2:3 (got %r)" % text(opts, "gpu_ids"))


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


def _first_error(lines, most=60):
    """The first traceback in a log, with the line before it ("Process Process-1:" when it was one of
    pytom's worker processes): the real error, which pytom's own "processes stopped unexpectedly" hides."""
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            start = i - 1 if i and lines[i - 1].startswith("Process ") else i
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("Traceback (most recent call")),
                       len(lines))
            return start, min(end, start + most)
    return None


def _show_tail(log, n=25):
    """The end of a failed pytom log in run.err, after the first error in it when that came earlier."""
    try:
        with open(log, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return
    out = []
    first = _first_error(lines)
    tail = max(0, len(lines) - n)
    if first is not None and first[0] < tail:
        start, end = first
        out += ["first error in the log (line %d):\n" % (start + 1)] + lines[start:end]
        out += ["...\n", "end of the log:\n"]
    out += lines[tail:]
    sys.stderr.write("".join("    | " + line if line.endswith("\n") else "    | " + line + "\n" for line in out))
    sys.stderr.flush()


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


def extract_command(opts, job_file, cut_off=None):
    cmd = conda_prefix(opts) + ["pytom_extract_candidates.py", "-j", job_file,
                                "-n", str(number(opts, "n_particles", int, True, "Number of particles"))]
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
        mask = text(opts, "tomogram_mask").replace("{name}", name)
        if not os.path.isfile(mask):
            raise Usage("%s: tomogram mask not found: %s" % (name, mask))
        return ["--tomogram-mask", mask]
    return []


def roc_command(opts, job_file):
    n = number(opts, "roc_particles", int) or 3 * number(opts, "n_particles", int, True, "Number of particles")
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


def write_selection(job_dir, thresholds=None):
    """particles.star: the candidates with rlnLCCmax at or above their tomogram's threshold (rlnCutOff
    becomes that threshold), and thresholds.star with ``thresholds`` ({tomogram: value or None}; None
    selects nothing) put in. Returns {tomogram: (candidates, selected)}."""
    cand = loop_block(os.path.join(job_dir, CANDIDATES), "particles")
    current = read_thresholds(job_dir)
    if thresholds:
        for name, thr in thresholds.items():
            if name in current:
                current[name] = (current[name][0], thr)
    cols = cand.columns
    li, ni = cols.index(LCC), cols.index(TOMO_NAME)
    ci = cols.index("rlnCutOff") if "rlnCutOff" in cols else None
    counts = dict((n, [0, 0]) for n in current)
    rows = []
    for r in cand.rows:
        name = r[ni]
        thr = current.get(name, (None, None))[1]
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
    table = [[n, "" if a is None else "%.6f" % a, "" if t is None else "%.6f" % t, counts[n][0], counts[n][1]]
             for n, (a, t) in current.items()]
    write_star(os.path.join(job_dir, THRESHOLDS),
               [("thresholds", None, [TOMO_NAME, COL_AUTO, COL_THRESHOLD, COL_CANDIDATES, COL_SELECTED], table)])
    return dict((n, tuple(c)) for n, c in counts.items())


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
    angles = ORIENTATIONS.get(text(opts, "orientations"), ORIENTATIONS["rlnAngleRot/Tilt/Psi"])
    tm, plots = os.path.join(job_dir, TM_DIR), os.path.join(job_dir, PLOTS_DIR)
    found = {}

    def task(row):
        name = row[TOMO_NAME]

        def factory(_slot):
            def gen():
                log = os.path.join(job_dir, LOG_DIR, name + ".log")
                job_file, stem = extract_job_file(row, job_dir)
                out = os.path.join(tm, stem + "_particles.star")
                rc = yield extract_command(opts, job_file, fixed), log
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
                    rc = yield extract_command(opts, job_file, low), log
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
                    rc = yield roc_command(opts, job_file), log
                    if rc != 0:
                        say("%s: pytom_estimate_roc.py failed (see %s); going on without it" % (name, log))
                    _move(os.path.join(tm, stem + "_roc.svg"), os.path.join(plots, name + "_roc.svg"))
                found[name] = (cut, to_relion5(cands, row, angles))
                say("%s: %d particle(s) above %s%s" % (name, len(sel), "the cut-off" if cut is None else "%.4f" % cut,
                                                         ", %d candidate(s)" % len(cands) if interactive else ""))
                return True
            return gen()
        return name, factory

    run_parallel(sup, [task(r) for r in ready], [None] * max(1, threads))
    failed = [r[TOMO_NAME] for r in ready if r[TOMO_NAME] not in found]
    if failed:
        raise Usage("extraction failed for: %s" % " ".join(failed))
    cols = candidate_columns(angles)
    table = []
    for r in ready:
        table += [[row.get(c, "") for c in cols] for row in found[r[TOMO_NAME]][1]]
    write_star(os.path.join(job_dir, CANDIDATES), [("particles", None, cols, table)])
    thr = [[r[TOMO_NAME], "" if found[r[TOMO_NAME]][0] is None else "%.6f" % found[r[TOMO_NAME]][0],
            "" if found[r[TOMO_NAME]][0] is None else "%.6f" % found[r[TOMO_NAME]][0], 0, 0] for r in ready]
    write_star(os.path.join(job_dir, THRESHOLDS),
               [("thresholds", None, [TOMO_NAME, COL_AUTO, COL_THRESHOLD, COL_CANDIDATES, COL_SELECTED], thr)])
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
    for f in (SUCCESS, FAILURE, ABORTED):
        if os.path.exists(os.path.join(job_dir, f)):
            os.remove(os.path.join(job_dir, f))
    sup = Supervisor(job_dir)
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


def _fail(argv):
    if "--o" in argv[:-1]:
        job_dir = argv[argv.index("--o") + 1]
        if os.path.isdir(job_dir):
            try:
                _touch(os.path.join(job_dir, FAILURE))
            except OSError:
                pass


def main():
    argv = sys.argv[1:]
    try:
        sys.exit(run(argv))
    except Usage as e:
        print("cryo-savanna pytom_runner: " + str(e), file=sys.stderr, flush=True)
        _fail(argv)
        sys.exit(2)
    except Exception:  # anything else: still an exit file, or RELION shows the job running for ever
        traceback.print_exc()
        sys.stderr.flush()
        _fail(argv)
        sys.exit(1)


if __name__ == "__main__":
    main()
