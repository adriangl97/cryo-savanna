#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its CLI tools, and travels with it.
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Run a RELION command-line program as an External job (cryo-savanna's "CLI tools" menu).

RELION runs an External job as

    <fn_exe> --o <job dir> [--in_movies|--in_mics|--in_parts|--in_coords|--in_3dref|--in_mask F]...
             [--<param label> <value>]... --j <threads> <additional arguments> [--pipeline_control <job dir>]

cryo-savanna sets fn_exe to "python3 .cryo-savanna/cli_runner.py <program>" (a copy of this file in
the project) and puts the program's own arguments in "Additional arguments", so this script gets
everything up to "--j N" from RELION and passes the rest to the program. The --in_* inputs are only there so RELION links the job to its inputs.
Parameters cryo-savanna may set: --mpi N (run <program>_mpi under $RELION_MPIRUN, default mpirun; at
most $RELION_ERROR_LOCAL_MPI processes, default 32, as RELION allows for local jobs)
and --stdout NAME (save what the program prints to NAME in the job directory).

As an External job must, it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED, and before
success RELION_OUTPUT_NODES.star with the files the program wrote in the job directory, so they
become outputs of the job that later jobs can use. RELION programs get --pipeline_control with a
private directory (<job dir>.cli/), so their own exit files do not end the job before the output
nodes are written; an abort (RELION_JOB_ABORT_NOW) is passed on to them, and programs that do not
stop by themselves are terminated (SIGTERM, then SIGKILL). The program runs in a session of its
own, so when the runner itself is told to stop (SIGTERM from scancel or kill, SIGINT, SIGHUP), it
passes that on to the program, writes RELION_JOB_EXIT_ABORTED at once (a SIGKILL may follow) and
kills the program if it is still there KILL_GRACE_S later.

Standard library only, Python 3.6+: RELION may start it on a compute node with the system python3.
"""

import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
import traceback

# Programs that take positional arguments (no IOParser, so no --pipeline_control).
POSITIONAL_TOOLS = {
    "relion_merge_particles", "relion_mrc2vtk", "relion_star_printtable", "relion_star_loopheader",
    "relion_star_datablock_stack", "relion_star_datablock_singlefiles", "relion_star_datablock_ctfdat",
}
EXTERNAL_INPUTS = ("--in_movies", "--in_mics", "--in_parts", "--in_coords", "--in_3dref", "--in_mask")
OWN_PARAMS = ("--mpi", "--stdout")

SUCCESS, FAILURE, ABORTED = "RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"
ABORT_NOW = "RELION_JOB_ABORT_NOW"
OUTPUT_NODES = "RELION_OUTPUT_NODES.star"
CONTROL_DIR = ".cli"
ABORT_GRACE_S = 30
KILL_GRACE_S = 30
DEFAULT_LOCAL_MPI_LIMIT = 32  # RELION's DEFAULTWARNINGLOCALMPI
MAX_NODES = 200
BOOKKEEPING = {"run.out", "run.err", "note.txt", "job.star", "run_submit.script", OUTPUT_NODES}


class Usage(Exception):
    pass


def parse(argv):
    """argv[1:] -> (program, job dir, {param: value}, program arguments)."""
    if not argv:
        raise Usage("no program given")
    program, rest = argv[0], list(argv[1:])
    if len(rest) >= 2 and rest[-2] == "--pipeline_control":
        rest = rest[:-2]
    params = {}
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag == "--j":  # RELION puts the thread count last, just before the additional arguments
            i += 2
            break
        if flag not in ("--o",) + EXTERNAL_INPUTS + OWN_PARAMS or i + 1 >= len(rest):
            raise Usage("unexpected argument before --j: %s" % flag)
        params[flag] = rest[i + 1]
        i += 2
    else:
        raise Usage("no --j in the arguments (not started by RELION?)")
    job_dir = params.pop("--o", "")
    if not job_dir:
        raise Usage("no --o <job directory>")
    if not job_dir.endswith("/"):
        job_dir += "/"
    return program, job_dir, params, rest[i:]


# ---------------------------------------------------------------------------- output nodes

def _star_blocks(path, limit=200):
    """Names of the data blocks and labels in the first lines of a STAR file."""
    blocks, labels = [], set()
    try:
        with open(path, "r", errors="replace") as fh:
            for n, line in enumerate(fh):
                if n > limit and blocks:
                    break
                s = line.strip()
                if s.startswith("data_"):
                    blocks.append(s[5:])
                elif s.startswith("_rln"):
                    labels.add(s.split()[0][1:])
    except OSError:
        pass
    return blocks, labels


def _mrc_nz(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
        if len(head) == 12:
            return struct.unpack("<3i", head)[2]
    except OSError:
        pass
    return None


def node_label(path):
    """A RELION node type for a file written by a tool (picks which inputs can use it)."""
    name = os.path.basename(path).lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext in ("mrc", "map"):
        if "mask" in name:
            return "Mask3D.mrc"
        nz = _mrc_nz(path)
        if nz is not None and nz <= 1:
            return "Image2D.mrc"
        return "DensityMap.mrc.halfmap" if "half" in name else "DensityMap.mrc"
    if ext == "mrcs":
        return "Image2DStack.mrcs"
    if ext == "star":
        blocks, labels = _star_blocks(path)
        if "rlnTomoParticlesFile" in labels or "rlnTomoTomogramsFile" in labels:
            return "TomoOptimisationSet.star.relion"
        if "rlnTomoTiltSeriesStarFile" in labels or "rlnTomoReconstructedTomogram" in labels:
            return "TomogramGroupMetadata.star.relion"
        if "particles" in blocks:
            return "ParticleGroupMetadata.star.relion"
        if "micrographs" in blocks:
            return "MicrographGroupMetadata.star.relion"
        if "movies" in blocks:
            return "MicrographMovieGroupMetadata.star.relion"
        if "coordinate_files" in blocks:
            return "MicrographCoordsGroup.star.relion"
        if "optimiser_general" in blocks:
            return "OptimiserData.star.relion"
        return "ProcessData.star"
    if ext == "pdf":
        return "LogFile.pdf"
    if ext in ("pdb", "cif"):
        return "AtomCoords." + ext
    return "ProcessData." + (ext or "txt")


def _job_files(job_dir):
    """Files the job wrote in its directory (two levels), with their modification times."""
    out = {}
    for root, dirs, files in os.walk(job_dir):
        rel = os.path.relpath(root, job_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirs[:] = [d for d in dirs if not d.startswith(".") and depth < 1]
        for f in files:
            if f.startswith(".") or f in BOOKKEEPING or f.startswith("RELION_") or f.endswith("_pipeline.star"):
                continue
            p = os.path.join(root, f)
            try:
                out[os.path.normpath(p)] = os.stat(p).st_mtime
            except OSError:
                pass
    return out


def write_output_nodes(job_dir, before):
    """RELION_OUTPUT_NODES.star with the files that are new or changed since ``before``."""
    now = _job_files(job_dir)
    new = sorted(p for p, t in now.items() if before.get(p) != t)
    top = [p for p in new if os.path.dirname(p) == os.path.normpath(job_dir)]
    if len(new) > MAX_NODES:  # e.g. one file per micrograph: keep the summary files
        new = top[:MAX_NODES]
    lines = ["", "# version 50001", "", "data_output_nodes", "", "loop_",
             "_rlnPipeLineNodeName #1", "_rlnPipeLineNodeTypeLabel #2"]
    for p in new:
        lines.append("%s %s" % (_quote(p), node_label(p)))
    with open(os.path.join(job_dir, OUTPUT_NODES), "w") as fh:
        fh.write("\n".join(lines) + "\n\n")
    return new


def _quote(s):
    return '"%s"' % s if any(c.isspace() for c in s) else s


# ---------------------------------------------------------------------------- running

def _touch(path):
    with open(path, "a"):
        os.utime(path, None)


def _which(program):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, program)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def local_mpi_limit():
    """RELION_ERROR_LOCAL_MPI (default 32), read as RELION's textToInteger does."""
    value = os.environ.get("RELION_ERROR_LOCAL_MPI")
    if value is None:
        return DEFAULT_LOCAL_MPI_LIMIT
    m = re.match(r"\s*([+-]?\d+)", value)
    if not m:
        raise Usage("RELION_ERROR_LOCAL_MPI=%r is not an integer" % value)
    return int(m.group(1))


def command(program, job_dir, params, args):
    try:
        mpi = int(params.get("--mpi", "1") or 1)
    except ValueError:
        raise Usage("--mpi %s: not a whole number of MPI processes" % params["--mpi"])
    exe = program
    if mpi > 1:
        limit = local_mpi_limit()
        if mpi > limit:  # RELION's prepareFinalCommand: these runs are always local
            raise Usage("You're submitting a local job with %d parallel MPI processes. That's more than allowed "
                        "by the RELION_ERROR_LOCAL_MPI environment variable (%d)." % (mpi, limit))
        if _which(program + "_mpi") is None:
            raise Usage("%s_mpi not found: load a RELION with MPI, or use 1 MPI process" % program)
        exe = program + "_mpi"
    cmd = [exe] + list(args)
    if program not in POSITIONAL_TOOLS:
        cmd += ["--pipeline_control", os.path.join(job_dir, CONTROL_DIR) + "/"]
    if mpi > 1:
        cmd = shlex.split(os.environ.get("RELION_MPIRUN", "mpirun")) + ["-n", str(mpi)] + cmd
    return cmd


def stop(pid, aborted_at, sent, now):
    """Signal the program's process group once an abort's grace periods are over: SIGTERM
    ABORT_GRACE_S after the abort, SIGKILL KILL_GRACE_S after that (a program that ignores SIGTERM
    would otherwise keep the job running for ever). Returns the signals sent so far."""
    if not sent and now - aborted_at > ABORT_GRACE_S:
        sig = signal.SIGTERM
    elif len(sent) == 1 and now - aborted_at > ABORT_GRACE_S + KILL_GRACE_S:
        sig = signal.SIGKILL
    else:
        return sent
    try:
        os.killpg(pid, sig)
    except OSError:
        pass
    return sent + [sig]


STOP_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP")


def on_stop_signals(handler):
    """Call handler(signum) when the runner is told to stop; returns the previous handlers."""
    old = {}
    for name in STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            old[sig] = signal.signal(sig, lambda signum, frame: handler(signum))
    return old


def run(argv):
    program, job_dir, params, args = parse(argv)
    control = os.path.join(job_dir, CONTROL_DIR)
    if not os.path.isdir(control):
        os.makedirs(control)
    for f in (SUCCESS, FAILURE, ABORTED, ABORT_NOW):  # RELION cleans the job directory itself
        if os.path.exists(os.path.join(control, f)):
            os.remove(os.path.join(control, f))
    cmd = command(program, job_dir, params, args)
    before = _job_files(job_dir)
    print(" ++++ cryo-savanna CLI tool: " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    out = None
    if params.get("--stdout"):
        try:
            out = open(os.path.join(job_dir, params["--stdout"]), "w")
        except OSError as e:
            raise Usage("cannot write the output to %s: %s" % (params["--stdout"], e))
    try:
        proc = subprocess.Popen(cmd, stdout=out if out is not None else None, start_new_session=True)
    except OSError as e:
        if out is not None:
            out.close()
        print("ERROR: cannot start %s: %s" % (cmd[0], e), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, FAILURE))
        return 1

    aborted_at = None
    sent = []
    stopped = []  # (signal, time) once the runner was told to stop
    killed = False

    def stop_now(signum):
        if stopped:
            return
        stopped.append((signum, time.time()))
        try:  # the signal may have come in the middle of another print
            print(" ++++ cryo-savanna CLI tool: signal %d, stopping %s" % (signum, program), flush=True)
        except (RuntimeError, OSError):
            pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:  # now: the next signal may be a SIGKILL that leaves no time for it
            _touch(os.path.join(job_dir, ABORTED))
        except OSError:
            pass

    on_stop_signals(stop_now)
    while True:
        try:
            rc = proc.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            pass
        if stopped:
            if not killed and time.time() - stopped[0][1] > KILL_GRACE_S:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                killed = True
        elif aborted_at is None and os.path.exists(os.path.join(job_dir, ABORT_NOW)):
            aborted_at = time.time()
            print(" ++++ abort requested", flush=True)
            _touch(os.path.join(control, ABORT_NOW))  # RELION programs stop at their next check
        elif aborted_at is not None:
            sent = stop(proc.pid, aborted_at, sent, time.time())
    if out is not None:
        out.close()

    if stopped:
        _touch(os.path.join(job_dir, ABORTED))
        return 128 + stopped[0][0]
    if aborted_at is not None:
        _touch(os.path.join(job_dir, ABORTED))
        return rc or 1
    if rc != 0:
        print("ERROR: %s exited with %s" % (program, rc), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, FAILURE))
        return rc
    nodes = write_output_nodes(job_dir, before)
    print(" ++++ %d output file(s) added to the pipeline" % len(nodes), flush=True)
    _touch(os.path.join(job_dir, SUCCESS))
    return 0


def _fail(argv):
    """Tell RELION the job ended: without an exit file it shows the job as running for ever."""
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
        print("cryo-savanna cli_runner: " + str(e), file=sys.stderr, flush=True)
        _fail(argv)
        sys.exit(2)
    except Exception:  # anything else (a full disk, a file that cannot be read): still an exit file
        traceback.print_exc()
        sys.stderr.flush()
        _fail(argv)
        sys.exit(1)


if __name__ == "__main__":
    main()
