#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its pytom, AreTomo3 or tomogram mask
# jobs, beside their runners, and travels with it.
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""What the runners of cryo-savanna's "Additional external jobs" share.

pytom_runner.py, aretomo_runner.py and mask_runner.py are started by RELION as External jobs
(``python3 .cryo-savanna/<runner>.py <mode> --o <job dir> [--in_* F]... --j <threads> --<option> <value> ...``)
and import this file from beside them: the arguments of an External job, the exit files and
RELION_OUTPUT_NODES.star, STAR files, MRC headers, environment modules, the programs they run (in process
groups of their own, stopped on abort or scancel), and the tomograms of a RELION 5 tomograms.star.

Like the runners it needs nothing but the standard library (Python 3.6+: RELION may start a runner on a
compute node with the system python3).
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

WHO = "cryo-savanna"  # what say() calls the program the job runs: each runner sets its own (C.WHO = ...)

EXTERNAL_INPUTS = ("--in_movies", "--in_mics", "--in_parts", "--in_coords", "--in_3dref", "--in_mask")
SUCCESS, FAILURE, ABORTED = "RELION_JOB_EXIT_SUCCESS", "RELION_JOB_EXIT_FAILURE", "RELION_JOB_EXIT_ABORTED"
ABORT_NOW = "RELION_JOB_ABORT_NOW"
OUTPUT_NODES = "RELION_OUTPUT_NODES.star"
KILL_GRACE_S = 30


def _poll_seconds(default=1.0):
    """How often the running programs are looked at. CRYO_SAVANNA_POLL_S changes it (cryo-savanna's own
    tests run stand-in programs that end at once: a second per look is most of their time)."""
    try:
        v = float(os.environ.get("CRYO_SAVANNA_POLL_S", ""))
    except ValueError:
        return default
    return v if 0 < v <= 60 else default


POLL_S = _poll_seconds()

TOMO_NAME = "rlnTomoName"
TOMO_MASK_SUFFIX = "_mask.mrc"  # a folder of tomogram masks: <rlnTomoName>_mask.mrc (as mask_runner.py names them)
TOMOGRAM_KINDS = {
    "Reconstructed": "rlnTomoReconstructedTomogram",
    "Denoised": "rlnTomoReconstructedTomogramDenoised",
    "Half 1": "rlnTomoReconstructedTomogramHalf1",
    "Half 2": "rlnTomoReconstructedTomogramHalf2",
}


class Usage(Exception):
    pass


class Aborted(Exception):
    pass


def say(text):
    print(" ++++ %s: %s" % (WHO, text), flush=True)


# ---------------------------------------------------------------------------- arguments


def parse(argv, modes):
    """argv[1:] -> (mode, job dir, number of threads, {option: value}). ``modes``: the first arguments
    the runner takes."""
    if not argv or argv[0] not in modes:
        raise Usage("the first argument must be %s" % " or ".join("'%s'" % m for m in modes))
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
    """A number as a program gets it on its command line (no trailing .0 for whole numbers)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


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
    environment, so every program the job runs (and conda run) sees that CUDA."""
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

    def __init__(self, job_dir, hint=""):
        self.job_dir = job_dir
        self.hint = hint  # after "cannot start <program>": what to check (" (is pytom's conda environment right?)")
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
            raise Usage("cannot start %s: %s%s" % (cmd[0], e, self.hint))
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
            say("abort requested: stopping wrapper")
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


# ---------------------------------------------------------------------------- RELION 5 tomograms

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


# ---------------------------------------------------------------------------- GPUs, logs, the end of a job

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


def _first_error(lines, most=60):
    """The first traceback in a log, with the line before it ("Process Process-1:" when it was one of
    the program's worker processes, as in pytom): the real error, which a later "processes stopped unexpectedly"
    hides."""
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            start = i - 1 if i and lines[i - 1].startswith("Process ") else i
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("Traceback (most recent call")),
                       len(lines))
            return start, min(end, start + most)
    return None


def _show_tail(log, n=25):
    """The end of a failed program's log in run.err, after the first error in it when that came earlier."""
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


def _fail(argv):
    if "--o" in argv[:-1]:
        job_dir = argv[argv.index("--o") + 1]
        if os.path.isdir(job_dir):
            try:
                _touch(os.path.join(job_dir, FAILURE))
            except OSError:
                pass


def main(run):
    """A runner's entry point: sys.exit(run(argv)), with a failure exit file whatever went wrong (a job with
    none stays Running in RELION for ever)."""
    argv = sys.argv[1:]
    runner = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    try:
        sys.exit(run(argv))
    except Usage as e:
        print("cryo-savanna %s: %s" % (runner, e), file=sys.stderr, flush=True)
        _fail(argv)
        sys.exit(2)
    except Exception:  # anything else: still an exit file, or RELION shows the job running for ever
        traceback.print_exc()
        sys.stderr.flush()
        _fail(argv)
        sys.exit(1)
