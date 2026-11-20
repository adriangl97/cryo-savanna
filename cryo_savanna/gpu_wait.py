#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's local GPU queue: wait until GPUs of this machine are free, then run a command that keeps them.

    python3 gpu_wait.py --locks <dir> --gpus 0,1 [--label <text>] -- <command> [<argument> ...]

Each GPU has a lock file in <dir> (``gpu-<id>.lock``, the id as CUDA_VISIBLE_DEVICES names it, else its
number). The command starts once this process holds exclusive locks (flock) on the files of all its GPUs, taken
all at once or not at all (no two runs each holding half of what the other needs). It is then exec'd in this
process, which keeps the locks: the command and what it starts (RELION's pipeliner and its jobs) hold them until
the last of them ends, also when cryo-savanna has been closed. A run that ends, or is killed, frees its GPUs:
nothing is left to clean up.

While it waits, a GPU also counts as busy when a RELION job computes on it without a lock (one started from
RELION's own GUI or a script): nvidia-smi lists a compute process there that is part of a RELION job, i.e. it,
or a process it was started by, is a RELION program (relion_refine, relion_run_motioncorr, relion_pipeliner, the
GUI ...) or gives a job directory as where to write (--o Class3D/job012/, ``>> External/job020/run.out``: how
RELION starts its own jobs and External ones, such as AreTomo3 or pytom from cryo-savanna's runners). Any other
program on the GPU (a notebook, a viewer, someone's own script) is not waited for.

project/gpu_queue.py starts it; standard library only (it runs with the python cryo-savanna runs with, but must
not depend on cryo-savanna's imports: it is started with RELION's environment).
"""

import errno
import fcntl
import os
import re
import subprocess
import sys
import time

POLL_S = 5.0


def _say(text):
    print(" ++++ cryo-savanna GPU queue: %s" % text, flush=True)


def parse(argv):
    """(lock dir, GPU ids, label, command)."""
    if "--" not in argv:
        raise SystemExit("usage: gpu_wait.py --locks DIR --gpus 0,1 [--label TEXT] -- COMMAND ...")
    i = argv.index("--")
    head, cmd = argv[:i], argv[i + 1:]
    opts = {}
    for k in range(0, len(head) - 1, 2):
        opts[head[k]] = head[k + 1]
    gpus = [g for g in opts.get("--gpus", "").split(",") if g]
    if "--locks" not in opts or not gpus or not cmd:
        raise SystemExit("gpu_wait.py: --locks, --gpus and a command are needed")
    return opts["--locks"], gpus, opts.get("--label", ""), cmd


def _lock_path(lock_dir, gpu):
    return os.path.join(lock_dir, "gpu-%s.lock" % "".join(c if c.isalnum() or c in "-_" else "_" for c in gpu))


def try_lock(lock_dir, gpus):
    """File descriptors holding the locks of all ``gpus``, or None (and nothing held) when one is taken."""
    held = []  # type: list[int]
    for gpu in sorted(gpus):
        fd = os.open(_lock_path(lock_dir, gpu), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            for h in held:
                os.close(h)
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return None
            raise
        held.append(fd)
    return held


def holder(lock_dir, gpu):
    """What the lock file of a GPU says holds it (the label written by the run holding it)."""
    try:
        with open(_lock_path(lock_dir, gpu)) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _nvidia_smi(args):
    try:
        r = subprocess.run(["nvidia-smi"] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


# a RELION program (by its name, also with a path or in `which ...`), or a job directory given as where to write
RELION_JOB = re.compile(r"(?:^|[\s/`'\"])relion(?:_\w+)?(?:[\s`'\"]|$)"
                        r"|(?:--o|--odir|\d?>>?)\s*['\"]?(?:\S*/)?[A-Za-z0-9_]+/job\d{3,}/")


def process_table():
    """{pid: (parent pid, command line)} of every process of this machine (ps); {} when it cannot be read."""
    try:
        r = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,args="], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in r.stdout.splitlines():
        w = line.split(None, 2)
        if len(w) >= 2 and w[0].isdigit() and w[1].isdigit():
            out[int(w[0])] = (int(w[1]), w[2] if len(w) > 2 else "")
    return out


def is_relion_job(pid, table):
    """Whether a process is part of a RELION job: it or one of the processes it was started by (as far up as
    the table goes) is a RELION program or writes to a job directory."""
    seen = set()
    while pid in table and pid not in seen and pid > 1:
        seen.add(pid)
        parent, args = table[pid]
        if RELION_JOB.search(args):
            return True
        pid = parent
    return False


def busy_gpus(table=process_table):
    """The GPUs (by number and by UUID) on which nvidia-smi lists a compute process of a RELION job; empty when it
    cannot tell. Other programs computing there do not count."""
    ids = {}
    for line in _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            ids[parts[1]] = parts[0]
    apps = []
    for line in _nvidia_smi(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] in ids and parts[1].isdigit():
            apps.append((parts[0], int(parts[1])))
    out = set()  # type: set[str]
    if apps:
        procs = table()
        for uuid, pid in apps:
            if is_relion_job(pid, procs):
                out.update((uuid, ids[uuid]))
    return out


def wait(lock_dir, gpus, label, poll_s=POLL_S, busy=busy_gpus):
    """Wait until all ``gpus`` are free; returns the descriptors holding their locks (inheritable)."""
    said = None
    while True:
        held = try_lock(lock_dir, gpus)
        if held is not None:
            busy_now = busy()  # once per look (two nvidia-smi and a ps), not once per GPU
            in_use = [g for g in gpus if g in busy_now]
            if not in_use:
                break
            for fd in held:
                os.close(fd)
            why = "GPU %s in use by a RELION job started without the GPU queue (nvidia-smi)" % ", ".join(in_use)
        else:
            taken = [g for g in gpus if holder(lock_dir, g)]
            why = "GPU %s in use by %s" % (", ".join(gpus), "; ".join(sorted({holder(lock_dir, g) for g in taken}))
                                            or "another run")
        if why != said:
            _say("waiting: " + why)
            said = why
        time.sleep(poll_s)
    for fd in held:
        os.ftruncate(fd, 0)
        os.write(fd, ("%s (process %d)\n" % (label or "a run", os.getpid())).encode())
        os.set_inheritable(fd, True)  # kept by the command, and by what it starts
    _say("GPU %s free: starting" % ", ".join(gpus))
    return held


def main(argv):
    lock_dir, gpus, label, cmd = parse(argv)
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError as e:
        print("gpu_wait.py: cannot make %s: %s" % (lock_dir, e), file=sys.stderr, flush=True)
        return 1
    wait(lock_dir, gpus, label)
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        print("gpu_wait.py: cannot start %s: %s" % (cmd[0], e), file=sys.stderr, flush=True)
        return 127
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
