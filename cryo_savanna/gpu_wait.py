#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's local queue: wait until the GPUs and threads a run needs on this machine are free, then run a
command that keeps them.

    python3 gpu_wait.py --locks <dir> [--gpus 0,1] [--threads N --max-threads M] [--label <text>] -- <command> ...

GPUs. Each GPU has a lock file in <dir> (``gpu-<id>.lock``, the id as CUDA_VISIBLE_DEVICES names it, else its
number). While it waits, a GPU also counts as busy when a RELION job computes on it without a lock (one started
from RELION's own GUI or a script): nvidia-smi lists a compute process there that is part of a RELION job, i.e.
it, or a process it was started by, is a RELION program (relion_refine, relion_run_motioncorr, relion_pipeliner,
the GUI ...) or gives a job directory as where to write (--o Class3D/job012/, ``>> External/job020/run.out``: how
RELION starts its own jobs and External ones, such as AreTomo3 or pytom from cryo-savanna's runners). Any other
program on the GPU (a notebook, a viewer, someone's own script) is not waited for.

Threads. A run that takes N threads (MPI processes x threads of its biggest job) starts once N more fit in M (the
limit of the machine): those taken by other runs of the queue (a ``threads-<pid>.lock`` each, holding how many)
and those of RELION jobs that run without it (RELION's own GUI, other projects, older cryo-savanna runs), counted
from their command lines (``--j N`` of each process of a RELION job, one per MPI rank of a relion_*_mpi without
it; see relion_threads). A run that asks for more than M waits until no RELION job runs here, then oversubscribes.
Looking and taking happen under ``threads.mutex``, so two runs never both take the last free threads.

The command starts once this process holds everything (GPUs and threads together, or nothing: no two runs each
holding half of what the other needs). It is then exec'd in this process, which keeps the locks: the command and
what it starts (RELION's pipeliner and its jobs) hold them until the last of them ends, also when cryo-savanna
has been closed. A run that ends, or is killed, frees its GPUs and threads: nothing is left to clean up.

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


THREAD_MUTEX = "threads.mutex"


def _say(text):
    print(" ++++ cryo-savanna local queue: %s" % text, flush=True)


def parse(argv):
    """(lock dir, GPU ids, label, command, threads, thread limit)."""
    usage = "usage: gpu_wait.py --locks DIR [--gpus 0,1] [--threads N --max-threads M] [--label TEXT] -- COMMAND ..."
    if "--" not in argv:
        raise SystemExit(usage)
    i = argv.index("--")
    head, cmd = argv[:i], argv[i + 1:]
    opts = {}
    for k in range(0, len(head) - 1, 2):
        opts[head[k]] = head[k + 1]
    gpus = [g for g in opts.get("--gpus", "").split(",") if g]
    try:
        threads, limit = int(opts.get("--threads", "0")), int(opts.get("--max-threads", "0"))
    except ValueError:
        raise SystemExit("gpu_wait.py: --threads and --max-threads take whole numbers")
    if threads <= 0 or limit <= 0:
        threads = limit = 0
    if "--locks" not in opts or not (gpus or threads) or not cmd:
        raise SystemExit("gpu_wait.py: --locks, --gpus or --threads, and a command are needed")
    return opts["--locks"], gpus, opts.get("--label", ""), cmd, threads, limit


def lock_path(lock_dir, gpu):
    """The lock file of a GPU (also read by cryo-savanna's force abort, project/processes.py)."""
    return os.path.join(lock_dir, "gpu-%s.lock" % "".join(c if c.isalnum() or c in "-_" else "_" for c in gpu))


def try_lock(lock_dir, gpus):
    """File descriptors holding the locks of all ``gpus``, or None (and nothing held) when one is taken."""
    held = []  # type: list[int]
    for gpu in sorted(gpus):
        path = lock_path(lock_dir, gpu)
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        except PermissionError:  # made by another user before files were made writable by all: lockable all the same
            fd = os.open(path, os.O_RDONLY)
        try:
            if os.fstat(fd).st_uid == os.getuid() and os.fstat(fd).st_mode & 0o777 != 0o666:
                os.fchmod(fd, 0o666)  # the umask took write access away: every user of the machine opens it
        except OSError:
            pass
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
        with open(lock_path(lock_dir, gpu)) as fh:
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


class ProcTable(dict):
    """{pid: (parent pid, command line)}, and in ``groups`` {pid: (process group, session)} when ps gave them."""

    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.groups = {}  # type: dict[int, tuple[int, int]]


def _ps(fields):
    try:
        r = subprocess.run(["ps", "-A", "-ww", "-o", fields], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def process_table():
    """{pid: (parent pid, command line)} of every process of this machine (ps), with their process groups and
    sessions (ProcTable.groups); {} when it cannot be read."""
    out = ProcTable()
    text = _ps("pid=,ppid=,pgid=,sess=,args=")  # sess: the session id on Linux (0 on macOS: no use, no harm)
    if text is not None:
        for line in text.splitlines():
            w = line.split(None, 4)
            if len(w) >= 4 and all(x.isdigit() for x in w[:4]):
                out[int(w[0])] = (int(w[1]), w[4] if len(w) > 4 else "")
                out.groups[int(w[0])] = (int(w[2]), int(w[3]))
        if out:
            return out
    text = _ps("pid=,ppid=,args=") or ""  # a ps without those fields
    for line in text.splitlines():
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


def gpu_index():
    """{UUID: number} of the GPUs nvidia-smi lists on this machine; {} when there are none (or no nvidia-smi)."""
    ids = {}
    for line in _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            ids[parts[1]] = parts[0]
    return ids


def busy_gpus(table=process_table, ids=None):
    """The GPUs (by number and by UUID) on which nvidia-smi lists a compute process of a RELION job; empty when it
    cannot tell. Other programs computing there do not count. ``ids``: gpu_index(), when already asked."""
    ids = gpu_index() if ids is None else ids
    if not ids:
        return set()
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


# ---------------------------------------------------------------------------- threads
THREADS_ARG = re.compile(r"(?:^|[\s'\"])--j\s+['\"]?(\d+)")
MPI_RANK = re.compile(r"^relion_\w+_mpi$")


def own_threads(args):
    """The threads a process says it computes with: its ``--j N``, else 1 for an MPI rank of a RELION program
    without one (relion_preprocess_mpi ...), else 0."""
    m = THREADS_ARG.search(args)
    if m:
        return int(m.group(1))
    first = args.split(None, 1)[0] if args.strip() else ""
    return 1 if MPI_RANK.match(os.path.basename(first)) else 0


def relion_threads(table, skip=()):
    """The threads RELION jobs of this machine compute with, from their command lines (``table``: process_table()),
    leaving out the processes of the runs ``skip`` (of the queue, which say how many they took): everything under
    them, and everything in their process group or session (each run is the leader of both: Pipeliner.run). RELION
    starts a local job in the background, so the job's processes are not under its run once the shell that started
    them has ended (they are the init's), but stay in its session.

    A job's threads show in several processes: the shell RELION starts it with (sh -c "mpirun -n 4 relion_refine_mpi
    --j 4 ..."), mpirun, its 4 ranks (--j 4 each). Each process counts as the larger of its own threads and those of
    the processes it started, together: 16 here, as RELION's MPI x threads; a runner with --j 8 that starts
    programs without --j counts 8. Only processes that are part of a RELION job (is_relion_job) count."""
    children = {}  # type: dict[int, list[int]]
    for pid, (ppid, _) in table.items():
        if ppid != pid:
            children.setdefault(ppid, []).append(pid)
    own = {}  # type: dict[int, int]
    for pid, (_, args) in table.items():
        n = own_threads(args)
        if n and is_relion_job(pid, table):
            own[pid] = n
    skip = set(skip)
    groups = getattr(table, "groups", {})
    if skip and groups:
        skip |= {pid for pid, (pgid, sid) in groups.items() if pgid in skip or sid in skip}
    total = 0
    value = {}  # type: dict[int, int]
    roots = [pid for pid, (ppid, _) in table.items() if ppid not in table or ppid == pid]
    for root in roots:
        stack = [(root, False)]  # type: list[tuple[int, bool]]
        while stack:  # after its children: a process's value is known once theirs are
            pid, done = stack.pop()
            if pid in skip:
                value[pid] = 0
            elif done:
                value[pid] = max(own.get(pid, 0), sum(value.get(c, 0) for c in children.get(pid, ())))
            elif pid not in value:
                value[pid] = 0  # also guards against a cycle in a table read while processes came and went
                stack.append((pid, True))
                stack.extend((c, False) for c in children.get(pid, ()) if c not in value)
        total += value.get(root, 0)
    return total


def thread_lock_path(lock_dir, pid, k=0):
    """The file by which the run of process ``pid`` holds its threads (also read by cryo-savanna's force abort)."""
    return os.path.join(lock_dir, "threads-%d%s.lock" % (pid, "-%d" % k if k else ""))


def held_threads(lock_dir, clean=False):
    """[(pid, threads, label)] of the runs holding threads now (their lock file is locked). ``clean``: remove the
    files of runs that ended (only under the mutex, which those taking threads hold)."""
    out = []  # type: list[tuple[int, int, str]]
    try:
        names = sorted(os.listdir(lock_dir))
    except OSError:
        return out
    for name in names:
        m = re.match(r"^threads-(\d+)(?:-\d+)?\.lock$", name)
        if not m:
            continue
        path = os.path.join(lock_dir, name)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    continue
                text = os.read(fd, 4096).decode("utf-8", "replace").strip()
                first = text.split(None, 1)
                if first and first[0].isdigit():
                    out.append((int(m.group(1)), int(first[0]), first[1] if len(first) > 1 else ""))
                continue
            if clean:  # nobody holds it: its run ended
                try:
                    os.unlink(path)
                except OSError:
                    pass
        finally:
            os.close(fd)
    return out


def threads_in_use(lock_dir, table, clean=False):
    """(threads taken by runs of the queue, threads of RELION jobs running without it, [(pid, threads, label)])."""
    held = held_threads(lock_dir, clean)
    holders = {pid for pid, _, _ in held}
    # not this process's own: a waiter has started nothing yet, and under cryo-savanna's window are the runs it
    # started (those that took no threads count)
    return sum(n for _, n, _ in held), relion_threads(table, holders), held


def _mutex(lock_dir):
    path = os.path.join(lock_dir, THREAD_MUTEX)
    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        if os.fstat(fd).st_uid == os.getuid():
            os.fchmod(fd, 0o666)  # every user of the machine takes it (flock needs no write access)
    except OSError:
        pass
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def try_threads(lock_dir, threads, limit, label, table):
    """(descriptor holding ``threads`` of the ``limit``, None), or (None, why not). ``table``: process_table, or
    what gives it."""
    mutex = _mutex(lock_dir)
    try:
        tracked, untracked, held = threads_in_use(lock_dir, table(), clean=True)
        used = tracked + untracked
        if used + threads > limit and not (threads > limit and used == 0):
            who = "; ".join("%s: %d" % (lab or "process %d" % pid, n) for pid, n, lab in held)
            parts = (["%d by runs of the queue (%s)" % (tracked, who)] if tracked else []) + \
                    (["%d by other RELION jobs" % untracked] if untracked else [])
            why = "%d of %d threads in use (%s), %d needed" % (used, limit, ", ".join(parts), threads)
            if threads > limit:
                why += ": more than the limit, so it waits until no RELION job runs on this machine"
            return None, why
        k = 0
        while True:
            path = thread_lock_path(lock_dir, os.getpid(), k)
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
                break
            except FileExistsError:  # left by an earlier process with this number, of another user
                k += 1
        fcntl.flock(fd, fcntl.LOCK_EX)  # a new file: at most a look at it (held_threads) for a moment
        os.write(fd, ("%d %s (process %d)\n" % (threads, label or "a run", os.getpid())).encode())
        return fd, None
    finally:
        os.close(mutex)


# ---------------------------------------------------------------------------- waiting
class _Once:
    """fn(), called the first time only."""

    def __init__(self, fn):
        self.fn, self.got = fn, None

    def __call__(self):
        if self.got is None:
            self.got = self.fn()
        return self.got


def wait(lock_dir, gpus, label, poll_s=POLL_S, busy=None, threads=0, max_threads=0, table=process_table):
    """Wait until all ``gpus`` and ``threads`` (of ``max_threads``) are free; returns the descriptors holding their
    locks (inheritable)."""
    said = None
    while True:
        table_once = _Once(table)  # one ps per look, for both GPUs and threads
        held = try_lock(lock_dir, gpus) if gpus else []
        why = None
        if held is not None:
            busy_now = (busy() if busy is not None else busy_gpus(table_once)) if gpus else set()
            in_use = [g for g in gpus if g in busy_now]  # once per look (two nvidia-smi and a ps), not once per GPU
            if in_use:
                why = "GPU %s in use by a RELION job started without the GPU queue (nvidia-smi)" % ", ".join(in_use)
            elif threads:
                fd, why = try_threads(lock_dir, threads, max_threads, label, table_once)
                if fd is not None:
                    held.append(fd)
            if why is None:
                break
            for fd in held:
                os.close(fd)
        else:
            taken = [g for g in gpus if holder(lock_dir, g)]
            why = "GPU %s in use by %s" % (", ".join(gpus), "; ".join(sorted({holder(lock_dir, g) for g in taken}))
                                            or "another run")
        if why != said:
            _say("waiting: " + why)
            said = why
        time.sleep(poll_s)
    for fd in held[:len(gpus)]:
        try:
            os.ftruncate(fd, 0)
            os.write(fd, ("%s (process %d)\n" % (label or "a run", os.getpid())).encode())
        except OSError:  # opened read-only (another user's file): locked, without a label
            pass
    for fd in held:
        os.set_inheritable(fd, True)  # kept by the command, and by what it starts
    got = (["GPU %s" % ", ".join(gpus)] if gpus else []) + \
        (["%d thread%s" % (threads, "s" if threads != 1 else "")] if threads else [])
    _say("%s free: starting" % " and ".join(got))
    if threads > max_threads > 0:
        _say("WARNING: %d threads asked for, more than the %d of the thread queue: this machine is oversubscribed"
             % (threads, max_threads))
    return held


def main(argv):
    lock_dir, gpus, label, cmd, threads, limit = parse(argv)
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError as e:
        print("gpu_wait.py: cannot make %s: %s" % (lock_dir, e), file=sys.stderr, flush=True)
        return 1
    wait(lock_dir, gpus, label, threads=threads, max_threads=limit)
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        print("gpu_wait.py: cannot start %s: %s" % (cmd[0], e), file=sys.stderr, flush=True)
        return 127
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
