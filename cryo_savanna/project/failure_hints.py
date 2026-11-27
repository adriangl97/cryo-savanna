# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Why a job may have failed: known messages in the end of its run.err and run.out, each with a hint.

The patterns are RELION 5.1.1's own error texts (``REPORT_ERROR`` and the ``ERR*`` messages of
``src/error.h``), those of CUDA, MPI and SLURM, and those of cryo-savanna's runners and SLURM monitor
(``cryo-savanna: SLURM job … ended TIMEOUT``, ui/actions/running.py). Read for failed jobs only, by the
job summaries (project/summary.py), off the GUI thread.

When none of them matches, RELION's own error message is shown instead (the line after its ``ERROR:``),
and failing that, a hint that MPI stopped the job (MPI_ABORT follows every RELION error under MPI, so it
says nothing on its own when something else matched).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

TAIL_BYTES = 64 * 1024  # of each log: errors are at the end
MAX_HINTS = 3
LOGS = ("run.err", "run.out")


@dataclass(frozen=True)
class Hint:
    id: str
    pattern: re.Pattern
    text: str
    fallback: bool = False  # shown only when no other hint matched
    covered_by: tuple[str, ...] = ()  # not shown when one of these matched (a more specific one)


def _h(id: str, pattern: str, text: str, **kw) -> Hint:
    return Hint(id, re.compile(pattern, re.IGNORECASE), text, **kw)


# most specific first: the order hints are shown in
HINTS: tuple[Hint, ...] = (
    _h("gpu_oom", r"You ran out of memory on the GPU|cudaErrorMemoryAllocation|CUDA error: out of memory"
                  r"|CUDA out of memory|hipErrorOutOfMemory",
       "The GPU ran out of memory: run fewer MPI processes per GPU (see 'is split between' at the start of "
       "run.out), use a smaller box (re-extract binned), or set 'Skip padding?' to Yes on the Compute tab "
       "(membrain-seg: a smaller sliding window, or fewer tomograms per GPU)."),
    _h("gpu_ids", r"There was an issue with the GPU-ids",
       "The GPU ids in 'Which GPUs to use' do not match this node: leave the field empty, or remember that "
       "GPUs are numbered from 0."),
    _h("gpu_mpi", r"not enough MPI nodes specified for the GPU IDs",
       "'Which GPUs to use' lists more GPU groups (separated by ':') than there are MPI followers: give one "
       "group per follower, or leave the field empty."),
    _h("gpu_kernel", r"A GPU-function failed to execute",
       "A GPU function failed: the GPU may be too old for this RELION build, or the node has GPUs of "
       "different kinds; try another node, or choose the GPU in 'Which GPUs to use'."),
    _h("time_limit", r"DUE TO TIME LIMIT|SLURM job \S+ ended TIMEOUT",
       "The queue's time limit ended the job: ask for more time in the submission settings, or continue the "
       "job (iterative jobs carry on from their last iteration)."),
    _h("ram_oom", r"There was an issue allocating CPU memory|std::bad_alloc|Allocate: No space left"
                  r"|oom[-_ ]kill|SLURM job \S+ ended OUT_OF_MEMORY",
       "The job ran out of memory (RAM): ask the queue for more memory, run fewer MPI processes per node, or "
       "set 'Pre-read all particles into RAM?' to No."),
    _h("node_fail", r"DUE TO NODE FAILURE|SLURM job \S+ ended NODE_FAIL",
       "A compute node failed: run the job again (continue it if it is iterative)."),
    _h("disk_full", r"No space left on device|Disk quota exceeded",
       "The file system is full or over quota: free space (Project → Disk usage…, then empty Trash/) and run "
       "the job again."),
    _h("permission", r"Permission denied",
       "A file or folder could not be read or written (permission denied): in a project shared by a group, "
       "files and folders must be group-writable."),
    _h("mpi_minimum", r"at least 3 MPI processes are required|needs to be run with at least two MPI processes"
                      r"|Cannot split data into random halves without using MPI",
       "This step needs more MPI processes (3D auto-refine: at least 3, a leader and one per half-set): raise "
       "'Number of MPI procs'."),
    _h("mpi_slots", r"There are not enough slots available|not enough slots",
       "MPI could not start as many processes as asked: lower 'Number of MPI procs', or ask the queue for more "
       "tasks."),
    _h("local_mpi", r"more than allowed by the RELION_ERROR_LOCAL_MPI",
       "Too many MPI processes for a job on this machine (RELION_ERROR_LOCAL_MPI): use fewer, or submit it to "
       "the queue."),
    _h("mask_size", r"Solvent mask does not have the same size|should have the same sizes|maps are different shapes"
                    r"|should all be 3D cubic|Diameter of spherical mask divided by the box size",
       "A mask or reference does not fit the particles: it needs the same box and pixel size (resample it with "
       "relion_image_handler), and the mask diameter must be smaller than the box."),
    _h("pixel_size", r"pixel sizes are different|different pixel size|Invalid pixel size|Pixel size is invalid",
       "The pixel sizes do not agree (or one is missing): check the pixel size of the inputs and their optics "
       "groups."),
    _h("zero_weights", r"resulted in all-zero|No orientation was found as better than any other"
                       r"|number of contributing orientations for an image|sum of weights for all orientations was"
                       r"|zero sum of weights",
       "A particle got no weight in any orientation, and RELION stopped on it (run.err names it: 'fn_img='): "
       "split it off with Additional external jobs → Split off zero-weight particles, then run the job "
       "again on its accepted_particles.star (tomography: its optimisation_set.star)."),
    _h("no_particles", r"there are no particles in|not enough particles to",
       "No particles were left at this step: check the selection or threshold of the job before (an empty "
       "class selection, a too-strict figure of merit)."),
    _h("python_module", r"ModuleNotFoundError|ImportError: No module named",
       "A Python module is missing: the program runs in the wrong Python environment (pytom, membrain-seg, "
       "pyem: its conda environment on the form; RELION's tomography tools: the Python of the RELION module)."),
    _h("cannot_start", r"cannot start \S+: ",
       "The program could not be started: check its path or environment on the form (pytom's, "
       "membrain-seg's or pyem's conda environment, the AreTomo3 executable)."),
    _h("missing_file", r"does not exist|Cannot open (input |output )?file|cannot read file|No such file or directory"
                       r"|Cannot find \S+ file",
       "An input file is missing or cannot be read: check the inputs on the I/O tab (was an earlier job "
       "deleted, moved, or harshly cleaned? A harsh clean of Extract removes its particle stacks).",
       covered_by=("python_module", "cannot_start", "permission", "disk_full")),
    _h("segfault", r"Segmentation fault|signal 11\b|SIGSEGV|core dumped",
       "The program crashed (segmentation fault): often a truncated or corrupt input file; if it happens again "
       "with the same inputs, report it with run.out and run.err.",
       covered_by=("gpu_oom", "ram_oom", "zero_weights")),  # (a debug build raises SIGSEGV for this error)
    _h("mpi_abort", r"MPI_ABORT was invoked|mpirun noticed that process rank|mpirun was unable to launch"
                    r"|ORTE was unable|PMIX ERROR",
       "MPI stopped the job because one process failed: the first error above in run.out or run.err is the "
       "cause.", fallback=True),
)
BY_ID = {h.id: h for h in HINTS}

RELION_ERROR = Hint("relion_error", re.compile(r"^ERROR:\s*$"), "RELION stopped with this error:")


@dataclass(frozen=True)
class Diagnosis:
    hint: Hint
    file: str  # "run.err"
    line_no: int  # 1-based, in the whole file
    line: str  # the line that matched (RELION's error: its message)

    @property
    def text(self) -> str:
        return f"{self.hint.text} {self.line}" if self.hint is RELION_ERROR else self.hint.text


def _tail(path: Path) -> tuple[list[str], int]:
    """The lines of the last TAIL_BYTES of a file, and the number (1-based) of the first of them."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            begin = max(0, size - TAIL_BYTES)
            first = 1
            if begin:  # count the lines before the tail (in chunks: run.out can be large)
                left = begin
                while left > 0:
                    chunk = fh.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    first += chunk.count(b"\n")
                    left -= len(chunk)
                fh.seek(begin)
            data = fh.read()
    except OSError:
        return [], 1
    lines = data.decode("utf-8", errors="replace").splitlines()
    if begin and lines:  # the first line is cut
        lines, first = lines[1:], first + 1
    return lines, first


def _relion_error(file: str, lines: list[str], first: int) -> Diagnosis | None:
    """The last of RELION's own error messages: "ERROR:" on a line of its own, the message on the next."""
    for i in range(len(lines) - 1, -1, -1):
        if RELION_ERROR.pattern.match(lines[i].strip()):
            for j in range(i + 1, min(i + 4, len(lines))):
                msg = re.sub(r"^ERROR:?\s*", "", lines[j].strip())  # some messages start with ERROR: again
                if msg:
                    return Diagnosis(RELION_ERROR, file, first + j, msg)
    return None


def diagnose(job_dir: str | os.PathLike) -> list[Diagnosis]:
    """Hints for a failed job, most specific first (at most MAX_HINTS); empty when nothing is recognised."""
    d = Path(job_dir)
    logs = [(name, *_tail(d / name)) for name in LOGS]
    found: dict[str, Diagnosis] = {}
    for hint in HINTS:
        for name, lines, first in logs:  # run.err first: where RELION writes its errors
            hit = next((i for i in range(len(lines) - 1, -1, -1) if hint.pattern.search(lines[i])), None)
            if hit is not None:
                found[hint.id] = Diagnosis(hint, name, first + hit, lines[hit].strip())
                break
    out = [x for x in found.values()
           if not x.hint.fallback and not any(c in found for c in x.hint.covered_by)]
    if not out:
        relion = next((e for name, lines, first in logs if (e := _relion_error(name, lines, first))), None)
        out = [relion] if relion is not None else [x for x in found.values() if x.hint.fallback]
    return out[:MAX_HINTS]
