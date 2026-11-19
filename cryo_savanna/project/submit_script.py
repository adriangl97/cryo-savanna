# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# fill() follows RelionJob::saveJobSubmissionScript in src/pipeline_jobs.cpp. RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
"""The script RELION writes when it sends a job to the queue (run_submit.script), before it does, and what may be
wrong with it (no Qt): the job editor's "Preview submission script…".

The template is filled in as RelionJob::saveJobSubmissionScript does (src/pipeline_jobs.cpp, 5.1.1): the MPI
processes, threads, cores, dedicated cores and nodes, the job's name and log files, the queue name and the extra
fields. Where RELION puts the job's commands (XXXcommandXXX), the preview has those of the job's last run (note.txt)
when there is one, else a line saying so: relion_pipeliner makes them when it sends the job.

The checks, for SLURM templates (#SBATCH lines): placeholders left (an extra field this RELION does not have),
options left empty, a partition sinfo does not know, a time limit SLURM cannot read or longer than the partition
allows, GPUs the job uses but does not ask for (or more than it asks for), and a submit command not on PATH.
"""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.env.machine import Runner, parse_queue_request, run_quiet

COMMAND_PLACEHOLDER = "<the job's command: relion_pipeliner puts it here when it sends the job>"


@dataclass
class Preview:
    script: str
    problems: list[str] = field(default_factory=list)  # what would make the job fail or not start
    notes: list[str] = field(default_factory=list)  # worth knowing


def _int(v: str | None, default: int) -> int:
    try:
        return int(float(v)) if v not in (None, "") else default
    except (TypeError, ValueError, OverflowError):
        return default


def last_commands(note: Path) -> list[str]:
    """The commands of a job's last run, from its note.txt (" ++++ with the following command(s): ")."""
    try:
        text = note.read_text(errors="replace")
    except OSError:
        return []
    i = text.rfind("with the following command(s):")
    if i < 0:
        return []
    out = []
    for line in text[i:].splitlines()[1:]:
        if line.lstrip().startswith("++++"):
            break
        if line.strip():
            out.append(line.strip())
    return out


def fill(template: str, values: Mapping[str, str], job: str, extra_count: int,
         commands: list[str] | None = None) -> str:
    """The template as RELION fills it in (saveJobSubmissionScript)."""
    nmpi = _int(values.get("nr_mpi"), 1) if "nr_mpi" in values else 1
    nthr = _int(values.get("nr_threads"), 1) if "nr_threads" in values else 1
    ncores = nmpi * nthr
    ndedi = max(1, _int(values.get("min_dedicated"), 1))
    nodes = math.ceil(ncores / ndedi)
    fixed = {"XXXmpinodesXXX": str(nmpi), "XXXthreadsXXX": str(nthr), "XXXcoresXXX": str(ncores),
             "XXXdedicatedXXX": str(ndedi), "XXXnodesXXX": str(nodes), "XXXnameXXX": job,
             "XXXerrfileXXX": job + "run.err", "XXXoutfileXXX": job + "run.out",
             "XXXqueueXXX": values.get("queuename", "")}
    out = []
    for line in template.split("\n"):
        for k, v in fixed.items():
            line = line.replace(k, v)
        for m in range(1, extra_count + 1):
            if f"qsub_extra{m}" in values:
                line = line.replace(f"XXXextra{m}XXX", values[f"qsub_extra{m}"])
        if "XXXcommandXXX" in line:
            out += [line.replace("XXXcommandXXX", c) for c in (commands or [COMMAND_PLACEHOLDER])]
        else:
            out.append(line)
    return "\n".join(out)


_TIME = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$")


def slurm_seconds(text: str) -> float | None:
    """A SLURM time ("90", "1:30:00", "2-00:00:00", "2-12"; minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS) in
    seconds, inf for UNLIMITED / INFINITE, None for what SLURM cannot read."""
    t = text.strip().lower()
    if t in ("unlimited", "infinite"):
        return math.inf
    m = _TIME.match(t)
    if not m:
        return None
    days, a, b, c = m.groups()
    if days is not None:  # D-HH[:MM[:SS]]
        return int(days) * 86400 + int(a) * 3600 + int(b or 0) * 60 + int(c or 0)
    if b is None:  # minutes
        return int(a) * 60
    if c is None:  # MM:SS
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)  # HH:MM:SS


def _sbatch_lines(script: str) -> list[str]:
    return [line.strip()[len("#SBATCH"):].split("#", 1)[0].strip() for line in script.splitlines()
            if line.strip().startswith("#SBATCH")]


def check(script: str, template: str, values: Mapping[str, str], gpu_ids: list[int] | None,
          run: Runner = run_quiet, which: Callable[[str], str | None] = shutil.which) -> tuple[list[str], list[str]]:
    """(problems, notes) of a filled-in script. ``gpu_ids``: the GPUs the job uses ([] all it is given, None none)."""
    problems: list[str] = []
    notes: list[str] = []
    left = sorted({p for p in re.findall(r"XXX\w+?XXX", script)})
    if left:
        problems.append("Not filled in: " + ", ".join(left) + " (an extra queue field this RELION does not have: "
                        "see RELION_QSUB_EXTRA_COUNT and RELION_QSUB_EXTRA<n>)")
    qsub = (values.get("qsub") or "").split()
    if qsub and which(qsub[0]) is None:
        problems.append(f"The submit command {qsub[0]} is not on PATH here")
    nmpi, nthr = _int(values.get("nr_mpi"), 1), _int(values.get("nr_threads"), 1)
    ndedi = max(1, _int(values.get("min_dedicated"), 1))
    if ("XXXnodesXXX" in template or "XXXdedicatedXXX" in template) and (nmpi * nthr) % ndedi:
        notes.append(f"{nmpi} MPI × {nthr} threads = {nmpi * nthr} cores is not a multiple of the {ndedi} dedicated "
                     "cores per node: the last node is only partly used (RELION warns about this too)")
    lines = _sbatch_lines(script)
    if not lines:
        notes.append("No #SBATCH lines: only SLURM templates are checked further")
        return problems, notes
    for opt in lines:
        if re.match(r"--[\w-]+=\s*$", opt):
            problems.append(f"#SBATCH {opt}: no value (an extra queue field left empty?)")
    req = parse_queue_request(template, values)
    have_sinfo = which("sinfo") is not None
    partitions = [p for p in (req.partition or "").split(",") if p]
    if partitions and have_sinfo:
        known = {p.rstrip("*") for p in run(["sinfo", "-h", "-o", "%P"]).split()}
        if known:
            for p in partitions:
                if p not in known:
                    problems.append(f"Partition {p} does not exist here (sinfo knows: {', '.join(sorted(known))})")
    times = [m.group(1) for opt in lines for m in [re.match(r"(?:--time[=\s]+|-t\s*)(\S+)$", opt)] if m]
    if times:
        limit = slurm_seconds(times[-1])
        if limit is None:
            problems.append(f"Time limit {times[-1]!r}: not a time SLURM reads (minutes, HH:MM:SS, D-HH:MM:SS, ...)")
        elif partitions and have_sinfo and math.isfinite(limit):
            words = run(["sinfo", "-h", "-p", partitions[0], "-o", "%l"]).split()
            allowed = [(t, w) for w in words for t in [slurm_seconds(w)] if t is not None]
            if allowed and limit > max(t for t, _ in allowed):
                longest = max(allowed)[1]
                problems.append(f"Time limit {times[-1]} is longer than partition {partitions[0]} allows ({longest})")
    if gpu_ids is not None:
        if not req.gpus:
            problems.append("The job uses GPUs, but the script asks for none (no --gres=gpu:N or --gpus=N)")
        elif gpu_ids and len(gpu_ids) > req.gpus:
            problems.append(f"The job uses GPUs {', '.join(map(str, gpu_ids))}, but the script asks for "
                            f"{req.gpus}: SLURM numbers the GPUs it gives from 0")
        elif gpu_ids and max(gpu_ids) >= req.gpus:
            problems.append(f"GPU id {max(gpu_ids)} is not among the {req.gpus} the script asks for (SLURM numbers "
                            "them from 0)")
    elif req.gpus:
        notes.append(f"The script asks for {req.gpus} GPU(s), which the job does not use (with these settings)")
    return problems, notes


def preview(template_path: str, values: Mapping[str, str], job: str, extra_count: int, project_dir: Path,
            gpu_ids: list[int] | None, run: Runner = run_quiet,
            which: Callable[[str], str | None] = shutil.which) -> Preview:
    """The script and its checks. The template is read here: call from a thread (a slow file system)."""
    try:
        template = Path(template_path).read_text(errors="replace")
    except OSError as e:
        return Preview("", [f"The submission script {template_path or '(none)'} cannot be read: "
                            f"{e.strerror or e}"])
    commands = last_commands(project_dir / job / "note.txt")
    script = fill(template, values, job, extra_count, commands)
    problems, notes = check(script, template, values, gpu_ids, run, which)
    if commands:
        notes.insert(0, "The commands are those of the job's last run (note.txt); RELION makes them again when it "
                        "sends the job")
    return Preview(script, problems, notes)
