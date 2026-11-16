# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What a job will run on: the CPUs, RAM and GPUs of this machine, or of a node of the SLURM partition a queued
job goes to. The job editor's compute suggestions (jobdefs/compute_advice.py) are made from these.

This machine: the CPUs this process may use (its affinity, so a cgroup or an interactive SLURM allocation counts),
the RAM (a cgroup limit when it is lower), and the GPUs nvidia-smi lists, only those CUDA_VISIBLE_DEVICES leaves.

The queue (SLURM only): the submission script (the job's queue template, as RELION fills it in) says which
partition, how many GPUs and how much memory the job asks for: its #SBATCH lines are read with the job's own
Queue name and RELION_QSUB_EXTRA values put in. sinfo then gives the CPUs, memory and GPUs (gres) of the
partition's nodes, the smallest where they differ. SLURM does not report GPU memory: it is taken from the gres
type's name (a100 -> 40 GB) where that is a known GPU, else left unknown.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCAL_TTL_S = 600.0  # this machine's GPUs and RAM are looked at again after this
QUEUE_TTL_S = 300.0  # a partition's nodes
CALL_TIMEOUT_S = 15.0

Runner = Callable[[list[str]], str]  # runs a command, returns its stdout ("" when it fails)


def run_quiet(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                           universal_newlines=True, timeout=CALL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


@dataclass(frozen=True)
class Gpu:
    index: int  # as the job will number it (CUDA_VISIBLE_DEVICES renumbers from 0)
    name: str
    mem_gb: float | None  # None: not known


@dataclass
class Specs:
    """The resources of the machine (or one queue node) a job runs on."""
    where: str  # "this machine", "partition gpu"
    cpus: int
    ram_gb: float | None
    gpus: list[Gpu] = field(default_factory=list)
    queue: bool = False
    gpus_requested: int | None = None  # queue: the GPUs the submission script asks for (None: not said)
    ram_requested_gb: float | None = None  # queue: --mem of the submission script
    uses_dedicated: bool = False  # queue: the template uses XXXdedicatedXXX or XXXnodesXXX
    notes: list[str] = field(default_factory=list)  # what was guessed or could not be found

    @property
    def gpu_mem_gb(self) -> float | None:
        """The smallest GPU's memory (None: no GPU, or one of unknown memory)."""
        mems = [g.mem_gb for g in self.gpus]
        return None if not mems or any(m is None for m in mems) else min(m for m in mems if m is not None)

    @property
    def usable_ram_gb(self) -> float | None:
        return self.ram_requested_gb if self.ram_requested_gb is not None else self.ram_gb

    def summary(self) -> str:
        """'32 CPUs, 251 GB RAM, 4× RTX 3090 (24 GB)'."""
        bits = [f"{self.cpus} CPUs"]
        ram = self.usable_ram_gb
        if ram is not None:
            bits.append(f"{ram:.0f} GB RAM" + (" requested" if self.ram_requested_gb is not None else ""))
        if self.gpus:
            names = sorted({g.name for g in self.gpus})
            mem = self.gpu_mem_gb
            what = names[0] if len(names) == 1 else "GPUs"
            n = len(self.gpus)
            bits.append(f"{n}× {what}" + (f" ({mem:.0f} GB)" if mem is not None else ""))
        else:
            bits.append("no GPU")
        return ", ".join(bits)


# ---------------------------------------------------------------------------- this machine
def local_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]  # Linux
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _meminfo_gb(text: str, key: str = "MemTotal") -> float | None:
    m = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.M)
    return int(m.group(1)) / 1024 ** 2 if m else None


def _cgroup_limit_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 1 << 60:  # "max", or the huge number of "no limit"
            return int(v) / 1024 ** 3
    return None


def local_ram_gb(run: Runner = run_quiet) -> float | None:
    total = None
    try:
        total = _meminfo_gb(Path("/proc/meminfo").read_text())
    except OSError:
        if sys.platform == "darwin":
            out = run(["sysctl", "-n", "hw.memsize"]).strip()
            total = int(out) / 1024 ** 3 if out.isdigit() else None
    limit = _cgroup_limit_gb()
    if limit is not None and (total is None or limit < total):
        return limit
    return total


def parse_nvidia_smi(text: str) -> list[tuple[int, str, str, float | None]]:
    """(index, uuid, name, memory GB) of `nvidia-smi --query-gpu=index,uuid,name,memory.total
    --format=csv,noheader,nounits` (memory in MiB)."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            mem = float(parts[3]) / 1024
        except ValueError:
            mem = None
        out.append((int(parts[0]), parts[1], parts[2], mem))
    return out


def visible_gpus(found: list[tuple[int, str, str, float | None]], environ: Mapping[str, str]) -> list[Gpu]:
    """The GPUs a job started from here sees, numbered as CUDA numbers them for it."""
    cvd = environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return [Gpu(i, _short_name(name), mem) for i, _, name, mem in found]
    chosen = []
    for tok in [t.strip() for t in cvd.split(",") if t.strip()]:
        hit = next((f for f in found if str(f[0]) == tok or f[1] == tok or f[1].startswith(tok)), None)
        if hit is None:
            break  # CUDA stops at the first id it does not know
        chosen.append(hit)
    return [Gpu(i, _short_name(name), mem) for i, (_, _, name, mem) in enumerate(chosen)]


def _short_name(name: str) -> str:
    return re.sub(r"^(NVIDIA|GeForce|Quadro|Tesla)\s+", "", re.sub(r"^NVIDIA\s+", "", name)).strip() or name


def local_specs(environ: Mapping[str, str] | None = None, run: Runner = run_quiet,
                which: Callable[[str], str | None] = shutil.which) -> Specs:
    environ = os.environ if environ is None else environ
    specs = Specs("this machine", local_cpus(), local_ram_gb(run))
    if which("nvidia-smi"):
        found = parse_nvidia_smi(run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                      "--format=csv,noheader,nounits"]))
        specs.gpus = visible_gpus(found, environ)
    if specs.ram_gb is None:
        specs.notes.append("the RAM could not be read")
    return specs


# ---------------------------------------------------------------------------- the queue template
@dataclass
class QueueRequest:
    """What a filled-in SLURM submission script asks for."""
    partition: str | None = None
    gpus: int | None = None
    mem_gb: float | None = None
    uses_dedicated: bool = False
    is_slurm: bool = False  # it has #SBATCH lines
    unresolved: list[str] = field(default_factory=list)  # placeholders that were not filled (XXXthreadsXXX)


_SIZE = {"": 1 / 1024, "K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}  # SLURM's default unit is MB


def fill_template(text: str, values: Mapping[str, str]) -> str:
    """The template's XXXqueueXXX and XXXextraNXXX as RELION puts them in, from the job's values."""
    text = text.replace("XXXqueueXXX", values.get("queuename", ""))
    return re.sub(r"XXXextra(\d+)XXX", lambda m: values.get(f"qsub_extra{m.group(1)}", m.group(0)), text)


def parse_queue_request(template: str, values: Mapping[str, str]) -> QueueRequest:
    req = QueueRequest(uses_dedicated="XXXdedicatedXXX" in template or "XXXnodesXXX" in template)
    for raw in fill_template(template, values).splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        req.is_slurm = True
        line = line[len("#SBATCH"):].split("#", 1)[0]
        m = re.search(r"(?:--partition[=\s]+|(?<!\S)-p\s*)(\S+)", line)
        if m:
            req.partition = m.group(1)
        m = re.search(r"--gres[=\s]+(\S+)", line)
        if m:
            for item in m.group(1).split(","):
                if "XXX" in item:  # a placeholder RELION does not fill (or a field this RELION lacks)
                    continue
                g = re.match(r"gpu(?::([\w.-]+?))??(?::(\d+))?$", item)
                if g:
                    req.gpus = int(g.group(2)) if g.group(2) else 1
        m = re.search(r"--gpus(?:-per-node)?[=\s]+(?:[\w.-]+:)?(\d+)\b", line) or \
            re.search(r"(?<!\S)-G\s*(\d+)\b", line)
        if m:
            req.gpus = int(m.group(1))
        m = re.search(r"--mem[=\s]+(\d+(?:\.\d+)?)([KMGT]?)B?\b", line, re.I)
        if m:
            req.mem_gb = float(m.group(1)) * _SIZE[m.group(2).upper()]
        req.unresolved += [p for p in re.findall(r"XXX\w+?XXX", line) if p not in req.unresolved]
    if req.partition and "XXX" in req.partition:
        req.partition = None
    return req


# ---------------------------------------------------------------------------- SLURM nodes
# GPU memory by gres type name (SLURM does not report it). Matched after lower-casing and dropping "-", "_",
# " ", longest name first; a type ending in a size ("a100_80gb", "a100-80g") gives that size instead.
GPU_MEMORY_GB = {
    "h200": 141, "h100": 80, "gh200": 96, "a100": 40, "a40": 48, "a30": 24, "a10": 24, "a16": 16, "a2": 16,
    "l40s": 48, "l40": 48, "l4": 24, "v100s": 32, "v100": 16, "p100": 16, "p40": 24, "t4": 16, "k80": 12,
    "rtxa6000": 48, "a6000": 48, "a5500": 24, "a5000": 24, "a4500": 20, "a4000": 16, "rtx6000ada": 48,
    "rtx6000": 24, "rtx8000": 48, "rtx5000": 16, "rtx4000": 8, "quadrortx6000": 24, "quadrortx8000": 48,
    "rtx5090": 32, "rtx4090": 24, "rtx4080": 16, "rtx3090": 24, "rtx3080ti": 12, "rtx3080": 10,
    "rtx2080ti": 11, "rtx2080": 8, "gtx1080ti": 11, "gtx1080": 8, "titanrtx": 24, "titanv": 12, "titanx": 12,
    "5090": 32, "4090": 24, "3090": 24, "2080ti": 11, "1080ti": 11, "b200": 180,
}


def gpu_memory_of(kind: str) -> float | None:
    m = re.search(r"(?:^|[-_ ])(\d+)\s*g(?:b)?$", kind.lower())  # "a100_80gb", "a100-80g"
    if m and int(m.group(1)) >= 4:
        return float(m.group(1))
    k = re.sub(r"[-_ ]", "", kind.lower())
    for name in sorted(GPU_MEMORY_GB, key=len, reverse=True):
        if name in k:
            return float(GPU_MEMORY_GB[name])
    return None


@dataclass(frozen=True)
class NodeKind:
    cpus: int
    mem_gb: float | None
    gpus: int
    gpu_type: str
    nodes: int


def parse_gres(gres: str) -> tuple[int, str]:
    """(GPUs, type) of a node's gres ("gpu:a100:4(S:0-1),tmpfs:100G" -> (4, "a100"))."""
    for item in re.split(r",(?![^()]*\))", gres):
        item = re.sub(r"\(.*?\)", "", item.strip())
        parts = item.split(":")
        if parts[0] != "gpu":
            continue
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1]), ""
        if len(parts) >= 3 and parts[-1].isdigit():
            return int(parts[-1]), parts[1]
        return 1, parts[1] if len(parts) > 1 else ""
    return 0, ""


def parse_sinfo(text: str) -> list[NodeKind]:
    """`sinfo -h -o "%c|%m|%G|%D"` (CPUs per node, memory MB, gres, node count): one line per kind of node."""
    out = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        try:
            cpus = int(re.match(r"\d+", parts[0]).group())  # type: ignore[union-attr]  # "48+" when they differ
        except (AttributeError, ValueError):
            continue
        mem = re.match(r"\d+", parts[1])
        gpus, kind = parse_gres(parts[2])
        nodes = re.match(r"\d+", parts[3])
        out.append(NodeKind(cpus, int(mem.group()) / 1024 if mem else None, gpus, kind,
                            int(nodes.group()) if nodes else 1))
    return out


def default_partition(run: Runner = run_quiet) -> str | None:
    for line in run(["sinfo", "-h", "-o", "%P"]).split():
        if line.endswith("*"):
            return line[:-1]
    return None


def queue_specs(template: str, values: Mapping[str, str], run: Runner = run_quiet,
                which: Callable[[str], str | None] = shutil.which) -> Specs | str:
    """The node a queued job runs on, or why it cannot be told."""
    if which("sinfo") is None:
        return "no SLURM here (sinfo not found)"
    req = parse_queue_request(template, values)
    if not req.is_slurm:
        return "the submission script has no #SBATCH lines (only SLURM is read)"
    partition = req.partition or default_partition(run)
    if not partition:
        return "the submission script names no partition and SLURM has no default one"
    kinds = parse_sinfo(run(["sinfo", "-h", "-p", partition, "-o", "%c|%m|%G|%D"]))
    if not kinds:
        return f"sinfo knows no nodes in partition {partition}"
    notes = []
    if req.gpus:  # the job asks for GPUs: only the nodes that have them can take it
        kinds = [k for k in kinds if k.gpus >= req.gpus] or kinds
    if len(kinds) > 1:
        notes.append(f"the nodes of {partition} differ: the smallest is assumed")
    cpus = min(k.cpus for k in kinds)
    mems = [k.mem_gb for k in kinds if k.mem_gb]
    ngpu = min(k.gpus for k in kinds)
    types = sorted({k.gpu_type for k in kinds if k.gpus})
    gpu_mem = [gpu_memory_of(t) for t in types]
    mem = None if not gpu_mem or any(m is None for m in gpu_mem) else min(m for m in gpu_mem if m is not None)
    if ngpu and mem is None:
        notes.append("GPU memory unknown (gres type " + (", ".join(types) or "not given") + ")")
    if req.gpus is None:
        n = ngpu
        if ngpu:
            notes.append(f"the submission script does not say how many GPUs: all {ngpu} of a node assumed")
    else:
        n = min(req.gpus, ngpu) if ngpu else req.gpus
    name = types[0].upper() if len(types) == 1 and types[0] else "GPU"
    return Specs(f"partition {partition}", cpus, min(mems) if mems else None,
                 [Gpu(i, name, mem) for i in range(n)], queue=True, gpus_requested=req.gpus,
                 ram_requested_gb=req.mem_gb, uses_dedicated=req.uses_dedicated, notes=notes)


# ---------------------------------------------------------------------------- cache
class SpecsCache:
    """Specs looked up once in a while, never on the GUI thread: get() gives what is known (None: not yet),
    and starts a lookup in a thread whose end calls ``on_ready`` (from that thread)."""

    def __init__(self, local: Callable[[], Specs] = local_specs,
                 queue: Callable[[str, Mapping[str, str]], Specs | str] = queue_specs):
        self._local, self._queue = local, queue
        self._known: dict[tuple, tuple[float, Any]] = {}
        self._pending: set[tuple] = set()
        self._lock = threading.Lock()

    def get(self, key: tuple, ttl: float, fn: Callable[[], Any],
            on_ready: Callable[[], None] | None = None) -> Any:
        """The value of ``key`` (None: not known yet); looked up with fn() in a thread when older than ``ttl``
        seconds. fn's errors become a text saying so."""
        with self._lock:
            hit = self._known.get(key)
            fresh = hit is not None and time.monotonic() - hit[0] < ttl
            if fresh or key in self._pending:
                return hit[1] if hit else None
            self._pending.add(key)

        def work() -> None:
            try:
                value = fn()
            except Exception as e:  # noqa: BLE001 - shown as the reason there is no suggestion
                value = f"could not look: {e}"
            with self._lock:
                self._known[key] = (time.monotonic(), value)
                self._pending.discard(key)
            if on_ready is not None:
                on_ready()

        threading.Thread(target=work, name="cryo-savanna-specs", daemon=True).start()
        return hit[1] if hit else None

    def local(self, on_ready: Callable[[], None] | None = None) -> Specs | str | None:
        return self.get(("local",), LOCAL_TTL_S, self._local, on_ready)

    def for_queue(self, template_path: str, values: Mapping[str, str],
                  on_ready: Callable[[], None] | None = None) -> Specs | str | None:
        """The node of a queued job: its submission script is read in the thread too (a slow file system
        does not freeze the window)."""
        fields = {k: values.get(k, "") for k in values if k == "queuename" or k.startswith("qsub_extra")}
        key = ("queue", template_path, tuple(sorted(fields.items())))

        def look() -> Specs | str:
            try:
                template = Path(template_path).read_text(errors="replace")
            except OSError:
                return f"the submission script {template_path or '(none)'} cannot be read"
            return self._queue(template, fields)

        return self.get(key, QUEUE_TTL_S, look, on_ready)

    def set(self, key: tuple, value: Any) -> None:
        """Tests: a value as if it had been looked up."""
        with self._lock:
            self._known[key] = (time.monotonic(), value)
