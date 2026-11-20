# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Suggested MPI processes, threads, pooled particles (and GPU ids) of a job, for the machine or queue node it
runs on (env/machine.py), with warnings about the values in the form that will not work or waste the machine.

The job editor shows the suggestion under the compute options, with Apply (ui/job_editor.py). Nothing is
changed unless Apply is clicked.

The rules, by job:

- relion_refine (2D/3D classification, 3D initial model, 3D auto-refine, multi-body). RELION refuses some
  counts (ml_optimiser_mpi.cpp): VDAM (InitialModel, Class2D with "Use VDAM algorithm") runs without MPI;
  with gold-standard halves (Refine3D, MultiBody) the count must be odd and at least 3; otherwise at least 2
  when MPI is used. The leader takes no GPU; the followers are spread over the GPUs. On GPUs: one follower per
  GPU, two where the GPU memory takes them; for VDAM, 1 MPI and about 6 threads per GPU. Threads: the CPUs
  shared out over every process, leader included (MPI x threads <= CPUs). Pooled particles: 30 on GPUs, as
  in RELION's own Schemes; RELION's default 3 on CPUs. These follow RELION's Schemes (scripts/Schemes/proc).
- Memory (when the box size of the input particles is known): per follower, the padded half Fourier volume
  of each class or body, ~20 bytes per voxel on the GPU (projector and backprojector in float), ~40 in RAM
  (in double), plus ~1 GB. That is the most a follower holds, at full resolution, so it errs on the safe side.
  "Pre-read all particles into RAM" adds box^2 x 4 bytes per particle to each follower (to the leader only
  without parallel disc I/O), as RELION's help says.
- CPU programs: those that only use MPI take a process per CPU (CtfFind, local resolution); Extract and
  Subtract at most 16, as they mostly wait for the disk. MPI and threads: 4 threads a process (12 for RELION's
  motion correction, whose help gives 12 or 18 threads for 36 frames). Polish's parameter training, ResMap
  and reverting subtracted labels: no MPI (RELION refuses it). MotionCor2, Topaz picking and AreTomo2 tilt
  series alignment: one process per GPU.

Locally the MPI count stays within RELION_ERROR_LOCAL_MPI (relion_pipeliner refuses more). On the queue one
node is assumed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.env.machine import Specs

MPI, THREADS, POOL, GPU_IDS, DEDICATED = "nr_mpi", "nr_threads", "nr_pool", "gpu_ids", "min_dedicated"
CLI_THREADS = "--j"
KEYS = (MPI, THREADS, CLI_THREADS, POOL, GPU_IDS, DEDICATED)  # the options a suggestion can set

REFINE = {"relion.class2d", "relion.class3d", "relion.refine3d", "relion.initialmodel", "relion.multibody"}
HALVES = {"relion.refine3d", "relion.multibody"}
HALVES_NAME = {"relion.refine3d": "3D auto-refine", "relion.multibody": "Multi-body refinement"}
GPU_POOL, CPU_POOL = 30, 3
VDAM_THREADS_PER_GPU = 6
MAX_RANKS_PER_GPU = 2
MIN_GPU_THREADS = 4  # a second follower on a GPU only while every process keeps this many threads
GPU_BASE_GB = RAM_BASE_GB = 1.0
GPU_BYTES_PER_VOXEL, RAM_BYTES_PER_VOXEL = 20, 40
GPU_USE, RAM_USE = 0.9, 0.85  # of a GPU's / the machine's memory a job may plan to take
CPU_THREADS = 4  # threads per MPI process of the CPU programs that use both
IO_MPI = 16  # most processes for the programs that mostly read and write
OWN_MOTIONCORR_THREADS = 12

MPI_ONLY = {"relion.ctffind": None, "relion.localres": None, "relion.extract": IO_MPI, "relion.subtract": IO_MPI}
MPI_THREADS = {"relion.polish", "relion.ctfrefine", "relion.pseudosubtomo", "relion.ctfrefinetomo",
               "relion.framealigntomo", "relion.reconstructparticletomo", "relion.reconstructtomograms"}


@dataclass(frozen=True)
class Particles:
    """What the input particles of a refinement say (None: not known)."""
    box: int | None
    count: int | None
    path: str = ""


@dataclass
class Advice:
    values: dict[str, str]  # the suggested values (only options the form has)
    why: list[str] = field(default_factory=list)  # how they were found
    warnings: list[str] = field(default_factory=list)  # about the values in the form
    basis: str = ""  # what they are for ("this machine: 32 CPUs, ...")

    def differs(self, current: Mapping[str, str]) -> bool:
        return any(current.get(k, "").strip() != v for k, v in self.values.items())

    def short(self) -> str:
        """'MPI 5 · threads 6 · pool 30'."""
        names = {MPI: "MPI", THREADS: "threads", CLI_THREADS: "threads", POOL: "pool", GPU_IDS: "GPUs",
                 DEDICATED: "dedicated cores"}
        return " · ".join(f"{names[k]} {v}" for k, v in self.values.items())


def base_key(variant_key: str) -> str:
    return variant_key.removesuffix(":tomo")


def covered(variant_key: str, keys: set[str] | Mapping[str, object]) -> bool:
    """Whether the form of this job gets suggestions."""
    k = base_key(variant_key)
    if k in REFINE or k in MPI_ONLY or k in MPI_THREADS or k in (
            "relion.motioncorr", "relion.autopick", "relion.maskcreate", "relion.aligntiltseries"):
        return True
    return k.startswith("cli:") and (MPI in keys or CLI_THREADS in keys)


# ---------------------------------------------------------------------------- the input particles
def input_star(variant_key: str, values: Mapping[str, str]) -> str:
    """The file that says where a refinement's particles are (project-relative), "" if none is given."""
    v = {k: (values.get(k) or "").strip() for k in ("fn_cont", "fn_img", "fn_in", "in_optimisation",
                                                     "in_particles", "use_direct_entries")}
    if v["fn_cont"]:
        return v["fn_cont"]
    if base_key(variant_key) == "relion.multibody":
        return v["fn_in"]
    if variant_key.endswith(":tomo") and "in_optimisation" in values:
        if v["use_direct_entries"] == "Yes" or not v["in_optimisation"]:
            return v["in_particles"]
        return v["in_optimisation"]
    return v["fn_img"]


def read_particles(project_dir: str | os.PathLike, star: str, count: bool = True) -> Particles | None:
    """Box size (optics rlnImageSize) and number of particles of a particles, optimiser or optimisation-set
    file. None when there is nothing to read."""
    from cryo_savanna.project.star import BIG_TABLES, loop_rows, read_star_head
    project = Path(project_dir)
    path = project / star
    for _ in range(3):  # optimiser -> data, optimisation set -> particles
        try:  # up to the particles: a particles file has none of the other blocks, and is not read through for them
            head = read_star_head(path, ["optics", "optimiser_general", "optimisation_set", ""], stop=BIG_TABLES)
        except (OSError, ValueError):
            return None
        gen = head.get("optimiser_general")
        if gen is not None and gen.get("rlnExperimentalDataStarFile"):
            path = project / str(gen.get("rlnExperimentalDataStarFile"))
            continue
        opt = head.get("optimisation_set") or head.get("")
        if opt is not None and opt.has("rlnTomoParticlesFile") and "optics" not in head:
            path = project / str(opt.get("rlnTomoParticlesFile"))
            continue
        break
    optics = head.get("optics")
    box = None
    if optics is not None and optics.has("rlnImageSize") and optics.rows:
        try:
            box = max(int(float(x)) for x in optics.column("rlnImageSize"))
        except (TypeError, ValueError, OverflowError):
            box = None
    n = None
    if count:
        try:
            rows = loop_rows(path, "particles")
        except OSError:
            rows = None
        n = rows[0] if rows else None
    if box is None and n is None:
        return None
    return Particles(box, n, str(path.relative_to(project)) if path.is_relative_to(project) else str(path))


# ---------------------------------------------------------------------------- helpers
def _int(text: str | None, default: int = 1) -> int:
    try:
        return int(float((text or "").strip()))
    except (ValueError, OverflowError):  # also "nan" and "inf" (OverflowError)
        return default


def _yes(values: Mapping[str, str], key: str) -> bool:
    return (values.get(key) or "").strip() == "Yes"


def typed_gpu_ids(values: Mapping[str, str]) -> list[int]:
    """The distinct GPU ids in 'Which GPUs to use' (0:1, 0,0:1,1, ...), in order."""
    out: list[int] = []
    for tok in re.split(r"[\s,:;]+", values.get(GPU_IDS, "") or ""):
        if tok.isdigit() and int(tok) not in out:
            out.append(int(tok))
    return out


def gb(x: float) -> str:
    return f"{x:.0f} GB" if x >= 10 else f"{x:.1f} GB"


class _Ctx:
    """What every rule needs: the machine, the form's options and values, and the MPI limit."""

    def __init__(self, variant_key: str, keys: set[str], values: Mapping[str, str], specs: Specs,
                 local_mpi_limit: int | None):
        self.key = base_key(variant_key)
        self.keys, self.values, self.specs = keys, values, specs
        self.cpus = max(1, specs.cpus)
        self.mpi_limit = None if specs.queue else local_mpi_limit
        self.threads_key = THREADS if THREADS in keys else CLI_THREADS if CLI_THREADS in keys else None
        self.out: dict[str, str] = {}
        self.why: list[str] = []
        self.warnings: list[str] = []
        typed = typed_gpu_ids(values) if GPU_IDS in keys else []
        self.typed_gpus = typed
        if typed and not specs.queue:
            have = {g.index for g in specs.gpus}
            missing = [i for i in typed if i not in have]
            if missing:
                self.warnings.append(
                    f"GPU {', '.join(map(str, missing))} is not on this machine " +
                    (f"(it has {', '.join(str(g.index) for g in specs.gpus)})." if specs.gpus else "(it has none)."))
            typed = [i for i in typed if i in have]
        self.gpus = len(typed) if typed else len(specs.gpus)
        self.gpu_note = (f"{self.gpus} of the {len(specs.gpus)} GPUs, as chosen in 'Which GPUs to use'"
                         if typed and len(typed) < len(specs.gpus) else "")

    @property
    def gpu_mem(self) -> float | None:
        return self.specs.gpu_mem_gb

    def set(self, key: str | None, value: int | str) -> None:
        if key is not None and key in self.keys:
            self.out[key] = str(value)

    def cap_mpi(self, n: int) -> int:
        if self.mpi_limit is not None and n > self.mpi_limit:
            self.why.append(f"at most {self.mpi_limit} MPI processes locally (RELION_ERROR_LOCAL_MPI)")
            return self.mpi_limit
        return n

    def current(self, key: str | None, default: int = 1) -> int:
        return _int(self.values.get(key, "") if key else "", default)

    def check_cpus(self) -> None:
        """Warn when the form's MPI x threads is more than the CPUs."""
        mpi = self.current(MPI) if MPI in self.keys else 1
        thr = self.current(self.threads_key)
        if mpi * thr > self.cpus:
            self.warnings.append(f"{mpi} MPI × {thr} threads = {mpi * thr} is more than the {self.cpus} CPUs "
                                 f"of {self.specs.where}.")
        if self.mpi_limit is not None and mpi > self.mpi_limit and MPI in self.keys:
            self.warnings.append(f"{mpi} MPI processes is more than RELION allows locally "
                                 f"(RELION_ERROR_LOCAL_MPI = {self.mpi_limit}).")

    def gpu_text(self, mpi: int) -> None:
        """'Which GPUs to use' in the form that spreads this job over the GPUs typed there (none typed: left
        empty, every GPU is used)."""
        ids = [i for i in self.typed_gpus if self.specs.queue or i in {g.index for g in self.specs.gpus}] \
            or self.typed_gpus
        if not ids:
            return
        self.set(GPU_IDS, ("," if mpi == 1 else ":").join(map(str, ids)))


# ---------------------------------------------------------------------------- relion_refine
def _refine(c: _Ctx, particles: Particles | None) -> None:
    v = c.values
    gpu = _yes(v, "use_gpu")
    vdam = c.key == "relion.initialmodel" or (c.key == "relion.class2d" and _yes(v, "do_grad"))
    halves = c.key in HALVES
    is2d = c.key == "relion.class2d"
    classes = max(1, _int(v.get("nr_classes"), 1)) if c.key in ("relion.class2d", "relion.class3d",
                                                                 "relion.initialmodel") else 1
    if c.key == "relion.multibody":
        classes = max(2, classes)  # the bodies: at least two (their number is in the bodies file)
    pad = 1 if _yes(v, "do_pad1") else 2
    box = particles.box if particles else None
    count = particles.count if particles else None

    gpu_rank = ram_rank = None
    if box:
        vox = (pad * box) ** (2 if is2d else 3) / 2
        gpu_rank = GPU_BASE_GB + classes * vox * GPU_BYTES_PER_VOXEL / 1e9
        ram_rank = RAM_BASE_GB + classes * vox * RAM_BYTES_PER_VOXEL / 1e9
    preread = 0.0
    if box and count and _yes(v, "do_preread_images"):
        preread = count * box * box * 4 / 1e9
    ram = c.specs.usable_ram_gb

    if gpu and c.gpus == 0:
        c.warnings.append(f"'Use GPU acceleration' is on, but {c.specs.where} has no GPU.")

    if vdam:
        mpi = 1
        threads = min(c.cpus, max(1, VDAM_THREADS_PER_GPU * c.gpus)) if gpu and c.gpus else c.cpus
        c.why.append("VDAM runs without MPI (RELION refuses it): its threads are spread over the GPUs"
                     if gpu and c.gpus else "VDAM runs without MPI (RELION refuses it): all CPUs as threads")
        if _int(v.get(MPI)) > 1:
            c.warnings.append("VDAM cannot run with MPI: RELION refuses more than 1 MPI process.")
        if gpu and gpu_rank and c.gpu_mem and gpu_rank > GPU_USE * c.gpu_mem:
            c.warnings.append(f"The references may need ~{gb(gpu_rank)} on each GPU, more than its "
                              f"{gb(c.gpu_mem)}.")
    else:
        if gpu and c.gpus:
            per_gpu = 1
            if c.gpu_mem and gpu_rank:
                fit = int(GPU_USE * c.gpu_mem // gpu_rank)
                per_gpu = max(1, min(MAX_RANKS_PER_GPU, fit))
                if fit < 1:
                    c.warnings.append(f"A follower may need up to ~{gb(gpu_rank)} of GPU memory at box {box} px, more "
                                      f"than the {gb(c.gpu_mem)} of a GPU" +
                                      ("" if pad == 1 else ": 'Skip padding' halves it") + ".")
            if _yes(v, "do_blush"):
                per_gpu = 1  # Blush's network takes GPU memory too
            few_cpus = per_gpu > 1 and (per_gpu * c.gpus + 1) * MIN_GPU_THREADS > c.cpus
            if few_cpus:
                per_gpu = 1  # not enough CPUs to feed two each: one, with more threads, as RELION's Schemes
            followers = per_gpu * c.gpus
            if halves and followers % 2:  # an even number of followers: one more on a GPU, or one less
                two_need = 2 * gpu_rank if gpu_rank else None
                two_fit = not (c.gpu_mem and two_need) or two_need <= GPU_USE * c.gpu_mem
                room = (followers + 2) * MIN_GPU_THREADS <= c.cpus  # one more follower keeps its threads
                if followers < 2 or (per_gpu < MAX_RANKS_PER_GPU and two_fit and room):
                    if not two_fit and two_need and c.gpu_mem:
                        c.warnings.append(f"Two halves need two followers: on one GPU they may need "
                                          f"~{gb(two_need)} of its {gb(c.gpu_mem)}.")
                    followers += 1
                else:
                    followers -= 1
            text = f"{per_gpu} follower{'s' if per_gpu > 1 else ''} per GPU"
            if gpu_rank and c.gpu_mem:
                text += f" (~{gb(gpu_rank)} of the {gb(c.gpu_mem)} each at box {box} px, padding {pad}×)"
            elif not box:
                text += " (box size of the particles not known: one, to be safe)"
            else:
                text += " (GPU memory not known: one, to be safe)"
            if few_cpus:
                text += f"; two would fit, but would leave fewer than {MIN_GPU_THREADS} threads each"
            c.why.append(text)
        else:
            # CPUs: as many followers as RAM takes, with a few threads each
            followers = max(1, c.cpus // CPU_THREADS - 1)
            c.why.append(f"on CPUs: about {CPU_THREADS} threads per MPI process")
        if halves:  # an even number of followers, one for each half
            followers = max(2, followers - followers % 2)
        if ram and ram_rank:
            def need(f: int) -> float:
                rd = preread * (f if _yes(v, "do_parallel_discio") else 1)
                return (f + 1) * ram_rank + rd
            # fewer followers for the RAM, but never fewer than the GPUs (the estimate is at full resolution,
            # which classifications seldom reach) nor than two halves
            least = 2 if halves else (c.gpus if gpu and c.gpus else 1)
            while followers - (2 if halves else 1) >= least and need(followers) > RAM_USE * ram:
                followers -= 2 if halves else 1
            if need(followers) > RAM_USE * ram:
                c.warnings.append(f"The job may need up to ~{gb(need(followers))} of RAM: too much for the "
                                  f"{gb(ram)} of {c.specs.where}" + (
                                      " (switch off 'Pre-read all particles into RAM')" if preread else "") + ".")
            elif preread:
                c.why.append(f"pre-reading {count:,} particles takes ~{gb(preread)} in each follower")
        mpi = followers + 1 if followers > 1 or halves else 1  # one follower: no MPI is as fast
        if halves and mpi % 2 == 0:
            mpi += 1
        mpi = c.cap_mpi(mpi)
        if halves and mpi % 2 == 0:
            mpi -= 1
        threads = max(1, c.cpus // mpi)
        cur = _int(v.get(MPI))
        if halves and (cur < 3 or cur % 2 == 0):
            c.warnings.append(f"{HALVES_NAME[c.key]} splits the data into two halves: RELION "
                              f"needs an odd number of MPI processes, at least 3 (not {cur}).")
        elif cur == 2 and not halves:
            c.warnings.append("2 MPI processes is a leader and a single follower: no faster than 1.")
        if gpu and c.gpus and cur > 1 and c.gpu_mem and gpu_rank:
            fit = int(GPU_USE * c.gpu_mem // gpu_rank) * c.gpus
            if cur - 1 > fit:
                c.warnings.append(f"{cur - 1} followers on {c.gpus} GPU{'s' if c.gpus > 1 else ''} may not fit "
                                  f"in GPU memory (~{gb(gpu_rank)} each, {gb(c.gpu_mem)} per GPU).")
    c.set(MPI, mpi)
    c.set(THREADS, threads)
    c.set(POOL, GPU_POOL if gpu and c.gpus else CPU_POOL)
    if gpu:
        c.gpu_text(mpi)
    if POOL in c.keys:
        c.why.append(f"pool {GPU_POOL} on GPUs, as RELION's Schemes use" if gpu and c.gpus
                     else f"pool {CPU_POOL} (RELION's default) on CPUs")


# ---------------------------------------------------------------------------- the others
def _cpu_split(c: _Ctx, threads: int) -> None:
    threads = max(1, min(threads, c.cpus))
    mpi = c.cap_mpi(max(1, round(c.cpus / threads))) if MPI in c.keys else 1
    threads = max(1, c.cpus // mpi)  # every CPU busy
    c.set(MPI, mpi)
    c.set(c.threads_key, threads if c.threads_key else 1)
    c.why.append(f"MPI × threads = the {c.cpus} CPUs")


def _no_mpi(c: _Ctx, why: str) -> None:
    c.set(MPI, 1)
    c.set(c.threads_key, c.cpus)
    c.why.append(why)
    if _int(c.values.get(MPI)) > 1:
        c.warnings.append(why[0].upper() + why[1:] + ".")


def _per_gpu(c: _Ctx, what: str) -> None:
    n = c.cap_mpi(max(1, c.gpus))
    c.set(MPI, n)
    c.set(c.threads_key, 1)
    c.gpu_text(2)  # one GPU per process: 0:1:2
    c.why.append(f"one {what} per GPU ({c.gpus or 'no'} GPU{'s' if c.gpus != 1 else ''})")
    if c.gpus == 0:
        c.warnings.append(f"{c.specs.where.capitalize()} has no GPU.")


def _other(c: _Ctx) -> None:
    v, k = c.values, c.key
    if k == "relion.subtract" and _yes(v, "do_fliplabel"):
        return _no_mpi(c, "reverting particle labels runs without MPI (RELION refuses it)")
    if k == "relion.localres" and not _yes(v, "do_relion_locres"):
        return _no_mpi(c, "ResMap runs without MPI (RELION refuses it)")
    if k == "relion.polish" and _yes(v, "do_param_optim"):
        return _no_mpi(c, "parameter training runs without MPI (RELION refuses it)")
    if k == "relion.motioncorr":
        if not _yes(v, "do_own_motioncor"):
            return _per_gpu(c, "MotionCor2 process")
        _cpu_split(c, OWN_MOTIONCORR_THREADS)
        c.why.append("RELION's motion correction is fastest when the frames divide by the threads")
        return
    if k == "relion.autopick":
        if _yes(v, "do_topaz") and _yes(v, "do_topaz_train"):
            return _no_mpi(c, "Topaz training uses one process")
        if _yes(v, "do_topaz") or (_yes(v, "use_gpu") and not _yes(v, "do_log")):
            return _per_gpu(c, "picking process")
        c.set(MPI, c.cap_mpi(c.cpus))
        c.why.append("picking on CPUs: one MPI process per CPU")
        return
    if k == "relion.aligntiltseries":
        if _yes(v, "do_aretomo2"):
            return _per_gpu(c, "AreTomo2 process")
        c.set(MPI, c.cap_mpi(c.cpus))
        c.why.append("IMOD: one tilt series per MPI process")
        return
    if k == "relion.maskcreate":
        c.set(THREADS, c.cpus)
        c.why.append("all CPUs as threads")
        return
    if k in MPI_ONLY:
        cap = MPI_ONLY[k]
        n = min(c.cpus, cap) if cap else c.cpus
        c.set(MPI, c.cap_mpi(n))
        c.why.append(f"one MPI process per CPU, at most {cap} (it mostly reads and writes)" if cap and c.cpus > cap
                     else "one MPI process per CPU")
        return
    if k.startswith("cli:") and c.specs.queue and MPI in c.keys:
        c.set(MPI, 1)  # External jobs cannot ask the queue for MPI slots (actions/forms.py refuses it)
        c.set(c.threads_key, c.cpus)
        c.why.append("CLI tools on the queue run without MPI")
        return
    if MPI in c.keys and c.threads_key:
        _cpu_split(c, CPU_THREADS)
    elif MPI in c.keys:
        c.set(MPI, c.cap_mpi(c.cpus))
        c.why.append("one MPI process per CPU")
    else:
        c.set(c.threads_key, c.cpus)
        c.why.append("all CPUs as threads")


def advise(variant_key: str, keys: set[str] | Mapping[str, object], values: Mapping[str, str], specs: Specs,
           particles: Particles | None = None, local_mpi_limit: int | None = None) -> Advice | None:
    """The suggestion for a job form (None: not one this covers)."""
    if not covered(variant_key, keys):
        return None
    c = _Ctx(variant_key, set(keys), values, specs, local_mpi_limit)
    if c.key in REFINE:
        _refine(c, particles)
    else:
        _other(c)
    if specs.queue and specs.uses_dedicated and DEDICATED in c.keys:
        c.set(DEDICATED, specs.cpus)
        c.why.append(f"dedicated cores = the {specs.cpus} CPUs of a node, so the job asks for one node")
    c.check_cpus()
    basis = f"{specs.where}: {specs.summary()}"
    if particles and particles.box and c.key in REFINE:
        basis += f"; box {particles.box} px" + (f", {particles.count:,} particles" if particles.count else "")
    if c.gpu_note and (GPU_IDS in c.out or c.key in REFINE and _yes(values, "use_gpu")):
        c.why.insert(0, c.gpu_note)
    c.why += specs.notes
    return Advice(c.out, c.why, c.warnings, basis)

