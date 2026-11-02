# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Short per-job summaries for the job cards and the tree: counts, resolution, key settings,
data quality and progress.

Everything comes from files RELION writes: the job's output nodes in the pipeline (counted by
node type), the last model.star of iterative jobs, postprocess.star, job.star and note.txt.
Reading can take a while on large particle files, so a SummaryReader is meant to run off the GUI
thread. It caches each job's summary and only reads a job again when its directory, status or
job.star changes.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.project.snapshot import JobInfo, stamp
from cryo_savanna.project.star import StarError, loop_rows, read_star_head
from cryo_savanna.relion_consts import STATUS_RUNNING, job_type_for_label

ITERATIVE = {"relion.class2d", "relion.class3d", "relion.initialmodel", "relion.refine3d", "relion.multibody"}
SHOWS_RESOLUTION = {"relion.class3d", "relion.initialmodel", "relion.refine3d", "relion.multibody"}
TOMOGRAM_JOBS = {"relion.reconstructtomograms", "relion.denoisetomo", "relion.ctfrefinetomo", "relion.framealigntomo"}

# Which summary lines a job type can have, so the graph can size its cards before any file is read
RESULT_TYPES = {
    "relion.import", "relion.motioncorr", "relion.ctffind", "relion.manualpick", "relion.autopick", "relion.extract",
    "relion.select", "relion.class2d", "relion.class3d", "relion.initialmodel", "relion.refine3d", "relion.multibody",
    "relion.ctfrefine", "relion.polish", "relion.subtract", "relion.joinstar", "relion.postprocess",
    "relion.importtomo", "relion.excludetilts", "relion.aligntiltseries", "relion.reconstructtomograms",
    "relion.denoisetomo", "relion.picktomo", "relion.pseudosubtomo", "relion.ctfrefinetomo", "relion.framealigntomo",
}
SETTINGS_TYPES = {
    "relion.import", "relion.importtomo", "relion.motioncorr", "relion.autopick", "relion.extract",
    "relion.pseudosubtomo", "relion.class2d", "relion.class3d", "relion.initialmodel", "relion.refine3d",
    "relion.reconstructtomograms", "relion.reconstructparticletomo",
}
QUALITY_TYPES = {"relion.motioncorr", "relion.ctffind", "relion.postprocess"}

_ITER_RE = re.compile(r"^run(?:_ct\d+)?_it(\d+)_(?:half1_)?model\.star$")  # also continued runs (run_ct5_...)
_CONTINUED = re.compile(r"^run_ct(\d+)_")
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def base_type(type_label: str) -> str:
    jt = job_type_for_label(type_label)
    return jt.label if jt else type_label


def continuation(name: str) -> int:
    """5 for a file of a run continued from iteration 5 (run_ct5_...), -1 for a job's first run."""
    m = _CONTINUED.match(name)
    return int(m.group(1)) if m else -1


def current_iterations(files) -> dict[int, str]:
    """iteration -> file, from (iteration, file name) pairs of a job's runs, for its latest run.

    A run continued from iteration N (run_ctN_...) replaces the earlier run's iterations after N:
    where both have an iteration the continuation's file counts, and iterations it has not reached
    (yet) are left out, so the latest iteration is the continuation's.
    """
    files = list(files)
    starts = {continuation(n) for _, n in files} - {-1}
    out: dict[int, tuple[int, str]] = {}
    for it, n in files:
        c = continuation(n)
        if any(c < s < it for s in starts):  # a later continuation started before this iteration
            continue
        if it not in out or c > out[it][0]:
            out[it] = (c, n)
    return {it: n for it, (_, n) in sorted(out.items())}


def summary_lines(type_label: str) -> int:
    """Number of summary lines a job of this type can show (0-3)."""
    t = base_type(type_label)
    return (t in RESULT_TYPES) + (t in SETTINGS_TYPES) + (t in QUALITY_TYPES)


@dataclass
class JobSummary:
    result: list[str] = field(default_factory=list)  # "12,345 particles", "3.20 Å"
    settings: list[str] = field(default_factory=list)  # "K=50", "T=2", "C1"
    quality: list[str] = field(default_factory=list)  # "CTF max res 4.1 Å (median)"
    iteration: int | None = None
    total_iterations: int | None = None
    started: float | None = None  # start of the latest run (note.txt)

    def lines(self) -> list[str]:
        return [" · ".join(x) for x in (self.result, self.settings, self.quality) if x]

    def text(self) -> str:
        return " · ".join(self.result + self.settings + self.quality)

    def progress(self, now: float | None = None) -> str:
        parts = []
        if self.iteration is not None:
            parts.append(f"iter {self.iteration}" + (f"/{self.total_iterations}" if self.total_iterations else ""))
        if self.started is not None:
            parts.append(duration((now or time.time()) - self.started))
        return " · ".join(parts)


def duration(s: float) -> str:
    s = max(0, int(s))
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    return f"{s // 86400} d {s % 86400 // 3600} h"


def resolution(a: float) -> str:
    return f"{a:.2f} Å"


def parse_ctime(text: str) -> float | None:
    """'Tue Oct 28 10:00:00 2026' (C ctime, independent of the locale) -> epoch seconds."""
    parts = text.split()
    try:
        _, mon, day, hms, year = parts[:5]
        h, m, s = (int(x) for x in hms.split(":"))
        return time.mktime((int(year), _MONTHS[mon], int(day), h, m, s, 0, 0, -1))
    except (ValueError, KeyError):
        return None


def last_start(note: Path) -> float | None:
    """When the latest run of a job started, from ' ++++ Executing new job on <ctime>' in note.txt."""
    try:
        text = note.read_text(errors="replace")
    except OSError:
        return None
    marker = "Executing new job on"
    i = text.rfind(marker)
    if i < 0:
        return None
    return parse_ctime(text[i + len(marker):].split("\n", 1)[0])


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _count(v) -> int | None:
    """A positive whole number from a job option ("25", "25.0"), else None ("", "inf", "nan", "0")."""
    f = _float(v)
    return int(f) if f is not None and math.isfinite(f) and f >= 1 else None


def _fmt_num(v: str) -> str:
    f = _float(v)
    return f"{f:g}" if f is not None else v


class SummaryReader:
    def __init__(self, project_dir: str | os.PathLike):
        self.project_dir = Path(project_dir)
        self._cache: dict[str, tuple[tuple, JobSummary]] = {}
        self._counts: dict[str, tuple[tuple, int | None]] = {}  # file -> (stamp, rows)

    def cached(self, name: str) -> JobSummary | None:
        hit = self._cache.get(name)
        return hit[1] if hit else None

    def needs_update(self, info: JobInfo) -> bool:
        """Finished jobs are only read again when their status changes; running ones on every change."""
        hit = self._cache.get(info.name)
        if hit is None or hit[0] is None:
            return True
        if info.status != STATUS_RUNNING and hit[0][0] == info.status:
            return False
        return hit[0] != self._key(info)

    def clear(self) -> None:
        """Forget all summaries (not the row counts, which are keyed by file stamps)."""
        self._cache.clear()

    def summarize(self, info: JobInfo, pipeline: Pipeline) -> JobSummary:
        key = self._key(info)
        hit = self._cache.get(info.name)
        if hit is not None and hit[0] == key:
            return hit[1]
        try:
            s = self._summarize(info.process, pipeline, info.status == STATUS_RUNNING)
        except (OSError, ValueError, KeyError, IndexError, StarError):
            s = JobSummary()  # half-written or unexpected files: try again on the next change
            key = None
        except Exception:  # noqa: BLE001 - a bug here must not stop the summaries of every other job
            print(f"cryo-savanna: cannot summarize {info.name}:", file=sys.stderr)
            traceback.print_exc()
            s = JobSummary()  # kept until the job changes: no traceback on every pass
        self._cache[info.name] = (key, s)
        return s

    def _key(self, info: JobInfo) -> tuple:
        d = self.project_dir / info.name
        return (info.status, stamp(d), stamp(d / "job.star"), stamp(d / "note.txt"))

    # ------------------------------------------------------------------ files
    def _count(self, rel: str, block: str) -> int | None:
        path = self.project_dir / rel
        st = stamp(path)
        if st is None:
            return None
        hit = self._counts.get(rel + "#" + block)
        if hit is not None and hit[0] == st:
            return hit[1]
        r = loop_rows(path, block)
        n = r[0] if r else None
        self._counts[rel + "#" + block] = (st, n)
        return n

    def _column(self, rel: str, block: str, label: str) -> list[float]:
        r = loop_rows(self.project_dir / rel, block, label)
        if not r:
            return []
        return [f for f in (_float(v) for v in r[1]) if f is not None]

    def _head(self, rel: str, block: str):
        path = self.project_dir / rel
        if not path.is_file():
            return None
        return read_star_head(path, [block]).get(block)

    # ------------------------------------------------------------------ per job
    def _summarize(self, pr: Process, pipeline: Pipeline, running: bool) -> JobSummary:
        t = base_type(pr.type_label)
        jobdir = self.project_dir / pr.name
        s = JobSummary()
        try:
            opts = JobStar.read(jobdir / "job.star").options
        except (OSError, ValueError, KeyError):
            opts = {}
        if running:
            s.started = last_start(jobdir / "note.txt")

        if t in ITERATIVE:
            self._iterative(t, pr, opts, s, running)
        else:
            self._counts_from_outputs(t, pr, pipeline, s)
        if t == "relion.postprocess":
            gen = self._head(pr.name + "postprocess.star", "general")
            if gen is not None:
                if _float(gen.get("rlnFinalResolution")) is not None:
                    s.result.append(resolution(float(gen.get("rlnFinalResolution"))))
                b = _float(gen.get("rlnBfactorUsedForSharpening"))
                if b is not None:
                    s.quality.append(f"B-factor {b:.0f} Å²")
        self._settings(t, pr, opts, s)
        self._quality(t, pr, pipeline, s)
        return s

    def _iterative(self, t: str, pr: Process, opts: dict, s: JobSummary, running: bool) -> None:
        jobdir = self.project_dir / pr.name
        try:
            names = os.listdir(jobdir)
        except OSError:
            return
        models = current_iterations((int(m.group(1)), m.group(0)) for m in map(_ITER_RE.match, names) if m)
        last = max(models) if models else None
        root = models[last].split("_it", 1)[0] if last is not None else "run"  # "run" or "run_ct5"
        final_model = "run_model.star" in names
        if last is not None:
            s.iteration = last
        if t == "relion.class2d":
            key = "nr_iter_grad" if opts.get("do_grad") == "Yes" else "nr_iter_em"
            s.total_iterations = _count(opts.get(key))
        elif t in ("relion.class3d", "relion.initialmodel"):
            s.total_iterations = _count(opts.get("nr_iter"))
        if not running:
            s.iteration = s.total_iterations = None  # progress is only shown while running

        data = "run_data.star" if "run_data.star" in names else (
            f"{root}_it{last:03d}_data.star" if last is not None else None)
        if data and data in names:
            n = self._count(pr.name + data, "particles")
            if n is not None:
                s.result.append(f"{n:,} particles")
        if t in SHOWS_RESOLUTION:
            model = "run_model.star" if final_model else (models[last] if last is not None else None)
            if model is not None:
                gen = self._head(pr.name + model, "model_general")
                res = _float(gen.get("rlnCurrentResolution")) if gen is not None else None
                if res is not None:
                    s.result.append(resolution(res))

    def _counts_from_outputs(self, t: str, pr: Process, pipeline: Pipeline, s: JobSummary) -> None:
        nodes = [pipeline.nodes[n] for n in pr.outputs if n in pipeline.nodes]

        def first(kind: str):
            return next((n for n in nodes if n.kind == kind and (self.project_dir / n.name).is_file()), None)

        node = first("ParticleGroupMetadata")
        if node is not None:
            n = self._count(node.name, "particles")
            if n is not None:
                s.result.append(f"{n:,} particles")
        node = first("MicrographCoordsGroup")
        if node is not None:
            r = loop_rows(self.project_dir / node.name, "coordinate_files", "rlnMicrographCoordinates")
            if r:
                total = sum(self._count(f, "") or 0 for f in r[1])
                s.result.append(f"{total:,} particles · {r[0]:,} micrographs")
        if not s.result:
            for kind, block, what in (("MicrographGroupMetadata", "micrographs", "micrographs"),
                                      ("MicrographMovieGroupMetadata", "movies", "movies")):
                node = first(kind)
                if node is not None:
                    n = self._count(node.name, block)
                    if n is not None:
                        s.result.append(f"{n:,} {what}")
                        break
        if not s.result:
            node = first("TomogramGroupMetadata")
            if node is not None:
                n = self._count(node.name, "global")
                if n is not None:
                    s.result.append(f"{n:,} {'tomograms' if t in TOMOGRAM_JOBS else 'tilt series'}")
        if t == "relion.select":
            node = first("Image2DGroupMetadata")
            if node is not None:
                n = self._count(node.name, "")
                if n:
                    s.result.append(f"{n} classes")

    def _settings(self, t: str, pr: Process, opts: dict, s: JobSummary) -> None:
        o = opts.get
        if t in ("relion.import", "relion.importtomo"):
            if o("angpix"):
                s.settings.append(f"{_fmt_num(o('angpix'))} Å/px")
            if t == "relion.import" and o("kV"):
                s.settings.append(f"{_fmt_num(o('kV'))} kV")
        elif t == "relion.motioncorr":
            if _float(o("bin_factor")) not in (None, 1.0):
                s.settings.append(f"bin {_fmt_num(o('bin_factor'))}")
        elif t == "relion.autopick":
            method = ("LoG" if o("do_log") == "Yes" else "Topaz" if o("do_topaz") == "Yes"
                      else "3D refs" if o("do_ref3d") == "Yes" else "2D refs" if o("do_refs") == "Yes" else "")
            if method:
                s.settings.append(method)
        elif t in ("relion.extract", "relion.pseudosubtomo"):
            s.settings += self._extract_boxes(t, pr, opts)
        elif t == "relion.reconstructtomograms":
            if o("binned_angpix"):
                s.settings.append(f"{_fmt_num(o('binned_angpix'))} Å/px")
        elif t == "relion.reconstructparticletomo":
            s.settings += [x for x in (o("sym_name"), f"box {o('box_size')} px" if o("box_size") else "",
                                       f"bin {_fmt_num(o('binning'))}" if o("binning") else "") if x]
        elif t in ("relion.class2d", "relion.class3d", "relion.initialmodel"):
            if o("nr_classes"):
                s.settings.append(f"K={o('nr_classes')}")
            if o("tau_fudge"):
                s.settings.append(f"T={_fmt_num(o('tau_fudge'))}")
            if t != "relion.class2d" and o("sym_name"):
                s.settings.append(o("sym_name"))
        elif t == "relion.refine3d":
            if o("sym_name"):
                s.settings.append(o("sym_name"))

    def _extract_boxes(self, t: str, pr: Process, opts: dict) -> list[str]:
        """Initial and final (binned) box size and the final pixel size of extracted particles.

        The final values come from the optics table of the output particles.star; before there
        is one (running jobs), from the job options.
        """
        o = opts.get
        node = next((n for n in pr.outputs if n.endswith("particles.star")), None)
        optics = self._head(node, "optics") if node else None

        def opt(label: str) -> float | None:
            return _float(optics.get(label)) if optics is not None and optics.has(label) else None

        final, pixel = opt("rlnImageSize"), opt("rlnImagePixelSize")
        if t == "relion.extract":
            initial = _float(o("extract_size"))
            if final is None and initial is not None:
                final = _float(o("rescale")) if o("do_rescale") == "Yes" else initial
        else:
            # relion_tomo_subtomo: box_size_angst is binned so that it becomes binned_size_pix
            binning = opt("rlnTomoSubtomogramBinning")
            initial = round(final * binning) if final is not None and binning else None
            if final is None:
                size_a, size_px = _float(o("box_size_angst")), _float(o("binned_size_pix"))
                if size_a and size_px:
                    return [f"box {size_a:g} Å → {size_px:g} px", f"{size_a / size_px:.4g} Å/px"]
                return []
        out = []
        if final is not None:
            out.append(f"box {initial:g} → {final:g} px" if initial is not None and initial != final
                       else f"box {final:g} px")
        if pixel:
            out.append(f"{pixel:.4g} Å/px")
        return out

    def _quality(self, t: str, pr: Process, pipeline: Pipeline, s: JobSummary) -> None:
        mics = next((n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == "MicrographGroupMetadata"),
                    None)
        tilts = next((n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == "TomogramGroupMetadata"),
                     None)
        if t == "relion.motioncorr" and mics:
            v = self._column(mics, "micrographs", "rlnAccumMotionTotal")
            if v:
                s.quality.append(f"motion {statistics.fmean(v):.1f} Å (mean)")
        elif t == "relion.ctffind":
            v: list[float] = []
            if mics:
                v = self._column(mics, "micrographs", "rlnCtfMaxResolution")
            elif tilts:
                # per-tilt values live in one STAR file per tilt series, in a block named after it
                glob = self._head(tilts, "global")
                for rec in (glob.records() if glob is not None else ()):
                    ts_file = rec.get("rlnTomoTiltSeriesStarFile", "")
                    if ts_file and (self.project_dir / ts_file).is_file():
                        v += self._column(ts_file, rec.get("rlnTomoName", ""), "rlnCtfMaxResolution")
            if v:
                s.quality.append(f"CTF max res {statistics.median(v):.1f} Å (median)")
