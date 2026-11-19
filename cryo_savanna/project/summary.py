# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Short per-job summaries for the job cards and the tree: counts, resolution, key settings,
data quality and progress, and for failed jobs the likely cause (project/failure_hints.py).

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

from cryo_savanna.project.failure_hints import Diagnosis, diagnose
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline, Process
from cryo_savanna.project.snapshot import JobInfo, stamp
from cryo_savanna.project.star import StarError, loop_rows, read_star, read_star_head
from cryo_savanna.relion_consts import STATUS_FAILED, STATUS_RUNNING, job_type_for_label

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
    "relion.reconstructtomograms", "relion.reconstructparticletomo", "relion.polish", "relion.subtract",
    "relion.maskcreate", "relion.localres", "relion.ctfrefine", "relion.ctfrefinetomo", "relion.framealigntomo",
}
QUALITY_TYPES = {"relion.motioncorr", "relion.ctffind", "relion.postprocess", "relion.localres", "relion.multibody"}

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


PACE_ITERATIONS = 4  # a running job's time per iteration is taken over this many of its latest iterations
MIN_PACE_S = 1.0


def _pace(jobdir: Path, models: dict[int, str], root: str) -> tuple[float | None, float | None]:
    """(seconds per iteration, when the latest iteration was written) of a running iterative job, from the times
    its latest model files were written. Only the files of its latest run count (``root``: "run", "run_ct5"): a
    continued run's earlier iterations may be days older. VDAM writes its files every few iterations
    (--grad_write_iter): the time between two files is divided by the iterations between them."""
    times = []
    for it, n in [(it, n) for it, n in models.items() if n.startswith(root + "_it")][-(PACE_ITERATIONS + 1):]:
        st = stamp(jobdir / n)
        if st is not None:
            times.append((it, st[0] / 1e9))
    if not times:
        return None, None
    (first_it, first_t), (last_it, last_t) = times[0], times[-1]
    per = (last_t - first_t) / (last_it - first_it) if last_it > first_it else 0.0
    # under a second: files written together (a job copied or restored), not the pace of its iterations
    return (per if per >= MIN_PACE_S else None), last_t


# The tables cryo-savanna's runners keep of the items they work through: (file, what the items are)
ITEM_TABLES = (("aretomo3.star", "tilt series"), ("pytom_match.star", "tomograms"))
# RELION's progress bar (src/time.cpp): "  3.12/  5.40 min ..........~~(,_,">", time so far / estimated total
_BAR = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s+(sec|min|hrs?|hours?|days?)\s*\.*~~\(,_,\">")
_UNIT_S = {"sec": 1, "min": 60, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "day": 86400, "days": 86400}
BAR_TAIL_BYTES = 8192


def _progress_bar(run_out: Path, s: "JobSummary") -> bool:
    """When the step RELION's progress bar is at should end (s.ends_at, s.step), if the bar is the last thing the
    program wrote and has not reached its end: run.out's time plus the estimated total less the time so far."""
    try:
        with open(run_out, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - BAR_TAIL_BYTES))
            tail = fh.read().decode("utf-8", errors="replace")
            written = os.fstat(fh.fileno()).st_mtime
    except OSError:
        return False
    last = next((x for x in reversed(re.split(r"[\r\n]+", tail)) if x.strip()), "")
    m = _BAR.search(last)
    if m is None:
        return False
    done, total, unit = float(m.group(1)), float(m.group(2)), _UNIT_S[m.group(3)]
    if total <= done:  # the step is done: what comes next is not known yet
        return False
    s.ends_at, s.step = written + (total - done) * unit, True
    return True


def _items(jobdir: Path, s: "JobSummary") -> bool:
    """The items one of cryo-savanna's runners works through (its table: status and seconds per item): how many
    are done, and when all should be, at the pace of those done in this run, as many at once as now run."""
    for name, what in ITEM_TABLES:
        path = jobdir / name
        try:
            blocks = read_star(path)
            written = path.stat().st_mtime
        except (OSError, StarError):
            continue
        b = next((b for b in blocks.values() if b.is_loop and any(c.endswith("Status") for c in b.columns)), None)
        if b is None:
            continue
        si = next(i for i, c in enumerate(b.columns) if c.endswith("Status"))
        ti = next((i for i, c in enumerate(b.columns) if c.endswith("Seconds")), None)
        status = [r[si] for r in b.rows]
        took = [f for r, st in zip(b.rows, status) if st == "done" and ti is not None
                for f in [_float(r[ti])] if f is not None and f > 0]  # 0: done in an earlier run, skipped now
        running, pending = status.count("running"), status.count("pending")
        s.items_done, s.items_total, s.items_what = status.count("done"), len(status), what
        if took and (running or pending):
            per = sum(took) / len(took)
            s.ends_at = written + (pending + 0.5 * running) * per / max(1, running)
        return True
    return False


def progress_lines(type_label: str, program: str = "") -> int:
    """Lines a running job's card has for its progress: how far (iteration, items) and the times, or the times."""
    from cryo_savanna.jobdefs import aretomo_jobs, pytom_jobs
    counted = set(aretomo_jobs.TITLES.values()) | set(pytom_jobs.TITLES.values())
    return 2 if base_type(type_label) in ITERATIVE or program in counted else 1


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
    failure: list[Diagnosis] = field(default_factory=list)  # a failed job: what its logs say, most likely first
    # a running iterative job: how long its recent iterations took, and when its latest one was written
    seconds_per_iteration: float | None = None
    iteration_written: float | None = None
    # a running job of other kinds: the items it works through (AreTomo3's tilt series, pytom's tomograms) and
    # when it should be done: at their pace, or as RELION's progress bar of the current step estimates it
    items_done: int | None = None
    items_total: int | None = None
    items_what: str = ""
    ends_at: float | None = None
    step: bool = False  # ends_at is the end of the step RELION's progress bar is at (a program may have several)

    def lines(self) -> list[str]:
        return [" · ".join(x) for x in (self.result, self.settings, self.quality) if x]

    def text(self) -> str:
        return " · ".join(self.result + self.settings + self.quality)

    def time_left(self, now: float | None = None) -> float | None:
        """Seconds until the last iteration is written, at the pace of the recent ones (None: not known, e.g.
        Refine3D, which runs until it converges). Can be negative: an iteration slower than the ones before."""
        if (self.iteration is None or not self.total_iterations or self.seconds_per_iteration is None
                or self.iteration_written is None):
            return None if self.ends_at is None else self.ends_at - (now or time.time())
        remaining = max(0, self.total_iterations - self.iteration) * self.seconds_per_iteration
        # never more than the iterations left take (a clock that differs from the file server's)
        return min(self.iteration_written + remaining - (now or time.time()), remaining)

    def progress_count(self) -> str:
        """How far it is: "iter 12/25", "5/20 tilt series" ("" when not known)."""
        if self.iteration is not None:
            return f"iter {self.iteration}" + (f"/{self.total_iterations}" if self.total_iterations else "")
        if self.items_total:
            return f"{self.items_done or 0}/{self.items_total} {self.items_what}".rstrip()
        return ""

    def progress_time(self, now: float | None = None, elapsed: float | None = None) -> str:
        """How long it has run and has left: "1 h 05 min · ~40 min left" ("" when not known). ``elapsed``: how long
        it has run when that is known better than from its start in note.txt (its SLURM job's time used: note.txt
        has when it was sent to the queue, so the time it waited there would count too)."""
        parts = []
        if elapsed is not None:
            parts.append(duration(elapsed))
        elif self.started is not None:
            parts.append(duration((now or time.time()) - self.started))
        left = self.time_left(now)
        if left is not None:
            what = " in this step" if self.step else ""
            parts.append(f"~{duration(left)} left{what}" if left >= 60 else "finishing" + (" this step" if self.step
                                                                                            else ""))
        elif self.seconds_per_iteration is not None:  # no last iteration to count to: the pace
            per = self.seconds_per_iteration
            parts.append(f"~{duration(per) if per >= 60 else f'{per:.0f} s'}/iter")
        return " · ".join(parts)

    def progress(self, now: float | None = None, elapsed: float | None = None) -> str:
        return " · ".join(x for x in (self.progress_count(), self.progress_time(now, elapsed)) if x)


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


def _fmt_num(v: str | None) -> str:
    f = _float(v)
    return f"{f:g}" if f is not None else str(v)


class SummaryReader:
    RETRY_S = 30.0  # a summary whose files could not be read is read again after this, or when the job changes

    def __init__(self, project_dir: str | os.PathLike):
        self.project_dir = Path(project_dir)
        self._cache: dict[str, tuple[tuple | None, JobSummary]] = {}
        self._counts: dict[str, tuple[tuple, int | None]] = {}  # file -> (stamp, rows)
        self._failed: dict[str, tuple[tuple | None, float]] = {}  # job -> (its key, time.monotonic()) of a failed read

    def cached(self, name: str) -> JobSummary | None:
        hit = self._cache.get(name)
        return hit[1] if hit else None

    def needs_update(self, info: JobInfo) -> bool:
        """Finished jobs are only read again when their status changes; running ones on every change. One whose
        files could not be read, when it changes or after RETRY_S (a file written again in place changes none
        of the stamps in its key): not on every pass, which comes every poll while any job of the project runs."""
        hit = self._cache.get(info.name)
        if hit is None:
            return True
        if hit[0] is None:
            failed = self._failed.get(info.name)
            return (failed is None or time.monotonic() - failed[1] > self.RETRY_S
                    or failed[0] != self._key(info))
        if info.status != STATUS_RUNNING and hit[0][0] == info.status:
            return False
        return hit[0] != self._key(info)

    def clear(self) -> None:
        """Forget all summaries (not the row counts, which are keyed by file stamps)."""
        self._cache.clear()
        self._failed.clear()

    def summarize(self, info: JobInfo, pipeline: Pipeline) -> JobSummary:
        key: tuple | None = self._key(info)
        hit = self._cache.get(info.name)
        if hit is not None and hit[0] == key:
            return hit[1]
        try:
            s = self._summarize(info.process, pipeline, info.status == STATUS_RUNNING)
            if info.status == STATUS_FAILED:
                s.failure = diagnose(self.project_dir / info.name)
        except (OSError, ValueError, KeyError, IndexError, StarError):
            s = JobSummary()  # half-written or unexpected files: try again when it changes (needs_update)
            self._failed[info.name] = (key, time.monotonic())
            key = None
        except Exception:  # noqa: BLE001 - a bug here must not stop the summaries of every other job
            print(f"cryo-savanna: cannot summarize {info.name}:", file=sys.stderr)
            traceback.print_exc()
            s = JobSummary()  # kept until the job changes: no traceback on every pass
        else:
            self._failed.pop(info.name, None)
        self._cache[info.name] = (key, s)
        return s

    def _key(self, info: JobInfo) -> tuple:
        d = self.project_dir / info.name
        # run.out: RELION's progress bar moves on in it (only a running job is read again on a change of it)
        return (info.status, stamp(d), stamp(d / "job.star"), stamp(d / "note.txt"), stamp(d / "run.out"))

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
            if t not in ITERATIVE:
                _items(jobdir, s) or _progress_bar(jobdir / "run.out", s)

        if t in ITERATIVE:
            self._iterative(t, pr, opts, s, running)
        else:
            self._counts_from_outputs(t, pr, pipeline, s)
        if t == "relion.polish" and opts.get("do_param_optim") == "Yes":
            from cryo_savanna.results.polish import trained_params
            found = trained_params(jobdir)
            s.result += [p.text() + (f" ({p.source})" if len(found) > 1 else "") for p in found]
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
        elif last is not None:
            s.seconds_per_iteration, s.iteration_written = _pace(jobdir, models, root)

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
        def o(k: str) -> str:
            return opts.get(k) or ""
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
        elif t == "relion.polish":
            if o("do_param_optim") == "Yes":
                s.settings.append("training")
            else:
                from cryo_savanna.results.polish import used_params
                p = used_params(self.project_dir / pr.name, opts, self.project_dir)
                if p is not None:
                    s.settings.append(p.text())
                first, last = _count(o("first_frame")), _float(o("last_frame"))
                if (first or 1) > 1 or (last is not None and last > 0):
                    s.settings.append(f"frames {first or 1}–{int(last) if last and last > 0 else 'last'}")
        elif t == "relion.subtract":
            if o("do_fliplabel") == "Yes":
                s.settings.append("reverted to the original particles")
            else:
                if o("do_center_mask") == "Yes":
                    s.settings.append("recentred on the mask")
                elif o("do_center_xyz") == "Yes":
                    s.settings.append(f"recentred on ({o('center_x')}, {o('center_y')}, {o('center_z')}) px")
                if (_float(o("new_box")) or -1) > 0:
                    s.settings.append(f"box {_fmt_num(o('new_box'))} px")
        elif t == "relion.maskcreate":
            parts = [f"threshold {_fmt_num(o('inimask_threshold'))}" if o("inimask_threshold") else "",
                     f"extend {_fmt_num(o('extend_inimask'))} px" if o("extend_inimask") else "",
                     f"soft edge {_fmt_num(o('width_mask_edge'))} px" if o("width_mask_edge") else ""]
            if (_float(o("lowpass_filter")) or 0) > 0:
                parts.append(f"lowpass {_fmt_num(o('lowpass_filter'))} Å")
            s.settings += [x for x in parts if x]
        elif t == "relion.localres":
            s.settings.append("ResMap" if o("do_resmap_locres") == "Yes" and o("do_relion_locres") != "Yes" else "RELION")
        elif t == "relion.ctfrefine":
            from cryo_savanna.results.ctfrefine import fitted
            for label, how in fitted(opts):
                s.settings.append(label.lower() if how == "per optics group" else f"{label.lower()} ({how})")
        elif t == "relion.ctfrefinetomo":
            if o("do_defocus") == "Yes":
                s.settings.append("defocus" + (" (regularised)" if o("do_reg_def") == "Yes" else ""))
            if o("do_scale") == "Yes":
                ways = [w for w, k in (("per frame", "do_frame_scale"), ("per tomogram", "do_tomo_scale")) if o(k) == "Yes"]
                s.settings.append("scale" + (f" ({', '.join(ways)})" if ways else ""))
        elif t == "relion.framealigntomo":
            if o("do_shift_align") == "Yes":
                s.settings.append(f"shifts ({(o('shift_align_type') or 'entire micrographs').lower()})")
            if o("do_motion") == "Yes":
                s.settings.append(f"motion σ vel {_fmt_num(o('sigma_vel'))} · σ div {_fmt_num(o('sigma_div'))}")

    def _extract_boxes(self, t: str, pr: Process, opts: dict) -> list[str]:
        """Initial and final (binned) box size and the final pixel size of extracted particles.

        The final values come from the optics table of the output particles.star; before there
        is one (running jobs), from the job options.
        """
        def o(k: str) -> str:
            return opts.get(k) or ""
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
        if t == "relion.localres":
            from cryo_savanna.results.mapstats import locres_stats
            locres = next((n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == "Image3D"), None)
            if locres and (self.project_dir / locres).is_file():
                try:
                    opts = JobStar.read(self.project_dir / pr.name / "job.star").options
                except (OSError, ValueError, KeyError):
                    opts = {}
                mask = opts.get("fn_mask", "")
                st = locres_stats(self.project_dir / locres, self.project_dir / mask if mask else None)
                if st is not None:
                    s.quality.append(st.text())
            return
        if t == "relion.multibody":
            from cryo_savanna.results.flex import logged_variance
            var = logged_variance(self.project_dir / pr.name)  # not the projections: one line per particle
            if var is not None and len(var) and math.isfinite(var[0]):
                s.quality.append(f"component 1: {var[0]:.1f}% of the variance")
            return
        mics = next((n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == "MicrographGroupMetadata"),
                    None)
        tilts = next((n for n in pr.outputs if n in pipeline.nodes and pipeline.nodes[n].kind == "TomogramGroupMetadata"),
                     None)
        if t == "relion.motioncorr" and mics:
            v = self._column(mics, "micrographs", "rlnAccumMotionTotal")
            if v:
                s.quality.append(f"motion {statistics.fmean(v):.1f} Å (mean)")
        elif t == "relion.ctffind":
            v = []
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
