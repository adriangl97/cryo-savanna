# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Rebuilding a pipeline from the job directories of a project.

For a project whose default_pipeline.star was lost, and for jobs that were run outside RELION's
pipeline (straight from the command line) in a project that has one. Each <Type>/jobNNN/ directory
is read on its own, using the best evidence it has, in this order:

1. ``job_pipeline.star``: the mini pipeline RELION writes in a job directory when the job is added
   (its process, input and output nodes, with their types).
2. ``job.star``: the job type and its options; the options that are input nodes (from the job
   definitions) are its inputs.
3. Command lines: ``note.txt``, ``run_submit.script``, the start of ``run.out`` and the general block
   of the last ``_optimiser.star``. Paths there that point at files in other job directories are
   inputs. Only files at the top of a job directory count (not, say, its Movies/ subfolder), so a
   log that names every micrograph does not make a job a child of the job that wrote them.
4. The directory name, for the job type when nothing else tells it.

Outputs come from job_pipeline.star, RELION_OUTPUT_NODES.star, the files RELION's jobs of that type
write (the last iteration of iterative jobs), the files other jobs take as inputs and, for jobs of
an unknown kind, the STAR, map and PDF files at the top of the directory. The status comes from
RELION's exit files; a job without one is Running when it wrote a file in the last minutes,
Scheduled when it was never started, otherwise Succeeded when it has outputs and Failed when not.

Aliases are the symlinks in the type folders that point at a job directory (RELION makes them).
Nothing here is certain the way a real pipeline is: the result is shown for review before it is
written (ui/rebuild_dialog.py).
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.cli_runner import node_label
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.ops import NODES_DIR, TRASH_DIR, OpsError, PipelineOps, decompose
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import StarError, read_star
from cryo_savanna.relion_consts import (
    GUI_PROJECTDIR,
    JOB_TYPE_BY_LABEL,
    JOB_TYPES,
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
    RELION_OUTPUT_NODES,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SCHEDULED,
    STATUS_SUCCEEDED,
    job_type_for_label,
)

PIPELINE_FILE = "default_pipeline.star"
MINI_PIPELINE = "job_pipeline.star"
RECENT_S = 15 * 60  # a job without an exit file that wrote a file this recently is still running
TYPE_DIRS = frozenset(t.dirname for t in JOB_TYPES)
SKIP_DIRS = frozenset({TRASH_DIR, NODES_DIR})
_JOB_NAME = re.compile(r"^(?P<dir>[^/]+)/job(?P<num>\d{3,})/$")
_JOB_DIR = re.compile(r"^job\d{3,}$")
_ITER = re.compile(r"_ct(\d+)|_it(\d+)")
_BOOKKEEPING = {"job.star", MINI_PIPELINE, "note.txt", "run.out", "run.err", "run_submit.script", RELION_OUTPUT_NODES,
                "default_pipeline.star", "deleted_pipeline.star"}

# What RELION's jobs of each type declare as outputs (getCommands*Job in src/pipeline_jobs.cpp, 5.1).
# (alternative file patterns, node type label, how): the first alternative that matches is used;
# "last" keeps the files of the last iteration only.
_PARTS, _MICS, _MOVIES = "ParticleGroupMetadata.star.relion", "MicrographGroupMetadata.star.relion", \
    "MicrographMovieGroupMetadata.star.relion"
_TOMOS = "TomogramGroupMetadata.star.relion.tomo"
_OPTSET = "TomoOptimisationSet.star.relion.tomo"


def _o(patterns, label: str, how: str = "one") -> tuple[tuple[str, ...], str, str]:
    return (patterns,) if isinstance(patterns, str) else tuple(patterns), label, how


OUTPUTS: dict[str, list[tuple[tuple[str, ...], str, str]]] = {
    "relion.import": [_o("movies.star", _MOVIES), _o("micrographs.star", _MICS)],
    "relion.motioncorr": [_o("corrected_micrographs.star", _MICS + ".motioncorr"),
                          _o("corrected_tilt_series.star", _TOMOS + ".motioncorr"),
                          _o("logfile.pdf", "LogFile.pdf.relion.motioncorr")],
    "relion.ctffind": [_o("micrographs_ctf.star", _MICS + ".ctf"), _o("tilt_series_ctf.star", _TOMOS + ".ctffind"),
                       _o("logfile.pdf", "LogFile.pdf.relion.ctffind"),
                       _o("power_spectra_fits.star", "Image2DGroupMetadata.star.relion.ctffind.power_spectra")],
    "relion.manualpick": [_o("manualpick.star", "MicrographCoordsGroup.star.relion.manualpick"),
                          _o("micrographs_selected.star", _MICS + ".manualpick")],
    "relion.autopick": [_o("autopick.star", "MicrographCoordsGroup.star.relion.autopick"),
                        _o("logfile.pdf", "LogFile.pdf.relion.autopick")],
    "relion.extract": [_o("particles.star", _PARTS)],
    "relion.select": [_o("particles.star", _PARTS), _o("micrographs.star", _MICS), _o("movies.star", _MOVIES),
                      _o("class_averages.star", "Image2DGroupMetadata.star.relion.classaverages"),
                      _o("particles_split*.star", _PARTS, "all"), _o("micrographs_split*.star", _MICS, "all")],
    "relion.class2d": [_o("run_it*_data.star", _PARTS + ".class2d", "last"),
                       _o("run_it*_optimiser.star", "OptimiserData.star.relion.class2d", "last")],
    "relion.initialmodel": [_o("initial_model.mrc", "DensityMap.mrc.relion.initialmodel"),
                            _o("run_it*_data.star", _PARTS + ".initialmodel", "last"),
                            _o("run_it*_optimiser.star", "OptimiserData.star.relion.initialmodel", "last")],
    "relion.class3d": [_o("run_it*_data.star", _PARTS + ".class3d", "last"),
                       _o("run_it*_optimiser.star", "OptimiserData.star.relion.class3d", "last"),
                       _o("run_it*_class*.mrc", "DensityMap.mrc.relion.class3d", "last")],
    "relion.refine3d": [_o(("run_data.star", "run_it*_data.star"), _PARTS + ".refine3d", "last"),
                        _o(("run_optimiser.star", "run_it*_optimiser.star"), "OptimiserData.star.relion.refine3d", "last"),
                        _o(("run_half1_class001_unfil.mrc", "run_it*_half1_class001.mrc"),
                           "DensityMap.mrc.relion.halfmap.refine3d", "last"),
                        _o("run_class001.mrc", "DensityMap.mrc.relion.refine3d"),
                        _o("run_optimisation_set.star", "TomoOptimisationSet.star.relion.refine3d")],
    "relion.maskcreate": [_o("mask.mrc", "Mask3D.mrc.relion")],
    "relion.joinstar": [_o("join_particles.star", _PARTS), _o("join_mics.star", _MICS),
                        _o("join_movies.star", _MOVIES)],
    "relion.subtract": [_o("particles_subtracted.star", _PARTS + ".subtracted"), _o("original.star", _PARTS)],
    "relion.postprocess": [_o("postprocess.mrc", "DensityMap.mrc.relion.postprocess"),
                           _o("postprocess_masked.mrc", "DensityMap.mrc.relion.postprocess.masked"),
                           _o("logfile.pdf", "LogFile.pdf.relion.postprocess"),
                           _o("postprocess.star", "ProcessData.star.relion.postprocess")],
    "relion.localres": [_o("histogram.pdf", "LogFile.pdf.relion.localres"),
                        _o("relion_locres_filtered.mrc", "DensityMap.mrc.relion.localresfiltered"),
                        _o("relion_locres.mrc", "Image3D.mrc.relion.localresmap")],
    "relion.multibody": [_o("run_half1_body*_unfil.mrc", "DensityMap.mrc.relion.halfmap.multibody", "all"),
                         _o("analyse_logfile.pdf", "LogFile.pdf.relion.flexanalysis")],
    "relion.polish": [_o("opt_params_all_groups.txt", "ParamsData.txt.relion.polish"),
                      _o("logfile.pdf", "LogFile.pdf.relion.polish"),
                      _o("shiny.star", _PARTS + ".polished")],
    "relion.ctfrefine": [_o("logfile.pdf", "LogFile.pdf.relion.ctfrefine"),
                         _o("particles_ctf_refine.star", _PARTS + ".ctfrefine")],
    "relion.importtomo": [_o("tilt_series.star", _TOMOS + ".import"),
                          _o("particles.star", _PARTS + ".tomo.import"),
                          _o("optimisation_set.star", _OPTSET + ".import")],
    "relion.excludetilts": [_o("selected_tilt_series.star", _TOMOS + ".excludeimages")],
    "relion.aligntiltseries": [_o("aligned_tilt_series.star", _TOMOS + ".aligntiltseries"),
                               _o("logfile.pdf", "LogFile.pdf.relion.tomo.aligntiltseries")],
    "relion.reconstructtomograms": [_o("tomograms.star", _TOMOS + ".reconstruct")],
    "relion.denoisetomo": [_o("tomograms.star", _TOMOS + ".denoise")],
    "relion.picktomo": [_o("particles.star", _PARTS + ".tomo.manualpick"),
                        _o("optimisation_set.star", _OPTSET + ".manualpick")],
    "relion.pseudosubtomo": [_o("optimisation_set.star", _OPTSET + ".extract"),
                             _o("particles.star", _PARTS + ".tomo.extract")],
    "relion.reconstructparticletomo": [_o("merged.mrc", "DensityMap.mrc.relion.tomo.map.reconstruct"),
                                       _o("optimisation_set.star", _OPTSET + ".reconstruct")],
    "relion.ctfrefinetomo": [_o("optimisation_set.star", _OPTSET + ".ctfrefine"),
                             _o("tomograms.star", _TOMOS + ".ctfrefine"),
                             _o("logfile.pdf", "LogFile.pdf.relion.tomo.ctfrefine")],
    "relion.framealigntomo": [_o("optimisation_set.star", _OPTSET + ".framealign"),
                              _o("motion.star", "TomoTrajectoryData.star.relion.tomo.framealign"),
                              _o("tomograms.star", _TOMOS + ".framealign")],
}
# job types whose files say nothing fixed: also take the summary files at the top of the directory
_SNIFF_TYPES = {"relion.external", "relion.import", "relion.importtomo", "dynamight", "modelangelo"}
_TOMO_FILES = ("tilt_series.star", "tomograms.star", "optimisation_set.star", "corrected_tilt_series.star",
               "tilt_series_ctf.star")


class RebuildError(RuntimeError):
    pass


@dataclass
class FoundJob:
    name: str  # "Class2D/job005/"
    type_label: str
    status: str
    alias: str = "None"  # "Class2D/my_alias/" or "None", as in the pipeline
    inputs: list[str] = field(default_factory=list)  # node names
    outputs: list[tuple[str, str, int]] = field(default_factory=list)  # (node name, type label, depth)
    sources: list[str] = field(default_factory=list)  # where the job was read from
    notes: list[str] = field(default_factory=list)  # what is uncertain about it
    is_tomo: bool = False

    @property
    def number(self) -> int:
        return job_number(self.name) or 0

    @property
    def type_dir(self) -> str:
        return self.name.split("/", 1)[0]

    def output_names(self) -> list[str]:
        return [n for n, _, _ in self.outputs]

    def add_output(self, name: str, label: str, depth: int = 1) -> None:
        if name not in self.output_names():
            self.outputs.append((name, label, depth))


@dataclass
class Scan:
    """Jobs found in a project, and outputs of jobs already in its pipeline that they use."""
    jobs: list[FoundJob] = field(default_factory=list)
    extra_outputs: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)  # pipeline job -> outputs
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (directory, why)
    labels: dict[str, tuple[str, int]] = field(default_factory=dict)  # node -> (type label, depth), for new nodes

    def job(self, name: str) -> FoundJob | None:
        return next((j for j in self.jobs if j.name == name), None)


def job_number(name: str) -> int | None:
    m = _JOB_NAME.match(name)
    return int(m.group("num")) if m else None


# ---------------------------------------------------------------------------- finding job directories

def find_job_dirs(project_dir: str | os.PathLike) -> list[str]:
    """Job directories (<Type>/jobNNN/, real directories with something in them) in the type folders
    RELION uses, sorted by job number. Alias symlinks, Trash/ and hidden folders are left out."""
    root = Path(project_dir)
    found = []
    try:
        tops = [e for e in os.scandir(root) if e.name in TYPE_DIRS and e.is_dir(follow_symlinks=False)]
    except OSError:
        return []
    for top in tops:
        try:
            subs = list(os.scandir(top.path))
        except OSError:
            continue
        for e in subs:
            if _JOB_DIR.match(e.name) and e.is_dir(follow_symlinks=False):
                try:
                    if any(True for _ in os.scandir(e.path)):
                        found.append(f"{top.name}/{e.name}/")
                except OSError:
                    continue
    return sorted(found, key=lambda n: (job_number(n) or 0, n))


def needs_pipeline(project_dir: str | os.PathLike) -> bool:
    """A folder without default_pipeline.star that has RELION job directories."""
    return not (Path(project_dir) / PIPELINE_FILE).is_file() and bool(find_job_dirs(project_dir))


def unregistered_job_dirs(project_dir: str | os.PathLike, pipeline: Pipeline | None = None) -> list[str]:
    """Job directories the pipeline does not have (run outside RELION's pipeline, or a lost pipeline)."""
    if pipeline is None and (Path(project_dir) / PIPELINE_FILE).is_file():
        pipeline = Pipeline.read(project_dir)
    known = set(pipeline.processes) if pipeline is not None else set()
    return [d for d in find_job_dirs(project_dir) if d not in known]


def highest_job_number(project_dir: str | os.PathLike) -> int:
    """The highest job number of any job directory, also in Trash/ (a new job must not reuse one)."""
    root = Path(project_dir)
    nums = [job_number(d) or 0 for d in find_job_dirs(root)]
    nums += [job_number(d) or 0 for d in find_job_dirs(root / TRASH_DIR)]
    return max(nums, default=0)


# ---------------------------------------------------------------------------- scanning

class _Scanner:
    def __init__(self, project_dir: Path, pipeline: Pipeline | None, defs, now: float, recent_s: float):
        self.root = project_dir
        self.real_root = os.path.realpath(project_dir)
        self.pipeline = pipeline or Pipeline()
        self.defs = defs
        self.now = now
        self.recent_s = recent_s
        self.aliases: dict[str, str] = {}  # "Refine3D/first_refine/" -> "Refine3D/job011/"
        for pr in self.pipeline.processes.values():
            if pr.has_alias:
                self.aliases[pr.alias if pr.alias.endswith("/") else pr.alias + "/"] = pr.name

    # -- helpers
    def _p(self, rel: str) -> Path:
        return self.root / rel

    def _text(self, rel: str, max_lines: int | None = None) -> str:
        try:
            with open(self._p(rel), "r", encoding="utf-8", errors="replace") as fh:
                if max_lines is None:
                    return fh.read(2_000_000)
                return "".join(line for _, line in zip(range(max_lines), fh))
        except OSError:
            return ""

    def find_aliases(self, jobs: list[str]) -> dict[str, str]:
        """Alias symlinks in the type folders: job -> "Type/alias/"."""
        wanted = set(jobs)
        out: dict[str, str] = {}
        for t in {j.split("/", 1)[0] for j in jobs}:
            try:
                entries = list(os.scandir(self._p(t)))
            except OSError:
                continue
            for e in entries:
                if not e.is_symlink() or _JOB_DIR.match(e.name):
                    continue
                target = os.path.realpath(e.path)
                rel = os.path.relpath(target, self.real_root).replace(os.sep, "/") + "/"
                if rel in wanted and rel not in out:
                    out[rel] = f"{t}/{e.name}/"
                    self.aliases[f"{t}/{e.name}/"] = rel
        return out

    def resolve(self, token: str) -> str | None:
        """A path from a file or command line -> the path relative to the project, through aliases."""
        token = token.strip().strip("'\"").rstrip(";,")
        if not token or any(c in token for c in "*?$`{}") or token.startswith("-"):
            return None
        p = Path(token) if os.path.isabs(token) else self.root / token
        rel = os.path.relpath(os.path.normpath(p), self.root).replace(os.sep, "/")
        if rel.startswith("../"):
            real = os.path.relpath(os.path.realpath(p), self.real_root).replace(os.sep, "/")
            if real.startswith("../"):
                return None
            rel = real
        parts = decompose(rel)
        if parts is None:  # perhaps through an alias: Refine3D/first_refine/run_class001.mrc
            for alias, job in self.aliases.items():
                if rel.startswith(alias):
                    return job + rel[len(alias):]
            real = os.path.relpath(os.path.realpath(p), self.real_root).replace(os.sep, "/")
            return real if decompose(real) is not None else rel
        return rel

    # -- one job
    def status(self, job: str, has_outputs: bool, started: bool) -> tuple[str, str | None]:
        d = self._p(job)
        if (d / RELION_JOB_EXIT_SUCCESS).exists():
            return STATUS_SUCCEEDED, None
        if (d / RELION_JOB_EXIT_FAILURE).exists():
            return STATUS_FAILED, None
        if (d / RELION_JOB_EXIT_ABORTED).exists():
            return STATUS_ABORTED, None
        if not started:
            return STATUS_SCHEDULED, None
        newest = 0.0
        try:
            for e in os.scandir(d):
                try:
                    newest = max(newest, e.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    pass
        except OSError:
            pass
        if self.now - newest < self.recent_s:
            return STATUS_RUNNING, ("no exit file, and it wrote files in the last minutes: taken as running. "
                                    "If it was run outside RELION's pipeline, mark it as finished once it is done.")
        if has_outputs:
            return STATUS_SUCCEEDED, "no exit file: taken as finished because its output files are there"
        return STATUS_FAILED, "no exit file and no outputs: taken as failed"

    def type_from_dir(self, job: str, text: str) -> tuple[str, bool]:
        """(type label, is tomo) from the directory name, with hints for the names SPA and tomo share."""
        dirname = job.split("/", 1)[0]
        cands = [t for t in JOB_TYPES if t.dirname == dirname]
        tomo_hint = "relion_tomo" in text or "--tomo" in text or any((self._p(job) / f).exists() for f in _TOMO_FILES)
        if not cands:
            return "relion.external", False
        if len(cands) > 1:
            pick = next((t for t in cands if (50 <= t.proc_id < 99) == tomo_hint), cands[0])
            return pick.label, tomo_hint
        return cands[0].label, tomo_hint

    def command_text(self, job: str) -> str:
        """Command lines and logs of a job: note.txt (without RELION's ++++ lines), run_submit.script,
        the start of run.out and the general block of its last _optimiser.star."""
        chunks = []
        note = self._text(job + "note.txt")
        chunks += [line for line in note.splitlines() if not line.lstrip().startswith("++++")]
        script = self._text(job + "run_submit.script", 400)
        chunks += [line for line in script.splitlines() if not line.lstrip().startswith("#")]
        chunks.append(self._text(job + "run.out", 200))
        opts = _last_iteration(self._glob(job, "run*_optimiser.star"))
        if opts:
            chunks.append(self._text(opts[-1], 80))
        return "\n".join(chunks)

    def _glob(self, job: str, pattern: str) -> list[str]:
        return sorted(glob.glob(job + pattern, root_dir=self.root))

    def outputs_by_type(self, job: str, type_label: str) -> list[tuple[str, str, int]]:
        jt = job_type_for_label(type_label)
        base = jt.label if jt else type_label
        out: list[tuple[str, str, int]] = []
        for patterns, label, how in OUTPUTS.get(base, []):
            for pat in patterns:
                hits = self._glob(job, pat)
                if hits:
                    if how == "one":
                        hits = hits[:1]
                    elif how == "last":
                        hits = _last_iteration(hits)
                    out += [(h, label, 1) for h in hits]
                    break
        return out

    @staticmethod
    def expected_outputs(job: str, type_label: str, is_tomo: bool) -> list[tuple[str, str, int]]:
        """The outputs with fixed file names of a job of this type, SPA or tomo, before it wrote them."""
        jt = job_type_for_label(type_label)
        out = []
        for patterns, label, how in OUTPUTS.get(jt.label if jt else type_label, []):
            first = patterns[0]
            if any(c in first for c in "*?["):
                continue
            tomo_only = label.startswith("Tomo") or ".tomo" in label
            if tomo_only and not is_tomo or label.startswith("Micrograph") and is_tomo:
                continue
            out.append((job + first, label, 1))
        return out

    def outputs_by_sniffing(self, job: str) -> list[tuple[str, str, int]]:
        """STAR, map and PDF files at the top of the directory (the last iteration of numbered ones)."""
        try:
            names = sorted(e.name for e in os.scandir(self._p(job)) if e.is_file())
        except OSError:
            return []
        keep = [n for n in names if n not in _BOOKKEEPING and not n.startswith(("RELION_", "."))
                and n.rsplit(".", 1)[-1].lower() in ("star", "mrc", "map", "pdf", "cif", "pdb")]
        families: dict[str, list[str]] = {}
        for n in keep:
            families.setdefault(_ITER.sub("_it#", n), []).append(n)
        files = sorted(f for fam in families.values() for f in _last_iteration(fam))
        if sum(1 for f in files if f.endswith((".mrc", ".map"))) > 12:  # e.g. one map per class: the STAR files
            files = [f for f in files if not f.endswith((".mrc", ".map"))]
        return [(job + f, node_label(str(self._p(job + f))), 1) for f in files]

    def read_job(self, job: str) -> tuple[FoundJob, list[tuple[str, str | None]]]:
        """The job without its inputs resolved: (job, [(input path, type label or None)])."""
        d = self._p(job)
        sources, notes, candidates = [], [], []
        type_label, is_tomo, mini = None, False, None
        outputs: list[tuple[str, str, int]] = []

        if (d / MINI_PIPELINE).is_file():
            try:
                mini = Pipeline.parse((d / MINI_PIPELINE).read_text(encoding="utf-8", errors="replace"))
            except (StarError, ValueError, KeyError, RuntimeError):
                notes.append(f"{MINI_PIPELINE} could not be read")
            if mini is not None and job not in mini.processes:
                notes.append(f"{MINI_PIPELINE} is about another job: not used")
                mini = None
            if mini is not None:
                pr = mini.processes[job]
                type_label = pr.type_label
                sources.append(MINI_PIPELINE)
                for n in pr.inputs:
                    node = mini.nodes[n]
                    candidates.append((n, node.type_label))
                outputs = [(n, mini.nodes[n].type_label, mini.nodes[n].type_depth) for n in pr.outputs]

        js = None
        if (d / "job.star").is_file():
            try:
                js = JobStar.read(d / "job.star")
            except (OSError, ValueError, KeyError, StarError) as e:
                notes.append(f"job.star could not be read ({e})")
        if js is not None:
            type_label = type_label or js.type_label
            is_tomo = js.is_tomo
            sources.append("job.star")
            if mini is None:
                candidates += self.jobstar_inputs(js)

        text = ""
        if mini is None and js is None:
            text = self.command_text(job)
            for tok in _tokens(text):
                candidates.append((tok, None))
            if text.strip():
                sources.append("command lines / logs")
        if type_label is None:
            type_label, is_tomo = self.type_from_dir(job, text)
            sources.append("folder name")
            notes.append("job type taken from the folder name")
        elif not is_tomo:
            is_tomo = job_type_for_label(type_label) is not None and 50 <= job_type_for_label(type_label).proc_id < 99

        # a job RELION only scheduled has a job.star but no run.out yet (nor a submit script, if queued)
        started = any((d / f).exists() for f in ("run.out", "run.err", "run_submit.script")) or (mini is None and js is None)
        # outputs: the mini pipeline's (kept when they exist), RELION_OUTPUT_NODES.star, the type's files
        declared = list(outputs)
        outputs = []
        for n, label, depth in declared + self.output_nodes_star(job) + self.outputs_by_type(job, type_label):
            if n not in [o[0] for o in outputs]:
                outputs.append((n, label, depth))
        jt = job_type_for_label(type_label)
        if (jt is None or jt.label in _SNIFF_TYPES) or not outputs:
            for o in self.outputs_by_sniffing(job):
                if o[0] not in [x[0] for x in outputs]:
                    outputs.append(o)
        existing = [o for o in outputs if self._p(o[0]).exists()]
        status, why = self.status(job, bool(existing), started)
        if why:
            notes.append(why)
        if status == STATUS_SUCCEEDED:
            outputs = existing  # RELION drops the declared outputs that were never written
        elif status in (STATUS_FAILED, STATUS_ABORTED):  # what its mini pipeline declares stays, as in RELION
            declared_names = {n for n, _, _ in declared}
            outputs = [o for o in outputs if o in existing or o[0] in declared_names]
        elif mini is None:  # what RELION declares when it adds the job (the files with fixed names)
            for n, label, depth in self.expected_outputs(job, type_label, is_tomo):
                if n not in [o[0] for o in outputs]:
                    outputs.append((n, label, depth))
        fj = FoundJob(job, type_label, status, inputs=[], outputs=outputs, sources=sources, notes=notes, is_tomo=is_tomo)
        return fj, candidates

    def jobstar_inputs(self, js: JobStar) -> list[tuple[str, str | None]]:
        """Values of the input-node options of a job.star, in the order of the job window."""
        variant = None
        if self.defs is not None:
            try:
                variant = self.defs.variant_for_jobstar(js)
            except KeyError:
                variant = None
        out = []
        if variant is not None:
            for key in variant.input_options():
                v = js.options.get(key, "").strip()
                if v:
                    out.append((v, variant.options[key].node_type or None))
            others = [k for k in js.options if k not in variant.options or variant.options[k].kind == "filename"]
        else:
            others = list(js.options)
        for key in others:  # unknown options and file names: only if they name a file of another job
            v = js.options.get(key, "").strip()
            if v and "/job" in v and not v.startswith("-"):
                out.append((v, None))
        return out

    def output_nodes_star(self, job: str) -> list[tuple[str, str, int]]:
        f = self._p(job + RELION_OUTPUT_NODES)
        if not f.is_file():
            return []
        try:
            block = read_star(f, ["output_nodes"]).get("output_nodes")
        except (OSError, StarError):
            return []
        out = []
        for rec in (block.records() if block is not None else ()):
            name = self.resolve(rec.get("rlnPipeLineNodeName", "")) or ""
            if name:
                out.append((name, rec.get("rlnPipeLineNodeTypeLabel", "") or node_label(str(self._p(name))),
                            int(rec.get("rlnPipeLineNodeTypeLabelDepth", "1") or 1)))
        return out


def scan(project_dir: str | os.PathLike, pipeline: Pipeline | None = None, jobs: list[str] | None = None,
         defs=None, now: float | None = None, recent_s: float = RECENT_S) -> Scan:
    """Read the job directories ``jobs`` (default: those ``pipeline`` does not have) into jobs for a pipeline.

    ``pipeline``: the project's pipeline, whose jobs and nodes the found jobs may take as inputs.
    ``defs``: job definitions (JobDefs), to know which job.star options are inputs.
    """
    root = Path(project_dir)
    sc = _Scanner(root, pipeline, defs, time.time() if now is None else now, recent_s)
    known = set(sc.pipeline.processes)
    if jobs is None:
        jobs = [d for d in find_job_dirs(root) if d not in known]
    result = Scan()
    jobs = [j if j.endswith("/") else j + "/" for j in jobs]
    for j in list(jobs):
        if j in known:
            result.skipped.append((j, "already in the pipeline"))
            jobs.remove(j)
    aliases = sc.find_aliases(jobs)

    read: list[tuple[FoundJob, list[tuple[str, str | None]]]] = []
    for job in sorted(jobs, key=lambda n: (job_number(n) or 0, n)):
        if job_number(job) is None:
            result.skipped.append((job, "not a <Type>/jobNNN/ directory"))
            continue
        fj, cands = sc.read_job(job)
        if job in aliases:
            alias = aliases[job]
            if any(pr.alias == alias for pr in sc.pipeline.processes.values()) or \
                    any(o.alias == alias for o, _ in read):
                fj.notes.append(f"alias {alias.rstrip('/')} is taken: left without one")
            else:
                fj.alias = alias
        read.append((fj, cands))
        result.jobs.append(fj)

    # inputs, now that every job's outputs are known
    by_name = {fj.name: fj for fj in result.jobs}
    node_owner: dict[str, str] = {n: node.producer for n, node in sc.pipeline.nodes.items() if node.producer}
    node_label_of: dict[str, tuple[str, int]] = {n: (node.type_label, node.type_depth)
                                                 for n, node in sc.pipeline.nodes.items()}
    for fj in result.jobs:
        for n, label, depth in fj.outputs:
            node_owner.setdefault(n, fj.name)
            node_label_of.setdefault(n, (label, depth))
    parents: dict[str, set[str]] = {name: set(sc.pipeline.parents(name)) for name in sc.pipeline.processes}

    def ancestors(job: str) -> set[str]:
        seen, todo = set(), [job]
        while todo:
            for p in parents.get(todo.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    todo.append(p)
        return seen

    for fj, cands in read:
        parents.setdefault(fj.name, set())
        from_text = "job.star" not in fj.sources and MINI_PIPELINE not in fj.sources
        for raw, hint in cands:
            path = sc.resolve(raw)
            if path is None or path.startswith(fj.name) or path in fj.inputs:
                continue
            parts = decompose(path)
            producer = node_owner.get(path)
            if producer is None and parts is not None:
                job = parts[0] + parts[1]
                top_level = "/" not in parts[2] and parts[2] != ""
                owner = by_name.get(job) or sc.pipeline.processes.get(job)
                waiting = owner is not None and owner.status in (STATUS_SCHEDULED, STATUS_RUNNING)
                if owner is not None and top_level and parts[2] not in _BOOKKEEPING and (sc._p(path).is_file() or waiting) \
                        and not _is_root_name(sc, path):
                    producer = job
                    label = node_label_of.get(path, (None, 1))[0] or _label_for(sc, path, hint)
                    node_label_of[path] = (label, 1)
                    node_owner[path] = job
                    if job in by_name:
                        by_name[job].add_output(path, label)
                    else:
                        result.extra_outputs.setdefault(job, []).append((path, label, 1))
            if producer is None and path not in node_label_of:
                # a file outside every job (a reference map from elsewhere): an input without a producer, as in
                # RELION; only when the job's own options name it as an input, not any path in a log
                if from_text or parts is not None or not sc._p(path).exists():
                    continue
                node_label_of[path] = (_label_for(sc, path, hint), 1)
            elif producer == fj.name or fj.name in ancestors(producer):
                fj.notes.append(f"input {path} left out: it would make a loop")
                continue
            fj.inputs.append(path)
            if producer is not None:
                parents[fj.name].add(producer)
        fj.inputs = list(dict.fromkeys(fj.inputs))
    result.labels = node_label_of
    return result


def _is_root_name(sc: _Scanner, path: str) -> bool:
    """'Refine3D/job011/run' (an output root, not a file) when it does not exist as such."""
    return not sc._p(path).exists() and "." not in path.rsplit("/", 1)[-1]


def _label_for(sc: _Scanner, path: str, hint: str | None) -> str:
    """The node type the input option (or mini pipeline) gives, else one from the file itself."""
    return hint or node_label(str(sc._p(path)))


def _last_iteration(paths: list[str]) -> list[str]:
    """The files of the highest (continuation, iteration) number; all of them when none is numbered."""
    def key(p: str) -> tuple[int, int]:
        ct = it = -1
        for m in _ITER.finditer(os.path.basename(p)):
            if m.group(1) is not None:
                ct = int(m.group(1))
            else:
                it = int(m.group(2))
        return it, ct
    if not paths:
        return []
    best = max(key(p) for p in paths)
    return [p for p in paths if key(p) == best]


_PATHISH = re.compile(r"[A-Za-z0-9_.+-]+/[^\s'\"]+")


def _tokens(text: str) -> list[str]:
    """Words of command lines and logs that look like relative or absolute paths with a folder."""
    out = []
    for line in text.splitlines():
        try:
            words = shlex.split(line, comments=False, posix=True)
        except ValueError:
            words = line.split()
        for w in words:
            if "=" in w and w.startswith("--"):
                w = w.split("=", 1)[1]
            if "/" in w and (_PATHISH.fullmatch(w.lstrip("./")) or w.startswith("/")):
                out.append(w)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------- writing

def write_pipeline(project_dir: str | os.PathLike, scan_result: Scan, jobs: list[str] | None = None,
                   statuses: dict[str, str] | None = None, new: bool = False) -> list[str]:
    """Add the found jobs (all, or ``jobs``) to the project's pipeline, under RELION's lock.

    ``new``: write a new default_pipeline.star (refused if there is one). ``statuses``: status
    chosen for a job in the review. Returns the jobs added, in pipeline order.
    """
    root = Path(project_dir)
    chosen = [j for j in scan_result.jobs if jobs is None or j.name in jobs]
    statuses = statuses or {}
    labels = scan_result.labels
    with PipelineLock(root, message="cryo-savanna: rebuild pipeline") as lock:
        path = root / PIPELINE_FILE
        if new:
            if path.exists():
                raise RebuildError(f"{path} exists already: nothing was changed.")
            p = Pipeline()
        else:
            if not path.exists():
                raise RebuildError(f"{path} does not exist: nothing was changed.")
            p = Pipeline.read(root)
        added: list[str] = []
        for proc, outs in scan_result.extra_outputs.items():
            pr = p.processes.get(proc)
            if pr is None:
                continue
            for name, label, depth in outs:
                if name not in p.nodes:
                    p.nodes[name] = Node(name, label, depth, producer=proc)
                    pr.outputs.append(name)
        for fj in sorted(chosen, key=lambda j: (j.number, j.name)):
            if fj.name in p.processes:
                continue
            alias = fj.alias
            if alias != "None" and any(o.alias == alias for o in p.processes.values()):
                alias = "None"
            pr = Process(fj.name, alias, fj.type_label, statuses.get(fj.name, fj.status))
            p.processes[fj.name] = pr
            for name, label, depth in fj.outputs:
                node = p.nodes.get(name)
                if node is None:
                    p.nodes[name] = Node(name, label, depth, producer=fj.name)
                elif node.producer is None:
                    node.producer = fj.name
                elif node.producer != fj.name:
                    continue
                pr.outputs.append(name)
            added.append(fj.name)
        for fj in chosen:
            pr = p.processes.get(fj.name)
            if pr is None or fj.name not in added:
                continue
            for name in fj.inputs:
                node = p.nodes.get(name)
                if node is None:
                    owner = decompose(name)
                    owner_job = owner[0] + owner[1] if owner else None
                    if owner_job is not None and owner_job not in p.processes:
                        continue  # a job that was not chosen
                    label, depth = labels.get(name, (node_label(str(root / name)), 1))
                    node = p.nodes[name] = Node(name, label, depth)
                if name not in pr.inputs:
                    pr.inputs.append(name)
                    node.consumers.append(fj.name)
        p.job_counter = max(p.job_counter, highest_job_number(root) + 1)
        p.write(root, lock=lock)
        ops = PipelineOps(root)
        for name in added:  # .Nodes/, as RELION keeps it (the display of node types in its GUI)
            ops._touch_node_files(p, p.processes[name])
        if new:
            try:
                (root / GUI_PROJECTDIR).touch()
            except OSError:
                pass
    return added


def adopt_folder(project_dir: str | os.PathLike, folder: str | os.PathLike, type_label: str) -> str:
    """Make a folder that is not a <Type>/jobNNN/ directory (output of a command run by hand) a job
    directory: move it to <Type>/jobNNN/ with the next free job number, and leave a symlink at its old
    place, so that paths in its files and in other jobs' files still lead there. A folder right inside
    the type folder keeps its name as the job's alias (the same symlink RELION makes for an alias).

    Takes the pipeline lock and raises the job counter past the new number, so no job added
    meanwhile gets it. Returns the new job name (add it with scan + write_pipeline)."""
    root = Path(project_dir)
    src = Path(folder)
    if not src.is_absolute():
        src = root / src
    src = Path(os.path.normpath(src))
    if src.is_symlink() or not src.is_dir():
        raise RebuildError(f"{src} is not a folder (or is a symlink).")
    rel = os.path.relpath(src, root).replace(os.sep, "/")
    if rel.startswith("../") or rel == ".":
        raise RebuildError(f"{src} is not inside the project {root}.")
    if job_number(rel + "/") is not None:
        raise RebuildError(f"{rel} is a job directory already: use Add jobs.")
    if rel.split("/", 1)[0] in SKIP_DIRS or rel.startswith("."):
        raise RebuildError(f"{rel} cannot be added as a job.")
    jt = JOB_TYPE_BY_LABEL.get(type_label)
    if jt is None:
        raise RebuildError(f"unknown job type {type_label}")
    pipe_path = root / PIPELINE_FILE
    with PipelineLock(root, message="cryo-savanna: add folder as job") as lock:
        p = Pipeline.read(root) if pipe_path.exists() else None
        number = max(p.job_counter if p else 1, highest_job_number(root) + 1)
        name = f"{jt.dirname}/job{number:03d}/"
        dest = root / name.rstrip("/")
        if os.path.lexists(dest):
            raise RebuildError(f"{name} exists already.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        try:  # in the type folder, the same link RELION makes for an alias
            os.symlink(f"../{name.rstrip('/')}" if src.parent == dest.parent else os.path.relpath(dest, src.parent),
                       src)
        except OSError:
            pass
        if src.parent != dest.parent:  # also an alias in the type folder, when the name can be one
            try:
                alias = PipelineOps.check_alias(p or Pipeline(), name, src.name)
                link = root / jt.dirname / alias.rstrip("/")
                if alias != "None" and not os.path.lexists(link):
                    os.symlink(f"../{name.rstrip('/')}", link)
            except OpsError:
                pass
        if p is not None:
            p.job_counter = number + 1
            p.write(root, lock=lock)
    return name
