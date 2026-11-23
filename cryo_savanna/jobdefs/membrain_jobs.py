# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""MemBrain-seg as a job ("Additional external jobs"): membrane segmentation of RELION 5 tomograms, optionally
with the skeletons of the segmentations.

Like the pytom jobs (jobdefs/pytom_jobs.py), a RELION External job (relion.external, tomo): the executable is
"python3 .cryo-savanna/membrain_runner.py segment" (a copy of cryo_savanna/membrain_runner.py in the project,
with the runner_common.py whose helpers it uses, put there before such a job is added or run), every option of
the form goes to External's "Additional arguments" as ``--key value``, and the form reads them back from there
(clone, continue, overwrite). The input tomogram set also goes in an External input field, so the job hangs
under the job it reads from.

membrain-seg itself is run by the runner from its conda environment (``conda run -n <env> membrain ...``):
nothing of it is imported here. Its segmentations are shown in the job's Membranes view (the tomogram viewer
on membrain_seg.star, the raw tomogram beside them).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from cryo_savanna import membrain_runner as M
from cryo_savanna import runner_common as RC
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import runner_files
from cryo_savanna.jobdefs.model import Group, JobDefs, OptionDef, Row, Tab
from cryo_savanna.jobdefs.pytom_jobs import (
    CUDA_ENV_VAR,
    EXTERNAL,
    TOMO_NAMES,
    PytomVariant,
    _o,
    _radio,
    _warmed_cuda_module,
    _yn,
    add_running_tab,
    decode,
    encode,
)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(M.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/membrain_runner.py"
SEGMENT = M.SEGMENT
TITLES = {SEGMENT: "MemBrain-seg membrane segmentation"}
DESCRIPTIONS = {
    SEGMENT: "Membrane segmentation of RELION 5 tomograms with MemBrain-seg (membrain segment), one tomogram per "
             "GPU, with optional score maps, connected components, uncertainty maps and skeletons (membrain "
             "skeletonize)",
}
MENU_TEXT = {SEGMENT: "MemBrain-seg: segment membranes"}
MAIN_INPUTS = {SEGMENT: "in_tomograms"}
ENV_VAR = "CRYO_SAVANNA_MEMBRAIN_ENV"  # default conda environment of membrain-seg
CKPT_VAR = "CRYO_SAVANNA_MEMBRAIN_CKPT"  # default model checkpoint
SHARED = ("cuda_module", "membrain_env", "conda", "ckpt_path")  # taken from the project's last membrain-seg job


def default_env() -> str:
    return os.environ.get(ENV_VAR, "membrain")


def default_ckpt() -> str:
    return os.environ.get(CKPT_VAR, "")


def segment_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        # Input
        _o("in_tomograms", "Input tomograms:", "", "The tomograms.star of a RELION 5 Reconstruct tomograms (or "
           "Denoise, or AreTomo3) job. Each tomogram is segmented into segmentations/<rlnTomoName>_segmented.mrc "
           "in the job, of the tomogram's size and voxel size.", kind="inputnode",
           node_type="TomogramGroupMetadata.star.relion", pattern="Tomogram set STAR file (*.star)"),
        _radio("tomo_kind", "Segment:", tuple(RC.TOMOGRAM_KINDS), "Reconstructed",
               "Which tomogram of each row: the reconstruction (rlnTomoReconstructedTomogram), the denoised one "
               "or a half tomogram. MemBrain-seg was trained on tomograms that were not denoised, but denoised "
               "ones (cryoCARE) often work as well."),
        TOMO_NAMES,
        _o("voxel_size", "Tomogram voxel size (Å):", "", "Given to membrain-seg as --in-pixel-size. Empty: "
           "rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning of each tomogram, else its MRC header."),
        # Segmentation
        _o("ckpt_path", "Model checkpoint:", "", "The MemBrain-seg model (.ckpt), downloaded as its README says "
           "(the newest model is best; models older than MONAI 1.3 need MONAI 1.2, or the adapted file the "
           f"installation guide links). The default comes from ${CKPT_VAR}; a new job takes the one of the "
           "project's last membrain-seg job.", kind="filename", pattern="Model checkpoint (*.ckpt)"),
        _yn("rescale_patches", "Rescale patches?", "Yes", "Rescale each patch to the model's pixel size during "
            "the segmentation (--rescale-patches): needed for tomograms whose voxel size is far from 10 Å (more "
            "than about 12 Å, or less than 8). Nothing is rescaled for tomograms at the model's pixel size; the "
            "segmentation keeps the tomogram's size either way."),
        _o("out_pixel_size", "Pixel size of the model (Å):", "10", "--out-pixel-size: what the patches are "
           "rescaled to; MemBrain-seg's models were trained at 10 Å (leave it, unless your model was trained at "
           "another pixel size)."),
        _yn("test_time_augmentation", "Test-time augmentation?", "Yes", "8-fold test-time augmentation "
            "(--test-time-augmentation): a slightly better segmentation, in about 8 times the time."),
        _o("segmentation_threshold", "Segmentation threshold:", "0", "Voxels with a membrane score above it are "
           "segmented (--segmentation-threshold). The scores are the network's raw output (logits): 0 is a "
           "probability of 0.5; MemBrain-seg suggests trying -1.5 to 0.5. Higher: fewer, more certain membranes. "
           "With score maps (below), the job's Scores view shows the segmentation at other thresholds, and sets "
           "them per tomogram."),
        _o("tomo_thresholds", "Threshold of some tomograms:", "", "Another segmentation threshold for some "
           "tomograms, as <rlnTomoName>=<threshold> separated by spaces (e.g. TS_01=-0.5); the job's Scores view "
           "sets it. When the job is continued, the tomograms whose threshold changed are thresholded again from "
           "their score maps (membrain thresholds, no GPU segmentation), then their connected components and "
           "skeletons made again; a tomogram without its score map is segmented again."),
        _o("sliding_window_size", "Sliding window size (px):", "160", "Size of the patches the network sees "
           "(--sliding-window-size; a multiple of 32). Smaller ones need less GPU memory (160 needs about 8 GB) "
           "but segment worse."),
        _yn("store_probabilities", "Store score maps?", "No", "Also write the membrane scores before the threshold "
            "(--store-probabilities), as segmentations/<rlnTomoName>_scores.mrc (float32, as large as the "
            "tomogram): the job's Scores view shows them over the tomogram, and the segmentation at other "
            "thresholds, which a continue applies from them without the GPU. Its \"Delete score maps…\" frees "
            "the space once they are not needed."),
        _yn("store_uncertainty", "Store uncertainty maps?", "No", "Also write the variance of the test-time "
            "augmentations (--store-uncertainty-map), as segmentations/<rlnTomoName>_uncertainty.mrc. Needs the "
            "test-time augmentation."),
        _yn("store_connected_components", "Connected components?", "No", "Label the connected components of the "
            "segmentation (--store-connected-components): the segmentation then holds the label of each "
            "membrane (1, 2, ...) instead of 1."),
        _o("connected_component_thres", "Smallest component (voxels):", "", "Components with fewer voxels are left "
           "out (--connected-component-thres). Empty: all are kept."),
        _o("segment_args", "More membrain segment arguments:", "", "Added to its command line."),
        # Skeletons
        _yn("do_skeleton", "Skeletonize?", "No", "Also make the skeleton of each segmentation (membrain "
            "skeletonize): thin membrane centre lines, as segmentations/<rlnTomoName>_skeleton.mrc."),
        _radio("skel_method", "Method:", M.SKEL_METHODS, M.SKEL_METHODS[0],
               "3D-NMS: non-maximum suppression along the membrane normals, in 3D (membrain-seg's default). "
               "2D-skimage: scikit-image's skeleton of each XY section."),
        _o("skel_batch_size", "Batch size:", "", "Voxels processed at once (--batch-size). Empty: the whole "
           "volume; with little GPU memory, 1000000 is suggested."),
        _o("skel_args", "More membrain skeletonize arguments:", "", "Added to its command line."),
        # Compute
        _o("gpu_ids", "GPUs to use:", "0", "GPU ids separated by colons, as in RELION, e.g. 0:1:2:3: one tomogram "
           "per GPU at a time. Empty: every GPU the job sees (CUDA_VISIBLE_DEVICES, nvidia-smi). On a cluster, ids "
           "count from 0 among the GPUs the queue gave the job; ask for them in the submission script."),
        _o("per_gpu", "Tomograms per GPU:", "1", "membrain-seg processes on each GPU at the same time (each needs "
           "its GPU memory: about 8 GB with a 160 px window)."),
        _yn("skip_done", "Skip tomograms already done?", "Yes", "When the job runs again (continue), tomograms "
            "with all their files are not processed again: after a failure only the rest run. A tomogram "
            "segmented before that lacks only its skeleton is only skeletonized."),
        _o("cuda_module", "CUDA module:", "", "Environment module(s) loaded with 'module load' before membrain-seg "
           "runs (e.g. cuda/12.4; several separated by spaces). Empty: nothing is loaded and PyTorch uses the "
           "CUDA it came with. The default is the CUDA module loaded where cryo-savanna runs, else the default "
           f"CUDA module of 'module avail' (${CUDA_ENV_VAR} overrides it)."),
        _o("membrain_env", "membrain-seg conda environment:", "", "Name (or path, with a /) of the conda "
           "environment where membrain-seg is installed (pip install membrain-seg): its 'membrain' runs with "
           "'conda run -n <environment>'. Empty: 'membrain' is on the PATH of the job. The default comes from "
           f"${ENV_VAR}, else 'membrain'."),
        _o("conda", "conda command:", "", "The conda (or mamba) executable for 'conda run'. Empty: $CONDA_EXE where "
           "the job runs, else 'conda' on its PATH. On a cluster, the compute nodes need it too."),
    ]
    tabs = [
        Tab("Input", tuple(Row((k,)) for k in ("in_tomograms", "tomo_kind", "tomo_names", "voxel_size"))),
        Tab("Segmentation", (
            Row(("ckpt_path",)), Row(("rescale_patches",)), Row(("out_pixel_size",)),
            Row(("test_time_augmentation",)), Row(("segmentation_threshold",)), Row(("tomo_thresholds",)),
            Row(("sliding_window_size",)),
            Row(("store_probabilities",)), Row(("store_uncertainty",), group=1),
            Row(("store_connected_components",)), Row(("connected_component_thres",), group=2),
            Row(("segment_args",)))),
        Tab("Skeletons", (Row(("do_skeleton",)), Row(("skel_method",), group=3), Row(("skel_batch_size",), group=3),
                          Row(("skel_args",), group=3))),
        Tab("Compute", tuple(Row((k,)) for k in ("gpu_ids", "per_gpu", "skip_done", "cuda_module", "membrain_env",
                                                   "conda"))),
    ]
    groups = {1: Group(1, None, "test_time_augmentation"), 2: Group(2, None, "store_connected_components"),
              3: Group(3, None, "do_skeleton")}
    return opts, tabs, groups


class MembrainVariant(PytomVariant):
    """The membrain-seg form: a PytomVariant (an own form written to an External job.star) with this runner."""


def runner_command(kind: str = SEGMENT) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def membrain_kind(fn_exe: str) -> str | None:
    """segment for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in M.MODES and \
            any(Path(p.strip("'\"")).name == "membrain_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def membrain_kind_of(js: JobStar) -> str | None:
    return membrain_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def membrain_main_input(js: JobStar) -> str | None:
    """The tomogram set a membrain-seg job works on (its main input field), else None."""
    kind = membrain_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    value = own.get(MAIN_INPUTS[kind], "").strip()
    while value.startswith("./"):
        value = value[2:]
    return value or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's membrain_runner.py and the runner_common.py it imports into the project."""
    return runner_files.install_runner(project_dir, RUNNER, RUNNER_IN_PROJECT)


class MembrainCatalog:
    """The membrain-seg form, built once per RELION environment (its queue settings and RELION_QSUB_EXTRA fields)."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[str, MembrainVariant] = {}

    def variant(self, kind: str = SEGMENT) -> MembrainVariant:
        if kind not in self._cache:
            self._cache[kind] = self._build(kind)
        return self._cache[kind]

    def _build(self, kind: str) -> MembrainVariant:
        if kind not in TITLES:
            raise KeyError(f"no membrain-seg job {kind!r}")
        external = self.defs.variant(EXTERNAL, True)
        ext = external.resolved_options(self.env)
        own, tabs, groups = segment_options()
        options = {o.key: o for o in own}
        options["membrain_env"] = replace(options["membrain_env"], default=default_env())
        options["ckpt_path"] = replace(options["ckpt_path"], default=default_ckpt())
        options["cuda_module"] = replace(options["cuda_module"], default=_warmed_cuda_module())
        options["nr_threads"] = replace(ext["nr_threads"], help=(
            "CPU threads to reserve on the queue (XXXthreadsXXX in the submission script). membrain-seg runs on "
            "the GPUs; a few CPUs per GPU are enough."))
        tabs, groups = add_running_tab(options, tabs, groups, ext)
        return MembrainVariant(key=f"membrain:{kind}", label=EXTERNAL, proc_id=external.proc_id,
                               dirname=external.dirname, is_tomo=True, options=options, tabs=tabs, groups=groups,
                               kind=kind, program=TITLES[kind], description=DESCRIPTIONS[kind],
                               own_keys=tuple(o.key for o in own), external=external, env=self.env,
                               fn_exe=runner_command(kind))


def last_settings(project_dir: Path, pipeline: Pipeline | None, jobs: list[str] | None = None) -> dict[str, str]:
    """CUDA module, conda environment and command, and checkpoint of the newest membrain-seg job in the project,
    so they need typing only once. ``jobs``: the project's membrain-seg jobs, when the caller knows them (only
    those are read)."""
    if pipeline is None:
        return {}
    for name, pr in reversed(list(pipeline.processes.items())):
        if pr.type_label != EXTERNAL or (jobs is not None and name not in jobs):
            continue
        try:
            js = JobStar.read(Path(project_dir) / name / "job.star")
        except (OSError, ValueError, KeyError):
            continue
        if membrain_kind_of(js) is None:
            continue
        own, _ = decode(js.options.get("other_args", ""))
        return {k: own[k] for k in SHARED if k in own}
    return {}


# ---------------------------------------------------------------------------- thresholds per tomogram


@dataclass
class ThresholdSettings:
    """The segmentation thresholds of a membrain-seg job's tomograms: ``job`` and ``per_tomogram`` are the job's
    options (job.star: what a continue uses), ``used`` what each tomogram's segmentation was made with
    (membrain_seg.star), ``scores`` the score map of each tomogram that has one."""
    job: float
    per_tomogram: dict[str, float]
    used: dict[str, float] = field(default_factory=dict)
    scores: dict[str, str] = field(default_factory=dict)

    def wanted(self, name: str) -> float:
        return self.per_tomogram.get(name, self.job)

    def to_redo(self, names) -> list[str]:
        """The tomograms done that a continue of the job thresholds again (or segments again) for these settings."""
        return [n for n in names if n in self.used and abs(self.used[n] - self.wanted(n)) > 1e-9]


def _options(project_dir: Path, job: str) -> tuple[JobStar, dict[str, str], list[str]]:
    js = JobStar.read(Path(project_dir) / job / "job.star")
    own, other = decode(js.options.get("other_args", ""))
    return js, own, other


def _write_options(project_dir: Path, job: str, js: JobStar, own: dict[str, str], other: list[str]) -> None:
    text = encode(own, list(own))
    if other:
        text += " " + " ".join(other)
    js.options["other_args"] = text
    js.write(Path(project_dir) / job / "job.star")


def read_thresholds(project_dir: Path, job: str) -> ThresholdSettings:
    """From the job's job.star and membrain_seg.star. Raises OSError / ValueError / KeyError / M.Usage."""
    _, own, _ = _options(project_dir, job)
    job_dir = str(Path(project_dir) / job)
    out = ThresholdSettings(M.job_threshold(own), M.tomo_thresholds(own))
    out.used = M.previous_thresholds(job_dir, out.job)
    out.scores = M.score_maps(job_dir)
    return out


def save_thresholds(project_dir: Path, job: str, per_tomogram: dict[str, float]) -> None:
    """Writes the segmentation threshold of some tomograms into the job's job.star (its option tomo_thresholds),
    for the next continue of the job. Thresholds equal to the job's own are left out."""
    js, own, other = _options(project_dir, job)
    job_t = M.job_threshold(own)
    own["tomo_thresholds"] = " ".join(f"{k}={M.num(v)}" for k, v in sorted(per_tomogram.items())
                                      if v is not None and abs(float(v) - job_t) > 1e-9)
    _write_options(project_dir, job, js, own, other)


def job_finished(project_dir: Path, job: str) -> bool:
    """The job ran and ended (an exit file): its files and job.star can be changed (not while it runs)."""
    d = Path(project_dir) / job
    return any((d / f).exists() for f in (M.C.SUCCESS, M.C.FAILURE, M.C.ABORTED))


def delete_score_maps(project_dir: Path, job: str) -> int:
    """Deletes the job's score maps, switches 'Store score maps?' off in its job.star (a continue would otherwise
    segment every tomogram again to write them) and takes them out of its tables. The bytes freed."""
    js, own, other = _options(project_dir, job)
    own["store_probabilities"] = "No"
    _write_options(project_dir, job, js, own, other)
    return M.delete_scores(str(Path(project_dir) / job))
