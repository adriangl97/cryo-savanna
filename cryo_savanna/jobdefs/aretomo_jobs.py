# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""AreTomo3 as a job ("Additional external jobs"): tilt-series alignment and tomogram reconstruction.

Like the pytom jobs (jobdefs/pytom_jobs.py), a RELION External job (relion.external, tomo): the executable
is "python3 .cryo-savanna/aretomo_runner.py align" (a copy of cryo_savanna/aretomo_runner.py in the project,
with the runner_common.py whose helpers it uses, put there before such a job is added or run), every option
of the form goes to External's "Additional arguments" as ``--key value``, and the form reads them back
from there (clone, continue, overwrite). The input tilt-series set also goes in an External input field,
so the job hangs under the job it reads from.

AreTomo3 itself is a program the runner starts (its executable, after the job's environment modules).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from cryo_savanna import aretomo_runner as A
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import runner_files
from cryo_savanna.jobdefs.model import Group, JobDefs, OptionDef, Row, Tab
from cryo_savanna.jobdefs.pytom_jobs import (
    CUDA_ENV_VAR,
    EXTERNAL,
    PytomVariant,
    _o,
    _radio,
    _warmed_cuda_module,
    _yn,
    add_running_tab,
    decode,
)
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(A.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/aretomo_runner.py"
ALIGN = A.ALIGN
TITLES = {ALIGN: "AreTomo3 alignment and reconstruction"}
DESCRIPTIONS = {
    ALIGN: "Tilt-series alignment and tomogram reconstruction of RELION 5 tilt series with AreTomo3 (CTF "
           "estimation, global and local alignment, CTF correction, SART or WBP), on one or more GPUs; the "
           "alignment goes back into RELION's tilt-series STAR files",
}
MENU_TEXT = {ALIGN: "AreTomo3: align and reconstruct tilt series"}
EXE_ENV_VAR = "CRYO_SAVANNA_ARETOMO3"  # default AreTomo3 executable
MAIN_INPUTS = {ALIGN: "in_tiltseries"}
SHARED = ("cuda_module", "aretomo_exe")  # taken from the project's last AreTomo3 job


def default_exe() -> str:
    return os.environ.get(EXE_ENV_VAR, "AreTomo3")


def align_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        # Input
        _o("in_tiltseries", "Input tilt series:", "", "A RELION 5 tilt-series set: the tilt_series.star of "
           "MotionCorr, CtfFind or Exclude tilt images (motion-corrected images, rlnMicrographName), or of an "
           "alignment job to reconstruct with its alignment. The images of each tilt series are stacked, "
           "sorted by tilt angle, with their tilt angles, acquisition order and dose "
           "(rlnMicrographPreExposure), for AreTomo3.", kind="inputnode",
           node_type="TomogramGroupMetadata.star.relion", pattern="Tilt-series set STAR file (*.star)"),
        _o("tomo_names", "Only these tilt series:", "", "rlnTomoName of the tilt series to process, separated by "
           "spaces. Empty: all of them."),
        _o("angpix", "Tilt-series pixel size (Å):", "", "Pixel size of the motion-corrected images. Empty: "
           "rlnTomoTiltSeriesPixelSize, else the MRC header of the images."),
        _yn("do_halves", "Odd/even tomograms?", "No", "Also reconstruct tomograms from the odd and even frames "
            "(rlnMicrographNameEven/Odd: MotionCorr's 'Save images for denoising?'), as "
            "rlnTomoReconstructedTomogramHalf1/2 for denoising (-SplitSum 1)."),
        _yn("keep_stacks", "Keep the tilt-series stacks?", "No", "The stacks AreTomo3 reads "
            "(inputs/<name>/<name>.mrc: every image of a tilt series, as float32) are deleted once the tilt series "
            "is done. They are made again when needed."),
        # Alignment
        _radio("alignment", "Alignment:", (A.ALIGN_ARETOMO, A.ALIGN_INPUT), A.ALIGN_ARETOMO,
               "Align with AreTomo3 (-Cmd 1: CTF estimation, alignment, reconstruction), writing its alignment "
               "into aligned_tilt_series.star as RELION's own AreTomo wrapper does (rlnTomoYTilt, rlnTomoZRot, "
               "rlnTomoX/YShiftAngst), so that RELION's Reconstruct tomograms can use it. Or only reconstruct "
               "(-Cmd 2) with the alignment of the input (an alignment job's tilt series, e.g. an earlier "
               "AreTomo3 job, whose local alignment is kept, or RELION's IMOD alignment) and its CTF values."),
        _radio("tilt_axis_mode", "Tilt axis:", (A.AXIS_REFINE, A.AXIS_FIXED, A.AXIS_SEARCH), A.AXIS_REFINE,
               "Start from the nominal tilt axis (rlnTomoNominalTiltAxisAngle, or the angle below) and refine "
               "it (-TiltAxis a), keep it (-TiltAxis a -1), or let AreTomo3 search all angles."),
        _o("tilt_axis", "Tilt axis angle (°):", "", "Empty: rlnTomoNominalTiltAxisAngle of the tilt series."),
        _o("align_z", "Alignment thickness (px):", "", "Height of the volume used for the alignment, in pixels "
           "of the tilt series (-AlignZ). Empty: the sample thickness AreTomo3 measures."),
        _o("at_patch", "Local alignment patches (x y):", "", "Patches for local alignment (-AtPatch), e.g. 4 4. "
           "Empty: global alignment only."),
        _radio("tilt_cor", "Tilt angle offset:", tuple(A.TILTCOR), next(iter(A.TILTCOR)),
               "-TiltCor: measure the offset of the tilt angles and use it in the alignment only (0, AreTomo3's "
               "default), also to level the tomogram (1), or neither (-1)."),
        _o("tilt_offset", "Tilt angle offset (°):", "", "Use this offset instead of the measured one (the "
           "second value of -TiltCor). Empty: measured."),
        _o("dark_tol", "Dark image tolerance:", "0.7", "-DarkTol, 0 to 1: higher is stricter. Tilts AreTomo3 "
           "finds dark are left out of the aligned tilt series."),
        _radio("ctf_to_star", "AreTomo3's CTF in the tilt series:", (A.CTF_IF_NONE, A.CTF_ALWAYS, A.CTF_NEVER),
               A.CTF_IF_NONE, "AreTomo3 estimates the CTF of every tilt. Write its defocus (rlnDefocusU/V/Angle, "
               "rlnCtfMaxResolution, rlnCtfFigureOfMerit) into the aligned tilt series when the input has none "
               "(before CtfFind), always, or never."),
        _o("ext_phase", "Phase shift and search range (°):", "", "For a phase plate (-ExtPhase), e.g. 90 30. "
           "Empty: none."),
        # Reconstruction
        _o("at_bin", "Binning:", "4", "Binning of the tomogram relative to the tilt series (-AtBin). Up to three "
           "numbers: the second and third give further tomograms (_2ND_Vol: WBP, _3RD_Vol: SART) in "
           "aretomo/<name>/; tomograms.star lists the first."),
        _o("vol_z", "Tomogram thickness (px):", "", "In pixels of the tilt series (-VolZ, before binning). Empty: "
           "the measured sample thickness plus the extra thickness below (-VolZ -1). Needed to reconstruct "
           "with an alignment that is not AreTomo3's."),
        _o("ext_z", "Extra thickness (px):", "300", "Added to the measured thickness (-ExtZ), in pixels of the "
           "tilt series."),
        _radio("recon_method", "Reconstruction:", (A.RECON_SART, A.RECON_WBP), A.RECON_SART,
               "SART (AreTomo3's default) or weighted back-projection (-Wbp 1; often preferred for template "
               "matching and subtomogram averaging)."),
        _o("sart", "SART iterations, projections per subset:", "", "-Sart. Empty: AreTomo3's (20 3)."),
        _radio("corr_ctf", "CTF correction:", tuple(A.CORR_CTF), next(iter(A.CORR_CTF)),
               "-CorrCTF, with AreTomo3's CTF (or the input's, when only reconstructing)."),
        _o("recon_range", "Tilt range (°):", "", "Only tilts in this range are reconstructed (-ReconRange), e.g. "
           "-50 50. Empty: all."),
        _radio("flip_vol", "Tomogram layout:", tuple(A.FLIP_VOL), next(iter(A.FLIP_VOL)),
               "Both write the tomograms with z as the slowest axis, as RELION's are. -FlipVol 1 (what "
               "AreTomo3's examples use) flips AreTomo3's xzy volume; -FlipVol 2 rotates it instead, keeping "
               "its hand. Check the hand against a RELION reconstruction of the aligned tilt series before using "
               "picks from these tomograms for subtomogram averaging."),
        _yn("flip_int", "Invert contrast?", "No", "Make the density white (-FlipInt 1)."),
        _o("aretomo_args", "More AreTomo3 arguments:", "", "Added to its command line, e.g. -OutImod 1."),
        # Compute
        _o("gpu_ids", "GPUs to use:", "0", "GPU ids separated by colons, as in RELION, e.g. 0:1:2:3: one tilt series "
           "per GPU at a time. Empty: every GPU the job sees (CUDA_VISIBLE_DEVICES, nvidia-smi). On a cluster, ids "
           "count from 0 among the GPUs the queue gave the job; ask for them in the submission script."),
        _o("per_gpu", "Tilt series per GPU:", "1", "AreTomo3 processes run on each GPU at the same time (more "
           "than one can keep a GPU busy while the other reads or writes files; needs the GPU memory)."),
        _yn("skip_done", "Skip tilt series already done?", "Yes", "When the job runs again (continue), tilt "
            "series with an alignment and a tomogram are not processed again: after a failure only the rest run."),
        _o("aretomo_exe", "AreTomo3 executable:", "", f"The AreTomo3 program (a name on the job's PATH, or a "
           f"path). The default comes from ${EXE_ENV_VAR}, else 'AreTomo3'."),
        _o("cuda_module", "Modules:", "", "Environment module(s) loaded with 'module load' before AreTomo3 runs "
           "(e.g. cuda/12.4, or AreTomo3's own module; several separated by spaces). Empty: nothing is loaded. "
           "The default is the CUDA module loaded where cryo-savanna runs, else the default CUDA module of "
           f"'module avail' (${CUDA_ENV_VAR} overrides it)."),
    ]
    tabs = [
        Tab("Input", tuple(Row((k,)) for k in ("in_tiltseries", "tomo_names", "angpix", "do_halves", "keep_stacks"))),
        Tab("Alignment", (
            Row(("alignment",)), Row(("tilt_axis_mode",), group=1), Row(("tilt_axis",), group=1),
            Row(("align_z",), group=1), Row(("at_patch",), group=1), Row(("tilt_cor",), group=1),
            Row(("tilt_offset",), group=1), Row(("dark_tol",), group=1), Row(("ctf_to_star",), group=1),
            Row(("ext_phase",), group=1))),
        Tab("Reconstruction", (
            Row(("at_bin",)), Row(("vol_z",)), Row(("ext_z",)), Row(("recon_method",)), Row(("sart",), group=2),
            Row(("corr_ctf",)), Row(("recon_range",)), Row(("flip_vol",)), Row(("flip_int",)),
            Row(("aretomo_args",)))),
        Tab("Compute", tuple(Row((k,)) for k in ("gpu_ids", "per_gpu", "skip_done", "aretomo_exe", "cuda_module"))),
    ]
    groups = {1: Group(1, None, "alignment", A.ALIGN_ARETOMO), 2: Group(2, None, "recon_method", A.RECON_SART)}
    return opts, tabs, groups


class AretomoVariant(PytomVariant):
    """The AreTomo3 form: a PytomVariant (an own form written to an External job.star) with this runner."""


def runner_command(kind: str) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def aretomo_kind(fn_exe: str) -> str | None:
    """align for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in A.MODES and \
            any(Path(p.strip("'\"")).name == "aretomo_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def aretomo_kind_of(js: JobStar) -> str | None:
    return aretomo_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def aretomo_main_input(js: JobStar) -> str | None:
    """The tilt-series set an AreTomo3 job works on (its main input field), else None."""
    kind = aretomo_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    value = own.get(MAIN_INPUTS[kind], "").strip()
    while value.startswith("./"):
        value = value[2:]
    return value or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's aretomo_runner.py and the runner_common.py it imports into the project."""
    return runner_files.install_runner(project_dir, RUNNER, RUNNER_IN_PROJECT)


class AretomoCatalog:
    """The AreTomo3 form, built once per RELION environment (its queue settings and RELION_QSUB_EXTRA fields)."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[str, AretomoVariant] = {}

    def variant(self, kind: str) -> AretomoVariant:
        if kind not in self._cache:
            self._cache[kind] = self._build(kind)
        return self._cache[kind]

    def _build(self, kind: str) -> AretomoVariant:
        external = self.defs.variant(EXTERNAL, True)
        ext = external.resolved_options(self.env)
        own, tabs, groups = align_options()
        options = {o.key: o for o in own}
        options["aretomo_exe"] = replace(options["aretomo_exe"], default=default_exe())
        options["cuda_module"] = replace(options["cuda_module"], default=_warmed_cuda_module())
        options["nr_threads"] = replace(ext["nr_threads"], help=(
            "CPU threads to reserve on the queue (XXXthreadsXXX in the submission script). AreTomo3 runs on "
            "the GPUs; making the stacks uses one CPU per tilt series being processed."))
        tabs, groups = add_running_tab(options, tabs, groups, ext)
        return AretomoVariant(key=f"aretomo:{kind}", label=EXTERNAL, proc_id=external.proc_id,
                              dirname=external.dirname, is_tomo=True, options=options, tabs=tabs, groups=groups,
                              kind=kind, program=TITLES[kind], description=DESCRIPTIONS[kind],
                              own_keys=tuple(o.key for o in own), external=external, env=self.env,
                              fn_exe=runner_command(kind))


def last_settings(project_dir: Path, pipeline: Pipeline | None, jobs: list[str] | None = None) -> dict[str, str]:
    """Modules and AreTomo3 executable of the newest AreTomo3 job in the project, so they need typing only
    once. ``jobs``: the project's AreTomo3 jobs, when the caller knows them (only those are read)."""
    if pipeline is None:
        return {}
    for name, pr in reversed(list(pipeline.processes.items())):
        if pr.type_label != EXTERNAL or (jobs is not None and name not in jobs):
            continue
        try:
            js = JobStar.read(Path(project_dir) / name / "job.star")
        except (OSError, ValueError, KeyError):
            continue
        if aretomo_kind_of(js) is None:
            continue
        own, _ = decode(js.options.get("other_args", ""))
        return {k: own[k] for k in SHARED if k in own}
    return {}


def newest_input(pipeline: Pipeline) -> dict[str, str]:
    """The newest tilt-series set (not a tomogram set) of the pipeline, as the form's input."""
    for name in reversed(list(pipeline.nodes)):
        n = pipeline.nodes[name]
        if n.producer and n.kind == "TomogramGroupMetadata" and not name.endswith("tomograms.star"):
            return {"in_tiltseries": name}
    return {}
