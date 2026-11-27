# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""CryoSPARC particles to RELION with pyem's csparc2star.py, as a job ("Additional external jobs").

Like the particle splits (jobdefs/split_jobs.py), a RELION External job (relion.external, SPA): the
executable is "python3 .cryo-savanna/csparc_runner.py csparc2star" (a copy of cryo_savanna/csparc_runner.py in
the project, with runner_common.py), the form's options go to External's "Additional arguments" as
``--key value``. The RELION micrographs (re-extraction) also go in an External input field, so the job hangs
under the job that made them; without them the job has no parent in the project (its input is a CryoSPARC
file).

pyem is run by the runner from its conda environment (``conda run -n <env> csparc2star.py ...``): nothing of
it is imported here. The job's particles.star gets the usual views (Particles, and Picks on the RELION
micrographs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from cryo_savanna import csparc_runner as CS
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import runner_files
from cryo_savanna.jobdefs.model import Group, JobDefs, OptionDef, Row, Tab
from cryo_savanna.jobdefs.pytom_jobs import EXTERNAL, PytomVariant, _o, _radio, _yn, add_running_tab, decode
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(CS.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/csparc_runner.py"
CSPARC2STAR = CS.CSPARC2STAR
TITLES = {CSPARC2STAR: "CryoSPARC particles to RELION (csparc2star)"}
DESCRIPTIONS = {
    CSPARC2STAR: "CryoSPARC particles (.cs) to a RELION particles.star with pyem's csparc2star.py: the "
                 "CryoSPARC stacks linked into the job as .mrcs (nothing copied or changed), or set up for "
                 "re-extraction from the project's RELION micrographs",
}
MENU_TEXT = {CSPARC2STAR: "CryoSPARC particles to RELION (pyem csparc2star)"}
MAIN_INPUTS = {CSPARC2STAR: "in_micrographs"}  # (the .cs file is no pipeline node)
ENV_VAR = "CRYO_SAVANNA_PYEM_ENV"  # default conda environment of pyem
SHARED = ("pyem_env", "conda", "cs_project")  # taken from the project's last csparc2star job
MICROGRAPHS_NODE = "MicrographGroupMetadata.star.relion"


def default_env() -> str:
    return os.environ.get(ENV_VAR, "pyem")


def csparc_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        # Input
        _o("in_cs", "CryoSPARC particles (.cs):", "", "The particles of a CryoSPARC job: an output's .cs file "
           "(e.g. J45/J45_008_particles.cs of a refinement, J31/J31_particles_selected.cs of a Select 2D), or the "
           ".cs of an exported particle set (exports/groups/...). The particle stacks are found from the paths in "
           "it, whichever job wrote them (Extract, Downsample, Restack, ...).", kind="filename",
           pattern="CryoSPARC particles (*.cs)"),
        _o("passthrough_cs", "Passthrough .cs files:", "", "More .cs files with the rest of the particles' "
           "fields (their stacks, micrographs, CTF), separated by spaces: a job's output holds only what the job "
           "changed, its J<n>_passthrough_<output>.cs the rest. Empty: the passthrough file beside the particles "
           "(below)."),
        _yn("auto_passthrough", "Add the job's passthrough file?", "Yes", "With no passthrough files given, use "
            "the one CryoSPARC wrote beside the particles (J45_passthrough_particles.cs for J45_008_particles.cs), "
            "when it is there. An exported particle set has everything in its own .cs file."),
        _o("cs_project", "CryoSPARC project folder:", "", "The folder of the CryoSPARC project (CS-<name>/, with "
           "J1/, J2/, ...): the .cs paths are relative to it. Empty: the folders above the particle .cs file are "
           "searched (right when the .cs file is in the project, or in its exports/).", kind="folder"),
        # RELION
        _radio("mode", "Particles for RELION:", (CS.DIRECT, CS.REEXTRACT), CS.DIRECT,
               "Use the stacks directly: each CryoSPARC stack is linked into the job as Stacks/<its CryoSPARC "
               "path>.mrcs (a link, absolute: the CryoSPARC files are not copied or changed), and rlnImageName "
               "points at the links; the particles.star is ready for 2D/3D classification and refinement. "
               "Re-extract in RELION: the particles are put on the RELION micrographs below (their names, their "
               "coordinates in those micrographs' pixels, their optics groups), for RELION's Extract with "
               "'Re-extract refined particles?' Yes and this particles.star (run.out says the settings, and "
               "the box size in the micrographs' pixels)."),
        _o("in_micrographs", "RELION micrographs:", "", "A micrographs STAR file of this project (Motion "
           "correction's corrected_micrographs.star, CTF estimation's micrographs_ctf.star) with the same "
           "micrographs as CryoSPARC's (made from the same movies). They are matched by name: CryoSPARC's without "
           "its UID, extension and motion-correction suffix (_patch_aligned_doseweighted, ...). The coordinates "
           "are scaled to each micrograph's size (binning, CryoSPARC's Fourier crop, downsampled micrographs). "
           "Needed to re-extract; to use the stacks directly, optional: rlnMicrographName and the coordinates "
           "then point at the RELION micrographs too (the Picks view, CTF refinement).", kind="inputnode",
           node_type=MICROGRAPHS_NODE, pattern="Micrographs STAR file (*.star)"),
        _yn("keep_ctf", "Keep CryoSPARC's CTF?", "Yes", "Yes: the particles keep CryoSPARC's defocus (per "
            "particle, after its CTF refinement), beam tilt and higher-order aberrations, which Extract keeps. "
            "No: they are left out, and Extract takes the CTF of the micrographs (CtfFind's)."),
        _radio("flip_y", "Y axis of the coordinates:", CS.FLIP_CHOICES, CS.FLIP_DEFAULT,
               "pyem's default: the coordinates in RELION's convention, for RELION micrographs the same way up "
               "as CryoSPARC's (re-extracted, the particles are then CryoSPARC's own images: poses and CTF stay "
               "as they are). --inverty: CryoSPARC's own values, to import the particles into CryoSPARC again. "
               "Upside-down micrographs: the RELION micrographs are upside-down against CryoSPARC's (e.g. "
               "CryoSPARC's Patch Motion against RELION's motion correction or MotionCor3): --inverty and, to "
               "re-extract, --flipy (the poses and the defocus angle mirrored too); the stacks used directly "
               "keep their poses. Check the picks in the job's Picks view (pyem's wiki; asarnow/pyem#133)."),
        _yn("strip_uid", "Strip CryoSPARC's UIDs?", "No", "Name the links (and, without RELION micrographs, the "
            "micrographs) without CryoSPARC's 21-digit UID prefix. Two stacks that only the UID tells apart "
            "keep it."),
        # csparc2star.py
        _o("classes", "Only these classes:", "", "csparc2star.py's --class: the class numbers to keep (from 1, "
           "separated by spaces), e.g. of a 3D classification. Empty: all."),
        _o("minphic", "Minimum class posterior:", "", "--minphic: leave out particles assigned to their class "
           "with a lower probability (0 to 1). Empty: none left out."),
        _o("boxsize", "Refinement box size (px):", "", "--boxsize: the box of the CryoSPARC refinement, when it "
           "is not the particles' (the shifts are scaled). Empty: the same."),
        _yn("noswapxy", "Do not swap X and Y?", "No", "--noswapxy: CryoSPARC's micrograph shape is (Y, X); "
            "csparc2star.py swaps it for RELION by default (rarely needs changing)."),
        _yn("invertx", "Invert X?", "No", "--invertx: mirror the coordinates in X (rare: K3 images that are "
            "'tall' in SerialEM)."),
        _o("transform", "Transform:", "", "--transform: a rotation matrix, or 3x4 rotation and translation, in "
           "numpy's format (e.g. [[1,0,0],[0,1,0],[0,0,1]]), applied to the particles' poses."),
        _o("copy_coords", "Copy micrograph coordinates from:", "", "--copy-micrograph-coordinates: a STAR file "
           "(or a quoted glob) whose micrograph names and coordinates the particles take, matched by image or "
           "micrograph. Not with RELION micrographs above."),
        _yn("cached", "Keep cache paths?", "No", "--cached: keep the paths of CryoSPARC's cache when copying the "
            "coordinates."),
        _o("micrograph_path", "Micrograph folder:", "", "--micrograph-path: replaces the folder of rlnMicrographName "
           "(only to use the stacks directly, without RELION micrographs)."),
        _yn("first10k", "Only the first 10,000 particles?", "No", "--10k: a quick test."),
        _yn("allow_pickle", "Allow pickle?", "No", "--allow-pickle: needed for some large .cs files. It lets the "
            "file run code: only for your own CryoSPARC files."),
        _o("max_header_size", "Largest .cs header:", "", "--max-header-size for large .cs files. Empty: "
           "csparc2star.py's (100000)."),
        _o("more_args", "More csparc2star.py arguments:", "", "Added to its command line."),
        _o("pyem_env", "pyem conda environment:", "", "Name (or path, with a /) of the conda environment where "
           "pyem is installed: csparc2star.py (and numpy) run with 'conda run -n <environment>'. Empty: "
           f"csparc2star.py is on the PATH of the job. The default comes from ${ENV_VAR}, else 'pyem'."),
        _o("conda", "conda command:", "", "The conda (or mamba) executable for 'conda run'. Empty: $CONDA_EXE where "
           "the job runs, else 'conda' on its PATH."),
    ]
    tabs = [
        Tab("Input", tuple(Row((k,)) for k in ("in_cs", "passthrough_cs", "auto_passthrough", "cs_project"))),
        Tab("RELION", (Row(("mode",)), Row(("in_micrographs",)), Row(("keep_ctf",), group=1), Row(("flip_y",)),
                       Row(("strip_uid",), group=2))),
        Tab("csparc2star", tuple(Row((k,)) for k in (
            "classes", "minphic", "boxsize", "noswapxy", "invertx", "transform", "copy_coords", "cached",
            "micrograph_path", "first10k", "allow_pickle", "max_header_size", "more_args", "pyem_env", "conda"))),
    ]
    groups = {1: Group(1, None, "mode", CS.REEXTRACT), 2: Group(2, None, "mode", CS.DIRECT)}
    return opts, tabs, groups


@dataclass
class CsparcVariant(PytomVariant):
    """The form: a PytomVariant (an own form written to an External job.star) with this runner, for SPA (the
    job.star's is-tomo flag follows the form, not always tomo)."""

    def make_jobstar(self, values: dict[str, str], is_continue: bool, job_dir: str,
                     pipeline: Pipeline | None) -> JobStar:
        js = super().make_jobstar(values, is_continue, job_dir, pipeline)
        js.is_tomo = self.is_tomo
        return js


def runner_command(kind: str = CSPARC2STAR) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def csparc_kind(fn_exe: str) -> str | None:
    """csparc2star for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in CS.MODES and \
            any(Path(p.strip("'\"")).name == "csparc_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def csparc_kind_of(js: JobStar) -> str | None:
    return csparc_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def csparc_main_input(js: JobStar) -> str | None:
    """The RELION micrographs of a csparc2star job (its main input field), else None."""
    kind = csparc_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    value = own.get(MAIN_INPUTS[kind], "").strip()
    while value.startswith("./"):
        value = value[2:]
    return value or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's csparc_runner.py and the runner_common.py it imports into the project."""
    return runner_files.install_runner(project_dir, RUNNER, RUNNER_IN_PROJECT)


class CsparcCatalog:
    """The form, built once per RELION environment (its queue settings and RELION_QSUB_EXTRA fields)."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[str, CsparcVariant] = {}

    def variant(self, kind: str = CSPARC2STAR) -> CsparcVariant:
        if kind not in self._cache:
            self._cache[kind] = self._build(kind)
        return self._cache[kind]

    def _build(self, kind: str) -> CsparcVariant:
        if kind not in TITLES:
            raise KeyError(f"no csparc2star job {kind!r}")
        external = self.defs.variant(EXTERNAL, False)
        ext = external.resolved_options(self.env)
        own, tabs, groups = csparc_options()
        options = {o.key: o for o in own}
        options["pyem_env"] = replace(options["pyem_env"], default=default_env())
        options["nr_threads"] = replace(ext["nr_threads"], help=(
            "CPU threads to reserve on the queue (XXXthreadsXXX). csparc2star.py uses one: the job reads and "
            "writes metadata and makes links, quick enough to run locally."))
        tabs, groups = add_running_tab(options, tabs, groups, ext)
        return CsparcVariant(key=f"csparc:{kind}", label=EXTERNAL, proc_id=external.proc_id,
                             dirname=external.dirname, is_tomo=False, options=options, tabs=tabs, groups=groups,
                             kind=kind, program=TITLES[kind], description=DESCRIPTIONS[kind],
                             own_keys=tuple(o.key for o in own), external=external, env=self.env,
                             fn_exe=runner_command(kind))


def last_settings(project_dir: Path, pipeline: Pipeline | None, jobs: list[str] | None = None) -> dict[str, str]:
    """pyem's conda environment and command, and the CryoSPARC project folder, of the newest csparc2star job in
    the project, so they need typing only once. ``jobs``: the project's csparc2star jobs, when the caller knows
    them (only those are read)."""
    if pipeline is None:
        return {}
    for name, pr in reversed(list(pipeline.processes.items())):
        if pr.type_label != EXTERNAL or (jobs is not None and name not in jobs):
            continue
        try:
            js = JobStar.read(Path(project_dir) / name / "job.star")
        except (OSError, ValueError, KeyError):
            continue
        if csparc_kind_of(js) is None:
            continue
        own, _ = decode(js.options.get("other_args", ""))
        return {k: own[k] for k in SHARED if k in own}
    return {}


def newest_micrographs(pipeline: Pipeline) -> dict[str, str]:
    """The newest micrographs STAR file of the project (with CTF, else any), to start the form with."""
    nodes = [n for n, node in pipeline.nodes.items() if node.kind == "MicrographGroupMetadata" and node.producer]
    with_ctf = [n for n in nodes if n.endswith("micrographs_ctf.star")]
    pick = (with_ctf or nodes)[-1:] if nodes else []
    return {"in_micrographs": pick[0]} if pick else {}
