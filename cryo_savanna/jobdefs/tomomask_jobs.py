# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomogram masks as a job ("Additional external jobs"): a mask per tomogram for pytom's --tomogram-mask.

Like the AreTomo3 job (jobdefs/aretomo_jobs.py), a RELION External job (relion.external, tomo): the executable
is "python3 .cryo-savanna/mask_runner.py masks" (a copy of cryo_savanna/mask_runner.py in the project, with the
pytom_runner.py whose helpers it uses), every option of the form goes to External's "Additional arguments" as
``--key value``, and the form reads them back from there. The job writes starting masks
(masks/<rlnTomoName>_mask.mrc); they are drawn in the job's Masks view (viewers/tomo_mask_view.py).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from cryo_savanna import mask_runner as M
from cryo_savanna import pytom_runner as PR
from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.aretomo_jobs import _install
from cryo_savanna.jobdefs.model import Group, JobDefs, OptionDef, Row, Tab
from cryo_savanna.jobdefs.pytom_jobs import EXTERNAL, TOMO_NAMES, PytomVariant, _o, _radio, add_running_tab, decode
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.project.pipeline import Pipeline

RUNNER = str(Path(M.__file__).resolve())
HELPERS = str(Path(PR.__file__).resolve())
RUNNER_IN_PROJECT = ".cryo-savanna/mask_runner.py"
HELPERS_IN_PROJECT = ".cryo-savanna/pytom_runner.py"
MASKS = M.MASKS
TITLES = {MASKS: "Tomogram masks"}
DESCRIPTIONS = {
    MASKS: "A mask per tomogram of a RELION 5 tomogram set, drawn interactively on the tomograms (a polygon in "
           "XY, through all Z or between two tilted planes), named <rlnTomoName>_mask.mrc for pytom's "
           "'Tomogram masks folder'",
}
MENU_TEXT = {MASKS: "Tomogram masks (for pytom)"}
MAIN_INPUTS = {MASKS: "in_tomograms"}


def mask_options() -> tuple[list[OptionDef], list[Tab], dict[int, Group]]:
    opts = [
        _o("in_tomograms", "Input tomograms:", "", "The tomograms.star of a RELION 5 Reconstruct tomograms (or "
           "Denoise, or AreTomo3) job. Each tomogram gets a mask of its size and voxel size, "
           "masks/<rlnTomoName>_mask.mrc in the job: the folder to give pytom's template matching or "
           "extraction ('Tomogram masks folder'). Use the tomograms pytom searches (they need the same size).",
           kind="inputnode", node_type="TomogramGroupMetadata.star.relion", pattern="Tomogram set STAR file (*.star)"),
        _radio("tomo_kind", "Tomograms:", tuple(PR.TOMOGRAM_KINDS), "Reconstructed",
               "Which tomogram of each row the masks are drawn on and take their size from (the denoised ones "
               "are often easier to see)."),
        TOMO_NAMES,
        _radio("start_z", "Starting masks:", (M.START_ALL, M.START_SLAB), M.START_ALL,
               "The mask each tomogram starts with, before it is drawn in the job's Masks view: the whole "
               "tomogram (a rectangle through all Z), or a slab around the tomogram's middle. Tomograms that "
               "already have a mask keep it when the job runs again."),
        _o("start_thickness", "Slab thickness (px):", "", "Of the starting slab, in voxels. Empty: half the "
           "tomogram's thickness."),
        _o("start_margin", "Margin in XY (px):", "0", "The starting rectangle stays this many voxels inside the "
           "edges of the tomogram."),
    ]
    tabs = [Tab("Input", (Row(("in_tomograms",)), Row(("tomo_kind",)), Row(("tomo_names",)), Row(("start_z",)),
                          Row(("start_thickness",), group=1), Row(("start_margin",))))]
    groups = {1: Group(1, None, "start_z", M.START_SLAB)}
    return opts, tabs, groups


class TomoMaskVariant(PytomVariant):
    """The tomogram mask form: a PytomVariant (an own form written to an External job.star) with this runner."""


def runner_command(kind: str = MASKS) -> str:
    return f"python3 {RUNNER_IN_PROJECT} {kind}"


def mask_kind(fn_exe: str) -> str | None:
    """masks for an External executable made by runner_command, else None."""
    parts = fn_exe.split()
    if len(parts) >= 2 and parts[-1] in M.MODES and \
            any(Path(p.strip("'\"")).name == "mask_runner.py" for p in parts[:-1]):
        return parts[-1]
    return None


def mask_kind_of(js: JobStar) -> str | None:
    return mask_kind(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def mask_main_input(js: JobStar) -> str | None:
    """The tomogram set a mask job works on (its main input field), else None."""
    kind = mask_kind_of(js)
    if kind is None:
        return None
    own, _ = decode(js.options.get("other_args", ""))
    value = own.get(MAIN_INPUTS[kind], "").strip()
    while value.startswith("./"):
        value = value[2:]
    return value or None


def install_runner(project_dir: str | os.PathLike) -> Path:
    """Put this cryo-savanna's mask_runner.py and the pytom_runner.py it imports into the project."""
    _install(HELPERS, Path(project_dir) / HELPERS_IN_PROJECT)
    dst = Path(project_dir) / RUNNER_IN_PROJECT
    _install(RUNNER, dst)
    return dst


class TomoMaskCatalog:
    """The tomogram mask form, built once per RELION environment (its queue settings)."""

    def __init__(self, defs: JobDefs, env: RelionEnv):
        self.defs, self.env = defs, env
        self._cache: dict[str, TomoMaskVariant] = {}

    def variant(self, kind: str = MASKS) -> TomoMaskVariant:
        if kind not in self._cache:
            self._cache[kind] = self._build(kind)
        return self._cache[kind]

    def _build(self, kind: str) -> TomoMaskVariant:
        external = self.defs.variant(EXTERNAL, True)
        ext = external.resolved_options(self.env)
        own, tabs, groups = mask_options()
        options = {o.key: o for o in own}
        options["nr_threads"] = replace(ext["nr_threads"], help=(
            "CPU threads to reserve on the queue (XXXthreadsXXX). The masks are written one after another: "
            "1 is enough."))
        tabs, groups = add_running_tab(options, tabs, groups, ext)
        return TomoMaskVariant(key=f"tomomask:{kind}", label=EXTERNAL, proc_id=external.proc_id,
                               dirname=external.dirname, is_tomo=True, options=options, tabs=tabs, groups=groups,
                               kind=kind, program=TITLES[kind], description=DESCRIPTIONS[kind],
                               own_keys=tuple(o.key for o in own), external=external, env=self.env,
                               fn_exe=runner_command(kind))


def newest_input(pipeline: Pipeline) -> dict[str, str]:
    """The newest tomogram set of the pipeline, as the form's input."""
    for name in reversed(list(pipeline.nodes)):
        n = pipeline.nodes[name]
        if n.producer and n.kind == "TomogramGroupMetadata" and name.endswith("tomograms.star"):
            return {"in_tomograms": name}
    return {}


def masks_job_of(folder: str, pipeline: Pipeline | None) -> str | None:
    """The tomogram_masks.star node of the mask job whose masks folder this is (a pipeline node), else None:
    a pytom job reading the folder hangs under that job."""
    if pipeline is None or not folder:
        return None
    f = folder.strip().rstrip("/")
    while f.startswith("./"):
        f = f[2:]
    if not f.endswith("/" + M.MASK_DIR):
        return None
    node = f[: -len(M.MASK_DIR)] + M.LIST
    n = pipeline.nodes.get(node)
    return node if n is not None and n.producer else None
