# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
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
"""Constants mirrored from RELION 5.1 (src/pipeline_jobs.h, src/pipeliner.h, src/pipeline_control.h).

tools/extract_jobdefs cross-checks these tables against the RELION source it extracts from,
so a mismatch with a newer RELION version shows up as a failing extraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobType:
    proc_id: int
    label: str  # e.g. "relion.class2d" (rlnJobTypeLabel / rlnPipeLineProcessTypeLabel)
    dirname: str  # e.g. "Class2D"


JOB_TYPES: tuple[JobType, ...] = (
    JobType(0, "relion.import", "Import"),
    JobType(1, "relion.motioncorr", "MotionCorr"),
    JobType(2, "relion.ctffind", "CtfFind"),
    JobType(3, "relion.manualpick", "ManualPick"),
    JobType(4, "relion.autopick", "AutoPick"),
    JobType(5, "relion.extract", "Extract"),
    JobType(7, "relion.select", "Select"),
    JobType(8, "relion.class2d", "Class2D"),
    JobType(9, "relion.class3d", "Class3D"),
    JobType(10, "relion.refine3d", "Refine3D"),
    JobType(12, "relion.maskcreate", "MaskCreate"),
    JobType(13, "relion.joinstar", "JoinStar"),
    JobType(14, "relion.subtract", "Subtract"),
    JobType(15, "relion.postprocess", "PostProcess"),
    JobType(16, "relion.localres", "LocalRes"),
    JobType(18, "relion.initialmodel", "InitialModel"),
    JobType(19, "relion.multibody", "MultiBody"),
    JobType(20, "relion.polish", "Polish"),
    JobType(21, "relion.ctfrefine", "CtfRefine"),
    JobType(22, "dynamight", "DynaMight"),
    JobType(23, "modelangelo", "ModelAngelo"),
    JobType(50, "relion.importtomo", "Import"),
    JobType(51, "relion.pseudosubtomo", "Extract"),
    JobType(52, "relion.ctfrefinetomo", "CtfRefine"),
    JobType(53, "relion.framealigntomo", "Polish"),
    JobType(54, "relion.reconstructparticletomo", "Reconstruct"),
    JobType(55, "relion.aligntiltseries", "AlignTiltSeries"),
    JobType(56, "relion.reconstructtomograms", "Tomograms"),
    JobType(57, "relion.excludetilts", "ExcludeTiltImages"),
    JobType(58, "relion.denoisetomo", "Denoise"),
    JobType(59, "relion.picktomo", "Picks"),
    JobType(99, "relion.external", "External"),
)

JOB_TYPE_BY_LABEL = {t.label: t for t in JOB_TYPES}
JOB_TYPE_BY_ID = {t.proc_id: t for t in JOB_TYPES}

TOMO_ONLY_LABELS = frozenset(t.label for t in JOB_TYPES if 50 <= t.proc_id < 99)


def job_type_for_label(label: str) -> JobType | None:
    """Like RELION's get_proc_type: strip trailing '.xxx' levels (ccpem-pipeliner style labels)."""
    lab = label
    while True:
        if lab in JOB_TYPE_BY_LABEL:
            return JOB_TYPE_BY_LABEL[lab]
        if "." not in lab:
            break
        lab = lab.rsplit(".", 1)[0]
    # Backward compatibility with alpha versions of relion-4.0 that used directory names.
    # Ambiguous for tomo jobs (e.g. "Import"): RELION's std::map initializer keeps the first
    # (SPA) entry for a duplicate key, so do the same.
    candidates = [t for t in JOB_TYPES if t.dirname == label]
    return candidates[0] if candidates else None


# Process status (rlnPipeLineProcessStatusLabel) and the integer codes used by RELION <= 3.1.
STATUS_RUNNING = "Running"
STATUS_SCHEDULED = "Scheduled"
STATUS_SUCCEEDED = "Succeeded"
STATUS_FAILED = "Failed"
STATUS_ABORTED = "Aborted"
STATUS_BY_CODE = {0: STATUS_RUNNING, 1: STATUS_SCHEDULED, 2: STATUS_SUCCEEDED, 3: STATUS_FAILED, 4: STATUS_ABORTED}
STATUS_LABELS = tuple(STATUS_BY_CODE.values())

# Marker files in a job directory
RELION_JOB_EXIT_SUCCESS = "RELION_JOB_EXIT_SUCCESS"
RELION_JOB_EXIT_FAILURE = "RELION_JOB_EXIT_FAILURE"
RELION_JOB_EXIT_ABORTED = "RELION_JOB_EXIT_ABORTED"
RELION_JOB_ABORT_NOW = "RELION_JOB_ABORT_NOW"
RELION_OUTPUT_NODES = "RELION_OUTPUT_NODES.star"

# Project-level files
PIPELINE_HAS_CHANGED = ".pipeline_has_changed"
LOCK_DIR = ".relion_lock"
GUI_PROJECTDIR = ".gui_projectdir"  # RELION's GUI marks a project directory with it
TMP_RUNFILES = ".TMP_runfiles"  # made by RELION's GUI for a new project (for the CCP-EM pipeliner)

# JobOption kinds (JOBOPTION_* in pipeline_jobs.h)
JOBOPTION_KINDS = {
    0: "undefined",
    1: "any",
    2: "filename",
    3: "inputnode",
    4: "radio",
    5: "boolean",
    6: "slider",
    7: "onlytext",
}

# GUI toggle behaviour when switching between new/continue (TOGGLE_* in pipeline_jobs.h)
TOGGLE_MODES = {0: "deactivate", 1: "reactivate", 2: "always_deactivate", 3: "leave_active"}


def convert_old_node_type_label(label: str) -> str:
    """Port of PipeLine::convertOldNodeTypeLabel for pipelines with version < 50001."""
    pairs = (
        ("ParticlesData", "ParticleGroupMetadata"),
        ("MicrographMoviesData", "MicrographMovieGroupMetadata"),
        ("MicrographsData", "MicrographGroupMetadata"),
        ("MicrographsCoords", "MicrographCoordsGroup"),
        ("ImagesData", "Image2DGroupMetadata"),
        ("OptimisationSet", "TomoOptimisationSet"),
        ("TomogramData", "TomogramGroupMetadata"),
        ("TrajectoryData", "TomoTrajectoryData"),
        ("ProcessData.star.relion.tomo.relion.tiltseries_set", "TomogramGroupMetadata.star.relion.tomo"),
        ("ProcessData.star.relion.tomo.relion.tomogram_set", "TomogramGroupMetadata.star.relion.tomo"),
    )
    for old, new in pairs:
        if old in label:
            return label.replace(old, new)
    return label


# Hidden "last used settings" files the original GUI and relion_pipeliner write in the project
# directory (RelionJob::hidden_name, src/pipeline_jobs.cpp): <name>job.star
HIDDEN_SETTINGS = {
    "relion.import": ".gui_import", "relion.motioncorr": ".gui_motioncorr", "relion.ctffind": ".gui_ctffind",
    "relion.manualpick": ".gui_manualpick", "relion.autopick": ".gui_autopick", "relion.extract": ".gui_extract",
    "relion.select": ".gui_select", "relion.class2d": ".gui_class2d", "relion.initialmodel": ".gui_inimodel",
    "relion.class3d": ".gui_class3d", "relion.refine3d": ".gui_auto3d", "relion.multibody": ".gui_multibody",
    "relion.maskcreate": ".gui_maskcreate", "relion.joinstar": ".gui_joinstar", "relion.subtract": ".gui_subtract",
    "relion.postprocess": ".gui_post", "relion.localres": ".gui_localres", "dynamight": ".gui_dynamight",
    "modelangelo": ".gui_modelangelo", "relion.polish": ".gui_bayespolish", "relion.ctfrefine": ".gui_ctfrefine",
    "relion.external": ".gui_external", "relion.importtomo": ".gui_tomo_import",
    "relion.aligntiltseries": ".gui_tomo_align_tiltseries",
    "relion.reconstructtomograms": ".gui_tomo_reconstruct_tomograms", "relion.denoisetomo": ".gui_tomo_denoise_tomogram",
    "relion.picktomo": ".gui_tomo_pick_tomogram", "relion.excludetilts": ".gui_tomo_exclude_tilt_images",
    "relion.pseudosubtomo": ".gui_tomo_subtomo", "relion.ctfrefinetomo": ".gui_tomo_refine_ctf",
    "relion.framealigntomo": ".gui_tomo_align", "relion.reconstructparticletomo": ".gui_tomo_reconstruct_particle",
}
