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
#
#   The XMIPP code in RELION's src/multidim_array.h and src/transformations.h (GPL-2.0-or-later;
#   centerOfMass, translateCenterOfMassToCenter, used for --recenter):
#   Authors: Carlos Oscar S. Sorzano (coss@cnb.csic.es), Sjors H.W. Scheres
#   Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
# The tilt-image exclusion follows RELION's tomography Python programs (Alister Burt; BSD-3-Clause): see NOTICE.md.
# Their notice, kept as their licence asks:
#   Copyright (c) Alister Burt and the RELION developers
#   (RELION 5.1.1, src/tomography_python_programs; licence as declared in RELION's pyproject.toml)
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted
#   provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this list of conditions
#      and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of
#      conditions and the following disclaimer in the documentation and/or other materials provided
#      with the distribution.
#   3. Neither the name of the copyright holder nor the names of its contributors may be used to
#      endorse or promote products derived from this software without specific prior written
#      permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
#   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#   FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#   IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#   OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""The files RELION's interactive selection tools write, written by cryo-savanna itself.

Select (interactive, on a Class2D/Class3D/InitialModel model or optimiser file) runs
``relion_display --gui --i <optimiser> --allow_save --fn_parts <job>particles.star
[--fn_imgs <job>class_averages.star] [--recenter] [--regroup N]``; saving writes (src/displayer.cpp):

- ``particles.star``: the data file's ``general`` (if any) and ``optics`` tables as they are, and
  every particle of the selected classes: class by class in class order, each class's particles in
  file order (``makeStarFileSelectedParticles``); with ``--regroup``, the groups rebuilt from the
  model's scale corrections (``regroupSelectedParticles``).
- ``class_averages.star`` (only for Class2D, or for Select's own ``run_optimiser.star``): the
  selected rows of ``model_classes`` with ``rlnClassNumber`` added, in class order (``saveSelected``);
  with ``--recenter``, each average moved to its centre of mass into ``class_averages.mrcs``.
- ``backup_selection.star``: ``rlnSelected`` 1/0 for every class (``saveBackupSelection``), read
  back when the selection is opened again.

Exclude tilt images runs ``relion_python_tomo_exclude_tilt_images``; saving writes
(tomography_python_programs/exclude_tilt_images): ``selected_tilt_series.star`` (the input's
``global`` table without the tilt series that kept no tilt, ``rlnTomoTiltSeriesStarFile`` moved to
the end and pointing to the new per-series files) and ``tilt_series/<name>.star`` (the kept tilts,
sorted by tilt angle).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna.project.star import FormattedLoop, StarBlock, format_row, loop_tokens, read_star, write_star

SELECT_FILES = ("particles.star", "class_averages.star", "backup_selection.star")


class SelectionError(ValueError):
    pass


# ---------------------------------------------------------------------------- classes

@dataclass
class ClassInput:
    """What Select reads: the classes (model_classes) and the particles (data.star) of one iteration."""

    project_dir: Path
    source: str  # the optimiser or model file given to Select (fn_model)
    model: str
    data: str
    classes: StarBlock  # model_classes, with rlnClassNumber
    groups: StarBlock | None = None  # model_groups (for regrouping)

    @property
    def numbers(self) -> list[int]:
        return [int(float(v)) for v in self.classes.column("rlnClassNumber")]

    @property
    def writes_class_averages(self) -> bool:
        """RELION writes class_averages.star only for 2D classes (pipeline_jobs.cpp)."""
        s = self.source
        return "Class2D/" in s or ("Select/" in s and "run_optimiser.star" in s)


def _without(s: str, suffix: str) -> str:
    return s[: -len(suffix)] if s.endswith(suffix) else s.replace(suffix, "")


def read_class_input(project_dir: str | os.PathLike, source: str) -> ClassInput:
    """The model and data files Select uses for an optimiser or model file (Displayer::initialise)."""
    project_dir = Path(project_dir)
    model = source
    if source.endswith("_optimiser.star"):
        data = _without(source, "_optimiser.star") + "_data.star"
        gen = read_star(project_dir / source, ["optimiser_general"]).get("optimiser_general")
        model = gen.get("rlnModelStarFile") if gen is not None and gen.get("rlnModelStarFile") else \
            _without(source, "_optimiser.star") + "_model.star"
    elif source.endswith("_half1_model.star"):
        data = _without(source, "_half1_model.star") + "_data.star"
    elif source.endswith("_half2_model.star"):
        data = _without(source, "_half2_model.star") + "_data.star"
    elif source.endswith("_model.star"):
        data = _without(source, "_model.star") + "_data.star"
    else:
        raise SelectionError(f"{source}: not a model or optimiser file")
    blocks = read_star(project_dir / model, ["model_classes", "model_groups"])
    classes = blocks.get("model_classes")
    if classes is None or not classes.has("rlnReferenceImage"):
        raise SelectionError(f"{model}: no model_classes")
    classes = StarBlock(classes.name, list(classes.columns), [list(r) for r in classes.rows], True, classes.version)
    if not classes.has("rlnClassNumber"):  # Displayer::run: "Store class number in metadata table"
        classes.columns.append("rlnClassNumber")
        for i, r in enumerate(classes.rows):
            r.append(str(i + 1))
    if not (project_dir / data).is_file():
        raise SelectionError(f"{data}: not found (the particles of {source})")
    return ClassInput(project_dir, source, model, data, classes, blocks.get("model_groups"))


@dataclass
class ClassSelectionResult:
    particles: int
    classes: int
    files: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def write_class_selection(inp: ClassInput, job: str, selected: list[int], recenter: bool = False,
                          regroup: int = 0) -> ClassSelectionResult:
    """Write particles.star, class_averages.star and backup_selection.star into ``job`` (e.g. "Select/job007/").

    Raises SelectionError, before writing anything, when the selected classes have no particles:
    the files of a job saved earlier stay as they are.
    """
    project = inp.project_dir
    out_dir = project / job
    chosen = set(selected)
    numbers = inp.numbers
    sel_flags = [n in chosen for n in numbers]
    result = ClassSelectionResult(0, sum(sel_flags))

    # particles.star. The particles are read one at a time and only those of the selected classes are
    # kept (as the line each becomes in the new file, unless they are regrouped first): a data.star
    # of millions of particles, held as rows of values, takes several times its size in memory
    rows_in = loop_tokens(project / inp.data, "particles")
    columns = next(rows_in, None)
    if columns is None or "rlnClassNumber" not in columns:
        raise SelectionError(f"{inp.data}: no particles with rlnClassNumber")
    ci = columns.index("rlnClassNumber")
    wanted = {n for n, on in zip(numbers, sel_flags) if on}
    keep = (lambda row: row) if regroup > 0 else format_row
    by_class: dict[int, list] = {}
    for row in rows_in:
        n = int(float(row[ci]))
        if n in wanted:
            by_class.setdefault(n, []).append(keep(row))
    rows = [r for n, on in zip(numbers, sel_flags) if on for r in by_class.get(n, [])]
    del by_class
    result.particles = len(rows)
    if not rows:
        raise SelectionError("No classes selected." if not any(sel_flags) else "The selected classes have no particles.")
    if regroup > 0:
        if inp.groups is None:
            raise SelectionError(f"{inp.model}: no model_groups to regroup with")
        out = StarBlock("particles", list(columns), rows, True)
        result.messages.append(regroup_particles(out, inp.groups, regroup))
    else:
        out = FormattedLoop("particles", list(columns), rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = read_star(project / inp.data, ["general", "optics"])  # in the file's order
    write_star(out_dir / "particles.star", list(blocks.values()) + [out])
    result.files.append(job + "particles.star")

    # class_averages.star (2D only)
    if inp.writes_class_averages:
        cls_rows = [list(r) for r, on in zip(inp.classes.rows, sel_flags) if on]
        avg = StarBlock("", list(inp.classes.columns), cls_rows, True, inp.classes.version)
        if recenter:
            _recenter(project, avg, job + "class_averages.mrcs")
            result.files.append(job + "class_averages.mrcs")
        write_star(out_dir / "class_averages.star", [avg])
        result.files.append(job + "class_averages.star")

    # backup_selection.star
    write_star(out_dir / "backup_selection.star",
               [StarBlock("", ["rlnSelected"], [[1 if on else 0] for on in sel_flags], True)])
    result.files.append(job + "backup_selection.star")
    return result


def read_backup_selection(project_dir: str | os.PathLike, job: str, n_classes: int) -> list[bool] | None:
    """The selection saved in a Select job (None if there is none, or for another number of classes)."""
    path = Path(project_dir) / job / "backup_selection.star"
    if not path.is_file():
        return None
    b = next((b for b in read_star(path).values() if b.has("rlnSelected")), None)
    if b is None or len(b) != n_classes:
        return None
    return [int(float(v)) == 1 for v in b.column("rlnSelected")]


def selected_numbers(inp: ClassInput, flags: list[bool] | None) -> list[int]:
    """The class numbers of a saved selection (backup_selection.star: one flag per class, in the
    order of model_classes)."""
    return [n for n, on in zip(inp.numbers, flags or []) if on]


def regroup_particles(parts: StarBlock, groups: StarBlock, nr_regroups: int) -> str:
    """regroupSelectedParticles: new rlnGroupName from groups sorted by scale, within optics groups."""
    if not (parts.has("rlnGroupNumber") and parts.has("rlnOpticsGroup")):
        raise SelectionError("regrouping needs rlnGroupNumber and rlnOpticsGroup in the particles")
    gi, oi = parts.col_index("rlnGroupNumber"), parts.col_index("rlnOpticsGroup")
    g_no = [int(float(v)) for v in groups.column("rlnGroupNumber")]
    g_scale = [float(v) for v in groups.column("rlnGroupScaleCorrection")]
    optics = {g: -1 for g in g_no}
    count = {g: 0 for g in g_no}
    max_optics = -1
    for row in parts.rows:
        g, o = int(float(row[gi])), int(float(row[oi]))
        if optics.get(g, -1) == -1:
            optics[g] = o
            max_optics = max(max_optics, o)
        count[g] = count.get(g, 0) + 1
    n = len(parts.rows)
    average = n // nr_regroups
    if average < 10:
        raise SelectionError("Each group should have at least 10 particles")
    width = int(math.floor(math.log(nr_regroups) / math.log(10))) + 1
    order = [g for _, g in sorted(zip(g_scale, g_no), key=lambda t: t[0])]  # MDgroups.sort(scale)
    names: dict[int, str] = {}
    new_id = 0
    for optics_id in range(1, max_optics + 1):
        in_group = 0
        new_id += 1
        for g in order:
            if optics.get(g) != optics_id:
                continue
            in_group += count.get(g, 0)
            if in_group > average:
                new_id += 1
                in_group = 0
            names[g] = f"group_{new_id:0{width}d}"
    missing = sorted({g for g in (int(float(r[gi])) for r in parts.rows) if g not in names})
    if missing:
        raise SelectionError(f"cannot regroup: group(s) {', '.join(map(str, missing))} of the particles are not in "
                             "the model's model_groups")
    if "rlnGroupName" not in parts.columns:
        parts.columns.append("rlnGroupName")
        for r in parts.rows:
            r.append("")
    ni = parts.col_index("rlnGroupName")
    for r in parts.rows:
        r[ni] = names[int(float(r[gi]))]
    del parts.columns[gi]  # MDdata.deactivateLabel(EMDL_MLMODEL_GROUP_NO)
    for r in parts.rows:
        del r[gi]
    return f"Regrouped particles into {new_id} groups"


def _recenter(project: Path, avg: StarBlock, stack_rel: str) -> None:
    """saveSelected with --recenter: averages moved to their centre of mass, in a new stack."""
    import mrcfile

    from cryo_savanna.images.mrc import ImageRef, header, read_image
    ri = avg.col_index("rlnReferenceImage")
    images = []
    voxel = 0.0
    for r in avg.rows:
        ref = ImageRef.parse(r[ri]).resolve(project)
        voxel = voxel or header(ref.path).pixel_size
        images.append(center_of_mass_to_center(read_image(ref)))
    stack = np.stack(images).astype(np.float32)
    with mrcfile.new(project / stack_rel, overwrite=True) as m:
        m.set_data(stack)
        m.voxel_size = voxel
        m.header.ispg = 0  # an image stack
    for i, r in enumerate(avg.rows):
        r[ri] = f"{i + 1:06d}@{stack_rel}"


def center_of_mass_to_center(img: np.ndarray) -> np.ndarray:
    """selfTranslateCenterOfMassToCenter: shift (bilinear, wrapping: applyGeometry with WRAP) so the
    centre of mass is at the image centre (RELION's origin: size // 2). As MultidimArray::centerOfMass,
    only positive pixels count."""
    ny, nx = img.shape
    w = np.where(img > 0, img, 0).astype(np.float64)
    total = float(w.sum())
    if total == 0:
        return img.copy()
    y, x = np.mgrid[:ny, :nx]
    cy = float((w * (y - ny // 2)).sum() / total)
    cx = float((w * (x - nx // 2)).sum() / total)
    # out(y, x) = img(y + cy, x + cx), bilinear with wrap
    sy, sx = (y + cy) % ny, (x + cx) % nx
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = sy - y0, sx - x0
    y1, x1 = (y0 + 1) % ny, (x0 + 1) % nx
    return ((1 - fy) * (1 - fx) * img[y0, x0] + (1 - fy) * fx * img[y0, x1] +
            fy * (1 - fx) * img[y1, x0] + fy * fx * img[y1, x1]).astype(np.float32)


# ---------------------------------------------------------------------------- tilt images

@dataclass
class TiltInput:
    project_dir: Path
    source: str  # the input tilt-series set (in_tiltseries)
    global_block: StarBlock
    tilts: dict[str, StarBlock]  # name -> per-tilt table, as in the input

    @property
    def names(self) -> list[str]:
        return self.global_block.column("rlnTomoName")


def read_tilt_input(project_dir: str | os.PathLike, source: str) -> TiltInput:
    project_dir = Path(project_dir)
    g = read_star(project_dir / source, ["global"]).get("global")
    if g is None or not g.has("rlnTomoTiltSeriesStarFile"):
        raise SelectionError(f"{source}: no tilt series (data_global with rlnTomoTiltSeriesStarFile)")
    tilts = {}
    for name, rel in zip(g.column("rlnTomoName"), g.column("rlnTomoTiltSeriesStarFile")):
        blocks = read_star(project_dir / rel)
        b = blocks.get(name)  # not `or`: a block without rows is falsy
        if b is None:
            b = next((b for b in blocks.values() if b.is_loop), None)
        if b is None:
            raise SelectionError(f"{rel}: no tilt table")
        tilts[name] = b
    return TiltInput(project_dir, source, g, tilts)


def _tilt_key(b: StarBlock) -> int:
    for col in ("rlnMicrographName", "rlnMicrographMovieName"):
        if b.has(col):
            return b.col_index(col)
    raise SelectionError("tilt table without rlnMicrographName")


def write_tilt_exclusion(inp: TiltInput, job: str, kept: dict[str, list[bool]]) -> dict[str, int]:
    """Write selected_tilt_series.star and tilt_series/<name>.star into ``job``; returns tilts kept per series.

    Raises SelectionError, before writing anything, when the choice does not fit or keeps no tilt.
    """
    out_dir = inp.project_dir / job
    g = inp.global_block
    si = g.col_index("rlnTomoTiltSeriesStarFile")
    cols = [c for c in g.columns if c != "rlnTomoTiltSeriesStarFile"] + ["rlnTomoTiltSeriesStarFile"]
    tables, rows, counts = [], [], {}
    for row in g.rows:
        name = row[g.col_index("rlnTomoName")]
        b = inp.tilts[name]
        mask = kept.get(name, [True] * len(b))
        if len(mask) != len(b):
            raise SelectionError(f"{name}: {len(mask)} choices for {len(b)} tilts")
        sel = [r for r, k in zip(b.rows, mask) if k]
        counts[name] = len(sel)
        if not sel:  # write_star_file drops tilt series without images
            continue
        if b.has("rlnTomoNominalStageTiltAngle"):
            ai = b.col_index("rlnTomoNominalStageTiltAngle")
            sel = sorted(sel, key=lambda r: float(r[ai]))  # the excluder sorts by tilt angle
        rel = f"{job}tilt_series/{name}.star"
        tables.append((rel, StarBlock(name, list(b.columns), [list(r) for r in sel], True, b.version)))
        rows.append([v for i, v in enumerate(row) if i != si] + [rel])
    if not rows:
        raise SelectionError("every tilt image is excluded")
    (out_dir / "tilt_series").mkdir(parents=True, exist_ok=True)
    for rel, block in tables:
        write_star(inp.project_dir / rel, [block])
    write_star(out_dir / "selected_tilt_series.star", [StarBlock("global", cols, rows, True, g.version)])
    return counts


def read_tilt_exclusion(inp: TiltInput, job: str) -> dict[str, list[bool]] | None:
    """The tilts an Exclude tilt images job kept, matched to the input by image name (None if it wrote nothing)."""
    path = inp.project_dir / job / "selected_tilt_series.star"
    if not path.is_file():
        return None
    out = read_tilt_input(inp.project_dir, str(path.relative_to(inp.project_dir)))
    kept = {}
    for name, b in inp.tilts.items():
        k = _tilt_key(b)
        if name not in out.tilts:
            kept[name] = [False] * len(b)
            continue
        ob = out.tilts[name]
        have = {r[_tilt_key(ob)] for r in ob.rows}
        kept[name] = [r[k] in have for r in b.rows]
    return kept


# ---------------------------------------------------------------------------- which jobs

SELECT_OTHER_MODES = ("do_class_ranker", "do_filaments", "do_remove_duplicates", "do_select_values", "do_discard",
                      "do_split")


def interactive_kind(type_label: str, options: dict[str, str]) -> str | None:
    """"classes" for an interactive Select on classes (relion_display), "tilts" for Exclude tilt
    images (napari), else None: the jobs whose interactive step cryo-savanna can do itself."""
    from cryo_savanna.relion_consts import job_type_for_label
    jt = job_type_for_label(type_label)
    t = jt.label if jt is not None else type_label
    if t == "relion.select" and options.get("fn_model") and not any(options.get(k) == "Yes" for k in SELECT_OTHER_MODES):
        return "classes"
    if t == "relion.excludetilts" and options.get("in_tiltseries"):
        return "tilts"
    return None


def relion_command(kind: str, options: dict[str, str], job: str, recenter: bool = False, regroup: int = 0) -> str:
    """The command RELION would have run for the job (written to note.txt)."""
    if kind == "classes":
        src = options.get("fn_model", "")
        cmd = f"`which relion_display` --gui --i {src} --allow_save --fn_parts {job}particles.star"
        if "Class2D/" in src or ("Select/" in src and "run_optimiser.star" in src):
            cmd += f" --fn_imgs {job}class_averages.star" + (" --recenter " if recenter else "")
        if regroup > 0:
            cmd += f" --regroup {regroup}"
        return cmd
    return (f"`which relion_python_tomo_exclude_tilt_images`  --tilt-series-star-file {options.get('in_tiltseries', '')}"
            f" --cache-size {options.get('cache_size', '5')}  --output-directory {job}")
