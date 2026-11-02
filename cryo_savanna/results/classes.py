# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""2D/3D classes per iteration, class averages chosen by Select, and particle stacks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryo_savanna.images.mrc import ImageRef
from cryo_savanna.project.star import loop_columns, read_star
from cryo_savanna.project.summary import current_iterations
from cryo_savanna.results.tables import Table, optics_value

_MODEL_RE = re.compile(r"^(?P<root>.*?)_it(?P<it>\d+)(?P<half>_half1)?_model\.star$")
# "run", or "run_ct<iteration>" for a continued run (MultiBody in RELION 5.1, every refine job before 3.1)
_RUN_ROOT = re.compile(r"^run(_ct\d+)?$")


@dataclass
class ClassInfo:
    number: int  # 1-based
    ref: ImageRef
    distribution: float | None = None  # fraction of the particles
    resolution: float | None = None  # Å (rlnEstimatedResolution)
    accuracy_rot: float | None = None
    accuracy_trans: float | None = None
    score: float | None = None  # rlnClassScore / rlnPredictedClassScore (class ranker)
    particles: int | None = None  # counted in the iteration's data.star

    def label(self) -> str:
        parts = [f"{self.number}"]
        if self.distribution is not None:
            parts.append(f"{100 * self.distribution:.1f}%")
        if self.resolution is not None and self.resolution < 900:
            parts.append(f"{self.resolution:.1f} Å")
        return "  ".join(parts)


@dataclass
class ClassSet:
    source: str  # model.star or class_averages.star, relative to the project
    classes: list[ClassInfo]
    dim: int = 2
    iteration: int | None = None
    pixel_size: float | None = None


def iterations(job_dir: Path) -> list[tuple[int, Path]]:
    """(iteration, model.star) of an iterative job, oldest first (Refine3D: the half1 models)."""
    try:
        names = os.listdir(job_dir)
    except OSError:
        return []
    files = []
    for n in names:
        m = _MODEL_RE.match(n)
        if m and _RUN_ROOT.match(m.group("root")):
            files.append((int(m.group("it")), n))
    # a continued run's models replace the earlier run's (current_iterations); a finished Refine3D
    # also has run_model.star (no iteration number): the final one
    return [(it, job_dir / n) for it, n in current_iterations(files).items()]


_ITER_MAP = re.compile(r"^(?P<root>.+)_it(?P<it>\d+)_(?:half(?P<half>[12])_)?(?P<what>class|body)(?P<n>\d+)\.mrc$")


def latest_iteration_maps(project_dir: str | os.PathLike, job: str) -> list[str]:
    """The maps of an iterative job's latest finished iteration (relative to the project), for a job that
    is still running or stopped before writing its final maps: Class3D/InitialModel classes
    (run_it012_class001.mrc ...), Refine3D/MultiBody half maps (run_it012_half1_class001.mrc ...).

    An iteration counts once its optimiser.star is there: RELION writes it after the maps.
    """
    job_dir = Path(project_dir) / job
    its = iterations(job_dir)
    if not its:
        return []

    def optimiser(model: Path) -> Path:
        return model.with_name(model.name.replace("_half1_model.star", "_optimiser.star")
                               .replace("_model.star", "_optimiser.star"))
    it, model = next(((i, m) for i, m in reversed(its) if optimiser(m).is_file()), its[-1])
    try:
        names = os.listdir(job_dir)
    except OSError:
        return []
    return _maps_of(job, names, model.name.split("_it", 1)[0], it)


def iteration_maps(project_dir: str | os.PathLike, job: str) -> list[tuple[int, list[str]]]:
    """(iteration, its maps) of every iteration of an iterative job that has maps, oldest first, for
    the job's latest run (a continued run's iterations replace the earlier run's, see iterations())."""
    job_dir = Path(project_dir) / job
    its = iterations(job_dir)
    try:
        names = os.listdir(job_dir) if its else []
    except OSError:
        return []
    out = []
    for it, model in its:
        maps = _maps_of(job, names, model.name.split("_it", 1)[0], it)
        if maps:
            out.append((it, maps))
    return out


def _maps_of(job: str, names: list[str], root: str, it: int) -> list[str]:
    """The maps of one iteration: classes, bodies, half maps (relative to the project), class by class."""
    maps = []
    for n in names:
        m = _ITER_MAP.match(n)
        if m and m["root"] == root and int(m["it"]) == it:
            maps.append(((m["what"], int(m["n"]), m["half"] or ""), job + n))
    return [rel for _, rel in sorted(maps)]


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def read_model_classes(project_dir: str | os.PathLike, model_rel: str, data_star: Path | None = None) -> ClassSet:
    project_dir = Path(project_dir)
    path = project_dir / model_rel
    blocks = read_star(path, ["model_general", "model_classes"])
    gen = blocks.get("model_general")
    cl = blocks.get("model_classes")
    if cl is None or not cl.has("rlnReferenceImage"):
        raise ValueError(f"{model_rel}: no model_classes")
    t = Table.from_block(cl)
    dim = int(_f(gen.get("rlnReferenceDimensionality")) or 2) if gen else 2
    px = _f(gen.get("rlnPixelSize")) if gen else None
    m = _MODEL_RE.match(path.name)
    counts = class_counts(data_star) if data_star is not None and data_star.is_file() else {}
    classes = []
    for i, ref in enumerate(t.column("rlnReferenceImage")):
        def val(label: str) -> float | None:
            return _f(t.column(label)[i]) if t.has(label) else None
        # the class's number, as the particles name it: rlnClassNumber where the table has it (as
        # project/selection.py reads it, so a selection saves the classes shown), else its row
        number = int(val("rlnClassNumber")) if val("rlnClassNumber") is not None else i + 1
        classes.append(ClassInfo(number, ImageRef.parse(ref).resolve(project_dir), val("rlnClassDistribution"),
                                 val("rlnEstimatedResolution"), val("rlnAccuracyRotations"),
                                 val("rlnAccuracyTranslationsAngst"),
                                 val("rlnClassScore") if t.has("rlnClassScore") else val("rlnPredictedClassScore"),
                                 counts.get(number)))
    return ClassSet(model_rel, classes, dim, int(m.group("it")) if m else None, px)


def data_star_for(model: Path) -> Path:
    """run_it025_model.star -> run_it025_data.star (Refine3D: run_it012_half1_model.star -> run_it012_data.star)."""
    return model.with_name(model.name.replace("_half1_model.star", "_data.star").replace("_model.star", "_data.star"))


def class_counts(data_star: Path) -> dict[int, int]:
    r = loop_columns(data_star, "particles", ["rlnClassNumber"])
    if r is None or "rlnClassNumber" not in r[1]:
        return {}
    nums, counts = np.unique(np.array(r[1]["rlnClassNumber"], dtype=float).astype(int), return_counts=True)
    return {int(n): int(c) for n, c in zip(nums, counts)}


def read_class_averages(project_dir: str | os.PathLike, star_rel: str) -> ClassSet:
    """Class averages picked by Select (``class_averages.star``) or any list of rlnReferenceImage."""
    project_dir = Path(project_dir)
    blocks = read_star(project_dir / star_rel)
    for b in blocks.values():
        if b.is_loop and b.has("rlnReferenceImage"):
            t = Table.from_block(b)
            break
    else:
        raise ValueError(f"{star_rel}: no rlnReferenceImage")
    classes = []
    for i, ref in enumerate(t.column("rlnReferenceImage")):
        r = ImageRef.parse(ref)
        def val(label: str) -> float | None:
            return _f(t.column(label)[i]) if t.has(label) else None
        num = int(val("rlnClassNumber") or (r.index + 1 if r.index is not None else i + 1))
        classes.append(ClassInfo(num, r.resolve(project_dir), val("rlnClassDistribution"),
                                 val("rlnEstimatedResolution"), val("rlnAccuracyRotations"),
                                 val("rlnAccuracyTranslationsAngst"), val("rlnClassScore")))
    return ClassSet(star_rel, classes, 2)


# ---------------------------------------------------------------------------- particles

PARTICLE_COLUMNS = ["rlnImageName", "rlnMicrographName", "rlnClassNumber", "rlnCoordinateX", "rlnCoordinateY",
                    "rlnAutopickFigureOfMerit", "rlnDefocusU", "rlnMaxValueProbDistribution", "rlnAngleRot",
                    "rlnAngleTilt", "rlnAnglePsi", "rlnOpticsGroup", "rlnGroupNumber"]


@dataclass
class ParticleSet:
    project_dir: Path
    source: str
    table: Table  # PARTICLE_COLUMNS that the file has
    pixel_size: float | None = None
    box: int | None = None

    def __len__(self) -> int:
        return len(self.table)

    def image(self, i: int) -> ImageRef:
        return ImageRef.parse(self.table.column("rlnImageName")[i]).resolve(self.project_dir)

    def tooltip(self, i: int) -> str:
        t = self.table
        lines = [f"Particle {i + 1}", t.column("rlnImageName")[i]]
        for label, title in (("rlnMicrographName", "micrograph"), ("rlnClassNumber", "class"),
                             ("rlnAutopickFigureOfMerit", "FOM"), ("rlnMaxValueProbDistribution", "max. prob."),
                             ("rlnDefocusU", "defocus U")):
            if t.has(label):
                lines.append(f"{title}: {t.column(label)[i]}")
        return "\n".join(lines)


def read_particles(project_dir: str | os.PathLike, star_rel: str) -> ParticleSet:
    project_dir = Path(project_dir)
    path = project_dir / star_rel
    for block in ("particles", ""):
        t = Table.read(path, block, PARTICLE_COLUMNS)
        if t is not None and t.has("rlnImageName"):
            break
    else:
        raise ValueError(f"{star_rel}: no particles (rlnImageName)")
    px = optics_value(path, "rlnImagePixelSize")
    box = optics_value(path, "rlnImageSize")
    return ParticleSet(project_dir, star_rel, t, px, int(box) if box else None)

