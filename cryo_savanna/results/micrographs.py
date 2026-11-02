# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Micrograph and movie lists (with their per-micrograph values), motion tracks and particle picks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna.images.mrc import ImageRef
from cryo_savanna.project.star import loop_columns, loop_rows, read_star
from cryo_savanna.results.tables import Table, optics_value

CLASS = "rlnClassNumber"

MIC = "rlnMicrographName"
MOVIE = "rlnMicrographMovieName"
COORDS = "rlnMicrographCoordinates"

# per-micrograph values worth a column, in this order (others are still in the STAR viewer)
METRICS = [
    ("rlnAccumMotionTotal", "Motion (Å)"), ("rlnAccumMotionEarly", "Early"), ("rlnAccumMotionLate", "Late"),
    ("rlnDefocusU", "Defocus U (Å)"), ("rlnDefocusV", "Defocus V (Å)"), ("rlnCtfAstigmatism", "Astig. (Å)"),
    ("rlnDefocusAngle", "Angle"), ("rlnCtfMaxResolution", "Max. res. (Å)"), ("rlnCtfFigureOfMerit", "FOM"),
    ("rlnCtfIceRingDensity", "Ice rings"), ("rlnCtfScalefactor", "Scale"),
]


@dataclass
class MicrographSet:
    """The micrographs (or movies) of one STAR file, with their per-micrograph columns."""

    project_dir: Path
    source: str  # STAR file, relative to the project
    table: Table
    name_col: str  # rlnMicrographName or rlnMicrographMovieName
    pixel_size: float | None = None

    @property
    def kind(self) -> str:
        return "movies" if self.name_col == MOVIE else "micrographs"

    def __len__(self) -> int:
        return len(self.table)

    @property
    def names(self) -> list[str]:
        return self.table.column(self.name_col)

    def image(self, i: int) -> ImageRef:
        return ImageRef.parse(self.names[i]).resolve(self.project_dir)

    def ctf_image(self, i: int) -> ImageRef | None:
        if not self.table.has("rlnCtfImage"):
            return None
        return ImageRef.parse(self.table.column("rlnCtfImage")[i]).resolve(self.project_dir)

    def motion_file(self, i: int) -> Path | None:
        if not self.table.has("rlnMicrographMetadata"):
            return None
        return self.project_dir / self.table.column("rlnMicrographMetadata")[i]

    def metric_columns(self) -> list[tuple[str, str]]:
        return [(c, t) for c, t in METRICS if self.table.has(c)]


def read_micrographs(project_dir: str | os.PathLike, star_rel: str) -> MicrographSet:
    project_dir = Path(project_dir)
    path = project_dir / star_rel
    blocks = read_star(path)
    for block, col in (("micrographs", MIC), ("movies", MOVIE), ("", MIC), ("", MOVIE)):
        b = blocks.get(block)
        if b is not None and b.has(col):
            table = Table.from_block(b, str(path))
            break
    else:
        raise ValueError(f"{star_rel}: no micrographs or movies")
    px = optics_value(path, "rlnMicrographPixelSize") or optics_value(path, "rlnMicrographOriginalPixelSize")
    return MicrographSet(project_dir, star_rel, table, col, px)


def micrographs_from_names(project_dir: str | os.PathLike, names: list[str], source: str = "") -> MicrographSet:
    """A bare list (e.g. the micrographs a coordinate list names, when their STAR file is gone)."""
    return MicrographSet(Path(project_dir), source, Table("micrographs", [MIC], {MIC: list(names)}), MIC)


def motion_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Frame numbers and global shifts (x, y, in unbinned movie pixels) of one micrograph."""
    try:
        b = read_star(path, ["global_shift"]).get("global_shift")
    except (OSError, ValueError):
        return None
    if b is None or not b.rows or not b.has("rlnMicrographShiftX"):
        return None
    t = Table.from_block(b)
    frames = t.numeric("rlnMicrographFrameNumber") if t.has("rlnMicrographFrameNumber") else np.arange(1, len(t) + 1)
    return frames, t.numeric("rlnMicrographShiftX"), t.numeric("rlnMicrographShiftY")


# ---------------------------------------------------------------------------- picks

@dataclass
class Picks:
    x: np.ndarray
    y: np.ndarray
    fom: np.ndarray | None = None  # rlnAutopickFigureOfMerit
    cls: np.ndarray | None = None  # rlnClassNumber (particle files of classification jobs)

    def __len__(self) -> int:
        return len(self.x)

    def in_classes(self, classes: set[int] | None) -> np.ndarray:
        """Which picks belong to one of the classes (all of them for None, or without class numbers)."""
        if classes is None or self.cls is None:
            return np.ones(len(self), bool)
        return np.isin(self.cls, list(classes))


@dataclass
class PickSet:
    """Coordinates per micrograph: from a coordinate list (AutoPick, ManualPick) or a particle file."""

    project_dir: Path
    source: str
    micrographs: list[str]
    coord_files: dict[str, str] = field(default_factory=dict)  # micrograph -> coordinate STAR (relative)
    pixel_size: float | None = None  # of the micrographs, from a particle file's optics
    classes: dict[int, int] = field(default_factory=dict)  # class -> particles (particle files with classes)
    _from_particles: dict[str, Picks] | None = None
    _cache: dict[str, tuple[tuple, Picks]] = field(default_factory=dict, repr=False)

    def picks(self, mic: str) -> Picks:
        if self._from_particles is not None:
            return self._from_particles.get(mic, Picks(np.empty(0), np.empty(0)))
        rel = self.coord_files.get(mic)
        if not rel:
            return Picks(np.empty(0), np.empty(0))
        path = self.project_dir / rel
        try:
            st = os.stat(path)
        except OSError:
            return Picks(np.empty(0), np.empty(0))
        key = (st.st_mtime_ns, st.st_size)
        hit = self._cache.get(mic)
        if hit is not None and hit[0] == key:
            return hit[1]
        p = _read_coords(path)
        self._cache[mic] = (key, p)
        return p

    def count(self, mic: str, classes: set[int] | None = None) -> int | None:
        if self._from_particles is not None:
            return int(self.picks(mic).in_classes(classes).sum())
        rel = self.coord_files.get(mic)
        if not rel:
            return 0
        path = self.project_dir / rel
        if not path.is_file():
            return 0
        try:
            for block in ("", "coordinates"):
                r = loop_rows(path, block)
                if r is not None:
                    return r[0]
        except OSError:
            return None
        return 0


def _read_coords(path: Path) -> Picks:
    labels = ["rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
    for block in ("", "coordinates"):
        r = loop_columns(path, block, labels)
        if r is not None and "rlnCoordinateX" in r[1]:
            t = Table(block, list(r[1]), r[1])
            fom = t.numeric("rlnAutopickFigureOfMerit") if t.has("rlnAutopickFigureOfMerit") else None
            return Picks(t.numeric("rlnCoordinateX"), t.numeric("rlnCoordinateY"), fom)
    return Picks(np.empty(0), np.empty(0))


def read_picks(project_dir: str | os.PathLike, star_rel: str) -> PickSet:
    """A coordinate list (``coordinate_files``: micrograph → coordinate file) or a particle file."""
    project_dir = Path(project_dir)
    path = project_dir / star_rel
    r = loop_columns(path, "coordinate_files", [MIC, COORDS])
    if r is not None and MIC in r[1]:
        mics = r[1][MIC]
        coords = r[1].get(COORDS, [""] * len(mics))
        return PickSet(project_dir, star_rel, list(mics), dict(zip(mics, coords)))
    labels = [MIC, "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit", CLASS]
    r = loop_columns(path, "particles", labels)
    if r is None or MIC not in r[1] or "rlnCoordinateX" not in r[1]:
        raise ValueError(f"{star_rel}: no coordinates")
    t = Table("particles", list(r[1]), r[1])
    x, y = t.numeric("rlnCoordinateX"), t.numeric("rlnCoordinateY")
    fom = t.numeric("rlnAutopickFigureOfMerit") if t.has("rlnAutopickFigureOfMerit") else None
    cls = t.numeric(CLASS).astype(int) if t.has(CLASS) else None
    order: dict[str, list[int]] = {}
    for i, m in enumerate(t.column(MIC)):
        order.setdefault(m, []).append(i)
    per = {m: Picks(x[idx], y[idx], fom[idx] if fom is not None else None, cls[idx] if cls is not None else None)
           for m, idx in ((m, np.array(v)) for m, v in order.items())}
    px = optics_value(path, "rlnMicrographPixelSize") or optics_value(path, "rlnMicrographOriginalPixelSize")
    return PickSet(project_dir, star_rel, list(order), pixel_size=px, classes=class_counts(cls),
                   _from_particles=per)


def class_counts(cls: np.ndarray | None) -> dict[int, int]:
    """class -> number of particles, in class order."""
    if cls is None or not len(cls):
        return {}
    nums, counts = np.unique(cls, return_counts=True)
    return {int(n): int(c) for n, c in zip(nums, counts)}
