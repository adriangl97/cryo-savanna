# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tomography results for the viewers: tilt-series sets, per-tilt values, tomograms, picked particles.

RELION 5 keeps one table per set (``data_global``: one row per tilt series, with its per-tilt STAR
file and, after reconstruction, its tomograms) and one table per tilt series (one row per tilt
image: tilt angle, dose, defocus, CTF, motion, alignment). Particles carry ``rlnTomoName`` and
coordinates in Å from the tomogram's centre (``rlnCenteredCoordinate[XYZ]Angst``; older files have
``rlnCoordinate[XYZ]`` in unbinned tilt-series pixels from the corner).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna.images.mrc import ImageRef
from cryo_savanna.project.star import loop_columns, read_star
from cryo_savanna.results.curves import Curve, Figure
from cryo_savanna.results.micrographs import CLASS, class_counts
from cryo_savanna.results.tables import Table

TOMO_NAME = "rlnTomoName"
TOMOGRAM_COLUMNS = [  # (column, title)
    ("rlnTomoReconstructedTomogramDenoised", "Denoised"),
    ("rlnTomoReconstructedTomogram", "Reconstructed"),
    ("rlnTomoReconstructedTomogramHalf1", "Half 1"),
    ("rlnTomoReconstructedTomogramHalf2", "Half 2"),
]
TILT_COLUMNS = [  # per-tilt values worth a column
    ("rlnTomoNominalStageTiltAngle", "Tilt (°)"), ("rlnMicrographPreExposure", "Dose (e/Å²)"),
    ("rlnDefocusU", "Defocus U (Å)"), ("rlnDefocusV", "Defocus V (Å)"), ("rlnCtfMaxResolution", "Max. res. (Å)"),
    ("rlnCtfFigureOfMerit", "FOM"), ("rlnAccumMotionTotal", "Motion (Å)"),
    ("rlnTomoXShiftAngst", "X shift (Å)"), ("rlnTomoYShiftAngst", "Y shift (Å)"),
    ("rlnTomoZRot", "Tilt axis (°)"), ("rlnTomoYTilt", "Fitted tilt (°)"),
]
SERIES_COLUMNS = [("rlnTomoSizeX", "X"), ("rlnTomoSizeY", "Y"), ("rlnTomoSizeZ", "Z"),
                  ("rlnTomoTomogramBinning", "Binning"), ("rlnTomoTiltSeriesPixelSize", "Pixel (Å)")]


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def columns_of(path: str | os.PathLike, block: str) -> list[str] | None:
    """Column labels of a loop block, reading only up to its first row."""
    target = f"data_{block}"
    cols: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("data_") and raw.split()[0] == target:
                break
        else:
            return None
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#" or line.startswith("loop_"):
                continue
            if line[0] == "_":
                cols.append(line[1:].split()[0])
                continue
            break
    return cols


# ---------------------------------------------------------------------------- tilt series and tomograms

@dataclass
class TiltSeriesSet:
    project_dir: Path
    source: str  # the set's STAR file, relative to the project
    table: Table  # data_global
    _tilts: dict[str, tuple[tuple, Table]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def names(self) -> list[str]:
        return self.table.column(TOMO_NAME)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def _value(self, i: int, label: str) -> float | None:
        v = self.table.numeric(label) if self.table.has(label) else None
        return float(v[i]) if v is not None and np.isfinite(v[i]) else None

    def pixel_size(self, i: int) -> float | None:
        """Pixel size of the tilt images (Å)."""
        return self._value(i, "rlnTomoTiltSeriesPixelSize") or self._value(i, "rlnMicrographOriginalPixelSize")

    def binning(self, i: int) -> float | None:
        return self._value(i, "rlnTomoTomogramBinning")

    def tomogram_kinds(self) -> list[tuple[str, str]]:
        return [(c, t) for c, t in TOMOGRAM_COLUMNS if self.table.has(c)]

    def tomogram(self, i: int, column: str | None = None) -> ImageRef | None:
        kinds = self.tomogram_kinds()
        column = column or (kinds[0][0] if kinds else None)
        if column is None or not self.table.has(column):
            return None
        v = self.table.column(column)[i]
        return ImageRef.parse(v).resolve(self.project_dir) if v and v != "None" else None

    def tilts(self, i: int) -> Table:
        """The per-tilt table of tilt series i (re-read when its file changes)."""
        name = self.names[i]
        if not self.table.has("rlnTomoTiltSeriesStarFile"):
            raise ValueError(f"{self.source}: no per-tilt STAR files (rlnTomoTiltSeriesStarFile)")
        path = self.project_dir / self.table.column("rlnTomoTiltSeriesStarFile")[i]
        st = _stamp(path)
        hit = self._tilts.get(name)
        if hit is not None and hit[0] == st:
            return hit[1]
        blocks = read_star(path)
        b = blocks.get(name)  # not `or`: a block without rows is falsy
        if b is None:
            b = next((b for b in blocks.values() if b.is_loop), None)
        if b is None:
            raise ValueError(f"{path}: no tilt table")
        t = Table.from_block(b, str(path))
        self._tilts[name] = (st, t)
        return t

    def tilt_image(self, tilts: Table, row: int) -> ImageRef | None:
        for col in ("rlnMicrographName", "rlnMicrographMovieName"):
            if tilts.has(col):
                return ImageRef.parse(tilts.column(col)[row]).resolve(self.project_dir)
        return None

    def ctf_image(self, tilts: Table, row: int) -> ImageRef | None:
        if not tilts.has("rlnCtfImage"):
            return None
        return ImageRef.parse(tilts.column("rlnCtfImage")[row]).resolve(self.project_dir)


def read_tilt_series_set(project_dir: str | os.PathLike, star_rel: str) -> TiltSeriesSet:
    project_dir = Path(project_dir)
    b = read_star(project_dir / star_rel, ["global"]).get("global")
    if b is None or not b.has(TOMO_NAME):
        raise ValueError(f"{star_rel}: no tilt series (data_global with rlnTomoName)")
    return TiltSeriesSet(project_dir, star_rel, Table.from_block(b, str(project_dir / star_rel)))


def has_tomograms(path: str | os.PathLike) -> bool:
    cols = columns_of(path, "global") or []
    return any(c in cols for c, _ in TOMOGRAM_COLUMNS)


def tilt_figures(tilts: Table, name: str = "") -> list[Figure]:
    """Per-tilt values against the tilt angle."""
    if not tilts.has("rlnTomoNominalStageTiltAngle"):
        return []
    x = tilts.numeric("rlnTomoNominalStageTiltAngle")
    order = np.argsort(x)
    x = x[order]
    figs = []
    groups = [
        ("defocus", "Defocus", "Å", [("rlnDefocusU", "U"), ("rlnDefocusV", "V")]),
        ("ctf", "CTF max. resolution", "Å", [("rlnCtfMaxResolution", "Max. resolution")]),
        ("motion", "Motion", "Å", [("rlnAccumMotionTotal", "Total"), ("rlnAccumMotionEarly", "Early"),
                                   ("rlnAccumMotionLate", "Late")]),
        ("shifts", "Alignment shifts", "Å", [("rlnTomoXShiftAngst", "X"), ("rlnTomoYShiftAngst", "Y")]),
        ("axis", "Tilt axis angle", "°", [("rlnTomoZRot", "Tilt axis")]),
        ("dose", "Pre-exposure", "e/Å²", [("rlnMicrographPreExposure", "Dose")]),
    ]
    for key, title, unit, cols in groups:
        fig = Figure(key, title + (f" — {name}" if name else ""), "Tilt angle (°)", f"{title} ({unit})")
        for col, label in cols:
            if tilts.has(col):
                y = tilts.numeric(col)
                if y is not None and np.isfinite(y).any():
                    fig.curves.append(Curve(label, x, y[order], symbols=True))
        if fig.curves:
            figs.append(fig)
    return figs


def series_summary(tilts: Table) -> dict[str, str]:
    out = {"Tilts": str(len(tilts))}
    if tilts.has("rlnTomoNominalStageTiltAngle"):
        a = tilts.numeric("rlnTomoNominalStageTiltAngle")
        out["Range (°)"] = f"{np.nanmin(a):.0f} … {np.nanmax(a):.0f}"
    if tilts.has("rlnCtfMaxResolution"):
        out["Max. res. (Å)"] = f"{np.nanmedian(tilts.numeric('rlnCtfMaxResolution')):.1f}"
    if tilts.has("rlnDefocusU"):
        out["Defocus (µm)"] = f"{np.nanmedian(tilts.numeric('rlnDefocusU')) / 1e4:.2f}"
    return out


# ---------------------------------------------------------------------------- particles

@dataclass
class TomoPicks:
    source: str
    by_tomo: dict[str, np.ndarray]  # tomogram -> [n, 3] (x, y, z)
    centered_angst: bool = True  # else unbinned tilt-series pixels from the corner
    cls_by_tomo: dict[str, np.ndarray] | None = None  # tomogram -> rlnClassNumber of its particles
    classes: dict[int, int] = field(default_factory=dict)  # class -> particles

    def count(self, name: str, classes: set[int] | None = None) -> int:
        return int(self.in_classes(name, classes).sum())

    def class_numbers(self, name: str) -> np.ndarray | None:
        return None if self.cls_by_tomo is None else self.cls_by_tomo.get(name, np.empty(0, int))

    def in_classes(self, name: str, classes: set[int] | None) -> np.ndarray:
        """Which of a tomogram's particles belong to one of the classes (all of them for None)."""
        n = len(self.by_tomo.get(name, ()))
        cls = self.class_numbers(name)
        if classes is None or cls is None:
            return np.ones(n, bool)
        return np.isin(cls, list(classes))

    def voxels(self, name: str, voxel: float, shape: tuple[int, int, int], binning: float | None = None) -> np.ndarray:
        """Coordinates of a tomogram's particles in its voxels: [n, 3] (x, y, z)."""
        xyz = self.by_tomo.get(name)
        if xyz is None or not len(xyz):
            return np.empty((0, 3))
        nz, ny, nx = shape
        if self.centered_angst:
            return xyz / voxel + np.array([nx / 2, ny / 2, nz / 2])
        return xyz / (binning or 1.0)


def is_tomo_particles(path: str | os.PathLike) -> bool:
    try:
        cols = columns_of(path, "particles")
    except OSError:
        return False
    return bool(cols) and TOMO_NAME in cols


def read_tomo_picks(project_dir: str | os.PathLike, star_rel: str) -> TomoPicks:
    path = Path(project_dir) / star_rel
    centred = ["rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"]
    corner = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    r = loop_columns(path, "particles", [TOMO_NAME] + centred + corner + [CLASS])
    if r is None or TOMO_NAME not in r[1]:
        raise ValueError(f"{star_rel}: no tomogram particles (rlnTomoName)")
    t = Table("particles", list(r[1]), r[1])
    cols = centred if all(t.has(c) for c in centred) else corner if all(t.has(c) for c in corner) else None
    if cols is None:
        raise ValueError(f"{star_rel}: no particle coordinates")
    xyz = np.stack([t.numeric(c) for c in cols], axis=1)
    names = np.array(t.column(TOMO_NAME))
    by = {str(n): xyz[names == n] for n in dict.fromkeys(names.tolist())}
    cls = t.numeric(CLASS).astype(int) if t.has(CLASS) else None
    cls_by = {k: cls[names == k] for k in by} if cls is not None else None
    return TomoPicks(star_rel, by, cols is centred, cls_by, class_counts(cls))


def tomograms_for_particles(project_dir: Path, particles_rel: str, options: dict[str, str]) -> str | None:
    """The tomogram set that particles were picked in: from the job's optimisation set, else its input."""
    job_dir = (project_dir / particles_rel).parent
    candidates = [job_dir / n for n in ("optimisation_set.star", "run_optimisation_set.star")]
    for key in ("in_optimisation", "in_optimisation_set"):
        if options.get(key):
            candidates.append(project_dir / options[key])
    for opt in candidates:
        if not opt.is_file():
            continue
        try:
            b = next(iter(read_star(opt).values()), None)
        except (OSError, ValueError):
            continue
        rel = b.get("rlnTomoTomogramsFile") if b is not None else None
        if rel and (project_dir / rel).is_file() and has_tomograms(project_dir / rel):
            return rel
    for key in ("in_tomoset", "in_tomograms"):
        rel = options.get(key, "")
        if rel and (project_dir / rel).is_file() and has_tomograms(project_dir / rel):
            return rel
    return None
