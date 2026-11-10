# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What pytom jobs wrote, read for the viewers (no Qt): the match summary, pytom's job.json, score maps,
the candidates and thresholds of an extraction.

The files are those of cryo_savanna/pytom_runner.py (the runner the jobs run): pytom_match.star (one row
per tomogram: the tomograms.star columns, the score maps and the status), tm/<tomogram>_job.json
(pytom's own), candidates.star / thresholds.star / particles.star of an extraction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna import pytom_runner as P
from cryo_savanna.images.mrc import MrcReader
from cryo_savanna.project.star import loop_columns, read_star
from cryo_savanna.results.tables import Table


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


@dataclass
class MatchSummary:
    source: str
    general: dict[str, str]
    table: Table  # data_global of pytom_match.star

    @property
    def names(self) -> list[str]:
        return self.table.column(P.TOMO_NAME)

    def value(self, i: int, label: str) -> str:
        return self.table.column(label)[i] if self.table.has(label) else ""

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.table.column(P.COL_STATUS) if self.table.has(P.COL_STATUS) else []:
            out[s] = out.get(s, 0) + 1
        return out


def read_match_summary(project_dir: Path, star_rel: str) -> MatchSummary:
    blocks = read_star(Path(project_dir) / star_rel, ["general", "global"])
    g, b = blocks.get("general"), blocks.get("global")
    if b is None or not b.has(P.TOMO_NAME):
        raise ValueError(f"{star_rel}: not a pytom match summary")
    general = {c: str(v) for c, v in zip(g.columns, g.rows[0])} if g is not None and g.rows else {}
    return MatchSummary(star_rel, general, Table.from_block(b, str(star_rel)))


def job_stats(path: Path) -> dict:
    """The numbers of pytom's job.json worth showing (empty when it is not there yet)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {}
    stats = d.get("job_stats") or {}
    out = {"search_space": stats.get("search_space"), "std": stats.get("std"), "variance": stats.get("variance"),
           "n_rotations": d.get("n_rotations"), "voxel_size": d.get("voxel_size"),
           "particle_diameter": d.get("particle_diameter"), "version": d.get("pytom_tm_version_number"),
           "tomo_shape": d.get("tomo_shape"), "template_shape": d.get("template_shape")}
    return {k: v for k, v in out.items() if v is not None}


@dataclass
class ScoreMap:
    """A score map, read once: its maximum projection along z, a sample of its values, its peak."""
    path: str
    shape: tuple[int, int, int]  # z, y, x
    mip: np.ndarray  # [y, x]
    sample: np.ndarray  # values, every few voxels
    peak: float
    peak_at: tuple[int, int, int]  # x, y, z
    mean: float
    std: float


_SCORES: dict[str, tuple[tuple, ScoreMap]] = {}


def read_score_map(path: str | os.PathLike, sample_every: int = 7) -> ScoreMap:
    """Reads the whole map (a section block at a time). Cached by the file's stamp: a map is read once."""
    key = str(path)
    st = _stamp(Path(path))
    hit = _SCORES.get(key)
    if hit is not None and hit[0] == st:
        return hit[1]
    with MrcReader(path) as r:
        nz, ny, nx = r.nz, r.ny, r.nx
        mip = np.full((ny, nx), -np.inf, np.float32)
        samples, peak, peak_at = [], -np.inf, (0, 0, 0)
        total, total2, count = 0.0, 0.0, 0
        step = 8
        for z0 in range(0, nz, step):
            block = np.asarray(r.planes(z0, min(nz, z0 + step)), dtype=np.float32)
            np.maximum(mip, block.max(axis=0), out=mip)
            flat = block.ravel()
            samples.append(flat[::sample_every].copy())
            i = int(np.argmax(flat))
            if flat[i] > peak:
                zz, yy, xx = np.unravel_index(i, block.shape)
                peak, peak_at = float(flat[i]), (int(xx), int(yy), int(z0 + zz))
            total += float(flat.sum(dtype=np.float64))
            total2 += float((flat.astype(np.float64) ** 2).sum())
            count += flat.size
    mean = total / count if count else 0.0
    std = float(np.sqrt(max(total2 / count - mean * mean, 0.0))) if count else 0.0
    out = ScoreMap(key, (nz, ny, nx), mip, np.concatenate(samples) if samples else np.empty(0), peak, peak_at,
                   mean, std)
    _SCORES[key] = (st, out)
    while len(_SCORES) > 8:
        _SCORES.pop(next(iter(_SCORES)))
    return out


def histogram(values: np.ndarray, bins: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres and counts."""
    values = values[np.isfinite(values)]
    if not len(values):
        return np.empty(0), np.empty(0)
    counts, edges = np.histogram(values, bins=bins)
    return (edges[:-1] + edges[1:]) / 2, counts


# ---------------------------------------------------------------------------- extraction

@dataclass
class Selection:
    """The candidates of an extraction job (LCC per tomogram, in the file's order) and their thresholds."""
    job: str
    lcc: dict[str, np.ndarray]
    auto: dict[str, float | None] = field(default_factory=dict)
    threshold: dict[str, float | None] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(list(self.threshold) + list(self.lcc)))

    def selected(self, name: str, threshold: float | None = None) -> np.ndarray:
        lcc = self.lcc.get(name, np.empty(0))
        thr = self.threshold.get(name) if threshold is None else threshold
        return lcc >= thr if thr is not None else np.zeros(len(lcc), bool)

    def counts(self, name: str) -> tuple[int, int]:
        return len(self.lcc.get(name, ())), int(self.selected(name).sum())


def read_selection(project_dir: Path, job: str) -> Selection:
    d = Path(project_dir) / job
    r = loop_columns(d / P.CANDIDATES, "particles", [P.TOMO_NAME, P.LCC])
    if r is None or P.TOMO_NAME not in r[1] or P.LCC not in r[1]:
        raise ValueError(f"{job}{P.CANDIDATES}: no candidates (rlnTomoName, rlnLCCmax)")
    names = np.array(r[1][P.TOMO_NAME])
    lcc = np.array([_float(v) if _float(v) is not None else np.nan for v in r[1][P.LCC]], dtype=float)
    by = {str(n): lcc[names == n] for n in dict.fromkeys(names.tolist())}
    sel = Selection(job, by)
    t = loop_columns(d / P.THRESHOLDS, "thresholds", [P.TOMO_NAME, P.COL_AUTO, P.COL_THRESHOLD])
    if t is not None and P.TOMO_NAME in t[1]:
        cols = t[1]
        for i, n in enumerate(cols[P.TOMO_NAME]):
            sel.auto[n] = _float(cols.get(P.COL_AUTO, [""] * (i + 1))[i])
            sel.threshold[n] = _float(cols.get(P.COL_THRESHOLD, [""] * (i + 1))[i])
    return sel


def save_thresholds(project_dir: Path, job: str, thresholds: dict[str, float | None]) -> dict[str, tuple[int, int]]:
    """particles.star and thresholds.star for these thresholds (the runner's own code writes them, as at the
    end of the job). Run from the project: the paths in the files are relative to it."""
    return P.write_selection(str(Path(project_dir) / job), thresholds)


def plot_files(project_dir: Path, job: str, folder: str = P.PLOTS_DIR) -> list[str]:
    d = Path(project_dir) / job / folder
    try:
        return sorted(f"{job}{folder}/{f.name}" for f in d.iterdir() if f.suffix.lower() == ".svg")
    except OSError:
        return []


def _erfcinv(y: float) -> float:
    """Inverse of math.erfc for 0 < y < 2 (bisection: no scipy needed)."""
    import math
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gaussian_cut_off(std: float | None, search_space: float | None, false_positives: float = 1.0) -> float | None:
    """The score that Gaussian noise of this σ exceeds ``false_positives`` times over the search space
    (Rickgauer et al. 2017, as pytom's automatic cut-off): σ·√2·erfc⁻¹(2·n / search space)."""
    if not std or not search_space or search_space <= 2 * false_positives:
        return None
    return std * np.sqrt(2) * _erfcinv(2 * false_positives / search_space)
