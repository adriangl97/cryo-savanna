# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bayesian polishing (relion_motion_refine): what a Polish job wrote, for its views and its card.

- Training (``do_param_optim``): ``opt_params_all_groups.txt`` (or ``opt_params_group_<name>.txt`` per
  optics group), one line ``s_vel s_div s_acc``: the parameters to polish with (``--s_vel``, ...).
- Polishing: ``bfactors.star`` (``data_perframe_bfactors``: per movie frame the B-factor and the scale,
  the intercept of the Guinier fit, used to weight the frames), ``shiny.star``, and per micrograph
  ``<micrograph>_tracks.star``: ``data_general`` (``rlnParticleNumber``), then one table per particle
  (``data_0``, ``data_1``, ...; the particles of that micrograph in the order of the particle file)
  with its shift in each frame (``rlnOriginXAngst``/``YAngst``). The file lives in the job directory
  where the micrograph is in its MotionCorr job (``MotionCorr/job002/Movies/m1.mrc`` ->
  ``Polish/job015/Movies/m1_tracks.star``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryo_savanna.project.star import StarError, read_star
from cryo_savanna.results.curves import Curve, Figure
from cryo_savanna.results.tables import Table
from cryo_savanna.stamps import Cache, stamp

PARAMS = "opt_params_all_groups.txt"
BFACTORS = "bfactors.star"
FRAME, BFACTOR, SCALE = "rlnMovieFrameNumber", "rlnBfactorUsedForSharpening", "rlnFittedInterceptGuinierPlot"
_JOB_PREFIX = re.compile(r"^(?:.*/)?job\d+/")


@dataclass
class PolishParams:
    """Trained (or given) motion parameters: velocity σ (Å/dose), divergence σ (Å), acceleration σ (Å/dose)."""

    s_vel: float
    s_div: float
    s_acc: float
    source: str = ""

    def text(self) -> str:
        acc = f"{self.s_acc:g}" if self.s_acc > 0 else "off"
        return f"σ vel {self.s_vel:g} · σ div {self.s_div:g} · σ acc {acc}"

    def flags(self) -> str:
        return f"--s_vel {self.s_vel:g} --s_div {self.s_div:g} --s_acc {self.s_acc:g}"


def read_params(path: Path) -> PolishParams | None:
    try:
        words = path.read_text().split()
        vel, div, acc = (float(w) for w in words[:3])
    except (OSError, ValueError):
        return None
    return PolishParams(vel, div, acc, path.name)


def trained_params(job_dir: Path) -> list[PolishParams]:
    """What a training job found: for all optics groups together, else one per optics group."""
    p = read_params(job_dir / PARAMS)
    if p is not None:
        return [p]
    out = []
    for f in sorted(job_dir.glob("opt_params_group_*.txt")):
        q = read_params(f)
        if q is not None:
            q.source = f.stem.removeprefix("opt_params_group_")
            out.append(q)
    return out


def used_params(job_dir: Path, options: dict[str, str], project_dir: Path | None = None) -> PolishParams | None:
    """The parameters a polishing job used: its own (``do_own_params``), else the training job's file."""
    if options.get("do_own_params") == "Yes":
        try:
            return PolishParams(float(options["sigma_vel"]), float(options["sigma_div"]),
                                float(options["sigma_acc"]), "job options")
        except (KeyError, ValueError):
            return None
    rel = options.get("opt_params", "")
    if rel and project_dir is not None:
        p = read_params(project_dir / rel)
        if p is not None:
            p.source = rel
        return p
    return None


def is_training(options: dict[str, str]) -> bool:
    return options.get("do_param_optim") == "Yes"


# ---------------------------------------------------------------------------- per-frame weights

def polish_figures(job_dir: Path) -> list[Figure]:
    """B-factor and scale of each movie frame (RELION's bfactors.eps and scalefactors.eps)."""
    path = job_dir / BFACTORS
    try:
        b = read_star(path, ["perframe_bfactors"]).get("perframe_bfactors")
    except (OSError, StarError):
        return []
    if b is None or not b.rows or not b.has(BFACTOR):
        return []
    t = Table.from_block(b)
    frames = t.numbers(FRAME) + 1 if t.has(FRAME) else np.arange(1, len(t) + 1)  # RELION counts from 0
    out = []
    bf = t.numbers(BFACTOR)
    fig = Figure("bfactors", "B-factor per frame", "Movie frame", "B-factor (Å²)")
    fig.curves.append(Curve("B-factor", frames, bf, symbols=True))
    ok = np.isfinite(bf)
    if ok.any():
        best = int(np.nanargmax(bf))
        fig.notes.append(f"highest {bf[best]:.0f} Å² at frame {frames[best]:g}, lowest {np.nanmin(bf):.0f} Å²")
    out.append(fig)
    if t.has(SCALE):
        fig = Figure("scale", "Scale per frame", "Movie frame", "Scale (Guinier intercept)")
        fig.curves.append(Curve("Scale", frames, t.numbers(SCALE), symbols=True))
        out.append(fig)
    return out


# ---------------------------------------------------------------------------- particle tracks

def tracks_file(job: str, micrograph: str) -> str:
    """Where a polishing job wrote the tracks of a micrograph's particles (relative to the project)."""
    post = _JOB_PREFIX.sub("", micrograph, count=1)
    return job + str(Path(post).with_suffix("")) + "_tracks.star"


_tracks = Cache(64)  # (file, stamp) -> tracks


def read_tracks(path: Path) -> list[np.ndarray]:
    """Each particle's shift in every frame, [frames, 2] in Å, in the order of the particle file."""
    st = stamp(path)
    if st is None:
        return []
    key = (str(path), st)
    hit = _tracks.get(key)
    if hit is not None:
        return hit
    blocks = read_star(path)
    out = []
    for name, b in blocks.items():
        if name == "general" or not b.is_loop or not b.has("rlnOriginXAngst"):
            continue
        t = Table.from_block(b)
        out.append(np.column_stack([t.numbers("rlnOriginXAngst"), t.numbers("rlnOriginYAngst")]))
    _tracks.put(key, out)
    return out


def track_lines(x: np.ndarray, y: np.ndarray, tracks: list[np.ndarray], pixel_size: float,
                scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Polylines of the tracks at each particle's position, ``scale`` times their length (in micrograph
    pixels), as RELION's per-micrograph plots; NaN between particles."""
    n = min(len(x), len(tracks))
    if n == 0 or not pixel_size:
        return np.empty(0), np.empty(0)
    xs, ys = [], []
    for i in range(n):
        tr = tracks[i] * (scale / pixel_size)  # around the particle's position, as RELION draws them
        xs.append(np.append(x[i] + tr[:, 0], np.nan))
        ys.append(np.append(y[i] + tr[:, 1], np.nan))
    return np.concatenate(xs), np.concatenate(ys)


def mean_track_length(tracks: list[np.ndarray]) -> float | None:
    """Mean path length (Å) of the tracks."""
    lengths = [float(np.sum(np.hypot(*np.diff(t, axis=0).T))) for t in tracks if len(t) > 1]
    return float(np.mean(lengths)) if lengths else None
