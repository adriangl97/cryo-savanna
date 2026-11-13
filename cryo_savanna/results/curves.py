# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Curves for the plots: FSC, Guinier, convergence per iteration, motion tracks, angular distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryo_savanna.project.star import StarError, loop_columns, read_star, read_star_head
from cryo_savanna.results.classes import iterations
from cryo_savanna.results.micrographs import motion_track
from cryo_savanna.results.tables import Table
from cryo_savanna.stamps import Cache, stamp


@dataclass
class Curve:
    label: str
    x: np.ndarray
    y: np.ndarray
    dashed: bool = False
    symbols: bool = False


@dataclass
class Figure:
    key: str
    title: str
    xlabel: str
    ylabel: str
    curves: list[Curve] = field(default_factory=list)
    hlines: list[tuple[float, str]] = field(default_factory=list)
    vlines: list[tuple[float, str]] = field(default_factory=list)
    x_resolution: bool = False  # x is 1/Å: the axis is labelled in Å
    notes: list[str] = field(default_factory=list)


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------- PostProcess

FSC_CURVES = [
    ("rlnFourierShellCorrelationCorrected", "Corrected", False),
    ("rlnFourierShellCorrelationUnmaskedMaps", "Unmasked maps", True),
    ("rlnFourierShellCorrelationMaskedMaps", "Masked maps", True),
    ("rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps", "Phase-randomised masked maps", True),
]
GUINIER_CURVES = [
    ("rlnLogAmplitudesOriginal", "Original"), ("rlnLogAmplitudesMTFCorrected", "MTF-corrected"),
    ("rlnLogAmplitudesWeighted", "FSC-weighted"), ("rlnLogAmplitudesSharpened", "Sharpened"),
    ("rlnLogAmplitudesIntercept", "Intercept"),
]


def postprocess_figures(path: Path) -> list[Figure]:
    blocks = read_star(path, ["general", "fsc", "guinier"])
    gen = blocks.get("general")
    out = []
    fsc = blocks.get("fsc")
    if fsc is not None and fsc.rows and fsc.has("rlnResolution"):
        t = Table.from_block(fsc)
        x = t.numeric("rlnResolution")
        fig = Figure("fsc", "Fourier shell correlation", "Resolution (Å)", "FSC", x_resolution=True,
                     hlines=[(0.143, "0.143")])
        for label, title, dashed in FSC_CURVES:
            if t.has(label):
                fig.curves.append(Curve(title, x[1:], t.numeric(label)[1:], dashed))
        res = _f(gen.get("rlnFinalResolution")) if gen else None
        if res:
            fig.vlines.append((1 / res, f"{res:.2f} Å"))
            fig.notes.append(f"Final resolution {res:.2f} Å")
        rnd = _f(gen.get("rlnRandomiseFrom")) if gen else None
        if rnd:
            fig.notes.append(f"phases randomised beyond {rnd:.2f} Å")
        out.append(fig)
    gui = blocks.get("guinier")
    if gui is not None and gui.rows and gui.has("rlnResolutionSquared"):
        t = Table.from_block(gui)
        x = t.numeric("rlnResolutionSquared")
        fig = Figure("guinier", "Guinier plot", "Resolution² (1/Å²)", "ln(amplitude)")
        for label, title in GUINIER_CURVES:
            if t.has(label):
                y = t.numeric(label)
                ok = np.isfinite(y) & (y > -99)  # RELION writes -99 where there is nothing
                fig.curves.append(Curve(title, x[ok], y[ok]))
        b = _f(gen.get("rlnBfactorUsedForSharpening")) if gen else None
        slope, icpt = (_f(gen.get("rlnFittedSlopeGuinierPlot")), _f(gen.get("rlnFittedInterceptGuinierPlot"))) \
            if gen else (None, None)
        finite = x[np.isfinite(x)]
        if slope is not None and icpt is not None and len(finite):
            xs = np.array([finite.min(), finite.max()])
            fig.curves.append(Curve("Fit", xs, icpt + slope * xs, dashed=True))
        if b is not None:
            fig.notes.append(f"B-factor used for sharpening {b:.1f} Å²")
        out.append(fig)
    return out


# ---------------------------------------------------------------------------- Refine3D / Class3D

def model_fsc(path: Path) -> Figure | None:
    """Gold-standard FSC between the half maps, from a (half1) model.star of Refine3D or MultiBody."""
    blocks = read_star(path)
    fig = Figure("fsc", "Gold-standard FSC (half maps)", "Resolution (Å)", "FSC", x_resolution=True,
                 hlines=[(0.143, "0.143"), (0.5, "0.5")])
    for name, b in blocks.items():
        if not (name.startswith("model_class_") or name.startswith("model_body_")) or not b.has("rlnGoldStandardFsc"):
            continue
        t = Table.from_block(b)
        y = t.numeric("rlnGoldStandardFsc")
        if not np.any(y[1:]):
            continue
        label = name.replace("model_class_", "Class ").replace("model_body_", "Body ")
        fig.curves.append(Curve(label, t.numeric("rlnResolution")[1:], y[1:]))
    gen = blocks.get("model_general")
    res = _f(gen.get("rlnCurrentResolution")) if gen else None
    if res:
        fig.vlines.append((1 / res, f"{res:.2f} Å"))
        fig.notes.append(f"Current resolution {res:.2f} Å")
    return fig if fig.curves else None


_head_cache = Cache(4000)  # (file, blocks, stamp) -> the blocks: one model.star per iteration of every job looked at


def _cached_head(path: Path, blocks: list[str]):
    st = stamp(path)
    if st is None:
        return {}
    key = (str(path), tuple(blocks), st)
    hit = _head_cache.get(key)
    if hit is None:
        try:
            hit = read_star_head(path, blocks)
        except (OSError, StarError, ValueError):
            return {}
        _head_cache.put(key, hit)
    return hit


def convergence(job_dir: Path) -> list[Figure]:
    """Per-iteration resolution, class occupancy, accuracies and changes of an iterative job."""
    its = iterations(job_dir)
    if not its:
        return []
    xs, res, dist, acc, changes = [], [], {}, {}, {"orient": [], "offset": [], "classes": []}
    n_classes = 0
    for it, model in its:
        h = _cached_head(model, ["model_general", "model_classes"])
        gen, cl = h.get("model_general"), h.get("model_classes")
        if gen is None:
            continue
        xs.append(it)
        res.append(_f(gen.get("rlnCurrentResolution")) or np.nan)
        if cl is not None and cl.rows:
            t = Table.from_block(cl)
            n_classes = max(n_classes, len(t))
            for label, store in (("rlnClassDistribution", dist), ("rlnAccuracyRotations", acc)):
                if t.has(label):
                    for k, v in enumerate(t.numeric(label)):
                        store.setdefault(k + 1, {})[it] = v
        opt = job_dir / model.name.replace("_half1_model.star", "_optimiser.star").replace("_model.star",
                                                                                            "_optimiser.star")
        og = _cached_head(opt, ["optimiser_general"]).get("optimiser_general")
        for key, label in (("orient", "rlnChangesOptimalOrientations"), ("offset", "rlnChangesOptimalOffsets"),
                           ("classes", "rlnChangesOptimalClasses")):
            v = _f(og.get(label)) if og is not None else None
            changes[key].append(np.nan if v is None or v >= 999 else v)
    x = np.array(xs, dtype=float)
    figs = []
    r = np.array(res, dtype=float)
    if np.isfinite(r).any():
        figs.append(Figure("resolution", "Resolution per iteration", "Iteration", "Resolution (Å)",
                           [Curve("Resolution", x, r, symbols=True)]))
    def per_class(store: dict, key: str, title: str, ylabel: str, scale: float = 1.0) -> None:
        if not store or n_classes < 1:
            return
        fig = Figure(key, title, "Iteration", ylabel)
        for k in sorted(store):
            ys = np.array([store[k].get(i, np.nan) for i in xs], dtype=float) * scale
            fig.curves.append(Curve(f"Class {k}", x, ys))
        figs.append(fig)
    per_class(dist, "occupancy", "Class occupancy", "Particles (%)", 100.0)
    if n_classes <= 20:  # one curve per class stops being readable beyond that
        per_class(acc, "accuracy", "Accuracy of rotations", "Degrees")
    ch = Figure("changes", "Changes per iteration", "Iteration", "Change")
    for key, title in (("orient", "Orientations (°)"), ("offset", "Offsets (px)"), ("classes", "Classes (fraction)")):
        y = np.array(changes[key], dtype=float)
        if np.isfinite(y).any():
            ch.curves.append(Curve(title, x, y, symbols=True))
    if ch.curves:
        figs.append(ch)
    return figs


def motion_figure(path: Path, pixel_size: float | None = None) -> Figure | None:
    tr = motion_track(path)
    if tr is None:
        return None
    frames, sx, sy = tr
    unit = "px"
    if pixel_size:
        sx, sy, unit = sx * pixel_size, sy * pixel_size, "Å"
    fig = Figure("motion", f"Motion of {path.stem}", f"Shift X ({unit})", f"Shift Y ({unit})",
                 [Curve("Track", sx - sx[0], sy - sy[0], symbols=True)])
    step = np.hypot(np.diff(sx), np.diff(sy))
    fig.notes.append(f"{len(frames)} frames, total {step.sum():.1f} {unit}")
    return fig


def angular_distribution(data_star: Path, bins: tuple[int, int] = (72, 36)) -> tuple[np.ndarray, int] | None:
    """Histogram of (rot, tilt) of the particles: (counts[tilt, rot], number of particles)."""
    r = loop_columns(data_star, "particles", ["rlnAngleRot", "rlnAngleTilt"])
    if r is None or not {"rlnAngleRot", "rlnAngleTilt"} <= r[1].keys():
        return None
    rot = np.array(r[1]["rlnAngleRot"], dtype=float)
    tilt = np.array(r[1]["rlnAngleTilt"], dtype=float)
    h, _, _ = np.histogram2d(tilt, ((rot + 180) % 360) - 180, bins=(bins[1], bins[0]),
                             range=((0, 180), (-180, 180)))
    return h, len(rot)
