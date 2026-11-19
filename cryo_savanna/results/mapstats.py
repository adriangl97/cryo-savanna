# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Figures computed from maps where RELION writes none: the FSC of two half maps (tomography's
Reconstruct particle) and the distribution of local resolution inside the mask (LocalRes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryo_savanna.images.mrc import MrcReader
from cryo_savanna.results.curves import Curve, Figure
from cryo_savanna.stamps import Cache, stamp

FSC_MAX_BOX = 400  # larger half maps are not Fourier transformed here (memory, time)
LOCRES_SAMPLE = 4_000_000  # voxels kept for the histogram of a large map
RESMAP_OUTSIDE = 100.0  # ResMap marks the voxels outside its mask with this value


def _read(path: Path, stride: int = 1) -> tuple[np.ndarray, float]:
    with MrcReader(path) as r:
        if r.nz < 2:
            raise ValueError(f"{path}: not a volume")
        if stride == 1:
            return np.asarray(r.planes(0, r.nz), np.float32), r.voxel
        planes = [np.asarray(r.planes(z, z + 1)[0, ::stride, ::stride], np.float32) for z in range(0, r.nz, stride)]
        return np.stack(planes), r.voxel


def _box(path: Path) -> tuple[int, int, int]:
    with MrcReader(path) as r:
        return r.nz, r.ny, r.nx


def _rfft_map(path: Path) -> tuple[np.ndarray, float]:
    """(np.fft.rfftn of a map, its voxel size), read and transformed one section at a time, then along z one
    row at a time: the transform and a section in memory. numpy's rfftn of the whole map, read whole, took
    about seven times the map (1.8 GB for 400³ in float32) and twice as long."""
    with MrcReader(path) as r:
        if r.nz < 2:
            raise ValueError(f"{path}: not a volume")
        f = np.empty((r.nz, r.ny, r.nx // 2 + 1), np.complex64)  # float32 sections: what rfftn gives them
        for z in range(r.nz):
            f[z] = np.fft.rfft2(r.section(z))
        voxel = r.voxel
    for y in range(f.shape[1]):
        f[:, y, :] = np.fft.fft(f[:, y, :], axis=0)
    return f, voxel


# ---------------------------------------------------------------------------- FSC

def fsc(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FSC per Fourier shell of two maps on the same cubic grid: (shell, FSC), shell 0 … box/2."""
    if a.shape != b.shape:
        raise ValueError(f"the half maps differ in size: {a.shape} and {b.shape}")
    return fsc_of_transforms(np.fft.rfftn(a), np.fft.rfftn(b), a.shape)


def fsc_of_transforms(fa: np.ndarray, fb: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """``fsc`` of the maps whose rfftn are ``fa`` and ``fb`` (maps of this ``shape``).

    Added up one plane of the transforms at a time: the shell of every Fourier voxel and the products, as
    whole arrays (and bincount's float64 copies of them), took several times the two transforms' memory
    (2.3 GB for two 400³ half maps)."""
    n = min(shape)
    top = n // 2
    rest = [np.fft.fftfreq(s, 1.0 / s) for s in shape[1:-1]] + [np.fft.rfftfreq(shape[-1], 1.0 / shape[-1])]
    r2 = sum(k ** 2 for k in np.meshgrid(*rest, indexing="ij", sparse=True))  # one plane: |k|² but for the first axis
    cross, pa, pb = np.zeros(top + 1), np.zeros(top + 1), np.zeros(top + 1)
    for kz, za, zb in zip(np.fft.fftfreq(shape[0], 1.0 / shape[0]), fa, fb):
        shell = np.rint(np.sqrt(r2 + kz * kz)).astype(np.intp).ravel()
        keep = shell <= top
        shell, za, zb = shell[keep], za.ravel()[keep], zb.ravel()[keep]
        cross += np.bincount(shell, za.real * zb.real + za.imag * zb.imag, minlength=top + 1)
        pa += np.bincount(shell, za.real ** 2 + za.imag ** 2, minlength=top + 1)
        pb += np.bincount(shell, zb.real ** 2 + zb.imag ** 2, minlength=top + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = cross / np.sqrt(pa * pb)
    return np.arange(top + 1), np.nan_to_num(curve)


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """The first x (after the origin) where y falls below level, interpolated linearly."""
    for i in range(1, len(y)):
        if y[i] < level:
            if y[i - 1] == y[i]:
                return float(x[i])
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            return float(x[i - 1] + f * (x[i] - x[i - 1]))
    return None


_fsc_cache = Cache(8)  # (half 1, half 2, stamps) -> Figure


def halfmap_fsc(half1: Path, half2: Path) -> Figure | None:
    """The FSC between two half maps (unmasked), with its 0.143 resolution."""
    key = (str(half1), str(half2), stamp(half1), stamp(half2))
    if None in key[2:]:
        return None
    hit = _fsc_cache.get(key)
    if hit is not None:
        return hit
    shape = _box(half1)
    if max(shape) > FSC_MAX_BOX:
        raise ValueError(f"{half1.name}: box larger than {FSC_MAX_BOX} voxels, FSC not computed here")
    if _box(half2) != shape:
        raise ValueError(f"the half maps differ in size: {shape} and {_box(half2)}")
    fa, voxel = _rfft_map(half1)
    fb, _ = _rfft_map(half2)
    shells, curve = fsc_of_transforms(fa, fb, shape)
    n = min(shape)
    if not voxel:
        voxel = 1.0
    x = shells / (n * voxel)  # 1/Å
    fig = Figure("fsc", "FSC between the half maps (unmasked)", "Resolution (Å)", "FSC", x_resolution=True,
                 hlines=[(0.143, "0.143"), (0.5, "0.5")])
    fig.curves.append(Curve("Unmasked", x[1:], curve[1:]))
    res = crossing(x, curve, 0.143)
    if res:
        fig.vlines.append((res, f"{1 / res:.2f} Å"))
        fig.notes.append(f"0.143 at {1 / res:.2f} Å (unmasked: a mask gives a better one)")
    else:
        fig.notes.append("above 0.143 up to Nyquist")
    _fsc_cache.put(key, fig)
    return fig


# ---------------------------------------------------------------------------- local resolution

@dataclass
class LocresStats:
    values: np.ndarray  # local resolution (Å) of the voxels inside the mask (a sample of them)
    median: float
    low: float  # 5th percentile (best)
    high: float  # 95th percentile (worst)
    masked: bool

    def text(self) -> str:
        return f"local res {self.median:.1f} Å (median; {self.low:.1f}–{self.high:.1f} Å)"


_locres_cache = Cache(8)


def locres_stats(locres: Path, mask: Path | None = None) -> LocresStats | None:
    """Local resolution inside the mask (or, without one, wherever the map has a finite value that is not
    ResMap's outside mark), read every ``stride`` voxel for large maps."""
    key = (str(locres), stamp(locres), str(mask) if mask else "", stamp(mask) if mask else None)
    if key[1] is None:
        return None
    hit = _locres_cache.get(key)
    if hit is not None:
        return hit
    shape = _box(locres)
    stride = 1
    while np.prod([-(-s // stride) for s in shape]) > LOCRES_SAMPLE:
        stride += 1
    v, _ = _read(locres, stride)
    inside = np.isfinite(v) & (v > 0) & (v < RESMAP_OUTSIDE - 1e-3)
    masked = False
    if mask is not None and mask.is_file() and _box(mask) == shape:
        m, _ = _read(mask, stride)
        inside &= m > 0.5
        masked = True
    vals = v[inside]
    if not vals.size:
        return None
    lo, med, hi = np.quantile(vals, [0.05, 0.5, 0.95])
    out = LocresStats(vals, float(med), float(lo), float(hi), masked)
    _locres_cache.put(key, out)
    return out


def locres_figure(stats: LocresStats, bins: int = 60) -> Figure:
    counts, edges = np.histogram(stats.values, bins=bins)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2).astype(float)
    fig = Figure("locres", "Local resolution" + (" inside the mask" if stats.masked else ""),
                 "Local resolution (Å)", "Voxels")
    fig.curves.append(Curve("Voxels", x, y))
    fig.vlines.append((stats.median, f"median {stats.median:.2f} Å"))
    fig.notes.append(f"5–95%: {stats.low:.2f}–{stats.high:.2f} Å")
    return fig
