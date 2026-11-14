# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Volumes (maps, masks, local-resolution maps) for display: binned copies, sections, projections, levels.

Projections and the 3D surface use a copy binned to at most ``max_size`` voxels per side. A map
of up to ``FULL_IN_MEMORY`` voxels (320³) is also kept whole, so its sections are instant; sections
of bigger ones are read from the file at full resolution when they are shown, and a 512³ map costs
64 MB in memory instead of 512 MB. Files are read, never memory-mapped (see images/mrc.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from cryo_savanna.images.mrc import ImageError, MrcReader
from cryo_savanna.stamps import Cache

AXES = ("z", "y", "x")  # numpy order of an MRC volume
FULL_IN_MEMORY = 32 * 1024 * 1024  # voxels (128 MB as float32)


@dataclass
class Volume:
    path: str
    data: np.ndarray  # binned copy, [z, y, x]
    shape: tuple[int, int, int]  # full size, [z, y, x]
    voxel: float  # Å per voxel of the full map (0 if the header has none)
    factor: int  # full voxels per binned voxel
    mean: float
    std: float
    minimum: float
    maximum: float
    full: np.ndarray | None = None  # the whole map, when small enough (sections without reading the file)

    @property
    def binned_voxel(self) -> float:
        return self.voxel * self.factor

    def level_for_fraction(self, fraction: float) -> float:
        """The density level that encloses this fraction of the box (ChimeraX starts at 1%)."""
        flat = self.data.ravel()
        if flat.size > 2_000_000:
            flat = flat[:: flat.size // 2_000_000]
        return float(np.quantile(flat, 1.0 - fraction))


def bin_volume(vol: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(vol, dtype=np.float32)
    z, y, x = (s // k for s in vol.shape)
    return np.asarray(vol[: z * k, : y * k, : x * k], np.float32).reshape(z, k, y, k, x, k).mean(axis=(1, 3, 5))


def load_volume(path: str | os.PathLike, max_size: int = 192) -> Volume:
    with MrcReader(path) as r:
        if r.nz < 2 or r.path.endswith(".mrcs"):
            raise ImageError(f"{path}: not a volume")
        shape = (r.nz, r.ny, r.nx)
        k = max(1, -(-max(shape) // max_size))
        full = r.planes(0, r.nz) if r.nz * r.ny * r.nx <= FULL_IN_MEMORY else None
        if full is not None:
            small = bin_volume(full, k)
        else:  # a slab of k sections at a time: never the whole map in memory
            z, y, x = (s // k for s in shape)
            small = np.empty((z, y, x), np.float32)
            for i in range(z):
                slab = r.planes(i * k, (i + 1) * k)[:, : y * k, : x * k]
                small[i] = slab.reshape(k, y, k, x, k).mean(axis=(0, 2, 4))
        voxel = r.voxel
    flat = small.ravel()
    return Volume(str(path), small, shape, voxel, k, float(flat.mean()), float(flat.std()) or 1.0,
                  float(flat.min()), float(flat.max()), full)


YZ_SLAB = 32  # columns kept per read of a YZ section: the next steps of the slider need no read
YZ_SLAB_BYTES = 32 * 1024 * 1024  # but at most this much per slab (fewer columns for very big maps)
_slabs = Cache(2)  # (path, stamp, first column, columns) -> [z, y, columns]


def section(path: str | os.PathLike, axis: int, index: int) -> np.ndarray:
    """One plane of the full map: axis 0 = XY plane at z, 1 = XZ plane at y, 2 = YZ plane at x.

    Returned so that it reads as RELION/ChimeraX show it: rows = the slower of the two axes.
    A YZ plane needs every XY plane of the file, so up to YZ_SLAB columns around it are kept (two
    such slabs of at most YZ_SLAB_BYTES, per file and its stamp): moving through the map reads it
    once per slab, not at every step.
    """
    with MrcReader(path) as r:
        if r.nz < 2:
            raise ImageError(f"{path}: not a volume")
        n = (r.nz, r.ny, r.nx)[axis]
        index = int(np.clip(index, 0, n - 1))
        if axis == 0:
            return r.section(index)
        if axis == 1:
            return np.stack([r.rows(z, index, index + 1)[0] for z in range(r.nz)])
        st = os.fstat(r.fh.fileno())
        width = max(1, min(YZ_SLAB, YZ_SLAB_BYTES // (r.nz * r.ny * 4)))
        x0 = index // width * width
        key = (os.path.realpath(r.path), st.st_mtime_ns, st.st_size, x0, width)
        slab = _slabs.get(key)
        if slab is None:
            slab = np.stack([r.section(z)[:, x0:x0 + width] for z in range(r.nz)])
            _slabs.put(key, slab)
        return np.ascontiguousarray(slab[:, :, index - x0])


def projection(vol: np.ndarray, axis: int) -> np.ndarray:
    return vol.sum(axis=axis, dtype=np.float32)


def same_grid(a: Volume, b: Volume) -> bool:
    return a.shape == b.shape


def outline(mask_plane: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Boolean edge of a mask plane: inside pixels with an outside neighbour."""
    inside = mask_plane > level
    edge = np.zeros_like(inside)
    edge[1:, :] |= inside[1:, :] & ~inside[:-1, :]
    edge[:-1, :] |= inside[:-1, :] & ~inside[1:, :]
    edge[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    edge[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    return edge


# ---------------------------------------------------------------------------- colours

# blue (best resolution) → cyan → green → yellow → red (worst), as local-resolution maps are coloured
_STOPS = np.array([[0.19, 0.25, 0.85], [0.1, 0.75, 0.95], [0.2, 0.8, 0.3], [0.98, 0.85, 0.15], [0.85, 0.15, 0.1]])


def colormap(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """RGB in 0..1 for values from lo (blue) to hi (red)."""
    t = np.clip((np.asarray(values, np.float32) - lo) / ((hi - lo) or 1.0), 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(t.astype(int), len(_STOPS) - 2)
    f = (t - i)[..., None]
    return _STOPS[i] * (1 - f) + _STOPS[i + 1] * f


def resolution_range(locres: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Colour range of a local-resolution map: 2nd–98th percentile of the values inside the mask
    (RELION writes the box outside the mask at the worst resolution)."""
    v = locres[mask > 0.5] if mask is not None and mask.shape == locres.shape else locres.ravel()
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0.0, 1.0
    if v.size > 2_000_000:
        v = v[:: v.size // 2_000_000]
    lo, hi = np.quantile(v, [0.02, 0.98])
    return float(lo), float(hi if hi > lo else lo + 1)
