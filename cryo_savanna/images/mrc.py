# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Reading images for display: RELION image names, MRC headers and slices, movies, downsampling, contrast.

Files are read with plain seeks and reads, so a slice or a region of a 4k×4k micrograph reads
only those bytes. They are never memory-mapped: when RELION truncates and rewrites a file that
is mapped (a running job writing its map or micrographs again), touching the old pages kills the
whole process (SIGBUS); a plain read of a short file is just an error. Everything returns float32 arrays indexed [y, x] (row 0 = first row in the file, which is the
top of the image as RELION's display shows it).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOVIE_SUFFIXES = (".tif", ".tiff", ".eer", ".mrc", ".mrcs")
_INDEXED = re.compile(r"^(\d+)@(.+)$")


class ImageError(OSError):
    pass


@dataclass(frozen=True)
class ImageRef:
    """One image: a file and, for stacks, a 0-based index (RELION writes "000012@stack.mrcs", 1-based)."""

    path: str
    index: int | None = None

    @classmethod
    def parse(cls, name: str) -> "ImageRef":
        name = name.strip()
        m = _INDEXED.match(name)
        index = int(m.group(1)) - 1 if m else None
        path = m.group(2) if m else name
        for suffix in (":mrcs", ":mrc"):  # "file.ctf:mrc": read as MRC whatever the extension
            if path.endswith(suffix):
                path = path[: -len(suffix)]
        return cls(path, index)

    def resolve(self, project_dir: str | os.PathLike) -> "ImageRef":
        p = Path(self.path)
        return self if p.is_absolute() else ImageRef(str(Path(project_dir) / p), self.index)

    def __str__(self) -> str:
        return self.path if self.index is None else f"{self.index + 1:06d}@{self.path}"


@dataclass(frozen=True)
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    pixel_size: float  # Å, from the header (0 if not set)
    is_stack: bool  # images (.mrcs or ispg 0), not a volume

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx


class MrcReader:
    """An open MRC file: the header, then sections, rows or whole planes read on demand."""

    def __init__(self, path: str | os.PathLike):
        import mrcfile
        import mrcfile.utils
        self.path = str(path)
        try:
            with mrcfile.open(self.path, header_only=True, permissive=True) as m:
                h = m.header
                self.nx, self.ny, self.nz = int(h.nx), int(h.ny), int(h.nz)
                self.mode, self.ispg = int(h.mode), int(h.ispg)
                self.dtype = mrcfile.utils.data_dtype_from_header(h)
                self.offset = int(h.nbytes) + int(h.nsymbt)
                self.voxel = float(m.voxel_size.x)
            self.fh = open(self.path, "rb")
        except (OSError, ValueError, AttributeError) as e:
            raise ImageError(f"{self.path}: {e}") from e
        if min(self.nx, self.ny, self.nz) < 1:
            self.fh.close()
            raise ImageError(f"{self.path}: no data")

    def __enter__(self) -> "MrcReader":
        return self

    def __exit__(self, *exc) -> None:
        self.fh.close()

    @property
    def is_stack(self) -> bool:
        return self.path.endswith(".mrcs") or (self.ispg == 0 and self.nz > 1)

    def header(self) -> MrcHeader:
        return MrcHeader(self.nx, self.ny, self.nz, self.mode, self.voxel, self.is_stack)

    def read(self, start: int, count: int) -> np.ndarray:
        """``count`` values from value number ``start`` on, as float32."""
        a = np.empty(count, self.dtype)
        self.fh.seek(self.offset + start * self.dtype.itemsize)
        n = self.fh.readinto(memoryview(a).cast("B"))
        if n != a.nbytes:
            raise ImageError(f"{self.path}: shorter than its header says (still being written?)")
        return a.astype(np.float32, copy=False)

    def index(self, index: int | None) -> int:
        """The section to show: the given one, or the central one of a volume."""
        if index is None:
            return self.nz // 2
        if not 0 <= index < self.nz:
            raise ImageError(f"image {index + 1} asked from {self.nz} in {self.path}")
        return index

    def section(self, z: int) -> np.ndarray:
        plane = self.ny * self.nx
        return self.read(z * plane, plane).reshape(self.ny, self.nx)

    def rows(self, z: int, y0: int, y1: int) -> np.ndarray:
        y0, y1 = max(0, y0), min(self.ny, y1)
        return self.read((z * self.ny + y0) * self.nx, (y1 - y0) * self.nx).reshape(y1 - y0, self.nx)

    def planes(self, z0: int, z1: int) -> np.ndarray:
        plane = self.ny * self.nx
        return self.read(z0 * plane, (z1 - z0) * plane).reshape(z1 - z0, self.ny, self.nx)


def header(path: str | os.PathLike) -> MrcHeader:
    with MrcReader(path) as r:
        return r.header()


def read_image(ref: ImageRef) -> np.ndarray:
    """One 2D image: a single-image file, one image of a stack, or the central section of a volume."""
    if Path(ref.path).suffix.lower() in (".tif", ".tiff", ".eer"):
        return read_movie_sum(ref.path)
    with MrcReader(ref.path) as r:
        return r.section(r.index(ref.index))


def read_images(path: str, indices: list[int | None]) -> dict[int | None, np.ndarray]:
    """Several images of one stack with a single open (class averages, particles); missing ones are left out."""
    out: dict = {}
    with MrcReader(path) as r:
        for i in indices:
            try:
                out[i] = r.section(r.index(i))
            except ImageError:
                pass
    return out


def read_region(ref: ImageRef, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """A rectangle of an image (only those rows are read from disk)."""
    with MrcReader(ref.path) as r:
        return np.ascontiguousarray(r.rows(r.index(ref.index), y0, y1)[:, max(0, x0):x1])


def read_movie_sum(path: str, max_frames: int = 8) -> np.ndarray:
    """Sum of the first frames of a movie (MRC or TIFF; EER needs RELION to be decoded)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".eer":
        raise ImageError("EER movies are not shown (RELION's MotionCorr renders them)")
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError as e:
            raise ImageError("TIFF movies need the tifffile package") from e
        try:
            with tifffile.TiffFile(path) as t:
                n = min(max_frames, len(t.pages))
                total = t.pages[0].asarray().astype(np.float32)  # frame by frame: one frame in memory, not all
                for i in range(1, n):
                    total += t.pages[i].asarray()
                return total
        except (OSError, ValueError, IndexError) as e:
            raise ImageError(f"{path}: {e}") from e
    with MrcReader(path) as r:
        total = r.section(0)  # frame by frame, into the array of the first: one frame in memory, not all
        for z in range(1, min(r.nz, max_frames)):
            total += r.section(z)
        return total


# ---------------------------------------------------------------------------- display helpers

def fourier_crop(img: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Downsample by cropping the Fourier transform: no aliasing, keeps the mean."""
    sy, sx = img.shape
    if (ny, nx) == (sy, sx):
        return img.astype(np.float32, copy=False)
    f = np.fft.rfft2(img.astype(np.float32, copy=False))
    top = ny // 2 + ny % 2
    rows = np.concatenate([f[:top], f[sy - ny // 2:]]) if ny // 2 else f[:top]
    out = np.fft.irfft2(rows[:, : nx // 2 + 1], s=(ny, nx))
    return (out * (ny * nx) / (sy * sx)).astype(np.float32)


def bin_mean(img: np.ndarray, k: int) -> np.ndarray:
    """Average k×k blocks (the edge rows/columns that do not fill a block are dropped)."""
    if k <= 1:
        return img.astype(np.float32, copy=False)
    ny, nx = img.shape[0] // k, img.shape[1] // k
    return img[: ny * k, : nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3), dtype=np.float32)


def downsample(img: np.ndarray, max_px: int) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is at most max_px; returns (image, factor), factor = original px per new px.

    Large images are first block-averaged by an integer factor (fast), then Fourier-cropped to
    the exact size (sharp, no aliasing), so a 4k×4k micrograph takes well under a second.
    """
    longest = max(img.shape)
    if longest <= max_px:
        return img.astype(np.float32, copy=False), 1.0
    factor = longest / max_px
    k = int(factor // 2) if factor >= 4 else 1  # leave 2-4× for the Fourier crop
    small = bin_mean(img, k) if k > 1 else img
    scale = max_px / max(small.shape)
    ny, nx = max(1, round(small.shape[0] * scale)), max(1, round(small.shape[1] * scale))
    out = fourier_crop(small, ny, nx)
    return out, img.shape[1] / nx


def contrast_limits(img: np.ndarray, sigma: float = 3.0) -> tuple[float, float]:
    """Display range: mean ± sigma·std (sigma 0 = min/max), estimated from a subsample of big images."""
    flat = img.ravel()
    if flat.size > 1_000_000:
        flat = flat[:: flat.size // 1_000_000]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    if sigma <= 0:
        lo, hi = float(flat.min()), float(flat.max())
    else:
        m, s = float(flat.mean()), float(flat.std())
        lo, hi = m - sigma * s, m + sigma * s
    return (lo, hi) if hi > lo else (lo - 0.5, lo + 0.5)


def to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (img - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)
