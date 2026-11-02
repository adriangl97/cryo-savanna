# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Thumbnail cache: downsampled images in memory (LRU) and on disk.

On disk: ``<root>/<hash of the project path>/``, one ``.npz`` per image and size (normalised float16
plus mean and std, so values come back on their own scale), keyed by path, index, file size and
mtime, so a rewritten file is read again. The root is ``$CRYO_SAVANNA_CACHE``, else
``$XDG_CACHE_HOME/cryo-savanna`` (``~/.cache/cryo-savanna``). Only images that are slow to make go to
disk (micrograph previews); the oldest files, of any project, go when all of them together grow past
``max_disk_mb``. A cache that cannot be written (read-only home, quota) is silently skipped.

Only cryo-savanna's own files are counted and deleted (``<root>/<16 hex digits>/*.npz``, and the
``*.tmp`` a cryo-savanna that ended while saving left there): the root may be a directory with other
things in it (``CRYO_SAVANNA_CACHE=/scratch/$USER``).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

from cryo_savanna.images.mrc import ImageRef


def cache_root() -> Path:
    env = os.environ.get("CRYO_SAVANNA_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cryo-savanna"


_PROJECT_DIR = re.compile(r"^[0-9a-f]{16}$")


def _project_dirs(root: Path) -> list[str]:
    try:
        return [e.path for e in os.scandir(root) if e.is_dir(follow_symlinks=False) and _PROJECT_DIR.match(e.name)]
    except OSError:
        return []


def cache_files(root: Path | None = None) -> list[tuple[float, int, str]]:
    """(mtime, size, path) of every cached image, of all projects."""
    out = []
    for d in _project_dirs(root or cache_root()):
        try:
            for e in os.scandir(d):
                if e.name.endswith(".npz") and e.is_file(follow_symlinks=False):
                    st = e.stat()
                    out.append((st.st_mtime, st.st_size, e.path))
        except OSError:
            continue
    return out


TMP_KEEP_S = 3600  # a .tmp older than this was left by a program that ended while saving


def stale_temp_files(root: Path | None = None, older_than: float = TMP_KEEP_S) -> list[str]:
    """Half-written images (``*.tmp``) left by a cryo-savanna that ended while saving one."""
    now, out = time.time(), []
    for d in _project_dirs(root or cache_root()):
        try:
            for e in os.scandir(d):
                if e.name.endswith(".tmp") and e.is_file(follow_symlinks=False) \
                        and now - e.stat().st_mtime >= older_than:
                    out.append(e.path)
        except OSError:
            continue
    return out


def cache_size_mb(root: Path | None = None) -> float:
    return sum(f[1] for f in cache_files(root)) / 1024 / 1024


def clear_cache(root: Path | None = None) -> None:
    """Delete every cached image (only those, and half-written ones: nothing else in the root is touched)."""
    root = root or cache_root()
    for path in stale_temp_files(root, older_than=60):  # not one being written right now
        try:
            os.unlink(path)
        except OSError:
            pass
    for _, _, path in cache_files(root):
        try:
            os.unlink(path)
        except OSError:
            pass
    for d in _project_dirs(root):
        try:
            os.rmdir(d)  # once empty
        except OSError:
            pass


class ThumbCache:
    def __init__(self, project_dir: str | os.PathLike, root: Path | None = None, memory_mb: int = 256,
                 disk: bool = True, max_disk_mb: int = 2048):
        project = os.path.realpath(project_dir)
        self.root = root or cache_root()
        self.dir = self.root / hashlib.sha1(project.encode()).hexdigest()[:16]
        self.disk = disk
        self._mem: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._limit = memory_mb * 1024 * 1024
        self.max_disk = max_disk_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._saved = 0

    @staticmethod
    def key(ref: ImageRef, size: int) -> tuple | None:
        try:
            st = os.stat(ref.path)
        except OSError:
            return None
        return ref.path, ref.index, size, st.st_size, st.st_mtime_ns

    def _file(self, key: tuple) -> Path:
        return self.dir / (hashlib.sha1(repr(key).encode()).hexdigest() + ".npz")

    def get(self, ref: ImageRef, size: int, make: Callable[[], np.ndarray], disk: bool = False) -> np.ndarray:
        """The cached image for (ref, size), or make() it and keep it (also on disk with disk=True)."""
        key = self.key(ref, size)
        if key is None:
            return make()  # missing file: let make() raise the error
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None:
                self._mem.move_to_end(key)
                return hit
        use_disk = self.disk and disk
        img = self._load(key) if use_disk else None
        if img is None:
            img = make()
            if use_disk:
                self._save(key, img)
        self._remember(key, img)
        return img

    def _load(self, key: tuple) -> np.ndarray | None:
        try:
            with np.load(self._file(key)) as z:
                mean, std = (float(v) for v in z["stats"])
                return z["img"].astype(np.float32) * std + mean
        except (OSError, ValueError, KeyError):
            return None

    def _save(self, key: tuple, img: np.ndarray) -> None:
        mean, std = float(img.mean()), float(img.std()) or 1.0
        f = self._file(key)
        tmp = f.with_name(f"{f.stem}.{uuid.uuid4().hex}.tmp")  # a cache root may be shared by hosts
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                np.savez(fh, img=((img - mean) / std).astype(np.float16), stats=np.array([mean, std], np.float64))
            os.replace(tmp, f)
        except OSError:
            try:  # a full disk or quota: no half-written file left behind
                tmp.unlink()
            except OSError:
                pass
            return
        with self._lock:
            self._saved += 1
            check = self._saved % 200 == 1
        if check:
            self.prune()

    def prune(self) -> None:
        """Delete the least recently written files (of any project) while all of them are over max_disk_mb."""
        for path in stale_temp_files(self.root):
            try:
                os.unlink(path)
            except OSError:
                pass
        files = cache_files(self.root)
        total = sum(f[1] for f in files)
        for _, size, path in sorted(files):
            if total <= self.max_disk:
                break
            try:
                os.unlink(path)
                total -= size
            except OSError:
                pass

    def _remember(self, key: tuple, img: np.ndarray) -> None:
        with self._lock:
            if key in self._mem:
                return
            self._mem[key] = img
            self._bytes += img.nbytes
            while self._bytes > self._limit and len(self._mem) > 1:
                _, old = self._mem.popitem(last=False)
                self._bytes -= old.nbytes
