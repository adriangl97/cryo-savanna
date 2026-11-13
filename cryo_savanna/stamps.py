# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Telling whether a file changed, and keeping what was read from it (no Qt).

``stamp(path)`` is what every reader and viewer compares to know whether to read a file again: its
modification time and size, None while it is not there. One stat, so it can be asked at every poll, also
over a network file system.

``Cache`` keeps the last few results of a slow read (a score map, a binned tomogram, the per-particle CTF
changes), under a key that has the file's stamp in it, so a file written again is read again. The viewers
read in a thread pool, several at once: a Cache can be used from all of them.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

Stamp = tuple[int, int] | None  # (mtime_ns, size), None when the file is missing

_MISSING: Any = object()


def stamp(path: str | os.PathLike | None) -> Stamp:
    """(mtime in ns, size) of a file or directory; None when it is missing (or no path was given). A
    directory's changes when files are added to or removed from it."""
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class Cache:
    """The ``size`` results used last, by key. Safe to use from several threads."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._making = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            if key not in self._items:
                return default
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def get_or_make(self, key: Hashable, make: Callable[[], Any], one_at_a_time: bool = False) -> Any:
        """The result kept under ``key``, else ``make()`` (kept from then on; also None).

        ``one_at_a_time``: a second thread asking while the first still makes a result waits for it, and
        finds it made if it asked for the same one: for reads so slow that two of them at once (two views
        of one job opening together) should not both happen.
        """
        hit = self.get(key, _MISSING)
        if hit is not _MISSING:
            return hit
        if not one_at_a_time:
            value = make()
            self.put(key, value)
            return value
        with self._making:
            hit = self.get(key, _MISSING)
            if hit is _MISSING:
                hit = make()
                self.put(key, hit)
            return hit

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._items
