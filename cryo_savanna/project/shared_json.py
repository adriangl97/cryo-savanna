# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's own JSON files in a project (``.cryo-savanna/*.json``), changed by every session that
has the project open, perhaps on other hosts.

A change is read → changed → written while holding a POSIX lock on ``<file>.lock`` (lockf, which NFS
passes on to the server), so two sessions saving at the same moment do not lose each other's change.
The file itself is replaced in one step, so a reader without the lock sees it whole. Where locks are
not available (a file system without them), the change is made without one, as before.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path

try:
    import fcntl
except ImportError:  # not on Windows
    fcntl = None


@contextlib.contextmanager
def _locked(path: Path):
    fd = None
    try:
        fd = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT, 0o666)
        if fcntl is not None:
            fcntl.lockf(fd, fcntl.LOCK_EX)
    except OSError:  # no locks here (ENOLCK), or the lock file cannot be made: go on without
        pass
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)  # releases the lock


def read_json(path: Path) -> dict:
    """The file's contents ({} when it is missing or not a JSON object)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update_json(path: Path, change: Callable[[dict], dict]) -> dict:
    """Write ``change(contents)`` to ``path``, with no other session changing it meanwhile; returns it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        data = change(read_json(path))
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")  # also if the lock is not available
        try:
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return data
