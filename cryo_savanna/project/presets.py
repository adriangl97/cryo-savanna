# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Setting presets: named sets of job options, saved per job type, the user's own (not the project's).

One JSON file per job type in ``<config>/presets/`` (``config_root()``: ``$CRYO_SAVANNA_CONFIG``, else
``$XDG_CONFIG_HOME/cryo-savanna``, else ``~/.config/cryo-savanna``), named from the form's key
(``JobVariant.key``: ``relion.class3d``, ``relion.class3d:tomo``, ``cli:relion_image_handler``, ``pytom:match``
...; ``:`` becomes ``__``)::

    {"format": 1, "presets": {"Class3D 200 kDa": {"options": {"nr_classes": "4", "tau_fudge": "4"},
                                                "relion_version": "5.1.1", "saved": "2026-11-16T10:00:00"}}}

A preset holds only the options chosen when it was saved; loading one (ui/job_editor.py) keeps the inputs of
the form, sets the rest to RELION's defaults and then the preset's options. Every change is a read-change-write
under a lock (project/shared_json.py), so two open windows do not lose each other's change. Keys a form does not
have (another RELION version) are kept, and skipped when loading.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.jobdefs.model import OptionDef
from cryo_savanna.project.shared_json import read_json, update_json

FORMAT = 1
INPUT_KINDS = frozenset({"inputnode", "filename", "folder"})


class PresetError(ValueError):
    pass


def config_root() -> Path:
    env = os.environ.get("CRYO_SAVANNA_CONFIG")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "cryo-savanna"


def is_input(opt: OptionDef) -> bool:
    """An input of the job (an input node, a file or a folder): kept as it is when a preset is loaded, and not
    saved in one unless asked. The queue submission script is a file too: specific to a machine."""
    return opt.kind in INPUT_KINDS


def file_name(key: str) -> str:
    """The preset file of a form's key ("relion.class3d:tomo" -> "relion.class3d__tomo.json")."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", key.replace(":", "__")) + ".json"


@dataclass
class Preset:
    name: str
    options: dict[str, str]
    relion_version: str = ""
    saved: str = ""  # ISO time
    extra: dict = field(default_factory=dict)  # anything else in the file, kept as it was


def check_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise PresetError("A preset needs a name.")
    if len(name) > 120 or any(c in name for c in "\n\r\t"):
        raise PresetError("A preset name is one line of at most 120 characters.")
    return name


class PresetStore:
    """The presets of every job type, in ``root`` (default: ``config_root() / "presets"``)."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root is not None else config_root() / "presets"

    def path(self, key: str) -> Path:
        return self.root / file_name(key)

    def _all(self, key: str) -> dict[str, dict]:
        presets = read_json(self.path(key)).get("presets", {})
        return {n: p for n, p in presets.items() if isinstance(p, dict)} if isinstance(presets, dict) else {}

    def names(self, key: str) -> list[str]:
        """The presets of this job type, sorted by name (case not counted)."""
        return sorted(self._all(key), key=str.casefold)

    def get(self, key: str, name: str) -> Preset | None:
        p = self._all(key).get(name)
        if p is None:
            return None
        options = {str(k): str(v) for k, v in (p.get("options") or {}).items()}
        extra = {k: v for k, v in p.items() if k not in ("options", "relion_version", "saved")}
        return Preset(name, options, str(p.get("relion_version", "")), str(p.get("saved", "")), extra)

    def _change(self, key: str, change) -> None:
        def apply(data: dict) -> dict:
            presets = data.get("presets")
            if not isinstance(presets, dict):
                presets = {}
            change(presets)
            data["format"], data["presets"] = FORMAT, presets
            return data
        try:
            update_json(self.path(key), apply)
        except OSError as e:
            raise PresetError(f"Cannot write {self.path(key)}: {e}") from e

    def save(self, key: str, name: str, options: dict[str, str], relion_version: str = "") -> Preset:
        """Save (or replace) a preset."""
        name = check_name(name)
        saved = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry = {"options": dict(options), "relion_version": relion_version, "saved": saved}
        self._change(key, lambda presets: presets.__setitem__(name, entry))
        return Preset(name, dict(options), relion_version, saved)

    def set_options(self, key: str, name: str, options: dict[str, str]) -> None:
        """New options for an existing preset (the manage dialog), keeping the rest of it."""
        def change(presets: dict) -> None:
            if name not in presets:
                raise PresetError(f"There is no preset “{name}” any more.")
            presets[name] = {**presets[name], "options": dict(options), "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._change(key, change)

    def rename(self, key: str, old: str, new: str) -> None:
        new = check_name(new)

        def change(presets: dict) -> None:
            if old not in presets:
                raise PresetError(f"There is no preset “{old}” any more.")
            if new != old and new in presets:
                raise PresetError(f"There is a preset “{new}” already.")
            presets[new] = presets.pop(old)
        self._change(key, change)

    def delete(self, key: str, name: str) -> None:
        self._change(key, lambda presets: presets.pop(name, None))

    def duplicate(self, key: str, name: str) -> str:
        """A copy of a preset, as "‹name› (copy)" (or "(copy 2)", ...); its new name."""
        made: list[str] = []

        def change(presets: dict) -> None:
            if name not in presets:
                raise PresetError(f"There is no preset “{name}” any more.")
            n, i = f"{name} (copy)", 2
            while n in presets:
                n, i = f"{name} (copy {i})", i + 1
            presets[n] = dict(presets[name])
            made.append(n)
        self._change(key, change)
        return made[0]
