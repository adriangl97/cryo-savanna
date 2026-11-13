# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
"""Job definitions extracted from the RELION source (see tools/extract_jobdefs).

A *variant* is one job type in SPA or tomo mode, e.g. "relion.class3d" and "relion.class3d:tomo".
It holds every job option (kind, label, default, range, help, environment bindings) and the
layout of RELION's own job window (tabs, rows, groups that a yes/no option switches on/off).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources

from cryo_savanna.env.relion_env import RelionEnv, float_to_string
from cryo_savanna.project.jobstar import JobStar
from cryo_savanna.relion_consts import job_type_for_label

DEFAULT_VERSION = "5.1.1"

# Hand-curated: input options that define a job's place in the job tree, where the job window's
# order does not follow the main data path. Other job types use the window order.
LINEAGE_INPUTS = {
    "relion.polish": ("fn_data", "fn_post", "fn_mic"),
    "relion.extract": ("fndata_reextract", "coords_suffix", "star_mics"),
}


@dataclass(frozen=True)
class OptionDef:
    key: str
    kind: str  # any | filename | folder | inputnode | radio | boolean | slider
    label: str
    default: str
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    pattern: str = ""
    directory: str = ""
    node_type: str = ""
    node_type_depth: int = 1
    toggle: str = "leave_active"  # behaviour when switching to "continue": see is_enabled_for_continue
    forced_inactive: bool = False
    env: tuple[tuple[str, str], ...] = ()  # (field, RELION_* variable)

    @classmethod
    def from_json(cls, key: str, d: dict) -> "OptionDef":
        return cls(
            key=key,
            kind=d["kind"],
            label=d["label"],
            default=d["default"],
            help=d.get("help", ""),
            min=d.get("min"),
            max=d.get("max"),
            step=d.get("step"),
            choices=tuple(d.get("choices", ())),
            pattern=d.get("pattern", ""),
            directory=d.get("directory", ""),
            node_type=d.get("node_type", ""),
            node_type_depth=d.get("node_type_depth", 1),
            toggle=d.get("toggle", "leave_active"),
            forced_inactive=d.get("forced_inactive", False),
            env=tuple(sorted(d.get("env", {}).items())),
        )

    def resolved(self, env: RelionEnv) -> "OptionDef":
        """This option with the environment applied, as RelionJob::initialise would build it."""
        changes: dict = {}
        for fld, var in self.env:
            if env.get(var) is None:
                continue
            if fld == "default":
                if self.kind == "slider":
                    changes["default"] = float_to_string(env.integer(var))
                elif self.kind == "boolean":
                    changes["default"] = "Yes" if env.boolean(var) else "No"
                else:
                    changes["default"] = env.get(var)
            elif fld in ("min", "max"):
                changes[fld] = env.integer(var)
            elif fld == "forced_inactive":
                changes["forced_inactive"] = not env.allow_change_minimum_dedicated
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class Row:
    keys: tuple[str, ...]
    label: str = ""  # only for rows with several options (RELION's place2/place3)
    group: int | None = None
    y: int = 0


@dataclass(frozen=True)
class Tab:
    label: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Group:
    id: int
    parent: int | None
    controller: str | None  # yes/no option that switches this group on and off
    active_when: str = "Yes"


@dataclass
class JobVariant:
    key: str  # "relion.class3d" or "relion.class3d:tomo"
    label: str  # job type label written to job.star
    proc_id: int
    dirname: str
    is_tomo: bool
    options: dict[str, OptionDef]
    tabs: list[Tab]
    groups: dict[int, Group] = field(default_factory=dict)

    @classmethod
    def from_json(cls, key: str, d: dict) -> "JobVariant":
        return cls(
            key=key,
            label=d["label"],
            proc_id=d["proc_id"],
            dirname=d["dirname"],
            is_tomo=d["is_tomo"],
            options={k: OptionDef.from_json(k, o) for k, o in d["options"].items()},
            tabs=[
                Tab(t["label"], tuple(Row(tuple(r["keys"]), r.get("label", ""), r.get("group"), r["y"]) for r in t["rows"]))
                for t in d["tabs"]
            ],
            groups={
                int(gid): Group(int(gid), g["parent"], g["controller"], g.get("active_when", "Yes"))
                for gid, g in d["groups"].items()
            },
        )

    # -- options with the environment applied ------------------------------------------------
    def resolved_options(self, env: RelionEnv) -> dict[str, OptionDef]:
        """All options for this environment, including the dynamic RELION_QSUB_EXTRA<i> fields."""
        out = {k: o.resolved(env) for k, o in self.options.items()}
        for ex in env.qsub_extras():
            out[ex.key] = OptionDef(key=ex.key, kind="any", label=ex.label, default=ex.default, help=ex.help)
        return out

    def resolved_tabs(self, env: RelionEnv) -> list[Tab]:
        """Tabs with the extra queue fields inserted after 'qsub', where setupRunTab places them."""
        extras = env.qsub_extras()
        if not extras:
            return list(self.tabs)
        tabs = []
        for t in self.tabs:
            rows: list[Row] = []
            for r in t.rows:
                rows.append(r)
                if r.keys == ("qsub",):
                    rows += [Row((ex.key,), group=r.group, y=r.y) for ex in extras]
            tabs.append(Tab(t.label, tuple(rows)))
        return tabs

    def default_values(self, env: RelionEnv) -> dict[str, str]:
        return {k: o.default for k, o in self.resolved_options(env).items()}

    def new_jobstar(self, env: RelionEnv, is_continue: bool = False) -> JobStar:
        return JobStar(type_label=self.label, is_continue=is_continue, is_tomo=self.is_tomo,
                       options=self.default_values(env))

    def values_from_jobstar(self, js: JobStar, env: RelionEnv) -> tuple[dict[str, str], list[str]]:
        """Defaults overlaid with the job.star values, like RelionJob::read(do_initialise=true).

        Returns (values, ignored) where ``ignored`` lists job.star options this RELION version
        does not define (RELION warns about and drops those).
        """
        values = self.default_values(env)
        ignored = []
        for k, v in js.options.items():
            if k in values:
                values[k] = v
            else:
                ignored.append(k)
        return values, ignored

    # -- layout logic -------------------------------------------------------------------------
    def group_active(self, gid: int | None, values: dict[str, str]) -> bool:
        """Whether a group (and all groups containing it) is switched on by its yes/no option.

        RELION activates a group whenever the controlling value starts with '$$' (a Schemes variable).
        """
        while gid is not None:
            g = self.groups[gid]
            if g.controller is not None:
                v = values.get(g.controller, "")
                if not v.startswith("$$") and v != g.active_when:
                    return False
            gid = g.parent
        return True

    @staticmethod
    def enabled_for_mode(opt: OptionDef, is_continue: bool) -> bool:
        """RELION's toggle_new_continue: which options can be edited in new vs continue mode."""
        if opt.forced_inactive:
            return False
        if opt.toggle == "deactivate":
            return not is_continue
        if opt.toggle == "reactivate":
            return is_continue
        if opt.toggle == "always_deactivate":
            return False
        return True

    def input_options(self) -> list[str]:
        """Input-node options in the order they appear in the job window."""
        seen = []
        for t in self.tabs:
            for r in t.rows:
                for k in r.keys:
                    if self.options.get(k) is not None and self.options[k].kind == "inputnode" and k not in seen:
                        seen.append(k)
        return seen

    def lineage_inputs(self) -> list[str]:
        """Input options in order of preference for placing this job in the job tree."""
        preferred = [k for k in LINEAGE_INPUTS.get(self.label, ()) if k in self.options]
        return preferred + [k for k in self.input_options() if k not in preferred]


@dataclass(frozen=True)
class MenuEntry:
    menu_label: str
    variant: str


class JobDefs:
    def __init__(self, data: dict):
        self.version: str = data["relion_version"]
        self.commit: str | None = data.get("relion_commit")
        self.variants: dict[str, JobVariant] = {k: JobVariant.from_json(k, v) for k, v in data["variants"].items()}
        self.menus: dict[str, list[MenuEntry]] = {
            mode: [MenuEntry(e["menu_label"], e["variant"]) for e in entries] for mode, entries in data["menus"].items()
        }

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, version: str = DEFAULT_VERSION) -> "JobDefs":
        text = resources.files("cryo_savanna.jobdefs").joinpath(f"relion-{version}.json").read_text(encoding="utf-8")
        return cls(json.loads(text))

    def variant(self, type_label: str, is_tomo: bool = False) -> JobVariant:
        key = f"{type_label}:tomo" if is_tomo else type_label
        try:
            return self.variants[key]
        except KeyError:
            raise KeyError(f"unknown job type {type_label!r} (RELION {self.version})") from None

    def variant_for_jobstar(self, js: JobStar) -> JobVariant:
        jt = job_type_for_label(js.type_label)
        if jt is None:
            raise KeyError(f"unknown job type {js.type_label!r}")
        return self.variant(jt.label, js.is_tomo)

    def menu(self, tomo: bool) -> list[tuple[str, JobVariant]]:
        return [(e.menu_label, self.variants[e.variant]) for e in self.menus["tomo" if tomo else "spa"]]
