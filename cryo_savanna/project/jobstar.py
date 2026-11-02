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
"""job.star files (RelionJob::read / RelionJob::write in RELION 5.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.star import StarBlock, format_star, read_star, write_star
from cryo_savanna.relion_consts import JOB_TYPE_BY_ID


def _star_bool(v: str | None) -> bool:
    # RELION writes bools as integers; accept the usual spellings too.
    return str(v).strip().lower() in ("1", "true", "yes")


@dataclass
class JobStar:
    type_label: str
    is_continue: bool = False
    is_tomo: bool = False
    options: dict[str, str] = field(default_factory=dict)  # joboption variable -> value

    @classmethod
    def read(cls, path: str | os.PathLike) -> "JobStar":
        """Read a job.star file, or the job.star inside a job directory."""
        p = Path(path)
        if p.is_dir():
            p = p / "job.star"
        blocks = read_star(p, blocks=("job", "joboptions_values"))
        if "job" not in blocks:
            raise ValueError(f"{p}: no data_job block")
        head = blocks["job"].as_dict()
        label = head.get("rlnJobTypeLabel")
        if label is None:
            # RELION 3.0 wrote the integer type instead
            jt = JOB_TYPE_BY_ID.get(int(head.get("rlnJobType", "-1")))
            if jt is None:
                raise ValueError(f"{p}: unknown job type")
            label = jt.label
        js = cls(
            type_label=label,
            is_continue=_star_bool(head.get("rlnJobIsContinue", "0")),
            is_tomo=_star_bool(head.get("rlnJobIsTomo", "0")),
        )
        vals = blocks.get("joboptions_values")
        if vals is not None:
            for rec in vals.records():
                js.options[rec["rlnJobOptionVariable"]] = rec["rlnJobOptionValue"]
        return js

    def to_blocks(self) -> list[StarBlock]:
        head = StarBlock.from_dict(
            "job",
            {"rlnJobTypeLabel": self.type_label, "rlnJobIsContinue": self.is_continue, "rlnJobIsTomo": self.is_tomo},
        )
        # RELION stores joboptions in a std::map, so they come out sorted by variable name.
        vals = StarBlock.from_records(
            "joboptions_values",
            ["rlnJobOptionVariable", "rlnJobOptionValue"],
            ([k, self.options[k]] for k in sorted(self.options, key=lambda s: s.encode())),
        )
        return [head, vals]

    def to_text(self) -> str:
        return format_star(self.to_blocks())

    def write(self, path: str | os.PathLike) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "job.star"
        write_star(p, self.to_blocks())


def read_last_used(project_dir: str | os.PathLike, type_label: str, is_tomo: bool) -> JobStar | None:
    """The settings RELION remembers for the next new job of this type (.gui_<name>job.star).

    Only used when it was written for the same SPA/tomo variant.
    """
    from cryo_savanna.relion_consts import HIDDEN_SETTINGS, job_type_for_label

    jt = job_type_for_label(type_label)
    name = HIDDEN_SETTINGS.get(jt.label if jt else type_label)
    if name is None:
        return None
    try:
        js = JobStar.read(Path(project_dir) / f"{name}job.star")
    except (OSError, ValueError, KeyError):
        return None
    return js if js.is_tomo == is_tomo else None
