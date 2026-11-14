# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The kinds of RELION External jobs cryo-savanna makes with a form and a runner of its own, in one place.

Each is a *family*: the CLI tools (jobdefs/cli_tools.py), pytom (pytom_jobs.py), AreTomo3 (aretomo_jobs.py)
and the tomogram masks (tomomask_jobs.py). To RELION they are all ``relion.external`` jobs; what tells them
apart is the runner their executable names (``python3 .cryo-savanna/<runner>.py <kind>``). The rest of
cryo-savanna asks here, not each family in turn:

- ``kind_of(js)`` / ``kind_of_exe(fn_exe)``: (family, kind) of a job, or None for any other job;
- ``install_runner(js, project)``: put the runner the job runs into the project (project/pipeliner.py);
- ``main_input(js)``: the file of the form's main input field (project/lineage.py);
- ``program_title(fn_exe)``: what the job runs, as shown with its type (project/snapshot.py);
- ``Forms``: the forms of every family, built once (ui/job_actions.py, ui/job_info.py), and
  ``Forms.variant_for(js)``: the form a job was made with.

A new wrapper is a jobs module like the others (its form, ``<x>_kind``, ``install_runner``, a main input),
its runner, and one more Family in ``FAMILIES`` below. What stays with the family: its menu entry and how a
new form is filled in (ui/job_actions.py), and its viewers (results/registry.py: EXTERNAL_VIEWS, by family
name).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs import aretomo_jobs, cli_tools, pytom_jobs, tomomask_jobs
from cryo_savanna.jobdefs.flags import FlagSource
from cryo_savanna.jobdefs.model import JobDefs, JobVariant
from cryo_savanna.project.jobstar import JobStar

EXTERNAL = cli_tools.EXTERNAL
CLI, PYTOM, ARETOMO, TOMOMASK = "cli", "pytom", "aretomo", "tomomask"


@dataclass(frozen=True)
class Family:
    name: str
    kind: Callable[[str], str | None]  # External's executable -> the kind (a program, a mode), None: not this family
    title: Callable[[str], str]  # kind -> what the job runs, in words
    install: Callable[[str | os.PathLike], Path]  # the runner (and what it imports) into a project
    main_input: Callable[[JobStar], str | None] | None  # the file of the form's main input field
    catalog: Callable[[JobDefs, RelionEnv, FlagSource], Any]  # the family's forms
    variant: Callable[[Any, str, bool], JobVariant | None]  # (catalog, kind, is tomo) -> the form (None: none)


FAMILIES: tuple[Family, ...] = (
    Family(CLI, cli_tools.runner_program, lambda program: program, cli_tools.install_runner, None,
           lambda defs, env, flags: cli_tools.CliCatalog(defs, env, flags),
           lambda catalog, program, tomo: catalog.variant(program, tomo)),
    Family(PYTOM, pytom_jobs.pytom_kind, lambda kind: pytom_jobs.TITLES[kind], pytom_jobs.install_runner,
           pytom_jobs.pytom_main_input, lambda defs, env, flags: pytom_jobs.PytomCatalog(defs, env),
           lambda catalog, kind, tomo: catalog.variant(kind)),
    Family(ARETOMO, aretomo_jobs.aretomo_kind, lambda kind: aretomo_jobs.TITLES[kind], aretomo_jobs.install_runner,
           aretomo_jobs.aretomo_main_input, lambda defs, env, flags: aretomo_jobs.AretomoCatalog(defs, env),
           lambda catalog, kind, tomo: catalog.variant(kind)),
    Family(TOMOMASK, tomomask_jobs.mask_kind, lambda kind: tomomask_jobs.TITLES[kind], tomomask_jobs.install_runner,
           tomomask_jobs.mask_main_input, lambda defs, env, flags: tomomask_jobs.TomoMaskCatalog(defs, env),
           lambda catalog, kind, tomo: catalog.variant(kind)),
)
BY_NAME = {f.name: f for f in FAMILIES}


def kind_of_exe(fn_exe: str) -> tuple[Family, str] | None:
    """(family, kind) for an External executable one of the families made, else None."""
    for family in FAMILIES:
        kind = family.kind(fn_exe)
        if kind is not None:
            return family, kind
    return None


def kind_of(js: JobStar) -> tuple[Family, str] | None:
    """(family, kind) of a job made from one of cryo-savanna's own forms, else None (any other job)."""
    return kind_of_exe(js.options.get("fn_exe", "")) if js.type_label == EXTERNAL else None


def family_of(js: JobStar) -> str | None:
    """The family's name (CLI, PYTOM, ...), or None."""
    found = kind_of(js)
    return found[0].name if found is not None else None


def install_runner(js: JobStar, project_dir: str | os.PathLike) -> Path | None:
    """Put the runner this job runs into the project (nothing for a job that has none)."""
    found = kind_of(js)
    return found[0].install(project_dir) if found is not None else None


def main_input(js: JobStar) -> str | None:
    """The file in the main input field of the job's own form, else None (also for other jobs)."""
    found = kind_of(js)
    if found is None or found[0].main_input is None:
        return None
    return found[0].main_input(js)


def program_title(fn_exe: str) -> str:
    """What an External job runs: the CLI tool or the wrapper's title, else its executable's name."""
    found = kind_of_exe(fn_exe)
    if found is not None:
        return found[0].title(found[1])
    parts = fn_exe.split()
    return Path(parts[0]).name if parts else ""


class Forms:
    """The forms of every family for one RELION environment: each family's catalog (which builds a form
    on first use), by family name and as ``.cli``, ``.pytom``, ``.aretomo``, ``.tomomask``."""

    def __init__(self, defs: JobDefs, env: RelionEnv, flag_source: FlagSource | None = None):
        self.defs, self.env = defs, env
        self.flag_source = flag_source or FlagSource()
        self.catalogs: dict[str, Any] = {f.name: f.catalog(defs, env, self.flag_source) for f in FAMILIES}

    def __getattr__(self, name: str) -> Any:  # forms.cli, forms.pytom, ...
        catalogs = self.__dict__.get("catalogs", {})
        if name in catalogs:
            return catalogs[name]
        raise AttributeError(name)

    def variant(self, family: str, kind: str, tomo: bool = False) -> JobVariant | None:
        """The form of a kind (None: a family without its forms here, or a CLI tool whose --help cannot be read)."""
        catalog = self.catalogs.get(family)
        return BY_NAME[family].variant(catalog, kind, tomo) if catalog is not None else None

    def variant_for(self, js: JobStar) -> JobVariant | None:
        """The form a job was made with: its family's, else None (a RELION job: the job type's form)."""
        found = kind_of(js)
        if found is None:
            return None
        return self.variant(found[0].name, found[1], js.is_tomo)
