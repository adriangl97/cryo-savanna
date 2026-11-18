# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Variations of a job (the job editor's "Variations…", ui/variations_dialog.py): some options each given a
list of values, and one job for every combination (K = 3,4,5 × T = 2,4: six jobs), each with an alias that
says what it varies (``K3_T2``).

Values are written as a list (``3, 4, 5``, also one per line) or, for numbers, as a range ``start:stop:step``
(``2:8:2`` is 2, 4, 6, 8). Yes/No and RELION's menus take their choices. Qt-free: the dialog and the tests use
it.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from cryo_savanna.jobdefs.model import OptionDef
from cryo_savanna.project.ops import ALIAS_FORBIDDEN
from cryo_savanna.project.presets import is_input

MAX_WITHOUT_ASKING = 20  # more jobs than this: the dialog asks again
MAX_JOBS = 500  # never more than this in one go

# short tags for options that are varied often (else the option's key)
TAGS = {
    "nr_classes": "K", "tau_fudge": "T", "nr_iter": "it", "nr_iter_em": "it", "nr_iter_grad": "it",
    "particle_diameter": "D", "sampling": "samp", "highres_limit": "res", "ini_high": "lp", "sym_name": "sym",
    "offset_range": "off", "offset_step": "offstep", "nr_mpi": "mpi", "nr_threads": "j", "do_blush": "blush",
    "nr_pool": "pool", "angular_search": "ang", "threshold": "thr", "lowpass": "lp", "box_size": "box",
}


class VariationError(ValueError):
    pass


@dataclass
class Variation:
    key: str
    values: list[str]
    tag: str = ""


def default_tag(key: str) -> str:
    return TAGS.get(key, key.lstrip("-"))  # a CLI tool's flag without its dashes


def is_choice(opt: OptionDef) -> bool:
    return opt.kind in ("radio", "boolean")


def choices(opt: OptionDef) -> list[str]:
    return list(opt.choices) if opt.kind == "radio" else ["Yes", "No"]


def _number(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.6g}"


def parse_values(opt: OptionDef, text: str) -> list[str]:
    """The values of a variation, in order, each once. A ``start:stop:step`` part is a range of numbers (stop
    included when the steps reach it)."""
    out: list[str] = []
    for part in re.split(r"[,\n]", text):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        nums = [_number(b.strip()) for b in bits] if len(bits) == 3 and not is_input(opt) else []
        if nums and all(n is not None for n in nums):
            start, stop, step = (n for n in nums if n is not None)
            if step == 0 or (stop - start) / step < 0:
                raise VariationError(f"{opt.label.rstrip(': ')}: the range {part} has no values "
                                     "(start:stop:step, the step towards stop)")
            n = int(math.floor((stop - start) / step + 1e-9)) + 1
            if n > MAX_JOBS:
                raise VariationError(f"{opt.label.rstrip(': ')}: the range {part} has {n} values")
            out += [_fmt(start + i * step) for i in range(n)]
        else:
            out.append(part)
    if is_choice(opt):
        allowed = choices(opt)
        bad = [v for v in out if v not in allowed]
        if bad:
            raise VariationError(f"{opt.label.rstrip(': ')}: {', '.join(bad)} is not one of {', '.join(allowed)}")
    if opt.kind == "slider" or (opt.min is not None or opt.max is not None):
        bad = [v for v in out if _number(v) is None]
        if bad:
            raise VariationError(f"{opt.label.rstrip(': ')}: {', '.join(bad)} is not a number")
    return list(dict.fromkeys(out))  # each once, in order


def combinations(variations: list[Variation]) -> list[dict[str, str]]:
    """Every combination of the values, the first variation changing slowest."""
    used = [v for v in variations if v.values]
    if not used:
        return []
    keys = [v.key for v in used]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(v.values for v in used))]


def count(variations: list[Variation]) -> int:
    used = [v for v in variations if v.values]
    return math.prod(len(v.values) for v in used) if used else 0


def _clean(text: str) -> str:
    text = "".join("_" if c in ALIAS_FORBIDDEN or c.isspace() else c for c in text)
    return re.sub(r"_+", "_", text).strip("_")


def value_tag(opt: OptionDef | None, value: str) -> str:
    """A value as it goes into an alias: an input file by its job (``Select_job009``), else the value."""
    if opt is not None and is_input(opt) and "/" in value:
        parts = PurePosixPath(value).parts
        return _clean("_".join(parts[:2]) if len(parts) >= 2 else value)
    return _clean(value)


def alias_for(base: str, combo: dict[str, str], variations: list[Variation],
              options: dict[str, OptionDef]) -> str:
    """``base_K3_T2``: the base (the editor's alias, else the job type) and each varied option's tag and value."""
    tags = {v.key: (v.tag or default_tag(v.key)) for v in variations}
    bits = [f"{_clean(tags[k])}{value_tag(options.get(k), val)}" for k, val in combo.items()]
    alias = "_".join(b for b in [_clean(base)] + bits if b)
    if alias.startswith("job"):  # RELION refuses aliases starting with "job"
        alias = "v_" + alias
    return alias


def duplicates(aliases: list[str]) -> list[str]:
    return sorted({a for a in aliases if a and aliases.count(a) > 1})
