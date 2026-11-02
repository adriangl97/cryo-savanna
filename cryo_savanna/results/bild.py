# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""RELION's angular-distribution .bild files, and the map each one belongs to.

Refine3D, Class3D and MultiBody write ``<root>_class001_angdist.bild`` (``_body001`` for bodies)
next to each map (HealpixSampling::writeBildFileOrientationalDistribution in RELION): one cylinder
per viewing direction that has particles, from radius R = box × pixel size / 2 around the map's
centre outwards, longer (up to 1.3 R) and redder the more particles have that view. Coordinates
are in Å with voxel 0 at 0 Å.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# run_class001, run_it025_class002, run_it012_half1_class001, run_body001, ...
_NAME = re.compile(r"^(?P<root>.+?)(?:_it(?P<it>\d+))?(?:_(?P<half>half[12]))?_(?P<what>class|body)(?P<n>\d+)"
                   r"_angdist\.bild$")


@dataclass
class Bild:
    """The cylinders of a .bild file (Å): from ``start`` to ``end``, with a radius and an RGB colour (0-1)."""

    start: np.ndarray  # [n, 3] x, y, z
    end: np.ndarray  # [n, 3]
    radius: np.ndarray  # [n]
    color: np.ndarray  # [n, 3]

    def __len__(self) -> int:
        return len(self.radius)


def read_bild(path: str | os.PathLike) -> Bild:
    """The cylinders (and arrows, drawn as cylinders) of a BILD file; other shapes are skipped."""
    starts, ends, radii, colors = [], [], [], []
    color = (0.5, 0.5, 0.5)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0]
            try:
                if cmd == ".color" and len(parts) >= 4:
                    color = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif cmd in (".cylinder", ".arrow") and len(parts) >= 7:
                    v = [float(x) for x in parts[1:7]]
                    starts.append(v[:3])
                    ends.append(v[3:])
                    radii.append(float(parts[7]) if len(parts) >= 8 else (0.1 if cmd == ".arrow" else 1.0))
                    colors.append(color)
            except ValueError:
                continue  # a named colour or a malformed line
    if not radii:
        return Bild(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)))
    return Bild(np.array(starts), np.array(ends), np.array(radii), np.clip(np.array(colors), 0, 1))


@dataclass
class AngdistPair:
    label: str  # "Final, class 1", "Iteration 25, class 2", ...
    map: str  # relative to the project
    bild: str


def angdist_pairs(project_dir: str | os.PathLike, job: str) -> list[AngdistPair]:
    """Every .bild file of a job with its map: the final ones first, then iterations from the last."""
    project_dir = Path(project_dir)
    found = []
    try:
        names = os.listdir(project_dir / job)
    except OSError:
        return []
    for name in names:
        m = _NAME.match(name)
        if m is None:
            continue
        map_name = name[: -len("_angdist.bild")] + ".mrc"
        if not (project_dir / job / map_name).is_file():
            continue
        it = int(m["it"]) if m["it"] else None
        label = "Final" if it is None else f"Iteration {it}"
        if m["half"]:
            label += f" ({m['half'][:4]} {m['half'][4]})"
        label += f", {m['what']} {int(m['n'])}"
        key = (it is not None, -(it or 0), m["half"] or "", int(m["n"]))
        found.append((key, AngdistPair(label, job + map_name, job + name)))
    return [p for _, p in sorted(found, key=lambda kp: kp[0])]
