# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""MultiBody's flexibility analysis (relion_flex_analyse, ``--o <job>analyse``).

It writes ``analyse_projections_along_eigenvectors_all_particles.txt`` (per particle its image name
and its projection on each eigenvector), ``analyse_component<k>_bin<b>.mrc`` (maps along the first
components), ``analyse_logfile.pdf`` and, when asked, ``analyse_eval<k>_select*.star``. How much
variance each component explains it only prints (``+ Component 1 explains 23.4% of variance.`` in
run.out); without that line, it is the variance of the projections, which is the eigenvalue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryo_savanna.results.curves import Curve, Figure
from cryo_savanna.stamps import Cache, stamp

PROJECTIONS = "analyse_projections_along_eigenvectors_all_particles.txt"
_EXPLAINS = re.compile(r"Component\s+(\d+)\s+explains\s+([-+0-9.eE]+)\s*%")
_COMPONENT_MAP = re.compile(r"^analyse_component(\d+)_bin(\d+)\.mrc$")
HISTOGRAMS = 6  # components with a histogram of their projections


@dataclass
class FlexAnalysis:
    variance: np.ndarray  # % of variance explained by each component
    projections: np.ndarray | None  # [particles, components]
    from_log: bool  # the variance as relion_flex_analyse printed it

    @property
    def components(self) -> int:
        return len(self.variance)


def logged_variance(job_dir: Path) -> np.ndarray | None:
    """% of variance per component, as relion_flex_analyse printed it in run.out."""
    try:
        text = (job_dir / "run.out").read_text(errors="replace")
    except OSError:
        return None
    found: dict[int, float] = {}
    for m in _EXPLAINS.finditer(text):
        found[int(m.group(1))] = float(m.group(2))  # the last analysis, if the job ran it again
    if not found:
        return None
    return np.array([found.get(k, np.nan) for k in range(1, max(found) + 1)])


def _projections(path: Path) -> np.ndarray | None:
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append([float(v) for v in parts[1:]])
                except ValueError:
                    continue
    except OSError:
        return None
    if not rows:
        return None
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


_cache = Cache(8)


def read_flex(job_dir: Path) -> FlexAnalysis | None:
    proj_path = job_dir / PROJECTIONS
    key = (str(job_dir), stamp(proj_path), stamp(job_dir / "run.out"))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    proj = _projections(proj_path) if key[1] is not None else None
    var = logged_variance(job_dir)
    from_log = var is not None
    if var is None and proj is not None and len(proj) > 1:
        v = proj.var(axis=0)
        var = 100 * v / v.sum() if v.sum() > 0 else None
    if var is None:
        return None
    out = FlexAnalysis(var, proj, from_log)
    _cache.put(key, out)
    return out


def flex_figures(job_dir: Path) -> list[Figure]:
    fa = read_flex(job_dir)
    if fa is None:
        return []
    k = np.arange(1, fa.components + 1)
    fig = Figure("variance", "Variance explained per component", "Component", "Variance explained (%)")
    fig.curves.append(Curve("Component", k, fa.variance, symbols=True))
    cum = np.nancumsum(fa.variance)
    fig.curves.append(Curve("Cumulative", k, cum, dashed=True))
    first = [f"{v:.1f}%" for v in fa.variance[:3] if np.isfinite(v)]
    if first:
        fig.notes.append("first components: " + ", ".join(first))
    out = [fig]
    if fa.projections is not None:
        for c in range(min(HISTOGRAMS, fa.projections.shape[1])):
            counts, edges = np.histogram(fa.projections[:, c], bins=50)
            h = Figure(f"hist{c + 1}", f"Component {c + 1}", "Projection on the eigenvector", "Particles")
            h.curves.append(Curve("Particles", np.repeat(edges, 2)[1:-1], np.repeat(counts, 2).astype(float)))
            if c < len(fa.variance) and np.isfinite(fa.variance[c]):
                h.notes.append(f"{fa.variance[c]:.1f}% of the variance")
            out.append(h)
    return out


def component_maps(job_dir: Path) -> list[str]:
    """The maps along the components, by component then bin (file names)."""
    found = []
    try:
        for p in job_dir.iterdir():
            m = _COMPONENT_MAP.match(p.name)
            if m:
                found.append(((int(m.group(1)), int(m.group(2))), p.name))
    except OSError:
        return []
    return [name for _, name in sorted(found)]
