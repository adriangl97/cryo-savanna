# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Layout of the job graph (no Qt, so it can be tested and timed without a display).

The layout follows the main data path, the same parent-child links as the job tree. Every job has
at most one such parent, so these links form trees, laid out top to bottom as tidy trees: each
subtree is pushed as close to its left sibling as their outlines allow, row by row, and a parent is
centred over its children. Other links (references, masks, ...) get no bend points; the view draws
them on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryo_savanna.project.lineage import cut_loops
from cryo_savanna.project.snapshot import ProjectSnapshot

NODE_W, NODE_H = 210.0, 46.0
LAYER_GAP = 44.0  # vertical space between rows of jobs, for the edges
NODE_GAP = 16.0  # horizontal space between jobs in one row
COMPONENT_GAP = 60.0
ARROW = 6.0


@dataclass
class GraphLayout:
    pos: dict[str, tuple[float, float]]  # job -> centre
    routes: dict[tuple[str, str], list[tuple[float, float]]]  # (parent, child) -> bend points between them
    primary: set[tuple[str, str]]  # main-data-path links; all other links have no bend points


def _tidy_tree(root: str, kids: dict[str, list[str]], width: dict[str, float]) -> dict[str, float]:
    """x of every job in the tree under ``root``, relative to ``root``.

    Bottom-up: a subtree is its job's x offsets plus its outline, the (left, right) extent of each
    row below its root. Children are added left to right, each moved right just enough to clear
    the outline of the ones before it in every row they share.
    """
    order, stack = [], [root]
    while stack:  # no recursion: chains of jobs can be long
        n = stack.pop()
        order.append(n)
        stack.extend(kids.get(n, []))
    shapes: dict[str, tuple[dict[str, float], list[list[float]]]] = {}
    for n in reversed(order):
        w = width[n]
        offsets: dict[str, float] = {}
        outline: list[list[float]] = []
        roots: list[float] = []
        for c in kids.get(n, []):
            c_offsets, c_outline = shapes.pop(c)
            shift = max((outline[d][1] + NODE_GAP - c_outline[d][0] for d in range(min(len(outline), len(c_outline)))),
                        default=0.0)
            roots.append(shift)
            for k, x in c_offsets.items():
                offsets[k] = x + shift
            for d, (left, right) in enumerate(c_outline):
                if d < len(outline):
                    outline[d][1] = right + shift
                else:
                    outline.append([left + shift, right + shift])
        mid = (roots[0] + roots[-1]) / 2 if roots else 0.0
        offsets = {k: x - mid for k, x in offsets.items()}
        offsets[n] = 0.0
        shapes[n] = (offsets, [[-w / 2, w / 2]] + [[left - mid, right - mid] for left, right in outline])
    return shapes[root][0]


def layout_graph(snap: ProjectSnapshot, sizes: dict[str, tuple[float, float]] | None = None) -> GraphLayout:
    """Layered layout of the main data path, top to bottom. Separate trees are placed side by side.

    Top to bottom suits RELION projects: long chains with few branches per level. ``sizes`` gives
    each job's card (width, height); the default is NODE_W x NODE_H.
    """
    names = list(snap.jobs)
    size = {n: (sizes or {}).get(n, (NODE_W, NODE_H)) for n in names}
    primary: set[tuple[str, str]] = set()
    routes: dict[tuple[str, str], list[tuple[float, float]]] = {}
    parent: dict[str, str] = {}
    for n, info in snap.jobs.items():
        if info.primary_parent in size and info.primary_parent != n:
            parent[n] = info.primary_parent
        for p in info.other_parents:
            if p in size:
                routes[(p, n)] = []
    # a loop of parents (never in a sound pipeline) is cut where it closes, as in the job list
    parent = {n: p for n, p in cut_loops({n: parent.get(n) for n in names}).items() if p is not None}
    kids: dict[str, list[str]] = {}
    for n in names:  # pipeline order, so the layout is stable between refreshes
        if n in parent:
            kids.setdefault(parent[n], []).append(n)
            primary.add((parent[n], n))
            routes[(parent[n], n)] = []  # one row apart: a straight curve, no bend points

    pos: dict[str, tuple[float, float]] = {}
    width = {n: w for n, (w, _) in size.items()}
    x_off = 0.0
    for root in (n for n in names if n not in parent):
        xs = _tidy_tree(root, kids, width)
        depth, stack = {root: 0}, [root]
        while stack:
            n = stack.pop()
            for c in kids.get(n, []):
                depth[c] = depth[n] + 1
                stack.append(c)
        rows = [0.0] * (max(depth.values()) + 1)
        for n, d in depth.items():
            rows[d] = max(rows[d], size[n][1])
        tops = [0.0]
        for h in rows[:-1]:
            tops.append(tops[-1] + h + LAYER_GAP)
        left = min(x - width[n] / 2 for n, x in xs.items())
        right = max(x + width[n] / 2 for n, x in xs.items())
        for n, x in xs.items():
            d = depth[n]
            pos[n] = (x - left + x_off, tops[d] + rows[d] / 2)
        x_off += (right - left) + COMPONENT_GAP
    return GraphLayout(pos, routes, primary)
