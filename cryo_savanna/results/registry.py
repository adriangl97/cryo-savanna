# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Which viewers a job gets, from its type and its output files.

Each view is a ``ViewSpec``: a viewer kind (the widget the UI builds) plus what it reads. Specific
job types come first (CtfFind gets its micrographs with CTF values and plots of them), then any
output a viewer can show (a particle file from any job gets the particle viewer), then "Files"
for everything in the job directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cryo_savanna.project.summary import base_type
from cryo_savanna.project.selection import interactive_kind
from cryo_savanna.results.classes import iterations, latest_iteration_maps
from cryo_savanna.results.tomo import has_tomograms, is_tomo_particles, tomograms_for_particles

# viewer kinds
MICROGRAPHS = "micrographs"  # list + image (+ CTF spectrum / motion track / picks)
CLASSES = "classes"  # gallery of class averages (2D) or class central sections (3D)
PARTICLES = "particles"  # paged particle grid
PLOTS = "plots"  # fixed figures (FSC, Guinier, convergence)
TABLE_PLOT = "table_plot"  # histograms / scatter of the columns of one STAR table
ANGLES = "angles"  # angular distribution
VOLUME = "volume"  # 3D map: sections, projections, surface (+ mask outline, local-resolution colour)
TILT_SERIES = "tilt_series"  # tomography: tilt series, their tilts and per-tilt values
TOMOGRAMS = "tomograms"  # tomography: Z slabs and side views of tomograms (+ picked particles)
SELECTION = "selection"  # what an interactive Select / Exclude tilt images job chose, read-only
FILES = "files"  # every file of the job: STAR tables, images, stacks


@dataclass
class ViewSpec:
    kind: str
    title: str
    args: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.title}"


@dataclass
class JobContext:
    project_dir: Path
    name: str  # "CtfFind/job003/"
    type_label: str
    status: str = ""
    outputs: list[tuple[str, str]] = field(default_factory=list)  # (node, node type)
    options: dict[str, str] = field(default_factory=dict)  # job.star

    @property
    def job_dir(self) -> Path:
        return self.project_dir / self.name

    def exists(self, rel: str) -> bool:
        return bool(rel) and (self.project_dir / rel).is_file()

    def output(self, kind: str, *, ending: str = "") -> str | None:
        """First output node of a kind (e.g. "ParticleGroupMetadata") whose file exists."""
        for node, t in self.outputs:
            if t.split(".", 1)[0] == kind and node.endswith(ending) and self.exists(node):
                return node
        return None

    def file(self, name: str) -> str | None:
        rel = self.name + name
        return rel if self.exists(rel) else None


def _option_file(ctx: JobContext, *keys: str) -> str | None:
    for k in keys:
        v = ctx.options.get(k, "")
        if v and ctx.exists(v):
            return v
    return None


def pick_diameter(ctx: JobContext) -> float | None:
    """Circle size for picks, in Å, from the picking job's settings."""
    o = ctx.options
    t = base_type(ctx.type_label)
    keys = []
    if t == "relion.manualpick":
        keys = ["diameter"]
    elif t == "relion.autopick":
        keys = (["topaz_particle_diameter"] if o.get("do_topaz") == "Yes" else []) + ["log_diam_max"]
    for k in keys:
        try:
            v = float(o.get(k, ""))
        except ValueError:
            continue
        if v > 0:
            return v
    return None


def tomo_pick_diameter(ctx: JobContext) -> float | None:
    """Circle size for tomogram particles (Å): the job's particle diameter, else half the box."""
    for key, scale in (("particle_diameter", 1.0), ("box_size_angst", 0.5)):
        try:
            v = float(ctx.options.get(key, ""))
        except ValueError:
            continue
        if v > 0:
            return v * scale
    return None


def views_for(ctx: JobContext) -> list[ViewSpec]:
    t = base_type(ctx.type_label)
    views: list[ViewSpec] = []
    seen_files: set[str] = set()

    def add(kind: str, title: str, **args) -> None:
        views.append(ViewSpec(kind, title, args))
        for v in args.values():
            if isinstance(v, str):
                seen_files.add(v)
            elif isinstance(v, list):
                seen_files.update(x for x in v if isinstance(x, str))

    def add_maps(maps: list[str], mask: str = "", color: str = "", masks: list[str] | None = None,
                 **extra) -> None:
        maps = [m for m in maps if ctx.exists(m)]
        if maps:
            add(VOLUME, "Maps" if len(maps) > 1 else "Map", maps=maps, mask=mask if ctx.exists(mask) else "",
                color=color if ctx.exists(color) else "", masks=list(masks or []), **extra)

    density = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "DensityMap"]

    def add_job_maps(maps: list[str], mask: str) -> None:
        """An iterative job's maps: while it runs (or when it stopped before writing its final maps), those
        of its latest iteration, followed by the viewer as new iterations come; else its output maps.
        Either way the viewer also offers every earlier iteration (``history``)."""
        if ctx.status == "Running" or not any(ctx.exists(m) for m in maps):
            latest = latest_iteration_maps(ctx.project_dir, ctx.name)
            if latest:
                seen_files.update(maps)  # the final maps come with the next look, once the job finished
                add(VOLUME, "Maps" if len(latest) > 1 else "Map", job=ctx.name, latest=True, history=True,
                    mask=mask if ctx.exists(mask) else "")
                return
        add_maps(maps, mask=mask, job=ctx.name, history=True)

    if t == "relion.import":
        movies = ctx.output("MicrographMovieGroupMetadata")
        mics = ctx.output("MicrographGroupMetadata")
        if movies:
            add(MICROGRAPHS, "Movies", star=movies)
        if mics:
            add(MICROGRAPHS, "Micrographs", star=mics)
    elif t == "relion.motioncorr":
        mics = ctx.output("MicrographGroupMetadata")
        if mics:
            add(MICROGRAPHS, "Micrographs", star=mics, motion=True)
            add(TABLE_PLOT, "Motion plots", star=mics, block="micrographs",
                columns=["rlnAccumMotionTotal", "rlnAccumMotionEarly", "rlnAccumMotionLate"])
    elif t == "relion.ctffind":
        mics = ctx.output("MicrographGroupMetadata")
        if mics:
            add(MICROGRAPHS, "Micrographs", star=mics, ctf=True)
            add(TABLE_PLOT, "CTF plots", star=mics, block="micrographs",
                columns=["rlnDefocusU", "rlnCtfAstigmatism", "rlnCtfMaxResolution", "rlnCtfFigureOfMerit"],
                scatter=("rlnDefocusU", "rlnCtfMaxResolution"))
    elif t in ("relion.manualpick", "relion.autopick"):
        coords = ctx.output("MicrographCoordsGroup")
        if coords:
            add(MICROGRAPHS, "Picks", star=_option_file(ctx, "fn_in", "fn_input_autopick") or "",
                picks=coords, diameter=pick_diameter(ctx))
    elif t == "relion.extract":
        parts = ctx.output("ParticleGroupMetadata")
        if parts:
            add(PARTICLES, "Particles", star=parts)
            add(MICROGRAPHS, "On micrographs", star=_option_file(ctx, "star_mics") or "", picks=parts,
                box=ctx.options.get("extract_size", ""))
            add(TABLE_PLOT, "Plots", star=parts, block="particles")
    elif t == "relion.class2d":
        if iterations(ctx.job_dir):
            add(CLASSES, "Classes", job=ctx.name, tool="classes")
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
    elif t in ("relion.class3d", "relion.initialmodel"):
        if iterations(ctx.job_dir):
            add(CLASSES, "Classes", job=ctx.name, tool="classes")
        add_job_maps(density, ctx.options.get("fn_mask", ""))
        if iterations(ctx.job_dir):
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
            add(ANGLES, "Angular distribution", job=ctx.name)
    elif t in ("relion.refine3d", "relion.multibody"):
        # the full map before the half maps
        add_job_maps(sorted(density, key=lambda n: "half" in n), ctx.options.get("fn_mask", ""))
        if iterations(ctx.job_dir) or ctx.file("run_model.star"):
            add(PLOTS, "FSC", job=ctx.name, figures="model_fsc")
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
            add(ANGLES, "Angular distribution", job=ctx.name)
    elif t == "relion.maskcreate":
        mask = ctx.output("Mask3D") or ""
        source = _option_file(ctx, "fn_in")
        add_maps([source, mask] if source else [mask], mask=mask, masks=[mask])
    elif t == "relion.postprocess":
        pp = ctx.file("postprocess.star")
        if pp:
            add(PLOTS, "FSC", star=pp, figures="postprocess")
        add_maps(sorted(density, key=lambda n: "masked" not in n), mask=ctx.options.get("fn_mask", ""))
    elif t == "relion.localres":
        locres = next((n for n, nt in ctx.outputs if nt.startswith("Image3D")), "")
        add_maps(density or [ctx.options.get("fn_in", "")], mask=ctx.options.get("fn_mask", ""), color=locres)
    elif t == "relion.select":
        if interactive_kind(ctx.type_label, ctx.options) == "classes":
            add(SELECTION, "Selection", job=ctx.name, what="classes")
        avg = ctx.file("class_averages.star")
        if avg:
            add(CLASSES, "Selected classes", star=avg)
    elif t == "relion.excludetilts" and interactive_kind(ctx.type_label, ctx.options) == "tilts":
        add(SELECTION, "Selection", job=ctx.name, what="tilts")

    # anything else a viewer can show
    for node, nt in ctx.outputs:
        kind = nt.split(".", 1)[0]
        if node in seen_files or not ctx.exists(node):
            continue
        if kind == "TomogramGroupMetadata":
            if has_tomograms(ctx.project_dir / node):
                add(TOMOGRAMS, "Tomograms", star=node)
            else:
                add(TILT_SERIES, "Tilt series", star=node, tool="tilts", tool_arg=node)
        elif kind == "ParticleGroupMetadata" and is_tomo_particles(ctx.project_dir / node):
            tomograms = tomograms_for_particles(ctx.project_dir, node, ctx.options)
            if tomograms:
                add(TOMOGRAMS, "Picks", star=tomograms, picks=node, diameter=tomo_pick_diameter(ctx))
        elif kind == "ParticleGroupMetadata":
            add(PARTICLES, "Particles" if not any(v.title == "Particles" for v in views) else os.path.basename(node),
                star=node)
        elif kind == "MicrographGroupMetadata":
            add(MICROGRAPHS, "Micrographs", star=node)
        elif kind == "MicrographMovieGroupMetadata":
            add(MICROGRAPHS, "Movies", star=node)
        elif kind == "Image2DGroupMetadata" and node.endswith(".star"):
            add(CLASSES, "Images", star=node)
    other_maps = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] in ("DensityMap", "Mask3D", "Image3D")
                  and n not in seen_files and ctx.exists(n) and n.endswith((".mrc", ".map"))]
    add_maps(other_maps, masks=[n for n, nt in ctx.outputs if nt.startswith("Mask3D")])
    add(FILES, "Files", job=ctx.name)
    return views


def context_for(project_dir: Path, info, pipeline, options: dict[str, str] | None = None) -> JobContext:
    """A JobContext from the snapshot's JobInfo and the pipeline's nodes."""
    pr = info.process
    outs = [(n, pipeline.nodes[n].type_label if n in pipeline.nodes else "") for n in pr.outputs]
    return JobContext(Path(project_dir), pr.name, pr.type_label, info.status, outs, dict(options or {}))
