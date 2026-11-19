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
from cryo_savanna.results import flex, polish
from cryo_savanna.results.classes import iterations, latest_iteration_maps, particle_iterations
from cryo_savanna.results.ctfrefine import aberration_images, fits_ctf
from cryo_savanna.results.tomo import (
    TOMO_NAME,
    columns_of,
    has_tomograms,
    input_tomograms,
    is_tomo_particles,
    is_warp_particles,
    tomograms_for_particles,
)

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
CTF_REFINE = "ctf_refine"  # CtfRefine: what it fitted, optics before/after, statistics of the changes
CTF_CHANGES = "ctf_changes"  # CtfRefine: per-particle CTF changes, and per micrograph
ABERRATIONS = "aberrations"  # CtfRefine: phase images of beam tilt / trefoil / 4th order, magnification
PYTOM_SCORES = "pytom_scores"  # pytom template matching: progress and score statistics per tomogram
PYTOM_THRESHOLDS = "pytom_thresholds"  # pytom extraction: thresholds per tomogram, chosen interactively
SVG_PLOTS = "svg_plots"  # the plots a program wrote as SVG files (pytom)
TOMO_MASKS = "tomo_masks"  # tomogram masks: drawn interactively on the tomograms
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


def _refinement_map(ctx: JobContext, optimiser: str) -> str:
    """The map of the refinement an optimiser file is from (Refine3D/job011/run_optimiser.star ->
    run_class001.mrc; MultiBody: its first body), if it exists."""
    if not optimiser.endswith("_optimiser.star"):
        return ""
    root = optimiser[: -len("optimiser.star")]
    for name in ("class001.mrc", "body001.mrc"):
        if ctx.exists(root + name):
            return root + name
    return ""


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


def _float(v: str | None) -> float | None:
    try:
        f = float(v or "")
    except ValueError:
        return None
    return f if f > 0 else None


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


ITERATIVE = ("relion.class2d", "relion.class3d", "relion.initialmodel", "relion.refine3d", "relion.multibody")
# jobs whose picks have a tab of their own already (coordinate lists; Extract's "On micrographs")
_OWN_PICKS = ("relion.manualpick", "relion.autopick", "relion.extract")


def job_particles(ctx: JobContext) -> str | None:
    """The particles of a job for its Picks tab: the newest of its particle outputs (a continued job lists
    its earlier ones too)."""
    outs = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "ParticleGroupMetadata" and ctx.exists(n)]
    return max(outs, key=lambda n: (ctx.project_dir / n).stat().st_mtime_ns) if outs else None


def is_spa_particles(path: str | os.PathLike) -> bool:
    """Particles on micrographs (rlnMicrographName and coordinates, no rlnTomoName). Warp particles have
    these too: callers tell them apart with is_warp_particles first."""
    try:
        cols = columns_of(path, "particles") or []
    except OSError:
        return False
    return TOMO_NAME not in cols and "rlnMicrographName" in cols and "rlnCoordinateX" in cols


def _pytom_views(ctx: JobContext, kind: str, add, add_maps) -> None:
    """pytom jobs (jobdefs/pytom_jobs.py): their views come from the files in the job directory, so that
    they show while the job runs (an External job has its outputs only at the end)."""
    from cryo_savanna import pytom_runner as P
    from cryo_savanna.jobdefs.pytom_jobs import decode
    own = decode(ctx.options.get("other_args", ""))[0]
    if kind == P.MATCH:
        summary = ctx.name + P.SUMMARY
        add(PYTOM_SCORES, "Scores", job=ctx.name)
        if ctx.exists(summary):
            add(TOMOGRAMS, "Tomograms", star=summary)
        add_maps([ctx.name + "template.mrc", ctx.name + "mask.mrc"], masks=[ctx.name + "mask.mrc"])
        return
    summary = own.get("in_match", "")
    add(PYTOM_THRESHOLDS, "Thresholds", job=ctx.name, summary=summary, diameter=_float(own.get("particle_diameter")))
    particles = ctx.name + P.PARTICLES
    if ctx.exists(particles) and ctx.exists(summary):
        add(TOMOGRAMS, "Picks", star=summary, picks=particles, diameter=_float(own.get("particle_diameter")))
    if (ctx.job_dir / P.PLOTS_DIR).is_dir():
        add(SVG_PLOTS, "pytom plots", job=ctx.name, folder=P.PLOTS_DIR)
    if ctx.exists(particles):
        add(TABLE_PLOT, "Particle plots", star=particles, block="particles",
            columns=[P.LCC, "rlnCenteredCoordinateZAngst"])


def _aretomo_views(ctx: JobContext, kind: str, add, add_maps) -> None:
    """AreTomo3 jobs (jobdefs/aretomo_jobs.py): the tilt series and tomograms done so far, from the files
    the job rewrites as tilt series finish (so they show while it runs), and its summary table."""
    from cryo_savanna import aretomo_runner as A
    tomograms, aligned, summary = (ctx.name + f for f in (A.TOMOGRAMS, A.ALIGNED, A.SUMMARY))
    if ctx.exists(tomograms) and has_tomograms(ctx.project_dir / tomograms):
        add(TOMOGRAMS, "Tomograms", star=tomograms, aretomo=ctx.name)  # with the alignment thickness to change
    if ctx.exists(aligned) and columns_of(ctx.project_dir / aligned, "global"):
        add(TILT_SERIES, "Tilt series", star=aligned)
    if ctx.exists(summary):
        add(TABLE_PLOT, "AreTomo3", star=summary, block="global",
            columns=[A.COL_THICKNESS, A.COL_AXIS, A.COL_DARK, A.COL_CTF_RES])


def _tomomask_views(ctx: JobContext, kind: str, add, add_maps) -> None:
    """Tomogram mask jobs (jobdefs/tomomask_jobs.py): the mask editor, once the job wrote masks.json."""
    add(TOMO_MASKS, "Masks", job=ctx.name)


# The viewers of the External jobs made from cryo-savanna's own forms, by family (jobdefs/external_jobs.py):
# (ctx, kind, add, add_maps). A CLI tool has none of its own: its output files get the usual ones.
EXTERNAL_VIEWS = {"pytom": _pytom_views, "aretomo": _aretomo_views, "tomomask": _tomomask_views}


def views_for(ctx: JobContext) -> list[ViewSpec]:
    t = base_type(ctx.type_label)
    views: list[ViewSpec] = []
    seen_files: set[str] = set()

    def add(kind: str, title: str, **args) -> None:
        if title == "Picks" and any(v.title == "Picks" for v in views):
            return  # one Picks tab per job, whatever else could show picks
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

    def add_picks(kind: str, **args) -> None:
        """The Picks tab, before the particles when the job's own views listed them already (CtfRefine)."""
        n = len(views)
        add(kind, "Picks", **args)
        at = next((i for i, v in enumerate(views[:n]) if v.kind == PARTICLES), None)
        if len(views) > n and at is not None:
            views.insert(at, views.pop())

    density = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "DensityMap"]
    listed: list[list] = []

    def job_iterations() -> list:
        """iterations() of the job, listed once: this runs in the window's thread at every refresh."""
        if not listed:
            listed.append(iterations(ctx.job_dir))
        return listed[0]

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
        if job_iterations():
            add(CLASSES, "Classes", job=ctx.name, tool="classes")
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
    elif t in ("relion.class3d", "relion.initialmodel"):
        if job_iterations():
            add(CLASSES, "Classes", job=ctx.name, tool="classes")
        add_job_maps(density, ctx.options.get("fn_mask", ""))
        if job_iterations():
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
            add(ANGLES, "Angular distribution", job=ctx.name)
    if t in ("relion.refine3d", "relion.multibody"):
        # the full map before the half maps
        add_job_maps(sorted(density, key=lambda n: "half" in n), ctx.options.get("fn_mask", ""))
        if job_iterations() or ctx.file("run_model.star"):
            add(PLOTS, "FSC", job=ctx.name, figures="model_fsc")
            add(PLOTS, "Convergence", job=ctx.name, figures="convergence")
            add(ANGLES, "Angular distribution", job=ctx.name)
        if t == "relion.multibody":
            if ctx.file(flex.PROJECTIONS):
                add(PLOTS, "Flexibility", job=ctx.name, figures="flex")
            components = flex.component_maps(ctx.job_dir)
            if components:
                maps = [ctx.name + n for n in components]
                add(VOLUME, "Components", maps=maps, mask="", color="", masks=[])
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
        if ctx.exists(locres):
            add(PLOTS, "Local resolution", figures="locres", star=locres, mask=_option_file(ctx, "fn_mask") or "")
    elif t == "relion.polish":
        if ctx.file(polish.BFACTORS):
            add(PLOTS, "Frame weights", figures="polish", job=ctx.name)
        shiny = ctx.output("ParticleGroupMetadata")
        if shiny and is_spa_particles(ctx.project_dir / shiny):
            # the particles on their micrographs, with the tracks of their motion
            add(MICROGRAPHS, "Picks", star=_option_file(ctx, "fn_mic") or "", picks=shiny, tracks=ctx.name)
            seen_files.discard(shiny)  # still gets its Particles tab below
    elif t == "relion.subtract":
        mask = _option_file(ctx, "fn_mask") or ""
        ref = _refinement_map(ctx, ctx.options.get("fn_opt", ""))
        if mask:
            # what is kept: the mask on the map the particles were subtracted from
            add_maps([ref, mask] if ref else [mask], mask=mask, masks=[mask])
    elif t in ("relion.ctfrefinetomo", "relion.framealigntomo"):
        tomos = ctx.output("TomogramGroupMetadata")
        if tomos:
            add(TILT_SERIES, "Tilt series", star=tomos,
                compare=input_tomograms(ctx.project_dir, ctx.options) or "")
    elif t == "relion.reconstructparticletomo":
        add_maps([m for m in (ctx.file(n) for n in ("merged.mrc", "half1.mrc", "half2.mrc")) if m])
        h1, h2 = ctx.file("half1.mrc"), ctx.file("half2.mrc")
        if h1 and h2:
            add(PLOTS, "FSC", figures="halfmap_fsc", star=h2, half1=h1, half2=h2)
    elif t == "relion.select":
        if interactive_kind(ctx.type_label, ctx.options) == "classes":
            add(SELECTION, "Selection", job=ctx.name, what="classes")
        avg = ctx.file("class_averages.star")
        if avg:
            add(CLASSES, "Selected classes", star=avg)
    elif t == "relion.ctfrefine":
        parts = ctx.output("ParticleGroupMetadata") or ""
        source = _option_file(ctx, "fn_data") or ""
        add(CTF_REFINE, "CTF refinement", job=ctx.name, star=parts, input=source)
        if parts and fits_ctf(ctx.options):
            add(CTF_CHANGES, "Per-particle CTF", star=parts, input=source)
        o = ctx.options
        # the images come at the end of the job: while it runs, a tab that waits for them
        if aberration_images(ctx.project_dir, ctx.name) or (
                ctx.status == "Running" and any(o.get(k) == "Yes" for k in ("do_aniso_mag", "do_tilt", "do_4thorder"))):
            add(ABERRATIONS, "Aberrations", job=ctx.name)
        if parts:
            add(PARTICLES, "Particles", star=parts)
    elif t == "relion.excludetilts" and interactive_kind(ctx.type_label, ctx.options) == "tilts":
        add(SELECTION, "Selection", job=ctx.name, what="tilts")
    elif t == "relion.external":
        from cryo_savanna.jobdefs.external_jobs import kind_of_exe
        found = kind_of_exe(ctx.options.get("fn_exe", ""))
        if found is not None and found[0].name in EXTERNAL_VIEWS:
            EXTERNAL_VIEWS[found[0].name](ctx, found[1], add, add_maps)

    if t in ITERATIVE:
        # the particles of each iteration that wrote them, the latest shown (and followed while the job
        # runs), then the final ones; with the classes to show when they carry several. The tab does not
        # change with the iterations (the viewer lists them), so it is not built again at each one.
        final = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "ParticleGroupMetadata"]
        its = particle_iterations(ctx.project_dir, ctx.name, final, job_iterations())
        newest = ctx.project_dir / its[-1][1] if its else None
        if newest is not None and is_warp_particles(newest):
            # Warp particles: on the tomograms Warp reconstructed (no RELION tomogram set)
            add_picks(TOMOGRAMS, star="", picks=its[-1][1], diameter=tomo_pick_diameter(ctx))
            seen_files.difference_update([its[-1][1]])
        elif newest is not None and is_spa_particles(newest):
            # no micrograph list in a particle file: the micrographs it names
            add_picks(MICROGRAPHS, job=ctx.name, final=final, diameter=_float(ctx.options.get("particle_diameter")))
            seen_files.difference_update(final)  # they still get their Particles tab below
    elif t not in _OWN_PICKS:
        # the particles of any other job on their micrographs (Warp particles: on the tomograms Warp
        # reconstructed), with the classes to show when they carry several; RELION tomography particles get
        # theirs below, as any tomogram particles do
        parts = job_particles(ctx)
        listed_parts = parts in seen_files
        if parts and is_warp_particles(ctx.project_dir / parts):
            add_picks(TOMOGRAMS, star="", picks=parts, diameter=tomo_pick_diameter(ctx))
        elif parts and is_spa_particles(ctx.project_dir / parts):
            add_picks(MICROGRAPHS, picks=parts, diameter=_float(ctx.options.get("particle_diameter")))
        else:
            listed_parts = True
        if not listed_parts:
            seen_files.discard(parts)  # still gets its Particles tab below

    # anything else a viewer can show; of several tomogram particle files (a continued job lists its
    # earlier ones too), the newest gets the Picks tab
    tomo_parts = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "ParticleGroupMetadata"
                  and n not in seen_files and ctx.exists(n) and is_tomo_particles(ctx.project_dir / n)]
    newest_parts = max(tomo_parts, key=lambda n: (ctx.project_dir / n).stat().st_mtime_ns, default=None)
    warp_parts = [n for n, nt in ctx.outputs if nt.split(".", 1)[0] == "ParticleGroupMetadata"
                  and n not in seen_files and n not in tomo_parts and ctx.exists(n)
                  and is_warp_particles(ctx.project_dir / n)]
    newest_warp = max(warp_parts, key=lambda n: (ctx.project_dir / n).stat().st_mtime_ns, default=None)
    for node, nt in ctx.outputs:
        kind = nt.split(".", 1)[0]
        if node in seen_files or not ctx.exists(node):
            continue
        if kind == "TomogramGroupMetadata":
            if has_tomograms(ctx.project_dir / node):
                add(TOMOGRAMS, "Tomograms", star=node)
            else:
                add(TILT_SERIES, "Tilt series", star=node, tool="tilts", tool_arg=node)
        elif node in tomo_parts:
            tomograms = tomograms_for_particles(ctx.project_dir, node, ctx.options) if node == newest_parts else None
            if tomograms:
                add(TOMOGRAMS, "Picks", star=tomograms, picks=node, diameter=tomo_pick_diameter(ctx))
        elif kind == "ParticleGroupMetadata":
            if node == newest_warp:
                add(TOMOGRAMS, "Picks", star="", picks=node, diameter=tomo_pick_diameter(ctx))
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
