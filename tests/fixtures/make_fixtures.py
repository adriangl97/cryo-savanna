# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Synthetic RELION 5.1 projects for tests and for trying out the GUI without real data.

Two projects are modelled on the public tutorials:
  * spa:  the beta-galactosidase SPA tutorial (Import -> ... -> Refine3D -> PostProcess -> Polish)
  * tomo: the HIV immature capsid subtomogram tutorial (ImportTomo -> ... -> Refine3D -> PostProcess)

Job directories, job.star files, output node files and default_pipeline.star follow what RELION
5.1 writes (node names and labels taken from src/pipeline_jobs.cpp). Images are tiny and
synthetic. The pipelines also contain Running, Scheduled, Failed and Aborted jobs and aliases.

    python -m tests.fixtures.make_fixtures OUTDIR      # writes OUTDIR/spa and OUTDIR/tomo
"""

from __future__ import annotations

import dataclasses
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.model import JobDefs
from cryo_savanna.project.lock import PipelineLock
from cryo_savanna.project.pipeline import Node, Pipeline, Process
from cryo_savanna.project.star import StarBlock, format_star, read_star, write_star
from cryo_savanna.relion_consts import (
    JOB_TYPE_BY_LABEL,
    RELION_JOB_EXIT_ABORTED,
    RELION_JOB_EXIT_FAILURE,
    RELION_JOB_EXIT_SUCCESS,
)

PDF_PLACEHOLDER = b"%PDF-1.4\n% cryo-savanna test fixture placeholder\n%%EOF\n"


@dataclass
class JobSpec:
    num: int
    label: str
    outputs: list[tuple[str, str]]  # (file relative to the job dir, node type label)
    inputs: list[str] = field(default_factory=list)  # node names (outputs of earlier jobs)
    options: dict[str, str] = field(default_factory=dict)
    tomo: bool = False
    alias: str | None = None
    status: str = "Succeeded"
    write: Callable[[Path, "Ctx"], None] | None = None  # writes the job's data files

    @property
    def dirname(self) -> str:
        return JOB_TYPE_BY_LABEL[self.label].dirname

    @property
    def name(self) -> str:
        return f"{self.dirname}/job{self.num:03d}/"


@dataclass
class Ctx:
    root: Path
    rng: np.random.Generator
    mics: list[str] = field(default_factory=list)
    tomos: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- small helpers

def star(path: Path, *blocks: StarBlock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_star(blocks))


def mrc(path: Path, data: np.ndarray, voxel: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data.astype(np.float32))
        m.voxel_size = voxel


def blob_volume(n: int, rng: np.random.Generator, centres: int = 6) -> np.ndarray:
    z, y, x = np.mgrid[:n, :n, :n] - n / 2
    vol = np.zeros((n, n, n), np.float32)
    for _ in range(centres):
        c = rng.normal(0, n / 8, 3)
        vol += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * (n / 10) ** 2))
    return vol


def projection_images(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[:n, :n] - n / 2
    out = np.empty((k, n, n), np.float32)
    for i in range(k):
        img = np.zeros((n, n), np.float32)
        for _ in range(4):
            c = rng.normal(0, n / 7, 2)
            img += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * (n / 9) ** 2))
        out[i] = img + rng.normal(0, 0.05, (n, n))
    return out


def optics_block(name: str = "optics", image: bool = False, pixel: float = 0.885, box: int = 64) -> StarBlock:
    cols = ["rlnOpticsGroupName", "rlnOpticsGroup", "rlnMicrographOriginalPixelSize", "rlnVoltage",
            "rlnSphericalAberration", "rlnAmplitudeContrast"]
    row: list = ["opticsGroup1", 1, 0.885, 200.0, 1.4, 0.1]
    if image:
        cols += ["rlnImagePixelSize", "rlnImageSize", "rlnImageDimensionality", "rlnCtfDataAreCtfPremultiplied"]
        row += [pixel, box, 2, 0]
    else:
        cols += ["rlnMicrographPixelSize"]
        row += [0.885]
    return StarBlock.from_records(name, cols, [row])


def model_star(path: Path, k: int, ref_stack: str, rng: np.random.Generator, resolution: float, dim: int = 2) -> None:
    dist = rng.dirichlet(np.ones(k))
    gen = StarBlock.from_dict("model_general", {
        "rlnReferenceDimensionality": dim, "rlnDataDimensionality": 2, "rlnOriginalImageSize": 64,
        "rlnCurrentResolution": resolution, "rlnCurrentImageSize": 64, "rlnPixelSize": 3.54,
        "rlnNrClasses": k, "rlnNrBodies": 1, "rlnNrGroups": 1, "rlnTau2FudgeFactor": 2.0,
        "rlnNormCorrectionAverage": 0.9, "rlnLogLikelihood": float(rng.normal(-1e7, 1e5)),
    })
    if dim == 2:
        refs = [f"{i + 1:06d}@{ref_stack}" for i in range(k)]
    else:
        refs = [f"{ref_stack}_class{i + 1:03d}.mrc" for i in range(k)]
    classes = StarBlock.from_records(
        "model_classes",
        ["rlnReferenceImage", "rlnClassDistribution", "rlnAccuracyRotations", "rlnAccuracyTranslationsAngst",
         "rlnEstimatedResolution", "rlnOverallFourierCompleteness", "rlnClassPriorOffsetX", "rlnClassPriorOffsetY"],
        [[refs[i], float(dist[i]), float(rng.uniform(2, 20)), float(rng.uniform(1, 5)),
          float(rng.uniform(resolution, 3 * resolution)), float(rng.uniform(0.8, 1.0)), 0.0, 0.0] for i in range(k)],
    )
    star(path, gen, classes)


def particles_star(path: Path, ctx: Ctx, n_per_mic: int, stack_dir: str, extra_cols: dict | None = None) -> None:
    rows = []
    for mic in ctx.mics:
        stem = Path(mic).stem
        for i in range(n_per_mic):
            rows.append([
                float(ctx.rng.uniform(50, 3700)), float(ctx.rng.uniform(50, 3700)), 0.0,
                f"{i + 1:06d}@{stack_dir}/Movies/{stem}.mrcs", mic, 1,
                float(ctx.rng.uniform(9000, 25000)), float(ctx.rng.uniform(9000, 25000)), float(ctx.rng.uniform(0, 180)),
            ])
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnAnglePsi", "rlnImageName", "rlnMicrographName",
            "rlnOpticsGroup", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]
    if extra_cols:
        for c, fn in extra_cols.items():
            cols.append(c)
            for r in rows:
                r.append(fn(ctx.rng))
    star(path, optics_block(image=True, pixel=3.54, box=64), StarBlock.from_records("particles", cols, rows))


# --------------------------------------------------------------------------- SPA data writers

def w_import_movies(d: Path, ctx: Ctx) -> None:
    movies = [f"Movies/20170629_{i:05d}_frameImage.tiff" for i in range(21, 45)]
    star(d / "movies.star", optics_block(),
         StarBlock.from_records("movies", ["rlnMicrographMovieName", "rlnOpticsGroup"], [[m, 1] for m in movies]))


def w_motioncorr(d: Path, ctx: Ctx) -> None:
    ctx.mics = [f"MotionCorr/job002/Movies/20170629_{i:05d}_frameImage.mrc" for i in range(21, 45)]
    rows = []
    for m in ctx.mics:
        early, late = float(ctx.rng.uniform(5, 25)), float(ctx.rng.uniform(1, 6))
        rows.append([m, m[:-4] + ".star", 1, early + late, early, late])
        mrc(ctx.root / m, ctx.rng.normal(0, 1, (128, 128)), 0.885 * 32)
    star(d / "corrected_micrographs.star", optics_block(), StarBlock.from_records(
        "micrographs",
        ["rlnMicrographName", "rlnMicrographMetadata", "rlnOpticsGroup", "rlnAccumMotionTotal",
         "rlnAccumMotionEarly", "rlnAccumMotionLate"], rows))
    (d / "logfile.pdf").write_bytes(PDF_PLACEHOLDER)


def w_ctffind(d: Path, ctx: Ctx) -> None:
    rows = []
    for m in ctx.mics:
        du = float(ctx.rng.uniform(8000, 26000))
        dv = du - float(ctx.rng.uniform(50, 600))
        rows.append([m, 1, m.replace("MotionCorr/job002", "CtfFind/job003")[:-4] + ".ctf:mrc", du, dv, du - dv,
                     float(ctx.rng.uniform(0, 180)), float(ctx.rng.uniform(0.05, 0.3)), float(ctx.rng.uniform(3, 8))])
    star(d / "micrographs_ctf.star", optics_block(), StarBlock.from_records(
        "micrographs",
        ["rlnMicrographName", "rlnOpticsGroup", "rlnCtfImage", "rlnDefocusU", "rlnDefocusV", "rlnCtfAstigmatism",
         "rlnDefocusAngle", "rlnCtfFigureOfMerit", "rlnCtfMaxResolution"], rows))
    star(d / "power_spectra_fits.star", StarBlock.from_records(
        "images", ["rlnImageName"], [[r[2]] for r in rows]))
    (d / "logfile.pdf").write_bytes(PDF_PLACEHOLDER)


def w_autopick(d: Path, ctx: Ctx) -> None:
    rows = []
    for m in ctx.mics:
        coords = f"AutoPick/job004/Movies/{Path(m).stem}_autopick.star"
        rows.append([m, coords])
        n = int(ctx.rng.integers(30, 80))
        star(ctx.root / coords, StarBlock.from_records(
            "", ["rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit", "rlnClassNumber", "rlnAnglePsi"],
            [[float(x), float(y), float(f), 0, -999.0] for x, y, f in
             zip(ctx.rng.uniform(50, 3700, n), ctx.rng.uniform(50, 3700, n), ctx.rng.normal(0.5, 0.3, n))]))
    star(d / "autopick.star", StarBlock.from_records(
        "coordinate_files", ["rlnMicrographName", "rlnMicrographCoordinates"], rows))
    (d / "logfile.pdf").write_bytes(PDF_PLACEHOLDER)


def w_extract(d: Path, ctx: Ctx) -> None:
    particles_star(d / "particles.star", ctx, 40, "Extract/job005",
                   {"rlnAutopickFigureOfMerit": lambda r: float(r.normal(0.5, 0.3))})


def _w_class2d(job: str, k: int, iters: list[int]) -> Callable[[Path, Ctx], None]:
    def w(d: Path, ctx: Ctx) -> None:
        for it in iters:
            stack = f"{job}run_it{it:03d}_classes.mrcs"
            model_star(d / f"run_it{it:03d}_model.star", k, stack, ctx.rng, resolution=40.0 - 0.1 * it)
        last = iters[-1]
        mrc(ctx.root / f"{job}run_it{last:03d}_classes.mrcs", projection_images(k, 48, ctx.rng), 3.54)
        particles_star(d / f"run_it{last:03d}_data.star", ctx, 40, "Extract/job005",
                       {"rlnClassNumber": lambda r: int(r.integers(1, k + 1)),
                        "rlnMaxValueProbDistribution": lambda r: float(r.uniform(0.1, 1.0))})
        star(d / f"run_it{last:03d}_optimiser.star", StarBlock.from_dict("optimiser_general", {
            "rlnOutputRootName": f"{job}run", "rlnModelStarFile": f"{job}run_it{last:03d}_model.star",
            "rlnExperimentalDataStarFile": f"{job}run_it{last:03d}_data.star"}))
    return w


def w_select_2d(d: Path, ctx: Ctx) -> None:
    particles_star(d / "particles.star", ctx, 25, "Extract/job005")
    star(d / "class_averages.star", StarBlock.from_records(
        "", ["rlnReferenceImage"], [[f"{i:06d}@Class2D/job006/run_it200_classes.mrcs"] for i in (3, 7, 12, 18, 22)]))
    star(d / "backup_selection.star", StarBlock.from_records(
        "", ["rlnSelected"], [[1 if i in (3, 7, 12, 18, 22) else 0] for i in range(1, 51)]))


def w_inimodel(d: Path, ctx: Ctx) -> None:
    mrc(d / "initial_model.mrc", blob_volume(32, ctx.rng), 3.54)


def angdist_bild(path: Path, box: int, angpix: float, n: int = 192) -> None:
    """An angular-distribution .bild file as RELION writes it (writeBildFileOrientationalDistribution):
    a cylinder per direction with particles, from R = box * angpix / 2 out to at most 1.3 R, blue to red.
    Its own random numbers (from the file name), so the other fixture files stay as they were."""
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))
    i = np.arange(n) + 0.5
    tilt = np.arccos(1 - 2 * i / n)  # directions spread evenly on the sphere
    rot = np.pi * (1 + 5 ** 0.5) * i
    v = np.stack([np.cos(rot) * np.sin(tilt), np.sin(rot) * np.sin(tilt), np.cos(tilt)], axis=1)
    pdf = np.exp(-((np.degrees(tilt) - 90) / 25) ** 2) + 0.2 * rng.random(n)  # side views preferred
    pdf[rng.random(n) < 0.1] = 0  # some directions without particles
    pdf /= pdf.sum()
    r = box * angpix / 2
    width = 0.5 * np.pi * r * (np.sqrt(41253 / n) / 360)
    mean, sigma = pdf.mean(), pdf.std()
    lines = []
    for vi, p in zip(v, pdf):
        if p <= 0:
            continue
        col = (np.clip((p - mean) / sigma, -1, 5) + 1) / 6
        a, b = r * vi + r, (r + 0.3 * r * p / pdf.max()) * vi + r
        lines.append(f".color {col:g} 0 {1 - col:g}\n.cylinder {a[0]:g} {a[1]:g} {a[2]:g} {b[0]:g} {b[1]:g} {b[2]:g}"
                     f" {width:g}\n")
    path.write_text("".join(lines))


def w_class3d(d: Path, ctx: Ctx) -> None:
    job = "Class3D/job009/"
    for it in (5, 10, 15, 20, 25):
        model_star(d / f"run_it{it:03d}_model.star", 4, f"{job}run_it{it:03d}", ctx.rng, resolution=20 - 0.3 * it, dim=3)
    for c in range(1, 5):
        mrc(d / f"run_it025_class{c:03d}.mrc", blob_volume(32, ctx.rng), 3.54)
        angdist_bild(d / f"run_it025_class{c:03d}_angdist.bild", 32, 3.54)
    particles_star(d / "run_it025_data.star", ctx, 25, "Extract/job005",
                   {"rlnClassNumber": lambda r: int(r.integers(1, 5))})
    star(d / "run_it025_optimiser.star", StarBlock.from_dict("optimiser_general", {"rlnOutputRootName": f"{job}run"}))


def w_select_3d(d: Path, ctx: Ctx) -> None:
    particles_star(d / "particles.star", ctx, 12, "Extract/job005")


def w_refine3d(d: Path, ctx: Ctx) -> None:
    for it in range(0, 16, 3):
        model_star(d / f"run_it{it:03d}_half1_model.star", 1, f"Refine3D/job011/run_it{it:03d}_half1", ctx.rng,
                   resolution=12 - 0.4 * it, dim=3)
    vol = blob_volume(32, ctx.rng)
    mrc(d / "run_class001.mrc", vol, 3.54)
    angdist_bild(d / "run_class001_angdist.bild", 32, 3.54)
    mrc(d / "run_it015_half1_class001.mrc", vol, 3.54)
    angdist_bild(d / "run_it015_half1_class001_angdist.bild", 32, 3.54, n=48)
    mrc(d / "run_half1_class001_unfil.mrc", vol + ctx.rng.normal(0, 0.05, vol.shape), 3.54)
    mrc(d / "run_half2_class001_unfil.mrc", vol + ctx.rng.normal(0, 0.05, vol.shape), 3.54)
    particles_star(d / "run_data.star", ctx, 12, "Extract/job005")
    star(d / "run_optimiser.star", StarBlock.from_dict("optimiser_general", {"rlnOutputRootName": "Refine3D/job011/run"}))


def w_maskcreate(d: Path, ctx: Ctx) -> None:
    mrc(d / "mask.mrc", (blob_volume(32, ctx.rng) > 0.2).astype(np.float32), 3.54)


def w_postprocess(d: Path, ctx: Ctx) -> None:
    vol = blob_volume(32, ctx.rng)
    mrc(d / "postprocess.mrc", vol, 3.54)
    mrc(d / "postprocess_masked.mrc", vol, 3.54)
    (d / "logfile.pdf").write_bytes(PDF_PLACEHOLDER)
    shells = np.arange(1, 33)
    res = 32 * 3.54 / shells
    fsc = 1 / (1 + np.exp((shells - 20) / 2.0))
    gen = StarBlock.from_dict("general", {"rlnFinalResolution": 3.2, "rlnBfactorUsedForSharpening": -112.5,
                                          "rlnUnfilteredMapHalf1": "Refine3D/job011/run_half1_class001_unfil.mrc",
                                          "rlnMaskName": "MaskCreate/job012/mask.mrc",
                                          "rlnRandomiseFrom": 7.1})
    tab = StarBlock.from_records(
        "fsc",
        ["rlnSpectralIndex", "rlnResolution", "rlnAngstromResolution", "rlnFourierShellCorrelationCorrected",
         "rlnFourierShellCorrelationUnmaskedMaps", "rlnFourierShellCorrelationMaskedMaps",
         "rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps"],
        [[int(s), float(1 / r), float(r), float(f), float(f * 0.9), float(f), float(f * 0.1 if s > 8 else f)]
         for s, r, f in zip(shells, res, fsc)])
    gui = StarBlock.from_records(
        "guinier", ["rlnResolutionSquared", "rlnLogAmplitudesOriginal", "rlnLogAmplitudesWeighted"],
        [[float(1 / r ** 2), float(-10 - 30 / r), float(-9 - 25 / r)] for r in res])
    star(d / "postprocess.star", gen, tab, gui)


def ctf_phase_images(n: int, pixel: float, tilt: tuple[float, float], trefoil: tuple[float, float],
                     spherical: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Phase differences in Fourier space as relion_ctf_refine writes them (origin in the middle): coma
    from beam tilt and trefoil (antisymmetric), a 4th-order term (symmetric); observed = model + noise."""
    ky, kx = (np.mgrid[:n, :n] - n // 2) / (n * pixel)  # 1/Å
    k = np.hypot(kx, ky) * 2 * pixel  # 0 at the origin, 1 at Nyquist
    phi = np.arctan2(ky, kx)
    asym = 2.5 * k ** 3 * (tilt[0] * np.cos(phi) + tilt[1] * np.sin(phi)) \
        + 1.5 * k ** 3 * (trefoil[0] * np.cos(3 * phi) + trefoil[1] * np.sin(3 * phi))
    sym = spherical * k ** 4
    noise = lambda: rng.normal(0, 0.15 + 0.6 * k)  # noisier towards Nyquist  # noqa: E731
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    return {"beamtilt_delta-phase_per-pixel_optics-group_1.mrc": wrap(asym + noise()),
            "beamtilt_delta-phase_lin-fit_optics-group_1_N-3.mrc": wrap(asym * 0.9),
            "beamtilt_delta-phase_iter-fit_optics-group_1_N-3.mrc": wrap(asym),
            "aberr_delta-phase_per-pixel_optics-group_1.mrc": wrap(sym + noise()),
            "aberr_delta-phase_iter-fit_optics-group_1_N-4.mrc": wrap(sym)}


def w_ctfrefine(d: Path, ctx: Ctx) -> None:
    """The particles of Refine3D/job011 with refined CTFs (per-particle defocus, per-micrograph
    astigmatism), beam tilt, trefoil and 4th-order aberrations; the output grouped by micrograph in
    another order than the input, as relion_ctf_refine may."""
    rng = np.random.default_rng(14)
    blocks = read_star(ctx.root / "Refine3D/job011/run_data.star")
    parts = blocks["particles"]
    iu, iv, ia, im = (parts.col_index(c) for c in ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle",
                                                   "rlnMicrographName"))
    mic_shift = {m: float(rng.normal(0, 150)) for m in dict.fromkeys(parts.column("rlnMicrographName"))}
    rows = []
    for r in parts.rows:
        r = list(r)
        dz = mic_shift[r[im]] + float(rng.normal(0, 60)) + 0.002 * (float(r[0]) - 1900)  # a tilted ice layer
        da = float(rng.normal(0, 30))
        r[iu], r[iv] = float(r[iu]) + dz + da, float(r[iv]) + dz - da
        r[ia] = (float(r[ia]) + float(rng.normal(0, 5))) % 180
        rows.append(r)
    order = {m: k for k, m in enumerate(reversed(list(mic_shift)))}
    rows.sort(key=lambda r: order[r[im]])
    optics = blocks["optics"]
    tilt, trefoil = (0.42, -0.27), (0.08, 0.05)
    odd = [0.0, 0.0, trefoil[1], -0.31, 0.19, trefoil[0]]
    even = [0.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.35, 0.03, -0.01]
    vec = lambda v: "[" + ",".join(f"{x:.6g}" for x in v) + "]"  # noqa: E731
    optics = StarBlock.from_records(
        "optics", optics.columns + ["rlnBeamTiltX", "rlnBeamTiltY", "rlnOddZernike", "rlnEvenZernike"],
        [r + [tilt[0], tilt[1], vec(odd), vec(even)] for r in optics.rows])
    star(d / "particles_ctf_refine.star", optics, StarBlock.from_records("particles", parts.columns, rows))
    for m in list(mic_shift)[:3]:  # per-micrograph fits, where the micrographs' folders are
        stem = Path(m).relative_to("MotionCorr/job002").with_suffix("")
        star(d / f"{stem}_defocus_fit.star", StarBlock.from_records(
            "particles", parts.columns, [r for r in rows if r[im] == m]))
    for name, img in ctf_phase_images(64, 3.54, tilt, trefoil, 0.35, rng).items():
        mrc(d / name, img, 3.54)
    (d / "logfile.pdf").write_bytes(PDF_PLACEHOLDER)


def w_polish_running(d: Path, ctx: Ctx) -> None:
    (d / "run.out").write_text(" + Reading references ...\n + Estimating motion for 24 micrographs ...\n")


def w_localres_failed(d: Path, ctx: Ctx) -> None:
    (d / "run.err").write_text(
        "in: /src/postprocessing.cpp, line 1188\nERROR: \nERROR: mask and map have different sizes\n")


def spa_jobs() -> list[JobSpec]:
    mc_mics = "MotionCorr/job002/corrected_micrographs.star"
    ctf_mics = "CtfFind/job003/micrographs_ctf.star"
    return [
        JobSpec(1, "relion.import", [("movies.star", "MicrographMovieGroupMetadata.star.relion")],
                options={"fn_in_raw": "Movies/*.tiff", "angpix": "0.885", "kV": "200", "Cs": "1.4"},
                write=w_import_movies),
        JobSpec(2, "relion.motioncorr",
                [("corrected_micrographs.star", "MicrographGroupMetadata.star.relion.motioncorr"),
                 ("logfile.pdf", "LogFile.pdf.relion.motioncorr")],
                inputs=["Import/job001/movies.star"],
                options={"input_star_mics": "Import/job001/movies.star", "dose_per_frame": "1.277", "patch_x": "5",
                         "patch_y": "5", "nr_threads": "12"},
                write=w_motioncorr),
        JobSpec(3, "relion.ctffind",
                [("micrographs_ctf.star", "MicrographGroupMetadata.star.relion.ctf"),
                 ("logfile.pdf", "LogFile.pdf.relion.ctffind"),
                 ("power_spectra_fits.star", "Image2DGroupMetadata.star.relion.ctffind.power_spectra")],
                inputs=[mc_mics], options={"input_star_mics": mc_mics}, write=w_ctffind),
        JobSpec(4, "relion.autopick",
                [("autopick.star", "MicrographCoordsGroup.star.relion.autopick"),
                 ("logfile.pdf", "LogFile.pdf.relion.autopick")],
                inputs=[ctf_mics], alias="LoG_based",
                options={"fn_input_autopick": ctf_mics, "do_log": "Yes", "log_diam_min": "150", "log_diam_max": "180"},
                write=w_autopick),
        JobSpec(5, "relion.extract", [("particles.star", "ParticleGroupMetadata.star.relion")],
                inputs=[ctf_mics, "AutoPick/job004/autopick.star"],
                options={"star_mics": ctf_mics, "coords_suffix": "AutoPick/job004/autopick.star", "extract_size": "256",
                         "do_rescale": "Yes", "rescale": "64"},
                write=w_extract),
        JobSpec(6, "relion.class2d",
                [("run_it200_data.star", "ParticleGroupMetadata.star.relion.class2d"),
                 ("run_it200_optimiser.star", "OptimiserData.star.relion.class2d")],
                inputs=["Extract/job005/particles.star"],
                options={"fn_img": "Extract/job005/particles.star", "nr_classes": "50", "particle_diameter": "200"},
                write=_w_class2d("Class2D/job006/", 50, [25, 50, 100, 150, 200])),
        JobSpec(7, "relion.select",
                [("particles.star", "ParticleGroupMetadata.star.relion"),
                 ("class_averages.star", "Image2DGroupMetadata.star.relion.classaverages")],
                inputs=["Class2D/job006/run_it200_optimiser.star"],
                options={"fn_model": "Class2D/job006/run_it200_optimiser.star"}, write=w_select_2d),
        JobSpec(8, "relion.initialmodel", [("initial_model.mrc", "DensityMap.mrc.relion.initialmodel")],
                inputs=["Select/job007/particles.star"],
                options={"fn_img": "Select/job007/particles.star", "sym_name": "D2"}, write=w_inimodel),
        JobSpec(9, "relion.class3d",
                [("run_it025_data.star", "ParticleGroupMetadata.star.relion.refine3d"),
                 ("run_it025_optimiser.star", "OptimiserData.star.relion.class3d")]
                + [(f"run_it025_class{c:03d}.mrc", "DensityMap.mrc.relion.class3d") for c in range(1, 5)],
                inputs=["Select/job007/particles.star", "InitialModel/job008/initial_model.mrc"],
                options={"fn_img": "Select/job007/particles.star", "fn_ref": "InitialModel/job008/initial_model.mrc",
                         "nr_classes": "4"},
                write=w_class3d),
        JobSpec(10, "relion.select", [("particles.star", "ParticleGroupMetadata.star.relion")],
                inputs=["Class3D/job009/run_it025_optimiser.star"],
                options={"fn_model": "Class3D/job009/run_it025_optimiser.star"}, write=w_select_3d),
        JobSpec(11, "relion.refine3d",
                [("run_data.star", "ParticleGroupMetadata.star.relion.refine3d"),
                 ("run_optimiser.star", "OptimiserData.star.relion.refine3d"),
                 ("run_half1_class001_unfil.mrc", "DensityMap.mrc.relion.halfmap.refine3d"),
                 ("run_class001.mrc", "DensityMap.mrc.relion.refine3d")],
                inputs=["Select/job010/particles.star", "Class3D/job009/run_it025_class001.mrc"],
                alias="first_refine",
                options={"fn_img": "Select/job010/particles.star", "fn_ref": "Class3D/job009/run_it025_class001.mrc",
                         "sym_name": "D2"},
                write=w_refine3d),
        JobSpec(12, "relion.maskcreate", [("mask.mrc", "Mask3D.mrc.relion")],
                inputs=["Refine3D/job011/run_class001.mrc"],
                options={"fn_in": "Refine3D/job011/run_class001.mrc", "inimask_threshold": "0.01"},
                write=w_maskcreate),
        JobSpec(13, "relion.postprocess",
                [("postprocess.mrc", "DensityMap.mrc.relion.postprocess"),
                 ("postprocess_masked.mrc", "DensityMap.mrc.relion.postprocess.masked"),
                 ("logfile.pdf", "LogFile.pdf.relion.postprocess"),
                 ("postprocess.star", "ProcessData.star.relion.postprocess")],
                inputs=["MaskCreate/job012/mask.mrc", "Refine3D/job011/run_half1_class001_unfil.mrc"],
                options={"fn_in": "Refine3D/job011/run_half1_class001_unfil.mrc", "fn_mask": "MaskCreate/job012/mask.mrc"},
                write=w_postprocess),
        JobSpec(14, "relion.ctfrefine",
                [("logfile.pdf", "LogFile.pdf.relion.ctfrefine"),
                 ("particles_ctf_refine.star", "ParticleGroupMetadata.star.relion.ctfrefine")],
                inputs=["Refine3D/job011/run_data.star", "PostProcess/job013/postprocess.star"],
                options={"fn_data": "Refine3D/job011/run_data.star", "fn_post": "PostProcess/job013/postprocess.star",
                         "do_ctf": "Yes", "do_defocus": "Per-particle", "do_astig": "Per-micrograph",
                         "do_bfactor": "No", "do_phase": "No", "do_tilt": "Yes", "do_trefoil": "Yes",
                         "do_4thorder": "Yes", "minres": "30"},
                write=w_ctfrefine),
        JobSpec(15, "relion.polish",
                [("opt_params_all_groups.txt", "ParamsData.txt.relion.polish"),
                 ("logfile.pdf", "LogFile.pdf.relion.polish"),
                 ("shiny.star", "ParticleGroupMetadata.star.relion.polished")],
                inputs=["CtfRefine/job014/particles_ctf_refine.star", "PostProcess/job013/postprocess.star", mc_mics],
                status="Running",
                options={"fn_data": "CtfRefine/job014/particles_ctf_refine.star",
                         "fn_post": "PostProcess/job013/postprocess.star", "fn_mic": mc_mics, "do_queue": "Yes",
                         "nr_mpi": "8"},
                write=w_polish_running),
        JobSpec(16, "relion.refine3d",
                [("run_data.star", "ParticleGroupMetadata.star.relion.refine3d"),
                 ("run_optimiser.star", "OptimiserData.star.relion.refine3d"),
                 ("run_half1_class001_unfil.mrc", "DensityMap.mrc.relion.halfmap.refine3d"),
                 ("run_class001.mrc", "DensityMap.mrc.relion.refine3d")],
                inputs=["Polish/job015/shiny.star", "Refine3D/job011/run_class001.mrc", "MaskCreate/job012/mask.mrc"],
                status="Scheduled",
                options={"fn_img": "Polish/job015/shiny.star", "fn_ref": "Refine3D/job011/run_class001.mrc",
                         "fn_mask": "MaskCreate/job012/mask.mrc", "sym_name": "D2"}),
        JobSpec(17, "relion.localres",
                [("histogram.pdf", "LogFile.pdf.relion.localres"),
                 ("relion_locres_filtered.mrc", "DensityMap.mrc.relion.localresfiltered"),
                 ("relion_locres.mrc", "Image3D.mrc.relion.localresmap")],
                inputs=["Refine3D/job011/run_half1_class001_unfil.mrc", "MaskCreate/job012/mask.mrc"],
                status="Failed",
                options={"fn_in": "Refine3D/job011/run_half1_class001_unfil.mrc", "fn_mask": "MaskCreate/job012/mask.mrc"},
                write=w_localres_failed),
        JobSpec(18, "relion.class2d",
                [("run_it025_data.star", "ParticleGroupMetadata.star.relion.class2d"),
                 ("run_it025_optimiser.star", "OptimiserData.star.relion.class2d")],
                inputs=["Extract/job005/particles.star"], status="Aborted",
                options={"fn_img": "Extract/job005/particles.star", "nr_classes": "100", "do_em": "Yes",
                         "do_grad": "No", "nr_iter_em": "25"}),
    ]


# --------------------------------------------------------------------------- tomo data writers

TILT_SERIES = ["TS_01", "TS_03", "TS_43", "TS_45", "TS_54"]
TILTS = np.arange(-60, 61, 3)


def _tomo_global(path: Path, per_ts_dir: str, extra: dict | None = None) -> None:
    cols = ["rlnTomoName", "rlnTomoTiltSeriesStarFile", "rlnVoltage", "rlnSphericalAberration",
            "rlnAmplitudeContrast", "rlnMicrographOriginalPixelSize", "rlnTomoHand", "rlnOpticsGroupName"]
    rows = [[ts, f"{per_ts_dir}/{ts}.star", 300.0, 2.7, 0.1, 0.675, -1.0, "optics1"] for ts in TILT_SERIES]
    if extra:
        for c, fn in extra.items():
            cols.append(c)
            for r in rows:
                r.append(fn(r[0]))
    star(path, StarBlock.from_records("global", cols, rows))


def _per_tilt(path: Path, ts: str, rng: np.random.Generator, cols: dict[str, Callable]) -> None:
    order = np.argsort(np.abs(TILTS), kind="stable")
    pre_exposure = np.empty(len(TILTS))
    pre_exposure[order] = np.arange(len(TILTS)) * 3.0
    rows = []
    for i, t in enumerate(TILTS):
        row = [f"frames/{ts}_{i + 1:03d}_{float(t):.1f}.mrc", 8, float(t), 85.0, float(pre_exposure[i]), -4.0]
        row += [fn(i, t, rng) for fn in cols.values()]
        rows.append(row)
    star(path, StarBlock.from_records(
        ts, ["rlnMicrographMovieName", "rlnTomoTiltMovieFrameCount", "rlnTomoNominalStageTiltAngle",
             "rlnTomoNominalTiltAxisAngle", "rlnMicrographPreExposure", "rlnTomoNominalDefocus"] + list(cols), rows))


def _w_tilt_stage(star_name: str, job: str, cols: dict[str, Callable], extra: dict | None = None):
    def w(d: Path, ctx: Ctx) -> None:
        _tomo_global(d / star_name, f"{job}tilt_series", extra)
        for ts in TILT_SERIES:
            _per_tilt(d / "tilt_series" / f"{ts}.star", ts, ctx.rng, cols)
    return w


def w_tomo_reconstruct(d: Path, ctx: Ctx, job: str = "Tomograms/job006/", denoised: bool = False) -> None:
    key = "rlnTomoReconstructedTomogramDenoised" if denoised else "rlnTomoReconstructedTomogram"
    _tomo_global(d / "tomograms.star", f"{job}tilt_series", {
        "rlnTomoSizeX": lambda ts: 4000, "rlnTomoSizeY": lambda ts: 4000, "rlnTomoSizeZ": lambda ts: 2000,
        "rlnTomoTomogramBinning": lambda ts: 10.0,
        # Denoise keeps the raw tomograms of its input and adds its own
        **({"rlnTomoReconstructedTomogram": lambda ts: f"Tomograms/job006/tomograms/rec_{ts}.mrc"} if denoised else {}),
        key: lambda ts: f"{job}tomograms/rec_{ts}.mrc"})
    for ts in TILT_SERIES:
        _per_tilt(d / "tilt_series" / f"{ts}.star", ts, ctx.rng, {})
        vol = ctx.rng.normal(0, 1, (16, 64, 64)) if not denoised else blob_volume(64, ctx.rng, 20)[:16]
        mrc(d / "tomograms" / f"rec_{ts}.mrc", vol, 6.75)


def w_tomo_picks(d: Path, ctx: Ctx) -> None:
    rows = []
    for ts in TILT_SERIES:
        for _ in range(int(ctx.rng.integers(20, 40))):
            rows.append([ts, float(ctx.rng.uniform(0, 400)), float(ctx.rng.uniform(0, 400)), float(ctx.rng.uniform(0, 200))])
    star(d / "particles.star", StarBlock.from_records(
        "particles", ["rlnTomoName", "rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
                      "rlnCenteredCoordinateZAngst"], rows))
    star(d / "optimisation_set.star", StarBlock.from_dict("", {
        "rlnTomoParticlesFile": "Picks/job008/particles.star", "rlnTomoTomogramsFile": "Denoise/job007/tomograms.star"}))


def w_tomo_extract(d: Path, ctx: Ctx) -> None:
    w_tomo_picks(d, ctx)
    # relion_tomo_subtomo adds an optics table with the binned pixel size and the box size
    parts = read_star(d / "particles.star")["particles"]
    parts.columns.append("rlnOpticsGroup")
    for r in parts.rows:
        r.append("1")
    optics = StarBlock.from_records(
        "optics", ["rlnOpticsGroupName", "rlnOpticsGroup", "rlnTomoTiltSeriesPixelSize", "rlnTomoSubtomogramBinning",
                   "rlnImagePixelSize", "rlnImageSize", "rlnImageDimensionality"],
        [["optics1", 1, 0.675, 10.0, 6.75, 64, 2]])
    star(d / "particles.star", optics, parts)
    star(d / "optimisation_set.star", StarBlock.from_dict("", {
        "rlnTomoParticlesFile": "Extract/job009/particles.star", "rlnTomoTomogramsFile": "Tomograms/job006/tomograms.star"}))


def w_tomo_refine(d: Path, ctx: Ctx) -> None:
    vol = blob_volume(32, ctx.rng)
    mrc(d / "run_class001.mrc", vol, 2.7)
    mrc(d / "run_half1_class001_unfil.mrc", vol, 2.7)
    mrc(d / "run_half2_class001_unfil.mrc", vol, 2.7)
    w_tomo_picks(d, ctx)
    (d / "particles.star").rename(d / "run_data.star")
    star(d / "run_optimisation_set.star", StarBlock.from_dict("", {"rlnTomoParticlesFile": "Refine3D/job011/run_data.star"}))
    star(d / "run_optimiser.star", StarBlock.from_dict("optimiser_general", {"rlnOutputRootName": "Refine3D/job011/run"}))


def tomo_jobs() -> list[JobSpec]:
    defocus = {"rlnDefocusU": lambda i, t, r: float(r.uniform(30000, 45000)),
               "rlnDefocusV": lambda i, t, r: float(r.uniform(30000, 45000)),
               "rlnCtfMaxResolution": lambda i, t, r: float(5 + abs(t) / 6 + r.uniform(0, 3))}
    return [
        JobSpec(1, "relion.importtomo", [("tilt_series.star", "TomogramGroupMetadata.star.relion.tomo.import")],
                options={"movie_files": "frames/*.mrc", "mdoc_files": "mdoc/*.mdoc", "angpix": "0.675", "kV": "300",
                         "Cs": "2.7", "dose_rate": "3", "tilt_axis_angle": "85"},
                write=_w_tilt_stage("tilt_series.star", "Import/job001/", {})),
        JobSpec(2, "relion.motioncorr",
                [("corrected_tilt_series.star", "TomogramGroupMetadata.star.relion.tomo.motioncorr"),
                 ("logfile.pdf", "LogFile.pdf.relion.motioncorr")],
                inputs=["Import/job001/tilt_series.star"], tomo=True,
                options={"input_star_mics": "Import/job001/tilt_series.star"},
                write=_w_tilt_stage("corrected_tilt_series.star", "MotionCorr/job002/", {
                    "rlnAccumMotionTotal": lambda i, t, r: float(r.uniform(2, 15))})),
        JobSpec(3, "relion.ctffind",
                [("tilt_series_ctf.star", "TomogramGroupMetadata.star.relion.tomo.ctffind"),
                 ("logfile.pdf", "LogFile.pdf.relion.ctffind")],
                inputs=["MotionCorr/job002/corrected_tilt_series.star"], tomo=True,
                options={"input_star_mics": "MotionCorr/job002/corrected_tilt_series.star"},
                write=_w_tilt_stage("tilt_series_ctf.star", "CtfFind/job003/", defocus)),
        JobSpec(4, "relion.excludetilts",
                [("selected_tilt_series.star", "TomogramGroupMetadata.star.relion.tomo.excludeimages")],
                inputs=["CtfFind/job003/tilt_series_ctf.star"],
                options={"in_tiltseries": "CtfFind/job003/tilt_series_ctf.star"},
                write=_w_tilt_stage("selected_tilt_series.star", "ExcludeTiltImages/job004/", defocus)),
        JobSpec(5, "relion.aligntiltseries",
                [("aligned_tilt_series.star", "TomogramGroupMetadata.star.relion.tomo.aligntiltseries"),
                 ("logfile.pdf", "LogFile.pdf.relion.tomo.aligntiltseries")],
                inputs=["ExcludeTiltImages/job004/selected_tilt_series.star"],
                options={"in_tiltseries": "ExcludeTiltImages/job004/selected_tilt_series.star", "do_imod_fiducials": "Yes",
                         "fiducial_diameter": "10"},
                write=_w_tilt_stage("aligned_tilt_series.star", "AlignTiltSeries/job005/", {
                    "rlnTomoXTilt": lambda i, t, r: 0.0, "rlnTomoYTilt": lambda i, t, r: float(t),
                    "rlnTomoZRot": lambda i, t, r: 85.0 + float(r.normal(0, 0.2)),
                    "rlnTomoXShiftAngst": lambda i, t, r: float(r.normal(0, 30)),
                    "rlnTomoYShiftAngst": lambda i, t, r: float(r.normal(0, 30))})),
        JobSpec(6, "relion.reconstructtomograms", [("tomograms.star", "TomogramGroupMetadata.star.relion.tomo.reconstruct")],
                inputs=["AlignTiltSeries/job005/aligned_tilt_series.star"],
                options={"in_tiltseries": "AlignTiltSeries/job005/aligned_tilt_series.star", "binned_angpix": "10",
                         "xdim": "4000", "ydim": "4000", "zdim": "2000"},
                write=lambda d, ctx: w_tomo_reconstruct(d, ctx)),
        JobSpec(7, "relion.denoisetomo", [("tomograms.star", "TomogramGroupMetadata.star.relion.tomo.denoise")],
                inputs=["Tomograms/job006/tomograms.star"],
                options={"in_tomoset": "Tomograms/job006/tomograms.star"},
                write=lambda d, ctx: w_tomo_reconstruct(d, ctx, "Denoise/job007/", denoised=True)),
        JobSpec(8, "relion.picktomo",
                [("particles.star", "ParticleGroupMetadata.star.relion.tomo.manualpick.spheres"),
                 ("optimisation_set.star", "TomoOptimisationSet.star.relion.tomo.manualpick")],
                inputs=["Denoise/job007/tomograms.star"], alias="spheres",
                options={"in_tomoset": "Denoise/job007/tomograms.star", "pick_mode": "spheres"},
                write=w_tomo_picks),
        JobSpec(9, "relion.pseudosubtomo",
                [("optimisation_set.star", "TomoOptimisationSet.star.relion.tomo.extract"),
                 ("particles.star", "ParticleGroupMetadata.star.relion.tomo.extract")],
                inputs=["Picks/job008/optimisation_set.star", "Tomograms/job006/tomograms.star"],
                options={"in_optimisation": "Picks/job008/optimisation_set.star", "box_size_angst": "512",
                         "binned_size_pix": "64"},
                write=w_tomo_extract),
        JobSpec(10, "relion.reconstructparticletomo",
                [("merged.mrc", "DensityMap.mrc.relion.tomo.map.reconstruct"),
                 ("optimisation_set.star", "TomoOptimisationSet.star.relion.tomo.reconstruct")],
                inputs=["Extract/job009/optimisation_set.star"],
                options={"in_optimisation": "Extract/job009/optimisation_set.star", "box_size": "128",
                         "crop_size": "64", "binning": "2"},
                write=lambda d, ctx: mrc(d / "merged.mrc", blob_volume(32, ctx.rng), 2.7)),
        JobSpec(11, "relion.refine3d",
                [("run_data.star", "ParticleGroupMetadata.star.relion.refine3d"),
                 ("run_optimiser.star", "OptimiserData.star.relion.refine3d"),
                 ("run_half1_class001_unfil.mrc", "DensityMap.mrc.relion.halfmap.refine3d"),
                 ("run_class001.mrc", "DensityMap.mrc.relion.refine3d"),
                 ("run_optimisation_set.star", "TomoOptimisationSet.star.relion.refine3d")],
                inputs=["Extract/job009/optimisation_set.star", "Reconstruct/job010/merged.mrc"], tomo=True,
                options={"in_optimisation": "Extract/job009/optimisation_set.star",
                         "fn_ref": "Reconstruct/job010/merged.mrc", "sym_name": "C6"},
                write=w_tomo_refine),
        JobSpec(12, "relion.maskcreate", [("mask.mrc", "Mask3D.mrc.relion")],
                inputs=["Refine3D/job011/run_class001.mrc"],
                options={"fn_in": "Refine3D/job011/run_class001.mrc"}, write=w_maskcreate),
        JobSpec(13, "relion.postprocess",
                [("postprocess.mrc", "DensityMap.mrc.relion.postprocess"),
                 ("postprocess_masked.mrc", "DensityMap.mrc.relion.postprocess.masked"),
                 ("logfile.pdf", "LogFile.pdf.relion.postprocess"),
                 ("postprocess.star", "ProcessData.star.relion.postprocess")],
                inputs=["MaskCreate/job012/mask.mrc", "Refine3D/job011/run_half1_class001_unfil.mrc"], tomo=True,
                options={"fn_in": "Refine3D/job011/run_half1_class001_unfil.mrc", "fn_mask": "MaskCreate/job012/mask.mrc"},
                write=w_postprocess),
        JobSpec(14, "relion.ctfrefinetomo",
                [("optimisation_set.star", "TomoOptimisationSet.star.relion.tomo.ctfrefine"),
                 ("tomograms.star", "TomogramGroupMetadata.star.relion.tomo.ctfrefine"),
                 ("logfile.pdf", "LogFile.pdf.relion.tomo.ctfrefine")],
                inputs=["Refine3D/job011/run_optimisation_set.star", "PostProcess/job013/postprocess.star",
                        "Refine3D/job011/run_half1_class001_unfil.mrc"],
                status="Running",
                options={"in_optimisation": "Refine3D/job011/run_optimisation_set.star",
                         "in_halfmaps": "Refine3D/job011/run_half1_class001_unfil.mrc",
                         "in_post": "PostProcess/job013/postprocess.star"}),
    ]


# --------------------------------------------------------------------------- project builder

def build_project(root: Path, jobs: list[JobSpec], seed: int = 1) -> Pipeline:
    """Create a RELION project directory from job specs and return its pipeline."""
    root.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(root=root, rng=np.random.default_rng(seed))
    pipe = Pipeline(job_counter=max(j.num for j in jobs) + 1)
    for spec in jobs:
        _add_to_project(root, spec, pipe, ctx)
    write_star(root / "default_pipeline.star", pipe.to_blocks(), atomic=False)
    return pipe


def add_job(root: Path, spec: JobSpec) -> None:
    """Add one job to an existing project like RELION does: under the lock, pipeline rewritten.

    The pipeline file's mtime is moved forward by a second so that filesystems with coarse
    timestamps still show the change.
    """
    with PipelineLock(root, message="cryo-savanna test fixture"):
        pipe = Pipeline.read(root)
        pipe.job_counter = max(pipe.job_counter, spec.num + 1)
        _add_to_project(root, spec, pipe, Ctx(root=root, rng=np.random.default_rng(spec.num)))
        path = root / "default_pipeline.star"
        old = path.stat().st_mtime_ns
        write_star(path, pipe.to_blocks(), atomic=False)
        os.utime(path, ns=(old + 1_000_000_000, old + 1_000_000_000))


def _add_to_project(root: Path, spec: JobSpec, pipe: Pipeline, ctx: Ctx) -> None:
    env = RelionEnv({})  # compiled-in defaults only, independent of the machine running the tests
    defs = JobDefs.load()
    jobdir = root / spec.name
    jobdir.mkdir(parents=True, exist_ok=True)
    variant = defs.variant(spec.label, spec.tomo)
    js = variant.new_jobstar(env)
    unknown = set(spec.options) - set(js.options)
    if unknown:
        raise KeyError(f"{spec.name}: {spec.label} has no options {sorted(unknown)}")
    js.options.update(spec.options)
    js.write(jobdir / "job.star")
    (jobdir / "note.txt").write_text(
        f" ++++ Executing new job on Tue Oct 28 10:00:00 2026\n ++++ with the following command(s):\n"
        f"`which relion_{spec.dirname.lower()}` --o {spec.name}  --pipeline_control {spec.name}\n ++++ \n")
    if spec.status != "Scheduled":
        (jobdir / "run.out").write_text(f"Running {spec.name}\n")
        (jobdir / "run.err").write_text("")
    if spec.write is not None:
        spec.write(jobdir, ctx)
    if spec.status == "Succeeded":
        for rel, _ in spec.outputs:
            p = jobdir / rel
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(PDF_PLACEHOLDER if rel.endswith(".pdf") else b"")
        (jobdir / RELION_JOB_EXIT_SUCCESS).touch()
    elif spec.status == "Failed":
        (jobdir / RELION_JOB_EXIT_FAILURE).touch()
    elif spec.status == "Aborted":
        (jobdir / RELION_JOB_EXIT_ABORTED).touch()

    alias = "None"
    if spec.alias:
        alias = f"{spec.dirname}/{spec.alias}/"
        link = root / spec.dirname / spec.alias
        if not link.exists():
            os.symlink(f"../{spec.name}", link)
    proc = Process(spec.name, alias, spec.label, spec.status)
    for n in spec.inputs:
        if n not in pipe.nodes:
            raise KeyError(f"{spec.name}: input {n} is not an output of an earlier job")
        proc.inputs.append(n)
        pipe.nodes[n].consumers.append(spec.name)
    for rel, label in spec.outputs:
        name = spec.name + rel
        depth = 99 if label == "DensityMap.mrc.relion.tomo.halfmap.reconstruct" else 1
        pipe.nodes[name] = Node(name, label, depth, producer=spec.name)
        proc.outputs.append(name)
    pipe.processes[spec.name] = proc


def large_spa_jobs(rounds: int = 35) -> list[JobSpec]:
    """The first 8 SPA jobs, then ``rounds`` branching rounds of 7 jobs (Class2D, Select, Class3D,
    Select, Refine3D, MaskCreate, PostProcess). No data files: for size and speed tests."""
    jobs = [dataclasses.replace(j, write=None) for j in spa_jobs()[:8]]
    particles = ["Extract/job005/particles.star", "Select/job007/particles.star"]
    ref = "InitialModel/job008/initial_model.mrc"
    n = 9
    for r in range(rounds):
        src = particles[(r * 7) % len(particles)]
        c2, s2, c3, s3, rf, mk, pp = range(n, n + 7)
        d = {k: f"{lbl}/job{num:03d}/" for k, lbl, num in (
            ("c2", "Class2D", c2), ("s2", "Select", s2), ("c3", "Class3D", c3), ("s3", "Select", s3),
            ("rf", "Refine3D", rf), ("mk", "MaskCreate", mk), ("pp", "PostProcess", pp))}
        jobs += [
            JobSpec(c2, "relion.class2d", [("run_it025_optimiser.star", "OptimiserData.star.relion.class2d")],
                    inputs=[src], options={"fn_img": src}),
            JobSpec(s2, "relion.select", [("particles.star", "ParticleGroupMetadata.star.relion")],
                    inputs=[d["c2"] + "run_it025_optimiser.star"], options={"fn_model": d["c2"] + "run_it025_optimiser.star"}),
            JobSpec(c3, "relion.class3d", [("run_it025_optimiser.star", "OptimiserData.star.relion.class3d"),
                                           ("run_it025_class001.mrc", "DensityMap.mrc.relion.class3d")],
                    inputs=[d["s2"] + "particles.star", ref], options={"fn_img": d["s2"] + "particles.star", "fn_ref": ref}),
            JobSpec(s3, "relion.select", [("particles.star", "ParticleGroupMetadata.star.relion")],
                    inputs=[d["c3"] + "run_it025_optimiser.star"], options={"fn_model": d["c3"] + "run_it025_optimiser.star"}),
            JobSpec(rf, "relion.refine3d", [("run_half1_class001_unfil.mrc", "DensityMap.mrc.relion.halfmap.refine3d"),
                                            ("run_class001.mrc", "DensityMap.mrc.relion.refine3d")],
                    inputs=[d["s3"] + "particles.star", d["c3"] + "run_it025_class001.mrc"],
                    options={"fn_img": d["s3"] + "particles.star", "fn_ref": d["c3"] + "run_it025_class001.mrc"}),
            JobSpec(mk, "relion.maskcreate", [("mask.mrc", "Mask3D.mrc.relion")],
                    inputs=[d["rf"] + "run_class001.mrc"], options={"fn_in": d["rf"] + "run_class001.mrc"}),
            JobSpec(pp, "relion.postprocess", [("postprocess.star", "ProcessData.star.relion.postprocess")],
                    inputs=[d["mk"] + "mask.mrc", d["rf"] + "run_half1_class001_unfil.mrc"],
                    options={"fn_in": d["rf"] + "run_half1_class001_unfil.mrc", "fn_mask": d["mk"] + "mask.mrc"}),
        ]
        particles += [d["s2"] + "particles.star", d["s3"] + "particles.star"]
        n += 7
    return jobs


def make_large_project(root: Path, rounds: int = 35) -> Pipeline:
    return build_project(root, large_spa_jobs(rounds), seed=3)


def make_spa_project(root: Path) -> Pipeline:
    pipe = build_project(root, spa_jobs(), seed=1)
    add_viewer_data(root)
    return pipe


def add_viewer_data(root: Path, seed: int = 11) -> None:
    """Files only the result viewers read: motion tracks, CTF spectra, particle stacks, FSC curves,
    optimiser changes and particle angles. A separate random generator, so the values the other
    tests check do not move."""
    rng = np.random.default_rng(seed)
    mics = read_star(root / "MotionCorr/job002/corrected_micrographs.star")["micrographs"]
    for meta in mics.column("rlnMicrographMetadata"):
        steps = rng.normal(0, 1.5, (24, 2)) * np.linspace(3, 0.5, 24)[:, None]
        track = np.cumsum(steps, axis=0)
        track -= track[0]
        star(root / meta, StarBlock.from_dict("general", {"rlnImageSizeX": 3838, "rlnImageSizeY": 3710,
                                                           "rlnImageSizeZ": 24, "rlnMicrographMovieName": "x.tiff",
                                                           "rlnMicrographBinning": 1.0,
                                                           "rlnMicrographOriginalPixelSize": 0.885}),
             StarBlock.from_records("global_shift", ["rlnMicrographFrameNumber", "rlnMicrographShiftX",
                                                     "rlnMicrographShiftY"],
                                    [[i + 1, float(x), float(y)] for i, (x, y) in enumerate(track)]))
    yy, xx = np.mgrid[:64, :64] - 32
    r = np.hypot(xx, yy)
    for ctf in read_star(root / "CtfFind/job003/micrographs_ctf.star")["micrographs"].column("rlnCtfImage"):
        rings = np.cos(r ** 2 / rng.uniform(20, 60)) ** 2 * np.exp(-r / 40) + rng.normal(0, 0.05, r.shape)
        mrc(root / ctf.removesuffix(":mrc"), rings, 1.0)
    y2, x2 = np.mgrid[:32, :32] - 16
    for m in mics.column("rlnMicrographName"):
        stack = np.empty((40, 32, 32), np.float32)
        for i in range(40):
            c = rng.normal(0, 2, 2)
            stack[i] = np.exp(-((x2 - c[0]) ** 2 + (y2 - c[1]) ** 2) / 40) + rng.normal(0, 0.3, (32, 32))
        mrc(root / f"Extract/job005/Movies/{Path(m).stem}.mrcs", stack, 3.54)
    # FSC between the half maps in Refine3D's models, and the optimiser's changes per iteration
    job = root / "Refine3D/job011"
    for model in sorted(job.glob("run_it*_half1_model.star")):
        it = int(model.name[6:9])
        shells = np.arange(0, 33)
        fsc = 1 / (1 + np.exp((shells - (8 + it)) / 1.5))
        fsc[0] = 1.0
        with open(model, "a") as fh:
            fh.write("\n" + format_star([StarBlock.from_records(
                "model_class_1", ["rlnSpectralIndex", "rlnResolution", "rlnAngstromResolution", "rlnGoldStandardFsc"],
                [[int(i), float(i / (64 * 3.54)), float(64 * 3.54 / i) if i else 999.0, float(f)]
                 for i, f in zip(shells, fsc)])]))
        _optimiser(job / f"run_it{it:03d}_optimiser.star", "Refine3D/job011/run", 30 / (1 + it), 2 / (1 + it), 0.0)
    for it in (5, 10, 15, 20, 25):
        _optimiser(root / f"Class3D/job009/run_it{it:03d}_optimiser.star", "Class3D/job009/run", 40 / it, 3 / it,
                   0.5 / it)
    for data in ("Refine3D/job011/run_data.star", "Class3D/job009/run_it025_data.star"):
        blocks = read_star(root / data)
        parts = blocks["particles"]
        parts.columns += ["rlnAngleRot", "rlnAngleTilt"]
        for row in parts.rows:
            row += [float(rng.uniform(-180, 180)), float(np.degrees(np.arccos(rng.uniform(-1, 1))))]
        write_star(root / data, blocks.values(), atomic=False)


def _optimiser(path: Path, root_name: str, orient: float, offset: float, classes: float) -> None:
    star(path, StarBlock.from_dict("optimiser_general", {
        "rlnOutputRootName": root_name, "rlnChangesOptimalOrientations": orient,
        "rlnChangesOptimalOffsets": offset, "rlnChangesOptimalClasses": classes}))


def make_tomo_project(root: Path) -> Pipeline:
    pipe = build_project(root, tomo_jobs(), seed=2)
    add_tomo_viewer_data(root)
    return pipe


def add_tomo_viewer_data(root: Path, seed: int = 12) -> None:
    """Files only the tomography viewers read: motion-corrected tilt images and CTF spectra (added to
    the per-tilt tables from MotionCorr on), particles inside their tomograms, and tomograms with
    something at the particles. Own random generator, so the other tests' values stay."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:48, :48] - 24
    r = np.hypot(xx, yy)
    for ts in TILT_SERIES:
        for i, t in enumerate(TILTS):
            img = rng.normal(0, 1, (64, 64)) + 2 * np.exp(-((np.arange(64) - 32 - t / 4) ** 2) / 40)[None, :]
            mrc(root / f"MotionCorr/job002/tilt_images/{ts}_{i + 1:03d}.mrc", img, 0.675 * 8)
            mrc(root / f"CtfFind/job003/tilt_images/{ts}_{i + 1:03d}.ctf",
                np.cos(r ** 2 / (25 + abs(t) / 3)) ** 2 * np.exp(-r / 30) + rng.normal(0, 0.05, r.shape), 1.0)
    later = ["MotionCorr/job002/", "CtfFind/job003/", "ExcludeTiltImages/job004/", "AlignTiltSeries/job005/",
             "Tomograms/job006/", "Denoise/job007/"]
    for k, job in enumerate(later):
        for ts in TILT_SERIES:
            f = root / job / "tilt_series" / f"{ts}.star"
            blocks = read_star(f)
            b = blocks[ts]
            b.columns.append("rlnMicrographName")
            if k >= 1:
                b.columns.append("rlnCtfImage")
            for i, row in enumerate(b.rows):
                row.append(f"MotionCorr/job002/tilt_images/{ts}_{i + 1:03d}.mrc")
                if k >= 1:
                    row.append(f"CtfFind/job003/tilt_images/{ts}_{i + 1:03d}.ctf:mrc")
            write_star(f, blocks.values(), atomic=False)
    # particles inside their tomograms (64 × 64 × 16 voxels of 6.75 Å: ±216 Å in x and y, ±54 Å in z)
    for rel in ("Picks/job008/particles.star", "Extract/job009/particles.star", "Refine3D/job011/run_data.star"):
        blocks = read_star(root / rel)
        b = blocks["particles"]
        cx = [b.col_index(c) for c in ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
                                       "rlnCenteredCoordinateZAngst")]
        for row in b.rows:
            for c, half in zip(cx, (180, 180, 40)):
                row[c] = f"{rng.uniform(-half, half):.3f}"
        write_star(root / rel, blocks.values(), atomic=False)
    picks = read_star(root / "Picks/job008/particles.star")["particles"]
    z, y, x = np.mgrid[:16, :64, :64]
    for job, denoised in (("Tomograms/job006/", False), ("Denoise/job007/", True)):
        for ts in TILT_SERIES:
            vol = rng.normal(0, 0.3 if denoised else 1.0, (16, 64, 64))
            for row in picks.rows:
                if row[picks.col_index("rlnTomoName")] != ts:
                    continue
                px, py, pz = (float(row[picks.col_index(c)]) / 6.75 + h for c, h in (
                    ("rlnCenteredCoordinateXAngst", 32), ("rlnCenteredCoordinateYAngst", 32),
                    ("rlnCenteredCoordinateZAngst", 8)))
                vol += 3 * np.exp(-((x + 0.5 - px) ** 2 + (y + 0.5 - py) ** 2 + (z + 0.5 - pz) ** 2) / 4)
            mrc(root / job / "tomograms" / f"rec_{ts}.mrc", vol, 6.75)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    make_spa_project(out / "spa")
    make_tomo_project(out / "tomo")
    print(f"wrote {out / 'spa'} and {out / 'tomo'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
