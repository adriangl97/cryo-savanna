# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The RELION 5.1 command-line programs that no GUI job runs, for the "CLI tools" menu.

Programs with RELION's usual --help (IOParser flags) get a form built from it. The few that take
positional arguments are described by hand in POSITIONAL. Left out:

- relion_pipeliner, relion_schemer: they run the pipeline itself, not a step in it;
- relion_star_plottable: opens a gnuplot window;
- relion_external_reconstruct: called by relion_refine --external_reconstruct, not on its own;
- relion_prepare_subtomo: RELION 2 sub-tomogram preparation (needs IMOD), superseded by the tomo jobs;
- relion_tomo_test: a developer test;
- relion_python_* tools: no --help texts collected yet (tools/validate.py skips them; some start
  napari), and the ones the jobs use are already in the job menus;
- the programs the GUI jobs run (jobdefs/flags.JOB_PROGRAMS): all their flags are in each job's
  "Additional arguments" list.
"""

from __future__ import annotations

from dataclasses import dataclass

# program -> (menu category, one-line description)
FLAG_TOOLS: dict[str, tuple[str, str]] = {
    # maps and images
    "relion_image_handler": ("Maps and images", "Operations on images, stacks and maps: rescale, filter, flip, invert, "
                             "add or subtract, statistics, FSC, power spectra, ..."),
    "relion_align_symmetry": ("Maps and images", "Rotate a map to the standard orientation of its point-group symmetry"),
    "relion_project": ("Maps and images", "Project a 3D map into 2D images, optionally with CTF and noise"),
    "relion_stack_create": ("Maps and images", "Write the images of a STAR file into a single stack"),
    "relion_convert_to_tiff": ("Maps and images", "Convert movies (MRC, EER) to compressed TIFF"),
    "relion_estimate_gain": ("Maps and images", "Estimate a gain reference from a set of movies"),
    # reconstruction and symmetry
    "relion_reconstruct": ("Reconstruction", "Reconstruct a 3D map from particles with known orientations"),
    "relion_movie_reconstruct": ("Reconstruction", "Reconstruct a 3D map from movie frames of particles"),
    "relion_backproject_2d": ("Reconstruction", "Back-project particles into 2D class images"),
    "relion_localsym": ("Reconstruction", "Local symmetry: search, refine and apply operators to masked regions"),
    "relion_particle_symmetry_expand": ("Reconstruction", "Expand particles by point-group, helical or local symmetry"),
    "relion_flex_analyse": ("Reconstruction", "Analyse multi-body motions (PCA) and make 3D movies or selections"),
    "relion_suggest_tvalue": ("Reconstruction", "Suggest a regularisation parameter T for focused classification"),
    # particles and STAR files
    "relion_convert_star": ("Particles and STAR files", "Convert a STAR file from RELION 3.0 to the current format"),
    "relion_particle_reposition": ("Particles and STAR files", "Paste projections of 2D classes or 3D maps back "
                                   "into the micrographs at the particle positions"),
    "relion_particle_select": ("Particles and STAR files", "Copy angles or offsets of matching particles from a "
                               "reference STAR file"),
    "relion_demodulate": ("Particles and STAR files", "Remove the phase modulation of particles in a STAR file"),
    "relion_batchrun": ("Particles and STAR files", "Run the commands listed in a text file, in parallel with MPI"),
    # CTF
    "relion_ctf_toolbox": ("CTF", "Simulate CTF images or pre-multiply particles by their CTF"),
    "relion_ctf_mask_test": ("CTF", "Test CTF-dependent masking of particles"),
    "relion_plot_delocalisation": ("CTF", "Plot how far the CTF delocalises signal outside the particle radius"),
    # tilt pairs
    "relion_find_tiltpairs": ("Tilt pairs", "Find matching particle coordinates on untilted and tilted micrographs"),
    "relion_tiltpair_plot": ("Tilt pairs", "Tilt-pair validation plot of a refined map"),
    # helical
    "relion_helix_toolbox": ("Helical", "Helical tools: impose or search symmetry, make masks and references, "
                             "convert coordinates, ..."),
    "relion_helix_inimodel2d": ("Helical", "Initial helical model from 2D class averages of amyloids"),
    "relion_helix_vote_classes": ("Helical", "Select helical segments by voting over their classes per filament"),
    # tomography
    "relion_run_align_tiltseries": ("Tomography", "Align tilt series with IMOD or AreTomo2 (the program behind "
                                    "the wrapper)"),
    "relion_pseudotomo_expansion": ("Tomography", "Expand 2D-stack subtomograms into pseudo-subtomograms"),
    "relion_tomo_make_optimisation_set": ("Tomography", "Make or modify an optimisation set"),
    "relion_tomo_local_particle_refine": ("Tomography", "Refine particle positions locally in the tilt series"),
    "relion_tomo_refine_mag": ("Tomography", "Refine the magnification (anisotropic) of the tilt series"),
    "relion_tomo_fit_bfactors": ("Tomography", "Fit per-tilt B-factors"),
    "relion_tomo_compute_FCC": ("Tomography", "Fourier cross-correlation of each tilt with the reference"),
    "relion_tomo_predict_tilt_series": ("Tomography", "Predict tilt series by projecting the reference at the particles"),
    "relion_tomo_split_optics": ("Tomography", "Split the tilts into optics groups"),
    "relion_tomo_find_fiducials": ("Tomography", "Find gold fiducials in the tilt series"),
    "relion_tomo_template_pick": ("Tomography", "Pick particles in tomograms by template matching"),
    "relion_tomo_find_lattice": ("Tomography", "Find a lattice of particles in a tomogram"),
    "relion_tomo_discover_motif": ("Tomography", "Find recurring arrangements in a point cloud"),
    "relion_tomo_catalogue_tomos": ("Tomography", "Write binned overview images of all tomograms"),
    "relion_tomo_convert_projections": ("Tomography", "Convert IMOD alignments into RELION projections"),
    "relion_tomo_tomo_ctf": ("Tomography", "Estimate the CTF of a tilt series"),
    "relion_tomo_powspec": ("Tomography", "Power spectra of tilt stacks"),
    "relion_tomo_bin_stack": ("Tomography", "Bin a tilt stack"),
    "relion_tomo_dark_erase": ("Tomography", "Erase dark spots (e.g. gold beads) in a tilt stack"),
    "relion_tomo_taper": ("Tomography", "Taper the edges of an image"),
    "relion_tomo_Wiener_divide": ("Tomography", "Wiener-divide a data term by a weight term"),
}


@dataclass(frozen=True)
class PositionalArg:
    key: str
    label: str
    help: str
    kind: str = "any"  # any | input (a file from the pipeline) | output (a file in the job directory)
    default: str = ""
    multi: bool = False  # several words, passed unquoted (at most one per program)


@dataclass(frozen=True)
class PositionalTool:
    category: str
    description: str
    usage: str
    args: tuple[PositionalArg, ...]
    stdout: str = ""  # what the program prints is saved to this file in the job directory ("" = not)


POSITIONAL: dict[str, PositionalTool] = {
    "relion_merge_particles": PositionalTool(
        "Particles and STAR files", "Merge particle STAR files (and their optics groups)",
        "relion_merge_particles <input1> <input2> ... <inputN> <output>",
        (PositionalArg("inputs", "Input STAR files:", "Particle STAR files to merge, separated by spaces.", multi=True),
         PositionalArg("output", "Output STAR file:", "The merged STAR file, in the job directory.", "output",
                       "particles.star"))),
    "relion_mrc2vtk": PositionalTool(
        "Maps and images", "Convert a map or stack to VTK (written next to the input file)",
        "relion_mrc2vtk X.(mrc/mrcs/tiff/spi)  ->  X.vtk",
        (PositionalArg("input", "Input map:", "Map, stack or image to convert. The .vtk file is written next to it.",
                       "input"),)),
    "relion_star_printtable": PositionalTool(
        "Particles and STAR files", "Print a table (or some of its columns) from a STAR file",
        "relion_star_printtable <starfile> <tablename> [<label1> <label2> ...]",
        (PositionalArg("starfile", "STAR file:", "The STAR file to read.", "input"),
         PositionalArg("table", "Table:", "Name of the table, e.g. data_particles or data_model_class_1.",
                       default="data_particles"),
         PositionalArg("labels", "Labels:", "Labels to print, separated by spaces, without the leading underscore "
                       "(e.g. rlnDefocusU rlnDefocusV). Empty prints every column.", multi=True)),
        stdout="table.txt"),
    "relion_star_loopheader": PositionalTool(
        "Particles and STAR files", "Write the header of a STAR table with the given labels",
        "relion_star_loopheader <label1> <label2> ...",
        (PositionalArg("labels", "Labels:", "Labels separated by spaces, e.g. rlnImageName rlnDefocusU.",
                       multi=True),),
        stdout="header.star"),
    "relion_star_datablock_stack": PositionalTool(
        "Particles and STAR files", "Write STAR table rows for the N images of a stack",
        "relion_star_datablock_stack <N> <stackname> <value1> <value2> ...",
        (PositionalArg("n", "Number of images:", "N: the number of images in the stack.", default="1"),
         PositionalArg("stack", "Stack:", "File name of the stack (.mrcs).", "input"),
         PositionalArg("values", "Other values:", "Values added to every row, separated by spaces.", multi=True)),
        stdout="datablock.star"),
    "relion_star_datablock_singlefiles": PositionalTool(
        "Particles and STAR files", "Write STAR table rows for all images matching a wildcard",
        'relion_star_datablock_singlefiles "*.spi" <value1> <value2> ...',
        (PositionalArg("pattern", "Images:", 'Wildcard for the images, e.g. "Images/*.mrc".'),
         PositionalArg("values", "Other values:", "Values added to every row, separated by spaces.", multi=True)),
        stdout="datablock.star"),
    "relion_star_datablock_ctfdat": PositionalTool(
        "Particles and STAR files", "Write STAR table rows from an XMIPP CTFDAT file",
        "relion_star_datablock_ctfdat <xmippctfdatfile>",
        (PositionalArg("ctfdat", "CTFDAT file:", "XMIPP-format CTFDAT file.", "input"),),
        stdout="datablock.star"),
}

CATEGORIES = ("Maps and images", "Reconstruction", "Particles and STAR files", "CTF", "Tilt pairs", "Helical",
              "Tomography")


def cli_programs() -> list[str]:
    return sorted(set(FLAG_TOOLS) | set(POSITIONAL))


def describe(program: str) -> tuple[str, str]:
    """(category, description) of a CLI tool."""
    if program in FLAG_TOOLS:
        return FLAG_TOOLS[program]
    t = POSITIONAL[program]
    return t.category, t.description
