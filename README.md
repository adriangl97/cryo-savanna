# cryo-savanna, where the RELIONs live

An alternative GUI for RELION 5.1, for both single-particle analysis (SPA) and tomography
(`--tomo`). It is not made or endorsed by the RELION developers.

cryo-savanna works on RELION's standard project files (`default_pipeline.star`, `job.star`), runs
jobs through RELION's own `relion_pipeliner`, and takes its queue settings from the `RELION_*`
environment of the loaded RELION module, as RELION's GUI does. The two GUIs are fully compatible
and can be used interchangeably, even at the same time, on the same project.

**Status:** version 0.9. Early but working. Made for and checked against RELION 5.1.1. See the
[user guide](docs/user_guide.md).

> **Disclaimer:** this project uses AI-assisted coding. Please report any issues you find.

## What it adds compared to RELION's GUI

**Fully works over X-11 forwarding**

- Nothing needs OpenGL, and images are downsampled to the size they are shown at before they are
  drawn, so the GUI stays usable over `ssh -X` and slow connections.
- Built-in windows for selecting 2D/3D classes and for excluding tilt images, so neither
  `relion_display` nor napari has to be started. They write the same files as RELION's tools, and
  RELION's own tools remain available.

**Visualizing the project**

- Two views of the pipeline: an indented list, where each job sits under the job its main input
  came from, and a graph tree whose job cards show the key information about each job. Which input
  counts as the main one can be changed per job.
- A filter (Ctrl+F) by name, alias, job type, the program behind External jobs, or SLURM state and
  node, plus a status menu and "Hide failed/aborted".
- An Options tab for every job, with the settings that differ from the defaults in bold.
- Live updates: jobs started from RELION's GUI or a script appear on their own, and the views of a
  running job follow its new files.
- Keyboard shortcuts for the common actions (Help → Keyboard shortcuts, F1).

**Results inside the GUI** (see [Results](#results) for the full list)

- Micrographs with their per-micrograph values, CTF spectra and fits, motion tracks, and picks.
- 2D class averages and 3D class sections per iteration, sortable by distribution, resolution,
  particle count or class score.
- Plots: CTF and motion histograms and scatter plots, FSC, Guinier, convergence per iteration,
  and the angular distribution in 2D and in 3D.
- A map viewer with orthogonal sections, an isosurface, mask outlines and local-resolution
  colouring, which follows the latest iteration of a running 3D job. "Open in ChimeraX" takes the
  same view across.
- Tilt series, tomograms and the particles picked in them.
- A Files tab for every job: STAR files as sortable tables with column plots, and MRC images,
  stacks and volumes.
- Any viewer can be opened in its own window (⧉ or Ctrl+Shift+P).

**Running jobs**

- Every flag of the program a job runs, read from its `--help`, is listed under "Additional
  arguments" with its default and help text, to tick and fill in.
- RELION's command-line-only programs (`relion_image_handler`, `relion_reconstruct`, the
  `relion_tomo_*` tools, ...) can be run from the **CLI tools** menu. They appear in the pipeline
  as ordinary External jobs, with their inputs and outputs.
- Class2D, Class3D, Refine3D and InitialModel jobs that have not written an iteration yet can be
  marked failed (RELION refuses and leaves them Running), and then overwritten.

**SLURM**

- Queued jobs show their SLURM state (queued and why, or running where and for how long) in the
  list, on the tree's job cards and in the job overview, with their `sacct` history.
- Abort can cancel a queued job with `scancel`.
- Jobs that SLURM ended without RELION noticing (time limit, out of memory, node failure,
  `scancel`) are marked failed, or aborted when cancelled, for RELION's GUI and `relion_pipeliner`
  as well.

## Installing

With conda or mamba, from a clone of this repository:

```bash
mamba env create -f environment.yml          # creates an environment "cryo-savanna"; conda works the same
mamba activate cryo-savanna
```

- If you run into problems, use `environment.lock.yml` instead: it pins the tested versions.
- On login nodes with glibc < 2.28, use `environment-pyqt5.yml`.
- To install with pip instead, include the GUI extra: `pip install '.[gui]'`. A plain
  `pip install .` gives only `--print-tree`.

To update after a `git pull`, run `mamba env update -f environment.yml` (or
`environment.lock.yml`, with the same `-n` / `-p` as when you created it).

## Usage

The command is `cryo-savanna`, or `savanna` for short. Load your RELION 5.1 module first.

```bash
cryo-savanna --project /path/to/relion/project               # opens the project
savanna --project /path/to/relion/project --readonly         # only look: nothing is changed
cryo-savanna --project /path/to/relion/project --tomo        # the tomography job menu
cryo-savanna --project /path/to/relion/project --print-tree  # no window: environment and job tree
```

The tomography job menu is chosen automatically when the project already has tomography jobs.

cryo-savanna shows the project's jobs as an indented list and a tree, with each job's inputs,
outputs, options, results and logs, and follows the changes RELION makes. Jobs can be created,
cloned, continued, overwritten, scheduled and run (through `relion_pipeliner`), aborted, cleaned,
renamed, marked finished or failed, deleted and undeleted. `--readonly` turns all of that off.
Right-click a job in the list or the tree for its menu.

### Keyboard shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| Ctrl+F | Filter | Ctrl+D | Clone |
| F5 | Refresh | Ctrl+R | Run scheduled jobs |
| Ctrl+1 | List | F2 | Set alias |
| Ctrl+2 | Tree | Del | Delete |
| Ctrl+0 | Fit tree | Ctrl+E | Job editor |
| Ctrl+Shift+P | Viewer in its own window | Ctrl+, | Preferences |

Help → Keyboard shortcuts (F1) lists them all.

### CLI tools

The **CLI tools** menu offers RELION's command-line-only programs as job forms. They run as
External jobs through `cryo_savanna/cli_runner.py`, which is copied into the project as
`.cryo-savanna/cli_runner.py`, so the jobs keep working when cryo-savanna is moved or updated.

### SLURM

On a SLURM cluster, cryo-savanna follows the jobs submitted to the queue (see
[What it adds to RELION's GUI](#what-it-adds-to-relions-gui)). For jobs that SLURM ended without
RELION noticing, it writes the exit file their program would have written, so that RELION's GUI
and a waiting `relion_pipeliner` see it too; with `--readonly` they are only flagged.
`--queue-poll S` sets how often `squeue` is asked (default 20 s, 0 = off).

### Files cryo-savanna keeps

- `<project>/.cryo-savanna/runs/`: a record of the pipeliner runs cryo-savanna starts.
- `<project>/.cryo-savanna/main_inputs.json`: the main inputs chosen with "make main input".
- Micrograph previews are cached in `$CRYO_SAVANNA_CACHE`, else `$XDG_CACHE_HOME/cryo-savanna/`
  (`~/.cache/cryo-savanna/`), 2 GB at most for all projects together (Preferences).

## Results

Each job's results open as tabs next to its Overview and Options:

- **Micrographs**: a sortable list of the micrographs with their values (motion, defocus,
  astigmatism, max. resolution, FOM, picks), and the selected one large. CtfFind adds the power
  spectrum with its fit, MotionCorr the motion track.
- **Picks**: AutoPick and ManualPick picks as circles, with a figure-of-merit threshold. Extract
  jobs show their particles on their micrographs.
- **Classes**: the class averages of each Class2D iteration, and the central sections of 3D
  classes. They can be sorted by distribution, resolution, particles or class score, and selected.
- **Particles**: any particle file, as a paged grid.
- **Plots**: CTF and motion histograms or scatter plots (a clicked point shows its row), FSC
  curves (PostProcess, Refine3D), Guinier, convergence per iteration (resolution, class
  occupancy, accuracies, changes), and the angular distribution: a 2D map of the viewing
  directions and, in 3D, the map's surface with RELION's `.bild` cylinders around it (final map or
  any iteration and class).
- **Maps** (InitialModel, Class3D, Refine3D, MaskCreate, PostProcess, LocalRes, any map output):
  - three sections that cross at one point, or projections; click a section to move the other two;
  - the isosurface, drawn by cryo-savanna itself (no OpenGL). Drag to rotate, wheel to zoom. The
    level starts at 1% of the box, or at 0.5 for masks;
  - the job's mask as a red outline (PostProcess, MaskCreate, Refine3D/Class3D with a mask);
  - LocalRes colours the sections and the surface by local resolution, blue (best) to red;
  - "Open in ChimeraX" opens the same map, level, mask and colours there (ChimeraX from PATH or
    `$CRYO_SAVANNA_CHIMERAX`);
  - while Class3D, InitialModel, Refine3D or MultiBody runs (or when it stopped before its final
    maps): the maps of its latest iteration (Refine3D: the half maps), replaced by each new
    iteration as it is written, keeping the class shown, the position and the level;
  - the Iteration menu shows any earlier iteration instead (next to the final maps, or the latest
    iteration while the job runs).
- **Tilt series** (tomography: Import, MotionCorr, CtfFind, ExcludeTiltImages, AlignTiltSeries):
  the tilt series of the set, the tilts of the selected one (angle, dose, defocus, CTF, motion,
  alignment), the tilt image and its CTF spectrum, and the per-tilt values against the tilt angle.
- **Tomograms** (ReconstructTomograms, Denoise, ...): an XY slab of a few averaged sections, moved
  with the Z slider, at full resolution; XZ and YZ side views; denoised and raw side by side.
- **Picks** (Pick tomograms, Extract subtomos, tomography Refine3D): the particles in their
  tomograms, as circles where the slab cuts them, with counts per tomogram.
- **Selection** (Select and Exclude tilt images jobs): the job's choice, shown read-only in the
  same view as the selection window; "Change selection…" opens the window again.
- **Files**: every file of the job. STAR files open as sortable tables with column plots; MRC
  images and stacks open as images, volumes in the map viewer.

"Open in napari" starts RELION's own `relion_python_tomo_view` (from the loaded RELION module) for
the same tilt series, tomograms or particles. The interactive tomography jobs (exclude tilt images,
picking) open RELION's napari tools when run, as in RELION's GUI.

Views of running jobs follow new files (a new iteration, more micrographs). ⧉ opens a view in its
own window.

### Selecting classes and excluding tilt images

"Select classes…" on a Classes tab (or Run of an interactive Select job) and "Exclude tilt
images…" on a Tilt series tab (or Run of an Exclude tilt images job) open cryo-savanna's own
selection windows. Saving adds the job through `relion_pipeliner`, writes the files that RELION's
`relion_display` and napari excluder write, and marks the job finished. The job editor's Running
tab ("Selection window") switches a job to RELION's own tool.

### Job tree placement

In the list and the tree, each job sits under the job its main input came from: its tomography
optimisation set, else its particles, else the first input of its job window. "make main input"
next to another input in the Overview changes that for the job.

## Layout

| Path | What |
|---|---|
| `cryo_savanna/project/` | STAR reader/writer, `default_pipeline.star` model, `job.star`, RELION's lock protocol, job lineage, read-only project snapshots |
| `cryo_savanna/ui/` | Qt windows: job tree, job graph (+ Qt-free layout), job info, logs, main window |
| `cryo_savanna/images/` | MRC reads (headers, slices, regions; TIFF movies with `tifffile`), downsampling, thumbnail cache |
| `cryo_savanna/results/` | What jobs produced, read for the viewers (no Qt), and which viewers a job gets (`registry.py`) |
| `cryo_savanna/viewers/` | The viewer widgets: image canvas, micrographs, galleries, plots, STAR tables, files |
| `cryo_savanna/jobdefs/relion-5.1.1.json` | Every job option and the job-window layout of RELION 5.1.1, SPA and tomo (generated) |
| `cryo_savanna/jobdefs/model.py` | Loads the definitions and applies the `RELION_*` environment |
| `cryo_savanna/jobdefs/cli_catalog.py`, `cli_tools.py`, `cryo_savanna/cli_runner.py` | CLI tools: the catalog, job forms from `--help`, and the runner RELION starts for them |
| `cryo_savanna/jobdefs/flags.py`, `jobdefs/help/` | Flags of the program behind each job's "Additional arguments", from `--help` (bundled 5.1.1 texts as fallback) |
| `cryo_savanna/env/relion_env.py` | Reads `RELION_*` variables the same way RELION does |
| `cryo_savanna/tools/validate.py` | Checks cryo-savanna against a real RELION install |
| `tools/extract_jobdefs/` | Regenerates the job definitions from a RELION source tree |
| `tools/bundle_help.py` | Regenerates the bundled `--help` texts from the validation dumps and the RELION source |
| `tools/pipeline_ops_ref/` | RELION's own alias/mark/delete/undelete code as a small program, to test `project/ops.py` against |
| `cryo_savanna/project/selection.py`, `cryo_savanna/viewers/selectors.py` | Class selection and tilt-image exclusion done in cryo-savanna, writing RELION's files |
| `environment.yml`, `environment.lock.yml`, `environment-pyqt5.yml` | conda/mamba environments that install cryo-savanna |
| `bin/cryo-savanna`, `bin/savanna` (a link to it) | Run cryo-savanna from a checkout without installing it (development) |
| `docs/user_guide.md` | The user guide |
| `docs/other_relion_versions.md` | How to move cryo-savanna to another RELION release (untested) |
| `tests/fixtures/make_fixtures.py` | Synthetic SPA (beta-gal-like) and tomo (HIV-like) projects |

## Development

```bash
mamba env create -f environment.yml -p ./env
./env/bin/python -m pip install --no-deps -e .   # edits in the checkout take effect without reinstalling
./env/bin/python -m pytest
```

Make the synthetic projects to look at:

```bash
./env/bin/python -m tests.fixtures.make_fixtures /tmp/cryo-savanna_fixtures
```

```bash
./env/bin/cryo-savanna --project /tmp/cryo-savanna_fixtures/spa
```

The Qt tests run offscreen (`QT_QPA_PLATFORM=offscreen`), so no display is needed.

### Checking against your RELION installation

On a workstation or an HPC login node, with RELION 5.1 loaded and the cryo-savanna env active:

```bash
tools/validate_against_relion.sh
```

It compares the default `job.star` of every job type with what `relion_pipeliner` writes, for
your environment and for the compiled-in defaults. It also runs a few External jobs through
`relion_pipeliner` in a scratch project (add, alias, run, chain, continue, abort, failure, clean,
lock), then two jobs of the CLI tools menu made as the window makes them (`relion_image_handler`
on a small map, the second on the first one's output), and saves `--help` of the RELION programs.
Run it from a shell with your cluster's RELION module loaded, so that its `RELION_QSUB_*` settings
are checked too. Nothing is submitted to the queue.

It prints a summary (also in `report.txt`), writes the details to `report.json` and packs
everything into a `.tar.gz`, for a bug report. That archive names the machine it ran on: its host
name (also in the file name), the value of every `RELION_*` variable (queue names and accounts,
template and program paths) and where the RELION programs are installed. Look through
`report.json` first if any of that is private.

### Regenerating job definitions for another RELION version

```bash
git clone --depth 1 --branch 5.1.1 https://github.com/3dem/relion.git /tmp/relion-src
python tools/extract_jobdefs/extract.py /tmp/relion-src -o cryo_savanna/jobdefs/relion-5.1.1.json --version 5.1.1
```

Needs a C++17 compiler. The extractor compiles RELION's own job and window initialisers against
small stand-in classes, so defaults, ranges, help text, tabs and on/off groups come straight from
RELION's code.

The job definitions are only one part:
[docs/other_relion_versions.md](docs/other_relion_versions.md) lists everything that follows 5.1.1
and the steps to move cryo-savanna to another release (untested).

## License

Copyright (C) 2026 Adrián González-López. cryo-savanna is free software under the GNU General
Public License, version 2 or (at your option) any later version ([LICENSE](LICENSE)); it comes with
NO WARRANTY. It contains material derived from RELION (Sjors H.W. Scheres, MRC Laboratory of
Molecular Biology; GPL version 2 or later): the job definitions and help texts, and the formats of
the pipeline, STAR, job.star and selection files; the tilt-image exclusion follows RELION's
tomography Python programs (Alister Burt; BSD 3-Clause). [NOTICE.md](NOTICE.md) lists what comes
from where, RELION's copyright notice, and the licences of the packages cryo-savanna uses. Every
source file starts with its copyright line and `SPDX-License-Identifier: GPL-2.0-or-later`; the
files ported from RELION also carry RELION's notice (and the XMIPP authors, where RELION's code
there is XMIPP's).
