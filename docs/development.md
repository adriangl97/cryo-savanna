# Developing cryo-savanna

Commands below are run from the root of a clone of this repository.

## Setting up

```bash
mamba env create -f environment.yml -p ./env
./env/bin/python -m pip install --no-deps -e .   # edits in the checkout take effect without reinstalling
./env/bin/python -m pytest
tools/install_hooks.sh   # once per clone: each commit writes its version (below)
```

The version is the date of the commit, `2026.11.14`, and `2026.11.14.2`, `.3`, ... for the 2nd, 3rd, ... commit of that day in the clone (all branches); the app shows it as `v2026.11.14`. The pre-commit hook writes it into `cryo_savanna/_version.py` (`pyproject.toml` reads it from there), and a merge commit gets its own through a merge driver of that file, so it is in the source and needs no git at runtime. Leading zeros are dropped (`2026.1.5`), as pip does. `git commit --amend` counts the commit it replaces, and `--no-verify` skips the hook.

Make the synthetic projects to look at:

```bash
./env/bin/python -m tests.fixtures.make_fixtures /tmp/cryo-savanna_fixtures
```

```bash
./env/bin/cryo-savanna --project /tmp/cryo-savanna_fixtures/spa
```

The Qt tests run offscreen (`QT_QPA_PLATFORM=offscreen`), so no display is needed.

Before a commit, also:

```bash
./env/bin/python -m ruff check .   # unused names, loop variables caught by mistake, ...
./env/bin/python -m mypy           # types, of everything but the windows (ui/, viewers/)
```

Both are set up in `pyproject.toml` and should say nothing. mypy does not look at `ui/` and `viewers/` yet:
most of what it says there is about Qt's enums through qtpy and about widgets made in a later step.

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
| `cryo_savanna/jobdefs/pytom_jobs.py`, `cryo_savanna/pytom_runner.py`, `cryo_savanna/results/pytom.py`, `cryo_savanna/viewers/pytom_view.py` | "Additional external jobs": pytom-match-pick's forms, the runner RELION starts for them (standard library only; imports `runner_common.py`), what they wrote, their viewers |
| `cryo_savanna/jobdefs/tomomask_jobs.py`, `cryo_savanna/mask_runner.py`, `cryo_savanna/viewers/tomo_mask_view.py` | "Additional external jobs": Tomogram masks' form, its runner (standard library only; writes the masks from `masks.json`, identical ones as links; imports `runner_common.py`) and the Masks view that draws them |
| `cryo_savanna/jobdefs/aretomo_jobs.py`, `cryo_savanna/aretomo_runner.py` | "Additional external jobs": AreTomo3's form and its runner (standard library only; it imports `runner_common.py`, and both are copied into the project) |
| `cryo_savanna/runner_common.py`, `cryo_savanna/jobdefs/runner_files.py` | What the runners of the "Additional external jobs" share (standard library only: External job arguments, exit files and output nodes, STAR, MRC headers, environment modules, the programs they run, RELION 5 tomograms), and how the runners and it are copied into a project's `.cryo-savanna/` |
| `cryo_savanna/jobdefs/flags.py`, `jobdefs/help/` | Flags of the program behind each job's "Additional arguments", from `--help` (bundled 5.1.1 texts as fallback) |
| `cryo_savanna/env/relion_env.py` | Reads `RELION_*` variables the same way RELION does |
| `cryo_savanna/jobdefs/external_jobs.py` | The kinds of External jobs with a form and a runner of cryo-savanna's own (CLI tools, pytom, AreTomo3, tomogram masks) in one place: which kind a job is, its runner, its form, its main input. A new wrapper is one more entry there |
| `cryo_savanna/ui/errors.py` | Errors nothing caught: in the status bar and in `errors.log` of the cache folder, not only in the terminal |
| `cryo_savanna/stamps.py` | `stamp()`: whether a file changed (what every reader and viewer compares before reading again), and `Cache`: the last few results of a slow read, shared by the viewers' threads |
| `cryo_savanna/tools/validate.py` | Checks cryo-savanna against a real RELION install |
| `tools/extract_jobdefs/` | Regenerates the job definitions from a RELION source tree |
| `tools/bundle_help.py` | Regenerates the bundled `--help` texts from the validation dumps and the RELION source |
| `tools/pipeline_ops_ref/` | RELION's own alias/mark/delete/undelete code as a small program, to test `project/ops.py` against |
| `cryo_savanna/project/disk_usage.py`, `cryo_savanna/ui/disk_usage.py` | Project → Disk usage: the size of each job, Trash/ and the rest (as `du`), what RELION's gentle and harsh clean would move (`PipeLine::cleanupJob`), emptying Trash/; and its window |
| `cryo_savanna/project/rebuild.py`, `cryo_savanna/ui/rebuild_dialog.py` | A pipeline rebuilt from job folders (lost `default_pipeline.star`, jobs run from the command line), and its review |
| `cryo_savanna/project/selection.py`, `cryo_savanna/viewers/selectors.py` | Class selection and tilt-image exclusion done in cryo-savanna, writing RELION's files |
| `tools/hooks/`, `tools/install_hooks.sh` | The version of each commit (the date), written by the pre-commit hook |
| `environment.yml`, `environment.lock.yml`, `environment-pyqt5.yml` | conda/mamba environments that install cryo-savanna |
| `bin/cryo-savanna`, `bin/savanna` (a link to it) | Run cryo-savanna from a checkout without installing it (development) |
| `docs/user_guide.md` | The user guide |
| `docs/development.md` | This file |
| `docs/other_relion_versions.md` | How to move cryo-savanna to another RELION release (untested) |
| `tests/fixtures/make_fixtures.py` | Synthetic SPA (beta-gal-like) and tomo (HIV-like) projects |

## Checking against your RELION installation

On a workstation or an HPC login node, with RELION 5.1 loaded and the cryo-savanna env active:

```bash
tools/validate_against_relion.sh
```

It compares the default `job.star` of every job type with what `relion_pipeliner` writes, for
your environment and for the compiled-in defaults. It also runs a few External jobs through
`relion_pipeliner` in a scratch project (add, alias, run, chain, continue, abort, failure, clean,
lock), then two jobs of the CLI tools menu made as the window makes them (`relion_image_handler`
on a small map, the second on the first one's output), then the "Additional external jobs" (AreTomo3 on a
tiny tilt series, tomogram masks on its tomogram, pytom template matching with those masks: AreTomo3 and
pytom are stand-ins that write the files the real ones write, so no GPU is needed), and saves `--help` of
the RELION programs.
Run it from a shell with your cluster's RELION module loaded, so that its `RELION_QSUB_*` settings
are checked too. Nothing is submitted to the queue.

It prints a summary (also in `report.txt`), writes the details to `report.json` and packs
everything into a `.tar.gz`, for a bug report. That archive names the machine it ran on: its host
name (also in the file name), the value of every `RELION_*` variable (queue names and accounts,
template and program paths) and where the RELION programs are installed. Look through
`report.json` first if any of that is private.

## Regenerating job definitions for another RELION version

```bash
git clone --depth 1 --branch 5.1.1 https://github.com/3dem/relion.git /tmp/relion-src
python tools/extract_jobdefs/extract.py /tmp/relion-src -o cryo_savanna/jobdefs/relion-5.1.1.json --version 5.1.1
```

Needs a C++17 compiler. The extractor compiles RELION's own job and window initialisers against
small stand-in classes, so defaults, ranges, help text, tabs and on/off groups come straight from
RELION's code.

The job definitions are only one part: [other_relion_versions.md](other_relion_versions.md) lists
everything that follows 5.1.1 and the steps to move cryo-savanna to another release (untested).
