# Using cryo-savanna with another RELION version

cryo-savanna is made for and tested with RELION **5.1.1**. These are the steps to move it to another
release (for example 5.0.1 or a future 5.2.0). **They have not been tried with any other version**:
treat them as a checklist, and expect some steps to need fixing on the way.

`X.Y.Z` below is the target release, as `relion_pipeliner --version` prints it without the
`-commit-...` part.

## What depends on the version

| What | Where | Follows |
|---|---|---|
| Job forms: types, options, defaults, tabs, menus | `cryo_savanna/jobdefs/relion-5.1.1.json` | `src/pipeline_jobs.cpp`, `src/gui_jobwindow.cpp`, `src/gui_mainwindow.cpp` |
| Job type ids, labels, directory names, status names, node labels | `cryo_savanna/relion_consts.py` | `src/pipeline_jobs.h`, `src/pipeliner.h`, `src/pipeline_control.h` |
| Bundled `--help` texts (only used when RELION is not loaded) | `cryo_savanna/jobdefs/help/5.1.1/` | each program's `read()` |
| Pipeline and STAR reading/writing | `cryo_savanna/project/pipeline.py`, `star.py` (`CURRENT_MDT_VERSION`) | `src/pipeliner.cpp` (`PipeLine::read`/`write`), `src/metadata_table.cpp` |
| Alias, mark finished/failed, delete, undelete | `cryo_savanna/project/ops.py` | `src/pipeliner.cpp` |
| Select and Exclude tilt images output files | `cryo_savanna/project/selection.py` | `src/displayer.cpp`, `tomography_python_programs/exclude_tilt_images` |
| Which program a job passes "Additional arguments" to | `cryo_savanna/jobdefs/flags.py` (`JOB_PROGRAMS`) | `src/pipeline_jobs.cpp` (`getCommands*`) |
| CLI tools menu, programs with an `_mpi` version | `cryo_savanna/jobdefs/cli_catalog.py`, `cli_tools.py` (`_MPI_PROGRAMS`) | `src/apps/` |
| Result viewers: output file names and STAR labels | `cryo_savanna/results/` | what each program writes |

Adding, running, continuing, aborting and cleaning jobs already go through the loaded
`relion_pipeliner`, and the "Additional arguments" flags are read from the loaded programs' `--help`,
so those follow whatever RELION is loaded.

## 1. Get the RELION source

A full clone, so the two releases can be compared:

```bash
git clone https://github.com/3dem/relion.git ~/src/relion
git -C ~/src/relion checkout X.Y.Z
```

## 2. Generate the job definitions

Needs a C++17 compiler (`CXX` to choose one):

```bash
python tools/extract_jobdefs/extract.py ~/src/relion -o cryo_savanna/jobdefs/relion-X.Y.Z.json --version X.Y.Z
```

- *"relion_consts.py JOB_TYPES differs from the RELION source"*: the release added, removed or
  renumbered job types. Update `JOB_TYPES` in `cryo_savanna/relion_consts.py` to match the message
  (and `TOMO_ONLY_LABELS` follows from the ids), then run the extractor again.
- *"harness failed to compile"*: RELION's job code uses something the stand-in classes in
  `tools/extract_jobdefs/stubs.h` do not have yet. Add what the compiler names to `stubs.h`, or
  adjust what `extract.py` copies out of the source if functions were moved or renamed.
- *"placed but undefined options"*: RELION's own GUI warns about these too; not an error.

Compare with the 5.1.1 definitions to see what changed for users:

```bash
diff <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(k, o) for k,v in sorted(d["variants"].items()) for o in sorted(v["options"])]' cryo_savanna/jobdefs/relion-5.1.1.json) \
     <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(k, o) for k,v in sorted(d["variants"].items()) for o in sorted(v["options"])]' cryo_savanna/jobdefs/relion-X.Y.Z.json)
```

## 3. Switch cryo-savanna to that version

In `cryo_savanna/jobdefs/model.py`:

```python
DEFAULT_VERSION = "X.Y.Z"
```

That one value chooses the definitions file, the bundled `--help` directory, the version the
warning compares with (so the "made for" warning now appears for 5.1.1 instead), and the version
shown in About and `--print-tree`. Keep `relion-5.1.1.json` and `help/5.1.1/`: switching back is
the same one line. Both are packaged (`pyproject.toml` takes `*.json` and `help/*/*.txt`).

## 4. Bundle the --help texts

With RELION X.Y.Z loaded, dump the help of its programs, then bundle them:

```bash
python -m cryo_savanna.tools.validate --out /tmp/savanna_validation --only help
python tools/bundle_help.py /tmp/savanna_validation/help ~/src/relion X.Y.Z
```

This writes `cryo_savanna/jobdefs/help/X.Y.Z/`. Without it, cryo-savanna still works when RELION is
loaded; it just has no flag list to fall back on when it is not.

## 5. Check the parts written by hand

See what changed in the RELION source between the two releases, in the files the table above
names:

```bash
git -C ~/src/relion diff 5.1.1 X.Y.Z --stat -- src/
git -C ~/src/relion diff 5.1.1 X.Y.Z -- src/pipeliner.cpp src/pipeliner.h src/pipeline_control.h \
    src/pipeline_jobs.h src/metadata_table.cpp src/metadata_label.h
git -C ~/src/relion diff 5.1.1 X.Y.Z --stat -- src/apps/
```

For each change, update the matching cryo-savanna file:

- `PipeLine::read`/`write`, the `# version` number of STAR files, node type labels, process
  status names → `project/pipeline.py`, `project/star.py`, `relion_consts.py`.
- `setAliasJob`, `markAsFinishedJob`, `deleteJobGetNodesAndProcesses` + `deleteNodesAndProcesses`,
  `undeleteJob` → `project/ops.py`. Then rebuild RELION's own version of these and compare (step 6).
- What `relion_display` saves for Select (`src/displayer.cpp`: `makeStarFileSelectedParticles`,
  `saveSelected`, `saveBackupSelection`, `regroupSelectedParticles`) and what
  `relion_python_tomo_exclude_tilt_images` saves (`git -C ~/src/relion diff 5.1.1 X.Y.Z --
  tomography_python_programs/exclude_tilt_images`) → `project/selection.py`.
- Which program each job runs (`getCommands*`) → `JOB_PROGRAMS` and the branches in `jobdefs/flags.py`.
- New or removed programs in `src/apps/` → `FLAG_TOOLS` / `POSITIONAL` in `jobdefs/cli_catalog.py`,
  `_MPI_PROGRAMS` in `jobdefs/cli_tools.py`.
- Renamed output files or STAR labels → the readers in `cryo_savanna/results/`.

## 6. Test against the real RELION

With RELION X.Y.Z loaded and the cryo-savanna env active:

```bash
tools/validate_against_relion.sh
```

It compares the default `job.star` of every job type with what `relion_pipeliner` writes, and runs
External jobs through it (add, alias, run, chain, continue, abort, failure, clean, lock). Every job
type should report an identical file.

The pipeline operations, against RELION's own code from a RELION X.Y.Z build directory:

```bash
python tools/pipeline_ops_ref/build.py /path/to/relion-X.Y.Z/build -o /tmp/ops_ref
RELION_OPS_REF=/tmp/ops_ref python -m pytest tests/test_ops.py
```

The extraction test, against the source:

```bash
RELION_SRC=~/src/relion python -m pytest tests/test_extract.py
```

Then the whole suite. Tests that assume 5.1.1 need the new version:
- `tests/test_jobdefs.py`, `tests/test_packaging.py`, `tests/test_preferences.py`, `tests/test_extract.py`
  (version strings and the JSON's name);
- `tests/test_against_relion.py`, `tests/test_pipeliner.py`, `tests/test_selection.py` skip unless the
  loaded RELION starts with `5.1`: change the prefix;
- `tests/test_flags.py`, `tests/test_cli_tools.py` read `tests/data/help-5.1.1/`: copy the new dumps
  from step 4 into `tests/data/help-X.Y.Z/`, or keep testing the parser on the 5.1.1 texts.

## 7. Texts

"Load RELION 5.1" appears in a few messages (`app.py`, `ui/job_actions.py`,
`project/pipeliner.py` and `tools/validate.py` in `cryo_savanna/`, and the repository's
`tools/validate_against_relion.sh`), and the README
and user guide say 5.1 / 5.1.1 throughout.

## Several versions at once

The steps above switch cryo-savanna from one release to another. Choosing the definitions from the
loaded RELION at start is possible (`JobDefs.load(version)` and `FlagSource(version)` already take
the version), but it has not been built:
- The version is only known after `relion_pipeliner --version` answers (in the background, up to
  20 s, `ui/main_window.py`), while the definitions are loaded before the window opens and are
  held by the main window, the project snapshot, the job actions and the CLI tools. Either ask
  before the window opens (slower start; cache the answer per binary), or reload all of them when
  the answer comes.
- A release without its own JSON needs a rule (nearest one of the same minor release, else the
  newest, with the warning).
- `relion_consts.JOB_TYPES` is one table; per version it should come from the JSON, which already
  holds `proc_types`.
- The warning should stay for `ops.py` and `selection.py` unless they have been checked against
  that release (step 6).
