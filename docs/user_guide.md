# cryo-savanna user guide

cryo-savanna is an alternative GUI for RELION 5.1 projects. It works on the same files as RELION's own GUI
(`default_pipeline.star`, each job's `job.star`), runs jobs with RELION's `relion_pipeliner`, and can
be used next to RELION's GUI and scripts on the same project.

## Starting

```bash
module load relion          # RELION's programs, relion_pipeliner, its queue settings
conda activate cryo-savanna  
cd /path/to/project
cryo-savanna                      # the project in the current directory
cryo-savanna --project /path/to/project --readonly    # only look: nothing is changed
cryo-savanna --tomo               # the tomography job menu (chosen by itself for tomography projects)
cryo-savanna --print-tree         # no window: the RELION environment and the job tree
savanna                           # short name alias
```

Over `ssh -X` it needs no OpenGL; images are reduced to the size they are shown at before they are
sent, so a slow connection stays usable.

## The window

- **Left: List / Tree** (Ctrl+1, Ctrl+2). In the list each job sits under the job its main input
  came from: its tomography optimisation set, else its particle set, else the first input of its job
  window (for PostProcess, the half map rather than the mask). To place a job under another of its
  inputs, click "make main input" next to that input in the Overview; "use the default" undoes it.
  The choice is saved in the project (`.cryo-savanna/main_inputs.json`; with `--readonly`, for the
  session only). The tree draws the same links as a graph; references and masks are dashed, for the selected job or for all
  with "All links". The filter (Ctrl+F) finds jobs by name, alias, type, the program of External
  jobs, and SLURM state or node; the status menu and "Hide failed/aborted" narrow it further.
  **Job → Star** (Ctrl+B, or right-click) marks a job with ★ in the list and on its card in the tree;
  "★ Starred" shows only the starred jobs and the jobs they come from. Stars are saved in the project
  (`.cryo-savanna/stars.json`, for everyone who opens it; with `--readonly`, for the session only).
- **Right, top: the selected job.** Overview (status, results, inputs and outputs with links to the
  jobs that made or used them, SLURM history), Options (the job's settings, changed ones in bold),
  then its results (below). ⧉ or Ctrl+Shift+P opens a result in its own window.
- **Right, bottom: logs.** run.out, run.err, note.txt, following the running job.
- **Status bar:** job counts, when the project was last read, SLURM queue counts.

The window follows the project: jobs started from RELION's GUI or a script show up within the
polling interval (Preferences, or `--poll`), and views of running jobs read new files as they come.

## Running jobs

- **New job** (toolbar, Job menu, right-click): RELION's job list, SPA or tomography. The job
  editor (Ctrl+E) opens in its own window (or docked in the main window: Preferences → Job editor;
  an open editor moves at once). It has RELION's tabs and options, with its help below; changed options are marked,
  "Defaults" resets them. New jobs start from the settings last used in this project, like RELION.
- **Additional arguments** lists every flag of the program the job runs, with its default and help:
  tick a flag or type a value and it is added to the arguments.
- **Clone** (Ctrl+D), **Continue**, **Overwrite** (the job goes to Trash/ and runs again in the same
  directory with new settings), **Schedule**, **Run scheduled** (Ctrl+R), **Stop repeating runs** (a repeating run stops after its
  current job), **Abort**, **Clean**,
  **Set alias** (F2), **Star** (Ctrl+B), **Mark as finished / failed**, **Delete** (Del, to Trash/), **Undelete**.
  Trash/ keeps one copy per job directory: overwriting or deleting a job whose earlier version is
  already there (after an overwrite) deletes that version for good, and the confirmation says so.
  A job whose directory was removed by hand can still be deleted (there is then nothing to undelete).
  A Class2D, Class3D, Refine3D or InitialModel job that has not written an iteration yet can be
  marked failed (RELION refuses, and leaves it Running, so that only Delete is left); it can then be
  overwritten, and the jobs scheduled after it stay linked to it. Marking it finished is refused.
  In a project shared by a group, files must be group-writable: cryo-savanna, like RELION, does
  not change a file (the pipeline, a job.star) that you cannot write.
- **CLI tools**: RELION programs that RELION's GUI has no job for (relion_image_handler,
  relion_reconstruct, …) as job forms; they run as External jobs and appear in the pipeline.
  With MPI they run locally only, with at most RELION_ERROR_LOCAL_MPI processes (default 32), as
  RELION allows for its own local jobs. Abort asks the program to stop; one that has not stopped
  after 30 s is terminated, and killed 30 s later.
- **Queue:** "Submit to queue" uses the queue settings of the loaded RELION module (RELION_QSUB_*).
  On a SLURM cluster the list shows each queued job's state (queued and why, or running where and
  for how long), Abort can cancel it with scancel, and jobs that SLURM ended without RELION noticing
  (time limit, memory, node failure, scancel elsewhere) are marked failed, or aborted when cancelled:
  cryo-savanna writes the exit file their program would have written (RELION_JOB_EXIT_FAILURE or
  _ABORTED) and says why at the end of run.err, so RELION's GUI and a relion_pipeliner waiting for
  the job see it too. With `--readonly` nothing is written: they are marked red in the Queue column.
  A SLURM job that ended before the job was last sent to the queue (an earlier run's) does not count. cryo-savanna finds a job's SLURM jobs by their name, their
  output file, and the sbatch lines of the runs it started; for jobs that ran while it was not
  looking, by sacct's submit line (newer SLURM releases only). A submit template with
  `#SBATCH --job-name=XXXnameXXX` (RELION fills in the job directory) makes this direct, and squeue
  readable for everyone.

## Results

Each job gets the tabs that fit what it made:

| Tab | For | What |
|---|---|---|
| Micrographs / Movies | Import, MotionCorr, CtfFind, Select | List with per-micrograph values (sort, filter); the image, zoomable to full detail; CTF spectrum, motion track |
| Picks / On micrographs | AutoPick, ManualPick, Extract, Class2D, Class3D, InitialModel | Particles as circles; FOM threshold, circle size; for particles of several classes, which classes to show (any set of them, each in its own colour) |
| CTF / Motion plots | CtfFind, MotionCorr | Histograms or scatter; a clicked point shows its micrograph |
| Classes | Class2D, Class3D, InitialModel, Select | Class averages (or central sections) per iteration; sort, size, select |
| Particles | any particle file | Paged grid |
| Maps | InitialModel … LocalRes, any map | Three sections or projections, 3D surface (drag to rotate as in ChimeraX), mask outline, local-resolution colours, Open in ChimeraX; while a 3D job runs, its latest iteration's maps, updated as new iterations are written; the Iteration menu shows any earlier iteration |
| FSC | Refine3D, MultiBody, PostProcess | Gold-standard FSC of the half maps; PostProcess also the Guinier plot |
| Convergence | Class2D, Class3D, InitialModel, Refine3D, MultiBody | Resolution, class occupancy, accuracies and changes per iteration |
| Angular distribution | Refine3D, Class3D, InitialModel, MultiBody | The particles' viewing directions as a 2D map (tilt against rot), and in 3D: the map's surface with RELION's .bild cylinders around it (longer and redder: more particles), for the final map or any iteration and class; drag to rotate, wheel to zoom |
| Tilt series | tomography Import … AlignTiltSeries | Tilt series, their tilts and values, the tilt image and CTF spectrum, values against tilt angle |
| Tomograms / Picks | ReconstructTomograms, Denoise, Pick, Extract, Class3D | Z slab with side views, denoised and raw side by side, particles where the slab cuts them (of the classes chosen) |
| Files | every job | Every file: STAR tables (with plots), images, stacks, volumes, text |

"Open in napari" starts RELION's own `relion_python_tomo_view`; "Open in ChimeraX" starts
ChimeraX (from PATH, Preferences, or `$CRYO_SAVANNA_CHIMERAX`).

## Selecting classes and tilt images

The interactive steps of two RELION jobs can be done in cryo-savanna. In the job editor, the Running
tab's "Selection window" chooses cryo-savanna's selector (the default) or RELION's own relion_display /
napari tool for Run now; scheduled jobs always run RELION's tool:

- **Select** (interactive, on a Class2D/Class3D/InitialModel optimiser or model file): click classes
  to select them, as in relion_display; sort them, select all/none/invert, recenter class averages,
  regroup particles. Start it with "Select classes…" on a job's Classes tab, or with Run in the
  editor for an interactive Select job.
- **Exclude tilt images**: ↑/↓ moves through the tilts (sorted by tilt angle), Enter or Space keeps
  or excludes one, [ and ] change tilt series. Start it with "Exclude tilt images…" on a Tilt series
  tab, or with Run in the editor.

**Save selection** (Ctrl+S) adds the job through relion_pipeliner, writes the files RELION's tools
write (Select: particles.star, class_averages.star, backup_selection.star; Exclude tilt images:
selected_tilt_series.star and tilt_series/*.star) and marks the job finished, so later jobs use it
as any RELION job.

The job's **Selection** tab shows what it chose, in the same view as the selection window but
read-only: all classes with the selected ones highlighted (and how many particles they hold), or
every tilt image with its Keep mark, excluded ones red and as lines in the plots. Clicks there change
nothing. **Change selection…** (top right of that tab), or Continue, opens the selection window
again and rewrites the job's files; the tab follows. Selections saved by relion_display or napari
show the same way.

## Preferences (Ctrl+,)

Theme (light, dark, or as the system:), where the job editor opens (its own window, the default, or
docked in the main window), polling interval (at least 0.5 s), SLURM polling (at least 5 s, or none), the size of the image cache
for all projects together (`$CRYO_SAVANNA_CACHE`, else `$XDG_CACHE_HOME/cryo-savanna`, i.e.
`~/.cache/cryo-savanna`) and emptying it (only cryo-savanna's own files there are deleted), ChimeraX.
Help → Keyboard shortcuts (F1) lists every key.

## Installing

With conda or mamba, from a clone of the cryo-savanna repository:

```bash
mamba env create -f environment.yml                         # for yourself: an environment "cryo-savanna"
mamba env create -f environment.lock.yml -p /opt/cryo-savanna    # for everyone: the tested versions, a fixed path
mamba env create -f environment-pyqt5.yml -p /opt/cryo-savanna-qt5   # login nodes with glibc < 2.28
mamba env update -f environment.yml -n cryo-savanna         # after a git pull, to update to the new version
```

The environment holds Python, Qt, the viewers' libraries and cryo-savanna itself, with its two
commands `cryo-savanna` and `savanna` in its `bin/`; RELION comes from its own module. On a cluster,
put only those two commands on PATH, not the whole `bin/`: its `python3`, `pip` and Qt tools would
otherwise come first for everything started from that shell, RELION jobs included (they inherit
PATH), so an External job's `#!/usr/bin/env python3` script would run in cryo-savanna's Python.

```bash
mkdir /opt/cryo-savanna/commands
ln -s ../bin/cryo-savanna ../bin/savanna /opt/cryo-savanna/commands/
```

```lua
whatis("cryo-savanna: an alternative GUI for RELION 5.1 projects (load a RELION 5.1 module too)")
prepend_path("PATH", "/opt/cryo-savanna/commands")
```

## When something is wrong

- *relion_pipeliner not found*: load the RELION 5.1 module before starting cryo-savanna.
- *⚠ RELION x (made for 5.1.1)* in the status bar: the loaded RELION is another release. Jobs are
  still added and run by its own relion_pipeliner, but the job forms show RELION 5.1.1's options,
  and aliases, mark finished/failed, delete/undelete and the Select / Exclude tilt images files
  are written by cryo-savanna the way 5.1.1 does. Check them, or use RELION's own GUI for these.
  [other_relion_versions.md](other_relion_versions.md) describes how to move cryo-savanna to
  another release (untested).
- *A program started from cryo-savanna fails with Qt or OpenGL errors* (napari, ChimeraX,
  relion_display): cryo-savanna gives them the environment you started it with, without its own Qt
  settings; check that they start from the same shell.
- *Thumbnails look stale*: Preferences → empty the image cache.
- *Slow over ssh -X*: make the window smaller (images are drawn at window size), or use
  `ssh -X -C` for compression.
