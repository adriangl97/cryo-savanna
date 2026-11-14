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

Started in a folder that is not a RELION project (no `default_pipeline.star`), it asks, as RELION's
GUI does, whether to start a new project there; or it can open a different folder, or close. A new
project is made by the installed RELION (`relion --do_projdir`, whose window shows for a second), or,
without `relion` on PATH, by cryo-savanna writing the same files (`.gui_projectdir`, `.TMP_runfiles/`
and an empty `default_pipeline.star`). With `--readonly` no project is started.

A folder that has RELION job folders (`Class2D/job005/` and so on) but no `default_pipeline.star`
(it was lost, or the jobs were run from the command line) is offered a pipeline generated from those
folders instead of a new project. The jobs found are listed for review first: untick a job to leave it out, change its
status, and see in the tooltip which inputs and outputs were found and what is uncertain. The same
happens for **Project → Open project…**.

**Project → Add jobs run outside the pipeline…** does the same for an open project: it lists the job
folders the pipeline does not have (a job run by hand into `Refine3D/job042/`, say), linked to the
jobs whose files they used, and adds the ticked ones. A command run with an output folder of its
own (`relion_refine --o Refine3D/bigbox/run …`) can be added with **Add folder as job…**: the folder
moves to the next free `<Type>/jobNNN/`, and a link is left at its old place (as the job's alias
when it was right inside the type folder), so paths in its files and in other files still work.

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
  an open editor moves at once), and closes once the job it runs has started. It has RELION's tabs and options, with its help below; changed options are marked,
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
- **Additional external jobs** (toolbar, menu bar): programs that are not RELION's, run as External
  jobs and in the pipeline like any job. See [pytom template matching](#pytom-template-matching),
  [Tomogram masks](#tomogram-masks-for-pytom) and [AreTomo3](#aretomo3-tilt-series-alignment-and-reconstruction).
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
| CTF refinement | CtfRefine | What the job fitted; mean, spread and range of the particles' defocus, astigmatism, B-factor and scale changes; per optics group beam tilt, magnification anisotropy and overall magnification (flagged when it changes the pixel size) and the Zernike coefficients, before and after; Open logfile.pdf |
| Per-particle CTF | CtfRefine | Histograms or scatter of the new values and of their changes from the input particles; the particles of a micrograph at their positions, coloured by a value (as RELION's per-micrograph plots); the mean change on every micrograph (click one to show it) |
| Aberrations | CtfRefine | Per optics group, the phase differences in Fourier space for beam tilt and trefoil, 4th-order aberrations and the magnification displacements: observed, fitted and residuals, at ±1 rad or ±π |
| Tilt series | tomography Import … AlignTiltSeries | Tilt series, their tilts and values, the tilt image and CTF spectrum, values against tilt angle |
| Tomograms / Picks | ReconstructTomograms, Denoise, Pick, Extract, Class3D | Z slab with side views, denoised and raw side by side, particles where the slab cuts them (of the classes chosen) |
| Scores | pytom template matching | Every tomogram's status (pending, running, done, failed), run time and pytom's search statistics, updated as the job goes; the score histogram of the tomogram chosen (with the score Gaussian noise passes once) and the maximum projection of its score map along z |
| Tomograms | pytom template matching | The tomograms and their score maps, Z slab and side views, one beside the other |
| Thresholds | pytom extraction | The threshold of each tomogram (or one for all, or for all the same fraction of each one's automatic cut-off), as a line to drag on the histogram of its candidates' scores, with the particles above it on the tomogram; also the maximum number of particles of each tomogram, for a continue of the job; Save rewrites particles.star (and the job's options) |
| pytom plots | pytom extraction | pytom's own SVG plots (extraction histograms, tophat filter, ROC) |
| Masks | Tomogram masks | The mask of each tomogram drawn on it: a polygon on the XY slab, through all Z or between two (tilted) planes dragged in the side views; one mask for all tomograms, or each its own; Save rewrites the masks |
| Tomograms, Tilt series, AreTomo3 | AreTomo3 | The tomograms and the aligned tilt series done so far, updated as the job goes; the AreTomo3 tab plots each tilt series' thickness, tilt axis, dark tilts and CTF resolution, with its status |
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

## Aditional external jobs 

### pytom template matching

[pytom-match-pick](https://github.com/SBC-Utrecht/pytom-match-pick) finds particles in tomograms by
template matching on GPUs. cryo-savanna runs its programs as two RELION External jobs, from
**Additional external jobs**. pytom is does not come with cryo-savanna; install it in a conda environment
of its own, and give the environment's name (or path) in the job's
Compute tab. Its programs then run with `conda run -n <environment>`; leave the name empty when they
are on the PATH already. `$CRYO_SAVANNA_PYTOM_ENV` sets the default name, and a new job
takes the environment of the project's last pytom job.

**Template matching** (`pytom_match_template.py`):

- *Inputs and template*: the tomograms.star of a Reconstruct tomograms or Denoise job. For each tomogram the job writes pytom's inputs from it and
  its tilt-series STAR files, in `inputs/<tomogram>/`: tilt angles (nominal or aligned), accumulated
  dose and defocus per tilt, voltage, Cs, amplitude contrast and the voxel size. The template can be
  prepared (`pytom_create_template.py`: final pixel size, by default the tomogram's; final box;
  contrast inverted; mirrored; low-pass). The mask is either made (`pytom_create_mask.py`: a sphere or
  an ellipsoid with a soft edge, in the template's box) or read from a file, which can be prepared
  too (final pixel size and box, mirrored: mirror it when the template is mirrored, or they no
  longer match).
- *Weighting and CTF*: per-tilt weighting, dose, CTF, phase shift, how the tomograms were
  CTF-corrected, defocus handedness, spectral whitening, band-pass filters.
- *Picking*: particle diameter or angular search, z-axis symmetry, non-spherical mask,
  random-phase correction, half precision, volume split,
  search ranges, the tomogram masks folder (a folder with `<rlnTomoName>_mask.mrc` for each tomogram,
  e.g. a [Tomogram masks](#tomogram-masks-for-pytom) job's `masks/`, chosen with **…**; a path with
  `{name}` for each tomogram also works), more pytom arguments.
- *Compute*: the GPU ids, separated by colons as in RELION (`0:1:2:3`), and how several GPUs are used: one tomogram per GPU at a time or each
  tomogram over all GPUs. Tomograms already done are skipped when the job is run
  again (Continue), so after a failure only the rest run.
- *CUDA module*: loaded with `module load` before pytom runs, so that pytom (CuPy) finds CUDA. The
  default is the CUDA module loaded where cryo-savanna runs, else the default CUDA module that
  `module avail` lists (`$CRYO_SAVANNA_CUDA_MODULE` overrides it). With no CUDA modules the field
  is empty: nothing is loaded and pytom uses the CUDA on the job's PATH. The same
  field is on the extraction job's *Compute* tab, and new pytom jobs take it, with the conda
  environment, from the project's last pytom job.
- *Running*: as for any job, locally or through the queue (RELION_QSUB_*). On a cluster, ask for
  the GPUs in your submission template.  

The job writes `pytom_match.star` (one row per tomogram: the tomograms.star columns, the score map,
the status) as tomograms finish, so its Scores and Tomograms tabs show the results while it runs.

**Extract candidates** (`pytom_extract_candidates.py`): from a template matching job, with pytom's
automatic cut-off (from the number of false positives allowed) or a fixed one, optionally with the
tophat filter, other tomogram masks, and `pytom_estimate_roc.py`. With "Keep candidates below the
cut-off", the peaks down to a fraction of each tomogram's cut-off are kept as candidates, and the
**Thresholds** tab sets the threshold of each tomogram (or one for all, or for all tomograms the
same fraction of each one's automatic cut-off, e.g. 0.8 ×) on the histogram of their scores, showing
the particles above it on the tomogram; **Save** rewrites particles.star. There too, the maximum
number of particles of the tomogram shown can be raised or lowered: Save writes it into the job's
options ("Max. particles of some tomograms"), and **Continue** extracts again only the tomograms whose
maximum changed; the others keep their candidates, and every tomogram keeps the threshold chosen for
it. Changing another setting of the job (e.g. "Candidates down to") extracts every tomogram again,
with its new cut-off as threshold.
pytom writes into the extraction job's own directory (the matching job's score maps are linked
there). The output is RELION 5's: `particles.star` (rlnTomoName, rlnCenteredCoordinate[XYZ]Angst,
the angles as rlnAngleRot/Tilt/Psi or rlnTomoSubtomogramRot/Tilt/Psi, rlnLCCmax and
rlnAutopickFigureOfMerit) and `optimisation_set.star` with the tomograms, for Extract subtomos.
Like matching, extraction has to run on a machine with a GPU and its NVIDIA driver: pytom's
programs look for a GPU as they start, even for the extraction itself. Submit it to a GPU queue
(one GPU is enough) or run it on a GPU workstation. On a machine without one (a CPU node) every tomogram fails with `cudaErrorInsufficientDriver`.

### Tomogram masks (for pytom)

**Additional external jobs → Tomogram masks (for pytom)** makes a mask per tomogram of a tomogram set,
for pytom's tomogram mask: pytom searches (and extracts) only where the mask is above 0, e.g. to leave
out the volume above and below a lamella, or ice and carbon at its edges.

The job (an External job, run like any other) writes a starting mask for each tomogram, the whole
tomogram or a slab around its middle, as `masks/<rlnTomoName>_mask.mrc` in the job: binary (int8),
with the tomogram's size and voxel size. Draw the masks in the job's **Masks** tab:

- Choose a tomogram in the table. On the XY slab, drag the corners of the polygon or the whole polygon. Click
  an edge to add a corner, and right-click a corner to remove it. What lies outside the mask at the
  slab's Z is shaded.
- *Z*: **All Z** leaves Z unmasked: the polygon goes through the whole tomogram, so tomograms of
  other thicknesses are not cut. **Between two planes** keeps only the slab between a bottom and a top
  plane, shown as lines in the XZ and YZ views. Drag a line to move its plane, or drag one end to tilt
  it (along x in XZ, along y in YZ) to follow a tilted lamella.
- **Same mask for all tomograms** gives every tomogram the mask of the one shown. While it is on,
  each edit goes to all of them. Switch it off to edit the tomograms one by one from there. Corners
  and planes are kept relative to each tomogram's centre, so a shared mask stays centred in
  tomograms of another size.
- **Undo** (Ctrl+Z) takes the last edit back, for every tomogram it changed; **Redo** (Ctrl+Shift+Z)
  makes it again. Edits can be taken back until the masks are read again (the job run again).
- **Save masks** rewrites the masks that changed. Identical masks are written once: the others are
  symbolic links to that file. **Reset** goes back to the whole tomogram.

Give the job's `masks/` folder to template matching (*Picking → Tomogram masks folder*) or to
extraction (*Tomogram mask → Other mask(s)*). The pytom job then appears under the mask job in the
pipeline. pytom stops with a message if a tomogram has no mask in the folder or its mask is not the
tomogram's size: draw the masks on the tomograms pytom searches. Running the mask job again
(Continue) keeps the masks already drawn and adds the new tomograms. The geometry is kept in
`masks.json`, and the list of masks in `tomogram_masks.star` (the job's output).

### AreTomo3: tilt-series alignment and reconstruction

[AreTomo3](https://github.com/czimaginginstitute/AreTomo3) aligns tilt series and reconstructs
tomograms on GPUs. **Additional external jobs → AreTomo3: align and reconstruct tilt series** runs it
on a RELION 5 tilt-series set, as a RELION External job. AreTomo3 does not come with cryo-savanna:
give its executable on the *Compute* tab (`$CRYO_SAVANNA_ARETOMO3` sets the default, else
`AreTomo3` on the PATH), and the module(s) it needs (CUDA, or AreTomo3's own module), which are
loaded with `module load` first. A new job takes both from the project's last AreTomo3 job.

- *Input*: the tilt_series.star of MotionCorr, CtfFind or Exclude tilt images (the job needs the
  motion-corrected images, `rlnMicrographName`). For each tilt series the job stacks its images,
  sorted by tilt angle, into `inputs/<name>/<name>.mrc` (float32, deleted when the tilt series is
  done unless kept), with `<name>_TLT.txt`: tilt angle, acquisition order and dose of each image
  (from `rlnMicrographPreExposure`). Pixel size, voltage, Cs and amplitude contrast come from the
  tilt-series set. With *Odd/even tomograms* the odd and even images (`rlnMicrographNameEven/Odd`,
  MotionCorr's "Save images for denoising?") are stacked too and reconstructed as half tomograms.
- *Alignment*: **Align with AreTomo3** (`-Cmd 1`: CTF estimation, alignment, CTF correction and
  reconstruction): tilt axis (refine the nominal `rlnTomoNominalTiltAxisAngle`, keep it, or search),
  alignment thickness, local alignment patches, tilt-angle offset, dark-image tolerance, phase
  plate. Or **Use the alignment of the input** (`-Cmd 2`, reconstruction only) with the tilt series
  of an alignment job: RELION's own (IMOD or AreTomo2), or an earlier AreTomo3 job, whose local
  alignment and measured thickness are kept when the tilts are still the same. The input's CTF
  values go to AreTomo3 for the CTF correction. With an alignment that is not AreTomo3's, give the
  tomogram thickness.
- *Reconstruction*: binning (up to three: the second and third give further tomograms in
  `aretomo/<name>/`), thickness (empty: the measured sample thickness plus the extra thickness),
  SART or WBP, CTF correction (local, phase flip, none, or CTF multiplication for template
  matching), tilt range, layout of the tomogram (`-FlipVol 1` or `2`), contrast, more AreTomo3
  arguments.
- *Compute*: GPU ids separated by colons (`0:1:2:3`): one tilt series per GPU at a time (or more,
  with *Tilt series per GPU*). Tilt series already done are skipped when the job runs again
  (Continue), so after a failure only the rest run.

The job writes, as tilt series finish (so that its tabs show them while it runs):

- `aligned_tilt_series.star`, with per-tilt STAR files in `tilt_series/`, the alignment in RELION's
  convention as RELION's own AreTomo wrapper writes it (`rlnTomoXTilt` 0, `rlnTomoYTilt`,
  `rlnTomoZRot`, `rlnTomoX/YShiftAngst`). Tilts AreTomo3 found dark are left out. It is an input for
  RELION's **Reconstruct tomograms** and the rest of RELION's tomography pipeline. AreTomo3's
  defocus per tilt goes into it when the input has none (before CtfFind), always, or never.
- `tomograms.star`: AreTomo3's tomograms (`aretomo/<name>/<name>_Vol.mrc`, the halves as
  `rlnTomoReconstructedTomogramHalf1/2`), their size and binning, for the Tomograms tab, template
  matching and denoising.
- `aretomo3.star`: one row per tilt series, with status, run time, tilt axis, sample thickness, tilt
  offsets, dark tilts and CTF resolution. AreTomo3's own output (`.aln`, `_CTF.txt`, logs) is in
  `aretomo/<name>/`; each tilt series' log is in `logs/`.

AreTomo3's tomograms are AreTomo3's reconstructions, not RELION's. Before you use particles picked
on them for subtomogram averaging in RELION, check that their hand and centre match a RELION
reconstruction of `aligned_tilt_series.star` (choose the other *Tomogram layout* if the hand is
inverted).

## Preferences (Ctrl+,)

Help → Keyboard shortcuts (F1) lists every key.

### Theme

Light, dark, or as the system.

### Where the job editor opens

Its own window, the default, or docked in the main window.

### Polling interval

At least 0.5 s.

### SLURM polling

At least 5 s, or none.

### Size of the image cache

For all projects together in `$CRYO_SAVANNA_CACHE`, else `$XDG_CACHE_HOME/cryo-savanna`, i.e.
`~/.cache/cryo-savanna`. There is an option to empty it (only cryo-savanna's own files there are
deleted).

### ChimeraX path

The ChimeraX that "Open in ChimeraX" starts; empty, the one on PATH. A `$CRYO_SAVANNA_CHIMERAX` set
in the shell wins over this preference.

### Scratch directory

The default of "Copy particles to scratch directory" for new jobs, as `$RELION_SCRATCH_DIR` would
set it. A `$RELION_SCRATCH_DIR` set in the shell wins over this preference.

### Scratch subdirectory of its own for each job

Off by default: the directory is used as typed, as in RELION's GUI. If set to on, every job started
or scheduled with a scratch directory (new, cloned or overwritten) gets a subdirectory of its own in
it, `savanna_NNN`, with a random 3-digit number that no other job of the project uses and that is
not on disk yet. RELION wipes `<scratch>/relion_volatile/` when a job starts and ends, so jobs
sharing one directory would destroy each other's copies. This can lead to an accidental crash of a
job if you run two at the same time sharing a scratch folder. This setting avoids that. Continuing a
job keeps its directory.

The caveat with this is that RELION removes only `relion_volatile/` of a job that finishes ok, so
the copies of failed, aborted or killed jobs stay until you delete them (with one shared directory,
the next job would have wiped them). The "Wipe savanna folders…" button deletes them: it lists every
`savanna_NNN` folder directly in the scratch directory (`$RELION_SCRATCH_DIR` from the shell, else
the one in Preferences), with full paths, and deletes them only after you confirm. Jobs running with
their scratch in one of them crash, so use it only when no job using scratch is running.

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
