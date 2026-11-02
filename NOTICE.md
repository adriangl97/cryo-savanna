# cryo-savanna: copyright and licence notices

cryo-savanna — a GUI for RELION projects
Copyright (C) 2026 Adrián González-López

This program is free software; you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details. You should have received a copy of the GNU General Public
License along with this program (the file `LICENSE`); if not, see
<https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.

SPDX-License-Identifier: GPL-2.0-or-later

## Material derived from RELION

cryo-savanna is not part of RELION and is not made or endorsed by the RELION developers. It
contains material derived from RELION 5.1.1 (<https://github.com/3dem/relion>): the job
definitions and the ported code from the sources at commit 29dfb4e35db63082e6321900d267966c55806176,
the `--help` texts from the programs of a RELION 5.1.1 build of a different commit of the same
release (captured with `cryo_savanna/tools/validate.py`; `tools/bundle_help.py` removes the build's
commit, its install and build paths and the host name). RELION's C++ code is distributed under the GNU General
Public License, version 2 or (at your option) any later version; its tomography Python programs
under the BSD 3-Clause License (see below):

| In cryo-savanna | Derived from (RELION source) |
|---|---|
| `cryo_savanna/jobdefs/relion-5.1.1.json`: job options, defaults, help texts, tabs and job menus | `src/pipeline_jobs.cpp`, `src/gui_jobwindow.cpp`, `src/gui_mainwindow.cpp` (extracted by `tools/extract_jobdefs/`, which compiles these functions against `stubs.h`) |
| `cryo_savanna/jobdefs/help/5.1.1/*.txt`, `tests/data/help-5.1.1/`: the `--help` output of RELION's programs (RELION 5.1.1, see above) | the programs' sources in `src/` and `src/apps/` |
| `cryo_savanna/project/pipeline.py`, `lock.py`, `ops.py`, `jobstar.py`, `cryo_savanna/relion_consts.py`: reading and writing `default_pipeline.star` and `job.star`, the pipeline lock, alias / mark finished / delete / undelete, job types and node labels | `src/pipeliner.cpp`, `src/pipeliner.h`, `src/pipeline_jobs.cpp`, `src/pipeline_jobs.h`, `src/pipeline_control.h`; for `ops.py` also `src/filename.cpp` (`decomposePipelineFileName`, `FileName::beforeNthOf`) |
| `cryo_savanna/project/star.py`: reading and writing STAR files | `src/strings.cpp`, `src/metadata_table.cpp` |
| `cryo_savanna/project/selection.py`: the files written by interactive Select and Exclude tilt images | `src/displayer.cpp`, `src/pipeline_jobs.cpp`, `src/multidim_array.h` (`centerOfMass`), `src/transformations.h` (`translateCenterOfMassToCenter`); for Exclude tilt images `src/tomography_python_programs/exclude_tilt_images/` and `_metadata_models/` (BSD 3-Clause, below) |
| `cryo_savanna/jobdefs/model.py`, `cryo_savanna/jobdefs/flags.py`: applying the `RELION_*` environment to job options, and which program gets a job's additional arguments | `src/pipeline_jobs.cpp` (`RelionJob::initialise`, `RelionJob::getCommands*`) |
| `cryo_savanna/ui/job_editor.py`: which pipeline nodes an input lists, RELION's file patterns | `src/filename.cpp`, `src/gui_entries.cpp` |
| `cryo_savanna/env/relion_env.py`: reading `RELION_*` environment variables | `src/pipeline_jobs.cpp`, `src/pipeline_jobs.h` |
| `tools/extract_jobdefs/stubs.h`: stand-ins with RELION's `JobOption` constructor signatures, and RELION's number and text conversions | `src/pipeline_jobs.h`; `src/strings.cpp` (`bestPrecision`, `floatToString`, `textToInteger`, `textToFloat`, `textToBool`) |
| `tools/pipeline_ops_ref/ops_ref.cpp`: test program linked against RELION | calls `src/pipeliner.cpp` |

RELION's source files carry this notice, which is reproduced here as it asks. The author named
differs per file: for the files of the table it is Sjors H.W. Scheres, except for the programs whose
`--help` texts are bundled, where some name other RELION authors as well, e.g. "Jasenko Zivanov &
Sjors H.W. Scheres" (`src/apps/ctf_refine.cpp`, `src/apps/motion_refine.cpp`) or Jasenko Zivanov
(`src/jaz/`, where many of those programs define their options); most tomography programs
(`src/jaz/tomography/apps/`) carry no notice. The notice, with the author of most of these files:

```
/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
```

`src/strings.cpp` and `src/metadata_table.cpp` also carry the notice of the XMIPP code they
include (GPL version 2 or later): "Authors: J.R. Bilbao-Castro (jrbcast@ace.ual.es), Unidad de
Bioinformatica of Centro Nacional de Biotecnologia, CSIC"; `src/filename.cpp` that of "Authors:
Carlos Oscar S. Sorzano (coss@cnb.csic.es), Unidad de Bioinformatica of Centro Nacional de
Biotecnologia, CSIC" (GPL version 2 or later); `src/multidim_array.h` that of "Authors: Carlos Oscar S.
Sorzano (coss@cnb.csic.es)", and `src/transformations.h` that of "Authors: Carlos Oscar S. Sorzano
(coss@cnb.csic.es), Sjors H.W. Scheres", both "Unidad de Bioinformatica of Centro Nacional de
Biotecnologia, CSIC" (GPL version 2 or later).

Each cryo-savanna file ported from these sources carries this notice in its header too, with
the XMIPP authors in `cryo_savanna/project/star.py`, `cryo_savanna/project/ops.py`,
`cryo_savanna/project/selection.py`, `cryo_savanna/ui/job_editor.py` and
`tools/extract_jobdefs/stubs.h`; `cryo_savanna/jobdefs/relion-5.1.1.json` carries it in its `notice`
field.

RELION's tomography Python programs (`src/tomography_python_programs/`, by Alister Burt) carry no
notice in their files; RELION's `pyproject.toml` declares them as "BSD 3-Clause License". Their
licence asks for this notice:

```
Copyright (c) Alister Burt and the RELION developers
(RELION 5.1.1, src/tomography_python_programs; licence as declared in RELION's pyproject.toml)

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

BSD 3-Clause code may be part of a GPL program; the BSD notice above stays with it.

RELION is developed in the group of Sjors Scheres at the MRC Laboratory of Molecular Biology, with
contributions from Tom Burnley, Alister Burt, Liyi Dong, Bjoern Forsberg, Shaoda He, Matt Iadanza,
Dari Kimanius, Erik Lindahl, Takanori Nakane, Joaquin Oton, Colin Palmer, Euan Pyle, Sjors Scheres
and Jasenko Zivanov (RELION's `AUTHORS` file). If you use RELION through cryo-savanna, please cite
the RELION papers.

## Programs cryo-savanna runs

cryo-savanna starts RELION's own programs (`relion_pipeliner`, `relion_display`, the tomography
Python tools), napari and UCSF ChimeraX as separate programs, from the user's own installation. None
of them is distributed with cryo-savanna; each keeps its own licence.

## Python packages cryo-savanna uses

Installed separately (from conda-forge or PyPI, see `environment.yml`); none is distributed with
cryo-savanna.

| Package | Licence |
|---|---|
| PySide6 / shiboken6 (Qt for Python) | LGPL-3.0-only (or GPL / commercial) |
| PyQt5 (`environment-pyqt5.yml` only) | GPL-3.0-only (or commercial) |
| qtpy | MIT |
| pyqtgraph | MIT |
| numpy | BSD-3-Clause (and others, see numpy) |
| mrcfile | BSD-3-Clause |
| tifffile (optional) | BSD-3-Clause |
| pytest, pytest-qt (tests only) | MIT |

The GNU GPL version 2 or later lets cryo-savanna be combined with LGPL-3.0 and GPL-3.0 libraries
under version 3 of the GPL.

## Source code

The complete source of cryo-savanna is its source distribution (`cryo-savanna-<version>.tar.gz`,
`MANIFEST.in`) or its git repository: the code, the tools that generate the bundled job definitions
and `--help` texts from RELION, the tests with their data, the docs and the environments. A wheel
holds only what runs, without those tools. Whoever passes on a wheel, an environment or another
built form of cryo-savanna must also give its recipients that source (GNU GPL version 2, section 3),
for example by distributing the source distribution of the same version alongside it.
