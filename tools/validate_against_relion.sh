#!/usr/bin/env bash
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Validate cryo-savanna against the RELION 5.1 installation in this shell.
#
#   module load relion/5.1          # or however RELION is set up here
#   tools/validate_against_relion.sh [output_dir]
#
# Needs a Python >= 3.11 with numpy and mrcfile (the cryo-savanna env, or set CRYO_SAVANNA_PYTHON).
# Runs no RELION computation and submits nothing to the queue; takes a few minutes.
# Everything is packed into <output_dir>.tar.gz, for a bug report. It holds this machine's host name (also in its file name),
# the values of every RELION_* variable (queue names, accounts, paths) and the RELION install paths:
# look through <output_dir>/report.json first if any of that is private.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-cryo_savanna_validation_$(hostname -s)_$(date +%Y%m%d_%H%M)}"
py="${CRYO_SAVANNA_PYTHON:-${CRYO_SAVANNA_ENV:+${CRYO_SAVANNA_ENV}/bin/python}}"
py="${py:-python3}"

if ! command -v relion_pipeliner >/dev/null 2>&1; then
    echo "relion_pipeliner not found: load RELION 5.1 first (e.g. module load relion/5.1)" >&2
    exit 2
fi
if ! "${py}" -c 'import sys, numpy, mrcfile; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "${py} is not Python >= 3.11 with numpy and mrcfile; set CRYO_SAVANNA_PYTHON to one that is" >&2
    exit 2
fi

echo "RELION: $(command -v relion_pipeliner)"
echo "Python: $(command -v "${py}")"
PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}" exec "${py}" -m cryo_savanna.tools.validate --out "${out}"
