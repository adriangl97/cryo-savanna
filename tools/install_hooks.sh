#!/usr/bin/env bash
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Once per clone: use the hooks in tools/hooks (the date version of each commit, tools/hooks/set_version.sh),
# and in a merge where both branches changed cryo_savanna/_version.py, write a new one for the merge (.gitattributes).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath tools/hooks
git config merge.version.name "cryo-savanna version of the merge commit"
git config merge.version.driver "tools/hooks/set_version.sh %A"
echo "hooks: tools/hooks; cryo_savanna/_version.py is set at each commit"
