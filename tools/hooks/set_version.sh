#!/usr/bin/env bash
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Writes cryo_savanna/_version.py for the commit being made: the date (2026.11.14), and .2, .3, ... for the
# 2nd, 3rd, ... commit of that day in this clone (all branches, local and fetched). Run by the pre-commit hook,
# and as the merge driver of that file with the path to write (tools/install_hooks.sh): a merge commit gets its own
# version too. The version is in the source, so no git is needed at runtime.
set -euo pipefail
out=${1:+$(cd "$(dirname "$1")" && pwd)/$(basename "$1")}  # the merge driver's file, before the cd
cd "$(git rev-parse --show-toplevel)"
read -r y m d <<<"$(date +'%Y %m %d')"
n=$(git log --branches --remotes --format=%cd --date=format-local:%Y.%m.%d 2>/dev/null | grep -cx "$y.$m.$d" || true)
# PEP 440 drops leading zeros (2026.01.05 is 2026.1.5): written that way, pip and the app show the same version
version=$(printf '%d.%d.%d' "$((10#$y))" "$((10#$m))" "$((10#$d))")
[ "${n:-0}" -gt 0 ] && version="$version.$((n + 1))"
file=${out:-cryo_savanna/_version.py}
new=$(printf '# Copyright (C) 2026 Adrián González-López\n# SPDX-License-Identifier: GPL-2.0-or-later\n# Written by tools/hooks/set_version.sh at each commit: the date, .N for the Nth commit of a day.\n__version__ = "%s"\n' "$version")
if [ "$(cat "$file" 2>/dev/null)" != "$new" ]; then
    printf '%s\n' "$new" >"$file"
fi
[ -n "$out" ] || git add "$file"
