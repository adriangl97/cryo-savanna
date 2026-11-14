# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
from cryo_savanna._version import __version__ as __version__  # the date of the commit: tools/hooks

COPYRIGHT = "Copyright (C) 2026 Adrián González-López"
# shown by --version and Help → About (GPL "how to apply": interactive programs show a short notice)
LICENSE_NOTICE = (
    "License GPLv2+: GNU GPL version 2 or later <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.\n"
    "Contains material derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; "
    "GPLv2+): see NOTICE.md."
)

# Record the user's environment before anything (qtpy sets QT_API) changes it: see env/child.py
from cryo_savanna.env import child  # noqa: E402

del child
