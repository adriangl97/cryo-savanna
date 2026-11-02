# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What a job produced, read for the viewers: micrographs, picks, classes, particles, curves.

No Qt here: everything is read in worker threads and tested without a display. ``registry``
decides which viewers a job gets.
"""
