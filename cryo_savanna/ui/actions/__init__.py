# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The job actions of the main window (ui/job_actions.py: JobActions), by group.

- ``base.py``: what they share: the project, its pipeliner and operations, background tasks, the job editor's
  window or dock, and helpers (``ActionsBase``);
- ``forms.py``: new, clone, continue and overwrite jobs, and submitting the job editor (``FormActions``);
- ``selection.py``: cryo-savanna's own class selection and tilt-image exclusion windows (``SelectionActions``);
- ``running.py``: run scheduled jobs, stop repeating runs, abort (and SLURM), clean, disk usage (``RunActions``);
- ``pipeline_ops.py``: alias, mark finished or failed, delete, undelete, add jobs run outside the pipeline
  (``PipelineActions``).

Each group is a mixin of ``JobActions``; for mypy it is an ``ActionsBase``, so what a group uses of the others is
declared there.
"""
