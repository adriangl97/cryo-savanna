# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""cryo-savanna's own selection windows for interactive jobs (viewers/selectors.py): Select (classes) and
Exclude tilt images, opened from a viewer or by running such a job; saving adds the job through
relion_pipeliner with the files RELION's own tools write (project/selection.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QMessageBox

from cryo_savanna.env.relion_env import RelionEnvError
from cryo_savanna.project.jobstar import JobStar, read_last_used
from cryo_savanna.project.selection import (
    SelectionError,
    interactive_kind,
    read_backup_selection,
    read_class_input,
    read_tilt_exclusion,
    read_tilt_input,
    regroup_count,
    relion_command,
    selected_numbers,
    write_class_selection,
    write_tilt_exclusion,
)
from cryo_savanna.ui.busy import run_busy
from cryo_savanna.ui.job_editor import MODE_CONTINUE, MODE_OVERWRITE

if TYPE_CHECKING:  # checked as the actions they are part of; at run time a plain class (no Qt diamond)
    from cryo_savanna.ui.actions.base import ActionsBase as _Base
else:
    _Base = object


class SelectionActions(_Base):
    """The class selection and tilt-image exclusion windows, and saving what they chose as a job."""

    def request_tool(self, kind: str, arg: str) -> None:
        """From a viewer: "classes" on an optimiser/model file, "tilts" on a tilt-series set (each makes
        a new job), or "change" for the Select / Exclude tilt images job ``arg``."""
        if self.readonly or self.project_dir is None:
            self.message.emit("read-only: no new jobs")
            return
        if kind == "change":
            try:
                js = JobStar.read(self.project_dir / arg / "job.star")
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self.window, "Selection", f"Cannot read {arg}job.star: {e}")
                return
            k = interactive_kind(js.type_label, js.options)
            if k is None:
                QMessageBox.information(self.window, "Selection", f"{self._label(arg)} is not an interactive selection.")
                return
            self.open_tool(k, js, "", MODE_CONTINUE, arg)
            return
        label, key = ("relion.select", "fn_model") if kind == "classes" else ("relion.excludetilts", "in_tiltseries")
        variant = self.defs.variant(label, kind == "tilts")
        last = read_last_used(self.project_dir, variant.label, variant.is_tomo)
        try:
            js = last if last is not None and interactive_kind(last.type_label, {**last.options, key: arg}) else \
                variant.new_jobstar(self.env)
        except RelionEnvError as e:
            self._env_error("Selection", e)
            return
        js.options[key] = arg
        if kind == "classes":
            for k in ("do_class_ranker", "do_filaments", "do_remove_duplicates", "do_select_values", "do_discard",
                      "do_split"):
                if k in js.options:
                    js.options[k] = "No"
        self.open_tool(kind, js, "", None, None)

    def open_tool(self, kind: str, js: JobStar, alias: str, mode, job: str | None) -> None:
        from cryo_savanna.viewers.selectors import ClassSelector, TiltExcluder
        project = self.project_dir
        if project is None:
            return
        o = js.options
        extra = o.get("other_args", "").strip()
        if extra:  # RELION passes them to its own tool; this window cannot follow them
            program = "relion_display" if kind == "classes" else "relion_python_tomo_exclude_tilt_images"
            ask = (f"This job's additional arguments ({extra}) are for {program}, which RELION runs for it. "
                   f"cryo-savanna's selection window does not use them, so what it saves can differ from what "
                   f"{program} would save (for example with --max_nr_parts_per_class).\n\nOpen cryo-savanna's "
                   f"window anyway, without them? To have them used, choose RELION's own tool under "
                   f"\"Selection window\" on the job editor's Running tab.")
            if QMessageBox.question(self.window, "Selection", ask) != QMessageBox.Yes:
                return
        what = (f"Change {self._label(job)}" if mode == MODE_CONTINUE and job else
                f"Overwrite {self._label(job)}" if mode == MODE_OVERWRITE and job else "New job")
        try:
            if kind == "classes":
                src = o["fn_model"]
                # the model and one STAR file per tilt series are read in a thread (ui/busy.py)
                inp = run_busy(lambda: read_class_input(project, src), "Reading the classes…", self.window)
                pre = read_backup_selection(project, job, len(inp.classes)) if job else None
                w = ClassSelector(project, src, selected_numbers(inp, pre),
                                  recenter=o.get("do_recenter") == "Yes",
                                  regroup=regroup_count(o),
                                  title=f"Select classes — {src} — {what}", parent=self.window)
            else:
                src = o["in_tiltseries"]
                kept = run_busy(lambda: read_tilt_exclusion(read_tilt_input(project, src), job),
                                "Reading the tilt series…", self.window) if job else None
                w = TiltExcluder(project, src, kept, title=f"Exclude tilt images — {src} — {what}",
                                 parent=self.window)
        except (OSError, ValueError, KeyError, SelectionError) as e:
            QMessageBox.warning(self.window, "Selection", str(e))
            return
        w.saved.connect(lambda choice, win=w: self._save_tool(kind, js, alias, mode, job, choice, win))
        self.tools.append(w)
        w.destroyed.connect(lambda *_, win=w: self.tools.remove(win) if win in self.tools else None)
        w.show()
        w.raise_()

    def _save_tool(self, kind: str, js: JobStar, alias: str, mode, job: str | None, choice, win) -> None:
        project, pl, _ = self._ready()
        o = dict(js.options)
        if kind == "classes":
            selected, recenter, regroup = choice
            js.options["do_recenter"] = "Yes" if recenter else "No"
            js.options["do_regroup"] = "Yes" if regroup > 0 else "No"
            if regroup > 0:
                js.options["nr_groups"] = str(regroup)
            src = o["fn_model"]

            def write(j):
                return write_class_selection(read_class_input(project, src), j, selected, recenter, regroup)
            command = lambda j: relion_command("classes", o, j, recenter, regroup)  # noqa: E731
        else:
            src = o["in_tiltseries"]

            def write(j):
                return write_tilt_exclusion(read_tilt_input(project, src), j, choice)
            command = lambda j: relion_command("tilts", o, j)  # noqa: E731

        def work():
            if mode == MODE_CONTINUE:
                assert job is not None  # a change of a job
                # the job's settings follow the choice (recenter, regroup), so that RELION's GUI
                # continues it with them too
                path = project / job / "job.star"
                if JobStar.read(path).options != js.options:
                    js.write(path)
                pl.finish_done_here(job, write, command)
                return job
            if mode == MODE_OVERWRITE:
                assert job is not None
                return pl.overwrite_done_here(job, js, write, command, alias or None).job
            return pl.add_done_here(js, write, command, alias or None).job

        if mode == MODE_OVERWRITE and job is not None:
            warning = self._trash_warning([job])
            if warning and QMessageBox.question(win, "Overwrite", f"Overwrite {self._label(job)}?{warning}") \
                    != QMessageBox.Yes:
                return

        def done(new_job):
            win.close()
            self.select.emit(new_job)
            self.editor.status.setText(f"{new_job.rstrip('/')}: selection saved.")

        win.set_busy(True, "Saving…")
        self._run("Save selection", work, done, on_error=lambda msg: win.set_busy(False, f"⚠ {msg}"))
