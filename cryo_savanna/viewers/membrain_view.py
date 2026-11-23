# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The Scores view of a membrain-seg job (jobdefs/membrain_jobs.py): its membrane score maps over the tomograms.

membrain-seg's scores are its network's raw output (logits): a voxel is membrane when its score is above the
segmentation threshold (0 by default, a probability of 0.5). For the tomogram chosen in the table, the view
shows

- the tomogram (as in the tomogram viewer: XY slab, XZ and YZ side views), with over its slab either the
  voxels whose score (averaged over the slab) is above the threshold, shaded, or the scores as a heatmap
  (low scores transparent), or nothing;
- the histogram of its scores (a sample of the voxels), with the threshold as a line to drag (or typed), the
  threshold its segmentation was made with as a dashed line, and the part of the tomogram above it.

The threshold is the tomogram's own, or one for all of them ("Same threshold for all tomograms"). "Save" writes
the thresholds into the job's options (job.star: tomo_thresholds); continuing the job thresholds again the
tomograms whose threshold changed, from their score maps (membrain thresholds: no GPU segmentation), and makes
their connected components and skeletons again. "Delete score maps…" deletes them once they are not needed (and
switches 'Store score maps?' off, so a continue does not segment everything again to write them).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna import membrain_runner as M
from cryo_savanna.images.mrc import MrcReader
from cryo_savanna.jobdefs import membrain_jobs as J
from cryo_savanna.project.disk_usage import human
from cryo_savanna.results.pytom import histogram, read_score_map
from cryo_savanna.stamps import stamp as _stamp
from cryo_savanna.viewers.pytom_view import STATUS_COLORS, _fmt, _item, _table
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.tomo_viewer import TomogramViewer

ABOVE, HEATMAP, NO_OVERLAY = "Above the threshold", "Score heatmap", "No overlay"
ABOVE_COLOR = (230, 159, 0)  # the shade of the voxels above the threshold
LINE_COLOR = (213, 94, 0)
HEAT = pg.colormap.get("inferno").getLookupTable(nPts=256, alpha=False)  # [256, 3] uint8


class _Viewer(TomogramViewer):
    """The tomogram viewer, telling the scores view when it drew a slab (its z range) or read a tomogram set."""

    after_slab: Callable[[], None] | None = None
    after_loaded: Callable[[], None] | None = None

    def slab_range(self) -> tuple[int, int] | None:
        if self.shape is None:
            return None
        nz, half = self.shape[0], self.slab.value() // 2
        return max(0, self.z - half), min(nz, self.z + half + 1)

    def _on_slab(self, imgs, refit: bool) -> None:
        super()._on_slab(imgs, refit)
        if self.after_slab is not None and not isinstance(imgs, TaskError):
            self.after_slab()

    def _on_loaded(self, result) -> None:
        super()._on_loaded(result)
        if self.after_loaded is not None and not isinstance(result, TaskError):
            self.after_loaded()


class MembrainScoresView(QWidget):
    COLUMNS = ["Tomogram", "Status", "Threshold", "Above (%)"]

    def __init__(self, project_dir: Path, job: str, cache=None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.cache = Path(project_dir), job, cache
        self.summary_rel = job + M.SUMMARY
        self.tasks = Tasks(self)
        self.names: list[str] = []
        self.status: dict[str, str] = {}
        self.settings: J.ThresholdSettings | None = None
        self.wanted: dict[str, float] = {}  # the threshold of each tomogram shown (saved or not)
        self.dirty = False
        self.current: str | None = None
        self.score = None  # the ScoreMap of the tomogram shown
        self._planes: tuple | None = None  # (path, stamp, z0, z1, mean scores of the slab)
        self._stamp: tuple = (None, None)
        self._syncing = False
        self.viewer: _Viewer | None = None

        self.same = QCheckBox("Same threshold for all tomograms")
        self.same.toggled.connect(self._on_same)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(-100, 100)
        self.spin.setSingleStep(0.1)
        self.spin.setPrefix("Threshold ")
        self.spin.setToolTip("membrain-seg's score (logit) above which a voxel is membrane: 0 is a probability of "
                             "0.5; lower: more membrane")
        self.spin.valueChanged.connect(self._on_spin)
        self.reset_btn = QPushButton("Job's threshold")
        self.reset_btn.setToolTip("Back to the job's 'Segmentation threshold' (for the tomogram shown, or all)")
        self.reset_btn.clicked.connect(self._reset)
        self.overlay = QComboBox()
        self.overlay.addItems([ABOVE, HEATMAP, NO_OVERLAY])
        self.overlay.setToolTip("What is drawn over the slab: the voxels above the threshold (scores averaged over "
                                "the slab's sections), the scores as a heatmap, or nothing")
        self.overlay.currentIndexChanged.connect(lambda _: self._request_planes())
        self.opacity = QSpinBox()
        self.opacity.setRange(5, 100)
        self.opacity.setValue(45)
        self.opacity.setPrefix("Opacity ")
        self.opacity.setSuffix(" %")
        self.opacity.valueChanged.connect(lambda _: self._draw_overlay())
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Write these thresholds into the job's options (job.star); continuing the job then "
                                 "thresholds again the tomograms whose threshold changed, from their score maps")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)
        self.delete_btn = QPushButton("Delete score maps…")
        self.delete_btn.setToolTip("Delete the job's score maps (they are as large as the tomograms) once they are "
                                   "not needed, and switch 'Store score maps?' off in the job")
        self.delete_btn.clicked.connect(lambda: self.delete_scores())
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.same, self.spin, self.reset_btn, self.overlay, self.opacity):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.save_btn)
        bar.addWidget(self.delete_btn)
        self.redo_label = QLabel()

        self.table = _table(self.COLUMNS)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Score (logit)")
        self.plot.setLabel("left", "Voxels (log₁₀)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for axis in ("bottom", "left"):
            self.plot.getPlotItem().getAxis(axis).enableAutoSIPrefix(False)
        self.curve = self.plot.plot([], [], stepMode="center", fillLevel=0, brush=pg.mkBrush(0, 114, 178, 120),
                                    pen=pg.mkPen(0, 114, 178))
        self.line = pg.InfiniteLine(0, angle=90, movable=True, pen=pg.mkPen(LINE_COLOR, width=2),
                                    label="threshold", labelOpts={"position": 0.9, "color": LINE_COLOR})
        self.line.sigPositionChanged.connect(self._on_line)
        self.used_line = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine),
                                         label="segmented at", labelOpts={"position": 0.75, "color": (120, 120, 120)})
        self.plot.addItem(self.line)
        self.plot.addItem(self.used_line)
        self.info = QLabel("Reading…")
        self.info.setWordWrap(True)

        self.overlay_item = pg.ImageItem()
        self.overlay_item.setZValue(1.5)  # over the slab (and its detail), under the picks

        left = QSplitter(Qt.Vertical)
        left.addWidget(self.table)
        left.addWidget(self.plot)
        left.setSizes([250, 300])
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(left)
        self.right = QWidget()
        QVBoxLayout(self.right).setContentsMargins(0, 0, 0, 0)
        self.split.addWidget(self.right)
        self.split.setSizes([420, 780])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.redo_label)
        lay.addWidget(self.split, 1)
        lay.addWidget(self.info)
        self.load()

    # ------------------------------------------------------------------ loading
    def _files(self):
        d = self.project_dir / self.job
        return _stamp(d / M.SUMMARY), _stamp(d / "job.star")

    def refresh(self) -> None:
        if self._files() != self._stamp and not self.dirty:
            self.load()
        elif self.viewer is not None:
            self.viewer.refresh()

    def load(self) -> None:
        self._stamp = self._files()
        project, job = self.project_dir, self.job
        if self._stamp[0] is None:
            self.info.setText(f"Waiting for the job to write {M.SUMMARY} (its tomograms and their progress)…")
            return

        def read():
            b = M.C.loop_block(str(project / job / M.SUMMARY), "global")
            rows = [(r.get(M.TOMO_NAME, ""), r.get(M.COL_STATUS, "")) for r in b.dicts()]
            return rows, J.read_thresholds(project, job)

        self.tasks.run(read, self._on_loaded, channel="load")

    def _on_loaded(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        rows, self.settings = result
        self.names = [n for n, _ in rows]
        self.status = dict(rows)
        self.wanted = {n: self.settings.wanted(n) for n in self.names}
        self.dirty = False
        self.save_btn.setEnabled(False)
        self.delete_btn.setEnabled(bool(self.settings.scores) and self._writable())
        self._fill_table()
        if self.viewer is None:
            self.viewer = _Viewer(self.project_dir, self.summary_rel, cache=self.cache)
            self.viewer.table.hide()  # the tomograms are chosen in this view's table
            self.viewer.after_slab = self._request_planes
            self.viewer.after_loaded = self._select_in_viewer
            self.viewer.canvas.view.addItem(self.overlay_item)
            self.right.layout().addWidget(self.viewer)
        else:
            self.viewer.load()
        keep = self.current if self.current in self.names else self._first_with_scores()
        if keep is not None:
            self.table.selectRow(self.names.index(keep))
        self._update_redo()
        self._update_info()

    def _first_with_scores(self) -> str | None:
        st = self.settings
        return next((n for n in self.names if st is not None and n in st.scores), self.names[0] if self.names else None)

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.names))
        for i, n in enumerate(self.names):
            status = self.status.get(n, "")
            cells = [_item(n), _item(status), _item(""), _item("—")]
            color = STATUS_COLORS.get(status)
            if color:
                cells[1].setForeground(pg.mkColor(*color))
            for c, it in enumerate(cells):
                self.table.setItem(i, c, it)
            self._update_row(n, i)
        self.table.blockSignals(False)

    def _update_row(self, name: str, row: int | None = None) -> None:
        st = self.settings
        if st is None:
            return
        i = self.names.index(name) if row is None else row
        used, want = st.used.get(name), self.wanted.get(name, st.job)
        text = _fmt(want) if used is None or abs(used - want) < 1e-9 else f"{_fmt(used)} → {_fmt(want)}"
        self.table.item(i, 2).setText(text)
        self.table.item(i, 2).setData(Qt.UserRole, want)
        if name == self.current and self.score is not None:
            above = 100.0 * float(np.mean(self.score.sample > want)) if len(self.score.sample) else 0.0
            self.table.item(i, 3).setText(f"{above:.2f}")
            self.table.item(i, 3).setData(Qt.UserRole, above)
            self.table.item(i, 3).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def pending(self) -> list[str]:
        """The tomograms done whose threshold shown (saved or not) is not the one they were segmented with."""
        st = self.settings
        if st is None:
            return []
        shown = J.ThresholdSettings(st.job, {n: t for n, t in self.wanted.items() if abs(t - st.job) > 1e-9},
                                    st.used, st.scores)
        return shown.to_redo(self.names)

    def _update_redo(self) -> None:
        st = self.settings
        if st is None:
            self.redo_label.setText("")
            return
        redo = self.pending()
        if not redo:
            self.redo_label.setText("")
            return
        missing = [n for n in redo if n not in st.scores]
        shown = ", ".join(redo[:6]) + (f" and {len(redo) - 6} more" if len(redo) > 6 else "")
        text = f"Continuing the job thresholds again: {shown}"
        if missing:
            text += f"   ·   without a score map, segmented again on the GPU: {', '.join(missing)}"
        self.redo_label.setText(text + ("   ·   not saved" if self.dirty else ""))

    def _update_info(self) -> None:
        st = self.settings
        if st is None:
            return
        size = sum(_size(p) for p in st.scores.values())
        maps = f"{len(st.scores)} score maps ({human(size)})" if st.scores else \
            "no score maps (switch on 'Store score maps?' and continue the job, or run it again, to see the scores)"
        self.info.setText(f"{len(self.names)} tomograms   ·   {maps}   ·   job's threshold {_fmt(st.job)}"
                          + ("   ·   not saved" if self.dirty else ""))

    # ------------------------------------------------------------------ the tomogram shown
    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.names:
            return
        self.current = self.table.item(rows[0].row(), 0).text()
        self._select_in_viewer()
        self._read_score()

    def _select_in_viewer(self) -> None:
        if self.viewer is not None and self.current is not None:
            self.viewer.select_name(self.current)

    def _score_path(self, name: str | None) -> str | None:
        st = self.settings
        return st.scores.get(name) if st is not None and name is not None else None

    def _read_score(self) -> None:
        name, path = self.current, self._score_path(self.current)
        self.score, self._planes = None, None
        self._show_threshold()
        if path is None:
            self.curve.setData([], [])
            self.overlay_item.hide()
            self.plot.setTitle(f"{name}: no score map" if name else "")
            return
        self.plot.setTitle(f"{name}: reading the score map…")
        self.tasks.run(lambda p=path: read_score_map(p), lambda res, n=name: self._on_score(n, res), channel="score")

    def _on_score(self, name: str, result) -> None:
        if isinstance(result, TaskError):
            self.plot.setTitle(f"⚠ {result}")
            return
        if name != self.current:
            return
        self.score = result
        x, counts = histogram(result.sample)
        if len(x):
            w = float(x[1] - x[0]) if len(x) > 1 else 1.0
            edges = np.append(x - w / 2, x[-1] + w / 2)
            self.curve.setData(edges, np.log10(np.maximum(counts, 0.5) + 1))
        self._show_threshold()
        self._request_planes()

    def _show_threshold(self) -> None:
        st, name = self.settings, self.current
        if st is None or name is None:
            return
        want = self.wanted.get(name, st.job)
        used = st.used.get(name)
        self._syncing = True
        self.spin.setValue(want)
        self.line.setValue(want)
        self._syncing = False
        self.used_line.setVisible(used is not None and abs(used - want) > 1e-9)  # (else under the threshold)
        if used is not None:
            self.used_line.setValue(used)
        self._update_row(name)
        self._update_title()

    def _update_title(self) -> None:
        name = self.current
        if name is None or self.score is None:
            return
        want = self.wanted.get(name, 0.0)
        above = 100.0 * float(np.mean(self.score.sample > want)) if len(self.score.sample) else 0.0
        self.plot.setTitle(f"{name}: {above:.2f} % of the tomogram above {_fmt(want)}")

    # ------------------------------------------------------------------ the overlay
    def _request_planes(self) -> None:
        """The scores of the slab shown (averaged over its sections, as the slab is), read when it moved."""
        v, path = self.viewer, self._score_path(self.current)
        rng = v.slab_range() if v is not None else None
        if path is None or rng is None or self.overlay.currentText() == NO_OVERLAY:
            self._draw_overlay()
            return
        z0, z1 = rng
        key = (path, _stamp(path), z0, z1)
        if self._planes is not None and self._planes[:4] == key:
            self._draw_overlay()
            return

        def read():
            with MrcReader(path) as r:
                return np.asarray(r.mean_planes(z0, z1), dtype=np.float32)

        self.tasks.run(read, lambda img, k=key: self._on_planes(k, img), channel="planes")

    def _on_planes(self, key: tuple, img) -> None:
        if isinstance(img, TaskError):
            self.info.setText(f"⚠ {img}")
            return
        if key[0] != self._score_path(self.current):
            return
        self._planes = key + (img,)
        self._draw_overlay()

    def overlay_rgba(self) -> np.ndarray | None:
        """What is drawn over the slab ([y, x, 4] RGBA), None for nothing."""
        mode = self.overlay.currentText()
        if self._planes is None or self.current is None or mode == NO_OVERLAY:
            return None
        scores = self._planes[4]
        alpha = int(round(2.55 * self.opacity.value()))
        rgba = np.zeros(scores.shape + (4,), np.uint8)
        if mode == ABOVE:
            rgba[scores > self.wanted.get(self.current, 0.0)] = ABOVE_COLOR + (alpha,)
            return rgba
        sc = self.score
        lo, hi = (np.percentile(sc.sample, [50, 99.9]) if sc is not None and len(sc.sample)
                  else (float(scores.min()), float(scores.max())))
        t = np.clip((scores - lo) / max(float(hi - lo), 1e-6), 0, 1)
        rgba[..., :3] = HEAT[(64 + t * 191).astype(np.uint8)]  # purple to yellow (not inferno's near-black start)
        rgba[..., 3] = (np.minimum(1.0, 2 * t) * alpha).astype(np.uint8)  # scores below the median transparent
        return rgba

    def _draw_overlay(self) -> None:
        rgba = self.overlay_rgba()
        if rgba is None:
            self.overlay_item.hide()
            return
        ny, nx = rgba.shape[:2]
        self.overlay_item.setImage(rgba, autoLevels=False)
        self.overlay_item.setRect(QRectF(0, 0, nx, ny))
        self.overlay_item.show()

    # ------------------------------------------------------------------ the thresholds
    def _on_line(self) -> None:
        if not self._syncing:
            self.set_threshold(float(self.line.value()))

    def _on_spin(self, v: float) -> None:
        if not self._syncing:
            self.set_threshold(float(v))

    def _on_same(self, on: bool) -> None:
        if on and self.current is not None and self.current in self.wanted:
            self.set_threshold(self.wanted[self.current])

    def _reset(self) -> None:
        if self.settings is not None:
            self.set_threshold(self.settings.job)

    def set_threshold(self, value: float) -> None:
        """The threshold of the tomogram shown (of all of them with "Same threshold for all tomograms")."""
        if self.settings is None or self.current is None:
            return
        names = list(self.names) if self.same.isChecked() else [self.current]
        for n in names:
            self.wanted[n] = float(value)
            self._update_row(n)
        self.dirty = True
        self.save_btn.setEnabled(self._writable())
        self._show_threshold()
        self._update_redo()
        self._update_info()
        self._draw_overlay()

    # ------------------------------------------------------------------ saving, deleting
    def _writable(self) -> bool:
        return J.job_finished(self.project_dir, self.job) and (self.project_dir / self.job / "job.star").is_file()

    def save(self) -> bool:
        if self.settings is None or not self._writable():
            if not self._writable():
                self.info.setText("The job is running: save the thresholds once it has finished")
            return False
        try:
            J.save_thresholds(self.project_dir, self.job, dict(self.wanted))
        except (OSError, ValueError, KeyError, M.Usage) as e:
            QMessageBox.warning(self, "Save thresholds", f"Could not write the job's options ({self.job}job.star): {e}")
            return False
        self.settings.per_tomogram = {n: t for n, t in self.wanted.items() if abs(t - self.settings.job) > 1e-9}
        self.dirty = False
        self._stamp = self._files()
        self.save_btn.setEnabled(False)
        redo = self.settings.to_redo(self.names)
        self._update_redo()
        self.info.setText("Saved in the job's options: " + (
            f"continue the job to threshold again {', '.join(redo)}" if redo else "nothing to threshold again"))
        return True

    def delete_scores(self, ask: bool = True) -> bool:
        st = self.settings
        if st is None or not st.scores:
            return False
        if not self._writable():
            self.info.setText("The job is running: delete the score maps once it has finished")
            return False
        redo = self.pending()
        size = sum(_size(p) for p in st.scores.values())
        if redo:
            if ask:
                QMessageBox.information(
                    self, "Delete score maps", "These tomograms have a new threshold the job has not applied yet: "
                    + ", ".join(redo) + ".\n\n" + ("Save the thresholds, then c" if self.dirty else "C")
                    + "ontinue the job first (it thresholds them again from their score maps); without the maps "
                    "they would be segmented again on the GPU.")
            return False
        if ask:
            text = (f"Delete the {len(st.scores)} score maps of {self.job} ({human(size)})?\n\nThe segmentations stay. "
                    "'Store score maps?' is switched off in the job, and other thresholds can then only be had by "
                    "segmenting again. This cannot be undone.")
            if QMessageBox.question(self, "Delete score maps", text) != QMessageBox.Yes:
                return False
        try:
            freed = J.delete_score_maps(self.project_dir, self.job)
        except (OSError, ValueError, KeyError, M.Usage) as e:
            QMessageBox.warning(self, "Delete score maps", f"Could not delete the score maps: {e}")
            return False
        self.load()
        self.info.setText(f"Deleted the score maps ({human(freed)})")
        return True


def _size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0
