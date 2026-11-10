# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The viewer widgets the registry names, each built from a ViewSpec and able to refresh itself.

Every view has ``refresh()``: it stats the files it shows and reads them again when they changed,
keeping the zoom, sort order and selection (a running Class2D gets a new iteration, a running
CtfFind more micrographs).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryo_savanna.env.child import external_environ
from cryo_savanna.images.cache import ThumbCache
from cryo_savanna.images.mrc import ImageRef, header
from cryo_savanna.results import registry as R
from cryo_savanna.results.classes import (
    data_star_for,
    iteration_maps,
    iterations,
    latest_iteration_maps,
    read_class_averages,
    read_model_classes,
    read_particles,
)
from cryo_savanna.results.curves import angular_distribution, convergence, model_fsc, postprocess_figures
from cryo_savanna.results.tables import read_compact
from cryo_savanna.ui import style
from cryo_savanna.viewers.angdist_view import AngdistPanel
from cryo_savanna.viewers.gallery import GalleryItem, GalleryView
from cryo_savanna.viewers.image_canvas import ImageCanvas
from cryo_savanna.viewers.micrograph_viewer import MicrographViewer
from cryo_savanna.viewers.plots import AnglesPlot, FigureGrid, TablePlot
from cryo_savanna.viewers.star_table import StarTableView
from cryo_savanna.viewers.tasks import TaskError, Tasks
from cryo_savanna.viewers.volume_viewer import VolumeViewer


def _stamp(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _dir_stamp(path: Path):
    """Changes when files are added to or removed from a directory."""
    return _stamp(path)


def _message(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    return lbl


# ---------------------------------------------------------------------------- classes

class ClassesView(QWidget):
    SORTS = [("distribution", "Class distribution"), ("resolution", "Estimated resolution"),
             ("particles", "Particles"), ("accuracy", "Accuracy of rotations"), ("score", "Class score")]

    def __init__(self, project_dir: Path, job: str = "", star: str = "", tool: str = "", request_tool=None,
                 parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.star = Path(project_dir), job, star
        self.tasks = Tasks(self)
        self.gallery = GalleryView(self.SORTS)
        self.tool_btn = None
        if tool == "classes" and request_tool is not None:
            self.tool_btn = QPushButton("Select classes…")
            self.tool_btn.setToolTip("Make an interactive Select job from these classes (as relion_display does)")
            self.tool_btn.clicked.connect(lambda: request_tool("classes", self.tool_source()))
            self.gallery.bar.addWidget(self.tool_btn)
        self.gallery.describe = self._describe
        self.iteration = QComboBox()
        self.iteration.setToolTip("Iteration")
        self.iteration.currentIndexChanged.connect(lambda _: self._load())
        self.iteration.setVisible(bool(job))
        self.gallery.bar.insertWidget(0, self.iteration)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.gallery)
        self._its: list = []
        self._stamp = None
        self.set_iterations()

    def tool_source(self) -> str:
        """The optimiser file of the iteration shown (what RELION's Select takes), else its model file."""
        if not self._its:
            return ""
        model = self._its[max(0, self.iteration.currentIndex())][1]
        opt = model.with_name(model.name.replace("_half1_model.star", "_optimiser.star")
                              .replace("_model.star", "_optimiser.star"))
        return str((opt if opt.is_file() else model).relative_to(self.project_dir))

    def set_iterations(self) -> None:
        if not self.job:
            self._load()
            return
        job_dir = self.project_dir / self.job
        self._stamp = _dir_stamp(job_dir)
        its = iterations(job_dir)
        at_last = self.iteration.currentIndex() in (-1, len(self._its) - 1)
        current = self.iteration.currentText()
        self._its = its
        self.iteration.blockSignals(True)
        self.iteration.clear()
        self.iteration.addItems([f"Iteration {it}" for it, _ in its])
        i = self.iteration.findText(current)
        self.iteration.setCurrentIndex(len(its) - 1 if at_last or i < 0 else i)
        self.iteration.blockSignals(False)
        self._load()

    def refresh(self) -> None:
        if self.job and _dir_stamp(self.project_dir / self.job) != self._stamp:
            self.set_iterations()
        elif self.star and _stamp(self.project_dir / self.star) != self._stamp:
            self._load()

    def _load(self) -> None:
        project = self.project_dir
        if self.job:
            if not self._its:
                return
            model = self._its[max(0, self.iteration.currentIndex())][1]
            rel = str(model.relative_to(project))
            fn = lambda: read_model_classes(project, rel, data_star_for(model))  # noqa: E731
        else:
            self._stamp = _stamp(project / self.star)
            star = self.star
            fn = lambda: read_class_averages(project, star)  # noqa: E731
        self.tasks.run(fn, self._on_classes, channel="classes")

    def _on_classes(self, cs) -> None:
        if isinstance(cs, TaskError):
            self.gallery.status.setText(f"⚠ {cs}")
            return
        self.classes = cs
        items = []
        for c in cs.classes:
            tip = [f"Class {c.number}"]
            if c.distribution is not None:
                tip.append(f"{100 * c.distribution:.2f}% of the particles")
            if c.particles is not None:
                tip.append(f"{c.particles} particles")
            if c.resolution is not None:
                tip.append(f"estimated resolution {c.resolution:.2f} Å")
            if c.accuracy_rot is not None:
                tip.append(f"accuracy {c.accuracy_rot:.2f}° / {c.accuracy_trans or 0:.2f} Å")
            if c.score is not None:
                tip.append(f"class score {c.score:.3f}")
            label = c.label() + (f"\n{c.particles} ptcls" if c.particles is not None else "")
            items.append(GalleryItem(c.ref, label, "\n".join(tip),
                                     {"distribution": c.distribution, "resolution": c.resolution,
                                      "particles": c.particles, "accuracy": c.accuracy_rot, "score": c.score},
                                     payload=c))
        self.gallery.set_items(items, keep_page=True)

    def _describe(self, sel: list[GalleryItem]) -> str:
        if not sel:
            return "3D classes: central sections" if getattr(self, "classes", None) and self.classes.dim == 3 else ""
        nums = sorted(i.payload.number for i in sel)
        parts = sum(i.payload.particles or 0 for i in sel)
        dist = sum(i.payload.distribution or 0 for i in sel)
        text = f"{len(sel)} selected: classes {', '.join(map(str, nums))}"
        if parts:
            text += f" · {parts} particles"
        elif dist:
            text += f" · {100 * dist:.1f}% of the particles"
        return text


# ---------------------------------------------------------------------------- particles

class ParticlesView(QWidget):
    SORTS = [("class", "Class"), ("fom", "Autopick FOM"), ("prob", "Max. probability"), ("defocus", "Defocus")]

    def __init__(self, project_dir: Path, star: str, parent=None):
        super().__init__(parent)
        self.project_dir, self.star = Path(project_dir), star
        self.tasks = Tasks(self)
        self.gallery = GalleryView(self.SORTS, paged=True)
        self.gallery.size.setValue(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.gallery)
        self._stamp = None
        self._load()

    def refresh(self) -> None:
        if _stamp(self.project_dir / self.star) != self._stamp:
            self._load()

    def _load(self) -> None:
        project, star = self.project_dir, self.star
        self._stamp = _stamp(project / star)
        self.gallery.status.setText("Reading…")

        def read():  # the items too: a million particles take seconds to list, not on the GUI thread
            ps = read_particles(project, star)
            return ps, particle_items(ps)
        self.tasks.run(read, self._on_particles, channel="particles")

    def _on_particles(self, result) -> None:
        if isinstance(result, TaskError):
            self.gallery.status.setText(f"⚠ {result}")
            return
        ps, items = result
        self.particles = ps
        self.gallery.set_items(items, keep_page=True)
        self.gallery.describe = lambda sel: (f"{len(sel)} selected" if sel else "") + \
            (f"   ·   {ps.box} px, {ps.pixel_size:g} Å/px" if ps.box and ps.pixel_size else "")
        self.gallery._update_status()


def particle_items(ps) -> list[GalleryItem]:
    """One gallery item per particle. Each stack's path is resolved once (particles share few stacks)."""
    t = ps.table
    cols = {k: t.numeric(c) if t.has(c) else None for k, c in (
        ("class", "rlnClassNumber"), ("fom", "rlnAutopickFigureOfMerit"),
        ("prob", "rlnMaxValueProbDistribution"), ("defocus", "rlnDefocusU"))}
    cols = {k: (v.tolist() if v is not None else None) for k, v in cols.items()}
    classes = cols["class"]
    paths: dict[str, str] = {}
    items = []
    for i, name in enumerate(t.column("rlnImageName")):
        ref = ImageRef.parse(name)
        path = paths.get(ref.path)
        if path is None:
            path = paths[ref.path] = ref.resolve(ps.project_dir).path
        keys = {k: (v[i] if v is not None else None) for k, v in cols.items()}
        label = str(i + 1) + (f" · c{int(classes[i])}" if classes is not None and classes[i] == classes[i] else "")
        items.append(GalleryItem(ImageRef(path, ref.index), label, "", keys, payload=i))
    return items


# ---------------------------------------------------------------------------- plots

class FiguresView(QWidget):
    def __init__(self, project_dir: Path, figures: str, job: str = "", star: str = "", parent=None):
        super().__init__(parent)
        self.project_dir, self.kind, self.job, self.star = Path(project_dir), figures, job, star
        self.tasks = Tasks(self)
        self.grid = FigureGrid()
        self.info = QLabel()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.info)
        self._stamp = None
        self._load()

    def _watch(self):
        if self.star:
            return _stamp(self.project_dir / self.star)
        return _dir_stamp(self.project_dir / self.job)

    def refresh(self) -> None:
        if self._watch() != self._stamp:
            self._load()

    def _load(self) -> None:
        self._stamp = self._watch()
        project, kind, job, star = self.project_dir, self.kind, self.job, self.star

        def read():
            if kind == "postprocess":
                return postprocess_figures(project / star), star
            if kind == "convergence":
                return convergence(project / job), job
            if kind == "model_fsc":
                job_dir = project / job
                final = job_dir / "run_model.star"
                its = iterations(job_dir)
                model = final if final.is_file() else (its[-1][1] if its else None)
                if model is None:
                    return [], ""
                fig = model_fsc(model)
                return ([fig] if fig else []), str(model.relative_to(project))
            raise ValueError(kind)

        self.tasks.run(read, self._on_figures, channel="figures")

    def _on_figures(self, result) -> None:
        if isinstance(result, TaskError):
            self.info.setText(f"⚠ {result}")
            return
        figs, source = result
        self.grid.set_figures(figs)
        self.info.setText(f"from {source}" if figs else "Nothing to plot yet.")


class TablePlotView(QWidget):
    def __init__(self, project_dir: Path, star: str, block: str, columns: list[str] | None = None,
                 scatter: tuple[str, str] | None = None, parent=None):
        super().__init__(parent)
        self.project_dir, self.star, self.block = Path(project_dir), star, block
        self.columns, self.scatter = columns, scatter
        self.tasks = Tasks(self)
        self.plot = TablePlot()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self._stamp = None
        self._load()

    def refresh(self) -> None:
        if _stamp(self.project_dir / self.star) != self._stamp:
            self._load()

    def _load(self) -> None:
        path, block = self.project_dir / self.star, self.block
        self._stamp = _stamp(path)
        self.tasks.run(lambda: read_compact(path, block), self._on_table, channel="table")  # big particle files

    def _on_table(self, t) -> None:
        if isinstance(t, TaskError) or t is None:
            self.plot.info.setText(f"⚠ {t}" if t is not None else "no such table")
            return
        self.plot.set_table(t, self.columns, self.scatter)


class AnglesView(QWidget):
    """The particles' viewing directions: a 2D map (tilt against rot), and in 3D the map's surface with
    RELION's .bild cylinders around it."""

    def __init__(self, project_dir: Path, job: str, parent=None):
        super().__init__(parent)
        self.project_dir, self.job = Path(project_dir), job
        self.tasks = Tasks(self)
        self.plot = AnglesPlot()
        self.panel3d = AngdistPanel(project_dir, job)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(self.panel3d)
        split.setSizes([500, 500])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self._stamp = None
        self._load()

    def _data_star(self) -> Path | None:
        d = self.project_dir / self.job
        if (d / "run_data.star").is_file():
            return d / "run_data.star"
        its = iterations(d)
        for _, model in reversed(its):
            ds = data_star_for(model)
            if ds.is_file():
                return ds
        return None

    def refresh(self) -> None:
        if _dir_stamp(self.project_dir / self.job) != self._stamp:
            self._load()
            self.panel3d.refresh()

    def _load(self) -> None:
        self._stamp = _dir_stamp(self.project_dir / self.job)
        ds = self._data_star()
        if ds is None:
            self.plot.info.setText("No particle orientations yet.")
            return
        self.tasks.run(lambda: (angular_distribution(ds), ds.name), self._on_hist, channel="angles")

    def _on_hist(self, result) -> None:
        if isinstance(result, TaskError) or result[0] is None:
            self.plot.info.setText(f"⚠ {result}" if isinstance(result, TaskError) else "No angles in the data file.")
            return
        (counts, n), source = result
        self.plot.set_histogram(counts, n, source)


# ---------------------------------------------------------------------------- files

TEXT_SUFFIXES = {".txt", ".out", ".err", ".log", ".script", ".py", ".sh", ".csh", ".json", ".dat", ".com", ".eps"}
IMAGE_SUFFIXES = {".mrc", ".mrcs", ".ctf", ".map", ".st", ".tif", ".tiff"}


def list_job_files(root: Path, limit: int) -> tuple[list[tuple[Path, list[tuple[str, int, float]], list[str]]], list[Path]]:
    """The files of a job directory, folder by folder (at most ``limit``): [(folder, [(name, size,
    mtime)], [subfolder names])], and every folder listed. Run off the GUI thread: a job with a file
    per micrograph, over a network file system, takes a while."""
    out, dirs, n = [], [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        if n >= limit:  # enough listed: do not walk the rest (one file per micrograph)
            break
        dirnames.sort()
        d = Path(dirpath)
        dirs.append(d)
        files = []
        for name in sorted(filenames):
            if n >= limit:
                break
            try:
                st = (d / name).stat()
            except OSError:
                continue
            files.append((name, st.st_size, st.st_mtime))
            n += 1
        out.append((d, files, list(dirnames)))
    return out, dirs


def _tree_stamp(dirs: list[Path]) -> tuple:
    """Changes when files are added to or removed from the job directory or a folder in it
    (a running job writes into Movies/, tilt_series/, ...): one stat per folder."""
    return tuple(_dir_stamp(d) for d in dirs)


class FilesView(QWidget):
    """The job's files; STAR files open as tables, MRC images and stacks as images, text as text.

    The list is read in the background, and again when a folder changes; the file shown stays as it
    is (a map keeps its level and view) while the list is rebuilt around it.
    """

    MAX_FILES = 3000

    def __init__(self, project_dir: Path, job: str, cache: ThumbCache | None = None, parent=None):
        super().__init__(parent)
        self.project_dir, self.job, self.cache = Path(project_dir), job, cache
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Size", "Modified"])
        self.tree.setRootIsDecorated(True)
        self.tree.currentItemChanged.connect(lambda cur, _: self._open(cur))
        self.stack = QStackedWidget()
        self.empty = _message("Select a file.")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.table: StarTableView | None = None
        self.image: ImageCanvas | None = None
        self.gallery: GalleryView | None = None
        self.other = QWidget()
        ov = QVBoxLayout(self.other)
        self.other_label = _message("")
        self.open_btn = QPushButton("Open with the system viewer")
        self.open_btn.clicked.connect(self._open_external)
        ov.addStretch(1)
        ov.addWidget(self.other_label)
        ov.addWidget(self.open_btn, 0, Qt.AlignCenter)
        ov.addStretch(1)
        for w in (self.empty, self.text, self.other):
            self.stack.addWidget(w)
        self.volume: VolumeViewer | None = None
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(self.stack)
        split.setSizes([300, 700])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        self._path: Path | None = None
        self._stamp = None
        self._dirs: list[Path] = []
        self.tasks = Tasks(self)
        self._listing = False
        self.fill()

    def fill(self, force: bool = True) -> None:
        """Read the list again (in the background); with force=False only when a folder changed."""
        if self._listing:
            return
        self._listing = True
        root, limit, dirs, old = self.project_dir / self.job, self.MAX_FILES, list(self._dirs), self._stamp

        def read():
            if not force and dirs and _tree_stamp(dirs) == old:
                return None  # nothing added or removed
            listing, found = list_job_files(root, limit)
            return listing, found, _tree_stamp(found)

        self.tasks.run(read, self._on_listing, channel="list")

    def _on_listing(self, result) -> None:
        self._listing = False
        if result is None:
            return
        if isinstance(result, TaskError):
            self.empty.setText(f"⚠ {result}")
            return
        listing, self._dirs, self._stamp = result
        current = str(self._path) if self._path is not None else None
        self.tree.blockSignals(True)  # rebuilding the list must not open the file shown again
        self.tree.clear()
        items: dict[Path, QTreeWidgetItem] = {}
        for d, files, dirnames in listing:
            parent = items.get(d)
            for name, size, mtime in files:
                p = d / name
                it = QTreeWidgetItem([name, _size(size), style.timestamp(mtime)])
                it.setData(0, Qt.UserRole, str(p))
                it.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                (parent.addChild(it) if parent else self.tree.addTopLevelItem(it))
                if str(p) == current:
                    self.tree.setCurrentItem(it)
            for name in dirnames:
                it = QTreeWidgetItem([name + "/", "", ""])
                (parent.addChild(it) if parent else self.tree.addTopLevelItem(it))
                items[d / name] = it
        self.tree.blockSignals(False)
        self.tree.resizeColumnToContents(0)

    def refresh(self) -> None:
        self.fill(force=False)
        if self.table is not None and self.stack.currentWidget() is self.table:
            self.table.refresh()
        if self.volume is not None and self.stack.currentWidget() is self.volume:
            self.volume.refresh()

    def select_file(self, name: str) -> bool:
        """Open a file of the job (relative to its directory), once a list being read is there."""
        if self._listing:
            self.tasks.wait()
        target = str(self.project_dir / self.job / name)
        for it in self.tree.findItems("*", Qt.MatchWildcard | Qt.MatchRecursive):
            if it.data(0, Qt.UserRole) == target:
                self.tree.setCurrentItem(it)
                return True
        return False

    def _open(self, item: QTreeWidgetItem | None) -> None:
        path = item.data(0, Qt.UserRole) if item is not None else None
        if not path:
            self.stack.setCurrentWidget(self.empty)
            return
        p = self._path = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".star":
            if self.table is None:
                self.table = StarTableView()
                self.stack.addWidget(self.table)
            self.table.set_file(p)
            self.stack.setCurrentWidget(self.table)
        elif suffix in IMAGE_SUFFIXES:
            self._open_image(p)
        elif suffix in TEXT_SUFFIXES or p.name in ("note.txt", "run.out", "run.err"):
            self.text.setPlainText(_tail(p))
            self.stack.setCurrentWidget(self.text)
        else:
            self.other_label.setText(f"{p.name}\n{_size(p.stat().st_size) if p.exists() else ''}")
            self.stack.setCurrentWidget(self.other)

    def _open_image(self, p: Path) -> None:
        try:
            h = header(p) if p.suffix.lower() not in (".tif", ".tiff") else None
        except OSError as e:
            self.other_label.setText(f"{p.name}\n{e}")
            self.stack.setCurrentWidget(self.other)
            return
        if h is not None and h.is_stack and h.nz > 1:
            if self.gallery is None:
                self.gallery = GalleryView(paged=True)
                self.stack.addWidget(self.gallery)
            self.gallery.set_items([GalleryItem(ImageRef(str(p), i), str(i + 1)) for i in range(h.nz)])
            self.stack.setCurrentWidget(self.gallery)
            return
        if h is not None and h.nz > 1:  # a volume
            if self.volume is not None:
                self.stack.removeWidget(self.volume)
                self.volume.deleteLater()
            self.volume = VolumeViewer(self.project_dir, [os.path.relpath(p, self.project_dir)])
            self.stack.addWidget(self.volume)
            self.stack.setCurrentWidget(self.volume)
            return
        if self.image is None:
            self.image = ImageCanvas(self.cache)
            self.stack.addWidget(self.image)
        self.stack.setCurrentWidget(self.image)
        self.image.set_image(ImageRef(str(p)))

    def _open_external(self) -> None:
        if self._path is None:
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, str(self._path)], start_new_session=True, env=external_environ(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            self.other_label.setText(f"{self._path.name}\n{opener}: {e}")


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _tail(p: Path, limit: int = 512 * 1024) -> str:
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            data = fh.read()
    except OSError as e:
        return str(e)
    text = data.decode("utf-8", errors="replace")
    return (f"[… first {size - limit} bytes not shown …]\n" if size > limit else "") + text


# ---------------------------------------------------------------------------- factory

def build_view(spec: R.ViewSpec, ctx: R.JobContext, cache: ThumbCache | None, request_tool=None) -> QWidget:
    """The widget for a view; ``request_tool(kind, arg)`` opens a selection tool (JobActions.request_tool)."""
    a, project = spec.args, ctx.project_dir
    if spec.kind == R.MICROGRAPHS:
        return MicrographViewer(project, a.get("star", ""), cache, ctf=a.get("ctf", False),
                                motion=a.get("motion", False), picks=a.get("picks", ""),
                                diameter=a.get("diameter"), box=a.get("box", ""))
    if spec.kind == R.CLASSES:
        return ClassesView(project, job=a.get("job", ""), star=a.get("star", ""), tool=a.get("tool", ""),
                           request_tool=request_tool)
    if spec.kind == R.PARTICLES:
        return ParticlesView(project, a["star"])
    if spec.kind == R.PLOTS:
        return FiguresView(project, a["figures"], job=a.get("job", ""), star=a.get("star", ""))
    if spec.kind == R.TABLE_PLOT:
        return TablePlotView(project, a["star"], a.get("block", ""), a.get("columns"), a.get("scatter"))
    if spec.kind == R.ANGLES:
        return AnglesView(project, a["job"])
    if spec.kind == R.VOLUME:
        follow = (lambda job=a["job"]: latest_iteration_maps(project, job)) if a.get("latest") else None
        history = (lambda job=a["job"]: iteration_maps(project, job)) if a.get("history") else None
        return VolumeViewer(project, a.get("maps", []), mask=a.get("mask", ""), color=a.get("color", ""),
                            masks=a.get("masks", []), follow=follow, history=history)
    if spec.kind == R.TILT_SERIES:
        from cryo_savanna.viewers.tomo_viewer import TiltSeriesViewer
        return TiltSeriesViewer(project, a["star"], cache, tool=a.get("tool", ""), tool_arg=a.get("tool_arg", ""),
                                request_tool=request_tool)
    if spec.kind == R.TOMOGRAMS:
        from cryo_savanna.viewers.tomo_viewer import TomogramViewer
        return TomogramViewer(project, a["star"], picks=a.get("picks", ""), diameter=a.get("diameter"), cache=cache)
    if spec.kind == R.SELECTION:
        from cryo_savanna.viewers.selectors import SelectionView
        return SelectionView(project, a["job"], a["what"], request_tool=request_tool)
    if spec.kind in (R.CTF_REFINE, R.CTF_CHANGES, R.ABERRATIONS):
        from cryo_savanna.viewers import ctfrefine_view as V
        if spec.kind == R.CTF_REFINE:
            return V.CtfRefineSummary(project, a["job"], a.get("star", ""), a.get("input", ""), ctx.options)
        if spec.kind == R.CTF_CHANGES:
            return V.CtfChangesView(project, a["star"], a.get("input", ""))
        return V.AberrationsView(project, a["job"], a.get("star", ""))
    if spec.kind in (R.PYTOM_SCORES, R.PYTOM_THRESHOLDS, R.SVG_PLOTS):
        from cryo_savanna.viewers import pytom_view as V
        if spec.kind == R.PYTOM_SCORES:
            return V.PytomScoresView(project, a["job"])
        if spec.kind == R.PYTOM_THRESHOLDS:
            return V.PytomThresholdView(project, a["job"], a.get("summary", ""), a.get("diameter"), cache)
        return V.SvgPlotsView(project, a["job"], a.get("folder", "plots"))
    if spec.kind == R.FILES:
        return FilesView(project, a["job"], cache)
    return _message(f"No viewer for {spec.kind}")
