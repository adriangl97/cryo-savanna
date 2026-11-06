# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Result viewers in the window: tabs per job, micrographs with CTF/motion/picks, galleries, plots, files."""

import os
import shutil
import time

import mrcfile
import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from qtpy.QtCore import QItemSelectionModel, QSettings, Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from cryo_savanna.env.relion_env import RelionEnv  # noqa: E402
from cryo_savanna.images.mrc import ImageRef  # noqa: E402
from cryo_savanna.jobdefs.flags import FlagSource  # noqa: E402
from cryo_savanna.project.slurm import SlurmClient  # noqa: E402
from cryo_savanna.results.tables import Table  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402
from cryo_savanna.viewers.gallery import GalleryView  # noqa: E402
from cryo_savanna.viewers.image_canvas import ImageCanvas  # noqa: E402
from cryo_savanna.viewers.micrograph_viewer import MicrographViewer  # noqa: E402
from cryo_savanna.viewers.plots import TablePlot  # noqa: E402
from cryo_savanna.viewers.star_table import StarTableView  # noqa: E402
from cryo_savanna.viewers.tasks import pool  # noqa: E402
from cryo_savanna.viewers.views import ClassesView, FilesView, FiguresView, ParticlesView  # noqa: E402


def settle(qtbot, rounds: int = 6) -> None:
    """Let the viewers' background reads finish and their results arrive (they may start more)."""
    for _ in range(rounds):
        pool().waitForDone(10000)
        QApplication.processEvents()
        qtbot.wait(15)


@pytest.fixture
def window(qtbot, tmp_path):
    def make(project):
        w = MainWindow(project, poll_s=1000, env=RelionEnv({}), flag_source=FlagSource(which=lambda p: None),
                       settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat),
                       slurm=SlurmClient(which=lambda p: None))
        qtbot.addWidget(w)
        w.resize(1400, 900)
        w.show()
        qtbot.waitExposed(w)
        w.load_now()
        return w
    return make


def open_view(w, qtbot, job, title):
    w.select_job(job)
    v = w.info.show_view(title)
    assert v is not None, w.info.view_titles()
    settle(qtbot)
    return v


def test_tabs_follow_the_job(window, spa_project, qtbot):
    w = window(spa_project)
    w.select_job("CtfFind/job003/")
    assert w.info.view_titles()[:2] == ["Micrographs", "CTF plots"]
    v = w.info.show_view("Micrographs")
    w.select_job("MotionCorr/job002/")  # the next job also has "Micrographs": stay on that tab
    assert w.info.tabText(w.info.currentIndex()) == "Micrographs"
    assert w.info.current_view() is not v
    w.select_job("Polish/job015/")
    assert w.info.view_titles() == ["Files"] and w.info.currentIndex() == 0
    w.info.refresh_viewers()  # nothing shown: nothing to do


def test_micrographs_with_ctf_spectrum_and_table(window, spa_project, qtbot):
    w = window(spa_project)
    v: MicrographViewer = open_view(w, qtbot, "CtfFind/job003/", "Micrographs")
    assert v.model.rowCount() == 24 and v.count.text() == "24 micrographs"
    assert v.canvas.preview is not None and v.canvas.full_shape == (128, 128)
    assert v.ctf_canvas.isVisible() and v.ctf_canvas.full_shape == (64, 64)
    # sort by defocus and step to another micrograph
    col = [v.model.headerData(c, Qt.Horizontal) for c in range(v.model.columnCount())].index("Defocus U (Å)")
    v.table.sortByColumn(col, Qt.DescendingOrder)
    v.table.setCurrentIndex(v.proxy.index(0, 0))
    settle(qtbot)
    top = v.proxy.mapToSource(v.proxy.index(0, 0)).row()
    assert v.current == top
    d = v.mics.table.numeric("rlnDefocusU")
    assert d[top] == d.max()
    assert v.mics.names[top].rsplit("/", 1)[-1] in v.canvas.info.text()


def test_clicking_a_ctf_plot_point_shows_its_micrograph(window, spa_project, qtbot):
    w = window(spa_project)
    plots = open_view(w, qtbot, "CtfFind/job003/", "CTF plots")
    plots.plot.mode.setCurrentIndex(1)  # scatter: defocus against max. resolution
    pt = plots.plot.scatter.points()[7]
    plots.plot._clicked(plots.plot.scatter, [pt])
    settle(qtbot)
    v = w.info.current_view()
    assert isinstance(v, MicrographViewer) and v.current == int(pt.data())


def test_motion_track(window, spa_project, qtbot):
    w = window(spa_project)
    v = open_view(w, qtbot, "MotionCorr/job002/", "Micrographs")
    assert v.motion_plot.isVisible() and len(v.motion_plot.plots) == 1
    assert "24 frames" in v.motion_plot.figures[0].notes[0]


def test_picks_with_fom_threshold(window, spa_project, qtbot):
    w = window(spa_project)
    v = open_view(w, qtbot, "AutoPick/job004/", "Picks")
    assert v.pick_bar.isVisible() and v._picks is not None
    n = len(v._picks)
    assert len(v.canvas.points.data) == n and v.pick_info.text() == f"{n} picks"
    assert v.circle.value() == pytest.approx(180 / 0.885, abs=1)  # LoG max. diameter in pixels
    assert v.model.columnCount() > 2 and v.model.headerData(v.model.columnCount() - 1, Qt.Horizontal) == "Picks"
    v.min_fom.setValue(0.5)
    shown = int((v._picks.fom >= 0.5).sum())
    assert len(v.canvas.points.data) == shown and "below the FOM threshold" in v.pick_info.text()
    v.show_picks.setChecked(False)
    assert len(v.canvas.points.data) == 0


def test_extracted_particles_on_their_micrographs(window, spa_project, qtbot):
    w = window(spa_project)
    v = open_view(w, qtbot, "Extract/job005/", "On micrographs")
    assert len(v._picks) == 40 and v.circle.value() == 256  # the box size
    p: ParticlesView = w.info.show_view("Particles")
    settle(qtbot)
    assert p.gallery.model.rowCount() == 24 * 40 and p.gallery.model.loaded() > 0


def test_picks_of_chosen_classes(window, spa_project, qtbot):
    from cryo_savanna.viewers.class_filter import PICK_COLOR, class_color
    w = window(spa_project)
    v = open_view(w, qtbot, "Class3D/job009/", "Picks")
    f, p = v.classes, v._picks
    assert f.isVisible() and f.list.count() == len(v.pickset.classes) > 1 and f.text() == "Classes: all"
    assert len(v.canvas.points.data) == len(p) == 25
    pens = {tuple(pt.pen().color().getRgb()[:3]) for pt in v.canvas.points.points()}
    assert pens == {class_color(n) for n in set(p.cls.tolist())}  # coloured by class
    f.set_selected({1, 3})
    shown = int(np.isin(p.cls, [1, 3]).sum())
    assert len(v.canvas.points.data) == shown and f.text() == "Classes: 1, 3"
    assert v.pick_info.text().startswith(f"{shown} picks (classes 1, 3")
    settle(qtbot)
    col = v.model.columnCount() - 1
    assert v.model.headerData(col, Qt.Horizontal) == "Picks"
    assert v.model.data(v.model.index(v.current, col)) in (shown, str(shown))
    assert "(classes 1, 3)" in v.count.text()
    f.color_by_class.setChecked(False)
    assert {tuple(pt.pen().color().getRgb()[:3]) for pt in v.canvas.points.points()} <= {PICK_COLOR}
    f._invert()
    assert len(v.canvas.points.data) == len(p) - shown
    f._tick_all(False)
    assert len(v.canvas.points.data) == 0 and f.text() == "Classes: none"
    v.select(v.current + 1)  # the choice stays for the next micrograph
    settle(qtbot)
    assert f.selected() == set() and len(v.canvas.points.data) == 0
    f._tick_all(True)
    assert f.selected() is None and len(v.canvas.points.data) == len(v._picks)


def test_no_class_choice_for_picks_of_one_class(window, spa_project, qtbot):
    w = window(spa_project)
    v = open_view(w, qtbot, "AutoPick/job004/", "Picks")
    assert not v.classes.isVisible() and v.classes.selected() is None


def test_class_gallery_sort_select_and_iterations(window, spa_project, qtbot):
    w = window(spa_project)
    v: ClassesView = open_view(w, qtbot, "Class2D/job006/", "Classes")
    g = v.gallery
    assert v.iteration.currentText() == "Iteration 200" and g.model.rowCount() == 50
    assert g.model.loaded() > 0  # thumbnails of the classes on screen
    g.sort.setCurrentIndex(1)  # class distribution
    g.descending.setChecked(True)
    dist = [it.payload.distribution for it in g.model.items]
    assert dist == sorted(dist, reverse=True)
    g.list.selectionModel().select(g.model.index(0), QItemSelectionModel.Select)
    g.list.selectionModel().select(g.model.index(1), QItemSelectionModel.Select)
    assert "2 selected" in g.status.text() and "particles" in g.status.text()
    v.iteration.setCurrentIndex(0)  # an older iteration (no data.star: no particle counts)
    settle(qtbot)
    assert g.model.items[0].payload.particles is None


def test_new_iteration_shows_up_while_running(window, spa_copy, qtbot):
    w = window(spa_copy)
    v = open_view(w, qtbot, "Class2D/job006/", "Classes")
    job = spa_copy / "Class2D/job006"
    text = (job / "run_it200_model.star").read_text().replace("run_it200_classes", "run_it201_classes")
    shutil.copy(job / "run_it200_classes.mrcs", job / "run_it201_classes.mrcs")
    (job / "run_it201_model.star").write_text(text)
    w.info.refresh_viewers()
    settle(qtbot)
    assert v.iteration.currentText() == "Iteration 201"
    assert "run_it201" in v.gallery.model.items[0].ref.path


def test_3d_classes_and_plots(window, spa_project, qtbot):
    w = window(spa_project)
    c = open_view(w, qtbot, "Class3D/job009/", "Classes")
    assert c.gallery.model.rowCount() == 4 and c.classes.dim == 3
    conv: FiguresView = w.info.show_view("Convergence")
    settle(qtbot)
    assert [f.key for f in conv.grid.figures] == ["resolution", "occupancy", "accuracy", "changes"]
    ang = w.info.show_view("Angular distribution")
    settle(qtbot)
    assert "particles from run_it025_data.star" in ang.plot.info.text()
    fsc = open_view(w, qtbot, "PostProcess/job013/", "FSC")
    assert [f.key for f in fsc.grid.figures] == ["fsc", "guinier"] and len(fsc.grid.plots) == 2
    ref = open_view(w, qtbot, "Refine3D/job011/", "FSC")
    assert ref.grid.figures[0].curves[0].label == "Class 1"
    assert "run_it015_half1_model.star" in ref.info.text()


def test_table_plots_and_star_tables(qtbot, spa_project):
    t = Table.read(spa_project / "CtfFind/job003/micrographs_ctf.star", "micrographs")
    tp = TablePlot()
    qtbot.addWidget(tp)
    tp.set_table(t, ["rlnDefocusU", "rlnCtfMaxResolution"], scatter=("rlnDefocusU", "rlnCtfMaxResolution"))
    assert len(tp.grid.ci.items) == 2  # two histograms
    tp.mode.setCurrentIndex(1)
    assert tp.scatter is not None and len(tp.scatter.data) == 24
    rows = []
    tp.row_clicked.connect(rows.append)
    tp._clicked(tp.scatter, [tp.scatter.points()[5]])
    assert rows == [int(tp.scatter.points()[5].data())]

    st = StarTableView()
    qtbot.addWidget(st)
    st.set_file(spa_project / "Extract/job005/particles.star")
    st.tasks.wait()
    assert st.block.currentText() == "particles" and st.model.rowCount() == 960
    st.filter.setText("00021")
    assert 0 < st.proxy.rowCount() < 960
    st.plot_btn.setChecked(True)
    assert st.plot.table is st.model.table


def test_files_tab(window, spa_project, qtbot):
    w = window(spa_project)
    f: FilesView = open_view(w, qtbot, "Refine3D/job011/", "Files")
    assert f.select_file("run_data.star")
    settle(qtbot)
    assert f.stack.currentWidget() is f.table and f.table.model.rowCount() == 24 * 12
    assert f.select_file("run_class001.mrc")  # a volume: the volume viewer
    settle(qtbot)
    assert f.stack.currentWidget() is f.volume and f.volume.vol.shape == (32, 32, 32)
    assert f.select_file("run.out")
    assert "Running Refine3D/job011/" in f.text.toPlainText()
    f2 = open_view(w, qtbot, "Extract/job005/", "Files")
    assert f2.select_file("Movies/20170629_00021_frameImage.mrcs")
    settle(qtbot)
    assert isinstance(f2.stack.currentWidget(), GalleryView) and f2.gallery.model.rowCount() == 40


def test_files_tab_shows_files_a_running_job_adds_in_its_folders(window, spa_copy, qtbot):
    w = window(spa_copy)
    f: FilesView = open_view(w, qtbot, "Extract/job005/", "Files")
    (spa_copy / "Extract/job005/Movies/new_particles.star").write_text("data_\n")
    f.refresh()
    assert f.select_file("Movies/new_particles.star")


def test_pop_out_window(window, spa_project, qtbot):
    w = window(spa_project)
    open_view(w, qtbot, "PostProcess/job013/", "FSC")
    w.info.pop_out()
    settle(qtbot)
    assert len(w.info.windows) == 1
    win = w.info.windows[0]
    assert "FSC" in win.windowTitle() and len(win.view.grid.plots) == 2
    w.info.refresh_viewers()
    win.close()
    QApplication.processEvents()
    qtbot.wait(20)
    QApplication.processEvents()
    assert w.info.windows == []


def test_zooming_in_reads_the_region_at_full_detail(qtbot, tmp_path):
    y, x = np.mgrid[:2048, :2048]
    img = np.sin(x / 3.0).astype(np.float32) + (y > 1024)
    with mrcfile.new(tmp_path / "big.mrc") as m:
        m.set_data(img)
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.resize(400, 400)
    c.show()
    qtbot.waitExposed(c)
    c.set_image(ImageRef(str(tmp_path / "big.mrc")))
    settle(qtbot)
    assert c.full_shape == (2048, 2048) and c.factor > 1 and not c.detail.isVisible()
    from qtpy.QtCore import QRectF
    c.view.setRange(QRectF(1000, 1000, 100, 100), padding=0)
    qtbot.wait(300)  # the debounce
    settle(qtbot)
    assert c.detail.isVisible()
    r = c.detail.mapRectToParent(c.detail.boundingRect())
    assert r.left() <= 1000 and r.right() >= 1100 and c.detail.image.shape[0] <= 2048
    c.view.setRange(QRectF(0, 0, 2048, 2048), padding=0)  # zoomed out again: the preview is enough
    qtbot.wait(300)
    settle(qtbot)
    assert not c.detail.isVisible()


def test_old_picks_and_images_are_repainted_away(qtbot, tmp_path):
    """Changing the circles (or losing the image) repaints where they were: with no image under them
    (a micrograph that does not load), old circles stayed on screen."""
    from qtpy.QtCore import QRectF
    c = ImageCanvas(controls=False)
    qtbot.addWidget(c)
    c.resize(400, 400)
    c.show()
    qtbot.waitExposed(c)
    c.view.setRange(QRectF(0, 0, 1000, 1000), padding=0)
    c.view.disableAutoRange()
    QApplication.processEvents()

    def repaints_old_area(item, change) -> bool:
        old = item.mapRectToScene(item.boundingRect()).center()
        dirty = []
        scene = c.gv.scene()
        scene.changed.connect(dirty.extend)
        change()
        QApplication.processEvents()
        scene.changed.disconnect(dirty.extend)
        return any(r.contains(old) for r in dirty)

    def old_picks():
        c.set_points(np.array([100.0, 150]), np.array([100.0, 150]), 50)
        QApplication.processEvents()

    old_picks()
    assert repaints_old_area(c.points, lambda: c.set_points([], [], 1))
    old_picks()
    assert repaints_old_area(c.points, lambda: c.set_points(np.array([800.0]), np.array([800.0]), 50))
    c.set_array(np.zeros((200, 200), np.float32))
    QApplication.processEvents()
    assert repaints_old_area(c.item, lambda: c.set_image(None))
    c.set_array(np.zeros((200, 200), np.float32))
    QApplication.processEvents()
    assert repaints_old_area(c.item, lambda: c.set_image(ImageRef(str(tmp_path / "missing.mrc"))) or settle(qtbot))
    assert c.error and c.item.image is None


def test_particle_items_resolve_each_stack_once(tmp_path):
    from cryo_savanna.results.classes import ParticleSet
    from cryo_savanna.results.tables import Table
    from cryo_savanna.viewers.views import particle_items
    t = Table("particles", ["rlnImageName", "rlnClassNumber"],
              {"rlnImageName": ["000001@Extract/job007/a.mrcs", "000002@Extract/job007/a.mrcs", "/abs/b.mrcs"],
               "rlnClassNumber": ["3", "nan", "1"]})
    items = particle_items(ParticleSet(tmp_path, "x.star", t))
    assert [str(i.ref) for i in items] == [f"000001@{tmp_path}/Extract/job007/a.mrcs",
                                           f"000002@{tmp_path}/Extract/job007/a.mrcs", "/abs/b.mrcs"]
    assert [i.label for i in items] == ["1 · c3", "2", "3 · c1"] and items[0].keys["class"] == 3.0


def test_star_table_sorts_and_filters_whole_columns(qtbot):
    """Sorted and filtered with numpy (a big particle file in a fraction of a second): numbers as numbers,
    no value first ascending and last descending, the current row kept, inf and nan shown as they are."""
    t = Table("particles", ["rlnImageName", "rlnDefocusU"],
              {"rlnImageName": [f"{i:06d}@Extract/job005/mic{i % 3}.mrcs" for i in range(1, 7)],
               "rlnDefocusU": ["12000", "9000", "nan", "15000", "inf", "800"]})
    st = StarTableView()
    qtbot.addWidget(st)
    st._on_table(t)
    st.view.selectRow(1)  # 9000

    def shown(col=2):
        return [st.proxy.index(r, col).data() for r in range(st.proxy.rowCount())]
    st.view.sortByColumn(2, Qt.AscendingOrder)
    assert shown() == ["nan", "inf", "800", "9000", "12000", "15000"]  # inf is not a number to sort by
    assert st.proxy.mapToSource(st.view.currentIndex()).row() == 1  # still the same particle
    st.view.sortByColumn(2, Qt.DescendingOrder)
    assert shown()[:4] == ["15000", "12000", "9000", "800"]
    st.filter.setText("MIC1.")  # case-insensitive, any column
    assert sorted(shown(1)) == ["000001@Extract/job005/mic1.mrcs", "000004@Extract/job005/mic1.mrcs"]
    assert st.view.currentIndex().isValid() is False  # the current row is hidden by the filter
    st.filter.setText("")
    assert st.proxy.rowCount() == 6


def test_files_tab_keeps_the_file_shown_when_files_are_added(window, spa_copy, qtbot):
    """A running job writes new files: the list follows, and the map shown keeps its viewer (level, view)."""
    w = window(spa_copy)
    f: FilesView = open_view(w, qtbot, "Refine3D/job011/", "Files")
    assert f.select_file("run_class001.mrc")
    settle(qtbot)
    vol = f.volume
    vol.level.setValue(0.123)
    job = spa_copy / "Refine3D/job011"
    (job / "run_it016_half1_model.star").write_text("data_\n")
    os.utime(job, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    f.refresh()
    f.tasks.wait()
    assert f.volume is vol and f.stack.currentWidget() is vol and vol.level.value() == pytest.approx(0.123)
    assert f.select_file("run_it016_half1_model.star")
