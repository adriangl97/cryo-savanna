# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""What the window paints again, and so sends over X11, when little changes (offscreen: conftest.py).

Over X11 every repainted pixel is sent to the display, so a refresh that changes one line must paint that
line, not the panel it is in: the job overview (ui/richtext.py), the logs (ui/log_panel.py), the list's
header (ui/job_tree.py) and the busy note (ui/busy.py).
"""

import dataclasses
import random
import time

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QEvent, QObject, QSettings  # noqa: E402
from qtpy.QtWidgets import QApplication, QTextBrowser  # noqa: E402

from cryo_savanna.ui import richtext  # noqa: E402
from cryo_savanna.ui.busy import run_busy  # noqa: E402
from cryo_savanna.ui.log_panel import LogPanel, LogView, collapse_progress  # noqa: E402
from cryo_savanna.ui.main_window import MainWindow  # noqa: E402


class Painted(QObject):
    """Pixels painted (paint events' regions) of a widget, or of every widget with ``widget=None``."""

    def __init__(self, widget=None):
        super().__init__()
        self.widget, self.px, self.events = widget, 0, 0
        (widget or QApplication.instance()).installEventFilter(self)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Paint and (self.widget is None or obj is self.widget):
            self.px += sum(r.width() * r.height() for r in e.region())
            self.events += 1
        return False

    def reset(self):
        self.px = self.events = 0


def settle(qtbot, ms=60):
    qtbot.wait(ms)
    QApplication.processEvents()


# ---------------------------------------------------------------------------- the overview

def _page(last_change, progress, extra=""):
    return ("<h3>Class2D/job005</h3><table cellspacing=0 cellpadding=2>"
            "<tr><td style='color:gray'>Status</td><td><b style='color:#1f7ad0'>Running</b></td></tr>"
            f"<tr><td style='color:gray'>Progress</td><td>{progress}</td></tr>"
            f"<tr><td style='color:gray'>Last change</td><td>{last_change}</td></tr></table>"
            "<h4>Inputs (1)</h4><table><tr><td>Extract/job004/particles.star</td>"
            "<td>from <a href='job:Extract/job004/'>Extract/job004</a> <b>(main input)</b></td></tr></table>" + extra)


def test_set_html_changes_only_what_differs(qtbot):
    b = QTextBrowser()
    qtbot.addWidget(b)
    b.resize(600, 400)
    b.show()
    pages = [_page("13:00:01", "iteration 3/25"), _page("13:00:03", "iteration 3/25"),
             _page("13:00:05", "iteration <b>4</b>/25 · 2 min left"), _page("13:00:05", ""),
             _page("x", "y", "<p>more: <a href='job:A/'>A</a></p>"), _page("x", "z", "<p>more: <a href='job:B/'>B</a></p>"),
             "<p>Select a job.</p>", _page("a", "b"), _page("a", "c" * 300)]
    for html in pages:
        richtext.set_html(b, html)
        fresh = richtext._scratch(b.document(), html)
        assert richtext.changed_blocks(b.document(), fresh) == []
        assert b.document().toHtml() == fresh.toHtml()  # as setHtml would have made it


def test_set_html_paints_only_the_changed_rows(qtbot):
    b = QTextBrowser()
    qtbot.addWidget(b)
    b.resize(600, 400)
    b.show()
    richtext.set_html(b, _page("13:00:01 (just now)", "iteration 3/25"))
    settle(qtbot)
    painted = Painted(b.viewport())
    richtext.set_html(b, _page("13:00:01 (just now)", "iteration 3/25"))  # the same again: nothing
    settle(qtbot)
    assert painted.px == 0
    richtext.set_html(b, _page("13:00:03 (just now)", "iteration 3/25"))
    settle(qtbot)
    full = b.viewport().width() * b.viewport().height()
    assert 0 < painted.px < full / 2


def test_selected_running_job_is_not_painted_whole_at_every_poll(qtbot, tmp_path, spa_copy):
    w = MainWindow(spa_copy, poll_s=1000, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.resize(1400, 850)
    w.show()
    w.load_now()
    w.select_job("Polish/job015/")  # running
    settle(qtbot, 200)
    overview, log = Painted(w.info.overview.viewport()), Painted(w.logs.views["run.out"].viewport())
    for i in range(3):
        with open(spa_copy / "Polish/job015/run.out", "a") as f:
            f.write(f"progress {i}\n")
        snap = w.loader.refresh()
        if snap is not None:
            w._on_loaded(snap)
        w.logs.refresh()
        settle(qtbot, 150)
    for p, widget in ((overview, w.info.overview), (log, w.logs.views["run.out"])):
        vp = widget.viewport()
        assert p.px < 3 * vp.width() * vp.height() / 2, (widget, p.px)


# ---------------------------------------------------------------------------- the logs

def test_log_view_paints_only_the_lines_added(qtbot, tmp_path):
    log = tmp_path / "run.out"
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    v = LogView()
    qtbot.addWidget(v)
    v.resize(600, 300)
    v.show()
    v.load(log)
    settle(qtbot)
    painted = Painted(v.viewport())
    full = v.viewport().width() * v.viewport().height()
    for chunk in ("one line\n", "two\nlines\n", "  0.10/1.35 min ~~(,_,\">   ", "[oo]", "\r  0.20/1.35 min [oo]",
                  "\rdone", "\n", "\n\nblank lines before\n"):
        painted.reset()
        with open(log, "a") as f:
            f.write(chunk)
        assert v.refresh()
        settle(qtbot)
        assert painted.px < full / 3, (chunk, painted.px, full)
    assert v.toPlainText() == collapse_progress(log.read_bytes().decode())


def test_log_view_text_is_the_file_as_a_terminal_shows_it(qtbot, tmp_path):
    log = tmp_path / "run.out"
    log.write_text("")
    v = LogView()
    qtbot.addWidget(v)
    v.load(log)
    rnd = random.Random(7)
    pieces = ["a", "bc", "\n", "\r", "  ", "é", "█", "😀", "\n\n", "\r\n"]
    for i in range(600):
        with open(log, "a", encoding="utf-8", newline="") as f:
            f.write("".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 12))))
        v.refresh()
        assert v.toPlainText() == collapse_progress(log.read_bytes().decode("utf-8")), i


def test_log_tab_text_set_only_when_it_changes(qtbot, tmp_path):
    job = tmp_path / "Class2D/job001"
    job.mkdir(parents=True)
    (job / "run.out").write_text("start\n")
    (job / "run.err").write_text("")
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.show_job(job)
    settle(qtbot)
    bar = Painted(panel.tabBar())
    with open(job / "run.out", "a") as f:
        f.write("more\n")
    panel.refresh()
    settle(qtbot)
    assert bar.px == 0  # the tab bar is not painted again for the same "run.err"
    (job / "run.err").write_text("an error\n")
    panel.refresh()
    assert panel.tabText(1) == "run.err ⚠"


# ---------------------------------------------------------------------------- the list's header

def test_list_header_not_painted_when_no_fitted_text_changes(qtbot, tmp_path, spa_copy):
    w = MainWindow(spa_copy, poll_s=1000, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.resize(1400, 850)
    w.show()
    w.load_now()
    settle(qtbot, 200)
    header = Painted(w.tree.header().viewport())
    snap = w.snapshot
    job = "Import/job001/"  # finished: no run time in its status, which would change with the clock
    for i in range(3):  # as at a poll: a job's last change moved, nothing the header fits
        jobs = dict(snap.jobs)
        jobs[job] = dataclasses.replace(jobs[job], modified=(jobs[job].modified or 0.0) + i + 1)
        w.tree.set_snapshot(dataclasses.replace(snap, jobs=jobs))
        settle(qtbot)
    assert header.px == 0


# ---------------------------------------------------------------------------- the busy note

def test_busy_note_is_not_animated(qtbot):
    painted = Painted()
    t = time.monotonic()
    assert run_busy(lambda: time.sleep(1.5) or 5, "Working…") == 5
    took = time.monotonic() - t
    # shown once, then its time written once a second (a busy bar: about 100 frames a second)
    assert painted.events < 15 * took, painted.events


# ---------------------------------------------------------------------------- the status bar

def test_status_bar_paints_only_the_time_at_a_poll(qtbot, tmp_path, spa_copy):
    w = MainWindow(spa_copy, poll_s=1000, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.resize(1400, 850)
    w.show()
    w.load_now()
    settle(qtbot, 200)
    before = w.status_state.text()
    bar = Painted(w.statusBar())
    project = Painted(w.status_project)
    w.snapshot.loaded_at -= 3600  # read an hour earlier: another time
    w._update_status()
    settle(qtbot)
    assert w.status_state.text() != before and "updated" in w.status_state.text()
    time_label = w.status_state.time
    assert project.px == 0  # only the time's own label (and the bar under it)
    assert 0 < bar.px <= time_label.width() * time_label.height()
