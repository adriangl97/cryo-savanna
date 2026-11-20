# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""The New job menu: a search field, the job types started last ("Recent"), RELION's jobs, the programs that are
not RELION's ("Other programs": pytom, AreTomo3, ...) and RELION's command-line tools.

Typing in the search field leaves only the entries whose name has the text (command-line tools too, listed
under their own heading); Enter starts the first one. The entries are kept as (id, text, what to call); the ids
("job:<variant>", "pytom:<kind>", "cli:<program>", ...) are what Recent remembers, in the settings when the window
gives them (``settings``), so they are the same in every project.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QAction, QApplication, QLineEdit, QMenu, QWidgetAction

RECENT_KEY = "recent_new_jobs"
RECENT_MAX = 5
CLI_MATCHES_MAX = 10  # command-line tools listed for a search (there are many)


@dataclass
class Entry:
    id: str  # what Recent keeps
    text: str
    start: Callable[[], None]
    tip: str = ""


class NewJobMenu(QMenu):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setToolTipsVisible(True)
        self.settings = None  # QSettings of the window: Recent is kept there (else for this session only)
        self._recent: list[str] = []
        self.jobs: list[Entry] = []  # RELION's jobs (SPA or tomography)
        self.others: list[Entry] = []  # programs that are not RELION's
        self.cli: list[Entry] = []  # RELION's command-line tools: only searched, and in their submenu
        self.cli_menu: QMenu | None = None
        self._entry_actions: list[tuple[QAction, Entry]] = []  # those the search filters
        self._search_found: list[QAction] = []  # what the search added (command-line tools, "No match")
        self._recent_actions: list[QAction] = []
        self._job_header: QAction | None = None
        self._others_header: QAction | None = None
        self._cli_action: QAction | None = None

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search job types…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._start_first)
        self._search_action = QWidgetAction(None)  # not the menu's: clear() would delete it
        self._search_action.setDefaultWidget(self.search)
        self.aboutToShow.connect(self._on_show)

    # ------------------------------------------------------------------ contents
    def set_entries(self, jobs: list[Entry], others: list[Entry], cli: list[Entry],
                    cli_menu: QMenu | None = None) -> None:
        self.jobs, self.others, self.cli, self.cli_menu = jobs, others, cli, cli_menu
        self._build()

    def _header(self, text: str) -> QAction:
        a = QAction(text, self)
        f = QFont(a.font())
        f.setBold(True)
        a.setFont(f)
        a.setEnabled(False)
        self.addAction(a)
        return a

    def _entry(self, e: Entry, recent: bool = False) -> QAction:
        a = QAction(e.text, self)
        if e.tip:
            a.setToolTip(e.tip)
            a.setStatusTip(e.tip)
        a.triggered.connect(lambda _=False, entry=e: self.start(entry))
        self.addAction(a)
        if not recent:
            self._entry_actions.append((a, e))
        return a

    def _build(self) -> None:
        self.clear()
        self._entry_actions, self._search_found = [], []
        self.addAction(self._search_action)
        known = {e.id: e for e in self.jobs + self.others + self.cli}
        recent = [known[i] for i in self.recent() if i in known]
        self._recent_actions = []
        self._cli_action = None
        if recent:
            self._recent_actions.append(self._header("Recent"))
            self._recent_actions += [self._entry(e, recent=True) for e in recent]
        self._job_header = self._header("RELION jobs") if recent or self.others else None
        for e in self.jobs:
            self._entry(e)
        self._others_header = self._header("Other programs") if self.others else None
        for e in self.others:
            self._entry(e)
        if self.cli_menu is not None:  # a menu used only here (its action is hidden while searching)
            self.addSeparator()
            self._cli_action = self.addMenu(self.cli_menu)

    def job_actions(self) -> list[QAction]:
        """The actions of RELION's jobs (not Recent's, not the other programs')."""
        jobs = {id(e) for e in self.jobs}
        return [a for a, e in self._entry_actions if id(e) in jobs]

    # ------------------------------------------------------------------ Recent
    def recent(self) -> list[str]:
        if self.settings is not None:
            text = self.settings.value(RECENT_KEY, "")
            return [i for i in str(text or "").split("\n") if i]
        return list(self._recent)

    def remember(self, entry_id: str) -> None:
        ids = [entry_id] + [i for i in self.recent() if i != entry_id]
        ids = ids[:RECENT_MAX * 4]  # more than shown: some are of the other job list (SPA / tomography)
        if self.settings is not None:
            self.settings.setValue(RECENT_KEY, "\n".join(ids))
        else:
            self._recent = ids

    def start(self, e: Entry) -> None:
        self.remember(e.id)
        e.start()

    # ------------------------------------------------------------------ search
    def _on_show(self) -> None:
        self._build()  # Recent as it is now
        self.search.clear()
        QTimer.singleShot(0, lambda: self.search.setFocus(Qt.PopupFocusReason))

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        for a in self._search_found:
            self.removeAction(a)
            a.deleteLater()
        self._search_found = []
        searching = bool(q)
        for a in self._recent_actions:
            a.setVisible(not searching)
        if self._cli_action is not None:
            self._cli_action.setVisible(not searching)
        shown = {"jobs": False, "others": False}
        others = {id(e) for e in self.others}
        for a, e in self._entry_actions:
            on = not searching or q in e.text.lower() or q in e.id.lower()
            a.setVisible(on)
            if on:
                shown["others" if id(e) in others else "jobs"] = True
        if self._job_header is not None:
            self._job_header.setVisible(shown["jobs"])
        if self._others_header is not None:
            self._others_header.setVisible(shown["others"])
        if not searching:
            return
        cli = [e for e in self.cli if q in e.text.lower()][:CLI_MATCHES_MAX]
        if cli:
            self._search_found.append(self._header("Command-line tools"))
            for e in cli:
                a = QAction(e.text, self)
                a.setToolTip(e.tip)
                a.triggered.connect(lambda _=False, entry=e: self.start(entry))
                self.addAction(a)
                self._search_found.append(a)
        if not cli and not shown["jobs"] and not shown["others"]:
            self._search_found.append(self._header("No match"))

    def matches(self) -> list[QAction]:
        """What the search leaves, in order (tests, and Enter)."""
        found = [a for a, _ in self._entry_actions if a.isVisible()]
        return found + [a for a in self._search_found if a.isEnabled()]

    def _start_first(self) -> None:
        found = self.matches() if self.search.text().strip() else []
        if not found:
            return
        for _ in range(5):  # this menu, and the menus it opened from (Job → New job)
            w = QApplication.activePopupWidget()
            if w is None:
                break
            w.close()
        self.close()
        found[0].trigger()
