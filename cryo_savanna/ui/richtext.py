# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Rich text shown again with a few changes, painted again only where it changed.

``QTextEdit.setHtml`` lays out and paints the whole view again, also for the very same text, and over
X11 every paint is sent whole: the job overview, set again at every poll while the job shown runs,
would send the panel every couple of seconds for a time that moved. ``set_html`` compares the new text
with what is shown, block by block, and replaces only the blocks that differ, so only they are laid
out and painted again (and the scroll position and a selection stay). When the structure is not the
same (other blocks, tables or block formats), it is set in full.
"""

from __future__ import annotations

from qtpy.QtGui import QTextBlock, QTextCursor, QTextDocument, QTextFormat
from qtpy.QtWidgets import QTextEdit

MAX_CHANGED = 0.5  # more than this fraction of the blocks changed: set in full


def _norm(fmt):
    """A format as it compares across documents: without the number of the object it belongs to (a table cell's),
    which depends on what the document held before."""
    if fmt is None or not fmt.hasProperty(QTextFormat.ObjectIndex):
        return fmt
    out = type(fmt)(fmt)
    out.clearProperty(QTextFormat.ObjectIndex)
    return out


def _blocks(doc: QTextDocument) -> list[QTextBlock]:
    out = []
    b = doc.begin()
    while b.isValid():
        out.append(b)
        b = b.next()
    return out


def _place(block: QTextBlock) -> tuple:
    """Where a block is: in which table cell (and that table's format), else in which frame's format."""
    cur = QTextCursor(block)
    table = cur.currentTable()
    if table is not None:
        cell = table.cellAt(block.position())
        return ("cell", _norm(table.format()), table.rows(), table.columns(), cell.row(), cell.column(),
                cell.rowSpan(), cell.columnSpan(), _norm(cell.format()))
    frame = cur.currentFrame()
    return ("frame", _norm(frame.frameFormat()) if frame is not None else None)


def _content(block: QTextBlock) -> list:
    out = []
    it = block.begin()
    while not it.atEnd():
        f = it.fragment()
        if f.isValid():
            out.append((f.text(), _norm(f.charFormat())))
        it += 1
    return out


def _scratch(doc: QTextDocument, html: str) -> QTextDocument:
    """``html`` in a document set up as ``doc`` is."""
    new = QTextDocument()
    new.setDefaultFont(doc.defaultFont())
    new.setDefaultStyleSheet(doc.defaultStyleSheet())
    new.setDocumentMargin(doc.documentMargin())
    new.setIndentWidth(doc.indentWidth())
    new.setDefaultTextOption(doc.defaultTextOption())
    new.setHtml(html)
    return new


def changed_blocks(doc: QTextDocument, new: QTextDocument) -> list[int] | None:
    """Indices of the blocks of ``doc`` whose text or character formats differ from ``new``'s; None when the
    two are not laid out alike (blocks, tables, block formats)."""
    old_blocks, new_blocks = _blocks(doc), _blocks(new)
    if len(old_blocks) != len(new_blocks):
        return None
    out = []
    for i, (a, b) in enumerate(zip(old_blocks, new_blocks)):
        if _norm(a.blockFormat()) != _norm(b.blockFormat()) or _norm(a.charFormat()) != _norm(b.charFormat()) \
                or _place(a) != _place(b):
            return None
        if a.text() != b.text() or _content(a) != _content(b):
            out.append(i)
    return out


def set_html(edit: QTextEdit, html: str) -> None:
    """``edit.setHtml(html)``, painting only what changed (see the module)."""
    doc = edit.document()
    new = _scratch(doc, html)
    changed = changed_blocks(doc, new)
    if changed is not None and not changed:
        return
    if changed is None or len(changed) > MAX_CHANGED * max(1, doc.blockCount()):
        bar = edit.verticalScrollBar()
        scroll = bar.value()
        edit.setHtml(html)
        bar.setValue(scroll)
        return
    old_blocks, new_blocks = _blocks(doc), _blocks(new)
    for i in reversed(changed):  # from the end: the positions of the blocks before stay as they are
        a, b = old_blocks[i], new_blocks[i]
        src = QTextCursor(new)
        src.setPosition(b.position())
        src.setPosition(b.position() + b.length() - 1, QTextCursor.KeepAnchor)
        dst = QTextCursor(doc)
        dst.setPosition(a.position())
        dst.setPosition(a.position() + a.length() - 1, QTextCursor.KeepAnchor)
        if b.length() > 1:
            dst.insertFragment(src.selection())
        else:
            dst.removeSelectedText()
