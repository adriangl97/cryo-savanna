# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from RELION (Sjors H.W. Scheres, MRC Laboratory of Molecular Biology; GPL-2.0-or-later): see NOTICE.md.
# RELION's notice, kept as it asks:
#   Author: "Sjors H.W. Scheres"
#   MRC Laboratory of Molecular Biology
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   This complete copyright notice must be included in any revised version of the
#   source code. Additional authorship citations may be added, but existing
#   author citations must be preserved.
#
#   The XMIPP code in RELION's src/strings.cpp and src/metadata_table.cpp (GPL-2.0-or-later):
#   Authors: J.R. Bilbao-Castro (jrbcast@ace.ual.es)
#   Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
"""Reading and writing STAR files the way RELION does.

The tokenizer mirrors ``nextTokenInSTAR`` and the writer mirrors
``MetaDataTable::write`` / ``escapeStringForSTAR`` / ``getValueToString`` from
RELION 5.1 (src/strings.cpp, src/metadata_table.cpp), so files written here are
byte-compatible with files written by RELION itself.

Values are kept as strings when reading. Typed values (bool/int/float/str) are
formatted RELION-style when writing.
"""

from __future__ import annotations

import errno
import io
import os
import uuid
from dataclasses import dataclass, field
from typing import IO, Iterable, Iterator

CURRENT_MDT_VERSION = 50001


class StarError(ValueError):
    pass


@dataclass
class StarBlock:
    """One ``data_`` block: either a list (single row of label/value pairs) or a loop table."""

    name: str
    columns: list[str] = field(default_factory=list)  # labels without the leading underscore
    rows: list[list] = field(default_factory=list)
    is_loop: bool = True
    version: int | None = None  # from a preceding "# version NNNNN" line
    comment: str = ""

    # -- convenience ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def col_index(self, label: str) -> int:
        try:
            return self.columns.index(label.lstrip("_"))
        except ValueError:
            raise KeyError(f"label {label} not in block data_{self.name}") from None

    def has(self, label: str) -> bool:
        return label.lstrip("_") in self.columns

    def column(self, label: str) -> list:
        i = self.col_index(label)
        return [r[i] for r in self.rows]

    def records(self) -> Iterator[dict]:
        for r in self.rows:
            yield dict(zip(self.columns, r))

    def as_dict(self) -> dict:
        """For list blocks: label -> value."""
        if not self.rows:
            return {}
        return dict(zip(self.columns, self.rows[0]))

    def get(self, label: str, default=None, row: int = 0):
        label = label.lstrip("_")
        if label not in self.columns or row >= len(self.rows):
            return default
        return self.rows[row][self.columns.index(label)]

    @classmethod
    def from_dict(cls, name: str, values: dict) -> "StarBlock":
        return cls(name=name, columns=list(values), rows=[list(values.values())], is_loop=False)

    @classmethod
    def from_records(cls, name: str, columns: list[str], records: Iterable[Iterable]) -> "StarBlock":
        return cls(name=name, columns=list(columns), rows=[list(r) for r in records], is_loop=True)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_line(line: str) -> list[str]:
    """Split one STAR data line into tokens (port of RELION's nextTokenInSTAR)."""
    if '"' not in line and "'" not in line:
        out = []
        for tok in line.split():
            if tok[0] == "#":
                break
            out.append(tok)
        return out

    tokens: list[str] = []
    n = len(line)
    i = 0
    while i < n:
        # skip blanks
        while i < n and line[i] in " \t\n\r":
            i += 1
        if i >= n or line[i] == "#":
            break
        if line[i] in "'\"":
            quote = line[i]
            start = i + 1
            pos = start
            buf = []
            end = -1
            while pos < n:
                c = line[pos]
                if c == quote and line[pos - 1] != "\a":
                    nxt = pos + 1
                    if nxt == n or line[nxt] in " \t\n\r":
                        end = pos - 1
                        break
                if c != "\a":
                    buf.append(c)
                pos += 1
            if end == -1:
                raise StarError(f"Could not find closing quote in STAR line: {line!r}")
            tokens.append("".join(buf))
            i = pos + 1
        else:
            j = i + 1
            while j < n and line[j] not in " \t\n\r":
                j += 1
            tokens.append(line[i:j])
            i = j + 1
    return tokens


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_star(source: str | os.PathLike | IO[str], blocks: Iterable[str] | None = None) -> dict[str, StarBlock]:
    """Read all (or only the named) data blocks of a STAR file, preserving order."""
    wanted = set(blocks) if blocks is not None else None
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            return _parse(fh, wanted)
    return _parse(source, wanted)


def read_star_head(path: str | os.PathLike, blocks: Iterable[str]) -> dict[str, StarBlock]:
    """Like read_star(path, blocks), but stops reading once all those blocks have been read.

    For the small blocks at the start of big files (optics before millions of particles).
    """
    wanted = set(blocks)
    seen: set[str] = set()

    def lines() -> Iterator[str]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if raw.startswith("data_"):
                    if wanted <= seen:
                        return
                    seen.add(raw[5:].split()[0] if raw[5:].strip() else "")
                yield raw

    return _parse(lines(), wanted)


def loop_rows(path: str | os.PathLike, block: str, label: str | None = None) -> tuple[int, list[str]] | None:
    """Stream one loop block: (number of rows, values of column ``label`` if given).

    Rows are counted without being parsed, so this is fast on large particle files. Returns
    None when the file has no such loop block (or no such column).
    """
    target = f"data_{block}"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("data_") and raw.split()[0] == target:
                break
        else:
            return None
        columns: list[str] = []
        n = 0
        values: list[str] = []
        col = -1
        is_loop = False
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#":
                continue
            if line.startswith("data_"):
                break
            if line.startswith("loop_"):
                is_loop = True
                continue
            if not is_loop:  # a list block, not a loop
                return None
            if line[0] == "_" and n == 0:
                columns.append(line[1:].split()[0])
                continue
            if label is not None and col < 0:
                if label not in columns:
                    return None
                col = columns.index(label)
            n += 1
            if col >= 0:
                toks = tokenize_line(line) if ('"' in line or "'" in line) else line.split()
                values.append(toks[col] if col < len(toks) else "")
    if not columns:
        return None
    if label is not None and col < 0:
        return (0, []) if label in columns else None
    return n, values


def loop_columns(path: str | os.PathLike, block: str, labels: Iterable[str]) -> tuple[list[str], dict[str, list[str]]] | None:
    """Stream some columns of one loop block: (all column labels, {label: values}) for the labels present.

    Only the wanted columns are kept, so a particle file with millions of rows stays cheap.
    None when the file has no such loop block.
    """
    target = f"data_{block}"
    wanted = list(labels)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("data_") and raw.split()[0] == target:
                break
        else:
            return None
        columns: list[str] = []
        picks: list[tuple[str, int]] | None = None
        out: dict[str, list[str]] = {}
        is_loop = False
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#":
                continue
            if line.startswith("data_"):
                break
            if line.startswith("loop_"):
                is_loop = True
                continue
            if not is_loop:
                return None
            if line[0] == "_" and picks is None:
                columns.append(line[1:].split()[0])
                continue
            if picks is None:
                picks = [(lab, columns.index(lab)) for lab in wanted if lab in columns]
                out = {lab: [] for lab, _ in picks}
            toks = tokenize_line(line) if ('"' in line or "'" in line) else line.split()
            for lab, i in picks:
                out[lab].append(toks[i] if i < len(toks) else "")
    if not columns:
        return None
    if picks is None:
        out = {lab: [] for lab in wanted if lab in columns}
    return columns, out


def loop_tokens(path: str | os.PathLike, block: str) -> Iterator[list[str]]:
    """Stream one loop block: first its column labels, then every row as its values (as read_star
    gives them, with the same checks), one at a time.

    For files too big to hold as rows of strings (millions of particles): the caller keeps what it
    needs of each row. Yields nothing when the file has no such loop block.
    """
    target = f"data_{block}"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("data_") and raw.split()[0] == target:
                break
        else:
            return
        columns: list[str] = []
        in_header = is_loop = False
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#":
                continue
            if line.startswith("data_"):
                break
            if line.startswith("loop_"):
                is_loop = in_header = True
                continue
            if not is_loop:  # a list block, not a loop
                return
            if line[0] == "_" and in_header:
                columns.append(line[1:].split()[0])
                continue
            if in_header:
                in_header = False
                yield columns
            toks = tokenize_line(line)
            if not toks:
                continue
            if len(toks) < len(columns) <= 2:  # as in _parse (RELION's readStarLoop)
                toks += [""] * (len(columns) - len(toks))
            elif len(toks) != len(columns):
                raise StarError(f"data_{block}: expected {len(columns)} values, got {len(toks)}: {line!r}")
            yield toks
        if in_header and columns:  # a loop without rows
            yield columns


def parse_star_text(text: str, blocks: Iterable[str] | None = None) -> dict[str, StarBlock]:
    return _parse(io.StringIO(text), set(blocks) if blocks is not None else None)


def _parse(fh: IO[str], wanted: set[str] | None) -> dict[str, StarBlock]:
    result: dict[str, StarBlock] = {}
    cur: StarBlock | None = None
    keep = False
    in_loop_header = False
    pending_version: int | None = None

    for raw in fh:
        line = raw.strip()
        if not line:
            continue
        if line[0] == "#":
            parts = line[1:].split()
            if len(parts) == 2 and parts[0] == "version" and parts[1].isdigit():
                pending_version = int(parts[1])
            elif cur is not None and not cur.comment and not cur.rows and not cur.columns:
                cur.comment = line[1:].strip()
            continue
        if line.startswith("data_"):
            name = line[5:].split()[0] if len(line) > 5 else ""
            cur = StarBlock(name=name, is_loop=False, version=pending_version)
            pending_version = None
            keep = wanted is None or name in wanted
            if keep:
                result[name] = cur
            in_loop_header = False
            continue
        if cur is None:
            continue
        if line.startswith("loop_"):
            cur.is_loop = True
            in_loop_header = True
            continue
        if not keep:
            continue
        if line[0] == "_" and (in_loop_header or not cur.is_loop):
            if cur.is_loop:
                cur.columns.append(line[1:].split()[0])
            else:
                parts = line[1:].split(None, 1)  # the label ends at the first space or tab
                label, rest = parts[0], parts[1] if len(parts) > 1 else ""
                toks = tokenize_line(rest)
                cur.columns.append(label)
                if not cur.rows:
                    cur.rows.append([])
                cur.rows[0].append(toks[0] if toks else "")
            continue
        # data line of a loop
        in_loop_header = False
        if not cur.is_loop:
            continue
        toks = tokenize_line(line)
        if not toks:
            continue
        if len(toks) < len(cur.columns) <= 2:
            # RELION's readStarLoop accepts this for two columns ("fn_mtf <empty>"); the value is empty
            toks += [""] * (len(cur.columns) - len(toks))
        elif len(toks) != len(cur.columns):
            raise StarError(
                f"data_{cur.name}: expected {len(cur.columns)} values, got {len(toks)}: {line!r}"
            )
        cur.rows.append(toks)
    return result


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def escape_string(value: str) -> str:
    """Port of escapeStringForSTAR, which also quotes a value starting with '#': RELION writes that bare,
    and every STAR reader (RELION's too) then takes it for a comment, so the value is lost."""
    if value == "":
        return '""'
    if value[0] in "\"'#" or " " in value or "\t" in value:
        return _quote(value)
    return value


def _quote(value: str) -> str:
    """escapeStringForSTAR's quoting: in double quotes, a quote that would end the value early marked with \\a."""
    out = ['"']
    n = len(value)
    for pos, c in enumerate(value):
        if c == '"':
            nxt = pos + 1
            if nxt == n - 1 or (nxt < n and value[nxt] in " \t"):
                out.append("\a")
        out.append(c)
    out.append('"')
    return "".join(out)


_LINE_STARTS = ("_", "data_", "loop_", ";")  # what a STAR line must not start with to be a row of values


def _c_trunc(s: str) -> str:
    # snprintf(buffer, 13, ...) keeps at most 12 characters
    return s[:12]


def format_value(v) -> str:
    """Format a value as RELION's getValueToString(escape=true) would."""
    if isinstance(v, bool):
        return _c_trunc("%12d" % int(v))
    if isinstance(v, int):
        return _c_trunc("%12d" % v)
    if isinstance(v, float):
        a = abs(v)
        if (0.0 < a < 0.001) or a > 100000.0:
            s = "%12.5e" % v if v < 0 else "%12.6e" % v
        else:
            s = "%12.5f" % v if v < 0 else "%12.6f" % v
        return _c_trunc(s)
    if v is None:
        return '""'
    return escape_string(str(v))


def format_row(row: Iterable) -> str:
    """One row of a loop as MetaDataTable::write writes it, with its newline."""
    values = [format_value(v) for v in row]
    if values and values[0].startswith(_LINE_STARTS):
        # RELION writes it bare, and every reader (RELION's too) then takes a row starting with
        # "_" for a label, "data_" / "loop_" for a new block or loop, and skips one starting
        # with ";" after the labels: quoted, it is read as the value it is
        values[0] = _quote(values[0])
    return "".join(f"{v:>10} " for v in values) + "\n"


@dataclass
class FormattedLoop:
    """A loop block whose rows are text already (format_row): for tables too big to hold as values."""

    name: str
    columns: list[str]
    lines: list[str]
    comment: str = ""


def _block_text(block: "StarBlock | FormattedLoop", version: int = CURRENT_MDT_VERSION) -> Iterator[str]:
    """The pieces of one block, exactly as MetaDataTable::write writes it (nothing for an empty block)."""
    formatted = isinstance(block, FormattedLoop)
    if not (block.lines if formatted else block.rows):
        return
    yield f"\n# version {version}\n\ndata_{block.name}\n"
    if block.comment:
        yield f"# {block.comment}\n"
    yield "\n"
    if formatted or block.is_loop:
        yield "loop_ \n"
        for i, label in enumerate(block.columns, start=1):
            yield f"_{label} #{i} \n"
        if formatted:
            yield from block.lines
        else:
            for row in block.rows:
                yield format_row(row)
        yield " \n"
    else:
        max_width = max([10] + [len(c) for c in block.columns])
        row = block.rows[0]
        for label, v in zip(block.columns, row):
            pad = 12 + max_width - len(label)
            yield f"_{label}{' ':>{pad}}{format_value(v)}\n"
        yield " \n"


def format_block(block: "StarBlock | FormattedLoop", version: int = CURRENT_MDT_VERSION) -> str:
    """Render one block exactly like MetaDataTable::write (empty blocks render as '')."""
    return "".join(_block_text(block, version))


def format_star(blocks: Iterable["StarBlock | FormattedLoop"]) -> str:
    return "".join(format_block(b) for b in blocks)


def write_star(path: str | os.PathLike, blocks: Iterable["StarBlock | FormattedLoop"], atomic: bool = True) -> None:
    """Write blocks to ``path``. With ``atomic`` the file is written to a temp file and renamed.

    A file this user may not write is refused, as RELION's in-place write would be: a rename would
    replace it anyway (the directory is writable), and the new file, now this user's, would shut
    out its owner's RELION (a group-shared project where the owner's files are not group-writable).

    A symlink is written through, as RELION writes it: the file it points to is replaced, the link stays.
    """
    blocks = list(blocks)  # written piece by piece: a big table is never one string in memory
    path = os.fspath(path)
    if os.path.lexists(path) and not os.access(path, os.W_OK):
        raise PermissionError(errno.EACCES, "Not writable by you (nor by RELION run as you); ask its owner "
                              "to make it group-writable", path)
    if not atomic:
        with open(path, "w", encoding="utf-8") as fh:
            for b in blocks:
                fh.writelines(_block_text(b))
        return
    if os.path.islink(path):
        path = os.path.realpath(path)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"  # one per write: other threads, processes and hosts (a shared project)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for b in blocks:
                fh.writelines(_block_text(b))
            fh.flush()
            os.fsync(fh.fileno())
        # Keep permissions/group of the file we replace (projects are often group-shared).
        try:
            st = os.stat(path)
        except FileNotFoundError:
            pass
        else:
            os.chmod(tmp, st.st_mode & 0o7777)
            try:
                os.chown(tmp, -1, st.st_gid)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:  # a full disk or quota: no half-written file left next to the real one
            os.unlink(tmp)
        except OSError:
            pass
        raise
