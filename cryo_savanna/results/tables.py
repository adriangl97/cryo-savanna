# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""STAR loops as columns, with numeric views for sorting and plotting."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from cryo_savanna.project.star import StarBlock, loop_columns, read_star


def is_text_label(label: str) -> bool:
    """A column of names or files (rlnTomoName, rlnMicrographName, rlnImageName, rlnOpticsGroupName, ...
    StarFile): text, also when every value in it looks like a number (tomograms named 001, 002)."""
    return label.endswith(("Name", "File"))


@dataclass
class Table:
    name: str
    columns: list[str]
    data: dict[str, list[str]]
    source: str = ""
    _numeric: dict[str, np.ndarray | None] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(next(iter(self.data.values()))) if self.data else 0

    @classmethod
    def from_block(cls, block: StarBlock, source: str = "") -> "Table":
        if not block.is_loop:  # a list block: one row
            return cls(block.name, list(block.columns), {c: [v] for c, v in zip(block.columns, block.rows[0] if
                                                                                 block.rows else [])}, source)
        cols = list(zip(*block.rows)) if block.rows else [() for _ in block.columns]
        return cls(block.name, list(block.columns), {c: list(v) for c, v in zip(block.columns, cols)}, source)

    @classmethod
    def read(cls, path: str | os.PathLike, block: str, labels: Iterable[str] | None = None) -> "Table | None":
        """One block; with labels, only those columns (streamed, fine for millions of particles)."""
        if labels is None:
            b = read_star(path, [block]).get(block)
            return cls.from_block(b, str(path)) if b is not None else None
        r = loop_columns(path, block, labels)
        if r is None:
            return None
        _, data = r
        return cls(block, list(data), data, str(path))

    def has(self, label: str) -> bool:
        return label in self.data

    def column(self, label: str) -> list[str]:
        return self.data[label]

    def numeric(self, label: str) -> np.ndarray | None:
        """The column as floats (NaN where a value is not a number), or None if it is text."""
        if label not in self._numeric:
            vals = self.data.get(label)
            arr = None
            if is_text_label(label):
                pass  # names are text, whatever they look like
            elif isinstance(vals, np.ndarray) and vals.dtype == np.float64:  # read_compact: already numbers
                arr = vals
            elif vals is not None:
                try:
                    arr = np.array(vals, dtype=np.float64)
                except ValueError:
                    conv = np.array([_num(v) for v in vals], dtype=np.float64)
                    arr = conv if np.isfinite(conv).sum() >= max(1, len(vals) // 2) else None
            self._numeric[label] = arr
        return self._numeric[label]

    def numbers(self, label: str) -> np.ndarray:
        """The column as floats, for a column that has to be numbers (coordinates, angles): NaN where a
        value is not one, also when most are not (numeric() says None then)."""
        arr = self.numeric(label)
        if arr is None:
            arr = np.array([_num(v) for v in self.data[label]], dtype=np.float64)
        return arr

    def numeric_columns(self) -> list[str]:
        return [c for c in self.columns if self.numeric(c) is not None]

    def add_column(self, label: str, values: list[str]) -> None:
        self.columns.append(label)
        self.data[label] = values
        self._numeric.pop(label, None)


COMPACT_CHUNK = 65536


class _Column:
    """One column of read_compact: float arrays while every value is a number, else text."""

    def __init__(self, text: bool = False):
        self.chunks: list = []
        self.text = text  # from the start: a column of names (is_text_label)

    def add(self, values) -> None:
        if not self.text:
            try:
                self.chunks.append(np.array(values, dtype=np.float64))
                return
            except ValueError:  # text after all: what was read so far goes back to text
                self.text = True
                self.chunks = [[f"{x:.10g}" for x in c] for c in self.chunks]
        self.chunks.append(list(values))

    def result(self):
        if self.text:
            return [v for c in self.chunks for v in c]
        return np.concatenate(self.chunks) if self.chunks else np.empty(0)


def read_compact(path: str | os.PathLike, block: str) -> Table | None:
    """A whole block, with each column that holds only numbers kept as a float array.

    For the table viewer and the column plots of big particle files: a value then takes 8 bytes
    instead of a Python string (about 60), so millions of rows fit in memory. Text columns stay
    strings. Numeric columns are floats in ``data`` (``column()`` gives the array), so this is not a
    drop-in for code that wants the file's text. List blocks are read as by Table.read.
    """
    from cryo_savanna.project.star import StarError, tokenize_line
    target = f"data_{block}"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("data_") and raw.split()[0] == target:
                break
        else:
            return None
        labels: list[str] = []
        columns: list[_Column] | None = None
        rows: list[list[str]] = []
        is_loop = False

        def flush() -> None:
            if rows and columns is not None:
                for col, values in zip(columns, zip(*rows)):
                    col.add(values)
                rows.clear()

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
                return Table.read(path, block)
            if line[0] == "_" and columns is None:
                labels.append(line[1:].split()[0])
                continue
            if columns is None:
                columns = [_Column(is_text_label(lab)) for lab in labels]
            toks = tokenize_line(line) if ('"' in line or "'" in line or "#" in line) else line.split()
            if not toks:
                continue
            if len(toks) != len(labels):
                if len(toks) < len(labels) <= 2:  # as RELION's readStarLoop
                    toks += [""] * (len(labels) - len(toks))
                else:
                    raise StarError(f"data_{block}: expected {len(labels)} values, got {len(toks)}: {line!r}")
            rows.append(toks)
            if len(rows) >= COMPACT_CHUNK:
                flush()
        if columns is None:
            columns = [_Column(is_text_label(lab)) for lab in labels]
        flush()
    return Table(block, labels, {lab: col.result() for lab, col in zip(labels, columns)}, str(path))


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def blocks_of(path: str | os.PathLike) -> list[str]:
    """Names of the data blocks of a STAR file, without parsing rows."""
    names = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("data_"):
                names.append(line[5:].split()[0] if line[5:].strip() else "")
    return names


def optics_value(path: Path, label: str) -> float | None:
    """A value from the first row of the optics block (pixel size, box size, ...)."""
    from cryo_savanna.project.star import read_star_head
    try:
        b = read_star_head(path, ["optics"]).get("optics")
    except (OSError, ValueError):
        return None
    if b is None or not b.has(label) or not b.rows:
        return None
    try:
        return float(b.get(label))
    except (TypeError, ValueError):
        return None
