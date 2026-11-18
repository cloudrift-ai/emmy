"""A fold in block form: a prototype, wired into no pass.

A fold walks a step over an ordered axis and carries a state. This module regroups the steps of ANY fold whose step
is affine in the state over a semiring, and it is one rule for all of them: a map, a reduction, a product, a prefix
sum, a recurrence with a dense step, a triangular solve and the delta rule differ only in what the rule finds.

A step's map is kept factored, never as a matrix over the state::

    S[X]  <-  alpha[X] * S[X]  +  U . C . R(S)

``U`` is where the step writes (some rows as they are, or every row through a tall basis read from an input), ``R``
is what it reads (state rows as they are, state rows through an input, or input rows), and ``C`` is a small
coefficient between them. Two maps compose into one of the same form, and the only arithmetic is on the small side::

    C  =  | alpha2 * C1        0           |        G21 = R2 . U1
          | C2 . G21 . C1      C2 * alpha1 |

So the coefficient of a block is block lower triangular BY CONSTRUCTION, and each merge is two small products
around ``G21``, the later steps' readers taken on what the earlier steps wrote. ``G21`` says how serial a fold is:

- empty: the steps do not see each other. The block is one scatter (a map), one reduce, or one product over the
  block (the K chunk of a tile);
- rows gathered from an input: a triangular solve. The merges are the blocked inverse;
- a product of two inputs: the delta rule. The merges are its chunk rule, the triangular system included.

Blocks compose bottom up in pairs (``compose_all``) and a block's map applies to the state the block starts from
(``Map.apply``). How many steps a block holds and how far the tree goes are choices, none taken here.

Which cells of ``C`` can be nonzero is the same computation over the Boolean semiring (``Fold.over(BOOL)``).

What it does not do: a step that multiplies two reads of its state is refused, unless one of them reads rows no
earlier step wrote, which are then an input (the solve in place is that case); a map acts along one axis of the state
and the other passes through; free batch axes, a scale that differs per row, and rows a step overwrites without
reading are left out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np


class NotAffine(ValueError):
    """The step is not affine in its state, so its steps do not compose."""


@dataclass(frozen=True)
class Semiring:
    """The arithmetic a fold runs over: ``mul`` distributes over ``add``, and both are associative."""

    name: str
    zero: object
    one: object
    dtype: type
    add: Callable
    mul: Callable
    product: Callable  # the matrix product of the two
    neg: Callable | None = None  # a ring's negation; a semiring has none

    def zeros(self, *shape: int) -> np.ndarray:
        return np.full(shape, self.zero, dtype=self.dtype)

    def eye(self, n: int) -> np.ndarray:
        out = self.zeros(n, n)
        out[np.arange(n), np.arange(n)] = self.one
        return out

    def mm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.product(a, b) if a.shape[1] else self.zeros(a.shape[0], b.shape[1])


REAL = Semiring("real", 0.0, 1.0, np.float64, np.add, np.multiply, np.matmul, np.negative)
MAX_PLUS = Semiring("max-plus", -np.inf, 0.0, np.float64, np.maximum, np.add, lambda a, b: (a[:, :, None] + b[None]).max(axis=1))
#: Which values can be nonzero. A difference is nonzero where either side is.
BOOL = Semiring(
    "pattern", False, True, np.bool_, np.logical_or, np.logical_and, lambda a, b: (a[:, :, None] & b[None]).any(axis=1), lambda a: a
)


@dataclass(frozen=True)
class Product:
    """One matrix product the block form runs: ``m x k`` times ``k x n``."""

    kind: str  # leaf, gram, merge, read, mix, write; "reduce" folds k rows into each of m
    m: int
    n: int
    k: int


def _selects(sr: Semiring, a: np.ndarray) -> bool:
    """Whether a product with ``a`` is a gather: only zeros and ones, at most one a row or one a column."""
    ones = a == sr.one
    return bool(np.all(ones | (a == sr.zero)) and (np.all(ones.sum(axis=1) <= 1) or np.all(ones.sum(axis=0) <= 1)))


def _mm(sr: Semiring, a: np.ndarray, b: np.ndarray, log: list | None, kind: str) -> np.ndarray:
    work = min(a.shape + b.shape) > 0 and a.size > 1 and b.size > 1
    if log is not None and work and not _selects(sr, a) and not _selects(sr, b):
        log.append(Product(kind, a.shape[0], b.shape[1], a.shape[1]))
    return sr.mm(a, b)


# ---------------------------------------------------------------------------------------------------------------
# The step, written once: run on numbers it is the serial walk, run on a ``Lin`` it is read as a map.
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Run:
    sr: Semiring
    log: list | None = None
    traced: bool = False


class Val:
    """A number, a row or a matrix a step computes from its inputs."""

    def __init__(self, run: _Run, a) -> None:
        self.run, self.a = run, np.asarray(a, dtype=run.sr.dtype)

    @property
    def T(self) -> Val:
        return Val(self.run, self.a.T)

    def __getitem__(self, key) -> Val:
        return Val(self.run, np.atleast_2d(self.a[key]))

    def _other(self, o) -> np.ndarray:
        return o.a if isinstance(o, Val) else np.asarray(o, dtype=self.run.sr.dtype)

    def __add__(self, o):
        return NotImplemented if isinstance(o, Lin) else Val(self.run, self.run.sr.add(self.a, self._other(o)))

    def __sub__(self, o):
        return NotImplemented if isinstance(o, Lin) else self + Val(self.run, _neg(self.run.sr, self._other(o)))

    def __mul__(self, o):
        return NotImplemented if isinstance(o, Lin) else Val(self.run, self.run.sr.mul(self.a, self._other(o)))

    def __matmul__(self, o):
        if isinstance(o, Lin):
            return NotImplemented
        if self.run.traced:  # a product of two inputs stays factored: its left side is where a step writes
            return Lin.of(o).left_times(self)
        return Val(self.run, _mm(self.run.sr, self.a, o.a, None, "leaf"))


def _neg(sr: Semiring, a: np.ndarray) -> np.ndarray:
    if sr.neg is None:
        raise NotAffine(f"the {sr.name} semiring has no difference")
    return sr.neg(a)


@dataclass(eq=False)
class _Reader:
    """What a term reads: state rows as they are (``rows``), state rows through ``mat``, or input rows (no array)."""

    array: str
    rows: tuple[int, ...] | None = None
    mat: np.ndarray | None = None

    @property
    def key(self) -> tuple[str, str]:
        return ("in", "") if not self.array else ("unit" if self.mat is None else "dense", self.array)


@dataclass
class _Group:
    left: np.ndarray | None  # None: the rows as they are
    coefs: dict[_Reader, np.ndarray]


class Lin:
    """A value that is linear in the state: a sum of ``left . coef . reader`` terms over ``k`` rows."""

    def __init__(self, run: _Run, k: int, widths: dict[str, int]) -> None:
        self.run, self.k, self.widths, self.groups = run, k, widths, []

    @staticmethod
    def of(value) -> Lin:
        out = Lin(value.run, value.a.shape[0], {})
        out._add(_Group(None, {_Reader("", mat=value.a): value.run.sr.eye(value.a.shape[0])}))
        return out

    def _add(self, group: _Group) -> None:
        sr = self.run.sr
        for mine in self.groups:
            if mine.left is group.left:
                for reader, coef in group.coefs.items():
                    mine.coefs[reader] = sr.add(mine.coefs[reader], coef) if reader in mine.coefs else coef
                return
        self.groups.append(_Group(group.left, dict(group.coefs)))

    def _like(self, k: int | None = None) -> Lin:
        return Lin(self.run, self.k if k is None else k, self.widths)

    def _map(self, f: Callable) -> Lin:
        out = self._like()
        out.groups = [_Group(g.left, {r: f(c) for r, c in g.coefs.items()}) for g in self.groups]
        return out

    def __add__(self, o):
        out, o = self._map(lambda c: c), (o if isinstance(o, Lin) else Lin.of(o))
        out.widths = self.widths | o.widths
        for group in o.groups:
            out._add(group)
        return out

    __radd__ = __add__

    def __neg__(self):
        return self._map(lambda c: _neg(self.run.sr, c))

    def __sub__(self, o):
        return self + (-o if isinstance(o, Lin) else Val(self.run, _neg(self.run.sr, o.a)))

    def __rsub__(self, o):
        return -self + o

    def __mul__(self, o):
        if isinstance(o, Lin) or np.size(scale := o.a if isinstance(o, Val) else o) != 1:
            raise NotAffine("the step multiplies two reads of its state")
        return self._map(lambda c: self.run.sr.mul(np.asarray(scale).reshape(1, 1), c))

    __rmul__ = __mul__

    def __matmul__(self, o):
        raise NotAffine("the step multiplies two reads of its state, or acts on the axis that passes through")

    def __rmatmul__(self, o):
        return self.left_times(o)

    def left_times(self, w: Val) -> Lin:
        """``w @ self``. A tall ``w`` is where the step writes; a wide one is folded into what it reads."""
        sr, log = self.run.sr, self.run.log
        out = self._like(w.a.shape[0])
        for group in self.groups:
            if group.left is not None:
                out._add(_Group(_mm(sr, w.a, group.left, log, "leaf"), group.coefs))
            elif w.a.shape[0] > w.a.shape[1]:
                out._add(_Group(w.a, group.coefs))
            else:
                for reader, coef in group.coefs.items():
                    scaled = np.array_equal(coef, sr.mul(coef[:1, :1], sr.eye(coef.shape[0])))
                    rows = sr.mul(coef[:1, :1], w.a) if scaled else _mm(sr, w.a, coef, log, "leaf")
                    if reader.mat is not None:
                        rows = _mm(sr, rows, reader.mat, log, "leaf")
                    else:  # state rows read as they are, through ``w``: rows of a matrix over the whole array
                        wide = sr.zeros(rows.shape[0], self.widths[reader.array])
                        for column, row in enumerate(reader.rows):
                            wide[:, row] = sr.add(wide[:, row], rows[:, column])
                        rows = wide
                    out._add(_Group(None, {_Reader(reader.array, mat=rows): sr.eye(rows.shape[0])}))
        return out


class _Array:
    """One state array inside one step. A read sees the state before the step."""

    def __init__(self, walk: _Walk, name: str) -> None:
        self.walk, self.name = walk, name

    def _rows(self, key) -> tuple[int, ...]:
        return tuple(int(i) for i in np.arange(self.walk.before[self.name].shape[0])[key].reshape(-1))

    def __getitem__(self, key):
        walk, (rows, columns) = self.walk, (key if isinstance(key, tuple) else (key, None))
        rows = self._rows(rows)
        if not walk.run.traced or columns is not None:
            if walk.run.traced and walk.written[self.name] & set(rows):
                raise NotAffine("the step multiplies two reads of its state")
            # Rows no earlier step wrote still hold what the fold started from: an input, not the state.
            value = walk.before[self.name][list(rows)]
            return Val(walk.run, value if columns is None else value[:, columns])
        out = Lin(walk.run, len(rows), {name: a.shape[0] for name, a in walk.before.items()})
        out._add(_Group(None, {_Reader(self.name, rows=rows): walk.run.sr.eye(len(rows))}))
        return out

    def __setitem__(self, key, value) -> None:
        self.walk.writes.append((self.name, self._rows(key), value))


class _Walk:
    def __init__(self, run: _Run, before: dict[str, np.ndarray], written: dict[str, set[int]]) -> None:
        self.run, self.before, self.written, self.writes = run, before, written, []

    def __getattr__(self, name: str) -> _Array:
        return _Array(self, name)


# ---------------------------------------------------------------------------------------------------------------
# The map of one step or of a block of steps.
# ---------------------------------------------------------------------------------------------------------------

Key = tuple[str, str]  # ("full" | "rows", array) on the left; ("unit" | "dense", array) or ("in", "") on the right


@dataclass
class Map:
    sr: Semiring
    outputs: frozenset[str]  # arrays a fold only writes, each row once, zero before
    sizes: dict[str, int]  # the rows of each array
    alpha: dict[str, object] = field(default_factory=dict)  # the scale of each array; absent is one
    full: dict[str, np.ndarray] = field(default_factory=dict)  # written through a basis: array -> n x p
    rows: dict[str, tuple[int, ...]] = field(default_factory=dict)  # written as they are: one row a coefficient row
    unit: dict[str, tuple[int, ...]] = field(default_factory=dict)  # read as they are: one row a coefficient column
    dense: dict[str, np.ndarray] = field(default_factory=dict)  # read through an input: array -> q x n
    inputs: np.ndarray | None = None  # q x m input rows
    coef: dict[tuple[Key, Key], np.ndarray] = field(default_factory=dict)
    written: dict[str, frozenset[int]] = field(default_factory=dict)

    def lefts(self) -> list[Key]:
        return [("full", x) for x in self.full] + [("rows", x) for x in self.rows]

    def readers(self) -> list[Key]:
        keys = [("unit", x) for x in self.unit] + [("dense", x) for x in self.dense]
        return keys + ([("in", "")] if self.inputs is not None else [])

    def p(self, key: Key) -> int:
        kind, x = key
        return (self.full[x].shape[1] if x in self.full else 0) if kind == "full" else len(self.rows.get(x, ()))

    def q(self, key: Key) -> int:
        kind, x = key
        if kind == "in":
            return 0 if self.inputs is None else self.inputs.shape[0]
        return len(self.unit.get(x, ())) if kind == "unit" else (self.dense[x].shape[0] if x in self.dense else 0)

    def apply(self, state: dict[str, np.ndarray], log: list | None = None) -> dict[str, np.ndarray]:
        """The state after the map's steps, from the state before them."""
        sr = self.sr
        read = {}
        for key in self.readers():
            kind, x = key
            if kind == "in":
                read[key] = self.inputs
            elif kind == "unit":
                read[key] = state[x][list(self.unit[x])]
            else:
                read[key] = _mm(sr, self.dense[x], state[x], log, "read")
        new = {x: sr.mul(np.asarray(self.alpha.get(x, sr.one), dtype=sr.dtype), a) for x, a in state.items()}
        for left in self.lefts():
            kind, x = left
            terms = [_mm(sr, c, read[r], log, "mix") for (lk, r), c in self.coef.items() if lk == left]
            if not terms:
                continue
            value = terms[0]
            for term in terms[1:]:
                value = sr.add(value, term)
            if kind == "full":
                new[x] = sr.add(new[x], _mm(sr, self.full[x], value, log, "write"))
                continue
            rows = self.rows[x]
            for at, row in enumerate(rows):
                new[x][row] = sr.add(new[x][row], value[at])
            if log is not None and len(set(rows)) < len(rows):
                log.append(Product("reduce", len(set(rows)), value.shape[1], len(rows) // len(set(rows))))
        return new


def _leaf(sr: Semiring, outputs: frozenset[str], shapes: dict[str, int], writes: list, log: list | None) -> Map:
    """The map of one step, read off what the step wrote."""
    out = Map(sr, outputs, shapes)
    entries: list[tuple[Key, object, dict[_Reader, np.ndarray]]] = []
    for name, rows, value in writes:
        value = value if isinstance(value, Lin) else Lin.of(value)
        whole = rows == tuple(range(shapes[name]))
        kept = None  # the scale of the rows' own previous value
        groups = [_Group(g.left, dict(g.coefs)) for g in value.groups]
        for group in groups:
            for reader, coef in list(group.coefs.items()):
                own = group.left is None and reader.array == name and reader.rows == rows
                if own and kept is None and np.array_equal(coef, sr.mul(coef[:1, :1], sr.eye(len(rows)))):
                    kept = coef[0, 0]
                    del group.coefs[reader]
        if whole:
            out.alpha[name] = kept if kept is not None else (sr.one if name in outputs else sr.zero)
        elif (kept is None and name not in outputs) or (kept is not None and kept != sr.one):
            raise NotImplementedError("a step that overwrites or scales some rows of a carried array")
        out.written[name] = out.written.get(name, frozenset()) | frozenset(rows)
        as_they_are: dict[_Reader, np.ndarray] = {}
        for group in groups:
            if group.left is not None and whole and group.left.shape[1] < len(rows):
                entries.append((("full", name), group.left, group.coefs))
                continue
            for reader, coef in group.coefs.items():
                coef = coef if group.left is None else _mm(sr, group.left, coef, log, "leaf")
                as_they_are[reader] = sr.add(as_they_are[reader], coef) if reader in as_they_are else coef
        if as_they_are:
            entries.append((("rows", name), rows, as_they_are))
    columns: dict[_Reader, tuple[Key, int]] = {}
    for _, _, coefs in entries:
        for reader in coefs:
            if reader in columns:
                continue
            key = reader.key
            columns[reader] = (key, out.q(key))
            if key[0] == "unit":
                out.unit[reader.array] = out.unit.get(reader.array, ()) + reader.rows
            elif key[0] == "dense":
                out.dense[reader.array] = _stack(out.dense.get(reader.array), reader.mat, 0)
            else:
                out.inputs = _stack(out.inputs, reader.mat, 0)
    placed = []
    for left, payload, coefs in entries:
        placed.append((left, out.p(left), coefs))
        if left[0] == "full":
            out.full[left[1]] = _stack(out.full.get(left[1]), payload, 1)
        else:
            out.rows[left[1]] = out.rows.get(left[1], ()) + payload
    for left, row, coefs in placed:
        for reader, coef in coefs.items():
            key, column = columns[reader]
            block = out.coef.setdefault((left, key), sr.zeros(out.p(left), out.q(key)))
            block[row : row + coef.shape[0], column : column + coef.shape[1]] = coef
    return _prune(out, log)


def _stack(a: np.ndarray | None, b: np.ndarray | None, axis: int) -> np.ndarray | None:
    return b if a is None else a if b is None else np.concatenate([a, b], axis=axis)


def compose(second: Map, first: Map, log: list | None = None) -> Map:
    """The map of ``first``'s steps and then ``second``'s."""
    sr = first.sr
    zero = sr.zero

    def gram(reader: Key, left: Key) -> np.ndarray | None:
        """``second``'s readers taken on what ``first`` wrote."""
        (rk, x), (lk, y) = reader, left
        if rk == "in" or x != y:
            return None
        if rk == "unit":
            rows = list(second.unit[x])
            if lk == "full":
                return first.full[x][rows]
            return np.where(np.equal.outer(rows, first.rows[x]), sr.one, zero).astype(sr.dtype)
        if lk == "full":
            return _mm(sr, second.dense[x], first.full[x], log, "gram")
        return second.dense[x][:, list(first.rows[x])]

    seen: dict[tuple[Key, Key], np.ndarray] = {}  # second's readers, as coefficients over first's readers
    for reader in second.readers():
        grams = {left: gram(reader, left) for left in first.lefts()}
        for (left, r1), c1 in first.coef.items():
            g = grams[left]
            if g is not None:
                term = _mm(sr, g, c1, log, "merge")
                seen[(reader, r1)] = sr.add(seen[(reader, r1)], term) if (reader, r1) in seen else term

    out = Map(sr, first.outputs, first.sizes)
    for x in {*first.alpha, *second.alpha}:
        out.alpha[x] = sr.mul(second.alpha.get(x, sr.one), first.alpha.get(x, sr.one))
    for x in {*first.written, *second.written}:
        out.written[x] = first.written.get(x, frozenset()) | second.written.get(x, frozenset())
    for x in {*first.full, *second.full}:
        out.full[x] = _stack(first.full.get(x), second.full.get(x), 1)
    for x in {*first.rows, *second.rows}:
        out.rows[x] = first.rows.get(x, ()) + second.rows.get(x, ())
    for x in {*first.unit, *second.unit}:
        out.unit[x] = first.unit.get(x, ()) + second.unit.get(x, ())
    for x in {*first.dense, *second.dense}:
        out.dense[x] = _stack(first.dense.get(x), second.dense.get(x), 0)
    out.inputs = _stack(first.inputs, second.inputs, 0)

    for left in out.lefts():
        for reader in out.readers():
            p1, q1 = first.p(left), first.q(reader)
            block = sr.zeros(out.p(left), out.q(reader))
            keep = second.alpha.get(left[1], sr.one)
            if (left, reader) in first.coef and keep != zero:
                block[:p1, :q1] = sr.mul(np.asarray(keep, dtype=sr.dtype), first.coef[(left, reader)])
            for (l2, r2), c2 in second.coef.items():
                if l2 == left and (r2, reader) in seen:
                    block[p1:, :q1] = sr.add(block[p1:, :q1], _mm(sr, c2, seen[(r2, reader)], log, "merge"))
            if (left, reader) in second.coef:
                c2 = second.coef[(left, reader)]
                if reader[0] != "in":  # a state reader of the second map sees the first map's scale of its array
                    c2 = sr.mul(np.asarray(first.alpha.get(reader[1], sr.one), dtype=sr.dtype), c2)
                if reader[0] == "unit" and reader[1] in out.outputs:
                    # A row the first map writes was zero before it: the second map reads it through ``gram`` alone.
                    stale = [row in first.written.get(reader[1], ()) for row in second.unit[reader[1]]]
                    c2 = np.where(stale, zero, c2).astype(sr.dtype)
                block[p1:, q1:] = c2
            out.coef[(left, reader)] = block
    return _prune(out, log)


def _prune(out: Map, log: list | None = None) -> Map:
    """Drop what no coefficient uses, read each state row once, and keep no more readers than rows written."""
    sr = out.sr
    _drop_unused(out)
    for key in [k for k in out.readers() if k[0] != "unit"]:
        # Factoring a map pays while it reads and writes less than the whole array. Input rows beyond the rows
        # written through them, and a state read through as many rows as the array has by a map that writes that
        # many, are multiplied in now. This is where a recurrence with a dense step composes its matrices and its
        # offsets, one product each.
        blocks = {left: c for (left, r), c in out.coef.items() if r == key}
        written = sum(c.shape[0] for c in blocks.values())
        if key[0] == "in":
            factored = out.q(key) <= written
        else:
            size = out.sizes[key[1]]
            factored = out.q(key) < size or sum(out.p(left) for left in out.lefts() if left[1] == key[1]) < size
        if factored:
            continue
        source = out.inputs if key[0] == "in" else out.dense[key[1]]
        taken, at = [], 0
        for left, coef in blocks.items():
            taken.append(_mm(sr, coef, source, log, "merge"))
            out.coef[(left, key)] = sr.zeros(coef.shape[0], written)
            out.coef[(left, key)][:, at : at + coef.shape[0]] = sr.eye(coef.shape[0])
            at += coef.shape[0]
        if key[0] == "in":
            out.inputs = np.concatenate(taken)
        else:
            out.dense[key[1]] = np.concatenate(taken)
    return out


def _drop_unused(out: Map) -> None:
    sr = out.sr
    for x, rows in list(out.unit.items()):
        first = [rows.index(row) for row in rows]
        if first != list(range(len(rows))):
            for key in [k for k in out.coef if k[1] == ("unit", x)]:
                block = out.coef[key].copy()
                for at, to in enumerate(first):
                    if to != at:
                        block[:, to] = sr.add(block[:, to], block[:, at])
                        block[:, at] = sr.zero
                out.coef[key] = block
    for key in out.lefts():
        blocks = [k for k in out.coef if k[0] == key]
        used = np.zeros(out.p(key), dtype=bool)
        for k in blocks:
            used |= (out.coef[k] != sr.zero).any(axis=1)
        for k in blocks:
            out.coef[k] = out.coef[k][used]
        kind, x = key
        if kind == "full":
            out.full[x] = out.full[x][:, used]
        else:
            out.rows[x] = tuple(row for row, keep in zip(out.rows[x], used, strict=True) if keep)
    for key in out.readers():
        blocks = [k for k in out.coef if k[1] == key]
        used = np.zeros(out.q(key), dtype=bool)
        for k in blocks:
            used |= (out.coef[k] != sr.zero).any(axis=0)
        for k in blocks:
            out.coef[k] = out.coef[k][:, used]
        kind, x = key
        if kind == "in":
            out.inputs = out.inputs[used] if used.any() else None
        elif kind == "unit":
            out.unit[x] = tuple(row for row, keep in zip(out.unit[x], used, strict=True) if keep)
        else:
            out.dense[x] = out.dense[x][used]
    out.coef = {k: c for k, c in out.coef.items() if c.size}
    out.full = {x: a for x, a in out.full.items() if a.shape[1]}
    out.rows = {x: r for x, r in out.rows.items() if r}
    out.unit = {x: r for x, r in out.unit.items() if r}
    out.dense = {x: a for x, a in out.dense.items() if a.shape[0]}


def compose_all(maps: list[Map], log: list | None = None) -> Map:
    """One map for consecutive steps, composed bottom up in pairs."""
    while len(maps) > 1:
        maps = [compose(maps[i + 1], maps[i], log) if i + 1 < len(maps) else maps[i] for i in range(0, len(maps), 2)]
    return maps[0]


# ---------------------------------------------------------------------------------------------------------------
# A fold.
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Fold:
    """``steps`` runs of ``step(t, state, inputs)`` over a state of named arrays.

    ``carried`` arrays start from the values given; ``outputs`` (name -> shape) start at zero and take each row once.
    The step reads ``state.NAME[rows]`` or, for a coefficient, ``state.NAME[rows, columns]``, assigns
    ``state.NAME[rows] = value``, and combines values with ``+``, ``-``, ``*`` by a scalar and ``@``."""

    steps: int
    step: Callable
    carried: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray]
    outputs: dict[str, tuple[int, int]] = field(default_factory=dict)
    sr: Semiring = REAL

    def start(self) -> dict[str, np.ndarray]:
        state = {name: np.array(a, dtype=self.sr.dtype) for name, a in self.carried.items()}
        return state | {name: self.sr.zeros(*shape) for name, shape in self.outputs.items()}

    def _walk(self, run: _Run, t: int, before: dict, written: dict) -> _Walk:
        walk = _Walk(run, before, written)
        self.step(t, walk, SimpleNamespace(**{name: Val(run, a) for name, a in self.inputs.items()}))
        return walk

    def serial(self) -> dict[str, np.ndarray]:
        """The fold as written: one step after another."""
        state = self.start()
        for t in range(self.steps):
            after = {name: a.copy() for name, a in state.items()}
            for name, rows, value in self._walk(_Run(self.sr), t, state, {}).writes:
                after[name][list(rows)] = value.a
            state = after
        return state

    def leaves(self, log: list | None = None) -> list[Map]:
        """Each step's map. Raises ``NotAffine`` for a step that has none."""
        start, shapes = self.start(), {name: a.shape[0] for name, a in self.start().items()}
        written = {name: set() for name in start}
        maps = []
        for t in range(self.steps):
            walk = self._walk(_Run(self.sr, log, traced=True), t, start, written)
            maps.append(_leaf(self.sr, frozenset(self.outputs), shapes, walk.writes, log))
            for name, rows, _ in walk.writes:
                written[name] |= set(rows)
        return maps

    def blocks(self, block: int, log: list | None = None) -> list[Map]:
        leaves = self.leaves(log)
        return [compose_all(leaves[at : at + block], log) for at in range(0, self.steps, block)]

    def blocked(self, block: int, log: list | None = None) -> dict[str, np.ndarray]:
        """The fold in blocks of ``block`` steps: each block's map composed bottom up, then applied in order."""
        state = self.start()
        for step in self.blocks(block, log):
            state = step.apply(state, log)
        return state

    def over(self, sr: Semiring) -> Fold:
        """The same fold over another semiring, each value replaced by whether it is nonzero."""
        nonzero = lambda a: np.asarray(a) != self.sr.zero  # noqa: E731
        return replace(
            self,
            sr=sr,
            carried={name: nonzero(a) for name, a in self.carried.items()},
            inputs={name: nonzero(a) for name, a in self.inputs.items()},
        )
