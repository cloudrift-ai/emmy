"""A fold in block form, in four steps: a prototype, wired into no pass.

A fold runs its statements once a step, each ``target[..] += product of reads`` summed over the statement's other
variables, under its conditions. A step reads the state as the step found it and its writes land when it ends.

1. **Split.** The fold's axis is cut into blocks of steps. So is every array axis the fold's variable indexes: a cell
   of such an axis has a block coordinate and a coordinate inside its block.
2. **Lift.** Each statement is cut by the block its state reads fall in, relative to the block of the step: an
   earlier one, the same one, a later one. A condition that is settled by the block coordinates alone is lifted: the
   part is whole blocks, taken or dropped, with no test inside. One that is not stays as a mask over the block.
3. **Isolate.** A read that no earlier step of the block wrote is a read of what the block started from, an input
   to the block. What is left is the dependence graph of the parts: a part that reads what another wrote earlier
   in the block runs after it. A part outside every cycle runs for the whole block at once. A cycle stays serial.
4. **Break.** A cycle whose statements are linear in what they read is a linear system one block wide: the reads of
   the block's steps through what the block's steps wrote. It is one product for the reads, one for that small
   system, a solve the width of the block, and one product for the writes.

A product, a reduction and a map have no cycle and end at step 3 as one operation a block. A triangular solve keeps a
cycle one block wide and its reads of earlier blocks become one product. The delta rule is all cycle, and step 4 on
it is its chunk rule.

The dependence questions are answered by evaluating the index conditions over the block's own index sets, which is
exact for the extents given and reads no data. A compiler with symbolic extents needs a dependence test there.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Semiring:
    """The arithmetic a fold runs over: ``mul`` distributes over ``add``, and both are associative."""

    name: str
    zero: float
    one: float
    add: np.ufunc
    mul: np.ufunc
    product: Callable  # the matrix product of the two

    def mm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.product(a, b) if a.shape[1] else np.full((a.shape[0], b.shape[1]), self.zero)


REAL = Semiring("real", 0.0, 1.0, np.add, np.multiply, np.matmul)
MAX_PLUS = Semiring("max-plus", -np.inf, 0.0, np.maximum, np.add, lambda a, b: (a[:, :, None] + b[None]).max(axis=1))


@dataclass(frozen=True)
class Ix:
    """An index: a variable plus a constant, or a constant alone."""

    var: str | None = None
    off: int = 0


Read = tuple[str, tuple[Ix, ...]]


@dataclass(frozen=True)
class Term:
    """``target += scale * product(factors)`` under ``conds``, summed over every variable the target does not name."""

    target: Read
    factors: tuple[Read, ...]
    conds: tuple[tuple[Ix, str, Ix], ...] = ()
    scale: float = 1.0

    @property
    def names(self) -> tuple[str, ...]:
        reads = (self.target, *self.factors)
        found = [ix.var for _, index in reads for ix in index] + [ix.var for a, _, b in self.conds for ix in (a, b)]
        return tuple(dict.fromkeys(name for name in found if name))


def _ix(text: str) -> Ix:
    text = text.strip()
    if text.isdigit():
        return Ix(None, int(text))
    name, sign, amount = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:([+-])\s*(\d+))?", text).groups()
    return Ix(name, int(sign + amount) if sign else 0)


def _read(text: str) -> Read:
    name, inside = re.fullmatch(r"\s*([A-Za-z_]\w*)\[(.*)\]\s*", text).groups()
    return name, tuple(_ix(part) for part in inside.split(","))


def term(spec: str, *conds: str) -> Term:
    """``term("S[i, j] += L[i, k] * S[k, j]", "k < i")``. A leading ``-`` or a number among the factors is the scale."""
    left, right = spec.split("+=")
    right, scale, factors = right.strip(), 1.0, []
    if right.startswith("-"):
        right, scale = right[1:], -1.0
    for piece in right.split("*"):
        try:
            scale *= float(piece)
        except ValueError:
            factors.append(_read(piece))
    parsed = tuple((_ix(a), op, _ix(b)) for a, op, b in (re.fullmatch(r"(.+?)(<=|<|==)(.+)", cond).groups() for cond in conds))
    return Term(_read(left), tuple(factors), parsed, scale)


@dataclass(frozen=True)
class Op:
    """One operation of the block form over ``m`` rows, ``n`` columns and ``k`` terms a cell.

    ``bulk`` is a part taken for a whole block at once; ``serial`` a cycle walked a step at a time; ``read``,
    ``gram`` and ``write`` the three products of a broken cycle, and ``solve`` its small system."""

    kind: str
    m: int
    n: int
    k: int


def _selects(sr: Semiring, a: np.ndarray) -> bool:
    """Whether a product with ``a`` is a gather: only zeros and ones, at most one a row or one a column."""
    ones = a == sr.one
    return bool(np.all(ones | (a == sr.zero)) and (np.all(ones.sum(axis=1) <= 1) or np.all(ones.sum(axis=0) <= 1)))


@dataclass
class Part:
    """The instances of one statement whose state reads fall in one block each, relative to the step's block."""

    term: Term
    at: int  # the statement's place among the fold's
    blocks: tuple[int, ...]  # per state read on a split axis: -1 an earlier block, 0 the step's, 1 a later one
    mask: np.ndarray  # over the block's steps, then the statement's other variables
    live: tuple[bool, ...] = ()  # per factor: read from the state as it stands, not from what the block started from
    lifted: bool = False  # whole blocks, taken or dropped: no test inside a block


@dataclass
class Group:
    """Parts that run together: one outside every cycle, for the whole block at once, or a cycle, step by step."""

    parts: list[Part]
    cycle: bool


class _Unbroken(Exception):
    """A cycle the fourth step does not take: it stays serial."""


@dataclass
class Fold:
    """``terms`` run once per value of ``axis`` over the arrays of ``state``; every other array is an input."""

    axis: str
    terms: tuple[Term, ...]
    state: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    sr: Semiring = REAL

    def __post_init__(self) -> None:
        self.extents: dict[str, int] = {}
        arrays = self.state | self.inputs
        for part in self.terms:
            for name, index in (part.target, *part.factors):
                for axis, ix in enumerate(index):
                    if ix.var and not ix.off:
                        self.extents[ix.var] = min(self.extents.get(ix.var, 1 << 30), arrays[name].shape[axis])
        #: The array axes the fold's variable indexes: the ones the split cuts into blocks with the steps.
        reads = [(name, index) for part in self.terms for name, index in (part.target, *part.factors) if name in self.state]
        self.split = {(name, axis) for name, index in reads for axis, ix in enumerate(index) if ix.var == self.axis}

    # ---- running statements ----------------------------------------------------------------------------- #

    def _grids(self, part: Term, steps: np.ndarray) -> dict[str, np.ndarray]:
        names = (self.axis, *(name for name in part.names if name != self.axis))
        grids = {}
        for axis, name in enumerate(names):
            values = np.asarray(steps) if name == self.axis else np.arange(self.extents[name])
            grids[name] = values.reshape([-1 if at == axis else 1 for at in range(len(names))])
        return grids

    @staticmethod
    def _index(grids: dict[str, np.ndarray], index: tuple[Ix, ...]) -> tuple:
        return tuple(grids[ix.var] + ix.off if ix.var else np.asarray(ix.off) for ix in index)

    def _mask(self, part: Term, grids: dict[str, np.ndarray]) -> np.ndarray:
        """Where the statement runs: its conditions, and every index inside its array."""
        arrays = self.state | self.inputs
        shape = np.broadcast_shapes(*(grid.shape for grid in grids.values()))
        mask = np.ones(shape, dtype=bool)
        for a, op, b in part.conds:
            (left,), (right,) = self._index(grids, (a,)), self._index(grids, (b,))
            mask &= left < right if op == "<" else left <= right if op == "<=" else left == right
        for name, index in (part.target, *part.factors):
            for extent, at in zip(arrays[name].shape, self._index(grids, index), strict=True):
                mask &= (at >= 0) & (at < extent)
        return mask

    def _add(self, delta: dict, part: Term, steps, mask: np.ndarray, sources: list[dict]) -> None:
        """Add the statement's instances under ``mask`` into ``delta``, factor ``f`` read from ``sources[f]``."""
        sr, grids = self.sr, self._grids(part, steps)
        value = np.full(mask.shape, sr.one if part.scale == 1.0 else part.scale)
        for source, (name, index) in zip(sources, part.factors, strict=True):
            array = source[name]
            at = tuple(np.clip(i, 0, extent - 1) for i, extent in zip(self._index(grids, index), array.shape, strict=True))
            value = sr.mul(value, array[at])
        name, index = part.target
        at = tuple(np.broadcast_to(i, mask.shape)[mask] for i in self._index(grids, index))
        sr.add.at(delta[name], at, value[mask])

    def _zero(self) -> dict[str, np.ndarray]:
        return {name: np.full(array.shape, self.sr.zero) for name, array in self.state.items()}

    def _sources(self, part: Term, live: dict, start: dict, which: tuple[bool, ...]) -> list[dict]:
        return [self.inputs if name in self.inputs else live if now else start for (name, _), now in zip(part.factors, which, strict=True)]

    def serial(self) -> dict[str, np.ndarray]:
        """The fold as written: one step after another."""
        state = {name: np.array(array, dtype=float) for name, array in self.state.items()}
        for step in range(self.extents[self.axis]):
            delta = self._zero()
            for part in self.terms:
                steps = np.array([step])
                self._add(
                    delta,
                    part,
                    steps,
                    self._mask(part, self._grids(part, steps)),
                    self._sources(part, state, state, (True,) * len(part.factors)),
                )
            state = {name: self.sr.add(array, delta[name]) for name, array in state.items()}
        return state

    # ---- steps 1 to 3: split, lift, isolate ------------------------------------------------------------ #

    def plan(self, steps: np.ndarray, width: int) -> list[Group]:
        """The groups one block of ``steps`` runs, in order."""
        block, first = int(steps[0]) // width, int(steps[0])
        parts: list[Part] = []
        for at, part in enumerate(self.terms):
            grids = self._grids(part, steps)
            mask = self._mask(part, grids)
            code = np.zeros(mask.shape, dtype=np.int64)
            for name, index in part.factors:
                for axis, where in enumerate(self._index(grids, index)):
                    if (name, axis) in self.split:
                        code = code * 3 + np.sign(where // width - block) + 1
            for value in np.unique(code[mask]):
                blocks, rest = [], int(value)
                while len(blocks) < sum((name, axis) in self.split for name, index in part.factors for axis in range(len(index))):
                    blocks.append(rest % 3 - 1)
                    rest //= 3
                parts.append(Part(part, at, tuple(reversed(blocks)), mask & (code == value)))

        # What each part writes, step by step, and so what is written before a step and from it on.
        position = {id(part): self._grids(part.term, steps)[self.axis] - first for part in parts}
        written = {}
        for part in parts:
            name, index = part.term.target
            cells = np.zeros((len(steps), *self.state[name].shape), dtype=bool)
            grids = self._grids(part.term, steps)
            at = tuple(np.broadcast_to(i, part.mask.shape)[part.mask] for i in (position[id(part)], *self._index(grids, index)))
            cells[at] = True
            before = np.zeros_like(cells)
            before[1:] = np.logical_or.accumulate(cells, axis=0)[:-1]
            written[id(part)] = (name, before, np.logical_or.accumulate(cells[::-1], axis=0)[::-1])

        def hits(reader: Part, factor: int, cells: np.ndarray) -> bool:
            grids = self._grids(reader.term, steps)
            at = (position[id(reader)], *self._index(grids, reader.term.factors[factor][1]))
            at = tuple(np.clip(i, 0, extent - 1) for i, extent in zip(at, cells.shape, strict=True))
            return bool((cells[at] & reader.mask).any())

        earlier: dict[int, set[int]] = {index: set() for index in range(len(parts))}  # who must run before whom
        loops = set()
        for r, reader in enumerate(parts):
            live = []
            for f, (name, _) in enumerate(reader.term.factors):
                writers = [(w, written[id(writer)]) for w, writer in enumerate(parts) if written[id(writer)][0] == name]
                live.append(name in self.state and any(hits(reader, f, before) for _, (_, before, _) in writers))
                if not live[-1]:
                    continue
                for w, (_, before, after) in writers:
                    if hits(reader, f, before):
                        earlier[r].add(w)  # it reads what the writer left earlier in the block
                        loops |= {r} if w == r else set()
                    if w != r and hits(reader, f, after):
                        earlier[w].add(r)  # and must not see what the writer adds from this step on
            reader.live = tuple(live)
            reader.lifted = self._lifted(reader, width)
        return [Group([parts[index] for index in members], len(members) > 1 or members[0] in loops) for members in _components(earlier)]

    def _lifted(self, part: Part, width: int) -> bool:
        """Whether the part is whole blocks: for each choice of block coordinates, every instance or none."""
        names = (self.axis, *(name for name in part.term.names if name != self.axis))
        reads = (part.term.target, *part.term.factors)
        blocked = {ix.var for name, index in reads for axis, ix in enumerate(index) if (name, axis) in self.split} - {self.axis}
        mask = np.pad(
            part.mask, [(0, -extent % width if name in blocked else 0) for name, extent in zip(names, part.mask.shape, strict=True)]
        )
        shape, inside = [], []
        for name, extent in zip(names, mask.shape, strict=True):
            shape += [extent // width, width] if name in blocked else [extent]
            inside.append(len(shape) - 1)
        cut = mask.reshape(shape)
        return bool((cut.any(axis=tuple(inside)) == cut.all(axis=tuple(inside))).all())

    # ---- running a block ---------------------------------------------------------------------------------- #

    def blocked(self, width: int, log: list | None = None, *, break_cycles: bool = True) -> dict[str, np.ndarray]:
        """The fold in blocks of ``width`` steps."""
        sr = self.sr
        state = {name: np.array(array, dtype=float) for name, array in self.state.items()}
        for first in range(0, self.extents[self.axis], width):
            steps = np.arange(first, min(first + width, self.extents[self.axis]))
            start = {name: array.copy() for name, array in state.items()}
            for group in self.plan(steps, width):
                if not group.cycle:
                    (part,) = group.parts
                    delta = self._zero()
                    self._add(delta, part.term, steps, part.mask, self._sources(part.term, state, start, part.live))
                    state = {name: sr.add(array, delta[name]) for name, array in state.items()}
                    if log is not None:
                        log.append(self._shape("bulk", part, steps))
                    continue
                if break_cycles:
                    try:
                        state = self._broken(group, steps, state, start, log)
                        continue
                    except _Unbroken:
                        pass
                for at in range(len(steps)):
                    delta = self._zero()
                    for part in group.parts:
                        self._add(
                            delta, part.term, steps[at : at + 1], part.mask[at : at + 1], self._sources(part.term, state, start, part.live)
                        )
                    state = {name: sr.add(array, delta[name]) for name, array in state.items()}
                if log is not None:
                    log.extend(self._shape("serial", part, steps) for part in group.parts)
        return state

    def _shape(self, kind: str, part: Part, steps: np.ndarray) -> Op:
        name, index = part.term.target
        grids = self._grids(part.term, steps)
        cells = np.zeros(self.state[name].shape, dtype=bool)
        cells[tuple(np.broadcast_to(i, part.mask.shape)[part.mask] for i in self._index(grids, index))] = True
        rows = int(cells.reshape(-1, cells.shape[-1]).any(axis=1).sum()) if cells.ndim > 1 else 1
        return Op(kind, rows, cells.shape[-1], int(part.mask.sum()) // max(1, int(cells.sum())))

    # ---- step 4: break a cycle ---------------------------------------------------------------------------- #

    def _broken(self, group: Group, steps: np.ndarray, state: dict, start: dict, log: list | None) -> dict:
        """The cycle's steps as one linear system. Each statement at each step is an event: it writes ``write``,
        a vector over the rows of its array, times a row of values, which is ``scale`` times what it reads through
        ``reader``, a vector over the rows it reads, or times a row of its own."""
        sr = self.sr
        events: list[tuple] = []  # (step, array written, write, array read or None, reader or row, scale)
        kept = {name: [sr.one] * len(steps) for name in self.state}  # what a step keeps of each array's own value
        for at, step in enumerate(steps):
            for part in group.parts:
                self._events(events, kept, part, at, int(step), state, start)

        def rows(name: str) -> np.ndarray:
            return state[name].reshape(-1, state[name].shape[-1])

        def carried(name: str, since: int, until: int) -> float:
            value = sr.one
            for at in range(since, until):
                value = sr.mul(value, kept[name][at])
            return value

        def taken(kind: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Only the rows both sides hold take part: a reader over one block of rows is a product that wide.
            met = (a != sr.zero).any(axis=0) & (b != sr.zero).any(axis=1)
            if log is not None and met.any() and not _selects(sr, a[:, met]) and not _selects(sr, b[met]):
                log.append(Op(kind, a.shape[0], b.shape[1], int(met.sum())))
            return sr.mm(a[:, met], b[met])

        width = next(iter(self.state.values())).shape[-1]
        values = np.full((len(events), width), sr.zero)
        for name in self.state:
            readers = [e for e, event in enumerate(events) if event[3] == name]
            writers = [e for e, event in enumerate(events) if event[1] == name]
            if not readers:
                continue
            through = np.stack([events[e][4] for e in readers])
            read = taken("read", through, rows(name))
            met = taken("gram", through, np.stack([events[e][2] for e in writers], axis=1)) if writers else None
            for row, e in enumerate(readers):
                events[e] = (*events[e], read[row], {w: met[row, column] for column, w in enumerate(writers)} if writers else {})
        for e, event in enumerate(events):  # the small system, a step's events after every earlier step's
            at, _, _, name, own, scale = event[:6]
            if name is None:
                values[e] = sr.mul(scale, own)
                continue
            total = sr.mul(carried(name, 0, at), event[6])
            for w, weight in event[7].items():
                if events[w][0] < at and weight != sr.zero:
                    total = sr.add(total, sr.mul(sr.mul(carried(name, events[w][0] + 1, at), weight), values[w]))
            values[e] = sr.mul(scale, total)
        if log is not None and len(events) > 1:
            log.append(Op("solve", len(events), width, len(events)))
        out = {}
        for name, array in state.items():
            writers = [e for e, event in enumerate(events) if event[1] == name]
            new = sr.mul(carried(name, 0, len(steps)), rows(name))
            if writers:
                scaled = np.stack([sr.mul(carried(name, events[e][0] + 1, len(steps)), values[e]) for e in writers])
                new = sr.add(new, taken("write", np.stack([events[e][2] for e in writers], axis=1), scaled))
            out[name] = new.reshape(array.shape)
        return out

    def _events(self, events: list, kept: dict, part: Part, at: int, step: int, state: dict, start: dict) -> None:
        """One statement at one step, as events. Raises ``_Unbroken`` for a shape the linear system does not hold.

        The statement's factors are sorted by what they vary with: the row it writes, the row it reads, the column,
        or nothing. A statement whose factors and mask keep the written row apart from the read one is ONE event,
        a vector over the rows written and one over the rows read. Otherwise it is an event per row written."""
        sr, term_ = self.sr, part.term
        if not part.mask[at].any():
            return  # the statement does not run at this step
        target, index = term_.target
        column = index[-1]
        live = [factor for factor, now in zip(term_.factors, part.live, strict=True) if now]
        if not column.var or column.off or column.var == self.axis or len(live) > 1:
            raise _Unbroken
        row = index[0] if len(index) == 2 else Ix(None, 0)
        height = self.state[target].shape[0] if len(index) == 2 else 1
        reads = live[0] if live else None
        if reads is not None and reads[1][-1] != column:
            raise _Unbroken
        read_row = None if reads is None else reads[1][0] if len(reads[1]) == 2 else Ix(None, 0)
        depth = 0 if reads is None else self.state[reads[0]].shape[0] if len(reads[1]) == 2 else 1

        def free(ix: Ix | None) -> str | None:
            return ix.var if ix is not None and ix.var and ix.var != self.axis else None

        i, k, j = free(row), free(read_row), column.var
        names = [name for name in term_.names if name != self.axis]
        full = tuple(self.extents[name] for name in names)
        if set(names) - {i, k, j} or (i and row.off) or (k and read_row.off):
            raise _Unbroken
        grids = self._grids(term_, np.array([step]))

        def along(values: np.ndarray, *over: str) -> np.ndarray:
            values = np.broadcast_to(values, full)[tuple(slice(None) if name in over else 0 for name in names)]
            order = [name for name in names if name in over]
            return np.transpose(values, [order.index(name) for name in over]) if len(over) > 1 else values

        products: dict[frozenset, np.ndarray] = {}
        for factor, now in zip(term_.factors, part.live, strict=True):
            if now:
                continue
            array = (self.inputs if factor[0] in self.inputs else start)[factor[0]]
            where = tuple(np.clip(at_, 0, extent - 1) for at_, extent in zip(self._index(grids, factor[1]), array.shape, strict=True))
            key = frozenset(ix.var for ix in factor[1] if ix.var and ix.var != self.axis)
            value = np.broadcast_to(array[where], np.broadcast_shapes(np.shape(array[where]), (1,) * (len(names) + 1)))[0]
            products[key] = sr.mul(products[key], value) if key in products else value
        scale = along(products.pop(frozenset(), np.asarray(sr.one)))
        scale = float(sr.mul(scale, term_.scale)) if term_.scale != 1.0 else float(scale)
        mask = part.mask[at]

        def ones(*over: str) -> np.ndarray:
            return np.full(tuple(self.extents[name] for name in over), sr.one)

        def taken(*over: str) -> np.ndarray:
            values = along(products.pop(frozenset(over)), *over) if frozenset(over) in products else ones(*over)
            others = tuple(axis for axis, name in enumerate(names) if name not in over)
            return np.where(along(mask.any(axis=others, keepdims=True), *over), values, sr.zero)

        def unit(size: int, where: Ix) -> np.ndarray:
            vector = np.full(size, sr.zero)
            vector[step + where.off if where.var == self.axis else where.off] = sr.one
            return vector

        def alone(*over: str) -> bool:
            """Whether the mask over ``over`` is each variable's own mask, with no tie between them."""
            others = tuple(axis for axis, name in enumerate(names) if name not in over)
            here = mask.any(axis=others) if others else mask
            each = [here.any(axis=tuple(a for a in range(here.ndim) if a != axis), keepdims=True) for axis in range(here.ndim)]
            return bool((np.logical_and.reduce(np.broadcast_arrays(*each)) == here).all()) if each else True

        present = [name for name in (i, k, j) if name]
        if not alone(*dict.fromkeys(present)) and not (i and (k or reads is None)):
            raise _Unbroken
        if reads is None:  # a row of the statement's own: it does not read the state
            across = taken(j)
            tied = frozenset((i, j)) in products or not alone(i, j) if i else False
            if i and tied:
                both = np.where(along(mask, i, j), along(products.pop(frozenset((i, j)), np.asarray(sr.one)), i, j), sr.zero)
                down = taken(i)
                for r in range(height):
                    if (both[r] != sr.zero).any() and down[r] != sr.zero:
                        events.append((at, target, unit(height, Ix(None, r)), None, sr.mul(sr.mul(down[r], both[r]), across), scale))
            else:
                events.append((at, target, taken(i) if i else unit(height, row), None, across, scale))
            if products:
                raise _Unbroken
            return
        if frozenset((j,)) in products or not (mask.all(axis=names.index(j)) == mask.any(axis=names.index(j))).all():
            raise _Unbroken  # a coefficient that differs by column
        if i and i == k:  # the cell's own previous value
            if target == reads[0] and not products and mask.all():
                kept[target][at] = sr.add(kept[target][at], scale)
                return
            down = taken(i)
            if products:
                raise _Unbroken
            for r in range(height):
                if down[r] != sr.zero:
                    events.append((at, target, unit(height, Ix(None, r)), reads[0], unit(depth, Ix(None, r)), sr.mul(scale, down[r])))
            return
        tied = bool(i and k) and (frozenset((i, k)) in products or not alone(i, k))
        if tied:
            both = np.where(
                along(mask.any(axis=names.index(j), keepdims=True), i, k),
                along(products.pop(frozenset((i, k)), np.asarray(sr.one)), i, k),
                sr.zero,
            )
            down, across = taken(i), taken(k)
            for r in range(height):
                reader = sr.mul(both[r], across)
                if down[r] != sr.zero and (reader != sr.zero).any():
                    events.append((at, target, unit(height, Ix(None, r)), reads[0], reader, sr.mul(scale, down[r])))
        else:
            write = taken(i) if i else unit(height, row)
            reader = taken(k) if k else unit(depth, read_row)
            events.append((at, target, write, reads[0], reader, scale))
        if products:
            raise _Unbroken


def _components(earlier: dict[int, set[int]]) -> list[list[int]]:
    """The strongly connected components of ``earlier`` (node -> the nodes that run before it), each after the
    ones it waits for."""
    order, low, stack, found, out = {}, {}, [], set(), []

    def visit(node: int) -> None:
        order[node] = low[node] = len(order)
        stack.append(node)
        for before in earlier[node]:
            if before not in order:
                visit(before)
                low[node] = min(low[node], low[before])
            elif before not in found:
                low[node] = min(low[node], order[before])
        if low[node] == order[node]:
            members = []
            while True:
                members.append(stack.pop())
                found.add(members[-1])
                if members[-1] == node:
                    break
            out.append(sorted(members))

    for node in earlier:
        if node not in order:
            visit(node)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The first three steps on a Loop IR kernel that carries a state.
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One read of the carried state in a Loop IR kernel, for one block of rows and one side of it."""

    name: str  # the read's SSA name
    block: int  # -1 the rows of earlier blocks, 0 the block's own, 1 later ones
    live: bool  # some step reads a cell an earlier step of the block wrote
    lifted: bool  # whole blocks of the variable it sums over: no test inside one


class Walk:
    """A Loop IR kernel that carries one state and moves along one axis of it, a row a step.

    What its steps read and write is ``ir/guards.py``'s reading: the kernel's selects and index conditionals as
    comparisons. Nothing here looks at an extent's size but the block bookkeeping below, which counts steps."""

    def __init__(self, body) -> None:
        from emmy.compiler.ir.guards import carried  # noqa: PLC0415 - the prototype above needs no IR
        from emmy.compiler.ir.stmt import Loop  # noqa: PLC0415

        self.facts = carried(body)
        self.loop, self.carry = self.facts.loop, self.facts.carry
        self.stmts = list(self.loop.body.iter())
        self.cells = list(self.facts.cells)
        axes = {stmt.axis.name: stmt.axis for stmt in self.stmts if isinstance(stmt, Loop)}
        #: The state's own shape.
        self.shape = tuple(axes[name].extent.as_static() for name in self.cells)
        found = self.facts.moving()
        #: The state's axis the steps move along, and the row a step writes less the step.
        self.moving, self.offset = found if found is not None else (None, 0)
        steps = self.loop.axis.extent.as_static()
        self.rows = np.arange(steps) + self.offset if found is not None else np.full(steps, -1)

    def steps_of(self, width: int, block: int) -> np.ndarray:
        """The steps that write the rows of ``block``."""
        return np.flatnonzero((self.rows >= 0) & (self.rows // width == block))


def survey(body, width: int, block: int) -> list[Reading]:
    """Split, lift and isolate, read off a Loop IR body that carries one state: each read of the state, on each
    side of the block of rows ``block`` it can fall on while a step of that block runs."""
    from emmy.compiler.ir.constraints import always, can_hold  # noqa: PLC0415
    from emmy.compiler.ir.expression import BinaryExpression, Literal, Var  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Loop  # noqa: PLC0415

    walk = Walk(body)
    if walk.moving is None:
        return []
    facts, at = walk.facts, walk.moving
    low, high = Literal(block * width, "int"), Literal((block + 1) * width, "int")
    row = BinaryExpression("+", Var(facts.step), Literal(walk.offset, "int"))
    # A step of this block: the row it writes is one of the block's.
    here = (*facts.bounds, BinaryExpression("<=", low, row), BinaryExpression("<", row, high))
    summed = {stmt.axis.name for stmt in walk.stmts if isinstance(stmt, Loop) and stmt.is_reduce}
    found: dict[tuple[str, int], Reading] = {}
    for read in facts.reads:
        cell = read.index[at]
        tied = [condition for condition in read.guard if condition.free_vars() & cell.free_vars() & summed]
        rest = [condition for condition in read.guard if condition not in tied]
        sides = (
            (-1, (BinaryExpression("<", cell, low),)),
            (0, (BinaryExpression("<=", low, cell), BinaryExpression("<", cell, high))),
            (1, (BinaryExpression("<=", high, cell),)),
        )
        for side, where in sides:
            if not can_hold(*read.guard, *here, *where):
                continue
            live = False
            if side == 0:
                # An earlier step of the block wrote this cell: the step whose row it is, where that step writes it.
                then = {facts.step: BinaryExpression("-", cell, Literal(walk.offset, "int"))}
                then |= dict(zip(facts.cells, read.index, strict=True))
                earlier = BinaryExpression("<", cell, row)
                live = any(can_hold(*read.guard, *here, *where, earlier, *(c.substitute(then) for c in guard)) for guard in facts.written)
            lifted = all(always(condition, *rest, *here, *where) for condition in tied)
            seen = found.get((read.name, side))
            found[(read.name, side)] = Reading(read.name, side, live or bool(seen and seen.live), lifted and (seen is None or seen.lifted))
    return list(found.values())
