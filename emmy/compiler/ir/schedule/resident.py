"""Shared-memory residency for a carried state: the block one CTA holds across the steps.

A carried state is an operand of its own step. Its ``STAGE`` says where it lives across the sequential axis:
``direct`` keeps it in the global buffer and launches once per step, the register tier gives a warp its
rows, and ``smem`` gives a CTA a BLOCK — the cells under one batch coordinate — which every step reads and
writes in place behind a barrier. :class:`BlockProgram` is the proof that a kernel's state can be held that
way: which coordinates of the state are the grid's and which are the block's, that no step reads outside
the CTA's own block, and what the block is seeded from. Derived once per kernel, before any schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt import Let, Load, Select

if TYPE_CHECKING:
    from emmy.compiler.ir.tile import OutputSpec, TileOp


def lagged(time: str) -> Expr:
    """The previous step's coordinate as the lift spells it: ``time > 0 ? time - 1 : 0``."""
    return TernaryExpr(BinaryExpr(">", Var(time), Literal(0, "int")), BinaryExpr("-", Var(time), Literal(1, "int")), Literal(0, "int"))


def first_step(time: str) -> Expr:
    """The predicate the lift selects a lagged read over its seed with."""
    return BinaryExpr(">", Var(time), Literal(0, "int"))


def _guard(entry: Expr, own: Expr) -> Expr | None | bool:
    """How a read addresses one BATCH position of the state: ``True`` at the CTA's own coordinate,
    the guard when it reads the own coordinate under a predicate and coordinate zero otherwise (the
    roll's clamped spelling of a read whose value the step then discards), ``None`` when it reads
    another block."""
    if entry == own:
        return True
    if (
        isinstance(own, Var)
        and isinstance(entry, TernaryExpr)
        and entry.if_true == own
        and isinstance(entry.if_false, Literal)
        and entry.if_false.value == 0
    ):
        return entry.cond
    return None


@dataclass(frozen=True, slots=True)
class BlockProgram:
    """A carried state a CTA can hold as one block in shared memory.

    ``batch`` are the free axes the launch grid binds — every read of the state takes the CTA's own
    coordinate there — and ``cells`` the block's axes, at ``cell_positions`` of the state's index
    (position 0 is time). ``seed`` is the buffer the block is filled from cell by cell before the
    first step, or the constant. ``bytes`` is one block's size."""

    state: OutputSpec
    batch: tuple[Axis, ...]
    cells: tuple[Axis, ...]
    cell_positions: tuple[int, ...]
    seed: str | float
    bytes: int

    @classmethod
    def from_tile(cls, tile: TileOp) -> BlockProgram | None:
        from emmy.compiler.ir.tile.ir import loaded_buffers  # noqa: PLC0415

        op = tile.op
        if op is None or op.axis is not None or len(tile.place.serial) != 1 or any(not a.extent.is_static for a in tile.axes):
            return None
        loads = tuple(loaded_buffers(op))
        own = {load.input for load in loads} & {spec.write.output for spec in tile.output_specs}
        states = [spec for spec in tile.output_specs if spec.write.output in own]
        if len(states) != 1:
            return None
        state = states[0]
        time = tile.place.serial[0].name
        index = state.write.index
        free = {axis.name: axis for axis in tile.place.free}
        if index[0] != Var(time) or len(state.write.values) != 1 or not state.write.is_scalar:
            return None
        for entry in index[1:]:
            if not ((isinstance(entry, Var) and entry.name in free) or isinstance(entry, Literal)):
                return None
        if any(name not in {entry.name for entry in index[1:] if isinstance(entry, Var)} for name in free):
            return None
        reads = tuple(load for load in loads if load.input == state.write.output)
        lag = lagged(time)
        if any(len(load.index) != len(index) or load.index[0] != lag or not load.is_scalar for load in reads):
            return None
        guards: dict[str, Expr] = {}
        cells: list[int] = []
        for position, own_entry in enumerate(index[1:], start=1):
            resident = True
            for load in reads:
                how = _guard(load.index[position], own_entry)
                if how is None:
                    resident = False
                    break
                if how is not True:
                    if guards.setdefault(load.names[0], how) != how:
                        return None  # one read under two predicates proves nothing about either
            if not resident:
                if not isinstance(own_entry, Var):
                    return None
                cells.append(position)
        if not cells or not _guarded_reads_discarded(tile, guards):
            return None
        seed = _seed_of(tile, {load.names[0] for load in reads}, time)
        if seed is None:
            return None
        cell_axes = tuple(tile.axis_of(index[position].name) for position in cells)
        batch = tuple(axis for axis in tile.place.free if axis.name not in {axis.name for axis in cell_axes})
        tensor = tile.outputs.get(state.write.output)
        width = tensor.dtype.nbytes if tensor is not None else 4
        return cls(state, batch, cell_axes, tuple(cells), seed, prod(axis.extent.as_static() for axis in cell_axes) * width)


def _seed_of(tile: TileOp, reads: frozenset[str] | set[str], time: str) -> str | float | None:
    """What the state holds before the first step. The lift spells every lagged read as a Select
    between the read and its seed on the first step; the seed is a buffer of the state's shape read
    cell by cell, or a constant, and one state has one seed."""
    start = first_step(time)
    sources: dict[str, str | float] = {}
    selects: list[Select] = []
    for site in tile.sites:
        for stmt in site.node.applied.body.iter():
            if isinstance(stmt, Load) and stmt.is_scalar:
                sources[stmt.names[0]] = stmt.input
            elif isinstance(stmt, Let) and isinstance(stmt.value, Literal):
                sources[stmt.name] = float(stmt.value.value)
            elif isinstance(stmt, Select):
                selects.append(stmt)
        for edge in site.node.operands:
            slab = edge.as_slab()
            if slab is not None:
                sources[slab.load.names[0]] = slab.load.input
    seeds: set[str | float] = set()
    seeded: set[str] = set()
    for stmt in selects:
        if len(stmt.branches) != 2 or stmt.branches[0].select != start or stmt.branches[0].value not in reads:
            continue
        source = sources.get(stmt.branches[1].value)
        if source is None:
            return None
        seeds.add(source)
        seeded.add(stmt.branches[0].value)
    if len(seeds) != 1 or seeded != set(reads):
        return None
    return seeds.pop()


def _guarded_reads_discarded(tile: TileOp, guards: dict[str, Expr]) -> bool:
    """Whether every read guarded at a batch position feeds the root's results only through a Select
    branch taken under that same guard — so the read's value is dead exactly when it addressed
    coordinate zero instead of the CTA's own block, and the resident kernel may read its own block
    unconditionally. Taint flows forward through the root's step from the reads (an operand's
    results, or the root's own loads); a Select launders the taint of the branches its guard picks."""
    if not guards:
        return True
    from emmy.compiler.ir.tile.ir import loaded_buffers  # noqa: PLC0415

    root = tile.op
    body = root.applied.body
    results = {*root.applied.results, *(name for spec in tile.output_specs for name in spec.write.values)}
    for guard in set(guards.values()):
        tainted = {name for name, own in guards.items() if own == guard}
        for edge in root.operands:
            if any(load.names[0] in tainted for load in loaded_buffers(edge)):
                tainted.update(edge.exposes)
        for stmt in body:
            if isinstance(stmt, Select):
                if any(branch.value in tainted and branch.select != guard for branch in stmt.branches):
                    tainted.add(stmt.name)
            elif any(name in tainted for name in stmt.deps()):
                tainted.update(stmt.defines())
        if tainted & results:
            return False
    return True


__all__ = ["BlockProgram", "first_step", "lagged"]
