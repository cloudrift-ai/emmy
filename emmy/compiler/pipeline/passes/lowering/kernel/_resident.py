"""Materialize a carried state resident in shared memory: one CTA holds its block across every step.

The launch grid is the block program's batch axes; each CTA declares the block, seeds it, and walks the
sequential axis inside the launch. Every thread owns a strided slice of the block's cells (the kernel's
``WORK`` width is the stride) and evaluates the step for each of them from the block, then the step's
writes go back into the block behind a barrier. At depth one the writes wait for every thread's reads
(two barriers per step); at depth two the step reads one copy of the block and writes the other, one
barrier per step. A per-step output — the state's own snapshot when something outside reads it, and any
other value the step stores — is written to global memory from the same write phase.
"""

from __future__ import annotations

from math import prod

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile
from emmy.compiler.ir.schedule.resident import first_step
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Cond, Let, Load, Select, Stmt, StridedLoop, Write

from ._atom import copy_cell

#: The block's shared-memory name and the thread's lane coordinate within the CTA.
BLOCK = "_state"
LANE = "_rt"


def _decode(flat: Expr, extents: tuple[int, ...], axes: tuple[Axis, ...]) -> dict[str, Expr]:
    """The block's cell coordinates of one flat cell ordinal, row-major over the cell axes."""
    coords: dict[str, Expr] = {}
    stride = prod(extents)
    for axis, extent in zip(axes, extents, strict=True):
        stride //= extent
        term = flat if stride == 1 else BinaryExpr("/", flat, Literal(stride, "int"))
        coords[axis.name] = term if axis is axes[0] else BinaryExpr("%", term, Literal(extent, "int"))
    return coords


def _resident_reads(step: Body, program, depth: int, time: str) -> tuple[Body, dict[str, str]]:
    """The step reading its state from the block, and the names it folded away: every lagged load
    of the state becomes a read of the block at its cell coordinates (a batch position is the CTA's
    own block, so it is dropped), and the Select that picks the seed on the first step folds away,
    because the block already holds the seed — the seed read it guarded is dropped with it, and a
    store of the Select's value stores the read's."""
    state = program.state.write.output
    reads = {stmt.names[0] for stmt in step.iter() if isinstance(stmt, Load) and stmt.input == state}
    slot = Literal(0, "int") if depth == 1 else BinaryExpr("%", Var(time), Literal(2, "int"))
    start = first_step(time)
    alias: dict[str, str] = {}
    seeds: set[str] = set()

    def redirect(stmt: Stmt):
        if isinstance(stmt, Load) and stmt.input == state:
            return Load(name=stmt.names[0], input=BLOCK, index=(slot, *(stmt.index[position] for position in program.cell_positions)))
        if isinstance(stmt, Select) and len(stmt.branches) == 2 and stmt.branches[0].select == start and stmt.branches[0].value in reads:
            alias[stmt.name] = stmt.branches[0].value
            seeds.add(stmt.branches[1].value)
            return None
        return stmt

    redirected = step.map(redirect)
    pruned = redirected.map(lambda stmt: None if isinstance(stmt, (Load, Let)) and set(stmt.defines()) <= seeds else stmt)
    return Body(tuple(stmt.rename(lambda name: alias.get(name, name)) for stmt in pruned)), alias


def factorize_resident(tile) -> Tile:
    """One launch over the batch axes; each CTA holds the block and walks every step."""
    program = tile.block_program
    depth = tile.schedule.kernel.state.depth
    threads = tile.schedule.kernel.work.units[0]
    time = tile.place.serial[0]
    state = program.state.write.output
    tensor = tile.outputs.get(state)
    extents = tuple(axis.extent.as_static() for axis in program.cells)
    count = prod(extents)
    copies = -(-count // threads)
    exact = copies * threads == count
    protected = frozenset({time.name, LANE, *(axis.name for axis in tile.axes)})

    # The step, lowered once with every coordinate bound and its boundary stores taken out: the
    # stored values are known by name, and the write phase below places them itself.
    bound = frozenset({time.name, *(axis.name for axis in tile.place.free)})
    step = tile.op.lower(bound, tile.output_specs, tile.axes).map(lambda stmt: None if isinstance(stmt, Write) else stmt)
    step, alias = _resident_reads(step, program, depth, time.name)
    stored = {spec.write.values[0]: alias.get(spec.write.values[0], spec.write.values[0]) for spec in tile.output_specs}

    def cell(k: int) -> tuple[Expr, dict[str, Expr], str]:
        """The k-th cell this thread owns: its raw ordinal, its coordinates (the last ordinal
        clamped, so a thread past the block's end evaluates a cell it never stores) and the SSA
        suffix of its copy of the step."""
        raw = Var(LANE) if k == 0 else BinaryExpr("+", Var(LANE), Literal(k * threads, "int"))
        flat = raw if exact else TernaryExpr(BinaryExpr("<", raw, Literal(count, "int")), raw, Literal(count - 1, "int"))
        return raw, _decode(flat, extents, program.cells), f"__k{k}"

    def guarded(raw: Expr, stmts: list[Stmt]) -> list[Stmt]:
        return stmts if exact else [Cond(cond=BinaryExpr("<", raw, Literal(count, "int")), body=tuple(stmts))]

    def block_index(slot: Expr, coords: dict[str, Expr]) -> tuple[Expr, ...]:
        return (slot, *(coords[axis.name] for axis in program.cells))

    fill: list[Stmt] = []
    for k in range(copies):
        raw, coords, suffix = cell(k)
        name = f"_seed{suffix}"
        if isinstance(program.seed, str):
            index = tuple(expr.substitute(coords) for expr in program.state.write.index[1:])
            seeded: Stmt = Load(name=name, input=program.seed, index=index)
        else:
            seeded = Let(name=name, value=Literal(program.seed))
        fill.extend(guarded(raw, [seeded, Write(output=BLOCK, index=block_index(Literal(0, "int"), coords), value=name)]))

    write_slot = Literal(0, "int") if depth == 1 else BinaryExpr("%", BinaryExpr("+", Var(time.name), Literal(1, "int")), Literal(2, "int"))
    evaluate: list[Stmt] = []
    commit: list[Stmt] = []
    for k in range(copies):
        raw, coords, suffix = cell(k)
        evaluate.extend(copy_cell(step, Sigma(coords), suffix, protected))
        writes = [Write(output=BLOCK, index=block_index(write_slot, coords), value=f"{stored[program.state.write.values[0]]}{suffix}")]
        writes.extend(
            Write(
                output=spec.write.output,
                index=tuple(expr.substitute(coords) for expr in spec.write.index),
                value=f"{stored[spec.write.values[0]]}{suffix}",
            )
            for spec in tile.output_specs
        )
        commit.extend(guarded(raw, writes))
    if depth == 1:
        body = (*evaluate, Sync(), *commit, Sync())
    else:
        # Reads take one copy of the block and writes fill the other, so a thread's writes never
        # race another's reads within the step; one barrier publishes the step.
        body = (*evaluate, *commit, Sync())
    loop = StridedLoop(axis=time, start=Literal(0, "int"), step=Literal(1, "int"), body=Body(body), unroll=False)
    smem = Smem(name=BLOCK, extents=(depth, *extents), dtype=cuda_name(tensor.dtype) if tensor is not None else "float")
    return Tile(axes=(*program.batch, Axis(LANE, threads)), body=Body((smem, *fill, Sync(), loop)), block_threads=threads)


__all__ = ["BLOCK", "LANE", "factorize_resident"]
