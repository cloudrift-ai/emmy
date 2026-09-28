"""Materialize a carried state held by one CTA: the block in shared memory across every step.

The launch grid is the block program's batch axes; each CTA declares the block, seeds it, and walks the
sequential axis inside the launch. Every thread strides over the block's cells (the kernel's ``WORK`` width
is the stride) and evaluates the step for each cell it owns from the block, keeping the results in a
register array; after a barrier the writes go back into the block, and a second barrier publishes the
step. A cell outside the step domain is neither evaluated nor written: the block already holds its
value. The per-step outputs the graph reads are written from the write phase — the state's own snapshot
and any other stored value at the cells the step defines, and at the cells it skips the block's value,
for the buffers something outside reads.
"""

from __future__ import annotations

from math import prod

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import RegFragment, Smem, Sync, Tile
from emmy.compiler.ir.schedule.resident import first_step
from emmy.compiler.ir.stmt import Body, Cond, Let, Load, Select, Stmt, StridedLoop, Write

#: The block's shared-memory name, the thread's lane within the CTA, and the flat cell ordinal it walks.
BLOCK = "_state"
LANE = "_rt"
CELL = "_ci"


def _decode(extents: tuple[int, ...], axes: tuple[Axis, ...]) -> list[Stmt]:
    """The block's cell coordinates of the flat cell ordinal, row-major over the cell axes, bound
    under the axes' own names so the step and the boundary stores read them verbatim."""
    out: list[Stmt] = []
    stride = prod(extents)
    for axis, extent in zip(axes, extents, strict=True):
        stride //= extent
        term: Expr = Var(CELL) if stride == 1 else BinaryExpr("/", Var(CELL), Literal(stride, "int"))
        out.append(Let(name=axis.name, value=term if axis is axes[0] else BinaryExpr("%", term, Literal(extent, "int"))))
    return out


def _resident_reads(step: Body, program, time: str) -> tuple[Body, dict[str, str]]:
    """The step reading its state from the block, and the names it folded away: every lagged load
    of the state becomes a read of the block at its cell coordinates (a batch position is the CTA's
    own block, so it is dropped), and the Select that picks the seed on the first step folds away,
    because the block already holds the seed — the seed read it guarded is dropped with it, and a
    store of the Select's value stores the read's."""
    state = program.state.write.output
    reads = {stmt.names[0] for stmt in step.iter() if isinstance(stmt, Load) and stmt.input == state}
    start = first_step(time)
    alias: dict[str, str] = {}
    seeds: set[str] = set()

    def redirect(stmt: Stmt):
        if isinstance(stmt, Load) and stmt.input == state:
            return Load(name=stmt.names[0], input=BLOCK, index=tuple(stmt.index[position] for position in program.cell_positions))
        if isinstance(stmt, Select) and len(stmt.branches) == 2 and stmt.branches[0].select == start and stmt.branches[0].value in reads:
            alias[stmt.name] = stmt.branches[0].value
            seeds.add(stmt.branches[1].value)
            return None
        return stmt

    redirected = step.map(redirect)
    pruned = redirected.map(lambda stmt: None if isinstance(stmt, (Load, Let)) and set(stmt.defines()) <= seeds else stmt)
    return Body(tuple(stmt.rename(lambda name: alias.get(name, name)) for stmt in pruned)), alias


def factorize_resident(tile, *, snapshots: dict[str, frozenset[int] | None]) -> Tile:
    """One launch over the batch axes; each CTA holds the block and walks every step. ``snapshots``
    maps each stored buffer something outside the kernel reads to the steps it is read at (``None``
    for every step): those steps' copies are written, complete at the cells the step skips too, and a
    buffer nobody reads is not written at all."""
    program = tile.block_program
    threads = tile.schedule.kernel.work.units[0]
    time = tile.place.serial[0]
    state = program.state.write.output
    tensor = tile.outputs.get(state)
    extents = tuple(axis.extent.as_static() for axis in program.cells)
    count = prod(extents)
    slots = -(-count // threads)
    cells = Axis(CELL, count)
    coords = _decode(extents, program.cells)
    block = tuple(Var(axis.name) for axis in program.cells)

    def stride(body: list[Stmt]) -> StridedLoop:
        return StridedLoop(axis=cells, start=Var(LANE), step=Literal(threads, "int"), body=Body((*coords, *body)), unroll=False)

    # The step, lowered once with every coordinate bound and its boundary stores taken out: the
    # stored values are known by name, and the write phase below places them itself.
    bound = frozenset({time.name, *(axis.name for axis in tile.place.free)})
    step = tile.op.lower(bound, tile.output_specs, tile.axes).map(lambda stmt: None if isinstance(stmt, Write) else stmt)
    step, alias = _resident_reads(step, program, time.name)
    values = tuple(dict.fromkeys(alias.get(spec.write.values[0], spec.write.values[0]) for spec in tile.output_specs))
    held = {value: f"_next{ordinal}" for ordinal, value in enumerate(values)}
    stored = {spec.write.output: alias.get(spec.write.values[0], spec.write.values[0]) for spec in tile.output_specs}
    slot = (BinaryExpr("/", BinaryExpr("-", Var(CELL), Var(LANE)), Literal(threads, "int")),)

    if isinstance(program.seed, str):
        seeded: Stmt = Load(name="_seed", input=program.seed, index=tuple(program.state.write.index[1:]))
    else:
        seeded = Let(name="_seed", value=Literal(program.seed))
    fill = stride([seeded, Write(output=BLOCK, index=block, value="_seed")])

    def snapshot(value) -> list[Stmt]:
        """The per-step copies something outside reads, at the steps it reads them: ``value`` names
        the SSA value a buffer's store takes, or the one name every store takes."""
        out: list[Stmt] = []
        for spec in tile.output_specs:
            steps = snapshots.get(spec.write.output, frozenset())
            if steps is not None and not steps:
                continue  # nothing reads this buffer: its port goes, and so do its writes
            write = Write(output=spec.write.output, index=spec.write.index, value=value(spec) if callable(value) else value)
            if steps is None:
                out.append(write)
                continue
            clauses = [BinaryExpr("==", Var(time.name), Literal(step, "int")) for step in sorted(steps)]
            cond = clauses[0]
            for clause in clauses[1:]:
                cond = BinaryExpr("||", cond, clause)
            out.append(Cond(cond=cond, body=(write,)))
        return out

    evaluate = [*step, *(Write(output=name, index=slot, value=value) for value, name in held.items())]
    commit: list[Stmt] = [Load(name=f"{name}_v", input=name, index=slot) for name in held.values()]
    commit.append(Write(output=BLOCK, index=block, value=f"{held[stored[state]]}_v"))
    commit.extend(snapshot(lambda spec: f"{held[stored[spec.write.output]]}_v"))
    if program.domain is None:
        phases = (stride(evaluate), Sync(), stride(commit), Sync())
    else:
        # A cell outside the domain keeps its value: nothing to evaluate, nothing to write into the
        # block; the buffers read outside still need that value in the snapshots of the steps they read.
        copies = snapshot("_kept")
        kept = (Load(name="_kept", input=BLOCK, index=block), *copies) if copies else ()
        phases = (
            stride([Cond(cond=program.domain, body=tuple(evaluate))]),
            Sync(),
            stride([Cond(cond=program.domain, body=tuple(commit), else_body=kept)]),
            Sync(),
        )
    loop = StridedLoop(axis=time, start=Literal(0, "int"), step=Literal(1, "int"), body=Body(phases), unroll=False)
    smem = Smem(name=BLOCK, extents=extents, dtype=cuda_name(tensor.dtype) if tensor is not None else "float")
    arrays = tuple(RegFragment(name=name, role="c", shape=(1, 1, 1), dtype=F32, nregs=slots) for name in held.values())
    return Tile(axes=(*program.batch, Axis(LANE, threads)), body=Body((smem, *arrays, fill, Sync(), loop)), block_threads=threads)


__all__ = ["BLOCK", "CELL", "LANE", "factorize_resident"]
