"""Give each value a kernel keeps for itself its storage inside the launch.

A kernel whose term reads a value by index computes that value once and loads cells of it (``TileOp.keeps``): its
body holds the ``Define`` that stores each cell and a ``Load`` per read. The value is a buffer no launch outside
this one sees, so it lives in shared memory, one copy per block. A block here is one cell of the grid: the nest
that defines the value walks the coordinates the grid does not bind, so a block holds the cells its own readers
load and no other block's.

One thread runs a block, so the definition is complete before the first load with no barrier. Spreading the
definition and the sweep that reads it over the threads of a block is the cooperative layout's, not this one's.
"""

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel import KernelOp, Smem, Tile
from emmy.compiler.ir.stmt import Body, Define, Load, Loop, Statement, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]


def _kept(tile: Tile) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Each value ``tile`` defines for itself: the shape one block holds of it, and the positions of its index that
    address that shape. A position a loop of the definition's own nest walks is a dimension of that loop's extent;
    a literal is a dimension that reaches it; any other position reads a coordinate the block binds, the same for
    every cell the block defines, and addresses nothing."""
    out: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def walk(stmts: Body, walked: dict[str, int]) -> None:
        for stmt in stmts:
            if isinstance(stmt, Define):
                shape, positions = [], []
                for position, entry in enumerate(stmt.index):
                    if isinstance(entry, Var) and entry.name in walked:
                        shape.append(walked[entry.name])
                    elif isinstance(entry, Literal):
                        shape.append(int(entry.value) + 1)
                    else:
                        continue
                    positions.append(position)
                out[stmt.name] = (tuple(shape) or (1,), tuple(positions))
            for child in stmt.nested():
                inner = walked
                if isinstance(stmt, Loop):
                    if not stmt.axis.extent.is_static:
                        raise RuleSkipped(f"the kernel keeps a value over {stmt.axis.name}, whose extent the launch binds", reject=True)
                    inner = {**walked, stmt.axis.name: int(stmt.axis.extent.as_static())}
                walk(child, inner)

    walk(tile.body, {})
    return out


def _stored(tile: Tile, kept: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]) -> Tile:
    """``tile`` with each kept value in shared memory: declared ahead of the body, every definition a store into
    it and every load a load of it, at the positions that address one block's cells."""
    held: dict[str, tuple] = {}

    def at(name: str, index: tuple) -> tuple:
        _, positions = kept[name]
        rest = tuple(entry for position, entry in enumerate(index) if position not in positions)
        if held.setdefault(name, rest) != rest:
            raise RuleSkipped(f"a load of {name} reads cells another block defines", reject=True)
        return tuple(index[position] for position in positions) or (Literal(0, "int"),)

    def store(stmt: Statement) -> Statement:
        if isinstance(stmt, Define):
            return Write(output=stmt.name, index=at(stmt.name, stmt.index), values=(stmt.value,))
        if isinstance(stmt, Load) and stmt.input in kept:
            return stmt.replace(index=at(stmt.input, stmt.index))
        return stmt

    # The definitions first: what a block holds is read off where each cell is defined.
    for stmt in tile.body.iter():
        if isinstance(stmt, Define):
            at(stmt.name, stmt.index)
    declared = tuple(Smem(name, shape, F32) for name, (shape, _) in kept.items())
    return tile.replace(body=Body((*declared, *tile.body.map(store))), block_threads=1)


def rewrite(root: Node, ctx=None) -> KernelOp:
    op = root.op
    if not any(isinstance(stmt, Define) for stmt in op.body.iter()):
        raise RuleSkipped("the kernel keeps no value")
    body = []
    for stmt in op.body:
        if isinstance(stmt, Tile) and (kept := _kept(stmt)):
            if stmt.block_threads not in (None, 1) or stmt.aux_threads:
                raise RuleSkipped("a block of this layout holds several cells, and a kept value is one cell's", reject=True)
            stmt = _stored(stmt, kept)
        body.append(stmt)
    lowered = replace(op, body=Body(body), source=op)
    if any(isinstance(stmt, Define) for stmt in lowered.body.iter()):
        raise RuleSkipped("a value is defined outside the kernel's grid", reject=True)
    if ctx is not None and lowered.smem_bytes() > ctx.max_dynamic_smem:
        raise RuleSkipped("the values the kernel keeps exceed shared memory", reject=True)
    return lowered
