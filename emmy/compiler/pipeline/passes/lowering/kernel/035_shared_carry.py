"""Realize a carried state the schedule keeps in shared memory (``STAGE=d1/smem`` at the carrying fold): two
padded shared buffers across the ordered steps, which run inside one launch."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from importlib import import_module
from math import prod

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as dtype_get
from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import BinaryExpression, Builtin, Literal, SimplifyCtx, TernaryExpression, Var
from emmy.compiler.ir.kernel import KernelOp, Smem, Sync, Tile
from emmy.compiler.ir.stmt import (
    Assign,
    Body,
    Cond,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    Write,
    dtype_promote,
    free_names,
)
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.types import value_dtypes
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.tile._fromloop import carried_program, conjuncts, seed_index, shared_state

PATTERN = [Pattern("root", KernelOp)]
_resolve = import_module("emmy.compiler.pipeline.passes.lowering.kernel.045_merge_select_loads")._resolve
_materialize = import_module("emmy.compiler.pipeline.passes.lowering.kernel.010_materialize")


def _resident(op, program, retained=frozenset()):
    loop, carry, batch = program
    if len(op.serial) != 1 or op.serial[0] != loop.axis or len(op.body) != 1 or not isinstance(op.body[0], Tile):
        return None
    tile = op.body[0]
    if tile.block_threads is None or tile.aux_threads or tile.raster_axes is not None or any(not a.extent.is_static for a in tile.axes):
        return None
    # The existing binder puts its cooperative inventory at the end of the flat decode.
    threads, split = 1, len(tile.axes)
    while threads < tile.block_threads and split:
        split -= 1
        threads *= tile.axes[split].extent.as_static()
    if threads != tile.block_threads or tuple(a.name for a in tile.axes[: len(batch)]) != batch:
        return None
    grid = tile.axes[len(batch) : split]
    if not grid:
        return None
    state = shared_state(op, program)
    if state is None:
        return None
    port, shape = state
    written = {s.output for s in loop.body.iter() if isinstance(s, Write)}
    tensor = op.outputs[port]
    kept = tuple(dim for dim, n in enumerate(tensor.shape) if n != 1)  # the dims a size-one coordinate pins to 0 hold no cell
    shared = "_carry"
    parity = Var(loop.axis.name) % Literal(2, "int")
    before = BinaryExpression("<", Literal(0, "int"), Var(loop.axis.name))  # the first-step predicate, in normal form
    stores = [s for s in tile.body.iter() if isinstance(s, Write) and s.output == port]
    if not stores or any(s.atomic or not s.is_scalar for s in stores):
        return None

    # A store that does not read the step keeps the state the loop ends with: it is stored once, after the last step.
    snapshots = tuple(s for s in loop.body.iter() if isinstance(s, Write))
    final = tuple(s for s in snapshots if Var(loop.axis.name) not in s.index)
    if any(not _stores_state(op, loop, carry, s, tensor.dtype) for s in final):
        return None

    def cells_of(index):
        return tuple(index[dim] for dim in kept[1 + len(batch) :])

    # The shared buffer is seeded ahead of the steps, so the first step's choice of the seed is the buffer's own read:
    # of a seed tensor's load, or of the constant a seed is.
    constants = {s.name: s for s in tile.body.iter() if isinstance(s, Let)} if not isinstance(carry.seed, str) else {}

    def rewrite(body):
        defs = {s.name: s for s in body if isinstance(s, Load)}
        uses = Counter(name for s in body.iter() for name in s.deps())
        discarded = set()
        out = []
        for stmt in body:
            if isinstance(stmt, Select) and len(stmt.branches) == 2 and stmt.branches[0].select == before:
                held = defs.get(stmt.branches[0].value)
                seed = defs.get(stmt.branches[1].value)
                constant = constants.get(stmt.branches[1].value)
                if held is None or held.input != port:
                    pass
                elif constant is not None and constant.value == Literal(float(carry.seed)):
                    stmt = Assign(stmt.name, Var(held.name), dtype=tensor.dtype)
                elif seed is not None and seed.input == carry.seed:
                    promoted = dtype_get(dtype_promote("add", [held.dtype.name, seed.dtype.name]))
                    if promoted == tensor.dtype:
                        if uses[seed.name] == 1:
                            discarded.add(seed.name)
                        stmt = Assign(stmt.name, Var(held.name), dtype=promoted)
            if isinstance(stmt, Load) and stmt.input == port:
                stmt = stmt.replace(input=shared, index=(parity, *cells_of(stmt.index)), dtype=tensor.dtype)
            elif isinstance(stmt, Write) and any(stmt.output == s.output for s in final):
                continue
            elif isinstance(stmt, Write) and stmt.output == port:
                if port in retained:
                    out.append(stmt)
                stmt = stmt.replace(output=shared, index=(Literal(1, "int") - parity, *cells_of(stmt.index)), value_dtype=tensor.dtype)
            elif stmt.nested():
                stmt = stmt.with_bodies(tuple(rewrite(child) for child in stmt.nested()))
            out.append(stmt)
        return Body(s for s in out if not isinstance(s, Load) or s.name not in discarded)

    step = rewrite(tile.body)
    snapshots = tuple(s for s in snapshots if s not in final)
    if port in retained and port not in written:
        snapshots += (Write(port, (Var(loop.axis.name), *carry.index), (carry.value,)),)
    snapshot = _parallel_copy(op, program, step, shared, parity, shape, batch, tile.block_threads, tensor.dtype, snapshots)
    if snapshot is not None:
        copying, step, predicate, lagging = snapshot
    else:
        copying, predicate, lagging = (), None, False
    step = Body((*step, Sync()))
    for axis in reversed(grid):
        start, end = Literal(0, "int"), axis.extent_expression()
        if predicate is not None:
            for condition in conjuncts(predicate):
                if not isinstance(condition, BinaryExpression):
                    continue
                if condition.op == "<" and condition.left == Var(axis.name) and condition.right.free_vars() <= {*batch, loop.axis.name}:
                    end = TernaryExpression(condition.right.lt(end), condition.right, end)
                if condition.op == "<=" and condition.right == Var(axis.name) and condition.left.free_vars() <= {*batch, loop.axis.name}:
                    start = TernaryExpression(start.lt(condition.left), condition.left, start)
        step = Body((Loop.strided(axis, start, Literal(1, "int"), step, end=end),))
    step = Body((Loop(loop.axis, (*copying, *step, Sync())),))
    suffix = _cells(Var("_carry_cell"), shape)
    initial = (*(Var(name) for name in batch), *suffix)
    if isinstance(carry.seed, str):
        seed = Load(("_carry_seed",), carry.seed, seed_index(initial, op.inputs[carry.seed].shape), dtype=op.inputs[carry.seed].dtype)
    else:
        seed = Let("_carry_seed", Literal(float(carry.seed)))
    # A copy that follows the changed cells finds the unchanged ones in both buffers.
    seeded = tuple(Write(shared, (Literal(slot, "int"), *suffix), ("_carry_seed",), value_dtype=seed.dtype) for slot in range(1 + lagging))
    threads = Literal(tile.block_threads, "int")
    initialize = Loop.strided(Axis("_carry_cell", prod(shape)), Builtin("thread_idx.x"), threads, (seed, *seeded))
    closing = ()
    if final:
        cells = _cells(Var("_carry_last"), shape)
        ended = Load(("_carry_end",), shared, (Literal(loop.axis.extent.as_static() % 2, "int"), *cells), dtype=tensor.dtype)
        coordinates = (*(Var(name) for name in batch), *cells)
        stores = tuple(
            Write(s.output, seed_index(coordinates, op.outputs[s.output].shape), ("_carry_end",), value_dtype=tensor.dtype) for s in final
        )
        closing = (Loop.strided(Axis("_carry_last", prod(shape)), Builtin("thread_idx.x"), threads, (ended, *stores)),)
    # One padding column per row keeps column-strided reads of the state off one bank.
    body = (Smem(shared, (2, *shape[:-1], shape[-1] + 1), tensor.dtype), initialize, Sync(), *step, *closing)
    return replace(op, serial=(), body=Body((tile.replace(axes=(*tile.axes[: len(batch)], *tile.axes[split:]), body=Body(body)),)))


def _cells(cell, shape):
    """The coordinates of flat ``cell`` in a state of ``shape``."""
    out, stride = [], prod(shape)
    for extent in shape:
        stride //= extent
        out.append(BinaryExpression("/", cell, Literal(stride, "int")) % Literal(extent, "int"))
    return out


def _stores_state(op, loop, carry, store, dtype) -> bool:
    """Whether ``store`` writes the state's next value itself, through copies that keep its ``dtype``."""
    defs = {s.name: s for s in loop.body.iter() for _ in s.defines() if hasattr(s, "name")}
    value = store.value
    while isinstance(defs.get(value), Assign) and isinstance(defs[value].value, Var):
        if defs[value].dtype not in (None, dtype):
            return False
        value = defs[value].value.name
    return value == carry.value and op.outputs[store.output].dtype == dtype


def _guard_stores(body: Body, types=None) -> Body:
    """Sink each coordinate select's private scalar cones and their stores into the branch that
    consumes them, so a branch that only copies the cell can be dropped once the whole CTA copies it.

    A select's dependent assignments and scalar stores must form one contiguous continuation.
    Other readers, nested effects and intervening writes to a cone's input keep the stream unchanged.
    """
    if types is None:
        types = value_dtypes(body)
    stmts = [stmt.with_bodies(tuple(_guard_stores(b, types) for b in stmt.nested())) if stmt.nested() else stmt for stmt in body]
    uses = Counter(name for stmt in Body(stmts).iter() for name in free_names(stmt))
    position = 0
    while position < len(stmts):
        select = stmts[position]
        if not isinstance(select, Select) or len(select.branches) != 2:
            position += 1
            continue
        end = position + 1
        names = {select.name}
        continuation = []
        while end < len(stmts) and isinstance(stmts[end], Assign) and names & set(stmts[end].deps()):
            continuation.append(stmts[end])
            names.add(stmts[end].name)
            end += 1
        start = end
        while end < len(stmts) and isinstance(stmts[end], Write) and stmts[end].is_scalar and stmts[end].value in names:
            end += 1
        stores = stmts[start:end]
        readers = Counter(name for stmt in (*continuation, *stores) for name in free_names(stmt))
        if (
            not stores
            or any(uses[name] != readers[name] for name in names)
            or any(s.atomic or any(names & e.free_vars() for e in s.index) for s in stores)
        ):
            position += 1
            continue
        defs = {name: i for i, stmt in enumerate(stmts[:position]) for name in stmt.defines()}

        def cone(name, found, defs=defs, stmts=stmts):
            i = defs.get(name)
            if i is None or uses[name] != 1 or not isinstance(stmts[i], (Assign, Let, Load, Select)):
                return
            if isinstance(stmts[i], Load) and (not stmts[i].is_scalar or stmts[i].carried):
                return
            found.add(i)
            for arg in free_names(stmts[i]):
                cone(arg, found)

        branches = []
        for branch in select.branches:
            found = set()
            cone(branch.value, found)
            branches.append(found)
        moved = branches[0] | branches[1]
        reads = {stmts[i].input for i in moved if isinstance(stmts[i], Load)}
        if (
            not moved
            or branches[0] & branches[1]
            or any(
                not isinstance(stmt, (Assign, Let, Load, Select, Write)) or getattr(stmt, "output", None) in reads
                for i, stmt in enumerate(stmts[:position])
                if min(moved) <= i and i not in moved
            )
        ):
            position += 1
            continue
        cond = select.branches[0].select.simplify(SimplifyCtx.empty())
        guarded = []
        for truth, (branch, found) in zip((True, False), zip(select.branches, branches, strict=True), strict=True):
            chain = []
            for i in sorted(found):
                stmt = stmts[i]
                if isinstance(stmt, Load):
                    stmt = stmt.replace(index=tuple(_resolve(e.simplify(SimplifyCtx.empty()), cond, truth) for e in stmt.index))
                chain.append(stmt)
            chain.append(Assign(select.name, Var(branch.value), dtype=types.get(select.name, F32)))
            guarded.append(_guard_stores(Body((*chain, *continuation, *stores)), types))
        replacement = Cond(cond, guarded[0], guarded[1])
        stmts = [stmt for i, stmt in enumerate(stmts) if i not in moved and not position <= i < end]
        position -= len(moved)
        stmts.insert(position, replacement)
        position += 1
    return Body(stmts)


def _parallel_copy(op, program, step, shared, parity, shape, batch, threads, dtype, outputs):
    """Copy unchanged cells with the whole CTA, then overwrite only the selected updates."""
    loop, carry, _ = program
    defs = {s.name: s for s in loop.body.iter() for _ in s.defines() if hasattr(s, "name")}
    selected = defs.get(carry.value)
    if not isinstance(selected, Select) or len(selected.branches) != 2:
        return None
    previous = defs.get(selected.branches[1].value)
    if not isinstance(previous, Pre) or previous.carrier != carry.name or previous.index != carry.index:
        return None
    if any(not _stores_state(op, loop, carry, store, dtype) for store in outputs):
        return None

    definitions = {name: s for s in step.iter() for name in s.defines()}

    def copy_branch(body):
        values = {}
        stores = []

        def address(name):
            if name in values:
                return values[name]
            stmt = definitions.get(name)
            if isinstance(stmt, Load) and stmt.input == shared:
                return stmt.index
            if isinstance(stmt, Assign) and isinstance(stmt.value, Var) and stmt.dtype in (None, dtype):
                return address(stmt.value.name)
            return None

        for s in body:
            if isinstance(s, Load) and s.input == shared:
                values[s.name] = s.index
            elif isinstance(s, Assign) and isinstance(s.value, Var) and s.dtype in (None, dtype) and address(s.value.name) is not None:
                values[s.name] = address(s.value.name)
            elif isinstance(s, Write) and address(s.value) is not None:
                stores.append(s)
                if s.output == shared and s.index[1:] != address(s.value)[1:]:
                    return False
            else:
                return False
        return any(s.output == shared for s in stores)

    removed = False

    def omit(body):
        nonlocal removed
        out = []
        for stmt in body:
            if isinstance(stmt, Cond) and copy_branch(stmt.else_body):
                stmt = stmt.replace(else_body=Body())
                removed = True
            if stmt.nested():
                stmt = stmt.with_bodies(tuple(omit(child) for child in stmt.nested()))
            out.append(stmt)
        return Body(out)

    changed = omit(_guard_stores(step))
    if not removed:
        return None
    indices = _cells(Var("_carry_copy"), shape)
    load = Load(("_carry_old",), shared, (parity, *indices), dtype=dtype)
    stores = [Write(shared, (Literal(1, "int") - parity, *indices), ("_carry_old",), value_dtype=dtype)]
    coordinates = (*(Var(name) for name in batch), *indices)
    for store in outputs:
        index = (Var(loop.axis.name), *seed_index(coordinates, op.outputs[store.output].shape[1:]))
        stores.append(Write(store.output, index, ("_carry_old",), value_dtype=dtype))
    predicate = selected.branches[0].select
    start, end = Builtin("thread_idx.x"), None
    # With no snapshot to store, the buffer a step writes differs from the state it reads only in the cells the
    # step before changed: where the predicate bounds those along the state's leading cell, only they are copied.
    rows = _bounds(predicate, next(e.name for e in carry.index if isinstance(e, Var) and e.name not in batch), {*batch, loop.axis.name})
    lagging = not outputs and None not in rows
    if lagging:
        before = {loop.axis.name: Var(loop.axis.name) - Literal(1, "int")}
        low, high = (bound.substitute(before) for bound in rows)
        zero, extent, width = Literal(0, "int"), Literal(shape[0], "int"), Literal(prod(shape[1:]), "int")
        start = start + TernaryExpression(zero.lt(low), low, zero) * width
        end = TernaryExpression(high.lt(extent), high, extent) * width
    copying = Loop.strided(Axis("_carry_copy", prod(shape)), start, Literal(threads, "int"), (load, *stores), end=end)
    return (copying, Sync()), changed, predicate, lagging


def _bounds(predicate, name, free):
    """``(low, high)`` with ``low <= name < high`` wherever ``predicate`` holds, each over ``free`` names alone
    or ``None`` where no conjunct bounds that side."""
    low = high = None
    for condition in conjuncts(predicate):
        if not isinstance(condition, BinaryExpression):
            continue
        if condition.op == "<" and condition.left == Var(name) and condition.right.free_vars() <= free:
            high = condition.right
        if condition.op == "<=" and condition.right == Var(name) and condition.left.free_vars() <= free:
            low = condition.left
    return low, high


def rewrite(match: Match, root: Node, ctx=None):
    op = root.op
    scheduled = next((s for s in op.source_chain() if isinstance(s, TileOp) and s.schedule is not None), None)
    state = getattr(getattr(scheduled.schedule, "kernel", None), "state", None) if scheduled is not None else None
    if not op.serial or state is None or state.is_direct:
        raise RuleSkipped("the schedule keeps this kernel's state in global memory, or it carries none")
    program = carried_program(op)
    retained = frozenset(t.name for t in root.outputs if t.name in match.graph.outputs or match.graph.buffer_users(t.name))
    selected = _resident(op, program, retained) if program is not None else None
    # The schedule offers shared storage on what it can prove before lowering; what only the lowered kernel
    # shows (its stores, the room beside the schedule's own scratch) declines the row here, and the pick moves on.
    if selected is None:
        raise RuleSkipped("this kernel has no CTA-owned state the row's shared storage can hold", reject=True)
    if selected.smem_bytes() > ctx.max_dynamic_smem:
        raise RuleSkipped("the carried state and the schedule's scratch exceed shared memory", reject=True)
    return _materialize._drop_private_ports(match, root, replace(selected, source=op, knobs=op.knobs), "shared")
