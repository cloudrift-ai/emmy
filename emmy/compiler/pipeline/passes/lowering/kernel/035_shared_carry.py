"""Keep an independently owned carried state in two shared buffers across ordered steps."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from importlib import import_module
from math import prod

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as dtype_get
from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.kernel import KernelOp, Smem, Sync, Tile
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Cond, Let, Load, Loop, Pre, Select, StridedLoop, Write
from emmy.compiler.ir.stmt.base import dtype_promote
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.passes.tile._fromloop import seed_index
from emmy.compiler.pipeline.search.space import SHARED_CARRY

PATTERN = [Pattern("root", KernelOp)]
_resolve = import_module("emmy.compiler.pipeline.passes.lowering.kernel.045_merge_select_loads")._resolve
_stamp = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types")
_materialize = import_module("emmy.compiler.pipeline.passes.lowering.kernel.010_materialize")


def _conjuncts(expr):
    return _conjuncts(expr.left) | _conjuncts(expr.right) if isinstance(expr, BinaryExpr) and expr.op == "&&" else frozenset((expr,))


def _owned(body, carry):
    """The leading cells every live state read addresses in its own independent batch.

    Coordinate selects can clamp a discarded branch to another batch. Each use contributes
    its predicate separately; a coordinate is owned only when every consuming path proves it.
    """
    demand = {}

    def need(name, predicates):
        demand.setdefault(name, set()).update(predicates)

    for stmt in reversed(tuple(body.iter())):
        wanted = demand.get(getattr(stmt, "name", ""), {frozenset()})
        if isinstance(stmt, Select):
            remaining = wanted
            for i, branch in enumerate(stmt.branches):
                if i == len(stmt.branches) - 1:
                    need(branch.value, remaining)
                else:
                    need(branch.value, {path | _conjuncts(branch.select) for path in remaining})
                    remaining = {path | {BinaryExpr("==", branch.select, Literal(0, "int"))} for path in remaining}
        elif isinstance(stmt, (Assign, Accum)):
            for name in stmt.deps():
                if name != stmt.name:
                    need(name, wanted)
        elif isinstance(stmt, (Carry, Write)):
            need(stmt.value, {frozenset()})
    reads = tuple(stmt for stmt in body.iter() if isinstance(stmt, Pre))
    owned = []
    for dim, cell in enumerate(carry.index):
        if not isinstance(cell, Var):
            continue
        for read in reads:
            if read.carrier != carry.name:
                return ()
            coordinate = read.index[dim]
            for path in demand.get(read.name, {frozenset()}):
                value = coordinate
                while isinstance(value, TernaryExpr) and _conjuncts(value.cond) <= path:
                    value = value.if_true
                if value != cell:
                    return tuple(owned)
        owned.append(cell.name)
    return tuple(owned)


def _program(op):
    """Recover the carried loop whose materialization produced this serial kernel."""
    source = op.source
    while source is not None:
        body = getattr(source, "loop_body", None)
        if body is not None:
            loops = [s for s in body.iter() if isinstance(s, Loop) and s.carries]
            if loops:
                if len(loops) != 1:
                    return None
                loop = loops[0]
                carries = tuple(s for s in loop.body.iter() if isinstance(s, Carry))
                if len(carries) != 1 or any(
                    not isinstance(s, (Loop, Let, Load, Pre, Assign, Select, Accum, Carry, Write)) for s in loop.body.iter()
                ):
                    return None
                carry = carries[0]
                if any(s.atomic for s in loop.body.iter() if isinstance(s, Write)):
                    return None
                if any(
                    tuple(e for e, n in zip(s.index, source.outputs[s.output].shape, strict=True) if n != 1)
                    != (Var(loop.axis.name), *(e for e in carry.index if isinstance(e, Var)))
                    for s in loop.body.iter()
                    if isinstance(s, Write)
                ):
                    return None
                return loop, carry, _owned(loop.body, carry)
        source = source.source
    return None


def _resident(op, program, retained=frozenset(), padding=0):
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
    ports = [name for name in op.outputs if name not in {s.output for s in program[0].body.iter() if isinstance(s, Write)}]
    if len(ports) != 1:
        return None
    port = ports[0]
    tensor = op.outputs[port]
    dimensions = tuple(tensor.shape[1 + len(batch) :])
    if not dimensions or any(not n.is_static for n in dimensions):
        return None
    shape = tuple(n.as_static() for n in dimensions)
    shared = "_carry"
    parity = Var(loop.axis.name) % Literal(2, "int")
    before = BinaryExpr(">", Var(loop.axis.name), Literal(0, "int"))
    stores = [s for s in tile.body.iter() if isinstance(s, Write) and s.output == port]
    if not stores or any(s.atomic or not s.is_scalar for s in stores):
        return None

    def rewrite(body):
        defs = {s.name: s for s in body if isinstance(s, Load)}
        uses = Counter(name for s in body.iter() for name in s.deps())
        discarded = set()
        out = []
        for stmt in body:
            if isinstance(stmt, Select) and len(stmt.branches) == 2 and stmt.branches[0].select == before:
                held = defs.get(stmt.branches[0].value)
                seed = defs.get(stmt.branches[1].value)
                if held is not None and held.input == port and seed is not None and seed.input == carry.seed:
                    promoted = dtype_get(dtype_promote("add", [held.dtype.name, seed.dtype.name]))
                    if promoted == tensor.dtype:
                        if uses[seed.name] == 1:
                            discarded.add(seed.name)
                        stmt = Assign(stmt.name, "copy", (held.name,), dtype=promoted)
            if isinstance(stmt, Load) and stmt.input == port:
                stmt = replace(stmt, input=shared, index=(parity, *stmt.index[1 + len(batch) :]), dtype=tensor.dtype)
            elif isinstance(stmt, Write) and stmt.output == port:
                if port in retained:
                    out.append(stmt)
                stmt = replace(
                    stmt, output=shared, index=(Literal(1, "int") - parity, *stmt.index[1 + len(batch) :]), value_dtype=tensor.dtype
                )
            elif stmt.nested():
                stmt = stmt.with_bodies(tuple(rewrite(child) for child in stmt.nested()))
            out.append(stmt)
        return Body(s for s in out if not isinstance(s, Load) or s.name not in discarded)

    step = rewrite(tile.body)
    snapshots = tuple(s for s in loop.body.iter() if isinstance(s, Write))
    if port in retained:
        snapshots += (Write(port, (Var(loop.axis.name), *carry.index), carry.value),)
    snapshot = _parallel_copy(op, program, step, shared, parity, shape, batch, tile.block_threads, tensor.dtype, snapshots)
    if snapshot is not None:
        copying, step, predicate = snapshot
    else:
        copying, predicate = (), None
    step = Body((*step, Sync()))
    for axis in reversed(grid):
        start, end = Literal(0, "int"), axis.extent_expr()
        if predicate is not None:
            for condition in _conjuncts(predicate):
                if not isinstance(condition, BinaryExpr):
                    continue
                if condition.op == "<" and condition.left == Var(axis.name) and condition.right.free_vars() <= {*batch, loop.axis.name}:
                    end = TernaryExpr(condition.right.lt(end), condition.right, end)
                if condition.op == "<=" and condition.right == Var(axis.name) and condition.left.free_vars() <= {*batch, loop.axis.name}:
                    start = TernaryExpr(start.lt(condition.left), condition.left, start)
        step = Body((StridedLoop(axis, start, Literal(1, "int"), step, end=end),))
    step = Body((Loop(loop.axis, (*copying, *step, Sync())),))
    cell = Var("_carry_cell")
    suffix = []
    stride = prod(shape)
    for extent in shape:
        stride //= extent
        suffix.append(BinaryExpr("/", cell, Literal(stride, "int")) % Literal(extent, "int"))
    initial = (*(Var(name) for name in batch), *suffix)
    if isinstance(carry.seed, str):
        seed = Load("_carry_seed", carry.seed, seed_index(initial, op.inputs[carry.seed].shape), dtype=op.inputs[carry.seed].dtype)
    else:
        seed = Let("_carry_seed", Literal(float(carry.seed)))
    initialize = StridedLoop(
        Axis("_carry_cell", prod(shape)),
        Builtin("thread_idx.x"),
        Literal(tile.block_threads, "int"),
        (seed, Write(shared, (Literal(0, "int"), *suffix), "_carry_seed", value_dtype=seed.dtype)),
    )
    body = (Smem(shared, (2, *shape[:-1], shape[-1] + padding), cuda_name(tensor.dtype)), initialize, Sync(), *step)
    return replace(op, serial=(), body=Body((replace(tile, axes=(*tile.axes[: len(batch)], *tile.axes[split:]), body=Body(body)),)))


def _guard_stores(body: Body, types=None) -> Body:
    """Sink each coordinate select's private scalar cones and their stores into the branch that
    consumes them, so a branch that only copies the cell can be dropped once the whole CTA copies it.

    A select's dependent assignments and scalar stores must form one contiguous continuation.
    Other readers, nested effects and intervening writes to a cone's input keep the stream unchanged.
    """
    if types is None:
        ctx = _stamp._StampCtx({})
        _stamp._seed_explicit_dtypes(body, ctx)
        _stamp._stamp_body(body, ctx)
        types = ctx.ssa_dtypes
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
                    stmt = replace(stmt, index=tuple(_resolve(e.simplify(SimplifyCtx.empty()), cond, truth) for e in stmt.index))
                chain.append(stmt)
            chain.append(Assign(select.name, "copy", (branch.value,), dtype=types.get(select.name, F32)))
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
    for store in outputs:
        value = store.value
        while isinstance(defs.get(value), Assign) and defs[value].op.name == "copy":
            if defs[value].dtype not in (None, dtype):
                return None
            value = defs[value].args[0]
        if value != carry.value or op.outputs[store.output].dtype != dtype:
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
            if isinstance(stmt, Assign) and stmt.op.name == "copy" and stmt.dtype in (None, dtype):
                return address(stmt.args[0])
            return None

        for s in body:
            if isinstance(s, Load) and s.input == shared:
                values[s.name] = s.index
            elif isinstance(s, Assign) and s.op.name == "copy" and s.dtype in (None, dtype) and address(s.args[0]) is not None:
                values[s.name] = address(s.args[0])
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
                stmt = replace(stmt, else_body=Body())
                removed = True
            if stmt.nested():
                stmt = stmt.with_bodies(tuple(omit(child) for child in stmt.nested()))
            out.append(stmt)
        return Body(out)

    changed = omit(_guard_stores(step))
    if not removed:
        return None
    cell = Var("_carry_copy")
    indices = []
    stride = prod(shape)
    for extent in shape:
        stride //= extent
        indices.append(BinaryExpr("/", cell, Literal(stride, "int")) % Literal(extent, "int"))
    load = Load("_carry_old", shared, (parity, *indices), dtype=dtype)
    stores = [Write(shared, (Literal(1, "int") - parity, *indices), "_carry_old", value_dtype=dtype)]
    coordinates = (*(Var(name) for name in batch), *indices)
    for store in outputs:
        index = (Var(loop.axis.name), *seed_index(coordinates, op.outputs[store.output].shape[1:]))
        stores.append(Write(store.output, index, "_carry_old", value_dtype=dtype))
    copying = StridedLoop(Axis("_carry_copy", prod(shape)), Builtin("thread_idx.x"), Literal(threads, "int"), (load, *stores))
    return (copying, Sync()), changed, selected.branches[0].select


def rewrite(match: Match, root: Node, ctx=None):
    op = root.op
    if not op.serial or SHARED_CARRY.name in op.knobs:
        raise RuleSkipped("shared carry already decided or no serial state")
    program = _program(op)
    retained = frozenset(t.name for t in root.outputs if t.name in match.graph.outputs or match.graph.buffer_users(t.name))
    if program is None or _resident(op, program, retained) is None:
        raise RuleSkipped("this kernel has no CTA-owned state that fits shared memory")
    variants = []
    for enabled in SHARED_CARRY.narrow((0, 1, 2)):
        selected = _resident(op, program, retained, padding=enabled - 1) if enabled else op
        if selected.smem_bytes() > ctx.max_dynamic_smem:
            continue
        selected = replace(selected, source=op, knobs={**op.knobs, SHARED_CARRY.name: enabled})
        variants.append(
            DeferredFork(lambda selected=selected: _materialize._drop_private_ports(match, root, selected, "shared"), knobs=selected.knobs)
            if enabled
            else selected
        )
    if not variants:
        raise RuleSkipped("the requested carried-state storage exceeds shared memory")
    return variants
