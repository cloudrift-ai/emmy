"""A carried state kept in register fragments: the walk's steps as one loop inside the CTA.

One warp owns sixteen positions of the state's last cell axis and every position of the one before it, as C-layout
fragments. Nothing in a step mixes two positions of the owned axis, so the warps never talk. A step is the carrying
fold's term evaluated block by block (:class:`_Blocks`): a read from memory is a fragment load, a lift is its
statements at fragment residence (:func:`~._atom._residence`, the evaluation the chunk tier uses), and a product
packs the fragments of its row-side operand into the atom's A, loads its other operand as B, and runs the atom.

The seed is selected in at the first step, every value of a step is computed and stored before the state's fragments
are overwritten, and the state itself has no store.

A step may keep part of its state: its next value is a choice whose last branch is the state as it stands. A block
that no position of an earlier branch reaches does not compute that branch (:meth:`_Blocks._chosen`), so a step that
writes one row of the state runs the product of one block, not of the state.

Leading ``_`` so the pass loader skips this module.
"""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import BinaryExpression, Interval, Literal, SimplifyCtx, TernaryExpression, Var, affine_form
from emmy.compiler.ir.kernel.ir import (
    COORD,
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    GMEM,
    UNIFORM,
    FragmentApply,
    FragmentLoad,
    FragmentPromote,
    FragmentRepack,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    Tile,
    fragment_args,
)
from emmy.compiler.ir.stmt import Assign, Body, Cond, Let, Load, Loop, Pre, Select
from emmy.compiler.ir.tile.ir import matrix_coordinates
from emmy.compiler.target import FLAGGED_WHOLE

from ._atom import _direct_operand, _residence, unroll_ok


class Unresident(ValueError):
    """The term has a shape these fragments do not hold: the row is not offered."""


def _at_unit_dims(index: tuple, shape: tuple) -> tuple:
    """``index`` over a buffer of ``shape``, a zero at every size-one dim the index has no coordinate for."""
    if len(index) == len(shape):
        return index
    units = [dim for dim, extent in enumerate(shape) if extent == 1]
    if len(shape) - len(index) != len(units):
        raise Unresident(f"an index of {len(index)} coordinates does not address a buffer of shape {shape}")
    rest = iter(index)
    return tuple(Literal(0, "int") if dim in units else next(rest) for dim in range(len(shape)))


def _given(expression, conditions: tuple, held: tuple = ()):
    """``expression`` where every one of ``conditions`` holds: a choice on one of them is its first arm, and a value
    one of them holds a coordinate to (``held``: the value, the coordinate's name) is that coordinate."""
    if isinstance(expression, TernaryExpression) and expression.cond in conditions:
        return _given(expression.if_true, conditions, held)
    coordinate = next((name for value, name in held if value == expression), None)
    if coordinate is not None:
        return Var(coordinate)
    children = expression.children()
    return expression.with_children(tuple(_given(child, conditions, held) for child in children)) if children else expression


def _needed(body: Body, result: str) -> tuple[list, set[str]]:
    """The statements of ``body`` that ``result`` needs, in order, and the names they read. A choice among them
    stands for its branches: what a branch reads is computed where the choice is, behind the branch's condition."""
    reads, members = {result}, []
    for stmt in reversed(tuple(body)):
        if reads.isdisjoint(stmt.defines()):
            continue
        members.append(stmt)
        if not isinstance(stmt, Select) or stmt.name == result:
            reads |= set(stmt.deps())
    return members[::-1], reads


def _bounds(condition, col: str, row: str) -> tuple[list, list] | None:
    """``(lows, highs)``: the values ``condition`` keeps ``col`` at or above, and the ones it keeps it below. ``None``
    where the condition is more than such bounds by values that read neither cell: nothing is known of where it
    holds."""
    lows, highs, terms, one = [], [], [condition], Literal(1, "int")
    while terms:
        term = terms.pop()
        if not isinstance(term, BinaryExpression):
            return None
        if term.op == "&&":
            terms += [term.left, term.right]
            continue
        over = term.left == Var(col)
        if not over and term.right != Var(col):
            return None
        bound = term.right if over else term.left
        if {col, row} & bound.free_vars() or term.op not in ("<", "<=", "=="):
            return None
        if term.op == "==" or not over:
            lows.append(bound + one if term.op == "<" else bound)
        if term.op == "==" or over:
            highs.append(bound if term.op == "<" else bound + one)
    return lows, highs


def _apart(high, low) -> int | None:
    """``high - low`` where it is one number whatever the coordinates are, else ``None``."""
    difference = BinaryExpression("-", high, low)
    form = affine_form(difference, difference.free_vars())
    span = form[0].range(SimplifyCtx.empty()) if form is not None and not form[1] else None
    return span.lo if span is not None and span.lo == span.hi else None


class _Blocks:
    """The values of a term over register fragments, a block of ``atom_m`` rows by ``atom_n`` columns at a time.

    A block is named by the node, the axis its rows run along and where they start (``None``: the warp's own
    sixteen), and the axis its columns run along and where they start. Equal blocks are computed once, and a value
    that does not read the column axis is one block for every column."""

    def __init__(self, tile, atom, period: int, row_base, state: tuple[str, ...], ahead: bool = False):
        self.tile, self.atom, self.period, self.row_base, self.state = tile, atom, period, row_base, state
        self.ahead = ahead
        self.carrying, self.batch, _, _ = tile.resident_state
        self.time = self.carrying.axis
        self.layout = atom.fragment_layout
        self.extents = {axis.name: axis.extent.as_static() for axis in (*tile.axes, *tile.place.free)}
        self.ranges = SimplifyCtx(ranges={name: Interval(0, extent - 1) for name, extent in self.extents.items()})
        self.out: list = []
        self.memo: dict = {}
        self.blocks: dict = {}
        self.uniform: set[str] = set()
        self.given: tuple = ()
        self.held: tuple = ()
        self.count = 0

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"_{stem}{self.count}"

    def settled(self, load):
        """``load`` with the conditions given taken off its index, where the index then stays inside the buffer
        at every position of every axis: the choice it loses only kept the read in bounds where the branch that
        reads it is not taken."""
        index = tuple(_given(entry, self.given, self.held) for entry in load.index)
        if index == tuple(load.index):
            return load
        spans = [entry.range(self.ranges) for entry in index]
        shape = self.tile.inputs[load.input].shape
        inside = all(span is not None and 0 <= span.lo and span.hi < dim.as_static() for span, dim in zip(spans, shape, strict=True))
        return load.replace(index=index) if inside else load

    def copied(self, args: tuple, kinds: tuple, rb: int | None, cb: int) -> str:
        """The fragment a copy fills: from memory, or with one value for every element."""
        name = self.fresh("g")
        self.out.append(
            FragmentApply(
                out=name,
                op=ElementwiseImpl("copy"),
                args=fragment_args(args, kinds),
                layout=self.layout,
                row_base=self.row_base if rb is None else Literal(rb, "int"),
                col_base=Literal(cb, "int"),
            )
        )
        return name

    def _seed(self, cb: int) -> tuple:
        """The seed at column ``cb`` as a fragment statement reads it: the operand, and its kind."""
        seed = self.carrying.init[0]
        if not isinstance(seed, str):
            return Literal(float(seed)), UNIFORM
        cells = (*(Var(b) for b in self.batch), Var(FRAG_COL), Var(FRAG_ROW))
        return (seed, _at_unit_dims(cells, self.tile.inputs[seed].shape)), GMEM

    def seeding(self) -> list:
        """The statements that put the seed in the state's fragments ahead of the steps. A zero takes none: a
        fragment is declared zero."""
        if self.carrying.init[0] == 0:
            return []
        fill = dict(op=ElementwiseImpl("copy"), layout=self.layout, row_base=self.row_base, in_place=True)
        return [
            FragmentApply(out=name, args=fragment_args(*zip(self._seed(cb))), col_base=Literal(cb, "int"), **fill)
            for cb, name in zip(range(0, len(self.state) * self.atom.atom_n, self.atom.atom_n), self.state, strict=True)
        ]

    def seeded(self, cb: int) -> str:
        """The state one step back at column ``cb``: the warp's own fragment of it, and at the first step the seed,
        a constant or the seed tensor read at the block. A zero seed is the fragment as declared, and a state
        seeded ahead of its steps (:meth:`seeding`) is the fragment too. Selecting any other seed inside the step is
        measured, not principle: seeding the fragments ahead of the loop does less work and ran 5 to 8% slower on an
        RTX 5090 for a step that writes the whole state."""
        key = ("seeded", cb)
        if self.ahead or self.carrying.init[0] == 0:
            return self.state[cb // self.atom.atom_n]
        if key not in self.memo:
            seed, kind = self._seed(cb)
            if kind == GMEM:
                seed, kind = self.copied((seed,), (GMEM,), None, cb), FRAG
            name = self.fresh("s")
            first = BinaryExpression(">", Var(self.time), Literal(0, "int"))
            self.out.append(
                FragmentApply(
                    out=name,
                    op=ElementwiseImpl("where"),
                    args=fragment_args((first, self.state[cb // self.atom.atom_n], seed), (COORD, FRAG, kind)),
                    layout=self.layout,
                    row_base=self.row_base,
                    col_base=Literal(cb, "int"),
                )
            )
            self.memo[key] = name
        return self.memo[key]

    def gmem(self, buffer: str, index: tuple, row: str, col: str, rb: int | None, cb: int) -> str:
        """The fragment of ``buffer`` at ``index``, the block's rows and columns standing for ``row`` and ``col``."""
        index = tuple(e.substitute({row: Var(FRAG_ROW), col: Var(FRAG_COL)}) for e in index)
        key = (buffer, tuple(str(e) for e in index), rb, cb)
        if key not in self.memo:
            self.memo[key] = self.copied(((buffer, index),), (GMEM,), rb, cb)
        return self.memo[key]

    def fragment(self, node, result: str, row: str, col: str, rb: int | None, cb: int) -> str:
        """:meth:`value` as a fragment, a value uniform over the block copied into one."""
        value = self.value(node, result, row, col, rb, cb)
        if value not in self.uniform:
            return value
        key = ("uniform", value)
        if key not in self.memo:
            self.memo[key] = self.copied((value,), (UNIFORM,), rb, cb)
        return self.memo[key]

    def value(self, node, result: str, row: str, col: str, rb: int | None, cb: int):
        """``result`` of ``node`` over the block: a fragment's name, or the name of a value uniform over it."""
        if node is not self.carrying and col not in node.free_axes:
            cb = 0
        key = (id(node), result, row, col, rb, cb)
        if key not in self.blocks:
            contracts = node.axis is not None and node is not self.carrying
            self.blocks[key] = self._contract(node, row, col, rb, cb) if contracts else self._lifted(node, result, row, col, rb, cb)
        return self.blocks[key]

    def _lifted(self, node, result: str, row: str, col: str, rb: int | None, cb: int):
        """A lift at fragment residence: the statements ``result`` needs, each operand it reads as its block."""
        bound = {param: (edge, edge.exposes[component]) for param, edge, component in node.bindings}
        if result in bound and result not in node.lift.body.defined:
            edge, name = bound[result]
            return self.value(edge, name, row, col, rb, cb)
        members, reads = _needed(node.lift.body, result)
        if members and isinstance(members[-1], Select):
            return self._chosen(node, members[-1], row, col, rb, cb)
        frags: dict[str, str] = {}
        # In the order the lift binds them: the reads are a set, and its order is another one every run.
        for param in node.lift.params:
            if param in reads and param in bound:
                edge, name = bound[param]
                operand = self.value(edge, name, row, col, rb, cb)
                if operand not in self.uniform:
                    frags[param] = operand
                elif operand != param and param not in self.uniform:
                    self.uniform.add(param)
                    self.out.append(Assign(param, Var(operand)))
        for stmt in members:
            if isinstance(stmt, Select):
                frags[stmt.name] = self.fragment(node, stmt.name, row, col, rb, cb)
            elif isinstance(stmt, Pre):
                if rb is not None or tuple(_given(e, self.given) for e in stmt.index[-2:]) != (Var(col), Var(row)):
                    raise Unresident("a read of the state off its own row and column")
                frags[stmt.name] = self.seeded(cb)
            elif isinstance(stmt, Load):
                # Every read is a fragment, one that reads neither axis too. Held as a scalar across the step it
                # costs a register the assembler has none to spare of: 8% on a 16-step dense recurrence.
                frags[stmt.name] = self.gmem(stmt.input, self.settled(stmt).index, row, col, rb, cb)
            elif isinstance(stmt, Let) or (isinstance(stmt, Assign) and not set(stmt.deps()) & frags.keys()):
                # One value for the whole block: a scalar statement, emitted once.
                if stmt.name not in self.uniform:
                    self.uniform.add(stmt.name)
                    self.out.append(stmt)
            elif isinstance(stmt, Assign):
                emitted, frags, _ = _residence([stmt], frags=frags, rows={}, tag=self.fresh("v"), memo=self.memo, layout=self.layout)
                self.out.extend(emitted)
            else:
                raise Unresident(f"a {type(stmt).__name__} in a step kept in registers")
        return frags.get(result, result)

    def _chosen(self, node, choice, row: str, col: str, rb: int | None, cb: int) -> str:
        """A choice over the block as one fragment: its last branch's value, and over it each branch before where
        that branch's condition holds. A branch whose condition bounds the column to positions outside the block is
        not computed: its statements run behind that test, and what they compute is no value outside it. Inside,
        the condition is given: a read it guards is the read itself, and where it holds the column to one value,
        an address spelled by that value moves with the column, as a product's operand in memory does."""
        *branches, rest = choice.branches
        chosen = self.fragment(node, rest.value, row, col, rb, cb)
        rows = self.row_base if rb is None else Literal(rb, "int")
        # Each side writes the fragment declared ahead of the test, so the value stands after it.
        base = dict(layout=self.layout, row_base=rows, col_base=Literal(cb, "int"), in_place=True)
        for branch in reversed(branches):
            name = self.fresh("w")
            lows, highs = _bounds(branch.select, col, row) or ((), ())
            one = [low for low in lows for high in highs if _apart(high, low) == 1]
            outside = self.out, self.given, self.held, dict(self.memo), dict(self.blocks), set(self.uniform)
            self.out, self.given, self.held = [], (*self.given, branch.select), (*self.held, *((low, col) for low in one))
            try:
                value, inside = self.fragment(node, branch.value, row, col, rb, cb), self.out
            finally:
                self.out, self.given, self.held, self.memo, self.blocks, self.uniform = outside
            at = branch.select.substitute({row: Var(FRAG_ROW), col: Var(FRAG_COL)})
            where = fragment_args((at, value, chosen), (COORD, FRAG, FRAG))
            taken = (*inside, FragmentApply(out=name, op=ElementwiseImpl("where"), args=where, **base))
            first, past = Literal(cb, "int"), Literal(cb + self.atom.atom_n, "int")
            reach = None
            for test in (*(BinaryExpression("<", low, past) for low in lows), *(BinaryExpression("<", first, high) for high in highs)):
                reach = test if reach is None else BinaryExpression("&&", reach, test)
            self.out.append(RegFragment(name=name, role="c", dtype=F32, fragment_layout=self.layout))
            if reach is None:
                self.out += taken
            else:
                kept = FragmentApply(out=name, op=ElementwiseImpl("copy"), args=fragment_args((chosen,), (FRAG,)), **base)
                self.out.append(Cond(reach, Body(taken), Body((kept,))))
            chosen = name
        return chosen

    def _contract(self, node, row: str, col: str, rb: int | None, cb: int) -> str:
        """A product over the block: for each ``atom_k`` positions of the node's axis, the row-side operand's
        fragments packed into A, the other operand as B, and the atom into one accumulator. B is loaded straight
        from memory where the operand is a read the fragment loader takes, else packed from the operand's own
        block, whose rows run along the node's axis."""
        view = node.as_contraction()
        if view is None or (view.product.name, view.plus.name) != ("multiply", "add") or len(node.operands) != 2:
            raise Unresident("a reduction that is no product under a sum")
        if node.range is not None:
            raise Unresident("a product that walks a range of its axis, where the atom folds every position")
        atom, axis = self.atom, node.axis
        # The two values the product reads: an operand shared with another product may expose more than this one's.
        read = node.lift.body.backward_cone(node.lift.results).external_reads
        bound = {param: (edge, edge.exposes[component]) for param, edge, component in node.bindings if param in read}
        if len(bound) != 2:
            raise Unresident("a product of other than two values")
        (left, left_name), (right, right_name) = sorted(bound.values(), key=lambda pair: row not in pair[0].free_axes)
        if rb is not None or row not in left.free_axes or self.extents[axis] % atom.atom_k:
            raise Unresident("a product that is not over the warp's own rows and whole atoms")
        slab, direct = right.as_slab(), None
        load = self.settled(slab.load) if slab is not None else None
        if slab is not None:
            try:
                trans, ldm = _direct_operand(load, self.tile.inputs, k_name=axis, own=col, legacy=(False, 0))
                direct = dict(b_trans=trans, ldm=ldm) if ldm else None
            except ValueError:
                direct = None
        steps = range(0, self.extents[axis], atom.atom_k)
        # Each step's operands are computed and packed beside the atom that reads them, so no step's fragments
        # are held across the others.
        begin, operands, prepared = len(self.out), [], []
        for k in steps:
            start = len(self.out)
            first = k // atom.atom_n * atom.atom_n
            span = range(0, max(atom.atom_k, atom.atom_n), atom.atom_n)
            a = tuple(self.fragment(left, left_name, row, axis, None, first + d) for d in span)
            if direct is not None:
                at = {axis: Literal(k, "int"), col: Literal(cb, "int")}
                b = tuple(str(e.substitute(at)) for e in load.index)
            else:
                b = self.fragment(right, right_name, axis, col, k // atom.atom_m * atom.atom_m, cb)
            operands.append((a, (k - first) // atom.atom_k, b, (k % atom.atom_m) // atom.atom_k))
            prepared.append(self.out[start:])
        del self.out[begin:]
        # Two products over the same fragments and the same memory are one product, whatever their axes are called.
        key = ("product", tuple(operands), load.input if direct is not None else None)
        if key in self.memo:
            self.out += [stmt for prepare in prepared for stmt in prepare]
            return self.memo[key]
        total = self.fresh("c")
        narrow = atom.operand_dtype("c").nbytes == 2
        partial = self.fresh("p") if narrow else total
        self.out.append(RegFragment(name=total, role="c", dtype=F32, fragment_layout=self.layout))
        if narrow:
            self.out.append(RegFragment(name=partial, role="c", dtype=atom.operand_dtype("c"), fragment_layout=self.layout))
        for step, (k, (a_sources, a_part, b_source, b_part), prepare) in enumerate(zip(steps, operands, prepared, strict=True), 1):
            a, b = self.fresh("a"), self.fresh("b")
            if direct is not None:
                at = {axis: Literal(k, "int"), col: Literal(cb, "int")}
                index = tuple(e.substitute(at) for e in load.index)
                filled = FragmentLoad(frag=b, src_buffer=load.input, src_index=index, role="b", fragment_layout=self.layout, **direct)
            else:
                filled = FragmentRepack(
                    frag=b, srcs=(b_source,), role="b", fragment_layout=self.layout, part=b_part, ab_dtype=atom.ab_dtype
                )
            self.out += [
                *prepare,
                RegFragment(name=a, role="a", dtype=atom.operand_dtype("a"), fragment_layout=self.layout),
                FragmentRepack(frag=a, srcs=a_sources, role="a", fragment_layout=self.layout, part=a_part, ab_dtype=atom.ab_dtype),
                RegFragment(name=b, role="b", dtype=atom.operand_dtype("b"), fragment_layout=self.layout),
                filled,
                MmaSyncPtx(
                    c_frag=partial,
                    a_frag=a,
                    b_frag=b,
                    shape=atom.ptx_shape,
                    ab_dtype=atom.ab_dtype,
                    c_dtype=atom.operand_dtype("c").name,
                ),
            ]
            if narrow and (step % self.period == 0 or step == len(steps)):
                self.out.append(FragmentPromote(dst=total, src=partial, fragment_layout=self.layout))
        self.memo[key] = total
        return total


def resident_walk(tile, atom, warps: int, period: int) -> Tile:
    """The kernel of ``tile``, whose one fold carries a state, with that state in the fragments of ``atom``:
    ``warps`` warps a CTA, a narrow accumulator promoted every ``period`` atom steps."""
    if tile.resident_state is None:
        raise Unresident("one state of static shape, its rows and columns whole fragments")
    fold, batch, col, row = tile.resident_state
    extents = {axis.name: axis.extent.as_static() for axis in tile.axes}
    stored = [(spec, *matrix_coordinates(spec.write.index)) for spec in tile.output_specs]
    widths = [extents[col], *(extents[spec.write.index[at].name] for spec, at, _ in stored)]
    if extents[row] % atom.atom_m or any(width % atom.atom_n for width in widths):
        raise Unresident("a state or an output the atom's fragments do not tile exactly")
    time = tile.axis_of(fold.axis)
    row_base = Literal(atom.atom_m, "int") * (Literal(warps, "int") * Var("_rb") + Var("_rw"))
    state = tuple(f"_state{j}" for j in range(extents[col] // atom.atom_n))
    # A step that keeps part of its state has its state seeded ahead of the steps, and is written out step by step:
    # the tests of its blocks read the step alone, so each copy is the work of the blocks that step writes, with no
    # test left in it, and a block it keeps is no work at all.
    members, _ = _needed(fold.lift.body, fold.lift.results[0])
    keeps = bool(members) and isinstance(members[-1], Select)
    blocks = _Blocks(tile, atom, period, row_base, state, ahead=keeps)
    declared = [RegFragment(name=name, role="c", dtype=F32, fragment_layout=atom.fragment_layout) for name in state]

    # Every stored value of the step, then the state's next value; the state's fragments are overwritten last. A
    # store that does not read the step writes the same cells every step, so the last step's alone is made.
    stores: list = []
    last = BinaryExpression("==", Var(time.name), Literal(time.extent.as_static() - 1, "int"))
    for spec, col_dim, row_dim in stored:
        (value,) = spec.write.values
        out_col, out_row = spec.write.index[col_dim].name, spec.write.index[row_dim].name
        cells = []
        for cb in range(0, extents[out_col], atom.atom_n):
            frag = blocks.fragment(fold, value, out_row, out_col, None, cb)
            index = tuple(
                Literal(cb, "int") if dim == col_dim else row_base if dim == row_dim else entry
                for dim, entry in enumerate(spec.write.index)
            )
            cells.append(
                RegStore(
                    dst_buffer=spec.write.output,
                    dst_index=index,
                    frag=frag,
                    shape=atom.shape,
                    fragment_layout=atom.fragment_layout,
                    row_dim=row_dim,
                    col_dim=col_dim,
                    m_guard=(row_base, Literal(extents[out_row], "int")),
                    n_guard=(Literal(cb, "int"), Literal(extents[out_col], "int")),
                )
            )
        every = any(time.name in entry.free_vars() for entry in spec.write.index)
        stores += cells if every else [Cond(last, Body(cells), Body())]
    following = [blocks.fragment(fold, fold.lift.results[0], row, col, None, j * atom.atom_n) for j in range(len(state))]
    commit = [
        FragmentApply(
            out=name, op=ElementwiseImpl("copy"), args=fragment_args((value,), (FRAG,)), layout=atom.fragment_layout, in_place=True
        )
        for name, value in zip(state, following, strict=True)
    ]
    unroll = keeps and unroll_ok(time.extent, FLAGGED_WHOLE)
    loop = Loop.strided(
        axis=time, start=Literal(0, "int"), step=Literal(1, "int"), body=Body((*blocks.out, *stores, *commit)), unroll=unroll
    )
    rows = extents[row]
    axes = (
        *tile.place.free,
        *(tile.axis_of(name) for name in batch),
        Axis("_rb", -(-rows // (warps * atom.atom_m))),
        Axis("_rw", warps),
        Axis("_rl", 32),
    )
    return Tile(axes=axes, body=Body((*declared, *(blocks.seeding() if keeps else ()), loop)), block_threads=warps * 32)
