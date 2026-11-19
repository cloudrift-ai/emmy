"""A carried state kept in register fragments: the walk's steps as one loop inside the CTA.

One warp owns sixteen positions of the state's last cell axis and every position of the one before it, as C-layout
fragments. Nothing in a step mixes two positions of the owned axis, so the warps never talk. A step is the carrying
fold's term evaluated block by block (:class:`_Blocks`): a read from memory is a fragment load, a lift is its
statements at fragment residence (:func:`~._atom._residence`, the evaluation the chunk tier uses), and a product
packs the fragments of its row-side operand into the atom's A, loads its other operand as B, and runs the atom.

The seed is selected in at the first step, every value of a step is computed and stored before the state's fragments
are overwritten, and the state itself has no store.

Leading ``_`` so the pass loader skips this module.
"""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
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
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Pre, StridedLoop

from ._atom import _direct_operand, _residence


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


class _Blocks:
    """The values of a term over register fragments, a block of ``atom_m`` rows by ``atom_n`` columns at a time.

    A block is named by the node, the axis its rows run along and where they start (``None``: the warp's own
    sixteen), and the axis its columns run along and where they start. Equal blocks are computed once, and a value
    that does not read the column axis is one block for every column."""

    def __init__(self, tile, atom, period: int, row_base, state: tuple[str, ...]):
        self.tile, self.atom, self.period, self.row_base, self.state = tile, atom, period, row_base, state
        self.carrying, self.batch, _, _ = tile.resident_state
        self.time = self.carrying.axis
        self.layout = atom.fragment_layout
        self.extents = {axis.name: axis.extent.as_static() for axis in tile.axes}
        self.out: list = []
        self.memo: dict = {}
        self.blocks: dict = {}
        self.uniform: set[str] = set()
        self.count = 0

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"_{stem}{self.count}"

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

    def seeded(self, cb: int) -> str:
        """The state one step back at column ``cb``: the warp's own fragment of it, and at the first step the seed,
        a constant or the seed tensor read at the block. Selecting the seed inside the step is measured, not
        principle: seeding the fragments ahead of the loop does less work and ran 5 to 8% slower on an RTX 5090."""
        key = ("seeded", cb)
        if key not in self.memo:
            seed = self.carrying.init[0]
            if isinstance(seed, str):
                cells = (*(Var(b) for b in self.batch), Var(FRAG_COL), Var(FRAG_ROW))
                seed = self.copied(((seed, _at_unit_dims(cells, self.tile.inputs[seed].shape)),), (GMEM,), None, cb)
                kinds = (COORD, FRAG, FRAG)
            else:
                seed, kinds = Literal(float(seed)), (COORD, FRAG, UNIFORM)
            name = self.fresh("s")
            first = BinaryExpression(">", Var(self.time), Literal(0, "int"))
            self.out.append(
                FragmentApply(
                    out=name,
                    op=ElementwiseImpl("where"),
                    args=fragment_args((first, self.state[cb // self.atom.atom_n], seed), kinds),
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
        cone = node.lift.body.backward_cone((result,))
        frags: dict[str, str] = {}
        # In the order the lift binds them: the cone's reads are a set, and its order is another one every run.
        for param in node.lift.params:
            if param in cone.external_reads and param in bound:
                edge, name = bound[param]
                operand = self.value(edge, name, row, col, rb, cb)
                if operand not in self.uniform:
                    frags[param] = operand
                elif operand != param and param not in self.uniform:
                    self.uniform.add(param)
                    self.out.append(Assign(param, Var(operand)))
        for stmt in cone.members:
            if isinstance(stmt, Pre):
                if rb is not None or stmt.index[-2:] != (Var(col), Var(row)):
                    raise Unresident("a read of the state off its own row and column")
                frags[stmt.name] = self.seeded(cb)
            elif isinstance(stmt, Load):
                # Every read is a fragment, one that reads neither axis too. Held as a scalar across the step it
                # costs a register the assembler has none to spare of: 8% on a 16-step dense recurrence.
                frags[stmt.name] = self.gmem(stmt.input, stmt.index, row, col, rb, cb)
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

    def _contract(self, node, row: str, col: str, rb: int | None, cb: int) -> str:
        """A product over the block: for each ``atom_k`` positions of the node's axis, the row-side operand's
        fragments packed into A, the other operand as B, and the atom into one accumulator. B is loaded straight
        from memory where the operand is a read the fragment loader takes, else packed from the operand's own
        block, whose rows run along the node's axis."""
        view = node.as_contraction()
        if view is None or (view.product.name, view.plus.name) != ("multiply", "add") or len(node.operands) != 2:
            raise Unresident("a reduction that is no product under a sum")
        atom, axis = self.atom, node.axis
        bound = {param: (edge, edge.exposes[component]) for param, edge, component in node.bindings}
        (left, left_name), (right, right_name) = sorted(bound.values(), key=lambda pair: row not in pair[0].free_axes)
        if rb is not None or row not in left.free_axes or self.extents[axis] % atom.atom_k:
            raise Unresident("a product that is not over the warp's own rows and whole atoms")
        slab, direct = right.as_slab(), None
        if slab is not None:
            try:
                trans, ldm = _direct_operand(slab.load, self.tile.inputs, k_name=axis, own=col, legacy=(False, 0))
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
                b = tuple(str(e.substitute(at)) for e in slab.load.index)
            else:
                b = self.fragment(right, right_name, axis, col, k // atom.atom_m * atom.atom_m, cb)
            operands.append((a, (k - first) // atom.atom_k, b, (k % atom.atom_m) // atom.atom_k))
            prepared.append(self.out[start:])
        del self.out[begin:]
        # Two products over the same fragments and the same memory are one product, whatever their axes are called.
        key = ("product", tuple(operands), slab.load.input if direct is not None else None)
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
                index = tuple(e.substitute(at) for e in slab.load.index)
                filled = FragmentLoad(frag=b, src_buffer=slab.load.input, src_index=index, role="b", fragment_layout=self.layout, **direct)
            else:
                filled = FragmentRepack(frag=b, srcs=(b_source,), role="b", fragment_layout=self.layout, part=b_part)
            self.out += [
                *prepare,
                RegFragment(name=a, role="a", dtype=atom.operand_dtype("a"), fragment_layout=self.layout),
                FragmentRepack(frag=a, srcs=a_sources, role="a", fragment_layout=self.layout, part=a_part),
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
    widths = [extents[col], *(extents[spec.write.index[-2].name] for spec in tile.output_specs)]
    if extents[row] % atom.atom_m or any(width % atom.atom_n for width in widths):
        raise Unresident("a state or an output the atom's fragments do not tile exactly")
    time = tile.axis_of(fold.axis)
    row_base = Literal(atom.atom_m, "int") * (Literal(warps, "int") * Var("_rb") + Var("_rw"))
    state = tuple(f"_state{j}" for j in range(extents[col] // atom.atom_n))
    blocks = _Blocks(tile, atom, period, row_base, state)
    declared = [RegFragment(name=name, role="c", dtype=F32, fragment_layout=atom.fragment_layout) for name in state]

    # Every stored value of the step, then the state's next value; the state's fragments are overwritten last.
    stores: list = []
    for spec in tile.output_specs:
        (value,) = spec.write.values
        out_col, out_row = (e.name for e in spec.write.index[-2:])
        for cb in range(0, extents[out_col], atom.atom_n):
            frag = blocks.fragment(fold, value, out_row, out_col, None, cb)
            index = (*spec.write.index[:-2], Literal(cb, "int"), row_base)
            stores.append(
                RegStore(
                    dst_buffer=spec.write.output,
                    dst_index=index,
                    frag=frag,
                    shape=atom.shape,
                    fragment_layout=atom.fragment_layout,
                    row_dim=len(index) - 1,
                    col_dim=len(index) - 2,
                    m_guard=(row_base, Literal(extents[out_row], "int")),
                    n_guard=(Literal(cb, "int"), Literal(extents[out_col], "int")),
                )
            )
    following = [blocks.fragment(fold, fold.lift.results[0], row, col, None, j * atom.atom_n) for j in range(len(state))]
    commit = [
        FragmentApply(
            out=name, op=ElementwiseImpl("copy"), args=fragment_args((value,), (FRAG,)), layout=atom.fragment_layout, in_place=True
        )
        for name, value in zip(state, following, strict=True)
    ]
    loop = StridedLoop(axis=time, start=Literal(0, "int"), step=Literal(1, "int"), body=Body((*blocks.out, *stores, *commit)), unroll=False)
    rows = extents[row]
    axes = (
        *(tile.axis_of(name) for name in batch),
        Axis("_rb", -(-rows // (warps * atom.atom_m))),
        Axis("_rw", warps),
        Axis("_rl", 32),
    )
    return Tile(axes=axes, body=Body((*declared, loop)), block_threads=warps * 32)
