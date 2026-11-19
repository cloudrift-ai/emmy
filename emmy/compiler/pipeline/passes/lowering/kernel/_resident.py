"""A carried state kept in register fragments: the walk's steps as one loop inside the CTA.

One warp owns sixteen positions of the state's last cell axis and every position of the one before it, as C-layout
fragments. Nothing in a step mixes two positions of the owned axis, so the warps never talk. A step is the carrying
fold's term evaluated block by block (:class:`_Blocks`): a read from memory is a fragment load, a lift is its
statements at fragment residence (:func:`~._atom._residence`, the evaluation the chunk tier uses), and a product
packs the fragments of its row-side operand into the atom's A, loads its other operand as B, and runs the atom.

The state is seeded ahead of the loop, every value of a step is computed and stored before the state's fragments are
overwritten, and the state itself has no store.

Leading ``_`` so the pass loader skips this module.
"""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    GMEM,
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
    """The values of a term over one warp's rows, a block of ``atom_m`` rows by ``atom_n`` columns at a time.

    A block is named by the node, the axis its rows run along (always the warp's sixteen), the axis its columns run
    along and the column it starts at. Equal blocks are computed once."""

    def __init__(self, tile, atom, period: int, row_base, state: tuple[str, ...], carrying):
        self.tile, self.atom, self.period, self.row_base = tile, atom, period, row_base
        self.state, self.carrying = state, carrying
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

    def gmem(self, buffer: str, index: tuple, row: str, col: str, cb: int) -> str:
        """The fragment of ``buffer`` at ``index``, the block's rows and columns standing for ``row`` and ``col``."""
        at = {row: Var(FRAG_ROW), col: Var(FRAG_COL)}
        key = (buffer, tuple(str(e) for e in index), row, col, cb)
        if key not in self.memo:
            name = self.fresh("g")
            index = tuple(e.substitute(at) for e in index)
            args = fragment_args(((buffer, index),), (GMEM,))
            self.out.append(
                FragmentApply(
                    out=name,
                    op=ElementwiseImpl("copy"),
                    args=args,
                    layout=self.layout,
                    row_base=self.row_base,
                    col_base=Literal(cb, "int"),
                )
            )
            self.memo[key] = name
        return self.memo[key]

    def value(self, node, result: str, row: str, col: str, cb: int):
        """``result`` of ``node`` over the block: a fragment's name, or the name of a value uniform over it."""
        key = (id(node), result, row, col, cb)
        if key not in self.blocks:
            self.blocks[key] = self._contract(node, row, col, cb) if self._contracts(node) else self._lifted(node, result, row, col, cb)
        return self.blocks[key]

    def _contracts(self, node) -> bool:
        return node.axis is not None and node is not self.carrying

    def _lifted(self, node, result: str, row: str, col: str, cb: int):
        """A lift at fragment residence: the statements ``result`` needs, each operand it reads as its block."""
        bound = {param: (edge, edge.exposes[component]) for param, edge, component in node.bindings}
        if result in bound and result not in node.lift.body.defined:
            edge, name = bound[result]
            return self.value(edge, name, row, col, cb)
        cone = node.lift.body.backward_cone((result,))
        frags: dict[str, str] = {}
        for param in cone.external_reads:
            if param in bound:
                edge, name = bound[param]
                operand = self.value(edge, name, row, col, cb)
                if operand not in self.uniform:
                    frags[param] = operand
                elif operand != param:
                    self.out.append(Assign(param, Var(operand)))
        for stmt in cone.members:
            if isinstance(stmt, Pre):
                if stmt.index[-2:] != (Var(col), Var(row)):
                    raise Unresident("a read of the state off its own row and column")
                frags[stmt.name] = self.state[cb // self.atom.atom_n]
            elif isinstance(stmt, Load) and {row, col} & {name for e in stmt.index for name in e.free_vars()}:
                frags[stmt.name] = self.gmem(stmt.input, stmt.index, row, col, cb)
            elif isinstance(stmt, (Load, Let)):
                if stmt.name not in self.uniform:
                    self.uniform.add(stmt.name)
                    self.out.append(stmt)
            elif isinstance(stmt, Assign):
                emitted, frags, _ = _residence([stmt], frags=frags, rows={}, tag=self.fresh("v"), memo=self.memo, layout=self.layout)
                self.out.extend(emitted)
            else:
                raise Unresident(f"a {type(stmt).__name__} in a step kept in registers")
        if result in frags:
            return frags[result]
        self.uniform.add(result)
        return result

    def _contract(self, node, row: str, col: str, cb: int) -> str:
        """A product over the block: for each ``atom_k`` positions of the node's axis, the row-side operand's
        fragments packed into A, the other operand loaded as B, and the atom into one accumulator."""
        view = node.as_contraction()
        if view is None or (view.product.name, view.plus.name) != ("multiply", "add") or len(node.operands) != 2:
            raise Unresident("a reduction that is no product under a sum")
        atom, axis = self.atom, node.axis
        bound = {param: (edge, edge.exposes[component]) for param, edge, component in node.bindings}
        (left, left_name), (right, _) = sorted(bound.values(), key=lambda pair: row not in pair[0].free_axes)
        slab = right.as_slab()
        if row not in left.free_axes or slab is None:
            raise Unresident("a product whose column-side operand is not read from memory")
        trans, ldm = _direct_operand(slab.load, self.tile.inputs, k_name=axis, own=col, legacy=(False, 0))
        if not ldm or self.extents[axis] % atom.atom_k:
            raise Unresident("a product whose column-side operand has no direct fragment load")
        total = self.fresh("c")
        narrow = atom.operand_dtype("c").nbytes == 2
        partial = self.fresh("p") if narrow else total
        self.out.append(RegFragment(name=total, role="c", dtype=F32, fragment_layout=self.layout))
        if narrow:
            self.out.append(RegFragment(name=partial, role="c", dtype=atom.operand_dtype("c"), fragment_layout=self.layout))
        steps = range(0, self.extents[axis], atom.atom_k)
        for step, k in enumerate(steps, 1):
            sources = tuple(self.value(left, left_name, row, axis, k + d) for d in range(0, atom.atom_k, atom.atom_n))
            a, b = self.fresh("a"), self.fresh("b")
            at = {axis: Literal(k, "int"), col: Literal(cb, "int")}
            self.out += [
                RegFragment(name=a, role="a", dtype=atom.operand_dtype("a"), fragment_layout=self.layout),
                FragmentRepack(frag=a, srcs=sources, role="a", fragment_layout=self.layout, part=0),
                RegFragment(name=b, role="b", dtype=atom.operand_dtype("b"), fragment_layout=self.layout),
                FragmentLoad(
                    frag=b,
                    src_buffer=slab.load.input,
                    src_index=tuple(e.substitute(at) for e in slab.load.index),
                    role="b",
                    b_trans=trans,
                    ldm=ldm,
                    fragment_layout=self.layout,
                ),
                MmaSyncPtx(
                    c_frag=partial, a_frag=a, b_frag=b, shape=atom.ptx_shape, ab_dtype=atom.ab_dtype, c_dtype=atom.operand_dtype("c").name
                ),
            ]
            if narrow and (step % self.period == 0 or step == len(steps)):
                self.out.append(FragmentPromote(dst=total, src=partial, fragment_layout=self.layout))
        return total


def resident_walk(tile, atom, warps: int, period: int) -> Tile:
    """The kernel of ``tile``, whose one fold carries a state, with that state in the fragments of ``atom``:
    ``warps`` warps a CTA, a narrow accumulator promoted every ``period`` atom steps."""
    carrying = [site.node for site in tile.sites if site.node.carries]
    if len(carrying) != 1 or tile.place.free or any(not axis.extent.is_static for axis in tile.axes):
        raise Unresident("one state of static shape under no free axis")
    (fold,) = carrying
    if len(fold.base.results) != 1 or len(fold.cells) < 2:
        raise Unresident("one state with a row and a column")
    *batch, col, row = fold.cells
    extents = {axis.name: axis.extent.as_static() for axis in tile.axes}
    if extents[row] % atom.atom_m or extents[col] % atom.atom_n:
        raise Unresident("a state the fragments do not tile exactly")
    time = tile.axis_of(fold.axis)
    row_base = Literal(atom.atom_m, "int") * (Literal(warps, "int") * Var("_rb") + Var("_rw"))
    state = tuple(f"_state{j}" for j in range(extents[col] // atom.atom_n))
    blocks = _Blocks(tile, atom, period, row_base, state, fold)

    # The seed, ahead of the loop: a constant, or the seed tensor read at the state's cells.
    seed = fold.init[0]
    prologue: list = []
    for j, name in enumerate(state):
        if isinstance(seed, str):
            cells = (*(Var(b) for b in batch), Var(FRAG_COL), Var(FRAG_ROW))
            args, kinds = ((seed, _at_unit_dims(cells, tile.inputs[seed].shape)),), (GMEM,)
        else:
            args, kinds = (Literal(float(seed)),), ("uniform",)
        prologue.append(
            FragmentApply(
                out=name,
                op=ElementwiseImpl("copy"),
                args=fragment_args(args, kinds),
                layout=atom.fragment_layout,
                row_base=row_base,
                col_base=Literal(j * atom.atom_n, "int"),
            )
        )

    # Every stored value of the step, then the state's next value; the state's fragments are overwritten last.
    stores: list = []
    for spec in tile.output_specs:
        (value,) = spec.write.values
        out_col, out_row = (e.name for e in spec.write.index[-2:])
        if extents[out_row] != extents[row] or extents[out_col] % atom.atom_n:
            raise Unresident("an output whose rows are not the state's")
        for cb in range(0, extents[out_col], atom.atom_n):
            frag = blocks.value(fold, value, out_row, out_col, cb)
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
    following = [blocks.value(fold, fold.lift.results[0], row, col, j * atom.atom_n) for j in range(len(state))]
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
    return Tile(axes=axes, body=Body((*prologue, loop)), block_threads=warps * 32)
