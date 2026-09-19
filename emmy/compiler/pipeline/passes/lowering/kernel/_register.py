"""Materialize an ordered Fold tree with its matrix values stored in register fragments."""

from __future__ import annotations

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    COORD,
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    GMEM,
    UNIFORM,
    FragmentApply,
    FragmentPromote,
    FragmentRepack,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    Tile,
)
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Select, StridedLoop


class _Fragments:
    """Emit the term once per coordinate tile, sharing equal loads, arithmetic and contractions."""

    def __init__(self, tile, row_base):
        self.tile = tile
        self.program = tile.register_program
        self.atom = tile.schedule.kernel.tile.atom
        self.period = tile.schedule.kernel.tile.bk
        self.row_base = row_base
        self.extents = {axis.name: axis.extent.as_static() for axis in tile.axes}
        self.body = []
        self.memo = {}
        self.cells = {}
        self.states = tuple(f"_state{j}" for j in range((self.program.columns + 7) // 8))
        self.counter = 0

    def name(self):
        name = f"_rf{self.counter}"
        self.counter += 1
        return name

    def apply(self, op, args, kinds, row_base=None, col_base=None):
        key = (op, args, kinds, row_base, col_base)
        if key not in self.memo:
            out = self.name()
            self.body.append(FragmentApply(out=out, op=ElementwiseImpl(op), args=args, kinds=kinds,
                                           row_base=row_base, col_base=col_base))
            self.memo[key] = out
        return self.memo[key]

    def fragment(self, value):
        return self.apply("copy", (value,), (UNIFORM,)) if isinstance(value, Literal) else value

    def pack(self, role, srcs):
        key = (role, srcs)
        if key not in self.memo:
            out = self.name()
            self.body.extend((
                RegFragment(name=out, role=role, shape=self.atom.shape, dtype=self.atom.operand_dtype(role)),
                FragmentRepack(frag=out, srcs=srcs, role=role),
            ))
            self.memo[key] = out
        return self.memo[key]

    def cell(self, node, row, col, rb, cb):
        key = (id(node), row, col, rb, cb)
        if key in self.cells:
            return self.cells[key]
        if isinstance(cb, Literal) and cb.value >= self.extents[col]:
            return (Literal(0.0),) * len(node.exposes)
        if node.axis is not None:
            result = (self.contract(node, row, col, rb, cb),)
        else:
            needed = node.lift.body.ssa_uses | set(node.lift.results)
            env = {}
            for param, edge, component in node.bindings:
                if param in needed:
                    env[param] = self.cell(edge, row, col, rb, cb)[component]
            coords = {row: Var(FRAG_ROW), col: Var(FRAG_COL)}
            # Clamp memory coordinates before masking a padded fragment. The padded value
            # is zero at the operand boundary, so it contributes the reduction's identity.
            clipped = {
                name: TernaryExpr(BinaryExpr("<", coord, Literal(self.extents[name], "int")),
                                  coord, Literal(self.extents[name] - 1, "int"))
                for name, coord in coords.items()
            }
            for stmt in node.lift.body:
                if isinstance(stmt, Let):
                    if not isinstance(stmt.value, Literal):
                        raise ValueError("register maps require literal Let values")
                    env[stmt.name] = stmt.value
                elif isinstance(stmt, Load):
                    if stmt.input == self.program.state.write.output:
                        if stmt.index[-2:] != (Var(col), Var(row)) or rb != self.row_base or not isinstance(cb, Literal):
                            raise ValueError("register state read crosses its owning warp rows")
                        env[stmt.name] = self.states[cb.value // 8]
                    else:
                        index = tuple(e.substitute(clipped).simplify(SimplifyCtx.empty()) for e in stmt.index)
                        env[stmt.name] = self.apply("copy", ((stmt.input, index),), (GMEM,), rb, cb)
                elif isinstance(stmt, Assign):
                    args = tuple(env[name] for name in stmt.args)
                    kinds = tuple(UNIFORM if isinstance(arg, Literal) else FRAG for arg in args)
                    env[stmt.name] = self.apply(stmt.op.name, args, kinds)
                elif isinstance(stmt, Select):
                    value = env[stmt.branches[-1].value]
                    for branch in reversed(stmt.branches[:-1]):
                        yes = env[branch.value]
                        predicate = branch.select.substitute(coords)
                        value = self.apply("where", (predicate, yes, value),
                                           (COORD, UNIFORM if isinstance(yes, Literal) else FRAG,
                                            UNIFORM if isinstance(value, Literal) else FRAG), rb, cb)
                    env[stmt.name] = value
                else:
                    raise ValueError(f"register map cannot emit {type(stmt).__name__}")
            result = tuple(env[name] for name in node.lift.results)
        self.cells[key] = result
        return result

    def operand(self, node, row, col, rb, cb):
        (value,) = self.cell(node, row, col, rb, cb)
        # A fragment can straddle either extent; all lanes still participate in MMA/shuffles.
        cond = BinaryExpr("&&", BinaryExpr("<", Var(FRAG_ROW), Literal(self.extents[row], "int")),
                          BinaryExpr("<", Var(FRAG_COL), Literal(self.extents[col], "int")))
        return self.apply("where", (cond, value, Literal(0.0)),
                          (COORD, UNIFORM if isinstance(value, Literal) else FRAG, UNIFORM), rb, cb)

    def contract(self, node, row, col, rb, cb):
        left, right = node.operands
        if row not in left.free_axes:
            left, right = right, left
        if row not in left.free_axes or row in right.free_axes or col in left.free_axes:
            raise ValueError("contraction does not preserve the register row ownership")
        axis = node.axis
        pairs = []
        for k in range(0, self.extents[axis], 16):
            a = self.pack("a", tuple(self.operand(left, row, axis, rb, Literal(k + d, "int")) for d in (0, 8)))
            b = self.pack("b", (self.operand(right, axis, col, Literal(k, "int"), cb),))
            pairs.append((a, b))
        key = ("mma", tuple(pairs), self.atom)
        if key not in self.memo:
            out = self.name()
            self.body.append(RegFragment(name=out, role="c", shape=self.atom.shape, dtype=F32))
            half = self.atom.operand_dtype("c").nbytes == 2
            partial = self.name() if half else out
            if half:
                self.body.append(RegFragment(name=partial, role="c", shape=self.atom.shape, dtype=self.atom.operand_dtype("c")))
            for step, (a, b) in enumerate(pairs, 1):
                self.body.append(MmaSyncPtx(c_frag=partial, a_frag=a, b_frag=b, shape=self.atom.ptx_shape,
                                            ab_dtype=self.atom.ab_dtype, c_dtype=self.atom.operand_dtype("c").name))
                if half and (step % self.period == 0 or step == len(pairs)):
                    self.body.append(FragmentPromote(dst=out, src=partial))
            self.memo[key] = out
        return self.memo[key]


def factorize_register(tile):
    """Run one ordered loop per CTA, with all previous-state reads preceding the commit."""
    program = tile.register_program
    choice = tile.schedule.kernel
    warps = choice.work.units[0]
    row_base = Literal(16, "int") * (Literal(warps, "int") * Var("_rb") + Var("_rw"))
    emit = _Fragments(tile, row_base)
    pending = []
    for spec, node in zip((*program.outputs, program.state), program.roots, strict=True):
        col, row = (expr.name for expr in spec.write.index[-2:])
        columns = emit.extents[col]
        values = []
        for j in range((columns + 7) // 8):
            cb = Literal(j * 8, "int")
            (value,) = emit.cell(node, row, col, row_base, cb)
            values.append(emit.fragment(value))
        pending.append((spec, values))
    # Outputs may observe both the previous state and results of this step. Emit all reads
    # before any carried register is overwritten, including readers outside the update cone.
    for spec, values in pending:
        for j, value in enumerate(values):
            index = (*spec.write.index[:-2], Literal(j * 8, "int"), row_base)
            emit.body.append(RegStore(dst_buffer=spec.write.output, dst_index=index, frag=value, shape=emit.atom.shape,
                                       row_dim=len(index) - 1, col_dim=len(index) - 2,
                                       m_guard=(row_base, Literal(program.rows, "int")),
                                       n_guard=(Literal(j * 8, "int"), Literal(emit.extents[spec.write.index[-2].name], "int"))))
    for state, value in zip(emit.states, pending[-1][1], strict=True):
        emit.body.append(FragmentApply(out=state, op=ElementwiseImpl("copy"), args=(value,), kinds=(FRAG,), in_place=True))
    declarations = tuple(RegFragment(name=name, role="c", shape=emit.atom.shape, dtype=F32) for name in emit.states)
    loop = StridedLoop(axis=tile.place.serial[0], start=Literal(0, "int"), step=Literal(1, "int"), body=Body(emit.body), unroll=False)
    axes = (*program.batch, Axis("_rb", (program.rows + warps * 16 - 1) // (warps * 16)), Axis("_rw", warps), Axis("_rl", 32))
    return Tile(axes=axes, body=Body((*declarations, loop)), block_threads=warps * 32)
