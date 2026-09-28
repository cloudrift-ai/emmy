"""Dependency-safety tests for Kernel-IR load interleaving."""

from importlib import import_module

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load

_vectorize_loads = import_module("emmy.compiler.pipeline.passes.lowering.kernel.050_vectorize_loads")._vectorize_body
_vectorize_stores = import_module("emmy.compiler.pipeline.passes.lowering.kernel.080_vectorize_stores")._vectorize_body
_interleave = import_module("emmy.compiler.pipeline.passes.lowering.kernel.095_interleave_loads")
_sink_loads = _interleave._sink_loads
_pair_ldmatrix = import_module("emmy.compiler.pipeline.passes.lowering.kernel.096_pair_ldmatrix_loads")._walk


def test_noop_kernel_peepholes_reuse_the_body() -> None:
    inner = Body(
        (
            Load(name="x", input="input", index=(Literal(0, "int"),), dtype=F32),
            Assign(name="y", op=ElementwiseImpl("abs"), args=("x",), dtype=F32),
        )
    )
    body = Body((Loop(axis=Axis("k", Dim(4)), body=inner),))
    op = KernelOp(body=body)
    interleaved, interleave_changed = _interleave._walk(body)
    paired, pair_changed = _pair_ldmatrix(body)

    assert _vectorize_loads(op, body) is body
    assert _vectorize_stores(op, body) is body
    assert interleaved is body and not interleave_changed
    assert paired is body and not pair_changed


def test_interleave_keeps_load_before_nested_consumer() -> None:
    """A load used in a serial reduction loop must remain in the enclosing scope."""
    invariant = Load(name="epsilon", input="epsilon_buffer", index=(Literal(0, "int"),), dtype=F32)
    loop = Loop(
        axis=Axis("k", Dim(4)),
        body=Body(
            (
                Load(name="x", input="input", index=(Literal(0, "int"),), dtype=F32),
                Assign(name="sum_term", op=ElementwiseImpl("add"), args=("epsilon", "x"), dtype=F32),
                Accum(name="sum", value="sum_term", op=ElementwiseImpl("add"), dtype=F32),
            )
        ),
    )
    trailing = Assign(name="result", op=ElementwiseImpl("add"), args=("sum", "epsilon"), dtype=F32)

    reordered = tuple(_sink_loads(Body((invariant, loop, trailing))))

    assert reordered.index(invariant) < reordered.index(loop)


def _unrolled_pointwise(*, between=()) -> Body:
    """Four unrolled cells of ``out = x * x``: each cell loads its f16 element and squares it
    before the next cell's load, as the unroller emits a pointwise body."""
    from emmy.compiler.dtype import F16
    from emmy.compiler.ir.expr import Var

    stmts = []
    for u in range(4):
        stmts.append(Load(name=f"x{u}", input="x", index=(Var("a0") * Literal(4, "int") + Literal(u, "int"),), dtype=F16))
        if u == 2:
            stmts.extend(between)
        stmts.append(Assign(name=f"y{u}", op=ElementwiseImpl("multiply"), args=(f"x{u}", f"x{u}"), dtype=F16))
    return Body(tuple(stmts))


def test_interleaved_loads_of_one_buffer_widen() -> None:
    """Consecutive elements loaded between other work still form one vector load: an unrolled
    pointwise body never places them side by side, and global f16 loads left 2 bytes wide cost
    the kernel most of its bandwidth."""
    out = _vectorize_loads(KernelOp(body=Body(())), _unrolled_pointwise())

    loads = [stmt for stmt in out if isinstance(stmt, Load)]
    assert len(loads) == 1 and loads[0].names == ("x0", "x1", "x2", "x3")
    assert out.index(loads[0]) == 0, "the widened load sits where the first element was loaded"


def test_a_load_whose_index_is_computed_in_between_stays_put() -> None:
    """A later load moves up only when its index needs nothing defined in between."""
    from emmy.compiler.dtype import F16
    from emmy.compiler.ir.expr import Var

    body = Body(
        (
            Load(name="x0", input="x", index=(Var("a0") * Literal(2, "int"),), dtype=F16),
            Assign(name="j", op=ElementwiseImpl("abs"), args=("x0",), dtype=F16),
            Load(name="x1", input="x", index=(Var("a0") * Literal(2, "int") + Var("j"),), dtype=F16),
        )
    )
    assert _vectorize_loads(KernelOp(body=body), body) is body


def test_a_write_to_the_buffer_in_between_stops_the_run() -> None:
    """A load never moves above a store into its own buffer."""
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt.leaves import Write

    store = Write(output="x", index=(Var("a0"),), values=("y0",))
    out = _vectorize_loads(KernelOp(body=Body(())), _unrolled_pointwise(between=(store,)))

    widths = sorted(len(stmt.names) for stmt in out if isinstance(stmt, Load))
    assert widths == [1, 1, 2], "x0 and x1 pair up; x3 cannot move above the store to meet x2"
