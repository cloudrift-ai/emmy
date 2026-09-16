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
