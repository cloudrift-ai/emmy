"""The reduce tier's ILP replication and the names it must NOT rename.

``_tile_reduce_axis`` copies the reduce body once per register chain, suffixing per-copy SSA
names. A name the body READS but does not DEFINE is one shared value (a hoisted scalar load, a
provider chain emitted ahead of the loop) — renaming its uses emits an undefined identifier at
nvcc. The Expr channel (``s.exprs()``) was already protected; ``Assign`` args travel the
SSA-deps channel (``s.deps()``), surfaced by DeepSeek-V4 post4096's two-cut piece
(``identifier "in3__r1" is undefined``).
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.schedule import Reduce
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering.kernel._factor import Ctx, _tile_reduce_axis, combine_tail


def _names_read(stmts) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        out.update(s.deps())
        for e in s.exprs():
            out.update(e.free_vars())
        for b in s.nested():
            out |= _names_read(b)
    return out


def test_an_external_ssa_read_is_shared_by_every_ilp_copy() -> None:
    from tests.compiler.terms import projection, reduction, slab

    # ``alpha`` is a per-row value the fold closes over through an operand term — hoisted ahead
    # of the loop by the lowering, one binding every ILP copy reads.
    alpha = projection((), (Load(name="alpha", input="s", index=(Var("m"),)),), results=("alpha",))
    scaled = Assign(name="acc__v", op="multiply", args=("x_e", "alpha"))
    red = reduction("k", (slab("x_e", "x", "m", "k"), alpha), (scaled,), ("acc",))
    axes = (Axis("m", 4), Axis("k", 128))
    sched = SimpleNamespace(
        get=lambda *_: None, tile=SimpleNamespace(axes=axes), axis_of=lambda name: next(a for a in axes if a.name == name)
    )
    ctx = Ctx(grid=(Axis("m", 4),), inputs={}, output="o", sched=sched)

    _state, fold, _close, _lane = _tile_reduce_axis(red, Reduce.of(reg=2), ctx, tail=(), out_val="acc")

    read = _names_read(fold)
    assert "alpha__r1" not in read, "the shared external value must not be renamed per copy"
    assert "alpha" in read
    assert any(name.endswith("__r1") for name in read), "the per-copy chains themselves still rename"


@pytest.mark.parametrize(("op", "dtype"), [("maximum", F16), ("add", F32)])
def test_ilp_merge_keeps_the_streaming_accumulator_dtype(op, dtype):
    """Register partials merge into the existing carrier; they cannot change its declaration's dtype."""
    from tests.compiler.terms import reduction, slab

    axis = Axis("k", 2)
    fold = reduction(axis, (slab("x_e", "x", "k"),), (Assign("acc__v", "copy", ("x_e",)),), ("acc",), op)
    body = Body(
        (
            Loop(
                axis=axis,
                body=(
                    Load(name="a", input="x", index=(Var("k"),)),
                    Accum(name="acc", value="a", op=op),
                    Load(name="b", input="x", index=(Var("k") + 2,)),
                    Accum(name="acc__r1", value="b", op=op),
                ),
            ),
            *combine_tail(fold, reg=2, coop=1, lane=None),
        )
    )
    kernel = KernelOp(name="ilp_merge", body=body, inputs={"x": Tensor("x", (4,), F16)})
    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(
        Node(id="out", op=kernel, inputs=["x"], outputs=())
    )
    accumulators = [stmt for stmt in stamped.body.iter() if isinstance(stmt, Accum)]
    assert len(accumulators) == 3
    assert {stmt.dtype for stmt in accumulators} == {dtype}
