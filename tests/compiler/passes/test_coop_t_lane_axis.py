"""The ``coop-t`` band lays its 32 lanes on the output axis its weight reads run along."""

from importlib import import_module

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Body, Loop
from emmy.compiler.ir.stmt.leaves import Load

_coalescing_axis = import_module("emmy.compiler.pipeline.passes.lowering.kernel._factor")._coalescing_axis

_HEAD_DIM, _HEADS, _K = Axis("d", Dim(128)), Axis("h", Dim(16)), Axis("k", Dim(1024))
_INPUTS = {"w": Tensor("w", (Dim(1024), Dim(2048)), dtype=F16)}


def _loop(column) -> Loop:
    return Loop(axis=_K, body=Body((Load(name="wv", input="w", index=(Var("k"), column), dtype=F16),)))


def test_the_lanes_follow_the_contiguous_output_axis() -> None:
    """A q projection read per head: the head dim is contiguous in the weight, the head strides it
    by 128. The tile declares the head last, which is where the band used to put its lanes."""
    column = Var("d") + Literal(128, "int") * Var("h")
    assert _coalescing_axis(_loop(column), (_HEAD_DIM, _HEADS), _INPUTS) is _HEAD_DIM


def test_without_a_contiguous_axis_the_last_axis_keeps_the_lanes() -> None:
    column = Literal(128, "int") * Var("d") + Literal(2, "int") * Var("h")
    assert _coalescing_axis(_loop(column), (_HEAD_DIM, _HEADS), _INPUTS) is _HEADS
