"""Static scalar binding becomes explicit before fusion, identity and CUDA rendering."""

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F16, F32, F64, I32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Let
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs

_RULE = "095_bind_scalar_constants"


def _program(value=0.1, *, dtype=F32, shape=(1,), producer=None, name="constant"):
    graph = Graph()
    producer = producer if producer is not None else ConstantOp(name, value=value)
    graph.add_node(producer, [], Tensor(name, shape, dtype), node_id=name)
    kernel = LoopOp(
        body=(
            Loop(
                Axis("i", 4),
                (
                    Load("loaded", name, tuple(Literal(0, "int") for _ in shape), dtype=dtype),
                    Write("out", (Var("i"),), "loaded", value_dtype=dtype),
                ),
            ),
        ),
    )
    graph.add_node(kernel, [name], Tensor("out", (4,), dtype), node_id="out")
    graph.inputs = [name] if isinstance(producer, InputOp) else []
    graph.outputs = ["out"]
    return graph


def _bind(graph):
    return Pipeline.build(["loop/lowering"], select=[_RULE]).run(graph)


@pytest.mark.parametrize("value", [0.1, 1e-6, 1024.0, 1.00000006, 0.0, -0.0, 1e-46, -1e-46])
def test_static_f32_binding_preserves_stored_bits(value):
    graph = _program(value)
    expected = NumpyBackend().run(graph)[0].outputs["out"]
    bound = _bind(graph)
    actual = NumpyBackend().run(bound)[0].outputs["out"]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    kernel = bound.nodes["out"].op
    assert not kernel.loads
    literal = next(stmt for stmt in kernel.body.iter() if isinstance(stmt, Let))
    assert literal.dtype == F32
    assert np.float32(literal.value.value).view(np.uint32) == np.float32(value).view(np.uint32)
    assert bound.nodes["out"].inputs == []
    assert _bind(bound).to_wire() == bound.to_wire()


@pytest.mark.parametrize(
    ("producer", "dtype", "shape"),
    [
        (InputOp(), F32, (1,)),
        (ConstantOp("constant", context_value=Var("seq_len")), F32, (1,)),
        (ConstantOp("constant", source_path="weight.scalar"), F32, (1,)),
        (ConstantOp("constant", value=0.1), F16, (1,)),
        (ConstantOp("constant", value=0.1), BF16, (1,)),
        (ConstantOp("constant", value=0.1), F64, (1,)),
        (ConstantOp("constant", value=1), I32, (1,)),
        (ConstantOp("constant", value=float("inf")), F32, (1,)),
        (ConstantOp("constant", value=float("nan")), F32, (1,)),
        (ConstantOp("constant", value=1e100), F32, (1,)),
        (ConstantOp("constant", value=0.1), F32, (2,)),
        (ConstantOp("constant", value=0.1), F32, (Dim("seq_len"),)),
    ],
)
def test_runtime_external_and_unsupported_storage_stay_loads(producer, dtype, shape):
    bound = _bind(_program(producer=producer, dtype=dtype, shape=shape))
    assert len(bound.nodes["out"].op.loads) == 1
    assert bound.nodes["out"].inputs == ["constant"]


def test_externally_bound_scalar_remains_a_runtime_value():
    bound = _bind(_program(producer=ConstantOp("constant")))
    for value in (0.1, 0.2):
        result = NumpyBackend().run(bound, input_data={"constant": np.array([value], np.float32)})[0]
        np.testing.assert_array_equal(result.outputs["out"], np.full((4,), value, np.float32))


@pytest.mark.parametrize("lift", [False, True])
def test_literal_value_participates_in_exact_identity_but_buffer_names_do_not(lift):
    def lower(graph):
        bound = _bind(graph)
        return Pipeline.build(["tile/lift"]).run(bound).nodes["out"].op if lift else bound.nodes["out"].op

    first = lower(_program(0.1))
    different = lower(_program(0.2))
    renamed = lower(_program(0.1, name="renamed"))
    key = lambda op: op.identity_key(structural=False, with_io=True)  # noqa: E731
    assert key(first) != key(different)
    assert key(first) == key(renamed)


def test_signed_zero_has_distinct_exact_identity():
    positive = _bind(_program(0.0)).nodes["out"].op
    negative = _bind(_program(-0.0)).nodes["out"].op
    assert positive.identity_key(structural=False, with_io=True) != negative.identity_key(structural=False, with_io=True)


def test_both_signed_zeros_survive_fusion_and_common_subexpressions():
    graph = Graph()
    graph.add_node(ConstantOp("positive", value=0.0), [], Tensor("positive", (1,), F32), node_id="positive")
    graph.add_node(ConstantOp("negative", value=-0.0), [], Tensor("negative", (1,), F32), node_id="negative")
    graph.add_node(InputOp(), [], Tensor("x", (4,), F32), node_id="x")
    positive = LoopOp(
        body=(
            Loop(
                Axis("i", 4),
                (
                    Load("element", "x", (Var("i"),), dtype=F32),
                    Load("zero", "positive", (Literal(0, "int"),), dtype=F32),
                    Assign("quotient", "divide", ("element", "zero"), dtype=F32),
                    Write("positive_values", (Var("i"),), "quotient", value_dtype=F32),
                ),
            ),
        )
    )
    graph.add_node(positive, ["x", "positive"], Tensor("positive_values", (4,), F32), node_id="positive_values")
    negative = LoopOp(
        body=(
            Loop(
                Axis("i", 4),
                (
                    Load("element", "x", (Var("i"),), dtype=F32),
                    Load("zero", "negative", (Literal(0, "int"),), dtype=F32),
                    Load("positive_value", "positive_values", (Var("i"),), dtype=F32),
                    Assign("quotient", "divide", ("element", "zero"), dtype=F32),
                    Write("out_positive", (Var("i"),), "positive_value", value_dtype=F32),
                    Write("out_negative", (Var("i"),), "quotient", value_dtype=F32),
                ),
            ),
        )
    )
    graph.add_node(
        negative,
        ["x", "negative", "positive_values"],
        outputs=[Tensor(name, (4,), F32) for name in ("out_positive", "out_negative")],
        node_id="out_positive",
    )
    graph.inputs = ["x"]
    graph.outputs = ["out_positive", "out_negative"]
    inputs = {"x": np.array([1, -1, 2, -2], np.float32)}
    with np.errstate(divide="ignore"):
        expected = NumpyBackend().run(graph, input_data=inputs)[0].outputs
        lowered = Pipeline.build(LOOP_PASSES).run(graph)
        actual = NumpyBackend().run(lowered, input_data=inputs)[0].outputs
    assert sum(isinstance(node.op, LoopOp) for node in lowered.nodes.values()) == 1
    for name in graph.outputs:
        np.testing.assert_array_equal(actual[name].view(np.uint32), expected[name].view(np.uint32))


def test_scalar_binding_removes_only_its_cuda_argument(monkeypatch):
    monkeypatch.setenv("EMMY_GOLDEN_FILE", "")
    graph = Graph()
    graph.add_node(ConstantOp("constant", value=0.1), [], Tensor("constant", (1,), F32), node_id="constant")
    graph.add_node(InputOp(), [], Tensor("x", (4,), F32), node_id="x")
    graph.inputs = ["x"]
    body = (
        Load("loaded", "constant", (Literal(0, "int"),), dtype=F32),
        Load("element", "x", (Var("i"),), dtype=F32),
        Assign("sum", "add", ("loaded", "element"), dtype=F32),
        Write("out", (Var("i"),), "sum", value_dtype=F32),
    )
    kernel = LoopOp(body=(Loop(Axis("i", 4), body),))
    graph.add_node(kernel, ["constant", "x"], Tensor("out", (4,), F32), node_id="out")
    graph.outputs = ["out"]
    with pinned_knobs({"PLACE": "fuse", "WORK": "t32", "REDUCE": "", "LOOPIFY": 0}):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((12, 0)))
    (cuda,) = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda.arg_order == ("x", "out")
    assert "constant[" not in cuda.kernel_source
    assert "0.10000000149011612" in cuda.kernel_source
