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
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Let, Load, parse_body
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, statements

_RULE = "095_bind_scalar_constants"


def _program(value=0.1, *, dtype=F32, shape=(1,), producer=None, name="constant"):
    graph = Graph()
    producer = producer if producer is not None else ConstantOp(name, value=value)
    graph.add_node(producer, [], Tensor(name, shape, dtype), node_id=name)
    kernel = LoopOp(
        body=parse_body(
            f"""
            for i in 0..4:
                {dtype.name} loaded = load {name}[{", ".join("0" for _ in shape)}]
                {dtype.name} out[i] = loaded
            """
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
        body=parse_body(
            """
            for i in 0..4:
                f32 element = load x[i]
                f32 zero = load positive[0]
                f32 quotient = divide(element, zero)
                f32 positive_values[i] = quotient
            """
        )
    )
    graph.add_node(positive, ["x", "positive"], Tensor("positive_values", (4,), F32), node_id="positive_values")
    negative = LoopOp(
        body=parse_body(
            """
            for i in 0..4:
                f32 element = load x[i]
                f32 zero = load negative[0]
                f32 positive_value = load positive_values[i]
                f32 quotient = divide(element, zero)
                f32 out_positive[i] = positive_value
                f32 out_negative[i] = quotient
            """
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
    kernel = LoopOp(
        body=parse_body(
            """
            for i in 0..4:
                f32 loaded = load constant[0]
                f32 element = load x[i]
                f32 sum = add(loaded, element)
                f32 out[i] = sum
            """
        )
    )
    graph.add_node(kernel, ["constant", "x"], Tensor("out", (4,), F32), node_id="out")
    graph.outputs = ["out"]
    with pinned_knobs({"PLACE": "fuse", "WORK": "t32", "REDUCE": "", "LOOPIFY": 0}):
        lowered, kernels = compile_with_kernels(graph, ctx=Context.from_target((12, 0)))
    (cuda,) = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda.arg_order == ("x", "out")
    assert not [load for load in statements(kernels, Load) if load.input == "constant"]
    # The constant is an immediate of the code: 0.1 as the nearest f32.
    assert "0f3DCCCCCD" in cuda.kernel_source
