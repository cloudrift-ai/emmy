"""Every IR object on the wire — expressions, dims, ops, programs and Loop IR kernels — round-trips through its class."""

from __future__ import annotations

import copy
import json

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.frontend.ir import (
    CatOp,
    Conv1dOp,
    LayerNormOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    RmsNormOp,
    SdpaOp,
    SliceOp,
    SoftmaxOp,
    TransposeOp,
    UnsqueezeOp,
)
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import (
    BitcastOp,
    CastOp,
    ElementwiseOp,
    GatherOp,
    IndexMapOp,
    IndexSource,
    RangeOp,
    ReduceOp,
    ScanOp,
    ScatterOp,
)
from emmy.compiler.tensor import Tensor
from emmy.compiler.wire import decode, encode, intern


@pytest.mark.parametrize(
    "expr",
    [
        Var("s"),
        Builtin("thread_idx_x"),
        Literal(3, "int"),
        BinaryExpr("+", Var("s"), Literal(2, "int")),
        FuncCallExpr("maximum", (Var("x"), Literal(0.0))),
        TernaryExpr(Var("p"), Literal(1, "int"), Literal(0, "int")),
        CastExpr("int", Var("x")),
    ],
)
def test_expression_round_trip(expr):
    assert decode(encode(expr)) == expr


def test_dimension_round_trip_preserves_composite_expression_and_hint():
    dims = [Dim(32), Dim("seq", hint=17), Dim(BinaryExpr("*", Var("seq"), Literal(2, "int")))]
    restored = [Dim.from_wire(dim.to_wire()) for dim in dims]
    assert [dim.expr for dim in restored] == [dim.expr for dim in dims]
    assert [dim.hint for dim in restored] == [dim.hint for dim in dims]


@pytest.mark.parametrize(
    "op",
    [
        InputOp(),
        ConstantOp(name="c", value=1.5),
        ConstantOp(name="ctx", context_value=Var("seq")),
        ConstantOp(name="w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f16"),
        ConstantOp(name="parts", source_parts=(("a", (4, 8)), ("b", (4, 8))), source_shape=(8, 8), source_dtype="f16"),
        TransposeOp((0, 1)),
        ReshapeOp((2, -1)),
        SliceOp((2, 4), dim=1, start=2),
        CatOp(),
        UnsqueezeOp(1),
        LinearOp(has_bias=True),
        MatmulOp(has_bias=True),
        SdpaOp(is_causal=True, sliding_window=128, scale=0.5),
        MeanOp(axis=-1),
        RmsNormOp(eps=1e-5),
        LayerNormOp(eps=1e-4),
        SoftmaxOp(axis=-1),
        RangeOp(0, 8, 2, "i64"),
        CastOp("f16"),
        BitcastOp("u16"),
        ElementwiseOp("silu"),
        ReduceOp("sum", -1),
        ScanOp("sum", 0),
        GatherOp(axis=1),
        ScatterOp(axis=1, reduce_fn="sum"),
        IndexMapOp(
            out_shape=(Dim("seq", hint=8), Dim(4)),
            sources=(
                IndexSource(
                    input_idx=0,
                    coord_map=(Var("out_coord_0"), BinaryExpr("%", Var("out_coord_1"), Literal(4, "int"))),
                    select=BinaryExpr("<", Var("out_coord_0"), Var("seq")),
                ),
            ),
        ),
    ],
)
def test_operation_round_trip(op):
    restored = decode(json.loads(json.dumps(encode(op))))
    assert restored == op


def test_conv1d_operation_wire_schema_round_trip():
    op = Conv1dOp(stride=2, padding=3, dilation=4, groups=5)
    wire = encode(op)

    assert wire == {"torch.conv1d": {"stride": 2, "padding": 3, "dilation": 4, "groups": 5}}
    assert decode(json.loads(json.dumps(wire))) == op


def _program() -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim("seq", hint=9), Dim(8)), "f16"), node_id="x")
    graph.add_node(ConstantOp(name="one", value=1.0), [], Tensor("one", (1,), "f16"), node_id="one")
    graph.add_node(ElementwiseOp("add"), ["x", "one"], Tensor("y", (Dim("seq", hint=9), Dim(8)), "f16"), node_id="y")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    return graph


def test_program_round_trip_is_deterministic():
    wire = _program().to_wire()
    assert set(wire) == {"inputs", "outputs", "nodes"}
    assert [node["id"] for node in wire["nodes"]] == ["one", "x", "y"]
    input_node = wire["nodes"][1]
    assert input_node == {
        "id": "x",
        "op": "input",
        "outputs": [["x", "f16", [{"sym": "seq", "hint": 9}, 8]]],
    }
    assert "attrs" not in input_node
    assert "inputs" not in input_node
    restored = Graph.from_wire(json.loads(json.dumps(wire)))
    assert restored.to_wire() == wire


def test_program_pool_uses_document_local_indexes_and_deduplicates():
    programs = []
    assert intern(programs, _program()) == 0
    assert intern(programs, _program()) == 0
    assert programs == [_program().to_wire()]


def test_constant_nested_load_ops_and_source_program_round_trip():
    source = Graph()
    source.add_node(ConstantOp(name="weight", source_path="weight"), [], Tensor("weight", (2, 3), "f16"), node_id="weight")
    source.inputs, source.outputs = [], ["weight"]
    original = ConstantOp(
        name="folded",
        load_ops=(TransposeOp((1, 0)),),
        source_shape=(3, 2),
        source_dtype="f16",
        source_graph=source,
    )

    wire = encode(original)
    restored = decode(json.loads(json.dumps(wire)))

    assert encode(restored) == wire


def test_program_rejects_unknown_ops_and_fields():
    wire = _program().to_wire()
    unknown_op = copy.deepcopy(wire)
    unknown_op["nodes"][-1]["op"] = "torch.future"
    with pytest.raises(ValueError, match="unknown op"):
        Graph.from_wire(unknown_op)

    unknown_field = copy.deepcopy(wire)
    unknown_field["nodes"][-1]["attrs"]["future"] = True
    with pytest.raises(ValueError, match="unknown field"):
        Graph.from_wire(unknown_field)

    removed_version = copy.deepcopy(wire)
    removed_version["ir_version"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        Graph.from_wire(removed_version)


def test_loop_graph_round_trip_preserves_structural_body_and_composite_dims() -> None:
    extent = Dim(BinaryExpr("*", Var("seq"), Literal(2, "int")))
    body = Body(
        (
            Loop(
                Axis("a0", extent),
                Body(
                    (
                        Load(name="in0", input="x", index=(Var("a0"),)),
                        Assign(name="v0", op=ElementwiseImpl("relu"), args=("in0",)),
                        Write(output="y", index=(Var("a0"),), value="v0"),
                    )
                ),
            ),
        )
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (extent,), "f16"), node_id="x")
    graph.add_node(LoopOp(body=body, name="k_relu"), ["x"], Tensor("y", (extent,), "f16"), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]

    wire = json.loads(json.dumps(graph.to_wire()))
    restored = Graph.from_wire(wire)

    assert restored.nodes["y"].op.body == body
    assert restored.nodes["y"].op.name == "k_relu"
    assert restored.buffer("y").shape == (extent,)
    assert restored.to_wire() == wire
