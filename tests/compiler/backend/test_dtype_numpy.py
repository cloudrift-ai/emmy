"""End-to-end fp16 elementwise chain on the numpy backend.

Pure-Python paths only; the CUDA fp16 path is exercised in
``tests/compiler/test_dtype_cuda.py`` (step 4).
"""

from __future__ import annotations

import numpy as np

from emmy.compiler import dtype as dt
from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dtype import DataType
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp, ElementwiseOp


def test_datatype_resolution_aliases():
    assert dt.get("float16") is dt.F16
    assert dt.get("half") is dt.F16
    assert dt.get("f16") is dt.F16
    assert dt.get(dt.F16) is dt.F16
    assert dt.F16.nbytes == 2
    assert dt.F32.nbytes == 4
    assert str(dt.F16) == "f16"


def test_int_datatype_resolution_aliases():
    # Integer dtypes appear on placeholder inputs from HF whole-model traces
    # (``input_ids``, ``position_ids``). The canonical name + numpy/PyTorch
    # aliases must all resolve to the same DataType singleton.
    assert dt.get("i32") is dt.I32
    assert dt.get("int32") is dt.I32
    assert dt.get("i64") is dt.I64
    assert dt.get("int64") is dt.I64
    assert dt.get("long") is dt.I64
    assert dt.I32.nbytes == 4
    assert dt.I64.nbytes == 8
    assert dt.I32.np == np.dtype(np.int32)
    assert dt.I64.np == np.dtype(np.int64)


def test_tensor_dtype_coerces_string():
    t = Tensor("a", (4,), "float16")
    assert isinstance(t.dtype, DataType)
    assert t.dtype is dt.F16


def test_numpy_backend_elementwise_chain_fp16():
    """Build a tiny exp -> negate -> add chain on fp16 inputs; compare to numpy eager."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8,), dt.F16), node_id="x")
    g.inputs.append(x)

    e = g.add_node(
        op=ElementwiseOp(op=np.exp),
        inputs=[x],
        output=Tensor("e", (8,), dt.F16),
        node_id="e",
    )
    n = g.add_node(
        op=ElementwiseOp(op=np.negative),
        inputs=[e],
        output=Tensor("n", (8,), dt.F16),
        node_id="n",
    )
    g.outputs.append(n)

    rng = np.random.default_rng(0)
    x_data = rng.standard_normal(8).astype(np.float16)

    be = NumpyBackend()
    out = be.run(be.compile(g), input_data={"x": x_data})[0].outputs["n"]

    assert out.dtype == np.float16, f"expected float16 output, got {out.dtype}"
    expected = (-np.exp(x_data.astype(np.float32))).astype(np.float16)
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)


def test_numpy_backend_bf16_numeric_ops_use_bit_carrier():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (3,), dt.BF16), node_id="x")
    g.add_node(ElementwiseOp(op="add"), ["x", "x"], Tensor("twice", (3,), dt.BF16), node_id="twice")
    g.add_node(CastOp(dtype="f32"), ["twice"], Tensor("decoded", (3,), dt.F32), node_id="decoded")
    g.add_node(BitcastOp(dtype="u16"), ["twice"], Tensor("bits", (3,), dt.U16), node_id="bits")
    g.outputs = ["twice", "decoded", "bits"]
    g.inputs = ["x"]

    values = np.array([1.0, -2.0, 3.140625], dtype=np.float32)
    expected = dt.encode_bf16(values * 2)
    backend = NumpyBackend()
    for supplied in (values, dt.encode_bf16(values)):
        got = backend.run(g, input_data={"x": supplied})[0].outputs
        np.testing.assert_array_equal(got["twice"], expected)
        np.testing.assert_array_equal(got["decoded"], dt.decode_bf16(expected))
        np.testing.assert_array_equal(got["bits"], expected)


def test_numpy_backend_bf16_computed_constant_encodes_values():
    from emmy.compiler.loader.binder import evaluate_source_graph
    from emmy.compiler.loader.quant import _f4_pair_table

    graph = Graph()
    table = _f4_pair_table(graph, name="pairs", out_name="pairs", dtype=dt.BF16)
    got = evaluate_source_graph(graph.nodes[table].op.source_graph, {})
    expected = np.stack((np.tile(dt.F4_VALUES, 16), np.repeat(dt.F4_VALUES, 16)), axis=1)
    np.testing.assert_array_equal(got, dt.encode_bf16(expected))
