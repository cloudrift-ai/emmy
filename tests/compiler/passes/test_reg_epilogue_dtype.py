"""Register epilogues retain the Loop tail's per-Assign dtype semantics."""

import numpy as np

from emmy.compiler.dtype import F16, F32, I32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, RenderCtx, Write
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue
from tests.compiler.helpers import requires_cuda, requires_sm90


def _render(*assigns) -> tuple[str, object]:
    tail = [*assigns, Write(output="out", index=(Var("m"), Var("n")), value=assigns[-1].name)]
    epilogue = _warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)
    assert epilogue is not None
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("m"), Var("n")),
        frag="_c",
        shape=(16, 8, 16),
        ldm=16,
        epilogue=epilogue,
    )
    ctx = RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})
    return "\n".join(store.render(ctx)), epilogue


def test_typed_copy_narrows_the_fragment_value_before_its_consumer() -> None:
    source, epilogue = _render(
        Assign(name="narrow", op=ElementwiseImpl("copy"), args=("acc",), dtype=F16),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("narrow",)),
    )

    assert epilogue.body[0].dtype is F16
    assert "const __half narrow_e0 = __float2half(_c[0]);" in source
    assert "const __half result_e0 = narrow_e0;" in source


def test_untyped_copy_keeps_the_existing_f32_epilogue() -> None:
    source, epilogue = _render(Assign(name="result", op=ElementwiseImpl("copy"), args=("acc",)))

    assert epilogue.body[0].dtype is None
    assert "const float result_e0 = _c[0];" in source
    assert "__float2half(_c[0])" not in source


def test_transposed_fragment_store_uses_both_output_strides() -> None:
    store = RegStore(
        dst_buffer="out",
        dst_index=(Var("n"), Var("m")),
        frag="_c",
        shape=(16, 8, 16),
        row_dim=1,
        col_dim=0,
    )
    source = "\n".join(store.render(RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "f16"})))

    assert "reinterpret_cast<__half2*>" not in source
    assert "(_t * 2 + 1) * 16" in source
    assert "out[n * 16 + m + _g + (_t * 2 + 0) * 16]" in source


def test_a_mask_over_an_op_result_renders_after_that_op() -> None:
    """A causal mask may select a value the chain computes (a scaled score), not only a loaded one:
    the ternary renders once that op has run."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.stmt import Load, Select, SelectBranch

    source, _ = _render(
        Load(name="ninf", input="neg_inf", index=(Literal(0, "int"),)),
        Assign(name="scaled", op=ElementwiseImpl("multiply"), args=("acc", "acc")),
        Select(
            name="masked",
            branches=(SelectBranch("scaled", BinaryExpr("<=", Var("n"), Var("m"))), SelectBranch("ninf", Literal(1, "int"))),
        ),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("masked",)),
    )

    assert source.index("scaled_e0 =") < source.index("masked_e0 = ((") < source.index("result_e0 = masked_e0")


def test_a_mask_over_a_narrowed_value_widens_both_branches() -> None:
    """A chain op keeps the tail's dtype, so a mask over a narrowed value converts back: a ternary
    mixing ``__half`` and ``float`` does not compile."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.stmt import Load, Select, SelectBranch

    source, _ = _render(
        Load(name="ninf", input="neg_inf", index=(Literal(0, "int"),)),
        Assign(name="narrow", op=ElementwiseImpl("copy"), args=("acc",), dtype=F16),
        Select(
            name="masked",
            branches=(SelectBranch("narrow", BinaryExpr("<=", Var("n"), Var("m"))), SelectBranch("ninf", Literal(1, "int"))),
        ),
        Assign(name="result", op=ElementwiseImpl("copy"), args=("masked",)),
    )

    assert "const __half narrow_e0 = __float2half(_c[0]);" in source
    assert "const float masked_e0 = ((n + (_t * 2 + 0) <= m + _g) ? __half2float(narrow_e0) : ninf_e0);" in source


def test_integer_load_in_fragment_epilogue_keeps_shift_operands_integer() -> None:
    """A packed output's shift amount must reach the epilogue as an integer, like a scalar Load."""
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.stmt import Load

    tail = [
        Load(name="shift", input="shift", index=(Literal(0, "int"),), dtype=I32),
        Assign(name="code", op="to_f4e2m1", args=("acc",), dtype=I32),
        Assign(name="bits", op="left_shift", args=("code", "shift"), dtype=I32),
        Write(output="out", index=(Var("m"), Var("n")), value="bits"),
    ]
    epilogue = _warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)
    assert epilogue is not None
    store = RegStore(dst_buffer="out", dst_index=(Var("m"), Var("n")), frag="_c", shape=(16, 8, 16), ldm=16, epilogue=epilogue)
    ctx = RenderCtx(shapes={"out": (16, 16), "shift": (1,)}, buffer_dtypes={"out": "i32", "shift": "i32"})
    source = "\n".join(store.render(ctx))

    assert "const int shift_e0 = shift[0];" in source
    assert "int bits_e0 = code_e0 << shift_e0;" in source

    literal = RenderCtx(shapes={"out": (16, 16)}, buffer_dtypes={"out": "i32"}, literal_constants={"shift": 4.0})
    literal_source = "\n".join(store.render(literal))
    assert "const int shift_e0 = 4;" in literal_source
    assert "int bits_e0 = code_e0 << shift_e0;" in literal_source


@requires_sm90
@requires_cuda
def test_integer_shift_stays_in_fused_mma_epilogue_on_gpu(monkeypatch) -> None:
    """An integer load and shift after MMA compute exactly the packed-bit oracle in one kernel."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp

    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f4x8/k2")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "")
    graph = Graph()
    for name, shape, dtype in (("a", (128, 128), F16), ("b", (128, 128), F16), ("shift", (128, 128), I32)):
        graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("mm", (128, 128), F32), node_id="mm")
    graph.add_node(ElementwiseOp("copy"), ["mm"], Tensor("code", (128, 128), I32), node_id="code")
    graph.add_node(ElementwiseOp("left_shift"), ["code", "shift"], Tensor("out", (128, 128), I32), node_id="out")
    graph.inputs, graph.outputs = ["a", "b", "shift"], ["out"]

    rng = np.random.default_rng(17)
    a = rng.integers(-2, 3, size=(128, 128)).astype(np.float16)
    b = rng.integers(-2, 3, size=(128, 128)).astype(np.float16)
    shift = rng.integers(0, 4, size=(128, 128), dtype=np.int32)
    backend = CudaBackend()
    compiled = backend.compile(graph)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None)]
    assert len(sources) == 1 and "mma.sync.aligned.m16n8k16" in sources[0] and " << " in sources[0]
    got = backend.run(compiled, input_data={"a": a, "b": b, "shift": shift})[0].outputs["out"]
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.int32) << shift
    np.testing.assert_array_equal(got, expected)
