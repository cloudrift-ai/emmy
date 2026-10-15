"""Register epilogues retain the Loop tail's per-Assign dtype semantics."""

import numpy as np
import pytest

from emmy.compiler.dtype import F16, F32, I32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import RegStore
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Let, RenderCtx, Write
from emmy.compiler.pipeline import RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue
from tests.compiler.helpers import requires_cuda, requires_sm, requires_sm90


def test_fp4_encode_in_register_epilogue_emits_its_helper() -> None:
    from emmy.compiler.ir.cuda.ir import CudaOp
    from tests.compiler.realization.helpers import CASES_DIR, load_case, lowered

    case = load_case(CASES_DIR / "matmul/mma-fp4-encode-epilogue.json")
    compiled, _ = lowered(case, case.context())
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1
    source = sources[0]
    assert "mma.sync.aligned.m16n8k16" in source
    assert "emmy_to_f4e2m1(" in source
    assert "unsigned char emmy_to_f4e2m1(float value)" in source


@requires_cuda
@requires_sm(12, 0)
@pytest.mark.parametrize("runtime_rows", [False, True], ids=["width16", "runtime65"])
def test_packed_fp4_epilogue_reads_each_elements_block_scale(runtime_rows) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.compiler.realization.helpers import CASES_DIR, load_case

    case = load_case(CASES_DIR / "matmul/mma-fp4-packed-block-scales.json")
    program = case.document.programs[0]
    rows = 65 if runtime_rows else 16
    if runtime_rows:
        for node in program["nodes"]:
            shapes = [output[2] for output in node["outputs"]]
            if "out_shape" in node.get("attrs", {}):
                shapes.append(node["attrs"]["out_shape"])
            for shape in shapes:
                if shape[0] == 16:
                    shape[0] = {"sym": "num_tokens", "hint": rows}
    graph = Graph.from_wire(program)
    rng = np.random.default_rng(17)
    a = rng.integers(-2, 3, (rows, 32)).astype(np.float16)
    b = rng.integers(-2, 3, (32, 128)).astype(np.float16)
    scales = rng.choice(np.array([8, 12, 20, 28], np.float32), (rows, 8))
    backend = CudaBackend()
    with pinned_knobs({**case.row.pins, **case.row.knobs}):
        compiled = backend.compile(graph)
    sources = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(sources) == 1 and "mma.sync.aligned.m16n8k16" in sources[0]
    assert "emmy_to_f4e2m1(" in sources[0]
    got = backend.run(compiled, input_data={"a": a, "b": b, "scales": scales})[0].outputs["packed"]

    normalized = (a.astype(np.float32) @ b.astype(np.float32)) / np.repeat(scales, 16, axis=1)
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    midpoints = (levels[:-1] + levels[1:]) / 2
    magnitude = np.abs(normalized)
    codes = np.searchsorted(midpoints, magnitude).astype(np.uint8)
    ties = magnitude == midpoints[np.minimum(codes, len(midpoints) - 1)]
    codes += (ties & ((codes & 1) != 0)).astype(np.uint8)
    codes |= np.signbit(normalized).astype(np.uint8) * 8
    expected = codes[:, ::2] | (codes[:, 1::2] << 4)
    assert np.unique(expected).size > 50
    np.testing.assert_array_equal(got, expected)


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


def test_typed_literal_binding_survives_fragment_epilogue_rendering() -> None:
    source, epilogue = _render(
        Let("scale", Literal(0.125), dtype=F32),
        Assign("result", "multiply", ("acc", "scale")),
    )
    assert epilogue.body[0] == Let("scale", Literal(0.125), dtype=F32)
    assert "const float scale_e0 = 0.125f;" in source
    assert "const float result_e0 = _c[0] * scale_e0;" in source


def test_fragment_epilogue_substitutes_binding_coordinates_and_rejects_unbound_values() -> None:
    source, _ = _render(Let("column", Var("n"), dtype=I32), Assign("result", "multiply", ("acc", "column")))
    assert "const int column_e0 = n + (_t * 2 + 0);" in source
    with pytest.raises(RuleSkipped, match="projection epilogue reads.*missing"):
        _render(Let("scale", Var("missing"), dtype=F32), Assign("result", "multiply", ("acc", "scale")))


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("fast_exp", [None, False, True])
def test_fast_exp_policy_reaches_generated_fragment_epilogues(monkeypatch, fast_math, fast_exp) -> None:
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.ir.kernel import KernelOp
    from emmy.compiler.pipeline import FINAL_LOWERING_PASS, KERNEL_PASSES, Pipeline
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    monkeypatch.delenv("EMMY_FAST_EXP", raising=False)
    graph = graph_from_code("F.silu(torch.randn(64, 64, dtype=torch.float16) @ torch.randn(64, 128, dtype=torch.float16))")[0]
    pins = {
        "PLACE": "fuse",
        "WORK": "w4x1",
        "TILE": "wgmma_m64n128k16_f16_f32/f1x16/k4",
        "REDUCE": "",
        "STAGE": "d1/smem-async",
        "FAST_MATH": fast_math,
    }
    if fast_exp is not None:
        pins["FAST_EXP"] = fast_exp
    with pinned_knobs(pins):
        ctx = Context.from_target((9, 0))
        lowered = Pipeline.build(KERNEL_PASSES).run(graph, ctx=ctx)
        (kernel,) = [node.op for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
        stores = [stmt for stmt in kernel.body.iter() if isinstance(stmt, RegStore)]
        assert stores and any(store.run for store in stores), "fixture must reach the grouped fragment stores"
        enabled = fast_math if fast_exp is None else fast_exp
        expected = "exp_fast" if enabled else "exp"
        for store in stores:
            for cell in (store, *store.run):
                assert cell.epilogue is not None
                ops = [stmt.op.name for stmt in cell.epilogue.body if isinstance(stmt, Assign)]
                assert expected in ops and ("exp" if enabled else "exp_fast") not in ops
        compiled = Pipeline.build([FINAL_LOWERING_PASS]).run(lowered, ctx=ctx)
    (source,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert ("__expf(" in source) == enabled
    assert "wgmma.mma_async" in source
    values = np.array([-12, -1, 0, 1, 12], dtype=np.float32)
    np.testing.assert_array_equal(ElementwiseImpl(expected)(values), np.exp(values))


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
