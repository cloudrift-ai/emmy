"""Register epilogues retain the Loop tail's per-Assign dtype semantics."""

import numpy as np
import pytest

from emmy.compiler.dtype import F16, F32, I32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.kernel.ir import MmaSyncPtx, RegStore, WgmmaMma
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Select, check_types, parse_body
from emmy.compiler.ir.types import epilogue_conversions
from emmy.compiler.pipeline import RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue
from emmy.compiler.target import native_f16_ops
from tests.compiler.helpers import (
    case_kernels,
    compile_with_kernels,
    operations,
    requires_cuda,
    requires_sm,
    requires_sm90,
    statements,
)


def _is_m16n8k16(kernels) -> bool:
    return any(tuple(mma.shape) == (16, 8, 16) for mma in statements(kernels, MmaSyncPtx))


def test_fp4_encode_stays_in_the_register_epilogue() -> None:
    from tests.compiler.realization.helpers import CASES_DIR, load_case

    case = load_case(CASES_DIR / "matmul/mma-fp4-encode-epilogue.json")
    _compiled, kernels = case_kernels(case)
    assert len(kernels) == 1
    assert _is_m16n8k16(kernels)
    assert "to_f4e2m1" in operations(kernels)


@requires_cuda
@requires_sm(12, 0)
@pytest.mark.parametrize("runtime_rows", [False, True], ids=["width16", "runtime65"])
def test_packed_fp4_epilogue_reads_each_elements_block_scale(runtime_rows) -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.graph import Graph
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
        compiled, kernels = compile_with_kernels(graph)
    assert len(kernels) == 1 and _is_m16n8k16(kernels)
    assert "to_f4e2m1" in operations(kernels)
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


def _typed_store(tail, buffers) -> RegStore:
    """The register store of the projection ``tail`` into ``out[m, n]``, its epilogue typed as the type pass types it."""
    epilogue = _warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)
    assert epilogue
    store = RegStore(dst_buffer="out", dst_index=(Var("m"), Var("n")), frag="_c", shape=(16, 8, 16), ldm=16, **epilogue)
    return store.replace(epilogue=epilogue_conversions(store.epilogue, store.epilogue_params, buffers, native_f16_ops()))


def _epilogue(chain: str) -> tuple[list[str], object]:
    """The typed register epilogue of the projection ``chain`` (a body's text) feeding ``out[m, n]``: its lines in the
    text form, and its statements."""
    chain = parse_body(chain)
    store = _typed_store([*chain, *parse_body(f"out[m, n] = {chain[-1].name}")], {"out": F16})
    return [line.strip() for stmt in store.epilogue for line in str(stmt).splitlines()], store.epilogue


def test_typed_copy_narrows_the_fragment_value_before_its_consumer() -> None:
    lines, epilogue = _epilogue(
        """
        f16 narrow = copy(acc)
        result = copy(narrow)
        """
    )

    assert epilogue[0].dtype is F16
    assert lines == ["f16 narrow = (f16)acc", "f16 result = narrow"]


def test_untyped_copy_keeps_the_existing_f32_epilogue() -> None:
    lines, epilogue = _epilogue("result = copy(acc)")

    assert epilogue[0].dtype is F32
    assert lines == ["f32 result = acc"]


def test_typed_literal_binding_survives_fragment_epilogue_typing() -> None:
    lines, epilogue = _epilogue(
        """
        let f32 scale = 0.125
        result = multiply(acc, scale)
        """
    )
    assert str(epilogue[0]) == "let f32 scale = 0.125"
    assert lines == ["let f32 scale = 0.125", "f32 result = acc * scale"]


def test_fragment_epilogue_substitutes_binding_coordinates_and_rejects_unbound_values() -> None:
    lines, _ = _epilogue(
        """
        let i32 column = n
        result = multiply(acc, column)
        """
    )
    # The coordinate is the element's own column (``__N__``) past the tile's.
    assert lines[0] == "let i32 column = __N__ + n"
    with pytest.raises(RuleSkipped, match="projection epilogue reads.*missing"):
        _epilogue(
            """
            let f32 scale = missing
            result = multiply(acc, scale)
            """
        )


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
                ops = [
                    node.operation.name
                    for stmt in cell.epilogue
                    if isinstance(stmt, Assign)
                    for node in stmt.value.subterms()
                    if node.operation
                ]
                assert expected in ops and ("exp" if enabled else "exp_fast") not in ops
        assert any(isinstance(stmt, WgmmaMma) for stmt in kernel.body.iter())
        compiled = Pipeline.build([FINAL_LOWERING_PASS]).run(lowered, ctx=ctx)
    assert [node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    values = np.array([-12, -1, 0, 1, 12], dtype=np.float32)
    np.testing.assert_array_equal(ElementwiseImpl(expected)(values), np.exp(values))


def test_a_mask_over_an_op_result_stands_after_that_op() -> None:
    """A causal mask may select a value the chain computes (a scaled score), not only a loaded one:
    the select stands once that op has run."""
    lines, _ = _epilogue(
        """
        ninf = load neg_inf[0]
        scaled = multiply(acc, acc)
        masked = select:
            scaled when n <= m
            ninf when 1
        result = copy(masked)
        """
    )

    assert lines.index("f32 scaled = acc * acc") < lines.index("masked = select:") < lines.index("f32 result = masked")


def test_a_mask_over_a_narrowed_value_widens_both_branches() -> None:
    """A chain op keeps the tail's dtype, so a mask over a narrowed value converts back: an epilogue's select is f32,
    every branch converted to it ahead of the select."""
    lines, epilogue = _epilogue(
        """
        ninf = load neg_inf[0]
        f16 narrow = copy(acc)
        masked = select:
            narrow when n <= m
            ninf when 1
        result = copy(masked)
        """
    )

    dtypes = {stmt.name: stmt.dtype for stmt in epilogue if isinstance(stmt, (Assign, Load))}
    (select,) = [stmt for stmt in epilogue if isinstance(stmt, Select)]
    assert dtypes["narrow"] is F16 and {dtypes[branch.value] for branch in select.branches} == {F32}
    assert "f32 narrow__f32 = (f32)narrow" in lines


def test_integer_load_in_fragment_epilogue_keeps_shift_operands_integer() -> None:
    """A packed output's shift amount must reach the epilogue as an integer, like a scalar Load."""
    tail = parse_body(
        """
        i32 shift = load shift[0]
        i32 code = to_f4e2m1(acc)
        i32 bits = left_shift(code, shift)
        out[m, n] = bits
        """
    )
    store = _typed_store(tail, {"out": I32, "shift": I32})
    lines = [str(stmt) for stmt in store.epilogue]

    assert "i32 shift = load shift[0]" in lines
    assert "i32 bits = code << shift" in lines


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
    compiled, kernels = compile_with_kernels(graph)
    assert len(kernels) == 1 and _is_m16n8k16(kernels) and "left_shift" in operations(kernels)
    got = backend.run(compiled, input_data={"a": a, "b": b, "shift": shift})[0].outputs["out"]
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.int32) << shift
    np.testing.assert_array_equal(got, expected)


def test_the_type_check_reads_a_stores_epilogue() -> None:
    """A store's epilogue is a body the type check holds like any: untyped as the lowering writes it, typed after the
    type pass."""
    tail = parse_body("in2 = load r[m, n]\nv1 = add(acc, in2)\nout[m, n] = v1")
    buffers = {"out": F16, "r": F16}
    store = _typed_store(tail, buffers)
    check_types(Body([store]), {name: dtype.name for name, dtype in buffers.items()})
    untyped = store.replace(epilogue=_warp_epilogue(tail, "acc", "m", "n", Sigma.IDENTITY)["epilogue"])
    with pytest.raises(TypeError, match="load r.*carries no dtype"):
        check_types(Body([untyped]), {name: dtype.name for name, dtype in buffers.items()})
