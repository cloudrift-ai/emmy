"""``008_flat_indices`` gives every element access one flat index, simplified under the ranges in scope, and keeps the
dimensions of the accesses they define."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.address import flat_index
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, KernelOp, Tile
from emmy.compiler.ir.stmt import Body, Let, Load, Write, check_input, parse_body
from emmy.compiler.pipeline import RuleSkipped
from emmy.compiler.tensor import Tensor

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.cuda.008_flat_indices")


def _flat(body: str, shapes: dict[str, tuple], paged: tuple = ()) -> KernelOp:
    """``body`` under a 4 x 132 x 8 tile ``(i, j, k)``, its buffers of ``shapes``, through the pass."""
    axes = tuple(Axis(name, extent) for name, extent in (("i", 4), ("j", 132), ("k", 8)))
    kernel = KernelOp(body=Body((Tile(axes=axes, body=parse_body(body)),)))
    tensors = {name: Tensor(name, shape, F32) for name, shape in shapes.items()}
    kernel = KernelOp(
        body=kernel.body,
        inputs={n: t for n, t in tensors.items() if n in kernel.inputs},
        outputs={n: t for n, t in tensors.items() if n in kernel.outputs},
    )
    graph = Graph()
    graph.hints.set("cuda.paged_buffers", paged)
    try:
        return _RULE.rewrite(SimpleNamespace(graph=graph), SimpleNamespace(op=kernel))
    except RuleSkipped:
        return kernel


def _accesses(kernel: KernelOp, kind) -> list[str]:
    return [", ".join(str(part) for part in stmt.index) for stmt in kernel.body.iter() if isinstance(stmt, kind)]


def test_an_index_is_row_major_over_the_shape_with_its_symbolic_sizes() -> None:
    index = (Var("a"), Var("b"), Var("c"))
    assert str(flat_index(index, (Dim(4), Dim(132), Dim(8)))) == "a * 1056 + b * 8 + c"
    assert str(flat_index(index, (Dim(4), Dim("seq_len"), Dim(8)))) == "a * (seq_len * 8) + b * 8 + c"
    assert flat_index((Var("a"),), None) == Var("a") and flat_index((), None) == Literal(0, "int")
    with pytest.raises(ValueError, match="an index spells every dim"):
        flat_index((Var("a"), Var("b")), (Dim(4), Dim(132), Dim(8)))


def test_a_buffer_past_the_32_bit_range_widens_each_coordinate() -> None:
    """The cast sits on each coordinate before it meets its stride, never on the finished sum."""
    wide = flat_index(tuple(Var(f"a{n}") for n in range(4)), (Dim(4), Dim(512), Dim(2048), Dim(1024)))
    assert str(wide) == "(long long)a0 * 1073741824 + (long long)a1 * 2097152 + (long long)a2 * 1024 + (long long)a3"


def test_a_load_and_a_write_take_one_index_spelled_affine() -> None:
    """``(2 * i + 1, j, k)`` into a 16 x 132 x 8 buffer is one sum of a term per name and a constant."""
    kernel = _flat("f32 x = load A[2 * i + 1, j, k]\nf32 O[i, j, k] = x", {"A": (16, 132, 8), "O": (4, 132, 8)})
    assert _accesses(kernel, Load) == ["2112 * i + 8 * j + k + 1056"]
    assert _accesses(kernel, Write) == ["1056 * i + 8 * j + k"]
    check_input(kernel.body, {"A": "f32", "O": "f32"})


def test_an_access_of_a_paged_buffer_keeps_its_dimensions() -> None:
    """Its index along the paged axis picks the page; a buffer one page spans is one allocation and goes flat."""
    body = "f32 x = load K[i, j, k]\nf32 K[i, j, k] = x"
    paged = _flat(body, {"K": (4, 132, 8)}, paged=(("K", 1, 16),))
    assert _accesses(paged, Load) == _accesses(paged, Write) == ["i, j, k"]
    check_input(paged.body, {"K": "f32"}, ["K"])
    with pytest.raises(ValueError, match="holds one index per dimension"):
        check_input(paged.body, {"K": "f32"})
    whole = _flat(body, {"K": (4, 132, 8)}, paged=(("K", 1, 256),))
    assert _accesses(whole, Load) == ["1056 * i + 8 * j + k"]


def test_what_its_dimensions_define_keeps_them() -> None:
    """A Volta layout is a function of row and column, and a copy's lane offset is split over the tile's row and
    column: both keep the shared slab's coordinates. The copy's source is one allocation and goes flat."""
    body = (
        "Smem(name=s, extents=(16, 64))\n"
        "f32 x = load A[i, j, k]\n"
        "f32 s[i, k] = x swizzle=V70A\n"
        "f32 s[i, k] = x swizzle=B128\n"
        "CpAsyncCopy(smem=s, smem_index=(i, 8), src=A, src_index=(i, j, 0), nbytes=16)\n"
        "CpAsyncCopy(smem=s, smem_index=(0, 0), src=A, src_index=(i, j, 0), nbytes=16, lane_index=(i, 8), lane_rows=8)"
    )
    kernel = _flat(body, {"A": (4, 132, 8)})
    assert _accesses(kernel, Write) == ["i, k", "64 * i + k"]
    plain, lane = (stmt for stmt in kernel.body.iter() if isinstance(stmt, CpAsyncCopy))
    assert [str(e) for e in plain.smem_index] == ["64 * i + 8"] and [str(e) for e in plain.src_index] == ["1056 * i + 8 * j"]
    assert [str(e) for e in lane.smem_index] == ["0", "0"] and [str(e) for e in lane.src_index] == ["1056 * i + 8 * j"]
    check_input(kernel.body, {"A": "f32"})


def test_a_flat_index_in_a_value_becomes_its_arithmetic() -> None:
    """A staging pass binds a copy's base and its stride along K as ``FlatIndex`` values: the stride is a constant."""
    body = "let base = A[i, j, 0]\nlet stride = A[i, j + 1, 0] - A[i, j, 0]\nf32 x = load A[base + stride]\nf32 O[i, j, k] = x"
    kernel = _flat(body, {"A": (4, 132, 8), "O": (4, 132, 8)})
    assert [(stmt.name, str(stmt.value)) for stmt in kernel.body.iter() if isinstance(stmt, Let)] == [("base", "1056 * i + 8 * j")]
    assert _accesses(kernel, Load) == ["base + 8"]


def test_a_lowered_matmul_reaches_code_generation_with_affine_indices() -> None:
    """The scalar tier reads ``a[2 * (16 * a0_b + a0_u) + r, k]`` of a 64-wide row: block and unit axes each take
    their own stride, and the row and K offsets fold into one constant."""
    from emmy.compiler.ir.cuda.ir import CudaOp
    from tests.compiler.realization import helpers

    case = helpers.load_case(helpers.CASES_DIR / "matmul" / "f32-epilogue-bias.json")
    graph, _ = helpers.lowered(case, case.context())
    (cuda,) = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    kernel = next(op for op in cuda.source_chain() if isinstance(op, KernelOp))
    loads = [stmt for stmt in kernel.body.iter() if isinstance(stmt, Load)]
    assert loads and all(len(stmt.index) == 1 for stmt in loads)
    assert "2048 * a0_b + 128 * a0_u + 64" in {str(stmt.index[0]) for stmt in loads if stmt.input == "a"}
