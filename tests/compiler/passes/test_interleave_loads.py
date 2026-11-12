"""The Kernel IR memory peepholes: vector loads and stores, fragment load pairs, and loads moved into their guard."""

from importlib import import_module

import pytest

from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.ir.expression import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Load, parse_body

_vectorize_loads = import_module("emmy.compiler.pipeline.passes.lowering.kernel.050_vectorize_loads")._vectorize_body
_vectorize_stores = import_module("emmy.compiler.pipeline.passes.lowering.kernel.080_vectorize_stores")._vectorize_body
_sink_guarded = import_module("emmy.compiler.pipeline.passes.lowering.kernel.095_interleave_loads")._walk
_pair_ldmatrix = import_module("emmy.compiler.pipeline.passes.lowering.kernel.096_pair_ldmatrix_loads")._walk


def test_noop_kernel_peepholes_reuse_the_body() -> None:
    body = parse_body(
        """
        for k in 0..4:
            f32 x = load input[0]
            f32 y = abs(x)
        """
    )
    op = KernelOp(body=body)
    interleaved, interleave_changed = _sink_guarded(body)
    paired, pair_changed = _pair_ldmatrix(body)

    assert _vectorize_loads(op, body) is body
    assert _vectorize_stores(op, body) is body
    assert interleaved is body and not interleave_changed
    assert paired is body and not pair_changed


def test_a_load_only_a_guard_reads_moves_into_it() -> None:
    """A cell the guard skips then never issues the load. A load the guard's own condition reads, or one another
    statement reads too, stays where every reader finds it."""
    body = parse_body(
        """
        f32 flag = load mask[0]
        f32 x = load input[0]
        f32 y = load input[1]
        if flag > 0.0:
            f32 z = add(x, y)
            out[0] = z
        out[1] = y
        """
    )

    moved, changed = _sink_guarded(body)

    assert changed
    assert [str(stmt) for stmt in moved if isinstance(stmt, Load)] == ["f32 flag = load mask[0]", "f32 y = load input[1]"]
    [guard] = [stmt for stmt in moved if stmt.nested()]
    assert str(guard.body[0]) == "f32 x = load input[0]"
    assert _sink_guarded(moved) == (moved, False)


def _unrolled_pointwise(*, between: str = "", dtype=F16) -> Body:
    """Four unrolled cells of ``out = x * x``: each cell loads its 16-bit element and squares it
    before the next cell's load, as the unroller emits a pointwise body; ``between`` is the text of
    the statements that sit between the third cell's load and its square."""
    lines = []
    for u in range(4):
        lines.append(f"{dtype.name} x{u} = load x[a0 * 4 + {u}]")
        if u == 2:
            lines.append(between)
        lines.append(f"{dtype.name} y{u} = multiply(x{u}, x{u})")
    return parse_body("\n".join(lines))


@pytest.mark.parametrize("dtype", [F16, BF16], ids=lambda dtype: dtype.name)
def test_interleaved_loads_of_one_buffer_widen(dtype) -> None:
    """Consecutive elements loaded between other work still form one vector load: an unrolled
    pointwise body never places them side by side, and global 16-bit loads left 2 bytes wide cost
    the kernel most of its bandwidth."""
    out = _vectorize_loads(KernelOp(body=Body(())), _unrolled_pointwise(dtype=dtype))

    loads = [stmt for stmt in out if isinstance(stmt, Load)]
    assert len(loads) == 1 and loads[0].names == ("x0", "x1", "x2", "x3")
    assert out[0] is loads[0], "the widened load sits where the first element was loaded"


def test_vector_loads_and_stores_require_aligned_row_offsets() -> None:
    from emmy.compiler.graph import Tensor

    for columns, row, aligned in (
        (3, "row", False),
        (4, "row", True),
        (3, "1", False),
        (3, "2", True),
    ):
        tensor = Tensor("x", (4, columns), F32)
        loads = parse_body("\n".join(f"f32 v{i} = load x[{row}, {i}]" for i in range(2)))
        stores = parse_body("\n".join(f"f32 x[{row}, {i}] = v{i}" for i in range(2)))
        op = KernelOp(body=loads, inputs={"x": tensor}, outputs={"x": tensor})
        assert len(_vectorize_loads(op, loads)) == (1 if aligned else 2)
        assert len(_vectorize_stores(op, stores)) == (1 if aligned else 2)


def _loads_widen(indices, tensor=None) -> bool:
    """Whether scalar loads of ``x`` at ``indices`` become one vector load."""
    loads = Body(tuple(Load(names=(f"v{i}",), input="x", index=index, dtype=F32) for i, index in enumerate(indices)))
    op = KernelOp(body=Body(()), inputs={"x": tensor} if tensor is not None else {})
    return len(_vectorize_loads(op, loads)) == 1


def test_vector_alignment_recomposes_split_coordinates() -> None:
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.expression import BinaryExpression, Var

    tensor = Tensor("x", (4, 8), F32)
    cells = [Var("i") * Literal(4, "int") + Literal(offset, "int") for offset in range(4)]
    split = [(BinaryExpression("//", cell, Literal(8, "int")), BinaryExpression("%", cell, Literal(8, "int"))) for cell in cells]
    assert _loads_widen(split, tensor)
    assert not _loads_widen([(Var("row"), Literal(i, "int")) for i in range(2)])


def test_a_gathered_selector_term_is_an_unknown_integer() -> None:
    """A routed expert's weight run starts at a gathered selector times the expert's stride, behind the gather's sign
    wrap. That term has no affine form, but it is an integer, so a stride that is a multiple of the width keeps the
    run aligned, and an odd one does not."""
    from emmy.compiler.ir.expression import BinaryExpression, CastExpression, TernaryExpression, Var

    sel = CastExpression(dtype="int", expression=Var("in0"))
    wrap = TernaryExpression(BinaryExpression("<", sel, Literal(0, "int")), BinaryExpression("+", sel, Literal(512, "int")), sel)

    def run(stride: int) -> list:
        base = BinaryExpression(
            "+", BinaryExpression("*", wrap, Literal(stride, "int")), BinaryExpression("*", Var("lane"), Literal(4, "int"))
        )
        return [(BinaryExpression("+", base, Literal(i, "int")) if i else base,) for i in range(4)]

    assert _loads_widen(run(409600))
    assert not _loads_widen(run(409601))


def test_a_run_widens_whatever_order_its_loads_stand_in() -> None:
    """The normal form orders sibling statements its own way, so a run is read by address, not by position: the
    widened load names its elements in address order."""
    from emmy.compiler.ir.expression import Var

    loads = Body(
        tuple(Load(names=(f"v{i}",), input="x", index=(Var("i") * Literal(4, "int") + Literal(i, "int"),), dtype=F32) for i in (2, 0, 3, 1))
    )
    [wide] = _vectorize_loads(KernelOp(body=Body(())), loads)
    assert wide.names == ("v0", "v1", "v2", "v3")


def test_a_write_between_two_loads_keeps_them_apart() -> None:
    """A load behind a write of its buffer reads the written value; one ahead of it reads the old one."""
    body = parse_body(
        """
        f32 v0 = load x[0]
        f32 x[1] = v0
        f32 v1 = load x[1]
        f32 y = add(v0, v1)
        """
    )
    assert _vectorize_loads(KernelOp(body=Body(())), body) is body


def test_a_strided_loop_variable_carries_its_alignment() -> None:
    """``coop/v<n>``: a lane starting at ``lane * 8`` and striding 1024 reads aligned runs of eight, so
    the loop variable's unit coefficient still proves an aligned vector. A lane starting at ``lane``
    proves nothing."""
    from emmy.compiler.graph import Tensor

    tensor = Tensor("w", (4, 1024), F16)
    for start, widened in (("lane * 8", True), ("lane", False)):
        loads = "\n".join(f"    f16 w{i} = load w[row, k + {i}]" if i else "    f16 w0 = load w[row, k]" for i in range(8))
        body = parse_body(f"for k in {start}..1024 step 1024:\n{loads}")
        op = KernelOp(body=body, inputs={"w": tensor}, outputs={})
        (out,) = _vectorize_loads(op, op.body)
        assert len(out.body) == (1 if widened else 8)


def test_a_load_whose_index_is_computed_in_between_stays_put() -> None:
    """A later load moves up only when its index needs nothing defined in between."""
    body = parse_body(
        """
        f16 x0 = load x[a0 * 2]
        f16 j = abs(x0)
        f16 x1 = load x[a0 * 2 + j]
        """
    )
    assert _vectorize_loads(KernelOp(body=body), body) is body


def test_a_write_to_the_buffer_in_between_stops_the_run() -> None:
    """A load never moves above a store into its own buffer."""
    out = _vectorize_loads(KernelOp(body=Body(())), _unrolled_pointwise(between="x[a0] = y0"))

    widths = sorted(len(stmt.names) for stmt in out if isinstance(stmt, Load))
    assert widths == [1, 1, 2], "x0 and x1 pair up; x3 cannot move above the store to meet x2"


def test_ring_slot_transposed_b_fragments_pair() -> None:
    """Two N-adjacent transposed-B fragments of a staged ring drain become one ldmatrix.x4.

    The row is ``slot * rows + (warp_row + 8)``: the +8 sits inside the chain, and the slot term is
    not affine. Left unpaired, every F.linear GEMM drained B with twice the ldmatrix count in its
    main loop."""

    def drain(frag: str, n: int) -> str:
        row = f"_ks / 32 % 4 * 64 + (a1_u * 32 + {n})"
        return f"LdmatrixLoad(frag={frag}, src_buffer=_b_smem, src_index=({row}, 0), role=b, ldm=32, b_trans=true)"

    paired, changed = _pair_ldmatrix(parse_body("\n".join((drain("b0", 0), drain("b1", 8), drain("b2", 16), drain("b3", 24)))))

    assert changed
    assert [(s.frag, s.pair_frag) for s in paired] == [("b0", "b1"), ("b2", "b3")]


def _drain(order: str) -> Body:
    """Two B fragments, eight slab rows apart, each drained over two K steps into its own accumulator. ``order`` is
    ``"steps"`` (both fragments of a step, then the next step: the order the drain is emitted in) or ``"chains"``
    (one fragment's whole chain, then the other's: an order the normal form can put them in)."""

    def step(frag: str, acc: str, row: int, k: int) -> list[str]:
        return [
            f"LdmatrixLoad(frag={frag}, src_buffer=_b_smem, src_index=({row}, {k}), role=b, ldm=128, b_trans=true)",
            f"MmaSyncPtx(c_frag={acc}, a_frag=_a, b_frag={frag}, shape=(16, 8, 16))",
        ]

    chains = [("_b0", "_c0", 0), ("_b1", "_c1", 8)]
    if order == "steps":
        lines = [line for k in (0, 16) for frag, acc, row in chains for line in step(frag, acc, row, k)]
    else:
        lines = [line for frag, acc, row in reversed(chains) for k in (0, 16) for line in step(frag, acc, row, k)]
    return parse_body("\n".join(lines))


@pytest.mark.parametrize("order", ["steps", "chains"])
def test_fragment_loads_pair_whatever_order_their_chains_stand_in(order) -> None:
    """The two halves of a pair are found by the scope's ordering constraints: each step's loads fuse, low half
    first, and the pair stands ahead of both of its consumers and behind the consumers of the step before."""
    from emmy.compiler.ir.kernel.ir import LdmatrixLoad

    paired, changed = _pair_ldmatrix(_drain(order))

    assert changed
    loads = [stmt for stmt in paired if isinstance(stmt, LdmatrixLoad)]
    assert [(load.frag, load.pair_frag) for load in loads] == [("_b0", "_b1")] * 2
    assert [type(stmt).__name__ for stmt in paired] == ["LdmatrixLoad", "MmaSyncPtx", "MmaSyncPtx"] * 2


def test_a_fragment_read_between_its_two_loads_keeps_a_pair_apart() -> None:
    """A consumer of the high fragment's earlier value that must run behind the low half's consumer leaves no place
    for the pair: it would fill the high fragment before that read."""
    body = parse_body(
        """
        LdmatrixLoad(frag=_b1, src_buffer=_b_smem, src_index=(8, 0), role=b, ldm=128, b_trans=true)
        LdmatrixLoad(frag=_b0, src_buffer=_b_smem, src_index=(0, 16), role=b, ldm=128, b_trans=true)
        MmaSyncPtx(c_frag=_c0, a_frag=_a, b_frag=_b0, shape=(16, 8, 16))
        MmaSyncPtx(c_frag=_c0, a_frag=_a, b_frag=_b1, shape=(16, 8, 16))
        LdmatrixLoad(frag=_b1, src_buffer=_b_smem, src_index=(8, 16), role=b, ldm=128, b_trans=true)
        MmaSyncPtx(c_frag=_c0, a_frag=_a, b_frag=_b1, shape=(16, 8, 16))
        """
    )
    paired, changed = _pair_ldmatrix(body)
    assert paired is body and not changed
