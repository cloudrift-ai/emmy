"""Generic sync compute-fill lowering invariants."""

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Literal
from emmy.compiler.ir.kernel.ir import pack_smem, swizzle_fn, swizzle_xor
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Assign, Body, Load, parse_body
from emmy.compiler.ir.tile.ops import cone_stat
from emmy.compiler.pipeline.passes.lowering.kernel._stage import (
    _SWIZZLE_SLAB_ALIGN,
    CtaTile,
    SyncOperand,
    SyncTransport,
    _fill_align,
    slab_smem,
    software_swizzle,
)
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import compile_with_kernels, statements
from tests.compiler.terms import projection, reduction, slab


def _sum_fold(axis: str, acc: str, *index: str) -> Fold:
    """``acc = Σ_axis x[index]`` — a plain sum over one slab."""
    value = f"{acc}_value"
    return reduction(axis, (slab(value, "x", *index),), parse_body(f"{acc}__v = copy({value})"), (acc,))


def test_cone_stat_follows_the_first_top_level_reduce_in_lowering_order() -> None:
    """The stat algebra must belong to the first reduce ``Loop`` that the cone prologue lowers.

    Attention's prologue passes its softmax statistic straight through, while the norm→linear form
    reaches its statistic through a chain over the operand. A reduction nested inside the first
    top-level fold — an operand of it, evaluated over its axis — must not win merely because a
    recursive tree walk encounters it.
    """
    nested = _sum_fold("nested", "nested_acc", "first", "nested")
    first = reduction("first", (nested,), parse_body("first_acc__v = copy(nested_acc)"), ("first_acc",))
    axes = (Axis("first", 4), Axis("nested", 4))
    body_prologue = projection((first,))
    body_cone = projection((body_prologue,), parse_body("cell = copy(first_acc)"))
    assert cone_stat(body_cone, "k", axes) is first

    operand_stat = _sum_fold("operand", "operand_acc", "operand")
    operand_prologue = projection((operand_stat,), parse_body("scale = copy(operand_acc)"))
    operand_cone = projection((operand_prologue,), parse_body("cell = copy(scale)"))
    assert cone_stat(operand_cone, "k", (Axis("operand", 4),)) is operand_stat


def test_cone_stat_finds_the_statistic_edge_wherever_formation_put_it() -> None:
    """Formation orders a cone's operands its own way, and the seam lowers every row-invariant edge in that
    order. A fused norm→linear cone is two gmem reads that vary with the contraction axis and THEN the norm's
    statistic, so a lookup that read only the first operand found no reduce, answered ``None``, and the
    prologue fell back to one serial row per thread: the Gemma 4 norm → gate/up → GeGLU kernel went from
    7.7 ms to 11.9 ms at 4096 tokens on an RTX 5090, same schedule. An edge that varies with the
    contraction axis is the cell's, never the prologue's, even when it reduces."""
    stat = _sum_fold("r", "acc", "m", "r")
    prologue = projection((stat,), parse_body("scale = copy(acc)"))
    per_cell = _sum_fold("s", "cell_acc", "k", "s")
    cone = projection(
        (slab("x_v", "x", "m", "k"), slab("w_v", "w", "k"), per_cell, prologue),
        parse_body("cell = multiply(x_v, scale)"),
    )
    axes = (Axis("m", 4), Axis("k", 4), Axis("r", 4), Axis("s", 4))
    assert cone_stat(cone, "k", axes) is stat


def test_compute_fill_suffixes_nested_ssa_for_every_vector_cell() -> None:
    """Replicated computed operands must rename definitions inside nested statistic folds."""

    def value(_k0, _row, col):
        stmts = parse_body(
            f"""
            for r in 0..4:
                value = load x[{col}, r]
                acc <- add(acc, value) over r
            out = add(acc, acc)
            """
        )
        return list(stmts), "out"

    operand = SyncOperand(tag="b", shape=(1, 8), value=value)
    transport = SyncTransport(operands=(operand,), slab_dtype="half", cta=CtaTile(Literal(0, "int"), 1), elem_bytes=2)
    fill = Body(tuple(transport.fill(k0=Literal(0, "int"), slot=Literal(0, "int"), k0_cur=Literal(0, "int"))))

    assert {acc.name for acc in fill.accums} == {f"acc__c{i}" for i in range(8)}
    assert {load.names[0] for load in fill.loads if load.input == "x"} == {f"value__c{i}" for i in range(8)}
    assert {assign.name for assign in fill.iter_of_type(Assign)} == {f"out__c{i}" for i in range(8)}


def test_software_swizzle_shifts_by_the_slab_row_not_the_atom() -> None:
    """A software-filled slab's XOR reads its ROW index — which needs the slab's OWN stride once a
    row is wider than one swizzle atom.

    The mode's default element shift is ``log2(atom elems)``; that IS the row only while a row is
    exactly one atom. A 128-elem fp16 row (the flash slabs' head dim) is TWO 128 B atoms, so the
    atom bit stays inside the shifted field, consecutive rows collapse onto a quarter of the chunk
    positions, and the ``ldmatrix`` drain over 16 rows goes multi-way bank-conflicted (measured
    7.8-way at ``D = 128`` on an A100, none at ``D = 64``). A one-atom row — and a non-power-of-two
    one, where no row bit is extractable — keeps the PLAIN spelling, so every slab that already
    drained conflict-free renders byte-identically.
    """
    assert software_swizzle(64, 2) == "B128"  # one 128 B atom: the default shift already is the row
    assert software_swizzle(32, 2) == "B64"  # one 64 B atom
    assert software_swizzle(128, 2) == "B128@7"  # two atoms: shift by the row (2**7 elems), not the atom
    assert software_swizzle(256, 2) == "B128@8"
    assert software_swizzle(96, 2) == "B64"  # not a power of two — no row bit to shift by
    assert software_swizzle(4, 2) == "NONE"  # no atom fits

    assert swizzle_xor("B128") == (6, 0x7)
    assert swizzle_xor("B128@7") == (7, 0x7)  # the override moves the shift, never the row mask
    assert swizzle_xor("NONE") is None
    # One emitted helper per distinct (mask, shift): the plain mode keeps its name, so an
    # unchanged slab's kernel source is unchanged.
    assert swizzle_fn("B128") == "emmy_swizzle_b128"
    assert swizzle_fn("B128@7") == "emmy_swizzle_b128_s7"


def test_a_staged_slab_aligns_to_its_fill_chunk_not_its_element() -> None:
    """A vector-filled slab's base must clear its FILL CHUNK, whatever its element width.

    Both fill kinds vectorize: the peer copies through ``cp.async``, and the COMPUTE fill through
    the ``v``-element runs :meth:`SyncTransport.fill` emits off the same width. So the rule covers
    every staged slab, not only the copied ones.

    :func:`_cp_async_width` picks the widest legal chunk that divides the inner span — 16 B for any
    16 B-divisible row, an f16 operand as much as an fp8 one — and both ``cp.async`` and the
    blocking vector store fault on a shared address that is not chunk-aligned. Aligning to the
    ELEMENT would leave an f16 slab 2 B-aligned, free to land at 8 mod 16 behind an odd-sized
    neighbour. That is a real fault, observed on an 8B serving compile, not a theoretical one.
    """
    assert _fill_align(64, 2, "NONE") == 16  # f16, 128 B rows — the case that used to align to 2
    assert _fill_align(32, 1, "NONE") == 16  # the packed-pair byte slab
    assert _fill_align(4, 2, "NONE") == 8  # 8 B rows admit only an 8 B chunk
    assert _fill_align(2, 2, "NONE") == 4
    # A swizzled slab still pins its atom when that is stricter.
    assert _fill_align(64, 2, "B128") == _SWIZZLE_SLAB_ALIGN["B128"]


def test_no_copied_slab_lands_misaligned_behind_an_odd_neighbour() -> None:
    """The layout property the alignment exists for, checked through the real packer.

    A packed weight stages three slabs, and the block-scale companion's size need not be a multiple
    of the next slab's chunk. Placing an f16 operand after such a neighbour is exactly what put a
    16 B ``cp.async`` write on an 8 mod 16 address; :func:`pack_smem` must now round it up.
    """
    [odd] = parse_body("Smem(name=_bs_smem, extents=(3, 4), dtype=half)")  # 24 B — leaves the cursor at 8 mod 16
    copied = slab_smem("_a_smem", 8, 64, "half", align=_fill_align(64, 2, "NONE"))
    offsets, _total = pack_smem([odd, copied])
    assert offsets["_a_smem"] % 16 == 0, f"copied slab at {offsets['_a_smem']}, not 16 B-aligned"


@pytest.mark.parametrize("width", [256, 260], ids=["dense-row", "row-off-the-16B-grid"])
def test_a_computed_operand_merges_its_reads_only_where_each_run_is_aligned(width, monkeypatch) -> None:
    """GPU-free. ``(x[:, :256] * 2.0) @ w.t()`` computes A in the fill, which merges each run's eight bf16 reads of ``x``
    into one 16 B load. A run starts on the 16 B grid along K, but its row does too only when the row stride is a
    multiple of eight elements: a 260-wide row is 520 B, so every odd row starts 8 B off the grid and a 16 B load there
    is misaligned. That row reads per element, and the 256-wide row keeps the merge. The later vectorizer still joins
    the 260-wide row's reads where it can prove them (8 B), so the plan is read off the fill itself."""
    plans: list[list[str]] = []
    run_plans = SyncTransport._run_plans
    monkeypatch.setattr(SyncTransport, "_run_plans", lambda self, op, v: plans.append(run_plans(self, op, v)) or plans[-1])
    code = (
        f"import torch\nx = torch.randn(64, {width}, dtype=torch.bfloat16)\nw = torch.randn(256, 256, dtype=torch.bfloat16)\n"
        "(x[:, :256] * 2.0) @ w.t()"
    )
    pin = {"TILE": "mma_m16n8k16_bf16_f32/f2x4/k4", "WORK": "w2x2", "STAGE": "d1/smem", "REDUCE": "", "PLACE": "fuse"}
    with pinned_knobs(pin):
        _, kernels = compile_with_kernels(graph_from_code(code)[0], ctx=Context.from_target((12, 0)))

    aligned = width % 8 == 0
    assert plans, "the pin must lower through the compute fill"
    assert any("vector" in plan for plan in plans) is aligned
    assert any(load.input == "x" and load.width == 8 for load in statements(kernels, Load)) is aligned, "one 16 B read of eight"
