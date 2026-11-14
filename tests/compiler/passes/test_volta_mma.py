"""Target selection and source lowering for the Volta m8n8k4 MMA family."""

from __future__ import annotations

import math

import pytest

from emmy.compiler.backend.cuda import ptxas
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SdpaOp
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCopy,
    FragmentApply,
    FragmentLoad,
    FragmentRepack,
    FragmentRowReduce,
    LdmatrixLoad,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    Smem,
    Sync,
    TmaLoad,
)
from emmy.compiler.ir.schedule.catalog import MAX_FRAGMENT_REGISTERS, warp_tile_moves
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, StridedLoop, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_value
from emmy.compiler.target import set_target
from tests.compiler.helpers import compile_with_kernels, operations, statements, text

VOLTA = "mma_m8n8k4_f16_f32"
AMPERE = "mma_m16n8k16_f16_f32"


def _graph(*, m: int = 16, n: int = 16, k: int = 4, trans: bool = False) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("a", (m, k), dtype=F16), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (n, k) if trans else (k, n), dtype=F16), node_id="b")
    graph.add_node(LinearOp() if trans else MatmulOp(), ["a", "b"], Tensor("c", (m, n), dtype=F16), node_id="c")
    graph.inputs, graph.outputs = ["a", "b"], ["c"]
    return graph


def _norm_linear_graph(*, m: int = 16, n: int = 16, k: int = 16) -> Graph:
    """A fused norm→linear: the projection's ``a`` edge is the norm's producer CONE, not a load."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(m), Dim(k)), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("wn", (Dim(k),), dtype=F16), node_id="wn")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(n), Dim(k)), dtype=F16), node_id="w")
    graph.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(m), Dim(k)), dtype=F16), node_id="xn")
    graph.add_node(LinearOp(), ["xn", "w"], Tensor("y", (Dim(m), Dim(n)), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "wn", "w"], ["y"]
    return graph


def _sdpa_graph(*, d: int = 32, causal: bool = False, dynamic: bool = False) -> Graph:
    graph = Graph()
    seq = Dim("seq_len") if dynamic else Dim(64)
    for name in ("q", "k", "v"):
        graph.add_node(InputOp(), [], Tensor(name, (Dim(1), Dim(1), seq, Dim(d)), dtype=F16), node_id=name)
    graph.add_node(
        SdpaOp(is_causal=causal),
        ["q", "k", "v"],
        Tensor("o", (Dim(1), Dim(1), seq, Dim(d)), dtype=F16),
        node_id="o",
    )
    graph.inputs, graph.outputs = ["q", "k", "v"], ["o"]
    return graph


def _pin(monkeypatch, atom: str, *, tile: str = "f1x1", stage: str = "") -> None:
    monkeypatch.setenv("EMMY_TILE", f"{atom}/{tile}")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_REDUCE", "")


def _pin_sdpa(monkeypatch) -> None:
    # One fused kernel is what these tests read the source of; the placement prior would otherwise decide the cut.
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_TILE@map.1/twist.1/inner", f"{VOLTA}/f1x1/k4")
    monkeypatch.setenv("EMMY_TILE@map.1/twist", f"{VOLTA}/f1x1/k4")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist.1/inner", "")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist", "")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_RASTER", "")


def _instructions(ptx: str) -> list[str]:
    """The instructions of a kernel's PTX, one a line, each without its guard and past the statement comments."""
    lines = (line.strip() for line in ptx.splitlines())
    code = (line.split(" ", 1)[1] if line.startswith("@") else line for line in lines if line and not line.startswith(("//", ".")))
    return [line for line in code if not line.endswith((":", "{", "}"))]


def _count(ptx: str, opcode: str) -> int:
    """How many instructions of ``ptx`` are ``opcode`` with any operands."""
    return sum(line.split()[0] == opcode for line in _instructions(ptx))


def _source(graph: Graph, ctx: Context) -> tuple:
    """The one kernel ``graph`` compiles to: its Kernel IR, its knobs and its PTX, which only an instruction check reads."""
    compiled, kernels = compile_with_kernels(graph, ctx=ctx)
    ops = [node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(ops) == 1
    (kernel,) = kernels.values()
    return kernel, ops[0].knobs, ops[0].kernel_source


def _slab_writes(kernel, slab: str, layout: str | None = None) -> list:
    """The writes that fill ``slab``; with ``layout``, those that place their element through it (``V70A`` is the
    crosswise layout, ``V70B`` the congruous one)."""
    return [w for w in statements(kernel, Write) if w.output == slab and (layout is None or w.swizzle == layout)]


def _drains(kernel, role: str) -> list:
    """The staged fragment loads of ``role``."""
    return [load for load in statements(kernel, FragmentLoad) if load.role == role and load.staged]


def _assert_volta_only(kernel) -> None:
    """Nothing in ``kernel`` asks for an instruction a newer target has and sm_70 does not: no matrix load, no
    asynchronous copy, and every product the Volta ``m8n8k4`` of f16 into f32."""
    assert not statements(kernel, LdmatrixLoad, CpAsyncCopy, TmaLoad)
    assert {(tuple(mma.shape), mma.ab_dtype, mma.c_dtype) for mma in statements(kernel, MmaSyncPtx)} == {((8, 8, 4), "f16", "f32")}


def _free_vars(index) -> set[str]:
    return {node.name for expression in index for node in expression.subterms() if isinstance(node, Var)}


def _register_ring_order(body) -> tuple[int, int, int, int]:
    """Where a register-staged ring's chunk body drains, deposits the carried vectors in the free slot, publishes them
    and issues the loads of the chunk after next: the first of each, by position."""
    drain = next(i for i, stmt in enumerate(body) if isinstance(stmt, MmaSyncPtx))
    deposit = next(i for i, stmt in enumerate(body) if isinstance(stmt, Write) and stmt.output in ("_a_smem", "_b_smem"))
    barrier = next(i for i in range(deposit, len(body)) if isinstance(body[i], Sync))
    issue = next(i for i in range(barrier, len(body)) if isinstance(body[i], Load) and body[i].carried)
    return drain, deposit, barrier, issue


@pytest.mark.parametrize(
    ("cc", "caps"),
    [
        ((7, 0), (True, False, False, False, False, False)),
        ((7, 5), (True, True, False, False, False, False)),
        ((8, 0), (False, True, True, False, True, False)),
        ((8, 9), (False, True, True, False, True, True)),
        ((9, 0), (False, True, True, True, True, True)),
    ],
)
def test_context_instruction_capabilities(cc, caps) -> None:
    ctx = Context(compute_capability=cc)
    assert (
        ctx.has_volta_mma,
        ctx.has_ldmatrix,
        ctx.has_cp_async,
        ctx.has_tma,
        ctx.has_bf16_mma,
        ctx.has_fp8_mma,
    ) == caps


def test_volta_atom_separates_logical_and_instruction_shapes() -> None:
    atom = ATOM_REGISTRY[VOLTA]
    assert atom.shape == (16, 16, 4)
    assert atom.ptx_shape == (8, 8, 4)
    assert tuple(atom.fragment_registers(role) for role in ("a", "b", "c")) == (2, 2, 8)
    assert atom.fragment_layout == "m8n8k4"
    assert atom.sync_copy_staging and atom.c_to_a_repack and atom.c_to_b_repack

    # The PTX C-fragment map covers the logical 16x16 output exactly once.
    coords = []
    for lane in range(32):
        comp = (lane & 15) >> 2
        row0 = (comp >> 1) * 8 + (lane >> 4) * 4 + (lane & 1)
        col0 = (comp & 1) * 8 + (lane & 2)
        coords.extend((row0 + (elem & 2), col0 + (elem & 4) + (elem & 1)) for elem in range(8))
    assert len(set(coords)) == 16 * 16
    assert set(coords) == {(row, col) for row in range(16) for col in range(16)}


def test_atom_selection_is_target_specific() -> None:
    assert atoms_for(F16, ctx=Context(compute_capability=(7, 0))) == (VOLTA,)
    assert atoms_for(F16, acc=F16, ctx=Context(compute_capability=(7, 0))) == ("mma_m8n8k4_f16_f16",)
    assert atoms_for(F16, ctx=Context(compute_capability=(7, 5))) == (VOLTA,)
    assert atoms_for(F16, ctx=Context(compute_capability=(8, 0))) == (AMPERE,)
    assert atoms_for(F16, ctx=Context(compute_capability=(12, 0))) == (AMPERE,)
    assert atoms_for(F16) == (AMPERE,)  # registry inspection retains the established modern view


def test_volta_warp_tiles_respect_accumulator_register_budget() -> None:
    volta = warp_tile_moves((VOLTA,))
    ampere = warp_tile_moves((AMPERE,))
    assert volta and ampere
    assert max(p.reg_m * p.reg_n for p in volta) == 16
    assert max(p.reg_m * p.reg_n for p in ampere) == 32
    assert all(p.reg_m * p.reg_n * p.atom.accumulator_registers_per_lane <= MAX_FRAGMENT_REGISTERS for p in volta + ampere)


def test_volta_chunks_reach_the_width_the_16_deep_atoms_do() -> None:
    """Volta's k4 atom reaches a 128-element K chunk at bk 32, the chunk a k16 atom reaches at bk 8: a skinny GEMM's
    K-contiguous rows stream in 256 B runs rather than 64 B. No other atom's chunk domain moves."""
    volta = {p.bk for p in warp_tile_moves((VOLTA,))}
    assert volta == {1, 2, 4, 8, 16, 32}
    for name in ATOM_REGISTRY:
        if ATOM_REGISTRY[name].atom_k > 4:
            assert {p.bk for p in warp_tile_moves((name,))} == {1, 2, 4, 8}, name


def test_sm70_deep_chunk_stages_k_contiguous_b_through_the_crosswise_layout(monkeypatch) -> None:
    """A bk 16 chunk is 64 K-contiguous elements per staged row: the copy and the crosswise drain cover both 32-column
    halves of the slab."""
    _pin(monkeypatch, VOLTA, tile="f2x4/k16", stage="d2/smem")
    monkeypatch.setenv("EMMY_WORK", "w1x2")
    kernel, knobs, _ = _source(_graph(m=32, n=128, k=256, trans=True), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f2x4/k16"
    assert _slab_writes(kernel, "_b_smem", "V70A")
    chunks = [loop for loop in statements(kernel, StridedLoop) if loop.axis.extent == Dim(256) and loop.step == Literal(64, "int")]
    assert chunks, "one chunk carries 64 K elements"


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_source_uses_only_the_volta_mma_family(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA)
    kernel, knobs, src = _source(_graph(trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == ""
    _assert_volta_only(kernel)
    fragments = statements(kernel, RegFragment)
    assert sorted(fragment.role for fragment in fragments if fragment.role in ("a", "b")) == ["a", "b"], "one A and one B fragment"
    assert [fragment.dtype.name for fragment in fragments if fragment.role == "c"] == ["f32"], "one accumulator"
    # A transposed B is K-contiguous in memory: its one unrolled K step starts at a literal offset, so it loads its
    # halves two at a time, as A does. A plain B strides K by its row and loads element by element.
    assert _count(src, "ld.global.nc.b16") == (0 if trans else 4)
    assert _count(src, "ld.global.nc.b32") == (4 if trans else 2)


@pytest.mark.parametrize("role,trans", [("a", False), ("b", True), ("b", False)])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize(
    "stride,offset,masked_k,aligned",
    [(32, 4, False, True), (32, 5, False, False), (31, 4, False, False), (32, 4, True, False)],
    ids=["aligned", "offset", "stride", "k-tail"],
)
def test_volta_direct_vectors_require_aligned_complete_runs(role, trans, dtype, stride, offset, masked_k, aligned):
    """A Volta fragment read from global memory takes its halves two at a time only where the run is complete and its
    address provably aligned; anything else is read an element at a time."""
    from emmy.emmy_ir import compile_ptx, parse_body

    from emmy.compiler.dtype import F16
    from emmy.compiler.dtype import get as dtype_get
    from emmy.compiler.ir.expression import Literal, Var
    from emmy.compiler.ir.kernel.interface import Interface, Param
    from emmy.compiler.ir.kernel.ir import FragmentLoad

    load = FragmentLoad(
        frag="r",
        src_buffer="x",
        src_index=(Var("t"), Literal(offset, "int")),
        role=role,
        b_trans=trans,
        ldm=stride,
        fragment_layout="m8n8k4",
        clamp=(Literal(0, "int"), Literal(17, "int")),
        k_zero=(Literal(0, "int"), Literal(3, "int")) if masked_k else None,
        src_dtype=dtype_get(dtype),
        dtype=F16,
    )
    fragment = f"RegFragment(name=r, role={role}, dtype=f16, fragment_layout=m8n8k4)"
    body = f"Grid(axes=(t in 0..32), decode=(thread_idx.x), block_threads=32):\n    {fragment}\n    {load}"
    interface = Interface((Param("x", "input", dtype_get(dtype), "x", shape=(17, stride)),), 32, {}, 0, False)
    ptx = compile_ptx("k", parse_body(body), interface, sm=70)
    wide = aligned and (role == "a" or trans) and dtype == "f16"
    assert _count(ptx, "ld.global.nc.b32") == (2 if wide else 0)
    assert _count(ptx, f"ld.global.nc.{'b16' if dtype == 'f16' else 'f32'}") == (0 if wide else 4)


def test_sm70_m1_linear_synthesizes_a_masked_mma_row(monkeypatch) -> None:
    """A literal unit output row is still the M side of a plain m1xKxN contraction.

    RESTORED: the decode row is the only shape a V100 deployment runs at serving time, and without
    it the m=1 linear falls to the scalar path with no assertion anywhere noticing.
    """
    _pin(monkeypatch, VOLTA, tile="f1x1", stage="")
    kernel, knobs, _ = _source(_graph(m=1, n=16, k=16, trans=True), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert knobs["WORK"] == "w1x1"
    _assert_volta_only(kernel)
    # The A load clamps to the one row (its bound is the rows there are), and the stores are guarded to it.
    a_loads = [load for load in statements(kernel, FragmentLoad) if load.role == "a"]
    assert a_loads and all(load.clamp is not None and load.clamp[1] == Literal(1, "int") for load in a_loads)
    stores = statements(kernel, RegStore)
    assert stores and all(store.m_guard is not None and store.m_guard[1] == Literal(1, "int") for store in stores)


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_sync_copy_stages_fragments_without_newer_instructions(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA, stage="d1/smem")
    kernel, knobs, _ = _source(_graph(k=16, trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "STAGE") == "d1/smem"
    slabs = {smem.name: (smem.dtype.name, math.prod(smem.extents)) for smem in statements(kernel, Smem)}
    assert slabs == {"_a_smem": ("f16", 64), "_b_smem": ("f16", 64)}
    assert [(load.src_buffer, load.b_trans) for load in _drains(kernel, "a")] == [("_a_smem", False)]
    assert [(load.src_buffer, load.b_trans) for load in _drains(kernel, "b")] == [("_b_smem", trans)]
    assert len(statements(kernel, Sync)) == 2
    _assert_volta_only(kernel)


def test_sm70_contiguous_staged_fragments_use_one_wide_load(monkeypatch) -> None:
    """A and N-major B fragments are contiguous in their staged slabs, so each drains as one uint2."""
    _pin(monkeypatch, VOLTA, stage="d1/smem")
    kernel, _, src = _source(_graph(k=16, trans=True), Context(compute_capability=(7, 0)))
    # The plain gather of each staged operand, the B one transposed: A and an N-major B in their slabs, unpaired.
    assert [(load.b_trans, load.slab_layout) for load in _drains(kernel, "a")] == [(False, "NONE")]
    assert [(load.b_trans, load.slab_layout) for load in _drains(kernel, "b")] == [(True, "NONE")]
    # Each drain reads its slab a word of two halves at a time.
    assert _count(src, "ld.shared.b32") == 4 and _count(src, "ld.shared.b16") == 0


def test_sm70_materialized_tiles_use_paired_volta_layout_loads(monkeypatch) -> None:
    """Even Volta register tiles derive the coupled CUTLASS fill/drain layouts by default."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    kernel, _, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    assert _slab_writes(kernel, "_a_smem", "V70A")
    assert _slab_writes(kernel, "_b_smem", "V70B")
    assert [(load.slab_layout, load.pair_frag is not None) for load in _drains(kernel, "a")] == [("V70A", True)]
    assert [(load.slab_layout, load.pair_frag is not None) for load in _drains(kernel, "b")] == [("V70B", True)]
    assert sum(mma.b_row_major for mma in statements(kernel, MmaSyncPtx)) == 4
    # The interleaved accumulator map is coupled to the paired layouts.
    assert all(store.volta_interleaved for store in statements(kernel, RegStore))


def test_sm70_transposed_b_drains_the_crosswise_layout(monkeypatch) -> None:
    """A transposed B stages K-contiguous like A, so it reads A's crosswise layout back rather
    than falling to the plain row-major tile whose 64-byte row is bank-conflicted."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    kernel, _, _ = _source(_graph(m=32, n=32, k=16, trans=True), Context(compute_capability=(7, 0)))
    assert _slab_writes(kernel, "_a_smem", "V70A")
    assert _slab_writes(kernel, "_b_smem", "V70A")
    assert not _slab_writes(kernel, "_b_smem", "V70B"), "the congruous layout is for an N-contiguous B"
    assert [(load.slab_layout, load.pair_frag is not None) for load in _drains(kernel, "a")] == [("V70A", True)]
    # The B reader is A's crosswise one (with lane bits 2 and 3 swapped where it is spelled): the conflicted plain
    # gather is gone.
    assert [(load.slab_layout, load.b_trans, load.pair_frag is not None) for load in _drains(kernel, "b")] == [("V70A", True, True)]
    # The row/col mma form stays.
    assert [mma.b_row_major for mma in statements(kernel, MmaSyncPtx)] == [False] * 4
    # The interleaved accumulator map is unchanged: the column half still comes from lane bit 3.
    assert all(store.volta_interleaved for store in statements(kernel, RegStore))


def test_sm70_pair_policy_off_retains_the_unpaired_gather(monkeypatch) -> None:
    """The existing policy override disables the coupled layout as one complete choice."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    monkeypatch.setenv("EMMY_PAIR_LDMATRIX", "0")
    kernel, knobs, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    assert not _slab_writes(kernel, "_a_smem", "V70A")
    assert not _slab_writes(kernel, "_b_smem", "V70B")
    drains = _drains(kernel, "a") + _drains(kernel, "b")
    assert {load.role for load in drains} == {"a", "b"}
    assert all(load.slab_layout == "NONE" and load.pair_frag is None and not load.b_trans for load in drains)
    assert statements(kernel, MmaSyncPtx) and not any(mma.b_row_major for mma in statements(kernel, MmaSyncPtx))
    assert knobs["PAIR_LDMATRIX"] is False


def test_sm70_computed_tiles_use_the_same_paired_layouts(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f2x2/k4", stage="d1/smem")
    graph = _norm_linear_graph(m=32, n=32, k=32)
    graph.nodes["y"].op = MatmulOp()
    kernel, _, _ = _source(graph, Context(compute_capability=(7, 0)))
    assert _slab_writes(kernel, "_a_smem", "V70A")
    assert _slab_writes(kernel, "_b_smem", "V70B")
    assert {(load.slab_layout, load.pair_frag is not None) for load in _drains(kernel, "a")} == {("V70A", True)}
    assert {(load.slab_layout, load.pair_frag is not None) for load in _drains(kernel, "b")} == {("V70B", True)}


def test_sm70_gmem_direct_tile_keeps_the_ordinary_accumulator_map(monkeypatch) -> None:
    """The paired accumulator map is coupled to a staged operand layout, never used alone."""
    _pin(monkeypatch, VOLTA, tile="f2x2")
    kernel, _, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    products = statements(kernel, MmaSyncPtx)
    assert not any(mma.b_row_major for mma in products)
    assert len(products) == 4 * (16 // 4)  # the 2x2 register tile, at each K step
    assert not any(store.volta_interleaved for store in statements(kernel, RegStore))


def test_sm70_output_stores_contiguous_fragment_pairs(monkeypatch) -> None:
    """The eight Volta accumulator elements leave registers as four contiguous half2 pairs."""
    _pin(monkeypatch, VOLTA)
    _, _, src = _source(_graph(), Context(compute_capability=(7, 0)))
    assert _count(src, "st.global.v2.b16") == 4 and _count(src, "st.global.b16") == 0


def test_sm70_masked_rows_keep_contiguous_fragment_pairs(monkeypatch) -> None:
    """A row guard covers both adjacent values, so it does not split their half2 store."""
    _pin(monkeypatch, VOLTA)
    kernel, _, src = _source(_graph(m=Dim("seq_len", hint=512)), Context(compute_capability=(7, 0)))
    (store,) = statements(kernel, RegStore)
    assert store.m_guard is not None and store.m_guard[1] == Var("seq_len")
    assert _count(src, "st.global.v2.b16") == 4 and _count(src, "st.global.b16") == 0


def test_sm70_sync_copy_composes_ring_and_register_pipelines(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f1x1/k2", stage="d2/smem/p2")
    kernel, knobs, _ = _source(_graph(k=32), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1/k2"
    assert family_value(knobs, "STAGE") == "d2/smem/p2"
    slabs = {smem.name: (smem.dtype.name, math.prod(smem.extents)) for smem in statements(kernel, Smem)}
    assert slabs == {"_a_smem": ("f16", 256), "_b_smem": ("f16", 256)}
    operands = [fragment for fragment in statements(kernel, RegFragment) if fragment.role in ("a", "b")]
    assert len(operands) == 4, "two register slots of the A and the B fragment"
    calls = [(mma.c_frag, mma.a_frag, mma.b_frag) for mma in statements(kernel, MmaSyncPtx)]
    assert len(calls) == 2 and len({c for c, _, _ in calls}) == 1, "both slots accumulate into the one C fragment"
    assert len({(a, b) for _, a, b in calls}) == 2, "each call reads its own slot pair"
    _assert_volta_only(kernel)


def test_sm70_ring_splits_the_blocking_copy_across_the_drain(monkeypatch) -> None:
    """A ring on a target without ``cp.async`` puts its in-flight chunk in REGISTERS.

    The resident chunk's mma drain runs, the registers land in the free slot, ONE barrier per chunk
    publishes them, and only past it do the next chunk's global loads issue — so the loads fly
    under the whole next drain rather than the part of it ptxas leaves after sinking them. Back-to-
    back load/store fills (what a ring emitted first) leave the latency fully exposed and need two
    barriers, which measured slower than no ring at all on every V100 shape tried."""
    monkeypatch.setenv("EMMY_TILE", f"{VOLTA}/f2x2/k8")
    monkeypatch.setenv("EMMY_WORK", "w2x2")  # 128 threads: the slabs stripe evenly, so the split engages
    monkeypatch.setenv("EMMY_STAGE", "d2/smem")
    monkeypatch.setenv("EMMY_REDUCE", "")
    kernel, knobs, _ = _source(_graph(m=64, n=64, k=64), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "STAGE") == "d2/smem"
    ((grid,),) = [kernel.body]
    (at,) = [i for i, stmt in enumerate(grid.body) if isinstance(stmt, StridedLoop) and stmt.axis.extent == Dim(64)]
    (chunk,) = [grid.body[at]]
    assert chunk.step == Literal(32, "int")
    prologue, body = grid.body[:at], list(chunk.body)
    drain, deposit, barrier, issue = _register_ring_order(body)
    assert drain < deposit < barrier < issue, "the loads issue past the barrier that publishes the deposit"
    # The deposit reads the carried vectors where they stand: nothing copies them into names of their own.
    carried = {name for load in statements(kernel, Load) if load.carried for name in load.names}
    assert not any(isinstance(stmt, Assign) and set(stmt.deps()) & carried for stmt in body)
    primes = [i for i, stmt in enumerate(prologue) if isinstance(stmt, Load) and stmt.carried]
    assert primes, "the prime declares the carried registers"
    assert min(primes) > max(i for i, stmt in enumerate(prologue) if isinstance(stmt, Sync)), "chunk 1 issues past the prime's barrier"
    lets = {stmt.name: stmt for stmt in prologue if isinstance(stmt, Let)}
    calls = " ".join(str(stmt.value) for stmt in lets.values())
    assert "emmy_volta_crosswise(" in calls and "emmy_volta_b_congruous(" in calls
    # The global addresses — a base per staged copy — are computed once ahead of the loop, and every load of the
    # loop reads nothing else: an operand's stride along K is a constant of its flat index.
    addresses = {name for name, stmt in lets.items() if stmt.dtype is None}
    assert len(addresses) == 4
    issued = [stmt for stmt in body if isinstance(stmt, Load) and stmt.input in ("a", "b")]
    assert len(issued) == 4 and all(_free_vars(stmt.index) <= addresses for stmt in issued)
    assert "emmy_volta" not in text(Body(tuple(body)))
    assert sum(isinstance(stmt, Sync) for stmt in body) == 1, "the deposit's barrier is the whole per-chunk handshake"
    assert sum(isinstance(stmt, Sync) for stmt in prologue) == 1, "the primed slot is published once before the loop"
    _assert_volta_only(kernel)


def test_sm70_depth_two_single_chunk_keeps_the_blocking_copy_before_the_drain(monkeypatch) -> None:
    """A requested ring that holds only one K chunk is the depth-one blocking copy.

    Splitting that lone copy leaves no resident chunk to overlap: issue-before-drain followed by
    deposit-after-drain makes the first drain read uninitialized shared memory. The depth-two row
    must therefore lower exactly like its correct depth-one control while retaining its authored
    schedule spelling.
    """
    monkeypatch.setenv("EMMY_TILE", f"{VOLTA}/f2x2/k8")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_REDUCE", "")
    ctx = Context(compute_capability=(7, 0))

    monkeypatch.setenv("EMMY_STAGE", "d1/smem")
    depth_one, _, _ = _source(_graph(m=64, n=64, k=32), ctx)
    monkeypatch.setenv("EMMY_STAGE", "d2/smem")
    depth_two, knobs, _ = _source(_graph(m=64, n=64, k=32), ctx)

    assert family_value(knobs, "STAGE") == "d2/smem"
    assert text(depth_two) == text(depth_one)


def test_sm70_shallow_k_tile_keeps_store_addresses_near_the_deposit(monkeypatch) -> None:
    """Hoisting Volta store addresses slows the shallower K tile despite reducing its SASS."""
    monkeypatch.setenv("EMMY_TILE", f"{VOLTA}/f2x2/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem")
    monkeypatch.setenv("EMMY_REDUCE", "")
    kernel, _, _ = _source(_graph(m=64, n=64, k=32), Context(compute_capability=(7, 0)))
    assert not statements(kernel, Let), "no slab store address, no global stride, is hoisted ahead of the loop"
    assert any(write.width > 1 for write in _slab_writes(kernel, "_a_smem", "V70A")), "the deposit places its vector itself"


def test_sm70_attention_uses_the_volta_fragment_layout(monkeypatch) -> None:
    """The softmax algebra and its P-to-A handoff use all eight Volta C registers and the lane
    butterflies that join one logical row."""
    _pin_sdpa(monkeypatch)
    kernel, _, src = _source(_sdpa_graph(), Context(compute_capability=(7, 0)))
    repacks = statements(kernel, FragmentRepack)
    assert repacks and all(repack.fragment_layout == "m8n8k4" for repack in repacks)
    assert any(repack.part in (None, 0) for repack in repacks)
    applies = statements(kernel, FragmentApply)
    assert applies and all(apply.layout == "m8n8k4" for apply in applies)
    reduces = statements(kernel, FragmentRowReduce)
    assert reduces and all(reduce.layout == "m8n8k4" for reduce in reduces)
    _assert_volta_only(kernel)
    # The lanes that share a row join through the butterflies of lane distance 4 and 2.
    butterflies = {line.split(",")[2].strip() for line in _instructions(src) if line.startswith("shfl.sync.bfly.b32")}
    assert {"4", "2"} <= butterflies, butterflies


def test_sm70_causal_attention_selects_the_mask_per_fragment_element(monkeypatch) -> None:
    _pin_sdpa(monkeypatch)
    kernel, _, src = _source(_sdpa_graph(d=256, causal=True, dynamic=True), Context(compute_capability=(7, 0)))
    (bias,) = [apply for apply in statements(kernel, FragmentApply) if apply.op.name == "add"]
    fragment = bias.out
    # The causal mask compares the fragment element's query row with its key column; the key's bound masks it too.
    masks = {str(apply.args[0].expressions[0]): apply for apply in statements(kernel, FragmentApply) if apply.op.name == "where"}
    assert set(masks) == {"__frow < __fcol", "seq_len <= __fcol"}
    assert all(mask.out == fragment and mask.layout == "m8n8k4" for mask in masks.values())
    causal, bound = masks["__frow < __fcol"], masks["seq_len <= __fcol"]
    assert causal.row_base is not None and causal.col_base == bound.col_base, "both read the same key column"
    assert {repack.fragment_layout for repack in statements(kernel, FragmentRepack)} == {"m8n8k4"}
    # The coordinates resolve to the lane's row and column: no instruction reads them by name.
    assert not any("__frow" in line or "__fcol" in line for line in _instructions(src))


def test_sm70_attention_rings_its_key_and_value_through_registers(monkeypatch) -> None:
    """A two-slot ring on the Volta chunk tier is the register-staged split: the next chunk's key
    and value load into packed registers past the barrier, the softmax and expectation of this
    chunk run, and the deposit lands them in the free slot, reading the vectors where it stands. The
    loads stay behind the barrier — a load moved above it would refill vectors the deposit has not
    read yet."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_WORK", "w4x1")
    monkeypatch.setenv("EMMY_TILE@map.1/twist.1/inner", f"{VOLTA}/f1x2/k2")
    monkeypatch.setenv("EMMY_TILE@map.1/twist", f"{VOLTA}/f1x8/k8")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist.1/inner", "")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist", "d2/smem")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    kernel, knobs, _ = _source(_sdpa_graph(d=128, causal=True), Context(compute_capability=(7, 0)))
    assert knobs["STAGE@map.1/twist"] == "d2/smem"
    (chunk,) = [loop for loop in statements(kernel, StridedLoop) if loop.axis.name.endswith("__ck")]
    body = list(chunk.body)
    drain, deposit, barrier, issue = _register_ring_order(body)
    assert drain < deposit < barrier < issue, "the key and value of the chunk after next issue past the barrier"
    carried = {name for load in statements(kernel, Load) if load.carried for name in load.names}
    assert not any(isinstance(stmt, Assign) and set(stmt.deps()) & carried for stmt in body), "nothing copies a carried vector"
    _assert_volta_only(kernel)


def test_sm70_computed_a_edge_stages_through_the_smem_compute_fill(monkeypatch) -> None:
    """A COMPUTED ``a`` edge reaches the Volta mma tier: the fill evaluates the norm cone into the
    A slab the Volta shared gather reads, and the materialized B peer rides the BLOCKING vector
    copy — sm_70 has no ``cp.async`` to fly it under the fill, which is also why the fill stages
    at depth 1 only here."""
    _pin(monkeypatch, VOLTA, tile="f1x1", stage="d1/smem")
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    kernel, knobs, _ = _source(_norm_linear_graph(), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == "d1/smem"
    assert [(load.src_buffer, load.b_trans) for load in _drains(kernel, "a")] == [("_a_smem", False)]
    assert [(load.src_buffer, load.b_trans) for load in _drains(kernel, "b")] == [("_b_smem", True)]
    assert "rsqrt" in operations(kernel)  # the norm cone itself, evaluated into the A slab
    assert any(write.width > 1 for write in _slab_writes(kernel, "_b_smem")), "the peer's blocking vector load/store"
    _assert_volta_only(kernel)


def test_modern_computed_a_edge_keeps_the_cp_async_peer_copy(monkeypatch) -> None:
    """The same fused edge on a cp.async target still flies its peer copy asynchronously — the
    blocking copy is the sm_70 fallback, not a new default."""
    _pin(monkeypatch, AMPERE, tile="f1x1", stage="d1/smem")
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    kernel, _, src = _source(_norm_linear_graph(k=32), Context(compute_capability=(8, 0)))
    assert statements(kernel, CpAsyncCopy)
    assert statements(kernel, LdmatrixLoad) and "rsqrt" in operations(kernel)
    assert not any(load.fragment_layout == "m8n8k4" for load in statements(kernel, FragmentLoad))
    assert not any(line.startswith("mma.sync.aligned.m8n8k4") for line in _instructions(src))


def test_modern_mma_source_holds_no_volta_product(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    kernel, _, src = _source(_graph(k=16), Context(compute_capability=(8, 0)))
    assert {tuple(mma.shape) for mma in statements(kernel, MmaSyncPtx)} == {(16, 8, 16)}
    products = [line for line in _instructions(src) if line.startswith("mma.sync.aligned")]
    assert products and not any(".m8n8k4." in line for line in products)


def test_sm70_modern_atom_pin_restricts_the_schedule_to_empty(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    out = Pipeline.build(TILE_PASSES).run(_graph(k=16), ctx=Context(compute_capability=(7, 0)))
    tile = next(node.op for node in out.nodes.values() if isinstance(node.op, TileOp))

    assert not tile.place.is_mapped and tile.schedule is None


@pytest.mark.parametrize("stage", ["d1/smem-async", "d1/smem-tma"])
def test_sm70_newer_stage_pin_refuses(monkeypatch, stage) -> None:
    _pin(monkeypatch, VOLTA, stage=stage)
    with pytest.raises(ValueError, match="requires sm_"):
        Pipeline.build(TILE_PASSES).run(_graph(), ctx=Context(compute_capability=(7, 0)))


def test_requested_target_reaches_the_ptxas_arch_and_cubin_key(monkeypatch) -> None:
    monkeypatch.setattr(ptxas, "version", lambda: "ptxas")
    try:
        set_target((7, 0))
        arch70 = ptxas.device_arch(False)
        key70 = ptxas._cubin_key("source", arch70)
        set_target((8, 0))
        arch80 = ptxas.device_arch(False)
        key80 = ptxas._cubin_key("source", arch80)
    finally:
        set_target(None)
    assert (arch70, arch80) == ("sm_70", "sm_80")
    assert key70 != key80
