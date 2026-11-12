"""Target selection and source lowering for the Volta m8n8k4 MMA family."""

from __future__ import annotations

import re

import pytest

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.atom import ATOM_REGISTRY, atoms_for
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SdpaOp
from emmy.compiler.ir.render import render_stmt
from emmy.compiler.ir.schedule.catalog import MAX_FRAGMENT_REGISTERS, warp_tile_moves
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_value
from emmy.compiler.target import set_target

VOLTA = "mma_m8n8k4_f16_f32"
AMPERE = "mma_m16n8k16_f16_f32"

# Every instruction a newer target has and sm_70 does not — the one list the source assertions share.
NEWER_INSTRUCTIONS = ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16")


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


def _kernel(src: str) -> str:
    """The kernel function of ``src``, past the prelude that defines every helper it may call. A kernel body is in
    normal form, so its fragments and axes carry the normal form's names (``v0``, ``a0``): the assertions below read
    the helpers, layouts and statement order, never those names."""
    return src[src.index("__global__") :]


def _source(graph: Graph, ctx: Context) -> tuple[str, dict]:
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    kernels = [node.op for node in lowered.nodes.values() if getattr(node.op, "kernel_source", None)]
    assert len(kernels) == 1
    return kernels[0].kernel_source, kernels[0].knobs


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
    assert tuple(atom.fragment_nregs(role) for role in ("a", "b", "c")) == (2, 2, 8)
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
    src, knobs = _source(_graph(m=32, n=128, k=256, trans=True), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f2x4/k16"
    assert "_b_smem[emmy_volta_crosswise(" in src
    assert re.search(r"for \(int (\w+) = 0; \1 < 256; \1 \+= 64\)", src), "one chunk carries 64 K elements"


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_source_uses_only_the_volta_mma_family(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA)
    src, knobs = _source(_graph(trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == ""
    assert "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32" in src
    kernel = _kernel(src)
    assert len(re.findall(r"unsigned \w+\[2\];", kernel)) == 2, "one A and one B fragment"
    assert len(re.findall(r"float \w+\[8\] = ", kernel)) == 1, "one accumulator"
    assert ("emmy_mma884_load_b_gmem_trans(" in kernel) == trans
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16", ".bf16", ".e4m3", ".e5m2"):
        assert forbidden not in src


@pytest.mark.parametrize("role,trans", [("a", False), ("b", True), ("b", False)])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize(
    "stride,offset,masked_k,aligned",
    [(32, 4, False, True), (32, 5, False, False), (31, 4, False, False), (32, 4, True, False)],
    ids=["aligned", "offset", "stride", "k-tail"],
)
def test_volta_direct_vectors_require_aligned_complete_runs(role, trans, dtype, stride, offset, masked_k, aligned):
    from emmy.compiler.ir.expression import Literal, Var
    from emmy.compiler.ir.kernel.ir import LdmatrixLoad
    from emmy.compiler.ir.render import RenderCtx

    ctx = RenderCtx(shapes={"x": (Dim(17), Dim(stride))}, buffer_dtypes={"x": dtype}, ssa_dtypes={"r": "f16"})
    load = LdmatrixLoad(
        frag="r",
        src_buffer="x",
        src_index=(Var("row"), Literal(offset, "int")),
        role=role,
        staged=False,
        b_trans=trans,
        ldm=stride,
        fragment_layout="m8n8k4",
        gmem_guard=(Literal(0, "int"), Literal(17, "int")),
        k_zero=(Literal(0, "int"), Literal(3, "int")) if masked_k else None,
    )
    source = "\n".join(render_stmt(load, ctx))
    assert ("load_gmem4<" in source) == (aligned and (role == "a" or trans))


def test_sm70_m1_linear_synthesizes_a_masked_mma_row(monkeypatch) -> None:
    """A literal unit output row is still the M side of a plain m1xKxN contraction.

    RESTORED: the decode row is the only shape a V100 deployment runs at serving time, and without
    it the m=1 linear falls to the scalar path with no assertion anywhere noticing.
    """
    _pin(monkeypatch, VOLTA, tile="f1x1", stage="")
    src, knobs = _source(_graph(m=1, n=16, k=16, trans=True), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert knobs["WORK"] == "w1x1"
    assert "emmy_mma_m8n8k4_f16_f32" in src
    assert "emmy_mma884_load_a_gmem_mclamp(" in _kernel(src) and "< (1)" in src


@pytest.mark.parametrize("trans", [False, True])
def test_sm70_sync_copy_stages_fragments_without_newer_instructions(monkeypatch, trans) -> None:
    _pin(monkeypatch, VOLTA, stage="d1/smem")
    src, knobs = _source(_graph(k=16, trans=trans), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "STAGE") == "d1/smem"
    # Every staged slab now declares its fill-chunk alignment, so match the declaration from the
    # dtype on — the attribute between ``__shared__`` and the type is not what this test is about.
    assert "__half _a_smem[64]" in src
    assert "__half _b_smem[64]" in src
    assert re.search(r"emmy_mma884_load_a_smem\(\w+, &_a_smem", src)
    b_helper = "emmy_mma884_load_b_smem_trans" if trans else "emmy_mma884_load_b_smem"
    assert re.search(rf"{b_helper}\(\w+, &_b_smem", src)
    assert src.count("__syncthreads();") == 2
    for forbidden in ("ldmatrix", "cp.async", "cp.async.bulk", "m16n8k16"):
        assert forbidden not in src


def test_sm70_contiguous_staged_fragments_use_one_wide_load(monkeypatch) -> None:
    """A and N-major B fragments are contiguous in their staged slabs, so each drains as one uint2."""
    _pin(monkeypatch, VOLTA, stage="d1/smem")
    src, _ = _source(_graph(k=16, trans=True), Context(compute_capability=(7, 0)))
    assert "uint2 v = *reinterpret_cast<const uint2*>(g);" in src
    assert "emmy_mma884_load4(r, s + row * ldm);" in src
    assert "emmy_mma884_load4(r, s + col * ldm);" in src


def test_sm70_materialized_tiles_use_paired_volta_layout_loads(monkeypatch) -> None:
    """Even Volta register tiles derive the coupled CUTLASS fill/drain layouts by default."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    src, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    assert "_a_smem[emmy_volta_crosswise(" in src
    assert "_b_smem[emmy_volta_b_congruous(" in src
    assert "int row = e / ldm;" not in src
    kernel = _kernel(src)
    assert re.search(r"emmy_mma884_load_a_crosswise_pair\(\w+, \w+, _a_smem", kernel)
    assert re.search(r"emmy_mma884_load_b_congruous_pair\(\w+, \w+, _b_smem", kernel)
    assert kernel.count("emmy_mma_m8n8k4_f16_f32_brow(") == 4
    assert "access ^ ((k_group >> 2) & 1)" in src
    assert "const int _vq = _vl >> 2;" in src
    assert "((_vq >> 1) & 1) * 8" in src


def test_sm70_transposed_b_drains_the_crosswise_layout(monkeypatch) -> None:
    """A transposed B stages K-contiguous like A, so it reads A's crosswise layout back rather
    than falling to the plain row-major tile whose 64-byte row is bank-conflicted."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    src, _ = _source(_graph(m=32, n=32, k=16, trans=True), Context(compute_capability=(7, 0)))
    assert "_a_smem[emmy_volta_crosswise(" in src
    assert "_b_smem[emmy_volta_crosswise(" in src
    assert "_b_smem[emmy_volta_b_congruous(" not in src, "the congruous layout is for an N-contiguous B"
    kernel = _kernel(src)
    assert re.search(r"emmy_mma884_load_a_crosswise_pair\(\w+, \w+, _a_smem", kernel)
    assert re.search(r"emmy_mma884_load_b_crosswise_pair\(\w+, \w+, _b_smem", kernel)
    assert "emmy_mma884_load_b_smem_trans(" not in kernel, "the conflicted plain gather must be gone"
    # The B reader is A's with lane bits 2 and 3 swapped, and the row/col mma form stays.
    assert "(lane & ~0xC) | ((lane & 4) << 1) | ((lane & 8) >> 1)" in src
    assert kernel.count("emmy_mma_m8n8k4_f16_f32(") == 4
    # The interleaved accumulator map is unchanged: the column half still comes from lane bit 3.
    assert "((_vq >> 1) & 1) * 8" in src


def test_sm70_pair_policy_off_retains_the_unpaired_gather(monkeypatch) -> None:
    """The existing policy override disables the coupled layout as one complete choice."""
    _pin(monkeypatch, VOLTA, tile="f2x2", stage="d1/smem")
    monkeypatch.setenv("EMMY_PAIR_LDMATRIX", "0")
    src, knobs = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    assert "_a_smem[emmy_volta_crosswise(" not in src
    assert "_b_smem[emmy_volta_b_congruous(" not in src
    kernel = _kernel(src)
    assert "emmy_mma884_load_a_smem(" in kernel
    assert "emmy_mma884_load_b_smem(" in kernel
    assert "emmy_mma_m8n8k4_f16_f32(" in kernel
    assert knobs["PAIR_LDMATRIX"] is False


def test_sm70_computed_tiles_use_the_same_paired_layouts(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f2x2/k4", stage="d1/smem")
    graph = _norm_linear_graph(m=32, n=32, k=32)
    graph.nodes["y"].op = MatmulOp()
    src, _ = _source(graph, Context(compute_capability=(7, 0)))
    assert "_a_smem[emmy_volta_crosswise(" in src
    assert "_b_smem[emmy_volta_b_congruous(" in src
    assert re.search(r"emmy_mma884_load_a_crosswise_pair\(\w+, \w+, _a_smem", src)
    assert re.search(r"emmy_mma884_load_b_congruous_pair\(\w+, \w+, _b_smem", src)


def test_sm70_gmem_direct_tile_keeps_the_ordinary_accumulator_map(monkeypatch) -> None:
    """The paired accumulator map is coupled to a staged operand layout, never used alone."""
    _pin(monkeypatch, VOLTA, tile="f2x2")
    src, _ = _source(_graph(m=32, n=32, k=16), Context(compute_capability=(7, 0)))
    kernel = _kernel(src)
    assert "emmy_mma_m8n8k4_f16_f32_brow(" not in kernel
    assert kernel.count("emmy_mma_m8n8k4_f16_f32(") == 4
    assert "const int _vq = (_vl & 15) >> 2;" in src


def test_sm70_output_stores_contiguous_fragment_pairs(monkeypatch) -> None:
    """The eight Volta accumulator elements leave registers as four contiguous half2 pairs."""
    _pin(monkeypatch, VOLTA)
    src, _ = _source(_graph(), Context(compute_capability=(7, 0)))
    assert src.count("*reinterpret_cast<__half2*>(&c[") == 4


def test_sm70_masked_rows_keep_contiguous_fragment_pairs(monkeypatch) -> None:
    """A row guard covers both adjacent values, so it does not split their half2 store."""
    _pin(monkeypatch, VOLTA)
    src, _ = _source(_graph(m=Dim("seq_len", hint=512)), Context(compute_capability=(7, 0)))
    assert src.count("*reinterpret_cast<__half2*>(&c[") == 4
    assert "< (seq_len))" in src


def test_sm70_sync_copy_composes_ring_and_register_pipelines(monkeypatch) -> None:
    _pin(monkeypatch, VOLTA, tile="f1x1/k2", stage="d2/smem/p2")
    src, knobs = _source(_graph(k=32), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1/k2"
    assert family_value(knobs, "STAGE") == "d2/smem/p2"
    assert "__half _a_smem[256]" in src
    assert "__half _b_smem[256]" in src
    kernel = _kernel(src)
    assert len(re.findall(r"unsigned \w+\[2\];", kernel)) == 4, "two register slots of the A and the B fragment"
    calls = re.findall(r"emmy_mma_m8n8k4_f16_f32\((\w+), (\w+), (\w+), \1\);", kernel)
    assert len(calls) == 2 and len({c for c, _, _ in calls}) == 1, "both slots accumulate into the one C fragment"
    assert len({(a, b) for _, a, b in calls}) == 2, "each call reads its own slot pair"
    assert "cp.async" not in src and "ldmatrix" not in src


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
    src, knobs = _source(_graph(m=64, n=64, k=64), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "STAGE") == "d2/smem"
    (chunk,) = re.findall(r"for \(int (\w+) = 0; \1 < 64; \1 \+= 32\)", src)
    prologue, _, body = _kernel(src).partition(f"for (int {chunk}")
    drain = body.index("emmy_mma_m8n8k4_f16_f32")
    unpack = re.search(r"\n\s+\w+ = _v_\w+_h\[0\];", body).start()  # the carried vector spreads to the staged names
    deposit = body.index("*reinterpret_cast<uint2*>(&_a_smem[")
    barrier = body.index("__syncthreads();")
    issue = re.search(r"_v_\w+ = __ldg\(", body).start()  # the gmem load of the chunk after next, past the barrier
    assert drain < unpack < deposit < barrier < issue, "the loads issue past the barrier that publishes the deposit"
    assert re.search(r"uint4 _v_\w+ = \*reinterpret_cast", prologue), "the prime declares the carried registers"
    assert re.search(r"_v_\w+ = __ldg\(", prologue).start() > prologue.rindex("__syncthreads();"), "chunk 1 issues past the prime's barrier"
    assert re.search(r"int \w+ = emmy_volta_crosswise\(", prologue)
    assert re.search(r"int \w+ = emmy_volta_b_congruous\(", prologue)
    # The global addresses — a base per staged copy and a stride per operand — are computed once ahead of the loop,
    # and every load of the loop reads nothing else.
    addresses = set(re.findall(r"\bauto (\w+) = ", prologue))
    assert len(addresses) == 6
    issued = re.findall(r"__ldg\(reinterpret_cast<const uint4\*>\(&[ab]\[([^\]]*)\]", body)
    assert len(issued) == 4 and all(set(re.findall(r"[A-Za-z_]\w*", index)) <= addresses for index in issued)
    assert "emmy_volta_crosswise(" not in body and "emmy_volta_b_congruous(" not in body
    assert body.count("__syncthreads();") == 1, "the deposit's barrier is the whole per-chunk handshake"
    assert prologue.count("__syncthreads();") == 1, "the primed slot is published once before the loop"
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


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
    depth_one, _ = _source(_graph(m=64, n=64, k=32), ctx)
    monkeypatch.setenv("EMMY_STAGE", "d2/smem")
    depth_two, knobs = _source(_graph(m=64, n=64, k=32), ctx)

    assert family_value(knobs, "STAGE") == "d2/smem"
    assert depth_two == depth_one


def test_sm70_shallow_k_tile_keeps_store_addresses_near_the_deposit(monkeypatch) -> None:
    """Hoisting Volta store addresses slows the shallower K tile despite reducing its SASS."""
    monkeypatch.setenv("EMMY_TILE", f"{VOLTA}/f2x2/k4")
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem")
    monkeypatch.setenv("EMMY_REDUCE", "")
    src, _ = _source(_graph(m=64, n=64, k=32), Context(compute_capability=(7, 0)))
    assert "int _a_smem_store" not in src and "int _b_smem_store" not in src
    assert "_a_gmem_stride" not in src and "_b_gmem_stride" not in src
    assert "*reinterpret_cast<uint2*>(&_a_smem[emmy_volta_crosswise(" in src


def test_sm70_attention_uses_the_volta_fragment_layout(monkeypatch) -> None:
    """The softmax algebra and its P-to-A handoff use all eight Volta C registers and the lane
    butterflies that join one logical row."""
    _pin_sdpa(monkeypatch)
    src, _ = _source(_sdpa_graph(), Context(compute_capability=(7, 0)))
    assert "emmy_c_to_a_f16_m8n8k4<0>" in src
    assert "for (int _e = 0; _e < 8; _e++)" in src
    assert "__shfl_xor_sync(0xffffffff" in src
    assert ", 4)" in src and ", 2)" in src
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


def test_sm70_causal_attention_selects_the_mask_per_fragment_element(monkeypatch) -> None:
    _pin_sdpa(monkeypatch)
    src, _ = _source(_sdpa_graph(d=256, causal=True, dynamic=True), Context(compute_capability=(7, 0)))
    (fragment,) = re.findall(r"(\w+)\[_e\] = \w+\[_e\] \+ __half2float\(\w+\);", src)
    assert f"float {fragment}[8]" in src
    # The causal mask compares the fragment element's query row with its key column; the key's bound masks it too.
    (key,) = re.findall(rf"{fragment}\[0\] = \(\(\w.* \+ _vr < (\w+) \+ _vc\) \? \(-1e\+30f\) : \({fragment}\[0\]\)\);", src)
    assert f"{fragment}[0] = ((seq_len <= {key} + _vc) ? (-1e+30f) : ({fragment}[0]));" in src
    assert "__frow" not in src and "__fcol" not in src
    assert "emmy_c_to_a_f16_m8n8k4" in src


def test_sm70_attention_rings_its_key_and_value_through_registers(monkeypatch) -> None:
    """A two-slot ring on the Volta chunk tier is the register-staged split: the next chunk's key
    and value load into packed registers past the barrier, the softmax and expectation of this
    chunk run, and the unpack and deposit land them in the free slot. The loads stay behind the
    barrier — a load moved above it would refill the vectors the unpack has not read yet."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    monkeypatch.setenv("EMMY_WORK", "w4x1")
    monkeypatch.setenv("EMMY_TILE@map.1/twist.1/inner", f"{VOLTA}/f1x2/k2")
    monkeypatch.setenv("EMMY_TILE@map.1/twist", f"{VOLTA}/f1x8/k8")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist.1/inner", "")
    monkeypatch.setenv("EMMY_STAGE@map.1/twist", "d2/smem")
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_RASTER", "")
    src, knobs = _source(_sdpa_graph(d=128, causal=True), Context(compute_capability=(7, 0)))
    assert knobs["STAGE@map.1/twist"] == "d2/smem"
    _, _, body = src.partition("for (int a")
    drain = body.index("emmy_mma_m8n8k4_f16_f32")
    unpack = body.index("_a_stage0_0 = _v__a_stage0_0_h[0];")
    barrier = body.index("__syncthreads();", unpack)
    issue = body.index("_v__a_stage0_0 = __ldg(")
    assert drain < unpack < barrier < issue, "the key and value of the chunk after next issue past the barrier"
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


def test_sm70_computed_a_edge_stages_through_the_smem_compute_fill(monkeypatch) -> None:
    """A COMPUTED ``a`` edge reaches the Volta mma tier: the fill evaluates the norm cone into the
    A slab the Volta shared gather reads, and the materialized B peer rides the BLOCKING vector
    copy — sm_70 has no ``cp.async`` to fly it under the fill, which is also why the fill stages
    at depth 1 only here."""
    _pin(monkeypatch, VOLTA, tile="f1x1", stage="d1/smem")
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    src, knobs = _source(_norm_linear_graph(), Context(compute_capability=(7, 0)))
    assert family_value(knobs, "TILE") == f"{VOLTA}/f1x1"
    assert family_value(knobs, "STAGE") == "d1/smem"
    assert re.search(r"emmy_mma884_load_a_smem\(\w+, &_a_smem", src)
    assert re.search(r"emmy_mma884_load_b_smem_trans\(\w+, &_b_smem", src)
    assert "rsqrtf" in src  # the norm cone itself, evaluated into the A slab
    assert "*reinterpret_cast<uint2*>(&_b_smem[" in src  # the peer's blocking vector load/store
    for forbidden in NEWER_INSTRUCTIONS:
        assert forbidden not in src


def test_modern_computed_a_edge_keeps_the_cp_async_peer_copy(monkeypatch) -> None:
    """The same fused edge on a cp.async target still flies its peer copy asynchronously — the
    blocking copy is the sm_70 fallback, not a new default."""
    _pin(monkeypatch, AMPERE, tile="f1x1", stage="d1/smem")
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    src, _ = _source(_norm_linear_graph(k=32), Context(compute_capability=(8, 0)))
    assert "emmy_cp_async" in src
    assert "ldmatrix" in src and "rsqrtf" in src
    assert "emmy_mma884" not in src


def test_modern_mma_source_does_not_gain_the_volta_prelude(monkeypatch) -> None:
    _pin(monkeypatch, AMPERE)
    src, _ = _source(_graph(k=16), Context(compute_capability=(8, 0)))
    assert "mma.sync.aligned.m16n8k16" in src
    assert "emmy_mma884" not in src and "mma.sync.aligned.m8n8k4" not in src


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


def test_requested_target_reaches_nvcc_arch_and_cubin_key(monkeypatch) -> None:
    monkeypatch.setattr(nvcc, "_toolkit_tag", lambda: "toolkit")
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    try:
        set_target((7, 0))
        arch70 = nvcc.device_arch(False)
        key70 = nvcc._cubin_key("source", "kernel", arch70)
        set_target((8, 0))
        arch80 = nvcc.device_arch(False)
        key80 = nvcc._cubin_key("source", "kernel", arch80)
    finally:
        set_target(None)
    assert (arch70, arch80) == ("sm_70", "sm_80")
    assert key70 != key80
