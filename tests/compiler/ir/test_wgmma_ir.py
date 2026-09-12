"""The Hopper warp-group MMA statements of the kernel IR: render snapshots, the def-use surface, the
rewrite round trip, the descriptor's bit layout, and a compile-only build of every wrapper form at
``sm_90a`` — none of which needs a Hopper card."""

from __future__ import annotations

import subprocess

import pytest

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.backend.plan import _needs_arch_isa
from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    Literal,
    MmaSyncPtx,
    RegFragment,
    Smem,
    Var,
    WgmmaCommit,
    WgmmaDescriptor,
    WgmmaFence,
    WgmmaMma,
    WgmmaWait,
    Write,
    wgmma_descriptor_bits,
)
from emmy.compiler.ir.kernel.render import _WGMMA_PRELUDE, _wgmma_prelude, render_kernelop
from emmy.compiler.ir.stmt import Body, RenderCtx
from emmy.compiler.ir.stmt.passes import rewrite

_NS = (64, 128, 256)
_DTYPES = ("f16", "bf16")
_FORMS = ("ss", "rs")


def _frags(n: int) -> tuple[str, ...]:
    return tuple(f"c{j}" for j in range(n // 8))


def _cell(n: int = 64, ab_dtype: str = "bf16", form: str = "ss", **kw) -> WgmmaMma:
    a = {"a_desc": "aD"} if form == "ss" else {"a_frag": "a"}
    return WgmmaMma(c_frags=_frags(n), b_desc="bD", shape=(64, n, 16), ab_dtype=ab_dtype, **a, **kw)


def _line(stmt) -> str:
    return stmt.render(RenderCtx())[0].strip()


@pytest.mark.parametrize(("swizzle", "mode"), [("NONE", 0), ("B128", 1), ("B64", 2), ("B32", 3), ("B128@7", 1)])
def test_the_descriptor_renders_its_swizzle_as_the_layout_type(swizzle: str, mode: int) -> None:
    desc = WgmmaDescriptor(name="bD", smem="K_s", smem_index=Var("k0") * 64, swizzle=swizzle, lbo_bytes=16, sbo_bytes=1024)
    assert _line(desc) == f"unsigned long long bD = emmy_wgmma_desc(emmy_smem_u32(&K_s[k0 * 64]), 16, 1024, {mode});"
    whole = WgmmaDescriptor(name="aD", smem="Q_s", smem_index=None, swizzle=swizzle, lbo_bytes=16, sbo_bytes=1024)
    assert _line(whole) == f"unsigned long long aD = emmy_wgmma_desc(emmy_smem_u32(Q_s), 16, 1024, {mode});"
    assert desc.defines() == ("bD",) and desc.deps() == () and desc.external_reads() == ("K_s",)


def test_the_descriptor_bits_of_a_128b_swizzled_bf16_slab() -> None:
    """A 64 x 64 bf16 K-major slab under the 128-byte swizzle: rows are 128 bytes, an 8-row core-matrix
    group is 1024 bytes (the stride byte offset), the leading byte offset is the conventional 16 for a
    swizzled K-major operand, and the layout type is B128."""
    bits = wgmma_descriptor_bits(0x400, 16, 1024, 1)
    assert bits & 0x3FFF == 0x40  # start address >> 4
    assert (bits >> 16) & 0x3FFF == 1  # leading byte offset >> 4
    assert (bits >> 32) & 0x3FFF == 64  # stride byte offset >> 4
    assert (bits >> 49) & 0x7 == 0  # base offset
    assert bits >> 62 == 1  # layout type
    assert bits == 0x4000004000010040
    # The fields are 14 bits wide: an address past 256 KiB of smem cannot leak into the next field.
    assert wgmma_descriptor_bits(0x40000, 0, 0, 0) == 0


def test_the_c_prelude_spells_the_same_descriptor_formula() -> None:
    for shift in ("(addr >> 4) & 0x3FFF", "((lbo >> 4) & 0x3FFF) << 16", "((sbo >> 4) & 0x3FFF) << 32", "(mode << 62)"):
        assert shift in _WGMMA_PRELUDE


@pytest.mark.parametrize("n", _NS)
@pytest.mark.parametrize("ab_dtype", _DTYPES)
@pytest.mark.parametrize("form", _FORMS)
def test_the_wrapper_name_spells_n_dtype_and_form(n: int, ab_dtype: str, form: str) -> None:
    cell = _cell(n, ab_dtype, form)
    acc = ", ".join(_frags(n))
    a, tnsp = ("aD", "<0, 0>") if form == "ss" else ("a", "<0>")
    assert _line(cell) == f"emmy_wgmma_m64n{n}k16_{ab_dtype}_f32_{form}{tnsp}({acc}, {a}, bD, 1);"
    assert cell.pretty()[0] == f"WgmmaMma {{{acc}}} += {a} @ bD (m64n{n}k16 {ab_dtype} {form})"


def test_the_transpose_bits_and_scale_d_reach_the_call() -> None:
    ss = _cell(64, "f16", "ss", scale_d=0, trans_a=1, trans_b=1)
    assert _line(ss).endswith("_f32_ss<1, 1>(c0, c1, c2, c3, c4, c5, c6, c7, aD, bD, 0);")
    assert ss.pretty()[0].endswith("(m64n64k16 f16 ss scale_d=0 trans_a trans_b)")
    rs = _cell(64, "f16", "rs", trans_b=1)
    assert _line(rs).endswith("_f32_rs<1>(c0, c1, c2, c3, c4, c5, c6, c7, a, bD, 1);")


def test_the_cell_rejects_malformed_operands() -> None:
    with pytest.raises(ValueError, match="exactly one of a_desc / a_frag"):
        WgmmaMma(c_frags=_frags(64), b_desc="bD", shape=(64, 64, 16), a_desc="aD", a_frag="a")
    with pytest.raises(ValueError, match="8 m16n8 C fragments"):
        WgmmaMma(c_frags=_frags(128), b_desc="bD", shape=(64, 64, 16), a_desc="aD")
    with pytest.raises(ValueError, match="K-major"):
        _cell(64, "f16", "rs", trans_a=1)
    with pytest.raises(ValueError, match="m64n"):
        WgmmaMma(c_frags=("c0",), b_desc="bD", shape=(16, 8, 16), a_desc="aD")


def test_fence_commit_wait_render_the_bare_instructions() -> None:
    assert _line(WgmmaFence()) == 'asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");'
    assert _line(WgmmaCommit()) == 'asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");'
    assert _line(WgmmaWait()) == 'asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");'
    assert _line(WgmmaWait(1)) == 'asm volatile("wgmma.wait_group.sync.aligned 1;\\n" ::: "memory");'
    for stmt in (WgmmaFence(), WgmmaCommit(), WgmmaWait()):
        assert stmt.deps() == () and stmt.defines() == ()
    assert [s.pretty()[0] for s in (WgmmaFence(), WgmmaCommit(), WgmmaWait(2))] == ["WgmmaFence", "WgmmaCommit", "WgmmaWait(2)"]


def test_the_cell_reads_its_operands_and_defines_its_accumulator() -> None:
    ss = _cell(64, form="ss")
    assert ss.deps() == (*_frags(64), "aD", "bD") and ss.defines() == _frags(64)
    rs = _cell(64, form="rs")
    assert rs.deps() == (*_frags(64), "a", "bD") and rs.defines() == _frags(64)


def test_rewrite_renames_ssa_names_and_keeps_buffers() -> None:
    body = Body(
        (
            WgmmaDescriptor(name="bD", smem="K_s", smem_index=Var("k0") * 64, swizzle="B128", lbo_bytes=16, sbo_bytes=1024),
            WgmmaFence(),
            _cell(64, form="ss"),
            _cell(64, form="rs"),
            WgmmaCommit(),
            WgmmaWait(),
        )
    )
    renamed = body.map(lambda s: rewrite(s, lambda name: f"{name}_r"))
    desc, fence, ss, rs, commit, wait = renamed
    assert desc == WgmmaDescriptor(name="bD_r", smem="K_s", smem_index=Var("k0_r") * 64, swizzle="B128", lbo_bytes=16, sbo_bytes=1024)
    assert ss.c_frags == tuple(f"{c}_r" for c in _frags(64)) and (ss.a_desc, ss.b_desc) == ("aD_r", "bD_r")
    assert (rs.a_frag, rs.a_desc, rs.b_desc) == ("a_r", None, "bD_r")
    assert (fence, commit, wait) == (WgmmaFence(), WgmmaCommit(), WgmmaWait())
    assert body.map(lambda s: rewrite(s, lambda name: name)) == body


def test_the_prelude_joins_only_with_a_wgmma_cell() -> None:
    plain = KernelOp(body=Body((MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 16)),)), name="k")
    assert _wgmma_prelude(plain) == ""
    assert "emmy_wgmma_" not in render_kernelop(plain)
    op = KernelOp(body=Body((_cell(64, "bf16", "ss"), _cell(64, "bf16", "ss", trans_b=1), _cell(64, "f16", "rs"))), name="k")
    prelude = _wgmma_prelude(op)
    assert prelude.count("emmy_smem_u32(") == 1 and prelude.count("emmy_wgmma_desc(") == 1
    # One wrapper per (N, dtype, form) — the two ss cells share theirs; the transpose bits are template
    # arguments, which is why the wrapper count does not grow with them.
    assert prelude.count("static __device__ __forceinline__ void emmy_wgmma_") == 2
    assert prelude.count("void emmy_wgmma_m64n64k16_bf16_f32_ss(") == 1 and prelude.count("void emmy_wgmma_m64n64k16_f16_f32_rs(") == 1
    # Operand order: d0..d31 fragment-major / register-minor, then A, B, the scale-d predicate and the
    # scale-A / scale-B / transpose immediates.
    d_regs = ", ".join(f"%{i}" for i in range(32))
    assert f"wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{d_regs}}}, %32, %33, p, 1, 1, %35, %36;" in prelude
    assert f"wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {{{d_regs}}}, {{%32, %33, %34, %35}}, %36, p, 1, 1, %38;" in prelude
    assert "setp.ne.b32 p, %34, 0;" in prelude and "setp.ne.b32 p, %37, 0;" in prelude
    assert ': "+f"(c0[0]), "+f"(c0[1]), "+f"(c0[2]), "+f"(c0[3]), "+f"(c1[0]),' in prelude
    assert '"+f"(c7[3])\n' in prelude
    assert ': "l"(a_desc), "l"(b_desc), "r"(scale_d), "n"(TnspA), "n"(TnspB));' in prelude
    assert ': "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b_desc), "r"(scale_d), "n"(TnspB));' in prelude
    assert render_kernelop(op).startswith(prelude)


def test_a_kernel_carrying_a_wgmma_cell_asks_for_the_arch_suffixed_target() -> None:
    class _Op:
        tma_descriptors = ()
        kernel_source = "... emmy_wgmma_m64n128k16_bf16_f32_ss<0, 0>(c0, c1, aD, bD, 1); ..."

    class _Plain:
        tma_descriptors = ()
        kernel_source = "... emmy_mma_m16n8k16_f16_f32(c, a, b, c); ..."

    assert _needs_arch_isa(_Op())
    assert not _needs_arch_isa(_Plain())


def _driver() -> str:
    """One kernel issuing every wrapper form from IR statements — the compile-only stand-in for the
    lowering's leaf: a 128B-swizzled slab read through two descriptors, the N/8 C fragments each
    cell accumulates into, the register-form A fragment, the fence / commit / wait around them, and
    a store of every accumulator register after the wait so no cell is dead. The first cell starts
    the accumulator (``scale_d=0``); every later one accumulates, so ptxas keeps the whole chain."""
    forms = [(n, ab, form) for n in _NS for ab in _DTYPES for form in _FORMS]
    stmts = [
        Smem(name="K_s", extents=(64 * 64,), dtype="unsigned short", align=1024),
        *(RegFragment(name=f"c{j}", role="c", shape=(16, 8, 16), dtype=F32) for j in range(256 // 8)),
        RegFragment(name="a", role="a", shape=(16, 8, 16), dtype=F16),
        WgmmaDescriptor(name="aD", smem="K_s", smem_index=None, swizzle="B128", lbo_bytes=16, sbo_bytes=1024),
        WgmmaDescriptor(name="bD", smem="K_s", smem_index=Literal(64 * 8, "int"), swizzle="B128", lbo_bytes=16, sbo_bytes=1024),
        WgmmaFence(),
        *(_cell(n, ab, form, scale_d=int(i > 0), trans_b=int(ab == "f16")) for i, (n, ab, form) in enumerate(forms)),
        WgmmaCommit(),
        WgmmaWait(),
        *(Write(output="out", index=(Literal(4 * j + k, "int"),), value=f"c{j}[{k}]") for j in range(256 // 8) for k in range(4)),
    ]
    return render_kernelop(KernelOp(body=Body(tuple(stmts)), name="k_wgmma_probe"), shapes={"out": (128,)})


@pytest.mark.skipif(nvcc.nvcc_path() is None, reason="nvcc unavailable")
def test_every_wrapper_form_builds_at_sm_90a(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nvcc, "cubin_cache_dir", lambda: tmp_path)
    source = _driver()
    assert nvcc.compile_to_cubin(source, "k_wgmma_probe", arch="sm_90a").exists()
    # The instruction is arch-specific: the same source is refused at the plain target, which is
    # what ``_needs_arch_isa`` guards.
    with pytest.raises(subprocess.CalledProcessError) as err:
        nvcc.compile_to_cubin(source, "k_wgmma_probe", arch="sm_90")
    assert b"wgmma" in err.value.stderr
