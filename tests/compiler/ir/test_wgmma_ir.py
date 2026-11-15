"""The Hopper warp-group MMA statements of the kernel IR: their text form, the descriptor's bit layout, and the
arch-suffixed target they ask for — none of which needs a Hopper card. The PTX each compiles to is the PTX compiler's
(``crates/emmy-ptx/tests/hopper.rs``)."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.kernel.ir import (
    Var,
    WgmmaCommit,
    WgmmaDescriptor,
    WgmmaFence,
    WgmmaMma,
    WgmmaWait,
    wgmma_descriptor_bits,
)

_NS = (64, 128, 256)
_DTYPES = ("f16", "bf16")
_FORMS = ("ss", "rs")


def _frags(n: int) -> tuple[str, ...]:
    return tuple(f"c{j}" for j in range(n // 8))


def _cell(n: int = 64, ab_dtype: str = "bf16", form: str = "ss", **kw) -> WgmmaMma:
    a = {"a_desc": "aD"} if form == "ss" else {"a_frag": "a"}
    return WgmmaMma(c_frags=_frags(n), b_desc="bD", shape=(64, n, 16), ab_dtype=ab_dtype, **a, **kw)


@pytest.mark.parametrize("swizzle", ["NONE", "B128", "B64", "B32", "B128@7"])
def test_the_descriptor_reads_its_slab(swizzle: str) -> None:
    desc = WgmmaDescriptor(name="bD", smem="K_s", smem_index=Var("k0") * 64, swizzle=swizzle, lbo_bytes=16, sbo_bytes=1024)
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


@pytest.mark.parametrize("n", _NS)
@pytest.mark.parametrize("ab_dtype", _DTYPES)
@pytest.mark.parametrize("form", _FORMS)
def test_the_text_form_spells_n_dtype_and_form(n: int, ab_dtype: str, form: str) -> None:
    cell = _cell(n, ab_dtype, form)
    acc = ", ".join(_frags(n))
    operand = "a_desc=aD" if form == "ss" else "a_frag=a"
    dtype = "" if ab_dtype == "f16" else f", ab_dtype={ab_dtype}"
    assert str(cell) == f"WgmmaMma(c_frags=({acc}), b_desc=bD, shape=(64, {n}, 16){dtype}, {operand})"


def test_the_transpose_bits_and_scale_d_stand_in_the_text_form() -> None:
    ss = _cell(64, "f16", "ss", scale_d=0, trans_a=1, trans_b=1)
    assert str(ss).endswith("a_desc=aD, scale_d=0, trans_a=1, trans_b=1)")


def test_fence_commit_wait_read_and_define_nothing() -> None:
    for stmt in (WgmmaFence(), WgmmaCommit(), WgmmaWait()):
        assert stmt.deps() == () and stmt.defines() == ()
    assert [str(s) for s in (WgmmaFence(), WgmmaCommit(), WgmmaWait(2))] == ["WgmmaFence", "WgmmaCommit", "WgmmaWait(group=2)"]
