"""The CUDA render target's scalar conversions: every pair of narrow formats has a rule or is refused."""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget


def test_the_two_half_formats_convert_through_f32():
    """No device intrinsic converts fp16 into bf16 or back: the spelling goes through f32 with the round explicit, as
    a typed definition of one over a name of the other needs (``bf16 v = x`` with ``x`` fp16)."""
    target = CudaRenderTarget()
    assert target.convert("x", "f16", "bf16") == "__float2bfloat16(__half2float(x))"
    assert target.convert("x", "bf16", "f16") == "__float2half(__bfloat162float(x))"


def test_an_unhandled_pair_of_narrow_formats_is_refused_at_render_time():
    """A pair of narrow formats with no rule has no conversion C++ supplies: it fails naming the pair, not in nvcc."""
    with pytest.raises(ValueError, match="no conversion from 'bf16' to 'f8e4m3'"):
        CudaRenderTarget().convert("x", "bf16", "f8e4m3")


def test_native_pairs_and_the_identity_pass_through():
    target = CudaRenderTarget()
    assert target.convert("x", "f32", "f32") == "x"
    assert target.convert("x", "u16", "u64") == "x"
