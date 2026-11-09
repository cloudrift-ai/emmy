"""The CUDA render target's scalar conversions: every pair of narrow formats has a rule or is refused."""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget


def test_two_narrow_floats_convert_through_f32():
    """No device intrinsic converts fp16 into bf16 or back, nor bf16 into fp8: the spelling goes through f32, which
    holds every value of either, so the one rounding is the destination's — as a typed definition of one over a name
    of the other needs (``bf16 v = x`` with ``x`` fp16)."""
    target = CudaRenderTarget()
    assert target.convert("x", "f16", "bf16") == "__float2bfloat16(__half2float(x))"
    assert target.convert("x", "bf16", "f16") == "__float2half(__bfloat162float(x))"
    assert target.convert("x", "bf16", "f8e4m3") == "__nv_fp8_e4m3(__bfloat162float(x))"
    assert target.convert("x", "f8e5m2", "bf16") == "__float2bfloat16(float(x))"


def test_an_unhandled_pair_of_packed_formats_is_refused_at_render_time():
    """A pair of packed formats with no rule has no conversion C++ supplies, or one it supplies wrongly — a pair of fp4
    codes is spelled ``unsigned char`` and would decode as an integer: it fails naming the pair, not in nvcc."""
    with pytest.raises(ValueError, match="no conversion from 'f16x2' to 'f8e4m3'"):
        CudaRenderTarget().convert("x", "f16x2", "f8e4m3")
    with pytest.raises(ValueError, match="no conversion from 'f4e2m1x2' to 'f16'"):
        CudaRenderTarget().convert("x", "f4e2m1x2", "f16")
    assert CudaRenderTarget().convert("x", "u8", "f4e2m1x2") == "x"


def test_native_pairs_and_the_identity_pass_through():
    target = CudaRenderTarget()
    assert target.convert("x", "f32", "f32") == "x"
    assert target.convert("x", "u16", "u64") == "x"
