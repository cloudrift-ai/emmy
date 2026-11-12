"""A kernel whose source is PTX is assembled by ptxas into the cubin cache, with no card."""

from __future__ import annotations

import pytest
from emmy.emmy_ir import compile_ptx, parse_body

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.dtype import F32
from emmy.compiler.ir.kernel.interface import Interface, Param

BODY = """
Grid(axes=(i in 0..1000), decode=(block_idx.x * 256 + thread_idx.x), block_threads=256):
    if block_idx.x * 256 + thread_idx.x < 1000:
        f32 x = load a[i]
        f32 o[i] = x
"""


@pytest.mark.parametrize("sm", [80, 120])
def test_a_ptx_source_is_assembled_into_the_cubin_cache(tmp_path, monkeypatch, sm):
    if nvcc.nvcc_path() is None:
        pytest.skip("the CUDA toolkit is not installed")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path))
    interface = Interface((Param("a", "input", F32, "a"), Param("o", "output", F32, "o")), 256, {}, 0, False)
    ptx = compile_ptx("k", parse_body(BODY), interface, sm=sm)
    assert nvcc.is_ptx(ptx)
    assert not nvcc.is_ptx('extern "C" __global__ void k(float* o) { o[0] = 0.f; }')
    cubin = nvcc.compile_to_cubin(ptx, "k", arch=f"sm_{sm}")
    assert cubin.parent == tmp_path and cubin.stat().st_size > 0
    assert nvcc.compile_to_cubin(ptx, "k", arch=f"sm_{sm}") == cubin
