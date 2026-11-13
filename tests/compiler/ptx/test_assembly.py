"""A kernel's PTX is assembled by ptxas into the cubin cache, with no card."""

from __future__ import annotations

import pytest
from emmy.emmy_ir import compile_ptx, parse_body

from emmy.compiler.backend.cuda import ptxas
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
    if ptxas.ptxas_path() is None:
        pytest.skip("ptxas is not installed")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path))
    interface = Interface((Param("a", "input", F32, "a"), Param("o", "output", F32, "o")), 256, {}, 0, False)
    ptx = compile_ptx("k", parse_body(BODY), interface, sm=sm)
    cubin = ptxas.compile_to_cubin(ptx, "k", arch=f"sm_{sm}")
    assert cubin.parent == tmp_path and cubin.stat().st_size > 0
    assert ptxas.compile_to_cubin(ptx, "k", arch=f"sm_{sm}") == cubin
    # The key is the text, the target and ptxas's version: one more of any is another cubin.
    assert ptxas.compile_to_cubin(ptx + "\n", "k", arch=f"sm_{sm}") != cubin
    if sm == 80:
        assert ptxas.compile_to_cubin(ptx, "k", arch="sm_90") != cubin
    monkeypatch.setattr(ptxas, "version", lambda: "another")
    assert ptxas.compile_to_cubin(ptx, "k", arch=f"sm_{sm}") != cubin


def test_no_ptxas_is_an_error_that_says_what_to_install(monkeypatch, tmp_path):
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path))
    monkeypatch.setattr(ptxas, "ptxas_path", lambda: None)
    monkeypatch.setattr(ptxas, "version", lambda: "absent")
    with pytest.raises(RuntimeError, match="ptxas not found.*install"):
        ptxas.compile_to_cubin(".version 8.7", "k", arch="sm_120")
