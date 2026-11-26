"""Flat-address algebra: what a vector copy's source alignment proof (``copies_aligned``) admits."""

from __future__ import annotations

import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.address import copies_aligned
from emmy.compiler.ir.stmt import parse_body

# --- copies_aligned --------------------------------------------------------------------------------------------------
# A copy of ``chunk`` elements starts wherever ``k`` is a multiple of ``chunk · per``; the whole flat address there must
# be a multiple of ``chunk``. Each case is one load over one buffer, with the copy running along ``k``.


@pytest.mark.parametrize(
    ("index", "shape", "chunk", "per", "aligned"),
    [
        ("x[m, k]", (64, 256), 8, 1, True),
        ("w[n, k + 1]", (256, 272), 8, 1, False),
        ("w[n, k + 8]", (256, 272), 8, 1, True),
        ("w[n, k]", (256, 260), 8, 1, False),
        ("x[m, k]", ("M", 256), 8, 1, True),
        ("x[m, k]", (64, "K"), 8, 1, False),
        ("x[0, m, k]", (1, 64, 256), 8, 1, True),
        ("x[m, k / 128, k % 128]", (64, 8, 128), 8, 1, True),
        ("x[m, k / 16, k % 16]", (64, 16, 17), 8, 1, False),
        ("x[(m * 256 + k) / 15, (m * 256 + k) % 15]", (1024, 15), 8, 1, True),
        ("w[n, (n * 4096 + k) / 2 % 2048]", (4096, 2048), 16, 2, True),
        ("w[n, (n * 4096 + k + 8) / 2 % 2048]", (4096, 2048), 16, 2, False),
        ("s[n, (n * 4096 + k) / 16 % 256]", (4096, 256), 16, 16, True),
    ],
    ids=[
        "dense",
        "2B-slice-offset",
        "16B-slice-offset",
        "520B-row-pitch",
        "symbolic-rows",
        "symbolic-row-pitch",
        "leading-unit-dim",
        "heads-flattened",
        "slice-reshaped-flat",
        "dense-reshape-across-rows",
        "packed-bits",
        "packed-bits-4B-offset",
        "block-scales",
    ],
)
def test_a_copy_is_aligned_only_when_its_whole_address_is(index, shape, chunk, per, aligned) -> None:
    (load,) = parse_body(f"bf16 a = load {index}")
    inputs = {load.input: Tensor(load.input, tuple(Dim(d) for d in shape), BF16)}
    assert copies_aligned(load, "k", chunk, inputs, per=per) is aligned


def test_an_unknown_buffer_is_not_proven() -> None:
    (load,) = parse_body("bf16 a = load x[m, k]")
    assert not copies_aligned(load, "k", 8, {})
