"""How the leaf statements render."""

from dataclasses import replace

import pytest

from emmy.compiler.dtype import BF16, I16
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Load, Write
from emmy.compiler.ir.stmt.base import RenderCtx


def test_a_carried_load_without_a_vector_type_reassigns_its_scalars():
    """A run the target has no vector type for carries the split blocking copy's staging registers across the chunk
    loop's back edge as scalars. Only the prime declares them: a second declaration in the same scope is what nvcc
    rejected on the vision tower's patch embedding, while bf16 still had no vector type."""
    prime = Load(names=("s0", "s1"), input="x", index=(Var("i"),), dtype=I16)
    ctx = RenderCtx(indent=0)
    assert prime.render(ctx) == ["short s0 = x[i];", "short s1 = x[i + 1];"]
    assert replace(prime, carried="load").render(ctx) == ["s0 = x[i];", "s1 = x[i + 1];"]
    assert replace(prime, carried="unpack").render(ctx) == [], "the registers already hold the chunk"


@pytest.mark.parametrize(("width", "vector"), [(2, "__nv_bfloat162"), (4, "uint2"), (8, "uint4")])
def test_a_bf16_run_is_one_access(width, vector):
    """A bf16 run takes f16's vector types: the pair type at 2, ``uint2`` / ``uint4`` punned to 4 and 8 elements. The
    load unpacks through an element pointer and the store packs through an element array; a staged copy's carried
    halves re-load the vector alone and unpack it where the chunk is consumed."""
    names = tuple(f"s{k}" for k in range(width))
    ctx = RenderCtx(indent=0, buffer_dtypes={"y": "bf16"})
    prime = Load(names=names, input="x", index=(Var("i"),), dtype=BF16)
    assert prime.render(ctx) == [
        f"{vector} _v_s0 = *reinterpret_cast<const {vector}*>(&x[i]);",
        "const __nv_bfloat16* _v_s0_h = reinterpret_cast<const __nv_bfloat16*>(&_v_s0);",
        *(f"__nv_bfloat16 s{k} = _v_s0_h[{k}];" for k in range(width)),
    ]
    assert replace(prime, carried="load").render(ctx) == [f"_v_s0 = __ldg(reinterpret_cast<const {vector}*>(&x[i]));"]
    assert replace(prime, carried="unpack").render(ctx) == [f"s{k} = _v_s0_h[{k}];" for k in range(width)]
    pack, store = Write(output="y", index=(Var("i"),), values=names, value_dtype=BF16).render(ctx)
    temp = pack.split()[1].split("[")[0]
    assert pack == f"__nv_bfloat16 {temp}[{width}] = {{ {', '.join(names)} }};"
    assert store == f"*reinterpret_cast<{vector}*>(&y[i]) = *reinterpret_cast<const {vector}*>({temp});"
