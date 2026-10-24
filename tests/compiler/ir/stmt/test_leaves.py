"""How the leaf statements render."""

from dataclasses import replace

from emmy.compiler.dtype import BF16
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.stmt.base import RenderCtx


def test_a_carried_load_without_a_vector_type_reassigns_its_scalars():
    """A bf16 chunk has no vector type, so the split blocking copy carries its staging registers across the chunk
    loop's back edge as scalars. Only the prime declares them: a second declaration in the same scope is what nvcc
    rejected on the vision tower's patch embedding."""
    prime = Load(names=("s0", "s1"), input="x", index=(Var("i"),), dtype=BF16)
    ctx = RenderCtx(indent=0)
    assert prime.render(ctx) == ["__nv_bfloat16 s0 = x[i];", "__nv_bfloat16 s1 = x[i + 1];"]
    assert replace(prime, carried="load").render(ctx) == ["s0 = x[i];", "s1 = x[i + 1];"]
    assert replace(prime, carried="unpack").render(ctx) == [], "the registers already hold the chunk"
