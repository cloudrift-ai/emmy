"""How the leaf statements render."""

import pytest

from emmy.compiler.ir.render import RenderCtx, render_stmt
from emmy.compiler.ir.stmt import Stmt, parse_body


def _stmt(text: str) -> Stmt:
    (stmt,) = parse_body(text)
    return stmt


def test_a_carried_load_without_a_vector_type_reassigns_its_scalars():
    """A run the target has no vector type for carries the split blocking copy's staging registers across the chunk
    loop's back edge as scalars. Only the prime declares them: a second declaration in the same scope is what nvcc
    rejected on the vision tower's patch embedding, while bf16 still had no vector type."""
    prime = "i16 s0, s1 = load x[i]"
    ctx = RenderCtx(indent=0)
    assert render_stmt(_stmt(prime), ctx) == ["short s0 = x[i];", "short s1 = x[i + 1];"]
    assert render_stmt(_stmt(f"{prime} carried=load"), ctx) == ["s0 = x[i];", "s1 = x[i + 1];"]
    assert render_stmt(_stmt(f"{prime} carried=unpack"), ctx) == [], "the registers already hold the chunk"


@pytest.mark.parametrize(("width", "vector"), [(2, "__nv_bfloat162"), (4, "uint2"), (8, "uint4")])
def test_a_bf16_run_is_one_access(width, vector):
    """A bf16 run takes f16's vector types: the pair type at 2, ``uint2`` / ``uint4`` punned to 4 and 8 elements. The
    load unpacks through an element pointer and the store packs through an element array; a staged copy's carried
    halves re-load the vector alone and unpack it where the chunk is consumed."""
    names = tuple(f"s{k}" for k in range(width))
    ctx = RenderCtx(indent=0, buffer_dtypes={"y": "bf16"})
    prime = f"bf16 {', '.join(names)} = load x[i]"
    assert render_stmt(_stmt(prime), ctx) == [
        f"{vector} _v_s0 = *reinterpret_cast<const {vector}*>(&x[i]);",
        "const __nv_bfloat16* _v_s0_h = reinterpret_cast<const __nv_bfloat16*>(&_v_s0);",
        *(f"__nv_bfloat16 s{k} = _v_s0_h[{k}];" for k in range(width)),
    ]
    assert render_stmt(_stmt(f"{prime} carried=load"), ctx) == [f"_v_s0 = __ldg(reinterpret_cast<const {vector}*>(&x[i]));"]
    assert render_stmt(_stmt(f"{prime} carried=unpack"), ctx) == [f"s{k} = _v_s0_h[{k}];" for k in range(width)]
    pack, store = render_stmt(_stmt(f"bf16 y[i] = ({', '.join(names)})"), ctx)
    temp = pack.split()[1].split("[")[0]
    assert pack == f"__nv_bfloat16 {temp}[{width}] = {{ {', '.join(names)} }};"
    assert store == f"*reinterpret_cast<{vector}*>(&y[i]) = *reinterpret_cast<const {vector}*>({temp});"
