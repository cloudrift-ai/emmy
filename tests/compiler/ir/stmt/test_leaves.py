"""How the leaf statements render."""

import pytest

from emmy.compiler.ir.render import RenderCtx, render_stmt
from emmy.compiler.ir.stmt import Statement, parse_body


def _stmt(text: str) -> Statement:
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


@pytest.mark.parametrize(("width", "vector"), [(2, "__nv_bfloat162"), (4, "uint2"), (8, "uint4")])
def test_a_bf16_run_is_one_access(width, vector):
    """A bf16 run takes f16's vector types: the pair type at 2, ``uint2`` / ``uint4`` punned to 4 and 8 elements. The
    load unpacks through an element pointer and the store packs through an element array."""
    names = tuple(f"s{k}" for k in range(width))
    ctx = RenderCtx(indent=0, buffer_dtypes={"y": "bf16"})
    assert render_stmt(_stmt(f"bf16 {', '.join(names)} = load x[i]"), ctx) == [
        f"{vector} _v_s0 = *reinterpret_cast<const {vector}*>(&x[i]);",
        "const __nv_bfloat16* _v_s0_h = reinterpret_cast<const __nv_bfloat16*>(&_v_s0);",
        *(f"__nv_bfloat16 s{k} = _v_s0_h[{k}];" for k in range(width)),
    ]
    pack, store = render_stmt(_stmt(f"bf16 y[i] = ({', '.join(names)})"), ctx)
    temp = pack.split()[1].split("[")[0]
    assert pack == f"__nv_bfloat16 {temp}[{width}] = {{ {', '.join(names)} }};"
    assert store == f"*reinterpret_cast<{vector}*>(&y[i]) = *reinterpret_cast<const {vector}*>({temp});"


@pytest.mark.parametrize(("width", "vector"), [(2, "__nv_bfloat162"), (4, "uint2"), (8, "uint4")])
def test_a_carried_run_is_read_from_its_vector(width, vector):
    """The names of a staged copy carried across a back edge are the lanes of the vector their first load declares:
    the carried load re-assigns the vector alone, and a store of the names reads its lanes where it stands."""
    names = tuple(f"s{k}" for k in range(width))
    ctx = RenderCtx(indent=0, buffer_dtypes={"y": "bf16"}, carried_names=frozenset(names))
    prime = f"bf16 {', '.join(names)} = load x[i]"
    assert render_stmt(_stmt(prime), ctx) == [
        f"{vector} _v_s0 = *reinterpret_cast<const {vector}*>(&x[i]);",
        "const __nv_bfloat16* _v_s0_h = reinterpret_cast<const __nv_bfloat16*>(&_v_s0);",
    ]
    assert render_stmt(_stmt(f"{prime} carried=load"), ctx) == [f"_v_s0 = __ldg(reinterpret_cast<const {vector}*>(&x[i]));"]
    pack, _ = render_stmt(_stmt(f"bf16 y[i] = ({', '.join(names)})"), ctx)
    assert pack.endswith(f"= {{ {', '.join(f'_v_s0_h[{k}]' for k in range(width))} }};")
