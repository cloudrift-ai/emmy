"""Tests for the structural LoopOp IR with SSA body (Assign/Accum/Write/Select)."""

from dataclasses import replace

import pytest
from frozendict import frozendict

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expression import Literal, Var, apply
from emmy.compiler.ir.loop import Assign, Axis, Loop, LoopOp, Select
from emmy.compiler.ir.stmt import normalize_body, parse_body


def _loop(text: str) -> LoopOp:
    """A Loop op over the body ``text`` states."""
    return LoopOp(body=parse_body(text))


# ---------------------------------------------------------------------------
# Axis / Port
# ---------------------------------------------------------------------------


def test_axis_construction():
    a = Axis("a0", 8)
    assert a.name == "a0"
    assert a.extent == 8


# ---------------------------------------------------------------------------
# Body-form Load / Accum (new IR: inputs + accumulators as statements)
# ---------------------------------------------------------------------------


def test_load_stmt_binding():
    """Load introduces an SSA name; LoopOp.loads collects them in pre-order."""
    k = _loop(
        """
        for a0 in 0..4:
            x_val = load src_0[a0]
            y = negative(x_val)
            out_0[a0] = y
        """
    )
    loads = k.body.loads
    # rename_ssa_sequential canonicalizes Load names to in0, in1, ...
    assert len(loads) == 1 and loads[0].name == "in0" and loads[0].input == "src_0"
    assert len(tuple(k.inputs)) == 1


def test_load_stmt_multiple_sources():
    """num_inputs is the count of distinct Load.source buf names."""
    k = _loop(
        """
        for a0 in 0..4:
            a = load src_0[a0]
            b = load src_2[a0]
            y = add(a, b)
            out_0[a0] = y
        """
    )
    # Two distinct source buf names → two inputs.
    assert tuple(k.inputs) == ("src_0", "src_2")


_SUM_OF_TWO = """
x = load X[]
y = load Y[]
z = add(x, y)
O[] = z
"""


def test_injective_buffer_rename_preserves_normal_form_cache() -> None:
    op = _loop(_SUM_OF_TWO)

    rename = {"X": "z_input", "Y": "a_input", "O": "renamed_output"}
    renamed = op.rename_buffers(rename)
    assert normalize_body(renamed.body) is renamed.body
    assert replace(renamed, outputs=renamed.outputs).body is renamed.body
    assert tuple(load.input for load in renamed.loads) == tuple(rename[load.input] for load in op.loads)
    assert renamed.identity_key() == op.identity_key()


def test_aliasing_buffer_rename_does_not_preserve_normal_form_cache() -> None:
    op = _loop(_SUM_OF_TWO)

    renamed = op.rename_buffers({"X": "input", "Y": "input"})
    assert normalize_body(renamed.body) is not renamed.body


def test_update_synthesizes_accum_decl():
    """An ``Accum`` in a reduce Loop implicitly declares its accumulator.
    ``LoopOp.accums`` synthesizes the declaration info (name, combine
    op, identity value) from the Accum. No explicit decl stmt is needed."""
    k = _loop(
        """
        for row in 0..4:
            for k in 0..8:
                x_val = load src_0[row, k]
                acc <- add(acc, x_val)
            out_0[row] = acc
        """
    )
    # rename_ssa_sequential canonicalizes Accum names to acc0, acc1, ...
    assert len(k.body.accums) == 1 and k.body.accums[0].name == "acc0"
    assert k.body.accums[0].op.name == "add"
    assert isinstance(k.body.accums[0].init, Literal) and k.body.accums[0].init.value == 0.0


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", apply(ElementwiseImpl("add"), "a", "b"))
    assert a.name == "out"
    assert a.value.operation.name == "add"
    assert [operand.name for operand in a.value.operands] == ["a", "b"]


# ---------------------------------------------------------------------------
# LoopOp construction
# ---------------------------------------------------------------------------


def test_kernel_pointwise():
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            in1_2 = load src_1[a0]
            z = add(in0_1, in1_2)
            out_0[a0] = z
        """
    )
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4  # 2 Loads + Assign + Write


def test_kernel_reduce():
    k = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                in0_1 = load src_0[a0, a1]
                s <- add(s, in0_1)
            out_0[a0] = s
        """
    )
    assert any(isinstance(lb.op, ElementwiseImpl) for lb in k.body.accums)


def test_kernel_matmul():
    k = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                in0_1 = load src_0[a0, a1]
                in1_2 = load src_1[a0, a1]
                multiply = multiply(in0_1, in1_2)
                dot <- add(dot, multiply)
            out_0[a0] = dot
        """
    )
    # 2 Loads + the product + Accum (over its name) + Write — accumulator info is carried on Accum.op now.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 5


def test_kernel_matmul_bias():
    k = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                in0_1 = load src_0[a0, a1]
                in1_2 = load src_1[a0, a1]
                multiply = multiply(in0_1, in1_2)
                dot <- add(dot, multiply)
            in2_3 = load src_2[a0]
            out = add(dot, in2_3)
            out_0[a0] = out
        """
    )
    # 3 Loads + the product + Accum (over its name) + Assign + Write (no decl stmt — Accum carries op).
    assert sum(1 for s in k if not isinstance(s, Loop)) == 7


def test_kernel_softmax_two_accumulators():
    # Under nested SSA scoping, per-segment SSA names (sub, ex) are scoped to
    # their own reduce Loop — the old flat fixture relied on laxness. Here we
    # just verify a kernel with two accumulators + two Updates builds fine.
    k = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                in0_1 = load src_0[a0, a1]
                mx <- maximum(mx, in0_1)
            for a1 in 0..8:
                in0_2 = load src_0[a0, a1]
                sm <- add(sm, in0_2)
            out_0[a0] = mx
            out_1[a0] = sm
        """
    )
    # rename_ssa_sequential renames Accums to acc0, acc1 in definition order.
    assert any(lb.name == "acc0" for lb in k.body.accums)
    assert any(lb.name == "acc1" for lb in k.body.accums)


def test_kernel_unary_chain():
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            negative = negative(in0_1)
            exp = exp(negative)
            out_0[a0] = exp
        """
    )
    # 1 Load + two Assigns (one operation each) + Write.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4


def test_kernel_scatter_output_via_select():
    """Select replaces Mux for coord-predicated dispatch on output."""
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            in1_2 = load src_1[a0]
            z = add(in0_1, in1_2)
            v = select:
                in0_1 when c1
                in1_2 when c2
            out_0[a0] = v
        """
    )
    assert any(isinstance(s, Select) for s in k)


# ---------------------------------------------------------------------------
# Validator — SSA / accumulator / v1 pins
# ---------------------------------------------------------------------------


def test_ssa_rejects_undefined_arg():
    with pytest.raises(ValueError, match="not defined"):
        _loop(
            """
            for a0 in 0..4:
                y = exp(z)
                out_0[a0] = y
            """
        )


def test_two_lexical_bindings_of_one_name_are_two_values():
    """A name bound twice in one scope (the second reading the first) is two values: the normal form renames them
    apart, one definition each."""
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            y = exp(in0_1)
            y = negative(y)
            out_0[a0] = y
        """
    )
    assert [str(s.value) for s in k.body[0].body if isinstance(s, Assign)] == ["exp(in0)", "-v0"]


def test_ssa_topo_sorts_forward_reference():
    """A body with a forward SSA reference is silently reordered by
    ``topological_sort`` in ``normalize_body`` so the def precedes its
    use. Validation only rejects truly undefined names / cycles."""
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            a = add(in0_1, b)
            b = exp(in0_1)
            out_0[a0] = a
        """
    )
    # After normalization the ``exp`` (originally "b") is defined ahead of the sum that reads it.
    values = [str(s.value) for s in k.body[0].body if isinstance(s, Assign)]
    assert values == ["exp(in0)", "in0 + v0"], values


def test_ssa_allows_input_name_reuse_in_multiple_args():
    k = _loop(
        """
        for a0 in 0..4:
            in0_1 = load src_0[a0]
            a = exp(in0_1)
            b = negative(in0_1)
            c = add(a, b)
            out_0[a0] = c
        """
    )
    # 1 Load + the exp, and the sum with the negation (one affine region) + 1 Write.
    assert sum(1 for s in k if not isinstance(s, Loop)) == 4


def test_update_op_conflict_rejected():
    """Multiple Updates to the same target must share the same op — mixing
    ``max`` and ``add`` on one accumulator is semantically ambiguous."""
    with pytest.raises(ValueError, match="conflicts with"):
        _loop(
            """
            for a0 in 0..4:
                in0_1 = load src_0[a0]
                acc <- maximum(acc, in0_1)
            for a0 in 0..4:
                in0_2 = load src_0[a0]
                acc <- add(acc, in0_2)
            out_0[] = acc
            """
        )


# ---------------------------------------------------------------------------
# Loop Statement — nested body form
# ---------------------------------------------------------------------------


def test_loop_stmt_basic():
    # Pointwise kernel with body wrapped in a free Loop.
    loop = _loop(
        """
        for a0 in 0..8:
            in0_1 = load src_0[a0]
            v = negative(in0_1)
            out_0[a0] = v
        """
    )
    assert isinstance(loop.body[0], Loop)
    assert loop.body[0].axis.name == "a0"


def test_loop_op_orders_free_axes_by_output_storage() -> None:
    op = _loop(
        """
        for d in 0..16:
            for h in 0..4:
                for m in 0..8:
                    value = load x[h, m, d]
                    out[h, m, d] = value
        """
    )

    assert op.writes[0].index == tuple(Var(axis.name) for axis in op.axes)


def test_loop_axis_sibling_same_name_allowed():
    """Sibling Loops with the same axis name are fine — that's softmax's two K-loops."""
    # Two reduce Loops over the same axis, sibling (not nested).
    _loop(
        """
        for k in 0..8:
            in0_1 = load src_0[a0, k]
            mx <- maximum(mx, in0_1)
        for k in 0..8:
            in1_2 = load src_1[a0, k]
            sm <- add(sm, in1_2)
        v = add(mx, sm)
        out_0[a0] = v
        """
    )


def test_loop_ssa_scoping():
    """Assign defined inside Loop body is invisible outside — caller Assign must fail.

    The inner Assign binds to the inner axis (via ``$0``'s index) so LICM
    can't hoist it out; the outer reference is a true scope violation.
    """
    with pytest.raises(ValueError, match="not defined"):
        _loop(
            """
            for a0_inner in 0..4:
                in0_1 = load src_0[a0_inner]
                v = negative(in0_1)
            out_0[a0] = v
            """
        )


# ---------------------------------------------------------------------------
# Phase 2: consumers handle nested bodies
# ---------------------------------------------------------------------------


def _flat_reduce_kernel() -> LoopOp:
    """Flat-body reduce kernel for parity comparison."""
    return _loop(
        """
        for a0 in 0..4:
            for k in 0..8:
                in0_1 = load src_0[a0, k]
                acc <- add(acc, in0_1)
            out_0[a0] = acc
        """
    )


def _nested_reduce_kernel() -> LoopOp:
    """Same kernel as _flat_reduce_kernel, but body is nested via explicit Loops."""
    return _loop(
        """
        for a0 in 0..4:
            for k in 0..8:
                x_k = load src_0[a0, k]
                acc <- add(acc, x_k)
            out_0[a0] = acc
        """
    )


def test_execute_loop_op_handles_nested_body():
    """LoopOp.forward should produce the same output for flat and nested forms."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)

    flat = _flat_reduce_kernel()
    nested = _nested_reduce_kernel()
    out_flat = flat.forward(x)
    out_nested = nested.forward(x)
    np.testing.assert_allclose(out_flat, out_nested, rtol=1e-5)


def test_structural_key_is_the_same_bare_and_through_a_loop_op() -> None:
    """Identity keys the executable normal form: hoisting a loop-invariant read at Loop op
    construction must not key the op apart from the bare body it was built from."""
    raw = parse_body(
        """
        for i in 0..8:
            s = load S[]
            x = load X[i]
            y = multiply(x, s)
            O[i] = y
        """
    )
    op = LoopOp(body=raw)
    assert op.body != raw, "the invariant read hoists"
    assert op.body.structural_key() == raw.structural_key()


def test_identity_binds_buffer_types_to_roles() -> None:
    """``identity_key(with_io=True)`` types the ROLE a buffer plays. Two kernels whose typed
    argument lists read the same in declaration order but assign the types to different roles
    key apart; declaring the same typed roles in another order keys the same."""
    from emmy.compiler.dtype import F16, F32  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    def body(product_left: str, product_right: str, subtrahend: str) -> str:
        return f"""
        for i in 0..8:
            p = load {product_left}[i]
            q = load {product_right}[i]
            r = load {subtrahend}[i]
            m = multiply(p, q)
            o = subtract(m, r)
            O[i] = o
        """

    def op(stmts: str, *declared: tuple[str, object]) -> LoopOp:
        inputs = frozendict({name: Tensor(name, (8,), dtype) for name, dtype in declared})
        return LoopOp(body=parse_body(stmts), inputs=inputs, outputs=frozendict({"O": Tensor("O", (8,), F32)}))

    halves_f16 = op(body("A", "B", "C"), ("A", F16), ("B", F16), ("C", F32))
    mixed_product = op(body("A", "C", "B"), ("A", F16), ("B", F16), ("C", F32))
    redeclared = op(body("A", "B", "C"), ("C", F32), ("B", F16), ("A", F16))
    assert halves_f16.identity_key() == mixed_product.identity_key(), "untyped, the bodies are one structure"
    assert halves_f16.identity_key(with_io=True) != mixed_product.identity_key(with_io=True)
    assert halves_f16.identity_key(with_io=True) == redeclared.identity_key(with_io=True)
