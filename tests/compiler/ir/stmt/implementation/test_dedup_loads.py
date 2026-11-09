"""Tests for Load deduplication during body normalization."""

from emmy.compiler.ir.expression import BinaryExpr
from emmy.compiler.ir.stmt import Affine, Assign, Body, Cond, Let, Load, Loop, Select, Write, normalize_body, parse_body
from emmy.compiler.ir.stmt.normalize import place_values
from tests.compiler.ir.stmt import listing


def test_dedup_loads_preserves_loads_under_a_rebound_coordinate() -> None:
    """A normalization sum must read every channel when its index shadows the output index."""
    import numpy as np

    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp

    body = parse_body(
        """
        for k in 0..4:
            outer = load x[k]
            for k in 0..4:
                inner = load x[k]
                total <- add(total, inner) over k
            value = divide(outer, total)
            out[k] = value
        """
    )
    values = np.array([1, 2, 4, 8], dtype=np.float32)
    actual = execute_loop_op_cpp(LoopOp(body=place_values(body)), {"x": values}, {"out": (4,)})

    np.testing.assert_allclose(actual, values / values.sum(), rtol=1e-6)


def test_normalize_body_dedups_loads_and_rewires_gather_indices() -> None:
    body = parse_body(
        """
        for a in 0..4:
            idx0 = load indices[a]
            idx1 = load indices[a]
            x0 = load values[idx0]
            x1 = load values[idx1]
            sum = add(x0, x1)
            out[a] = sum
        """
    )

    (loop,) = normalize_body(body)

    assert [str(stmt) for stmt in loop.body if isinstance(stmt, Load)] == ["in0 = load indices[a0]", "in1 = load values[in0]"]
    assert str(loop.body[-2]) == "v0 = 2*in1"


def test_normalize_hoisted_shadowed_loads_keep_their_values() -> None:
    """Motion preserves lexical bindings before CSE merges equal loads."""
    from emmy.compiler.ir.loop import LoopOp

    for inner_input in ("x", "y"):
        op = LoopOp(
            body=parse_body(
                f"""
                value = load x[0]
                for i in 0..4:
                    value = load {inner_input}[0]
                    inner[i] = value
                outer[0] = value
                """
            )
        )
        loads = {stmt.name: stmt.input for stmt in op.body if isinstance(stmt, Load)}
        writes = {stmt.output: stmt.value for stmt in op.body.iter() if isinstance(stmt, Write)}
        assert loads[writes["outer"]] == "x"
        assert loads[writes["inner"]] == inner_input
        assert len(loads) == (1 if inner_input == "x" else 2)
        assert normalize_body(op.body) == op.body


def test_cse_does_not_use_expression_printing(monkeypatch) -> None:
    body = parse_body("x0 = load x[i + 1]\nx1 = load x[i + 2]")
    monkeypatch.setattr(BinaryExpr, "__str__", lambda self: "index")
    assert place_values(body) == body


def test_cse_shares_selections_and_their_downstream_cones() -> None:
    body = parse_body(
        """
        let `zero` = 0
        let other_zero = 0
        left = select:
            x when p
            `zero` when 1
        right = select:
            x when p
            other_zero when 1
        a = exp(left)
        b = exp(right)
        out[0] = b
        """
    )
    out = place_values(body)
    assert [type(s) for s in out] == [Let, Select, Assign, Write]
    assert out[-1].value == "a"


def test_cse_selection_predicates_observe_rebound_coordinates() -> None:
    body = parse_body(
        """
        a = select:
            x when k
            y when 1
        for k in 0..4:
            b = select:
                x when k
                y when 1
            out[k] = b
        """
    )
    assert place_values(body) == body


def test_cse_does_not_drop_repeated_accumulator_updates() -> None:
    body = parse_body(
        """
        for k in 0..8:
            sum <- add(sum, x) over k
            sum <- add(sum, x) over k
        """
    )
    assert place_values(body) == body


def test_cse_does_not_alias_distinct_unseeded_accumulators() -> None:
    body = parse_body(
        """
        for k in 0..8 seed=false:
            left <- add(left, x) over k
            right <- add(right, x) over k
        """
    )
    assert place_values(body) == body


def test_cse_partial_state_changes_invalidate_dependent_values() -> None:
    body = parse_body(
        """
        for k in 0..8:
            before = exp(sum)
            sum <- add(sum, x) over k
            after = exp(sum)
            out[k] = after
        """
    )
    assert place_values(body) == body


def test_cse_nested_state_changes_invalidate_enclosing_values() -> None:
    body = parse_body(
        """
        before = exp(sum)
        for k in 0..4 seed=false:
            sum <- add(sum, x) over k
        after = exp(sum)
        out[0] = after
        """
    )
    assert place_values(body) == body


def test_cse_does_not_reuse_a_staged_load_assignment() -> None:
    body = parse_body("value = load x[0]\nvalue = load x[0] carried=load")
    assert place_values(body) == body


def test_cse_removes_repeated_copies_of_one_binding() -> None:
    from emmy.compiler.ir.loop import LoopOp

    body = parse_body(
        """
        value = load x[0]
        first[0] = value
        value = load x[0]
        second[0] = value
        """
    )
    assert str(place_values(body)) == listing(
        """
        value = load x[0]
        first[0] = value
        second[0] = value
        """
    )
    assert len([stmt for stmt in LoopOp(body=body).body if isinstance(stmt, Load)]) == 1


def test_cse_never_aliases_a_value_to_an_overwritten_representative() -> None:
    body = parse_body(
        """
        value = load x[0]
        copy = load x[0]
        value = load y[0]
        out[0] = copy
        """
    )
    assert place_values(body) == body
    updates = parse_body("value = exp(value)\nvalue = exp(value)")
    assert place_values(updates) == updates


def test_cse_does_not_merge_reductions_with_distinct_explicit_seeds() -> None:
    body = parse_body(
        """
        init f32 left = 0
        init f32 right = 1
        for k in 0..8:
            left <- add(left, x)
            right <- add(right, x)
        """
    )
    assert place_values(body) == body


def test_normalization_closes_simplification_cse_and_invariant_motion() -> None:
    """An equal gather index exposes a constant predicate, then an invariant shared cone."""
    body = parse_body(
        """
        x = load x[0]
        for i in 0..4:
            a = load indices[i]
            b = load indices[i]
            s = select:
                x when a - b < 1
                a when 1
            v = exp(s)
            out[i] = v
        """
    )
    normalized = normalize_body(body)
    assert any(isinstance(s, Assign) and s.op.name == "exp" for s in normalized)
    assert normalize_body(Body(tuple(normalized))) == normalized


def test_cse_factors_common_branch_cones_without_speculation() -> None:
    def branch(prefix, extra):
        return "\n".join(
            f"    {line}"
            for line in (
                f"{prefix}i = load indices[0]",
                f"{prefix}x = load x[{prefix}i]",
                f"{prefix}v = exp({prefix}x)",
                f"{prefix}only = load {extra}[0]",
                f"{prefix}out = add({prefix}v, {prefix}only)",
                f"out[0] = {prefix}out",
            )
        )

    body = parse_body(f"if predicate:\n{branch('a', 'left')}\nelse:\n{branch('b', 'right')}")
    out = normalize_body(body)
    assert [type(stmt) for stmt in out] == [Load, Load, Assign, Cond]
    assert out[2].op.name == "exp"
    for inner in out[-1].nested():
        assert [type(stmt) for stmt in inner] == [Load, Affine, Write]
        assert out[2].name in inner[1].args
    assert normalize_body(Body(tuple(out))) == out


def test_common_branch_load_cannot_cross_a_write_on_either_path() -> None:
    read = "    value = load x[0]\n    out[0] = value\n"
    write = "    x[0] = replacement\n"
    for left, right in ((write + read, read), (read, write + read)):
        body = parse_body(f"if predicate:\n{left}else:\n{right}")
        assert place_values(body) == body


def test_normalized_quotient_addresses_fuse_and_share_the_whole_reduction() -> None:
    def reduction(axis, name, nested):
        index = f"{axis} / 128 / 2" if nested else f"{axis} / 256"
        return "\n".join(
            (
                f"for {axis} in 0..1024:",
                f"    {name}x = load x[{index}]",
                f"    {name}v = exp({name}x)",
                f"    {name} <- add({name}, {name}v) over {axis}",
            )
        )

    body = parse_body("\n".join((reduction("i", "left", True), reduction("j", "right", False), "out[0] = right", "other[0] = left")))
    out = normalize_body(body)
    assert len(out.loads) == len(out.accums) == 1
    assert len([s for s in out.iter() if isinstance(s, Loop)]) == 1
    assert out[-1].value == out[-2].value
    assert normalize_body(Body(tuple(out))) == out


def test_dedup_loads_closes_commutative_chains_after_aliasing() -> None:
    """An alias that reverses sorted operands must not hide the rest of a duplicate chain."""
    body = parse_body(
        """
        z = load X[0]
        a = load X[0]
        m = load Y[0]
        z1 = multiply(m, z)
        a1 = multiply(a, m)
        z2 = add(m, z1)
        a2 = add(a1, m)
        left = subtract(m, z1)
        right = subtract(a1, m)
        O[0] = a2
        L[0] = left
        R[0] = right
        """
    )

    out = place_values(body)

    assert [stmt.name for stmt in out if isinstance(stmt, Assign)] == ["z1", "z2", "left", "right"]
    assert str(out[-4]) == "right = subtract(z1, m)"
    assert str(out[-3]) == "O[0] = z2"
    assert place_values(out) == out


def test_dedup_loads_still_rewires_an_inner_use_of_the_dropped_name() -> None:
    """The mirror case: the loop only *reads* the dropped name, so the alias must reach inside."""
    body = parse_body(
        """
        in0 = load const[0]
        in1 = load const[0]
        for a in 0..4:
            v = add(in0, in1)
        """
    )

    out = place_values(body)

    assert [s.name for s in out if isinstance(s, Load)] == ["in0"]
    assert [str(stmt) for stmt in out.iter() if isinstance(stmt, Assign)] == ["v = add(in0, in0)"]


def test_cse_reuses_a_dominating_read_until_the_write_on_that_branch() -> None:
    body = parse_body(
        """
        old = load x[0]
        if p:
            before = load x[0]
            out[0] = before
            x[0] = replacement
            after = load x[0]
            new[0] = after
        """
    )
    out = place_values(body)
    assert str(out[1].body[0]) == "out[0] = old"
    assert [load.name for load in out.loads] == ["old", "after"]


def test_dedup_loads_rewires_every_vector_lane() -> None:
    """A duplicate vector load aliases each lane to the corresponding kept lane."""
    body = parse_body(
        """
        x0, x1 = load X[0]
        y0, y1 = load X[0]
        sum = add(y0, y1)
        """
    )

    out = place_values(body)

    assert str(out) == listing(
        """
        x0, x1 = load X[0]
        sum = add(x0, x1)
        """
    )


def test_dedup_loads_invalidates_a_read_after_writing_its_buffer() -> None:
    body = parse_body(
        """
        old = load B[0]
        replacement = load R[0]
        B[0] = replacement
        new = load B[0]
        O[0] = new
        """
    )

    out = place_values(body)

    assert [stmt.name for stmt in out if isinstance(stmt, Load) and stmt.input == "B"] == ["old", "new"]


def test_dedup_loads_does_not_reuse_a_read_across_a_loop_that_writes_its_buffer() -> None:
    body = parse_body(
        """
        before = load B[0]
        for a in 0..4:
            current = load B[0]
            B[0] = current
        """
    )

    out = place_values(body)

    assert str(out[1].body[0]) == "current = load B[0]"


def test_normalize_body_dedups_identical_accumulations() -> None:
    """Two consumers of one fused value carry two copies of its accumulation in one reduce loop
    after the splice; the second is the first under another name."""
    body = parse_body(
        """
        for m in 0..4:
            for k in 0..8:
                x0 = load x[m, k]
                w0 = load w[k]
                p0 = multiply(x0, w0)
                acc0 <- add(acc0, p0) over k
                x1 = load x[m, k]
                w1 = load w[k]
                p1 = multiply(x1, w1)
                acc1 <- add(acc1, p1) over k
            gate = exp(acc0)
            up = exp(acc1)
            out = multiply(gate, up)
            out[m] = out
        """
    )
    (loop,) = normalize_body(body)
    (inner,) = [stmt for stmt in loop.body if isinstance(stmt, Loop)]
    assert [type(stmt).__name__ for stmt in inner.body] == ["Load", "Load", "Assign", "Accum"]
    assert [stmt for stmt in loop.body if isinstance(stmt, Assign)][-1].args == ("v1", "v1")
