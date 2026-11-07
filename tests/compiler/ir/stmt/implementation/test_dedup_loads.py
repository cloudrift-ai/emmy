"""Normalization shares equal loads and the cones they feed: what the whole driver leaves of a body with duplicates."""

from emmy.compiler.ir.stmt import Affine, Assign, Body, Cond, Load, Loop, Write, normalize_body, parse_body


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
    actual = execute_loop_op_cpp(LoopOp(body=normalize_body(body)), {"x": values}, {"out": (4,)})

    np.testing.assert_allclose(actual, values / values.sum(), rtol=1e-6)


def test_normalize_dedups_loads_and_rewires_gather_indices() -> None:
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


def test_normalize_dedups_identical_accumulations() -> None:
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
