"""Tests for the hoist ``place_values`` (``stmt/normalize.py``) performs.

Builds bodies by hand and asserts on the post-hoist structure so failures
point at the pass itself rather than upstream lowering.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, StridedLoop, Write, normalize_body, parse_body
from emmy.compiler.ir.stmt.normalize import place_values


def test_hoists_invariant_load_above_loop() -> None:
    body = parse_body(
        """
        for a in 0..4:
            x = load X[]
            o[a] = x
        """
    )

    out = place_values(body)

    assert len(out) == 2
    assert isinstance(out[0], Load) and out[0].name == "x"
    assert isinstance(out[1], Loop) and out[1].axis.name == "a"
    inner = tuple(out[1].body)
    assert len(inner) == 1 and isinstance(inner[0], Write)


def test_hoists_inner_loop_and_consumer_together() -> None:
    """The motivating case: an inner reduce-Loop is invariant in the outer
    axis, and a downstream Assign consumes its Accum result. Both must move
    above the outer Loop together — hoisting only the Assign would leave it
    referencing an Accum still defined inside the outer Loop body."""
    body = parse_body(
        """
        for a in 0..4:
            for b in 0..8:
                xb = load X[b]
                acc <- add(acc, xb)
            v = exp(acc)
            out[a] = v
        """
    )

    out = place_values(body)

    assert len(out) == 3, f"expected [Loop(b), Assign v, Loop(a)], got {[type(s).__name__ for s in out]}"
    assert isinstance(out[0], Loop) and out[0].axis.name == "b"
    assert isinstance(out[1], Assign) and out[1].name == "v" and out[1].args == ("acc",)
    assert isinstance(out[2], Loop) and out[2].axis.name == "a"
    inner_a = tuple(out[2].body)
    assert len(inner_a) == 1 and isinstance(inner_a[0], Write)
    inner_b = tuple(out[0].body)
    assert any(isinstance(s, Accum) and s.name == "acc" for s in inner_b)


def test_keeps_inner_loop_when_it_references_outer_axis() -> None:
    body = parse_body(
        """
        for a in 0..4:
            for b in 0..8:
                xab = load X[a, b]
                acc <- add(acc, xab)
            out[a] = acc
        """
    )

    out = place_values(body)

    assert len(out) == 1
    assert isinstance(out[0], Loop) and out[0].axis.name == "a"
    inner = tuple(out[0].body)
    assert any(isinstance(s, Loop) and s.axis.name == "b" for s in inner)


def test_hoists_invariant_load_above_strided_loop() -> None:
    """A ``StridedLoop`` is the same kind of sequential axis-binder as ``Loop``;
    hoisting partitions its body the same way."""
    body = parse_body(
        """
        for a in 0..4 step 1:
            x = load X[]
            o[a] = x
        """
    )

    out = place_values(body)

    assert len(out) == 2
    assert isinstance(out[0], Load) and out[0].name == "x"
    assert isinstance(out[1], StridedLoop) and out[1].axis.name == "a"
    inner = tuple(out[1].body)
    assert len(inner) == 1 and isinstance(inner[0], Write)


def test_hoists_inner_strided_loop_and_consumer_together() -> None:
    """Same shape as the Loop-block test, but the inner reduce is a ``StridedLoop``."""
    body = parse_body(
        """
        for a in 0..4:
            for b in 0..8 step 1:
                xb = load X[b]
                acc <- add(acc, xb)
            v = exp(acc)
            out[a] = v
        """
    )

    out = place_values(body)

    assert len(out) == 3, f"expected [StridedLoop(b), Assign v, Loop(a)], got {[type(s).__name__ for s in out]}"
    assert isinstance(out[0], StridedLoop) and out[0].axis.name == "b"
    assert isinstance(out[1], Assign) and out[1].name == "v" and out[1].args == ("acc",)
    assert isinstance(out[2], Loop) and out[2].axis.name == "a"
    inner_a = tuple(out[2].body)
    assert len(inner_a) == 1 and isinstance(inner_a[0], Write)


def test_does_not_hoist_block_containing_write() -> None:
    """Inner Loop has no axis dependency on the outer axis but contains a
    Write — hoisting would change observable side-effect count, so it stays."""
    body = parse_body(
        """
        for a in 0..4:
            for b in 0..8:
                xb = load X[b]
                scratch[b] = xb
            out[a] = xb
        """
    )

    out = place_values(body)

    assert len(out) == 1 and isinstance(out[0], Loop) and out[0].axis.name == "a"
    inner = tuple(out[0].body)
    assert any(isinstance(s, Loop) and s.axis.name == "b" for s in inner)


def _nested(depth: int, leaf: str) -> str:
    """``leaf`` under ``depth`` loops ``a0`` … over two."""
    return "\n".join([*(f"{'    ' * level}for a{level} in 0..2:" for level in range(depth)), f"{'    ' * depth}{leaf}"])


def test_nested_side_effect_summary_is_cached_bottom_up() -> None:
    """One root query populates the immutable summary on every nested loop."""
    (stmt,) = parse_body(_nested(32, "out[] = value"))
    loops = []
    loop = stmt
    while isinstance(loop, Loop):
        loops.append(loop)
        (loop,) = loop.body

    assert stmt.has_side_effects
    assert all(loop.__dict__.get("has_side_effects") is True for loop in loops)


def test_nested_axis_dependency_summary_does_not_rescan_each_loop(monkeypatch) -> None:
    """LICM does not rediscover one leaf through every enclosing loop."""
    calls = 0
    original = Load.deps

    def counted(load):
        nonlocal calls
        calls += 1
        return original(load)

    monkeypatch.setattr(Load, "deps", counted)
    depth = 32
    body = parse_body(_nested(depth, f"value = load x[{', '.join(f'a{level}' for level in range(depth))}]"))

    place_values(body)

    assert calls <= 2 * (depth + 1), "linear in the statements, not in statements times depth"


def test_normalization_does_not_build_the_full_ssa_dependency_closure() -> None:
    body = parse_body(
        """
        for a in 0..4:
            x = load X[a]
            scale = load scale[]
            value = divide(x, scale)
            out[a] = value
        """
    )

    normalized = normalize_body(body)
    assert [s.op.name for s in normalized.iter() if isinstance(s, Assign)] == ["divide"]

    assert "deps_closure" not in body.__dict__
    assert "deps_closure" not in normalized.__dict__


def test_hoist_keeps_a_read_of_a_buffer_the_loop_writes() -> None:
    """Axis-invariance alone does not earn a hoist: a read behind a write to its own buffer, or
    behind an ordered execution protocol, follows the scope's ordering constraints and stays."""
    import emmy.compiler.ir.kernel.ir  # noqa: F401  # declares ``Sync`` for the reader

    body = parse_body(
        """
        for i in 0..4:
            O[0] = one
            back = load O[0]
            Sync
            staged = load S[0]
            P[i] = back
            Q[i] = staged
        """
    )
    (loop,) = place_values(body)
    assert [type(stmt).__name__ for stmt in loop.body] == ["Write", "Load", "Sync", "Load", "Write", "Write"]


def test_escaped_reduce_loop_keeps_its_state_at_the_scope_it_moved_to() -> None:
    """An invariant reduce loop two scopes down moves to the top. The loops it left no longer hold its state, so
    they must not settle it at their own scope: the read of ``accc`` inside ``b`` would otherwise pin the ``a``
    loop below its own scope and the read of ``acca`` after it would index a scope that does not exist.
    The shape is a GDN decoder layer's fused normalization (Qwen3.5-9B, layer 0)."""
    body = parse_body(
        """
        z = load Z[]
        for a in 0..4:
            for b in 0..4:
                for c in 0..4:
                    accc <- add(accc, z)
                x = load X[a, b]
                v1 = add(accc, x)
                accb <- add(accb, v1)
            acca <- add(acca, accb)
        v3 = exp(acca)
        Y[] = v3
        """
    )

    out = place_values(body)

    kinds = [(type(s).__name__, getattr(s, "axis", None) and s.axis.name) for s in out]
    assert kinds == [("Load", None), ("Loop", "c"), ("Loop", "a"), ("Assign", None), ("Write", None)], kinds
    inner_b = tuple(tuple(out[2].body)[0].body)
    assert any(isinstance(s, Assign) and s.args == ("accc", "x") for s in inner_b)


def test_folded_reduction_is_read_at_its_representatives_scope() -> None:
    """Two seeded reductions of one value in one loop are one reduction: the second's name aliases the first.
    A read of that name after the loop sees the first's scope, not the loop body the dropped update sat in."""
    body = parse_body(
        """
        for a in 0..4:
            for b in 0..4:
                x = load X[a, b]
                acc1 <- add(acc1, x)
                y = load X[a, b]
                acc2 <- add(acc2, y)
            v = add(acc1, acc2)
            Y[a] = v
        """
    )

    out = place_values(body)

    assert len(out) == 1 and isinstance(out[0], Loop) and out[0].axis.name == "a"
    inner_a = tuple(out[0].body)
    assert [type(s).__name__ for s in inner_a] == ["Loop", "Assign", "Write"]
    assert [s.name for s in inner_a[0].body if isinstance(s, Accum)] == ["acc1"]
    assert inner_a[1].args == ("acc1", "acc1")
