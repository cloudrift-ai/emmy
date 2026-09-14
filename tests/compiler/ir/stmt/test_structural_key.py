"""Tests for ``Body.structural_key()`` and the two new normalization
passes (``sort_commutative_args``, ``canonicalize_buffer_names``).

The structural key is the canonical text rendering used for dedup
queries — two bodies that differ only by SSA / axis names, commutative
arg order, or external-buffer names must produce the same key.
"""

from __future__ import annotations

from itertools import permutations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import (
    canonicalize_buffer_names,
    normalize_body,
    sort_commutative_args,
)

# ---------------------------------------------------------------------------
# sort_commutative_args
# ---------------------------------------------------------------------------


def test_sort_commutative_args_orders_add() -> None:
    body = Body((Assign(name="v0", op="add", args=("y", "x")),))
    out = sort_commutative_args(body)
    assert out[0].args == ("x", "y")


def test_sort_commutative_args_leaves_subtract_alone() -> None:
    body = Body((Assign(name="v0", op="subtract", args=("y", "x")),))
    out = sort_commutative_args(body)
    assert out[0].args == ("y", "x")


def test_sort_commutative_args_recurses_into_loop() -> None:
    a = Axis("a", 4)
    body = Body((Loop(axis=a, body=(Assign(name="v0", op="multiply", args=("y", "x")),)),))
    out = sort_commutative_args(body)
    assert out[0].body[0].args == ("x", "y")


def test_sort_commutative_args_idempotent() -> None:
    body = Body((Assign(name="v0", op="add", args=("y", "x")),))
    once = sort_commutative_args(body)
    twice = sort_commutative_args(once)
    assert tuple(once) == tuple(twice)


# ---------------------------------------------------------------------------
# canonicalize_buffer_names
# ---------------------------------------------------------------------------


def test_canonicalize_buffer_names_renames_loads_and_writes() -> None:
    a = Axis("a", 4)
    body = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="x", input="my_input", index=(Var("a"),)),
                    Write(output="my_output", index=(Var("a"),), value="x"),
                ),
            ),
        )
    )
    out = canonicalize_buffer_names(body)
    inner = tuple(out[0].body)
    assert isinstance(inner[0], Load) and inner[0].input == "b0"
    assert isinstance(inner[1], Write) and inner[1].output == "b1"


def test_canonicalize_buffer_names_first_seen_wins() -> None:
    """Encounter order via ``Body.iter`` — the first reference to each
    buffer name fixes its canonical id."""
    body = Body(
        (
            Load(name="x", input="Z", index=(Var("a"),)),
            Load(name="y", input="A", index=(Var("a"),)),
            Load(name="z", input="Z", index=(Var("a"),)),  # reuse Z
        )
    )
    out = canonicalize_buffer_names(body)
    assert out[0].input == "b0"  # Z first
    assert out[1].input == "b1"  # A second
    assert out[2].input == "b0"  # Z again


def test_canonicalize_buffer_names_already_canonical_returns_same() -> None:
    body = Body((Load(name="x", input="b0", index=()),))
    out = canonicalize_buffer_names(body)
    assert out is body


# ---------------------------------------------------------------------------
# Body.structural_key()
# ---------------------------------------------------------------------------


def _matmul_body(input_x: str, input_y: str, output: str) -> Body:
    """Tiny multiply-and-write body parameterized by buffer names."""
    a = Axis("a", 4)
    return Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="x", input=input_x, index=(Var("a"),)),
                    Load(name="y", input=input_y, index=(Var("a"),)),
                    Assign(name="z", op="multiply", args=("x", "y")),
                    Write(output=output, index=(Var("a"),), value="z"),
                ),
            ),
        )
    )


def test_structural_key_equal_for_renamed_buffers() -> None:
    a = _matmul_body("X", "Y", "O")
    b = _matmul_body("foo", "bar", "baz")
    assert a.structural_key() == b.structural_key()


def test_structural_key_equal_for_swapped_commutative_args() -> None:
    a = Axis("a", 4)
    body_xy = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="lx", input="X", index=(Var("a"),)),
                    Load(name="ly", input="Y", index=(Var("a"),)),
                    Assign(name="z", op="add", args=("lx", "ly")),
                    Write(output="O", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    body_yx = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="ly", input="Y", index=(Var("a"),)),
                    Load(name="lx", input="X", index=(Var("a"),)),
                    Assign(name="z", op="add", args=("ly", "lx")),
                    Write(output="O", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    assert body_xy.structural_key() == body_yx.structural_key()


def _pointwise_body(*, inputs: tuple[tuple[str, str, str], ...], op: str, args: tuple[str, ...]) -> Body:
    """One pointwise kernel with an authored load order over row/column arguments."""
    row, column = Axis("row", 4), Axis("column", 4)
    loads = tuple(Load(name=value, input=buffer, index=(Var(index),)) for value, buffer, index in inputs)
    return Body(
        (
            Loop(
                axis=row,
                body=(
                    Loop(
                        axis=column,
                        body=(
                            *loads,
                            Assign(name="result", op=op, args=args),
                            Write(output="output", index=(Var("row"), Var("column")), value="result"),
                        ),
                    ),
                ),
            ),
        )
    )


def test_structural_key_equal_when_arguments_and_their_loads_are_reordered() -> None:
    """A Q·K-style product has one identity whether the trace encounters Q or K first."""
    query_first = _pointwise_body(
        inputs=(("query", "query_buffer", "row"), ("key", "key_buffer", "column")),
        op="multiply",
        args=("query", "key"),
    )
    key_first = _pointwise_body(
        inputs=(("renamed_key", "renamed_key_buffer", "column"), ("renamed_query", "renamed_query_buffer", "row")),
        op="multiply",
        args=("renamed_key", "renamed_query"),
    )
    assert query_first.structural_key(structural=False) == key_first.structural_key(structural=False)


def test_structural_key_equal_for_reordered_independent_operations() -> None:
    """Every valid topological order of pure definitions has one identity."""
    load = Load(name="input", input="input_buffer", index=(Var("element"),))
    absolute = Assign(name="absolute", op="abs", args=("input",))
    negated = Assign(name="negated", op="negative", args=("input",))
    combine = Assign(name="result", op="add", args=("absolute", "negated"))
    write = Write(output="output", index=(Var("element"),), value="result")
    axis = Axis("element", 4)
    valid_orders = (
        order
        for order in permutations((load, absolute, negated, combine))
        if order.index(load) < order.index(absolute) < order.index(combine)
        and order.index(load) < order.index(negated) < order.index(combine)
    )
    keys = {Body((Loop(axis=axis, body=(*order, write)),)).structural_key(structural=False) for order in valid_orders}
    assert len(keys) == 1


def test_structural_key_equal_for_permuted_same_shape_arguments() -> None:
    """Argument roles, not same-shaped Loads or their names, fix the canonical buffer order."""
    entries = (("x", "X", "row"), ("y", "Y", "row"), ("z", "Z", "row"))
    keys = {
        _pointwise_body(inputs=order, op="where", args=("x", "y", "z")).structural_key(structural=False)
        for order in permutations(entries)
    }
    assert len(keys) == 1


def test_structural_key_equal_when_duplicate_producers_are_reordered() -> None:
    """Use roles disambiguate equal producers without relying on which duplicate came first."""
    load = Load(name="input", input="input_buffer", index=(Var("element"),))
    first = Assign(name="first", op="abs", args=("input",))
    second = Assign(name="second", op="abs", args=("input",))
    exponential = Assign(name="exponential", op="exp", args=("first",))
    logarithm = Assign(name="logarithm", op="log", args=("second",))
    result = Assign(name="result", op="subtract", args=("exponential", "logarithm"))
    write = Write(output="output", index=(Var("element"),), value="result")
    axis = Axis("element", 4)
    one = Body((Loop(axis=axis, body=(load, first, second, exponential, logarithm, result, write)),))
    two = Body((Loop(axis=axis, body=(load, second, first, exponential, logarithm, result, write)),))
    assert one.structural_key(structural=False) == two.structural_key(structural=False)


def test_structural_key_distinguishes_reordered_noncommutative_arguments() -> None:
    """Statement order is free; the operand order of an exact subtract remains identity."""
    inputs = (("left", "left_buffer", "row"), ("right", "right_buffer", "column"))
    left_minus_right = _pointwise_body(inputs=inputs, op="subtract", args=("left", "right"))
    right_minus_left = _pointwise_body(inputs=inputs, op="subtract", args=("right", "left"))
    assert left_minus_right.structural_key(structural=False) != right_minus_left.structural_key(structural=False)


def test_structural_key_distinguishes_repeated_computation() -> None:
    """Canonical ordering retains duplicate instructions because they change kernel work."""
    axis = Axis("element", 4)
    load = Load(name="input", input="input_buffer", index=(Var("element"),))
    first = Assign(name="first", op="abs", args=("input",))
    second = Assign(name="second", op="abs", args=("input",))
    repeated = Body(
        (
            Loop(
                axis=axis,
                body=(
                    load,
                    first,
                    second,
                    Assign(name="result", op="add", args=("first", "second")),
                    Write(output="output", index=(Var("element"),), value="result"),
                ),
            ),
        )
    )
    shared = Body(
        (
            Loop(
                axis=axis,
                body=(
                    load,
                    first,
                    Assign(name="result", op="add", args=("first", "first")),
                    Write(output="output", index=(Var("element"),), value="result"),
                ),
            ),
        )
    )
    assert repeated.structural_key(structural=False) != shared.structural_key(structural=False)


def test_structural_key_preserves_effect_order() -> None:
    """Writes are ordering barriers even when their operands are independent."""
    axis = Axis("element", 4)
    load_x = Load(name="x", input="X", index=(Var("element"),))
    load_y = Load(name="y", input="Y", index=(Var("element"),))
    absolute = Assign(name="absolute", op="abs", args=("x",))
    exponential = Assign(name="exponential", op="exp", args=("y",))
    write_x = Write(output="output", index=(Var("element"),), value="absolute")
    write_y = Write(output="output", index=(Var("element"),), value="exponential")
    xy = Body((Loop(axis=axis, body=(load_x, load_y, absolute, exponential, write_x, write_y)),))
    yx = Body((Loop(axis=axis, body=(load_x, load_y, absolute, exponential, write_y, write_x)),))
    assert xy.structural_key(structural=False) != yx.structural_key(structural=False)


def test_structural_key_distinguishes_buffer_aliasing() -> None:
    """Two argument names and one argument used twice are different signatures."""
    separate = _pointwise_body(
        inputs=(("left", "X", "row"), ("right", "Y", "column")), op="multiply", args=("left", "right")
    )
    aliased = _pointwise_body(
        inputs=(("left", "X", "row"), ("right", "X", "column")), op="multiply", args=("left", "right")
    )
    assert separate.structural_key(structural=False) != aliased.structural_key(structural=False)


def _binary_body(op: str, args: tuple[str, str] = ("x", "y")) -> Body:
    """Two-operand body builder used by the op-clustering tests below."""
    a = Axis("a", 4)
    return Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="x", input="X", index=(Var("a"),)),
                    Load(name="y", input="Y", index=(Var("a"),)),
                    Assign(name="z", op=op, args=args),
                    Write(output="O", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )


def test_structural_key_clusters_fma_ops() -> None:
    """add / subtract / multiply share the FMA cluster — all hash equal.

    The cluster representative is ``add``; two bodies that differ only
    in *which* FMA-issued op sits at the same position are
    structurally equivalent for autotune search purposes."""
    keys = {_binary_body(op).structural_key() for op in ("add", "subtract", "multiply", "negative")}
    assert len(keys) == 1, f"expected 1 cluster key for FMA ops, got {len(keys)}"


def test_structural_key_clusters_sfu_div_ops() -> None:
    """divide / mod / reciprocal share the SFU-div cluster."""
    keys = {_binary_body(op).structural_key() for op in ("divide", "true_divide", "floor_divide", "remainder", "mod")}
    assert len(keys) == 1


def test_structural_key_distinguishes_across_clusters() -> None:
    """Cross-cluster ops still hash distinct — clustering doesn't
    collapse FMA into SFU."""
    fma = _binary_body("add").structural_key()
    sfu_div = _binary_body("divide").structural_key()
    sfu_trans = _binary_body("exp").structural_key()
    compare = _binary_body("maximum").structural_key()
    assert len({fma, sfu_div, sfu_trans, compare}) == 4


def test_structural_key_clusters_collapse_noncommutative_to_commutative() -> None:
    """Side effect of clustering: ``subtract`` (non-commutative) folds
    to ``add`` (commutative), so swapped args sort to the same form.
    Document the consequence explicitly — autotune treats ``x - y`` and
    ``y - x`` as the same kernel shape."""
    assert _binary_body("subtract", ("x", "y")).structural_key() == _binary_body("subtract", ("y", "x")).structural_key()


def test_structural_key_idempotent() -> None:
    """Round-tripping a body through ``normalize_body(canonical_buffers=True)``
    fixes its structural key — re-querying yields the same string."""
    body = _matmul_body("X", "Y", "O")
    k1 = body.structural_key()
    renormalized = Body(normalize_body(body, hoist=False, canonical_buffers=True))
    assert renormalized.structural_key() == k1


def test_structural_key_is_string_and_hashable() -> None:
    body = _matmul_body("X", "Y", "O")
    key = body.structural_key()
    assert isinstance(key, str)
    assert hash(key) == hash(body.structural_key())  # cached, deterministic


def test_structural_key_cached_property() -> None:
    """Accessing twice returns the *same* string object (cached)."""
    body = _matmul_body("X", "Y", "O")
    a = body.structural_key()
    b = body.structural_key()
    assert a is b
