"""Tests for ``Body.structural_key()`` and identity normalization.

The structural key is the canonical structural rendering used for dedup queries — two bodies that differ only by
SSA / axis names, dependency-valid order, equivalent expressions, or external-buffer names must produce the same key.
"""

from __future__ import annotations

from itertools import permutations

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.identity import canonicalize_identity
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Const, Init, Load, Write
from emmy.compiler.ir.stmt.normalize import normalize_body, sort_commutative_args

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
        _pointwise_body(inputs=order, op="where", args=("x", "y", "z")).structural_key(structural=False) for order in permutations(entries)
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


def test_structural_key_equal_when_pure_work_moves_across_a_write() -> None:
    """An unrelated write does not pin a pure definition to one side of it."""
    axis = Axis("element", 4)
    load = Load(name="x", input="X", index=(Var("element"),))
    absolute = Assign(name="absolute", op="abs", args=("x",))
    exponential = Assign(name="exponential", op="exp", args=("x",))
    write_a = Write(output="A", index=(Var("element"),), value="absolute")
    write_b = Write(output="B", index=(Var("element"),), value="exponential")
    one = Body((Loop(axis=axis, body=(load, absolute, write_a, exponential, write_b)),))
    two = Body((Loop(axis=axis, body=(load, absolute, exponential, write_a, write_b)),))
    assert one.structural_key(structural=False) == two.structural_key(structural=False)


def test_structural_key_equal_for_reordered_independent_read_write_chains() -> None:
    """Buffer role discovery cannot invent aliases among unrelated chains."""
    axis = Axis("element", 4)
    chain_one = (
        Load(name="scalar", input="Scalar", index=()),
        Write(output="Vector", index=(Var("element"),), value="scalar"),
    )
    chain_two = (
        Load(name="element_value", input="Input", index=(Var("element"),)),
        Write(output="ScalarOutput", index=(), value="element_value"),
    )
    one = Body((Loop(axis=axis, body=(*chain_one, *chain_two)),))
    two = Body((Loop(axis=axis, body=(*chain_two, *chain_one)),))
    assert one.structural_key(structural=False) == two.structural_key(structural=False)


def test_structural_key_equal_for_reordered_independent_writes() -> None:
    """Writes to distinct buffers are dependency-independent effects."""
    axis = Axis("element", 4)
    load = Load(name="x", input="X", index=(Var("element"),))
    absolute = Assign(name="absolute", op="abs", args=("x",))
    exponential = Assign(name="exponential", op="exp", args=("x",))
    write_a = Write(output="A", index=(Var("element"),), value="absolute")
    write_b = Write(output="B", index=(Var("element"),), value="exponential")
    one = Body((Loop(axis=axis, body=(load, absolute, exponential, write_a, write_b)),))
    two = Body((Loop(axis=axis, body=(load, absolute, exponential, write_b, write_a)),))
    assert one.structural_key(structural=False) == two.structural_key(structural=False)


def test_structural_key_equal_for_reordered_independent_accumulators() -> None:
    """Updates of distinct reduction states have no ordering edge."""
    axis = Axis("reduce", 4)
    load_x = Load(name="x", input="X", index=(Var("reduce"),))
    load_y = Load(name="y", input="Y", index=(Var("reduce"),))
    accum_x = Accum(name="sum_x", value="x", axes=("reduce",))
    accum_y = Accum(name="sum_y", value="y", axes=("reduce",))
    writes = (Write(output="A", index=(), value="sum_x"), Write(output="B", index=(), value="sum_y"))
    one = Body((Loop(axis=axis, body=(load_x, load_y, accum_x, accum_y)), *writes))
    two = Body((Loop(axis=axis, body=(load_x, load_y, accum_y, accum_x)), *writes))
    assert one.structural_key(structural=False) == two.structural_key(structural=False)


def test_structural_key_equal_for_reordered_reduction_axis_metadata() -> None:
    """Reduction-axis metadata is set-like, not an authored-order constraint."""

    def make(axes: tuple[str, ...]) -> Body:
        inner = Body(
            (
                Load(name="x", input="X", index=(Var("row"), Var("column"))),
                Accum(name="total", value="x", axes=axes),
            )
        )
        return Body(
            (
                Loop(axis=Axis("row", 4), body=(Loop(axis=Axis("column", 4), body=inner),)),
                Write(output="O", index=(), value="total"),
            )
        )

    assert make(("row", "column")).structural_key(structural=False) == make(("column", "row")).structural_key(structural=False)
    assert make(("row",)).structural_key(structural=False) != make(("column",)).structural_key(structural=False)


def test_structural_key_equal_for_ambiguous_free_axis_renaming() -> None:
    """Axis spelling cannot decide a loop order when scalar output geometry does not."""

    def make(outer: str, inner: str) -> Body:
        terminal = Body(
            (
                Load(name="x", input="X", index=(Var(outer), Var(inner))),
                Write(output="O", index=(), value="x"),
            )
        )
        return Body((Loop(axis=Axis(outer, 2), body=(Loop(axis=Axis(inner, 3), body=terminal),)),))

    assert make("z", "a").structural_key(structural=False) == make("a", "z").structural_key(structural=False)


def test_structural_key_equal_for_renamed_and_reordered_axis_windows() -> None:
    """Axis and parent provenance names cannot choose the order of otherwise ambiguous loops."""

    def make(names: tuple[str, str], parents: tuple[str, str], reverse: bool) -> Body:
        axes = (
            Axis(
                names[0],
                4,
                window=Window(
                    parent=Axis(parents[0], 8),
                    base=Var(parents[0]),
                    bound=BinaryExpr("+", Var(parents[0]), Literal(4, "int")),
                ),
            ),
            Axis(
                names[1],
                4,
                window=Window(
                    parent=Axis(parents[1], 8),
                    base=BinaryExpr("+", Var(parents[1]), Literal(4, "int")),
                    bound=BinaryExpr("+", Var(parents[1]), Literal(8, "int")),
                    partition=True,
                ),
            ),
        )
        terminal = Body((Load(name="x", input="X", index=()), Write(output="O", index=(), value="x")))
        for axis in reversed(tuple(reversed(axes)) if reverse else axes):
            terminal = Body((Loop(axis=axis, body=terminal),))
        return terminal

    assert make(("outer", "inner"), ("source_a", "source_b"), False).structural_key(structural=False) == make(
        ("renamed_outer", "renamed_inner"), ("renamed_a", "renamed_b"), True
    ).structural_key(structural=False)


def test_structural_key_distinguishes_axis_windows_even_when_bodies_compare_equal() -> None:
    """The shared cache includes codegen-relevant fields excluded from dataclass equality."""

    def make(window: Window) -> Body:
        axis = Axis("element", 4, window=window)
        return Body(
            (
                Loop(
                    axis=axis,
                    body=(
                        Load(name="x", input="X", index=(Var("element"),)),
                        Write(output="O", index=(Var("element"),), value="x"),
                    ),
                ),
            )
        )

    first = make(Window(parent=Axis("source", 8), base=Literal(0, "int"), bound=Literal(4, "int")))
    second = make(
        Window(
            parent=Axis("other_source", 8),
            base=Literal(4, "int"),
            bound=Literal(8, "int"),
            partition=True,
        )
    )
    assert first == second
    assert first.structural_key(structural=False) != second.structural_key(structural=False)


def test_structural_key_ignores_symbolic_dimension_hints() -> None:
    """An expected size guides tuning but does not change the kernel's structural identity."""

    def make(hint: int) -> Body:
        axis = Axis("element", Dim("size", hint=hint))
        return Body(
            (
                Loop(
                    axis=axis,
                    body=(
                        Load(name="x", input="X", index=(Var("element"),)),
                        Write(output="O", index=(Var("element"),), value="x"),
                    ),
                ),
            )
        )

    assert make(32).structural_key(structural=False) == make(64).structural_key(structural=False)


def test_structural_key_equal_for_renamed_condition_and_reduction_axis() -> None:
    """Every axis reference follows its binder through conditions and reduction metadata."""

    def make(element: str, reduce: str, predicate: str, value: str, total: str) -> Body:
        reduction = Loop(
            axis=Axis(reduce, 4),
            body=(
                Load(name=value, input="X", index=(Var(element), Var(reduce))),
                Accum(name=total, value=value, axes=(reduce,)),
            ),
        )
        return Body(
            (
                Loop(
                    axis=Axis(element, 4),
                    body=(
                        Load(name=predicate, input="Mask", index=(Var(element),)),
                        reduction,
                        Cond(
                            cond=Var(predicate),
                            body=(Write(output="O", index=(Var(element),), value=total),),
                        ),
                    ),
                ),
            )
        )

    assert make("row", "k", "predicate", "value", "total").structural_key(structural=False) == make(
        "renamed_row", "renamed_k", "renamed_predicate", "renamed_value", "renamed_total"
    ).structural_key(structural=False)


def test_structural_key_equal_for_renamed_exported_accumulators() -> None:
    """Several exported states are allocated in body order, never source-name set order."""

    def make(total: str, maximum: str) -> Body:
        reduction = Loop(
            axis=Axis("reduce", 4),
            body=(
                Load(name="x", input="X", index=(Var("reduce"),)),
                Load(name="y", input="Y", index=(Var("reduce"),)),
                Assign(name="absolute", op="abs", args=("x",)),
                Assign(name="exponential", op="exp", args=("y",)),
                Accum(name=total, value="absolute", op="add", axes=("reduce",)),
                Accum(name=maximum, value="exponential", op="maximum", axes=("reduce",)),
            ),
        )
        return Body(
            (
                reduction,
                Write(output="Sum", index=(), value=total),
                Write(output="Max", index=(), value=maximum),
            )
        )

    assert make("total", "maximum").structural_key(structural=False) == make("alpha_ssa_0", "alpha_ssa_1").structural_key(structural=False)


def test_structural_key_equal_for_equivalent_index_and_comparison_expressions() -> None:
    """Identity canonicalizes commutative expressions and comparison duals."""

    def make(index, condition) -> Body:
        return Body(
            (
                Loop(
                    axis=Axis("element", 8),
                    body=(
                        Load(name="x", input="X", index=(index,)),
                        Cond(condition, body=(Write(output="O", index=(Var("element"),), value="x"),)),
                    ),
                ),
            )
        )

    element, one, four = Var("element"), Literal(1, "int"), Literal(4, "int")
    left = make(BinaryExpr("+", element, one), BinaryExpr("<", element, four))
    right = make(BinaryExpr("+", one, element), BinaryExpr(">", four, element))
    assert left.structural_key(structural=False) == right.structural_key(structural=False)


def test_structural_key_equal_for_affine_index_spellings() -> None:
    """Constant folding and association do not split equivalent affine addresses."""
    element = Var("element")
    one, two, three, six = (Literal(value, "int") for value in (1, 2, 3, 6))

    def key(index) -> str:
        return Body(
            (
                Loop(
                    axis=Axis("element", 8),
                    body=(Load(name="x", input="X", index=(index,)), Write(output="O", index=(), value="x")),
                ),
            )
        ).structural_key(structural=False)

    assert key(BinaryExpr("+", BinaryExpr("+", element, one), two)) == key(BinaryExpr("+", element, three))
    assert key(BinaryExpr("*", BinaryExpr("*", element, two), three)) == key(BinaryExpr("*", element, six))
    assert key(BinaryExpr("-", element, one)) == key(BinaryExpr("+", element, Literal(-1, "int")))


def test_structural_key_distinguishes_noncommutative_index_expressions() -> None:
    """Identity does not reorder subtraction operands."""
    axis = Axis("element", 8)
    one = Literal(1, "int")

    def make(index) -> Body:
        return Body(
            (
                Loop(
                    axis=axis,
                    body=(Load(name="x", input="X", index=(index,)), Write(output="O", index=(), value="x")),
                ),
            )
        )

    assert make(BinaryExpr("-", Var("element"), one)).structural_key(structural=False) != make(
        BinaryExpr("-", one, Var("element"))
    ).structural_key(structural=False)


def test_structural_key_distinguishes_reassociated_ssa_arithmetic() -> None:
    """Affine coordinate folding cannot collapse a floating-point SSA operation tree."""

    def make(value: BinaryExpr) -> Body:
        return Body(
            (
                Load(name="x", input="X", index=()),
                Cond(
                    BinaryExpr("<", value, Literal(0.0)),
                    body=(Write(output="O", index=(), value="x"),),
                ),
            )
        )

    x = Var("x")
    associated = BinaryExpr("+", BinaryExpr("+", x, Literal(1.0)), Literal(2.0))
    folded = BinaryExpr("+", x, Literal(3.0))
    assert make(associated).structural_key(structural=False) != make(folded).structural_key(structural=False)


def test_structural_key_equal_for_renamed_const_and_init() -> None:
    """All definition-bearing Loop IR statements participate in alpha-renaming."""

    def make(constant: str, initial: str) -> Body:
        return Body(
            (
                Const(name=constant, value=2),
                Init(name=initial, identity=0, dtype=F32),
                Assign(name="result", op="add", args=(constant, initial)),
                Write(output="O", index=(), value="result"),
            )
        )

    assert make("constant", "initial").structural_key(structural=False) == make("renamed_constant", "renamed_initial").structural_key(
        structural=False
    )


def test_structural_key_equal_for_renamed_vector_load_lanes() -> None:
    """Vector Load lane binders are alpha-renamed together."""

    def make(names: tuple[str, str]) -> Body:
        return Body(
            (
                Load(names=names, input="X", index=()),
                Assign(name="result", op="add", args=names),
                Write(output="O", index=(), value="result"),
            )
        )

    assert make(("x0", "x1")).structural_key(structural=False) == make(("renamed0", "renamed1")).structural_key(structural=False)


def test_structural_key_equal_when_sibling_scopes_reuse_local_names() -> None:
    """A spelling reused in sibling scopes still denotes two lexical binders."""

    def make(
        axes: tuple[str, str],
        values: tuple[str, str],
        buffers: tuple[str, str],
        outputs: tuple[str, str],
    ) -> Body:
        return Body(
            tuple(
                Loop(
                    axis=Axis(axis, 4),
                    body=(
                        Load(name=value, input=input_buffer, index=(Var(axis),)),
                        Assign(
                            name=f"{value}_result",
                            op=operation,
                            args=(value,),
                        ),
                        Write(output=output, index=(Var(axis),), value=f"{value}_result"),
                    ),
                )
                for axis, value, input_buffer, output, operation in zip(axes, values, buffers, outputs, ("abs", "exp"), strict=True)
            )
        )

    reused = make(("i", "i"), ("x", "x"), ("A", "B"), ("OA", "OB"))
    distinct = make(
        ("left_axis", "right_axis"),
        ("left_value", "right_value"),
        ("left_input", "right_input"),
        ("left_output", "right_output"),
    )
    renamed_and_reordered = Body(
        reversed(
            make(
                ("renamed_left_axis", "renamed_right_axis"),
                ("renamed_left_value", "renamed_right_value"),
                ("renamed_left_input", "renamed_right_input"),
                ("renamed_left_output", "renamed_right_output"),
            )
        )
    )
    assert (
        len(
            {
                reused.structural_key(structural=False),
                distinct.structural_key(structural=False),
                renamed_and_reordered.structural_key(structural=False),
            }
        )
        == 1
    )


def test_structural_key_equal_when_a_copy_alias_precedes_a_sibling_scope() -> None:
    def make(reuse_names: bool) -> Body:
        right_axis = "i" if reuse_names else "j"
        right_input = "x" if reuse_names else "right_input"
        right_result = "y" if reuse_names else "right_result"
        return Body(
            (
                Loop(
                    axis=Axis("i", 4),
                    body=(
                        Load(name="x", input="A", index=(Var("i"),)),
                        Assign(name="y", op="copy", args=("x",)),
                        Write(output="OA", index=(Var("i"),), value="y"),
                    ),
                ),
                Loop(
                    axis=Axis(right_axis, 4),
                    body=(
                        Load(name=right_input, input="B", index=(Var(right_axis),)),
                        Assign(name=right_result, op="exp", args=(right_input,)),
                        Write(output="OB", index=(Var(right_axis),), value=right_result),
                    ),
                ),
            )
        )

    assert make(True).structural_key(structural=False) == make(False).structural_key(structural=False)


def test_structural_key_handles_large_symmetric_partitions() -> None:
    """Exact symmetry does not trigger factorial buffer, producer, or axis searches."""
    axis = Axis("element", 4)
    loads = tuple(Load(name=f"x{i}", input=f"X{i}", index=(Var("element"),)) for i in range(9))
    buffers = Body(
        (
            Loop(
                axis=axis,
                body=(
                    *loads,
                    Assign(name="result", op="add", args=tuple(f"x{i}" for i in range(9))),
                    Write(output="O", index=(Var("element"),), value="result"),
                ),
            ),
        )
    )
    load = Load(name="x", input="X", index=(Var("element"),))
    producers = tuple(Assign(name=f"p{i}", op="abs", args=("x",)) for i in range(9))
    duplicate_producers = Body(
        (
            Loop(
                axis=axis,
                body=(
                    load,
                    *producers,
                    Assign(name="result", op="add", args=tuple(f"p{i}" for i in range(9))),
                    Write(output="O", index=(Var("element"),), value="result"),
                ),
            ),
        )
    )
    nested = Body((Load(name="x", input="X", index=()), Write(output="O", index=(), value="x")))
    for i in reversed(range(9)):
        nested = Body((Loop(axis=Axis(f"axis{i}", 2), body=nested),))

    assert (
        len(
            {
                buffers.structural_key(structural=False),
                duplicate_producers.structural_key(structural=False),
                nested.structural_key(structural=False),
            }
        )
        == 3
    )


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
    """Writes to the same buffer retain their order even when their operands are independent."""
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


def test_structural_key_preserves_a_read_across_a_write_to_its_buffer() -> None:
    prefix = (
        Load(name="old", input="B", index=()),
        Const(name="replacement", value=7),
        Write(output="P", index=(), value="old"),
    )
    write = Write(output="B", index=(), value="replacement")
    read = Load(name="new", input="B", index=())
    observe = Write(output="O", index=(), value="new")
    after_write = Body((*prefix, write, read, observe))
    before_write = Body((*prefix, read, write, observe))

    assert after_write.structural_key(structural=False) != before_write.structural_key(structural=False)


def test_structural_key_preserves_shared_accumulator_order() -> None:
    """Updates to one reduction state retain their execution order."""
    axis = Axis("reduce", 4)
    prefix = (
        Load(name="x", input="X", index=(Var("reduce"),)),
        Load(name="y", input="Y", index=(Var("reduce"),)),
        Assign(name="absolute", op="abs", args=("x",)),
        Assign(name="exponential", op="exp", args=("y",)),
    )
    absolute = Accum(name="total", value="absolute", axes=("reduce",))
    exponential = Accum(name="total", value="exponential", axes=("reduce",))
    suffix = (Write(output="O", index=(), value="total"),)
    one = Body((Loop(axis=axis, body=(*prefix, absolute, exponential)), *suffix))
    two = Body((Loop(axis=axis, body=(*prefix, exponential, absolute)), *suffix))
    assert one.structural_key(structural=False) != two.structural_key(structural=False)


def test_structural_key_preserves_when_accumulator_state_is_observed() -> None:
    """A state read remains after the update whose value it observes."""
    prefix = (
        Init(name="total", identity=0, dtype=F32),
        Const(name="first", value=1),
        Const(name="second", value=2),
        Accum(name="total", value="first"),
    )
    final_update = Accum(name="total", value="second")
    observe = Write(output="O", index=(), value="total")
    after_both = Body((*prefix, final_update, observe))
    between = Body((*prefix, observe, final_update))
    assert after_both.structural_key(structural=False) != between.structural_key(structural=False)


def test_structural_key_distinguishes_reduction_seed_policy() -> None:
    """A reduction seed changes execution and must remain in exact identity."""
    axis = Axis("reduce", 4)
    body = (
        Load(name="x", input="X", index=(Var("reduce"),)),
        Accum(name="total", value="x", axes=("reduce",)),
    )
    seeded = Body((Loop(axis=axis, body=body, seed=True), Write(output="O", index=(), value="total")))
    continuing = Body((Loop(axis=axis, body=body, seed=False), Write(output="O", index=(), value="total")))
    assert seeded.structural_key(structural=False) != continuing.structural_key(structural=False)


def test_structural_key_distinguishes_buffer_aliasing() -> None:
    """Two argument names and one argument used twice are different signatures."""
    separate = _pointwise_body(inputs=(("left", "X", "row"), ("right", "Y", "column")), op="multiply", args=("left", "right"))
    aliased = _pointwise_body(inputs=(("left", "X", "row"), ("right", "X", "column")), op="multiply", args=("left", "right"))
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
    """Identity normalization reaches a fixed canonical body."""
    body = _matmul_body("X", "Y", "O")
    canonical = canonicalize_identity(normalize_body(body, hoist=False))
    assert canonicalize_identity(canonical) == canonical


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
