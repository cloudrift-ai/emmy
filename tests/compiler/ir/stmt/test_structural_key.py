"""Tests for ``Body.structural_key()`` and identity normalization.

The structural key is the canonical structural rendering used for dedup queries — two bodies that differ only by
SSA / axis names, dependency-valid order, equivalent expressions, or external-buffer names must produce the same key.
"""

from __future__ import annotations

from itertools import permutations, product

from emmy.compiler.ir.stmt import Body, canonicalize_identity, normalize_body, parse_body


def _key(body: Body | str) -> str:
    """The exact identity of a body, or of the body a text spells."""
    return (parse_body(body) if isinstance(body, str) else body).structural_key(structural=False)


def _normal(body: Body | str) -> str:
    """The text of the normal form of a body, or of the body a text spells."""
    return str(normalize_body(parse_body(body) if isinstance(body, str) else body))


def _loop(axis: str, extent: int, *lines: str) -> str:
    """The text of one loop over ``lines``, each of which may itself span several lines."""
    return "\n".join([f"for {axis} in 0..{extent}:", *(f"    {line}" for block in lines for line in block.splitlines())])


# ---------------------------------------------------------------------------
# sort_commutative_args
# ---------------------------------------------------------------------------


def test_affine_index_equivalence_and_cse_have_one_exact_identity() -> None:
    keys, normals = set(), set()
    for i, j in (("i", "j"), ("z", "q")):
        forms = (f"{i} + 2 * {j} + 3", f"3 + {j} + ({i} + {j})", f"3 * ({i} + {j} + 1) - (2 * {i} + {j})")
        for first, second in product(forms, repeat=2):
            body = f"""
                for {i} in 0..4:
                    for {j} in 0..4:
                        a = load x[{first}]
                        b = load x[{second}]
                        left = exp(a)
                        right = exp(b)
                        sum = add(right, left)
                        out[{i}, {j}] = sum
                """
            keys.add(_key(body))
            normals.add(_normal(body))
    assert len(keys) == 1
    (normal,) = normals
    lines = [line.strip() for line in normal.splitlines()]
    assert sum(" = load " in line for line in lines) == 1
    assert sum(not line.startswith("for ") and " = load " not in line and not line.startswith("out[") for line in lines) == 2
    assert _normal(normal) == normal


# ---------------------------------------------------------------------------
# Body.structural_key()
# ---------------------------------------------------------------------------


def _matmul_body(input_x: str, input_y: str, output: str) -> Body:
    """Tiny multiply-and-write body parameterized by buffer names."""
    return parse_body(
        f"""
        for a in 0..4:
            x = load {input_x}[a]
            y = load {input_y}[a]
            z = multiply(x, y)
            {output}[a] = z
        """
    )


def test_structural_key_equal_for_renamed_buffers() -> None:
    a = _matmul_body("X", "Y", "O")
    b = _matmul_body("foo", "bar", "baz")
    assert _normal(a) != _normal(b)
    assert a.structural_key() == b.structural_key()


def test_structural_key_equal_for_swapped_commutative_args() -> None:
    body_xy = parse_body(
        """
        for a in 0..4:
            lx = load X[a]
            ly = load Y[a]
            z = add(lx, ly)
            O[a] = z
        """
    )
    body_yx = parse_body(
        """
        for a in 0..4:
            ly = load Y[a]
            lx = load X[a]
            z = add(ly, lx)
            O[a] = z
        """
    )
    assert _normal(body_xy) == _normal(body_yx)
    assert body_xy.structural_key() == body_yx.structural_key()


def _pointwise_body(*, inputs: tuple[tuple[str, str, str], ...], op: str, args: tuple[str, ...]) -> Body:
    """One pointwise kernel with an authored load order over row/column arguments."""
    loads = (f"{value} = load {buffer}[{index}]" for value, buffer, index in inputs)
    inner = _loop("column", 4, *loads, f"result = {op}({', '.join(args)})", "output[row, column] = result")
    return parse_body(_loop("row", 4, inner))


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
    assert _key(query_first) == _key(key_first)


def test_structural_key_equal_for_reordered_independent_operations() -> None:
    """Every valid topological order of pure definitions has one identity."""
    load = "input = load input_buffer[element]"
    absolute = "absolute = abs(input)"
    negated = "negated = negative(input)"
    combine = "result = add(absolute, negated)"
    write = "output[element] = result"
    valid_orders = tuple(
        order
        for order in permutations((load, absolute, negated, combine))
        if order.index(load) < order.index(absolute) < order.index(combine)
        and order.index(load) < order.index(negated) < order.index(combine)
    )
    bodies = [_loop("element", 4, *order, write) for order in valid_orders]
    assert len(valid_orders) == 2
    assert len({_normal(body) for body in bodies}) == 1
    assert len({_key(body) for body in bodies}) == 1


def test_structural_key_equal_for_permuted_same_shape_arguments() -> None:
    """Argument roles, not same-shaped Loads or their names, fix the canonical buffer order."""
    entries = (("x", "X", "row"), ("y", "Y", "row"), ("z", "Z", "row"))
    keys = {_key(_pointwise_body(inputs=order, op="where", args=("x", "y", "z"))) for order in permutations(entries)}
    assert len(keys) == 1


def test_structural_key_equal_when_duplicate_producers_are_reordered() -> None:
    """Use roles disambiguate equal producers without relying on which duplicate came first."""
    load = "input = load input_buffer[element]"
    first = "first = abs(input)"
    second = "second = abs(input)"
    tail = ("exponential = exp(first)", "logarithm = log(second)", "result = subtract(exponential, logarithm)", "output[element] = result")
    assert _key(_loop("element", 4, load, first, second, *tail)) == _key(_loop("element", 4, load, second, first, *tail))


def test_structural_key_equal_when_pure_work_moves_across_a_write() -> None:
    """An unrelated write does not pin a pure definition to one side of it."""
    load = "x = load X[element]"
    absolute = "absolute = abs(x)"
    exponential = "exponential = exp(x)"
    write_a = "A[element] = absolute"
    write_b = "B[element] = exponential"
    one = _loop("element", 4, load, absolute, write_a, exponential, write_b)
    two = _loop("element", 4, load, absolute, exponential, write_a, write_b)
    assert _key(one) == _key(two)


def test_structural_key_equal_for_reordered_independent_read_write_chains() -> None:
    """Buffer role discovery cannot invent aliases among unrelated chains."""
    chain_one = ("scalar = load Scalar[]", "Vector[element] = scalar")
    chain_two = ("element_value = load Input[element]", "ScalarOutput[] = element_value")
    assert _key(_loop("element", 4, *chain_one, *chain_two)) == _key(_loop("element", 4, *chain_two, *chain_one))


def test_structural_key_equal_for_reordered_independent_writes() -> None:
    """Writes to distinct buffers are dependency-independent effects."""
    prefix = ("x = load X[element]", "absolute = abs(x)", "exponential = exp(x)")
    write_a = "A[element] = absolute"
    write_b = "B[element] = exponential"
    assert _key(_loop("element", 4, *prefix, write_a, write_b)) == _key(_loop("element", 4, *prefix, write_b, write_a))


def test_structural_key_equal_for_reordered_independent_accumulators() -> None:
    """Updates of distinct reduction states have no ordering edge."""
    loads = ("x = load X[reduce]", "y = load Y[reduce]")
    accum_x = "sum_x <- add(sum_x, x) over reduce"
    accum_y = "sum_y <- add(sum_y, y) over reduce"
    writes = "A[] = sum_x\nB[] = sum_y"
    one = _loop("reduce", 4, *loads, accum_x, accum_y) + "\n" + writes
    two = _loop("reduce", 4, *loads, accum_y, accum_x) + "\n" + writes
    assert _key(one) == _key(two)


def test_structural_key_equal_for_reordered_reduction_axis_metadata() -> None:
    """Reduction-axis metadata is set-like, not an authored-order constraint."""

    def make(axes: str) -> str:
        return f"""
            for row in 0..4:
                for column in 0..4:
                    x = load X[row, column]
                    total <- add(total, x) over {axes}
            O[] = total
            """

    per_row = """
        for row in 0..4:
            for column in 0..4:
                x = load X[row, column]
                total <- add(total, x) over column
            O[row] = total
        """
    assert _key(make("row, column")) == _key(make("column, row"))
    assert _key(make("row, column")) != _key(per_row)


def test_structural_key_equal_for_ambiguous_free_axis_renaming() -> None:
    """Axis spelling cannot decide a loop order when scalar output geometry does not."""

    def make(outer: str, inner: str) -> str:
        return f"""
            for {outer} in 0..2:
                for {inner} in 0..3:
                    x = load X[{outer}, {inner}]
                    O[] = x
            """

    assert _normal(make("z", "a")) == _normal(make("a", "z"))
    assert _key(make("z", "a")) == _key(make("a", "z"))


def test_normalize_body_keeps_shared_axes_outside_known_output_geometry() -> None:
    """A composed batch coordinate stays outside matrix rows and columns after alpha-renaming."""

    def make(names: tuple[str, str, str], order: tuple[int, int, int]) -> str:
        batch, row, column = names
        index = f"{batch} // 3, {batch} % 3, {row}, {column}"
        result = f"x = load X[{index}]\nO[{index}] = x"
        axes = ((batch, 6), (row, 128), (column, 128))
        for position in reversed(order):
            result = _loop(*axes[position], result)
        return result

    first = _normal(make(("batch", "row", "column"), (0, 1, 2)))
    second = _normal(make(("b", "m", "n"), (2, 0, 1)))
    assert first == second
    # The three loops nest in that order: each header one level deeper, the batch outermost.
    headers = first.splitlines()[:3]
    assert [header.index("for ") for header in headers] == [0, 4, 8]
    assert [header.split(" in ")[1] for header in headers] == ["0..6:", "0..128:", "0..128:"]


def test_structural_key_equal_for_renamed_and_reordered_axis_windows() -> None:
    """Axis and parent provenance names cannot choose the order of otherwise ambiguous loops."""

    def make(names: tuple[str, str], parents: tuple[str, str], reverse: bool) -> str:
        loops = (
            f"for {names[0]} in 0..4 window(parent={parents[0]} in 0..8, base={parents[0]}, bound={parents[0]} + 4):",
            f"for {names[1]} in 0..4 window(parent={parents[1]} in 0..8, base={parents[1]} + 4, bound={parents[1]} + 8, partition=true):",
        )
        outer, inner = reversed(loops) if reverse else loops
        return "\n".join([outer, f"    {inner}", "        x = load X[]", "        O[] = x"])

    assert _key(make(("outer", "inner"), ("source_a", "source_b"), False)) == _key(
        make(("renamed_outer", "renamed_inner"), ("renamed_a", "renamed_b"), True)
    )


def test_structural_key_distinguishes_axis_windows_even_when_bodies_compare_equal() -> None:
    """The key includes codegen-relevant fields excluded from equality."""

    def make(window: str) -> Body:
        return parse_body(
            f"""
            for element in 0..4 window({window}):
                x = load X[element]
                O[element] = x
            """
        )

    first = make("parent=source in 0..8, base=0, bound=4")
    second = make("parent=other_source in 0..8, base=4, bound=8, partition=true")
    assert first == second
    assert _key(first) != _key(second)


def test_structural_key_ignores_symbolic_dimension_hints() -> None:
    """An expected size guides tuning but does not change the kernel's structural identity."""

    def make(hint: int) -> str:
        return f"""
            for element in 0..size~{hint}:
                x = load X[element]
                O[element] = x
            """

    assert _key(make(32)) == _key(make(64))


def test_structural_key_equal_for_renamed_condition_and_reduction_axis() -> None:
    """Every axis reference follows its binder through conditions and reduction metadata."""

    def make(element: str, reduce: str, predicate: str, value: str, total: str) -> str:
        return f"""
            for {element} in 0..4:
                {predicate} = load Mask[{element}]
                for {reduce} in 0..4:
                    {value} = load X[{element}, {reduce}]
                    {total} <- add({total}, {value}) over {reduce}
                if {predicate}:
                    O[{element}] = {total}
            """

    assert _key(make("row", "k", "predicate", "value", "total")) == _key(
        make("renamed_row", "renamed_k", "renamed_predicate", "renamed_value", "renamed_total")
    )


def test_structural_key_equal_for_renamed_exported_accumulators() -> None:
    """Several exported states are allocated in body order, never source-name set order."""

    def make(total: str, maximum: str) -> str:
        return f"""
            for reduce in 0..4:
                x = load X[reduce]
                y = load Y[reduce]
                absolute = abs(x)
                exponential = exp(y)
                {total} <- add({total}, absolute) over reduce
                {maximum} <- maximum({maximum}, exponential) over reduce
            Sum[] = {total}
            Max[] = {maximum}
            """

    assert _key(make("total", "maximum")) == _key(make("alpha_ssa_0", "alpha_ssa_1"))


def test_structural_key_equal_for_equivalent_index_and_comparison_expressions() -> None:
    """Identity canonicalizes commutative expressions and comparison duals."""

    def make(index: str, condition: str) -> str:
        return f"""
            for element in 0..8:
                x = load X[{index}]
                if {condition}:
                    O[element] = x
            """

    left = make("element + 1", "element < 4")
    right = make("1 + element", "4 > element")
    assert _normal(left) == _normal(right)
    assert _key(left) == _key(right)


def _indexed(index: str) -> str:
    """A loop that reads ``X`` at ``index`` and stores the value in a scalar."""
    return f"""
        for element in 0..8:
            x = load X[{index}]
            O[] = x
        """


def test_structural_key_equal_for_affine_index_spellings() -> None:
    """Constant folding and association do not split equivalent affine addresses."""
    assert _key(_indexed("element + 1 + 2")) == _key(_indexed("element + 3"))
    assert _key(_indexed("element * 2 * 3")) == _key(_indexed("element * 6"))
    assert _key(_indexed("element - 1")) == _key(_indexed("element + -1"))


def test_structural_key_distinguishes_noncommutative_index_expressions() -> None:
    """Identity does not reorder subtraction operands."""
    assert _key(_indexed("element - 1")) != _key(_indexed("1 - element"))


def test_structural_key_distinguishes_reassociated_ssa_arithmetic() -> None:
    """Affine coordinate folding cannot collapse a floating-point SSA operation tree."""

    def make(value: str) -> str:
        return f"""
            x = load X[]
            if {value} < 0.0:
                O[] = x
            """

    assert _key(make("x + 1.0 + 2.0")) != _key(make("x + 3.0"))


def test_structural_key_equal_for_renamed_const_and_init() -> None:
    """All definition-bearing Loop IR statements participate in alpha-renaming."""

    def make(constant: str, initial: str) -> str:
        return f"""
            let {constant} = 2.0
            init f32 {initial} = 0
            result = add({constant}, {initial})
            O[] = result
            """

    assert _key(make("constant", "initial")) == _key(make("renamed_constant", "renamed_initial"))


def test_structural_key_equal_for_renamed_vector_load_lanes() -> None:
    """Vector Load lane binders are alpha-renamed together."""

    def make(first: str, second: str) -> str:
        return f"""
            {first}, {second} = load X[]
            result = add({first}, {second})
            O[] = result
            """

    assert _key(make("x0", "x1")) == _key(make("renamed0", "renamed1"))


def test_structural_key_equal_when_sibling_scopes_reuse_local_names() -> None:
    """A spelling reused in sibling scopes still denotes two lexical binders."""

    def make(axes: tuple[str, str], values: tuple[str, str], buffers: tuple[str, str], outputs: tuple[str, str]) -> list[str]:
        return [
            _loop(axis, 4, f"{value} = load {buffer}[{axis}]", f"{value}_result = {op}({value})", f"{output}[{axis}] = {value}_result")
            for axis, value, buffer, output, op in zip(axes, values, buffers, outputs, ("abs", "exp"), strict=True)
        ]

    reused = make(("i", "i"), ("x", "x"), ("A", "B"), ("OA", "OB"))
    distinct = make(
        ("left_axis", "right_axis"),
        ("left_value", "right_value"),
        ("left_input", "right_input"),
        ("left_output", "right_output"),
    )
    renamed_and_reordered = reversed(
        make(
            ("renamed_left_axis", "renamed_right_axis"),
            ("renamed_left_value", "renamed_right_value"),
            ("renamed_left_input", "renamed_right_input"),
            ("renamed_left_output", "renamed_right_output"),
        )
    )
    assert len({_key("\n".join(loops)) for loops in (reused, distinct, renamed_and_reordered)}) == 1


def test_normalize_body_canonicalizes_nested_scope_orders_independently() -> None:
    """Two ambiguous child scopes do not multiply their order choices or leak alpha names."""

    def make(axis: str, *, reverse_scopes: bool, reverse_left: bool, reverse_right: bool) -> str:
        def child(stem: str, source: str, outputs: tuple[str, str], operations: tuple[str, str], reverse: bool) -> str:
            value = f"{stem}_input"
            chains = [
                (f"{stem}_value_{index} = {operation}({value})", f"{output}[{axis}] = {stem}_value_{index}")
                for index, (operation, output) in enumerate(zip(operations, outputs, strict=True))
            ]
            ordered = reversed(chains) if reverse else chains
            lines = (f"{value} = load {source}[{axis}]", *(line for chain in ordered for line in chain))
            return "\n".join([f"if {axis} < 4:", *(f"    {line}" for line in lines)])

        scopes = (
            child("left", "X", ("A0", "A1"), ("abs", "negative"), reverse_left),
            child("right", "Y", ("B0", "B1"), ("exp", "reciprocal"), reverse_right),
        )
        return _loop(axis, 4, *(reversed(scopes) if reverse_scopes else scopes))

    normalized = {
        _normal(
            make(
                "renamed_axis" if renamed else "element",
                reverse_scopes=reverse_scopes,
                reverse_left=reverse_left,
                reverse_right=reverse_right,
            )
        )
        for renamed, reverse_scopes, reverse_left, reverse_right in product((False, True), repeat=4)
    }
    assert len(normalized) == 1


def test_normalize_body_canonicalizes_outer_captures_before_nested_order() -> None:
    """A nested order cannot depend on the source spelling of an outer SSA value."""

    def make(left: str) -> Body:
        return parse_body(
            f"""
            {left} = load X[]
            right = load Y[]
            if 1:
                left_abs = abs({left})
                right_abs = abs(right)
                result = add(left_abs, right_abs)
                O[] = result
            """
        )

    canonical = normalize_body(make("x"))
    assert normalize_body(make("renamed_x")) == canonical
    assert normalize_body(canonical) == canonical
    assert normalize_body(canonical) is canonical


def test_structural_key_equal_when_a_copy_alias_precedes_a_sibling_scope() -> None:
    def make(reuse_names: bool) -> str:
        right_axis = "i" if reuse_names else "j"
        right_input = "x" if reuse_names else "right_input"
        right_result = "y" if reuse_names else "right_result"
        return f"""
            for i in 0..4:
                x = load A[i]
                y = copy(x)
                OA[i] = y
            for {right_axis} in 0..4:
                {right_input} = load B[{right_axis}]
                {right_result} = exp({right_input})
                OB[{right_axis}] = {right_result}
            """

    assert _key(make(True)) == _key(make(False))


def test_structural_key_handles_large_symmetric_partitions() -> None:
    """Exact symmetry does not trigger factorial buffer, producer, or axis searches."""
    loads = (f"x{i} = load X{i}[element]" for i in range(9))
    buffers = _loop("element", 4, *loads, f"result = add({', '.join(f'x{i}' for i in range(9))})", "O[element] = result")
    producers = (f"p{i} = abs(x)" for i in range(9))
    duplicate_producers = _loop(
        "element",
        4,
        "x = load X[element]",
        *producers,
        f"result = add({', '.join(f'p{i}' for i in range(9))})",
        "O[element] = result",
    )
    nested = "x = load X[]\nO[] = x"
    for i in reversed(range(9)):
        nested = _loop(f"axis{i}", 2, nested)

    assert len({_key(buffers), _key(duplicate_producers), _key(nested)}) == 3


def test_normalize_body_refines_large_asymmetric_sibling_partition() -> None:
    """Graph context splits a local-form tie before the exact ordering fallback."""

    def make(prefix: str, *, reverse: bool) -> str:
        constants = [f"let {prefix}_source_{index} = {index}.0" for index in range(9)]
        consumers = [f"if 1:\n    {prefix}_value_{index} = abs({prefix}_source_{index})" for index in range(9)]
        return "\n".join([*constants, *(reversed(consumers) if reverse else consumers)])

    assert _normal(make("first", reverse=False)) == _normal(make("renamed", reverse=True))


def test_normalize_body_keeps_independent_sibling_scopes_together() -> None:
    """Ready blocks stay ahead of leaf epilogues so normalization does not widen scheduling."""
    normal = _normal(
        """
        for first_axis in 0..4:
            first_input = load X[first_axis]
            first_sum <- add(first_sum, first_input) over first_axis
        first_result = abs(first_sum)
        for second_axis in 0..4:
            second_input = load Y[second_axis]
            second_sum <- add(second_sum, second_input) over second_axis
        second_result = exp(second_sum)
        O0[] = first_result
        O1[] = second_result
        """
    )
    first, second, *_ = (line for line in normal.splitlines() if not line.startswith(" "))
    assert first.startswith("for ") and second.startswith("for ")


def test_structural_key_distinguishes_reordered_noncommutative_arguments() -> None:
    """Statement order is free; the operand order of an exact subtract remains identity."""
    inputs = (("left", "left_buffer", "row"), ("right", "right_buffer", "column"))
    left_minus_right = _pointwise_body(inputs=inputs, op="subtract", args=("left", "right"))
    right_minus_left = _pointwise_body(inputs=inputs, op="subtract", args=("right", "left"))
    assert _key(left_minus_right) != _key(right_minus_left)


def test_structural_key_folds_repeated_computation() -> None:
    """A pure statement spelled twice over the same names is one value: normalization keeps the
    first and aliases the second, so the two spellings are one kernel and one key."""
    repeated = """
        for element in 0..4:
            input = load input_buffer[element]
            first = abs(input)
            second = abs(input)
            result = add(first, second)
            output[element] = result
        """
    shared = """
        for element in 0..4:
            input = load input_buffer[element]
            first = abs(input)
            result = add(first, first)
            output[element] = result
        """
    assert _key(repeated) == _key(shared)


def test_structural_key_preserves_effect_order() -> None:
    """Writes to the same buffer retain their order even when their operands are independent."""
    prefix = ("x = load X[element]", "y = load Y[element]", "absolute = abs(x)", "exponential = exp(y)")
    write_x = "output[element] = absolute"
    write_y = "output[element] = exponential"
    assert _key(_loop("element", 4, *prefix, write_x, write_y)) != _key(_loop("element", 4, *prefix, write_y, write_x))


def test_structural_key_preserves_a_read_across_a_write_to_its_buffer() -> None:
    prefix = ("old = load B[]", "let replacement = 7.0", "P[] = old")
    write = "B[] = replacement"
    read = "new = load B[]"
    observe = "O[] = new"
    after_write = "\n".join([*prefix, write, read, observe])
    before_write = "\n".join([*prefix, read, write, observe])
    assert _key(after_write) != _key(before_write)


def test_structural_key_preserves_shared_accumulator_order() -> None:
    """Updates to one reduction state retain their execution order."""
    prefix = ("x = load X[reduce]", "y = load Y[reduce]", "absolute = abs(x)", "exponential = exp(y)")
    absolute = "total <- add(total, absolute) over reduce"
    exponential = "total <- add(total, exponential) over reduce"
    one = _loop("reduce", 4, *prefix, absolute, exponential) + "\nO[] = total"
    two = _loop("reduce", 4, *prefix, exponential, absolute) + "\nO[] = total"
    assert _key(one) != _key(two)


def test_structural_key_preserves_when_accumulator_state_is_observed() -> None:
    """A state read remains after the update whose value it observes."""
    prefix = ("init f32 total = 0", "let first = 1.0", "let second = 2.0", "total <- add(total, first)")
    final_update = "total <- add(total, second)"
    observe = "O[] = total"
    after_both = "\n".join([*prefix, final_update, observe])
    between = "\n".join([*prefix, observe, final_update])
    assert _key(after_both) != _key(between)


def test_structural_key_distinguishes_reduction_seed_policy() -> None:
    """A reduction seed changes execution and must remain in exact identity."""

    def make(flags: str) -> str:
        return f"""
            for reduce in 0..4{flags}:
                x = load X[reduce]
                total <- add(total, x) over reduce
            O[] = total
            """

    assert _key(make("")) != _key(make(" seed=false"))


def test_structural_key_distinguishes_buffer_aliasing() -> None:
    """Two argument names and one argument used twice are different signatures."""
    separate = _pointwise_body(inputs=(("left", "X", "row"), ("right", "Y", "column")), op="multiply", args=("left", "right"))
    aliased = _pointwise_body(inputs=(("left", "X", "row"), ("right", "X", "column")), op="multiply", args=("left", "right"))
    assert _key(separate) != _key(aliased)


def _binary_body(op: str, args: tuple[str, str] = ("x", "y")) -> Body:
    """Two-operand body builder used by the op-clustering tests below."""
    return parse_body(
        f"""
        for a in 0..4:
            x = load X[a]
            y = load Y[a]
            z = {op}({", ".join(args)})
            O[a] = z
        """
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


def test_normalize_body_does_not_cluster_operations() -> None:
    add = _binary_body("add")
    subtract = _binary_body("subtract")

    assert _normal(add) != _normal(subtract)
    assert add.structural_key() == subtract.structural_key()


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
    """Identity is a fixed point of normalization: the normal form keys as the body it came from."""
    body = _matmul_body("X", "Y", "O")
    canonical = canonicalize_identity(body)
    assert canonicalize_identity(normalize_body(body)) == canonical
    assert set(canonical.arguments) == {"X", "Y", "O"}


def test_identity_coordinate_order_survives_axis_rename() -> None:
    """Renaming axes across lexical order must not re-key a saved kernel's measured evidence."""
    keys = set()
    for outer, inner in (("a9", "a10"), ("outer", "inner"), ("i", "j")):
        body = parse_body(
            f"""
            for {outer} in 0..40:
                for {inner} in 0..128:
                    v = load X[128 * {outer} + {inner}]
                    Y[{outer}, {inner}] = v
            """
        )
        canonical = canonicalize_identity(body)
        assert canonicalize_identity(normalize_body(body)).key == canonical.key
        normal = _normal(body)
        assert _normal(normal) == normal
        keys.add(canonical.key)
    assert len(keys) == 1


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
