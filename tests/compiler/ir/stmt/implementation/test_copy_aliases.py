"""Copy aliases through value placement, one inner step of normalization."""

from emmy.emmy_ir import place_values

from emmy.compiler.ir.stmt import parse_body


def test_typed_copy_is_a_cast_not_an_alias():
    cast = parse_body("f32 wide = copy(narrow)")
    assert place_values(cast) == cast


def test_copy_aliases_do_not_leak_between_sibling_scopes():
    body = parse_body(
        """
        for i in 0..4:
            x = load A[i]
            y = copy(x)
            OA[i] = y
        for i in 0..4:
            x = load B[i]
            y = exp(x)
            OB[i] = y
        """
    )

    out = place_values(body)

    assert str(out[1].body[1]) == "y = exp(x)"
