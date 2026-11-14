""":class:`Lambda` — formation (the ``Statement.pure`` gate, results-defined, closedness), α-invariance
by canonical renumbering, and the componentwise program a plain fold's combine is."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.twist import SOFTMAX
from emmy.compiler.ir.stmt import Accum, Body, Init, Loop, Write, parse_body

# --- Lambda formation: the Statement.pure gate + results-defined ------------------------------------- #


def test_lambda_accepts_pure_stmts_and_defines_results() -> None:
    lam = Lambda(params=("k", "x"), body=parse_body("x2 = multiply(x, x)"), results=("x2",))
    assert lam.results == ("x2",)
    # CLOSED: every name the body reads is a param or one of its own defs, so there is no
    # residual for the former to append and the two spellings coincide.
    assert lam.defined == frozenset({"k", "x", "x2"})
    assert Lambda.closing(("k", "x"), lam.body, lam.results) == lam


def test_lambda_post_init_canonicalizes_body_order() -> None:
    left = "left = load x[m, k]"
    right = "right = load w[n, k]"
    product = "product = multiply(left, right)"

    first = Lambda.closing(("k",), parse_body("\n".join((left, right, product))), ("product",))
    second = Lambda.closing(("k",), parse_body("\n".join((right, left, product))), ("product",))

    assert first == second


def test_lambda_body_order_does_not_follow_the_written_order_where_values_tie() -> None:
    """Two products of one shape tie on the old token order, which kept them as written; the crate's order ranks them
    by what they compute, so the two spellings are one lambda up to renaming."""
    loads = "a = load x[k]\nb = load w[k]\nc = load y[k]"
    ab, bc = "ab = multiply(a, b)", "bc = multiply(b, c)"
    first = Lambda.closing(("k",), parse_body("\n".join((loads, ab, bc))), ("ab", "bc"))
    second = Lambda.closing(("k",), parse_body("\n".join((loads, bc, ab))), ("ab", "bc"))
    assert first.body == second.body


def test_lambda_body_reads_a_param_it_rebinds_ahead_of_the_rebinding() -> None:
    lam = Lambda(params=("m", "d"), body=parse_body("e = subtract(d, m)\nm = add(m, e)"), results=("m",))
    assert [stmt.name for stmt in lam.body] == ["e", "m"]


def test_lambda_refuses_a_body_whose_statements_wait_on_each_other() -> None:
    with pytest.raises(ValueError, match="wait on each other"):
        Lambda(params=("x",), body=parse_body("a = add(b, x)\nb = add(a, x)"), results=("b",))


def test_lambda_orders_the_definition_read_by_a_predicate_first() -> None:
    predicate = """
        predicate = select:
            x when column
            y when true
        """
    value = """
        value = select:
            x when predicate
            y when true
        """

    forward = Lambda.closing((), parse_body(predicate + value), ("value",))
    reversed_body = Lambda.closing((), parse_body(value + predicate), ("value",))

    assert forward == reversed_body
    assert forward.body == parse_body(predicate + value)


@pytest.mark.parametrize(
    "stmt",
    [
        Accum(name="acc", value="x", op="add"),
        Write(output="out", index=(Var("m"),), values=("x",)),
        Init(name="acc", identity=0.0, dtype="float32"),
        Loop(axis=Axis("k", 4), body=Body(())),
    ],
    ids=lambda s: type(s).__name__,
)
def test_lambda_rejects_impure_statement_types(stmt) -> None:
    """The conservative ``Statement.pure`` default: effectful / scope-bound statement types never enter a stored
    lambda body — the purity-erosion guard."""
    with pytest.raises(ValueError, match="pure"):
        Lambda(params=("x",), body=Body((stmt,)), results=("x",))


def test_lambda_rejects_undefined_results() -> None:
    with pytest.raises(ValueError, match="not defined"):
        Lambda(params=("x",), body=Body(()), results=("y",))


def test_lambda_allows_param_results_and_constant_results() -> None:
    """ι is spelled in the lift: softmax's singleton is ``(x, 1)`` — the constant component is a
    ``Let`` def, a name like any other result."""
    lam = Lambda(params=("k", "x"), body=parse_body("let one = 1.0"), results=("x", "one"))
    assert lam.results == ("x", "one") and lam.defined == frozenset({"k", "x", "one"})


def test_lambda_refuses_an_open_body_and_closing_binds_the_residual() -> None:
    """A term carries no free names. What the retired ``free_names`` reported as a contextual read
    is now a FORMATION ERROR, and :meth:`Lambda.closing` is the former that turns it into a
    trailing param — trailing, never interleaved, so the operand prefix keeps its positions."""
    body = parse_body("y = add(x, outer)")

    with pytest.raises(ValueError, match=r"reads \['outer'\]"):
        Lambda(params=("x",), body=body, results=("y",))

    lam = Lambda.closing(("x",), body, ("y",))
    assert lam.params == ("x", "outer")
    assert lam.defined == frozenset({"x", "outer", "y"})
    assert lam.results == ("y",)


def test_select_predicate_is_bound_and_renamed_with_the_lambda() -> None:
    body = parse_body(
        """
        v = select:
            x when column == 0.0
            y when 1.0
        """
    )

    with pytest.raises(ValueError, match=r"reads \['column'\]"):
        Lambda(params=("x", "y"), body=body, results=("v",))

    lam = Lambda.closing(("x", "y"), body, ("v",))
    renamed = lam.rename({"column": "j"})
    assert renamed.params == ("x", "y", "j")
    assert renamed.body[0].branches[0].select.free_vars() == {"j"}
    assert lam.alpha_eq(renamed)


# --- α-invariance: canonical renumbering -------------------------------------------------------- #


def test_cone_is_one_definition_closed_over_what_it_reads() -> None:
    """The cone of a result is the sub-lambda defining it — its dependence within the body, the
    reads it takes from the params as its own params; a param's cone is the identity on it."""
    fn = Lambda.closing(
        ("k", "a", "b"),
        parse_body(
            """
            s = multiply(a, b)
            d = subtract(s, k)
            w = exp(d)
            """
        ),
        ("w",),
    )
    score = fn.cone("s")
    assert score.params == ("a", "b") and [stmt.name for stmt in score.body] == ["s"] and score.results == ("s",)
    assert fn.cone("a") == Lambda(params=("a",), body=Body(()), results=("a",))
    assert set(fn.cone("w").params) == {"a", "b", "k"} and len(fn.cone("w").body) == 3


def test_cone_reads_a_param_as_what_came_in_not_as_the_bodys_write_back() -> None:
    """A stored combine is not SSA in its states: it takes ``m`` in and writes ``m`` back on the way
    out to spell its result. The cone of the incoming side's factor must read the ORIGINAL ``m`` —
    resolving that read forward to the write-back drags the pivot's own advance into the weight and
    leaves the kernel reading an uninitialized accumulator."""
    combine = SOFTMAX.program(("m", "l"))
    beta = combine.cone("m__o__beta")
    assert beta.params == ("m", "m__o")
    assert [stmt.name for stmt in beta.body] == ["m__o__gn", "m__o__dg_o", "m__o__beta"]
    assert all(not isinstance(stmt.value, Var) for stmt in beta.body)
    # The channel's rescale reads its OWN state the same way — as the param, never as its result.
    assert "l" in combine.cone("m__o__l_sa").params


def test_rename_maps_params_body_and_results_in_lockstep() -> None:
    fn = Lambda(
        params=("m", "m__o"),
        body=parse_body(
            """
            t = maximum(m, m__o)
            m = t
            """
        ),
        results=("m",),
    )
    renamed = fn.rename({"m": "acc", "m__o": "acc__o", "t": "acc__t"})
    assert renamed.params == ("acc", "acc__o") and renamed.results == ("acc",)
    assert [(stmt.name, str(stmt.value)) for stmt in renamed.body] == [("acc__t", "maximum(acc, acc__o)"), ("acc", "acc__t")]
    assert fn.rename(lambda name: name) == fn
    assert fn.alpha_eq(renamed)


def test_rename_that_merges_names_forms_the_lambda_it_spells() -> None:
    """A rename that keeps every name apart skips the formation checks; one that merges two names is built like any
    lambda, and both agree with building the renamed parts by hand."""
    fn = Lambda.closing(("x", "y"), parse_body("t = add(x, y)\nu = multiply(t, x)"), ("u",))
    for names in ({"x": "a", "t": "b"}, {"t": "x"}, lambda name: f"{name}_1"):
        renamed = fn.rename(names)
        spell = names.get if isinstance(names, dict) else None
        rn = (lambda n: spell(n, n)) if spell else names
        expected = Lambda(
            params=tuple(rn(p) for p in fn.params),
            body=parse_body("\n".join(str(stmt.rename({n: rn(n) for n in stmt._names()})) for stmt in fn.body)),
            results=tuple(rn(r) for r in fn.results),
        )
        assert renamed == expected


def test_alpha_equality_under_bound_renaming() -> None:
    a = Lambda(params=("k", "x"), body=parse_body("t = multiply(x, x)"), results=("t",))
    b = Lambda(params=("j", "v"), body=parse_body("sq = multiply(v, v)"), results=("sq",))
    assert a.alpha_eq(b)
    assert a.canonical() == b.canonical()
    assert hash(a.canonical()) == hash(b.canonical())


def test_alpha_inequality_on_structure_and_residual_arity() -> None:
    sq = Lambda(params=("x",), body=parse_body("t = multiply(x, x)"), results=("t",))
    dbl = Lambda(params=("x",), body=parse_body("t = add(x, x)"), results=("t",))
    closed = Lambda.closing(("x",), parse_body("t = multiply(x, w)"), ("t",))
    renamed = Lambda.closing(("x",), parse_body("t = multiply(x, z)"), ("t",))
    assert not sq.alpha_eq(dbl)
    # The residual read is BOUND, so it changes the arity — ``x·w`` is a two-param function.
    assert closed.params == ("x", "w") and not sq.alpha_eq(closed)
    # And being bound, it renumbers like any other param: its spelling no longer has to match.
    assert closed.alpha_eq(renamed)
    assert sq.canonical().canonical() == sq.canonical()  # idempotent


# --- the componentwise program: a plain fold's ⊕, and the shape read back off any combine -------- #


def test_componentwise_program_reads_back_its_ops() -> None:
    """``Lambda.componentwise`` spells one independent ⊕ per state, ``S × S → S`` with the second
    operand ``<n>__o``, and ``components`` reads the op vector back — in either argument order for
    a commutative ⊕, since a rename can reorder the sorted arguments."""
    combine = Lambda.componentwise(("add", "maximum"), ("acc", "mx"))
    assert combine.params == ("acc", "mx", "acc__o", "mx__o") and combine.results == ("acc", "mx")
    assert [op.name for op in combine.components()] == ["add", "maximum"]
    assert [op.name for op in combine.rename({"acc": "z"}).components()] == ["add", "maximum"]


def test_a_twisted_program_has_no_components() -> None:
    """A cross-component read — a recipe's rescale — fails the componentwise shape: the
    planar-vs-twisted reading every partition legality question asks, derived, never stored."""
    assert SOFTMAX.program(("m", "l")).components() is None
