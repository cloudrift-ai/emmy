"""The hand-spelled term builders spell what the total lift forms from Loop IR — so a fixture built
through ``tests.compiler.terms`` exercises the same term the compiler would schedule."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop, scan_from_loop
from tests.compiler.terms import contraction, projection, slab


def test_a_contraction_is_the_lifted_matmul_loop() -> None:
    (loop,) = parse_body(
        """
        for k in 0..16:
            a = load x[m, k]
            b = load w[k, n]
            acc__v = multiply(a, b)
            acc <- add(acc, acc__v) over k
        """
    )
    built = contraction("k", parse_body("a = load x[m, k]")[0], (parse_body("b = load w[k, n]")[0], "acc"))
    assert built.canonical() == fold_from_loop(loop).canonical()
    view = built.as_contraction()
    assert (view.axis, view.left, view.right, view.product.name, view.plus.name) == ("k", "m", "n", "multiply", "add")


def test_several_own_axes_on_both_sides_still_orient_a_pair() -> None:
    """A chunked row (``64*m1 + m2``) against a packed channel group (``8*n1 + n2``) is an ordinary
    contraction: each side's own axes stay its role, and the tile picks one of them per side. The
    Qwen3.8 AWQ down projection has this shape; refusing it left the piece off the tensor cores."""
    a = parse_body("a = load x[64 * m1 + m2, k]")[0]
    b = parse_body("b = load w[k, 8 * n1 + n2]")[0]
    view = contraction("k", a, (b, "acc")).as_contraction()
    assert view is not None and (view.left_axes, view.right_axes) == ({"m1", "m2"}, {"n1", "n2"})


def test_two_channels_over_one_a_are_one_contraction() -> None:
    a = slab("a", "x", "m", "k")
    fused = contraction(Axis("k", 16), a, (slab("g", "Wg", "k", "n"), "acc_g"), (slab("u", "Wu", "k", "n"), "acc_u"))
    assert fused.operands[0] is a and len(fused.operands) == 3
    assert fused.as_contraction() is not None and fused.combine.results == ("acc_g", "acc_u")
    assert fused.as_contraction().plus.name == "add" and fused.init == (0.0, 0.0)


def test_a_projection_exposes_its_last_definition_or_passes_its_operand_through() -> None:
    stat = contraction("k", slab("a", "x", "m", "k"), (slab("b", "w", "k", "n"), "acc"))
    cell = projection(
        (stat,),
        """
        r = rsqrt(acc)
        o = multiply(r, r)
        """,
    )
    assert cell.axis is None and cell.exposes == ("o",) and cell.free_axes == {"m", "n"}
    assert projection((stat,)).exposes == ("acc",)


def test_a_load_indexed_by_a_loaded_value_is_a_gather_not_a_slab() -> None:
    # The index reads ``i``, a value the step loads — not a coordinate. A slab would declare it as
    # one, and the closed program would then ask the kernel for an extent no axis table can hold.
    (loop,) = parse_body(
        """
        for k in 0..16:
            a = load x[m, k]
            i = load idx[k]
            b = load w[(int)i, n]
            acc__v = multiply(a, b)
            acc <- add(acc, acc__v) over k
        """
    )
    fold, _ = scan_from_loop(loop, axes=(Axis("m", 8), Axis("n", 32)))
    assert fold.free_axes == {"m", "n"}
