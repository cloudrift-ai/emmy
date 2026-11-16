"""Twist recipes — the streaming carriers a two-pass reduce pair fuses into.

A *dependent* reduce ``F`` reads an earlier reduce ``G`` over the same stream: ``m = max_k s_k``
then ``l = Σ_k exp(s_k − m)``. After the lift that dependency is in the tree — ``G`` is an operand
of ``F`` — and a recipe is what says the pair fuses into ONE fold with state ``(m, l)``.

A recipe is a TWISTED MONOID stated as its mathematics — transport of structure: a componentwise
monoid ``base`` (one ⊕ per state, with ``lift`` the per-element contribution to each) conjugated
by a bijection ``psi`` on the carrier, ``x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))``, associative because the base
is. Online softmax is ``(max, Σeˢ, Σeˢv)`` under ``ψ(m, D, O) = (m, De⁻ᵐ, Oe⁻ᵐ)``; Welford's variance
is ``(Σx, Σ1, Σx, Σx²)`` under ``ψ(S, n, T, W) = (S, n, T/n, W − T²/n)``. What the tree is matched
on and what the kernel runs are DATA beside that definition, because stability is not preserved by
conjugation: the ``channels`` say which per-element maps a dependent's lift must spell (over ROLES,
never over any term's names) and what each state is at the singleton, and the fused ⊕ is spelled
in its numerically stable form (softmax's as a pivot advance and a per-channel rescale, so one
recipe serves any channel count; Welford's as one lambda). The definition certifies the data: the
program is the conjugate of the base on every state pair, the seeds are the base identities under
ψ⁻¹, the injections are the lift seen through ψ (``tests/compiler/ir/pure/test_twist.py``).

The one generic algorithm that applies any recipe is :meth:`Fold.fuse`, which finds the pivot
among ``F``'s own operands: matching is alpha-invariant by construction — ``F``'s lift binds its
operands positionally, so the pivot's state is the param bound to ``G``; the score is whatever
sub-cone of ``F``'s lift is alpha-equal to ``G``'s own per-element map, operand for operand; and
what remains of ``F``'s lift with that cone cut out, its params in role order, compares with a
channel's ``pattern`` by canonical form. A click gives the role-to-name map, and the recipe is
instantiated by renaming — no algebra engine, no op-name table: either a recipe clicks or it does
not.
"""

from __future__ import annotations

import emmy.emmy_ir as _ir
from emmy.compiler.ir.expression import Var, apply
from emmy.compiler.ir.pure._fields import MISSING, declare
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt import Assign, Body, Let, Statement

# The types are the ``emmy-pure`` crate's: a ``Channel`` is one state of the fused carrier beyond the pivot's (its
# ``injection`` at the singleton over ``(score, *extras)``; with a ``pattern`` the per-element map over
# ``(score, pivot, *extras)`` a dependent fold's lift must spell, without one a state the two-pass form never had,
# suffixed ``name`` and seeded ``init``). A ``Recipe`` is the twisted monoid as data — ``base`` the componentwise ⊕ of
# every state, ``lift`` its per-element contribution over ``(score, *extras)``, ``psi`` / ``psi_inv`` the conjugation,
# the fused ⊕ as ``advance`` / ``scale`` at an open channel count or one ``combine`` — and ``program(states)`` the
# fused ⊕ over a carrier's state names. A ``Twist`` is a recipe instantiated on a term: ``channels`` says which recipe
# channel each carried state past the pivot is; ``program`` and ``inject`` derive the two halves a fold does not store.
# The recipes themselves are defined here, in Python.
Channel = _ir.Channel
Recipe = _ir.Recipe
Twist = _ir.Twist
_LAMBDA_OR_NONE = Lambda | None
declare(Channel, (("injection", Lambda, MISSING), ("pattern", _LAMBDA_OR_NONE, None), ("name", str, ""), ("init", float, 0.0)))
declare(
    Recipe,
    (
        ("name", str, MISSING),
        ("base", tuple[str, ...], MISSING),
        ("lift", Lambda, MISSING),
        ("psi", Lambda, MISSING),
        ("psi_inv", Lambda, MISSING),
        ("channels", tuple[Channel, ...], MISSING),
        ("advance", _LAMBDA_OR_NONE, None),
        ("scale", _LAMBDA_OR_NONE, None),
        ("combine", _LAMBDA_OR_NONE, None),
    ),
)
declare(Twist, (("recipe", Recipe, MISSING), ("channels", tuple[int, ...], MISSING)))


def _lam(params: tuple[str, ...], body: tuple[Statement, ...], *results: str) -> Lambda:
    return Lambda(params=params, body=Body(body), results=results)


def _const(name: str, value: float) -> Lambda:
    return _lam(("s",), (Let(name=name, value=value),), name)


SOFTMAX = Recipe(
    name="softmax",
    base=("maximum", "add", "add"),
    lift=_lam(("s", "v"), (Assign("e", apply("exp", "s")), Assign("ev", apply("multiply", "e", "v"))), "s", "e", "ev"),
    psi=_lam(
        ("m", "D", "O"),
        (
            Assign("nm", apply("negative", "m")),
            Assign("f", apply("exp", "nm")),
            Assign("d", apply("multiply", "D", "f")),
            Assign("o", apply("multiply", "O", "f")),
        ),
        "m",
        "d",
        "o",
    ),
    psi_inv=_lam(
        ("m", "d", "o"),
        (Assign("f", apply("exp", "m")), Assign("D", apply("multiply", "d", "f")), Assign("O", apply("multiply", "o", "f"))),
        "m",
        "D",
        "O",
    ),
    channels=(
        # The denominator: one weight per element, ``exp(s − m)``; at the singleton the pivot is
        # the score, so the channel injects ``1``.
        Channel(
            pattern=_lam(("s", "g"), (Assign("w", apply("exp", apply("subtract", "s", "g"))),), "w"),
            injection=_lam(("s",), (Let(name="one", value=1.0),), "one"),
        ),
        # An expectation: the weight times a streamed value; it injects the value itself.
        Channel(
            pattern=_lam(
                ("s", "g", "v"),
                (Assign("p", apply("multiply", apply("exp", apply("subtract", "s", "g")), "v")),),
                "p",
            ),
            injection=_lam(("s", "v"), (), "v"),
        ),
    ),
    # The pivot advances to the larger of the pair, and each side's factor is ``exp`` of its distance
    # below the new pivot — never positive, so the program cannot overflow.
    advance=_lam(
        ("g", "g_o"),
        (
            Assign("gn", apply("maximum", "g", "g_o")),
            Assign("dg", apply("subtract", "g", "gn")),
            Assign("alpha", apply("exp", "dg")),
            Assign("dg_o", apply("subtract", "g_o", "gn")),
            Assign("beta", apply("exp", "dg_o")),
        ),
        "gn",
        "alpha",
        "beta",
    ),
    # A side moves to the advanced pivot by its factor; the channels' ⊕ (a sum) then joins the two.
    scale=_lam(("s", "f"), (Assign("sa", apply("multiply", "s", "f")),), "sa"),
)

# Welford's variance: ``mean = Σ_k x_k / N`` then ``Σ_k (x_k − mean)²``, fused into the carrier
# ``(sum, count, mean, M2)`` — the running mean beside the sum so the seed ``(0, 0, 0, 0)`` is a true
# identity (a mean of nothing is any value, a sum over nothing divided by its count is not), the sum
# kept so the pivot's readers keep their value. Chan's merge: ``δ = mean′ − mean``, ``mean +=
# δ·n′/n``, ``M2 += M2′ + δ²·n·n′/(n + n′)``.
WELFORD = Recipe(
    name="welford",
    base=("add", "add", "add", "add"),
    lift=_lam(("s", "c"), (Let(name="one", value=1.0), Assign("sq", apply("multiply", "s", "s"))), "s", "one", "s", "sq"),
    psi=_lam(
        ("S", "n", "T", "W"),
        (Assign("mu", apply("divide", "T", "n")), Assign("Tmu", apply("multiply", "T", "mu")), Assign("M2", apply("subtract", "W", "Tmu"))),
        "S",
        "n",
        "mu",
        "M2",
    ),
    psi_inv=_lam(
        ("S", "n", "mu", "M2"),
        (Assign("T", apply("multiply", "n", "mu")), Assign("Tmu", apply("multiply", "T", "mu")), Assign("W", apply("add", "M2", "Tmu"))),
        "S",
        "n",
        "T",
        "W",
    ),
    channels=(
        # The count and the running mean: states the two-pass form never had, one element counting
        # one and averaging to itself.
        Channel(injection=_const("one", 1.0), name="n"),
        Channel(injection=_lam(("s",), (), "s"), name="mean"),
        # The squared deviation from the mean, the mean being the pivot scaled by ``1/N`` — a scalar
        # the tree reads like any operand; a single element deviates from its own mean by nothing.
        Channel(
            pattern=_lam(
                ("s", "g", "c"),
                (Assign("d", apply("subtract", "s", apply("multiply", "g", "c"))), Assign("sq", apply("multiply", "d", "d"))),
                "sq",
            ),
            injection=_const("zero", 0.0),
        ),
    ),
    combine=_lam(
        ("g", "n", "m", "q", "g__o", "n__o", "m__o", "q__o"),
        (
            Assign("tn", apply("add", "n", "n__o")),
            Assign("d", apply("subtract", "m__o", "m")),
            Assign("w", apply("divide", "n__o", "tn")),
            Assign("dw", apply("multiply", "d", "w")),
            Assign("nn", apply("multiply", "n", "n__o")),
            Assign("r", apply("divide", "nn", "tn")),
            Assign("dd", apply("multiply", "d", "d")),
            Assign("corr", apply("multiply", "dd", "r")),
            Assign("q1", apply("add", "q", "q__o")),
            Assign("g", apply("add", "g", "g__o")),
            Assign("m", apply("add", "m", "dw")),
            Assign("q", apply("add", "q1", "corr")),
            Assign("n", Var("tn")),
        ),
        "g",
        "n",
        "m",
        "q",
    ),
)

RECIPES = (SOFTMAX, WELFORD)

__all__ = ["RECIPES", "SOFTMAX", "WELFORD", "Channel", "Recipe", "Twist"]
