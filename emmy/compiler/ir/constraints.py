"""Whether integer comparisons can all hold: the test a dependence question comes down to.

"Does this read touch a cell an earlier step wrote?" is a handful of comparisons between index expressions: the two
cells are the same, the step is earlier, each variable is inside its loop. If the comparisons cannot all hold there
is no dependence. The answer does not look at any extent's size, so it costs the same for 64 rows and for ``N``.

A comparison's sides are read as affine forms (``expression.affine_terms``); a term that is not a variable, such as
a quotient, is a name of its own. The test is Fourier-Motzkin elimination over the rationals with ``a < b`` taken as
``a + 1 <= b``. So it is exact when it says the comparisons cannot hold, and may say they can where no integers
make them: a caller that then keeps a dependence it did not need stays right.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from functools import lru_cache
from itertools import product

from emmy.compiler.ir.expression import BinaryExpression, Expression, Literal, affine_terms

#: ``sum(coefficient * name) + constant <= 0``.
Form = tuple[tuple[tuple[str, Fraction], ...], Fraction]

_FLIPPED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def _form(left: Expression, right: Expression, slack: int) -> Form:
    """``left - right + slack <= 0``."""
    sums: dict[str, Fraction] = {}
    constant = Fraction(slack)
    for sign, side in ((1, left), (-1, right)):
        terms, offset, _ = affine_terms(side)
        constant += sign * Fraction(offset)
        for coefficient, operand in terms:
            name = str(operand)
            sums[name] = sums.get(name, Fraction(0)) + sign * Fraction(coefficient)
    return tuple(sorted((name, value) for name, value in sums.items() if value)), constant


def alternatives(condition: Expression, *, holds: bool = True) -> list[tuple[Form, ...]]:
    """``condition`` (or, with ``holds`` false, its negation) as alternatives, each a set of forms that all hold."""
    if isinstance(condition, Literal):
        return [()] if bool(condition.value) == holds else []
    if not isinstance(condition, BinaryExpression):
        raise ValueError(f"not a comparison: {condition}")
    op = condition.op
    if op in ("&&", "||"):
        left, right = alternatives(condition.left, holds=holds), alternatives(condition.right, holds=holds)
        if (op == "&&") == holds:
            return [(*a, *b) for a, b in product(left, right)]
        return [*left, *right]
    op = op if holds else _FLIPPED[op]
    a, b = condition.left, condition.right
    if op == "<":
        return [(_form(a, b, 1),)]
    if op == "<=":
        return [(_form(a, b, 0),)]
    if op == ">":
        return [(_form(b, a, 1),)]
    if op == ">=":
        return [(_form(b, a, 0),)]
    if op == "==":
        return [(_form(a, b, 0), _form(b, a, 0))]
    if op == "!=":
        return [(_form(a, b, 1),), (_form(b, a, 1),)]
    raise ValueError(f"not a comparison: {condition}")


def negated(condition: Expression) -> Expression:
    """The comparison that holds exactly where ``condition`` does not."""
    if isinstance(condition, Literal):
        return Literal(not condition.value, "bool")
    if condition.op in ("&&", "||"):
        return BinaryExpression("||" if condition.op == "&&" else "&&", negated(condition.left), negated(condition.right))
    return BinaryExpression(_FLIPPED[condition.op], condition.left, condition.right)


def _solvable(forms: Iterable[Form]) -> bool:
    """Whether rationals make every form at most zero: eliminate one name at a time."""
    rows = [(dict(terms), constant) for terms, constant in set(forms)]
    while True:
        if any(not terms and constant > 0 for terms, constant in rows):
            return False
        names = {name for terms, _ in rows for name in terms}
        if not names:
            return True
        # The name whose elimination makes the fewest new rows.
        name = min(names, key=lambda n: sum(t.get(n, 0) > 0 for t, _ in rows) * sum(t.get(n, 0) < 0 for t, _ in rows))
        above = [row for row in rows if row[0].get(name, 0) > 0]
        below = [row for row in rows if row[0].get(name, 0) < 0]
        kept = [row for row in rows if not row[0].get(name, 0)]
        for (up, up_constant), (down, down_constant) in product(above, below):
            scale_up, scale_down = -down[name], up[name]
            terms = {}
            for other in {*up, *down} - {name}:
                value = scale_up * up.get(other, 0) + scale_down * down.get(other, 0)
                if value:
                    terms[other] = value
            kept.append((terms, scale_up * up_constant + scale_down * down_constant))
        rows = [(dict(terms), constant) for terms, constant in {(tuple(sorted(terms.items())), constant) for terms, constant in kept}]


@lru_cache(maxsize=1 << 16)
def _some(cases: tuple[tuple[tuple[Form, ...], ...], ...]) -> bool:
    return any(_solvable(form for case in choice for form in case) for choice in product(*cases))


def can_hold(*conditions: Expression) -> bool:
    """Whether every one of ``conditions`` can hold at once, for some integers."""
    return _some(tuple(sorted({tuple(alternatives(condition)) for condition in conditions})))


def always(condition: Expression, *given: Expression) -> bool:
    """Whether ``condition`` holds wherever all of ``given`` do."""
    cases = {tuple(alternatives(known)) for known in given}
    return not _some((tuple(alternatives(condition, holds=False)), *sorted(cases)))
