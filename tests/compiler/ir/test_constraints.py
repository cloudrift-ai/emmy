"""Whether integer comparisons can all hold (``ir/constraints.py``): the test a dependence question comes down to."""

import pytest

from emmy.compiler.ir.constraints import always, can_hold
from emmy.compiler.ir.expression import parse


def _all(*texts: str) -> tuple:
    return tuple(parse(text) for text in texts)


def test_a_read_of_an_earlier_block_meets_no_row_the_block_writes():
    """The solve: a step of block ``b`` writes row ``i``; it read row ``k < i``; an earlier step of the block wrote
    row ``w``. With ``k`` in an earlier block the two rows cannot be one, at any block size."""
    block = _all("16 * b <= i", "i < 16 * b + 16", "16 * b <= w", "w < i", "k == w", "k < i")
    assert not can_hold(*block, parse("k < 16 * b"))
    assert can_hold(*block, parse("16 * b <= k"))
    # The same with the block's width and the system's height left as names.
    named = _all("lo <= i", "i < lo + width", "lo <= w", "w < i", "k == w", "k < i", "i < n")
    assert not can_hold(*named, parse("k < lo")) and can_hold(*named, parse("lo <= k"))


def test_a_test_the_block_settles_is_always_true_there():
    """``k < i`` holds for every row of an earlier block, and for no row of a later one."""
    inside = _all("16 * b <= i", "i < 16 * b + 16")
    assert always(parse("k < i"), *inside, parse("k < 16 * b"))
    assert not always(parse("k < i"), *inside, parse("16 * b <= k"), parse("k < 16 * b + 16"))
    assert not can_hold(parse("k < i"), *inside, parse("16 * b + 16 <= k"))


@pytest.mark.parametrize(
    ("conditions", "holds"),
    [
        (("a < b", "b < a"), False),
        (("a < b", "b < c", "c < a + 2"), False),  # integers: a + 1 <= b, b + 1 <= c, so c >= a + 2
        (("a < b", "b < c", "c < a + 3"), True),
        (("a == b + 1", "a <= b"), False),
        (("a != b", "a <= b", "b <= a"), False),
        (("a < 4 || 8 <= a", "4 <= a", "a < 8"), False),
        (("a < b && c < d", "d < c || b < a + 1"), False),
        (("x / 16 < y", "y < x / 16"), False),  # a quotient is a name of its own
        (("true",), True),
        (("false",), False),
    ],
)
def test_comparisons_that_can_and_cannot_hold(conditions, holds):
    assert can_hold(*_all(*conditions)) is holds


def test_what_follows_and_what_does_not():
    assert always(parse("a < c"), *_all("a < b", "b < c"))
    assert not always(parse("a < c"), *_all("a < b"))
    assert always(parse("a + 16 < b + 17"), parse("a < b + 1")) and always(parse("a < b + 1"), parse("a + 16 < b + 17"))
