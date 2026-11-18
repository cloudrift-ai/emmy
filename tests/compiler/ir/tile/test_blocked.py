"""The block form of a fold (``ir/tile/blocked.py``): split, lift, isolate, break, checked on folds of different kinds.

Every fold is written once, as statements. Run a step at a time it is the reference. In blocks it must give the same
arrays at every block width, and what each block runs says what the four steps found in that fold."""

from collections import Counter

import numpy as np
import pytest

from emmy.compiler.ir.tile.blocked import MAX_PLUS, REAL, Fold, term

RNG = np.random.default_rng(0)
WIDTHS = (1, 2, 4, 8, 16, 64)


def _product(rows=12, columns=10, depth=32):
    a, b = RNG.standard_normal((rows, depth)), RNG.standard_normal((depth, columns))
    fold = Fold("k", (term("C[i, j] += A[i, k] * B[k, j]"),), {"C": np.zeros((rows, columns))}, {"A": a, "B": b})
    return fold, {"C": a @ b}


def _sum(steps=32, width=8, sr=REAL):
    x = RNG.standard_normal((steps, width))
    fold = Fold("t", (term("acc[j] += X[t, j]"),), {"acc": np.full(width, sr.zero)}, {"X": x}, sr)
    return fold, {"acc": x.sum(0) if sr is REAL else x.max(0)}


def _map(steps=32, width=8):
    x = RNG.standard_normal((steps, width))
    fold = Fold("t", (term("out[t, j] += F[t, j]"),), {"out": np.zeros((steps, width))}, {"F": x * x + x})
    return fold, {"out": x * x + x}


def _prefix_sum(steps=32, width=8):
    x = RNG.standard_normal((steps, width))
    terms = (term("out[t, j] += X[t, j]"), term("out[t, j] += out[t - 1, j]", "1 <= t"))
    return Fold("t", terms, {"out": np.zeros((steps, width))}, {"X": x}), {"out": x.cumsum(0)}


def _solve_in_place(n=32):
    """Forward substitution in place, a row a step: the step multiplies two reads of its state."""
    system = np.tril(RNG.standard_normal((n, n)) * 0.3, -1)
    fold = Fold("i", (term("S[i, j] += S[i, k] * S[k, j]", "k < i"),), {"S": -system})
    return fold, {"S": np.linalg.inv(np.eye(n) + system) - np.eye(n)}


def _solve(n=32):
    """The same solve with the system read from an input."""
    system = np.tril(RNG.standard_normal((n, n)) * 0.3, -1)
    fold = Fold("i", (term("S[i, j] += L[i, k] * S[k, j]", "k < i"),), {"S": -system}, {"L": -system})
    return fold, {"S": np.linalg.inv(np.eye(n) + system) - np.eye(n)}


def _delta_rule(steps=32, keys=24, values=20):
    """One head of the gated delta rule, one row a step, with its outputs: ``S <- g S + k (b (v - g k S))`` and
    ``out = q S`` after the update."""
    q, k = RNG.standard_normal((steps, keys)) / keys**0.5, RNG.standard_normal((steps, keys)) / keys**0.5
    v, beta, decay = RNG.standard_normal((steps, values)), RNG.uniform(0.2, 0.9, steps), RNG.uniform(0.7, 1.0, steps)
    seed = RNG.standard_normal((keys, values))
    s, out = seed, np.zeros((steps, values))
    for t in range(steps):
        s = s * decay[t]
        s = s + np.outer(k[t], (v[t] - k[t] @ s) * beta[t])
        out[t] = q[t] @ s
    met = (q * k).sum(1)
    inputs = {
        "K": k,
        "V": v,
        "GM1": decay - 1,
        "BK": beta[:, None] * k,
        "NBGK": -(beta * decay)[:, None] * k,
        "GQ": decay[:, None] * q,
        "BQK": beta * met,
        "NBGQK": -beta * decay * met,
    }
    terms = (
        term("S[i, j] += GM1[t] * S[i, j]"),
        term("S[i, j] += BK[t, i] * V[t, j]"),
        term("S[i, j] += NBGK[t, i] * K[t, k] * S[k, j]"),
        term("O[t, j] += GQ[t, k] * S[k, j]"),
        term("O[t, j] += BQK[t] * V[t, j]"),
        term("O[t, j] += NBGQK[t] * K[t, k] * S[k, j]"),
    )
    return Fold("t", terms, {"S": seed, "O": np.zeros((steps, values))}, inputs), {"S": s, "O": out}


def _dense_recurrence(steps=16, width=6, sr=REAL):
    """``S <- A S + b`` over the reals; ``S <- max(S, A S)`` over max-plus."""
    a, b = RNG.standard_normal((steps, width, width)) * 0.5, RNG.standard_normal((steps, width, 3))
    seed = RNG.standard_normal((width, 3))
    expected = seed
    if sr is REAL:
        terms = (term("S[i, j] += A[t, i, k] * S[k, j]"), term("S[i, j] += -S[i, j]"), term("S[i, j] += B[t, i, j]"))
        for t in range(steps):
            expected = a[t] @ expected + b[t]
    else:
        terms = (term("S[i, j] += A[t, i, k] * S[k, j]"),)
        for t in range(steps):
            expected = np.maximum(expected, (a[t][:, :, None] + expected[None]).max(1))
    return Fold("t", terms, {"S": seed}, {"A": a, "B": b}, sr), {"S": expected}


def _squares(steps=4, n=5):
    """Not linear in its state: every step multiplies two values the step before wrote."""
    seed = RNG.standard_normal((n, n)) * 0.3
    expected = seed
    for _ in range(steps):
        expected = expected + expected @ expected
    fold = Fold("t", (term("S[i, j] += S[i, k] * S[k, j] * ONE[t]"),), {"S": seed}, {"ONE": np.ones(steps)})
    return fold, {"S": expected}


CASES = {
    "product": _product,
    "sum": _sum,
    "max": lambda: _sum(sr=MAX_PLUS),
    "map": _map,
    "prefix_sum": _prefix_sum,
    "solve_in_place": _solve_in_place,
    "solve": _solve,
    "delta_rule": _delta_rule,
    "dense_recurrence": _dense_recurrence,
    "max_plus_recurrence": lambda: _dense_recurrence(sr=MAX_PLUS),
    "squares": _squares,
}


def _shapes(log, *kinds):
    return Counter((op.kind, op.m, op.n, op.k) for op in log if not kinds or op.kind in kinds)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("break_cycles", [True, False], ids=["broken", "serial"])
def test_the_block_form_computes_what_the_serial_walk_does(name, width, break_cycles):
    fold, expected = CASES[name]()
    serial, blocked = fold.serial(), fold.blocked(width, break_cycles=break_cycles)
    for array, value in expected.items():
        np.testing.assert_allclose(serial[array], value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(blocked[array], value, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("case", "op"),
    [
        (_product, ("bulk", 12, 10, 16)),
        (_sum, ("bulk", 1, 8, 16)),
        (lambda: _sum(sr=MAX_PLUS), ("bulk", 1, 8, 16)),
        (_map, ("bulk", 16, 8, 1)),
    ],
    ids=["product", "sum", "max", "map"],
)
def test_a_fold_with_no_cycle_is_one_operation_a_block(case, op):
    """A product is one product over a K chunk, a reduction one reduce of the block's rows, a map one scatter."""
    fold, _ = case()
    log = []
    fold.blocked(16, log)
    assert _shapes(log) == {op: 2}
    assert all(not group.cycle and part.lifted for group in fold.plan(np.arange(16, 32), 16) for part in group.parts)


@pytest.mark.parametrize("case", [_solve_in_place, _solve], ids=["in-place", "from-an-input"])
def test_a_triangular_solve_keeps_a_cycle_one_block_wide(case):
    """The rows of earlier blocks lift out whole, with no test inside a block, and are one product a block. What is
    left reads only the block's own rows, under the triangle that could not lift."""
    fold, _ = case(32)
    earlier, own = fold.plan(np.arange(16, 32), 16)
    (bulk,), (cycle,) = earlier.parts, own.parts
    assert not earlier.cycle and bulk.lifted and bulk.blocks[-1] == -1 and not any(bulk.live)
    assert own.cycle and not cycle.lifted and cycle.blocks[-1] == 0 and sum(cycle.live) == 1
    assert int(bulk.mask.sum()) == 16 * 32 * 16 and int(cycle.mask.sum()) == (16 * 15 // 2) * 32
    # The first block has no earlier rows: it is the cycle alone.
    (first,) = fold.plan(np.arange(0, 16), 16)
    assert first.cycle

    log = []
    fold.blocked(16, log)
    assert _shapes(log, "bulk") == {("bulk", 16, 32, 16): 1}
    # The cycle, broken: the block's own rows read through the system's own block, and a solve as wide. A block's
    # first step reads none of its rows, so fifteen steps are in it. Row 0 of a strictly lower system is zero, and
    # the first block's product is one row narrower for it.
    assert _shapes(log, "read", "solve") == {("read", 15, 32, 15): 1, ("read", 15, 32, 14): 1, ("solve", 15, 32, 15): 2}
    assert not _shapes(log, "serial", "gram", "write"), "the rows are written as they are: gathers, not products"


def test_a_prefix_sum_carries_one_row_between_blocks():
    fold, _ = _prefix_sum(32, 8)
    groups = fold.plan(np.arange(16, 32), 16)
    kinds = [(group.cycle, part.at, part.blocks) for group in groups for part in group.parts]
    # The inputs and the row carried in from the block before run at once; the chain inside the block is the cycle.
    assert kinds == [(False, 0, ()), (False, 1, (-1,)), (True, 1, (0,))]


def test_the_delta_rule_is_all_cycle_and_breaks_into_its_chunk_rule():
    steps, keys, values = 32, 24, 20
    fold, _ = _delta_rule(steps, keys, values)
    cycle, alone = fold.plan(np.arange(16, 32), 16)
    # Everything that reads or writes the state is one cycle; the outputs' own inputs stand outside it.
    assert cycle.cycle and sorted(part.at for part in cycle.parts) == [0, 1, 2, 3, 5]
    assert not alone.cycle and [part.at for part in alone.parts] == [4]
    log = []
    fold.blocked(16, log)
    shapes = _shapes(log)
    # A block: the state read once through its keys and queries, their products with the keys the block writes
    # through, a small solve, and the state written once through the keys.
    assert shapes[("read", 48, values, keys)] == 2
    assert shapes[("gram", 48, 32, keys)] == 2
    assert shapes[("write", keys, values, 32)] == 2
    assert not _shapes(log, "serial")


def test_a_cycle_that_is_not_linear_stays_serial_and_stays_right():
    fold, expected = _squares()
    log = []
    np.testing.assert_allclose(fold.blocked(2, log)["S"], expected["S"], rtol=1e-9)
    assert {op.kind for op in log} == {"serial"}
