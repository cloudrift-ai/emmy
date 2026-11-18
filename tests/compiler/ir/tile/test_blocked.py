"""The block form of a fold (``ir/tile/blocked.py``): one rule, checked on folds of different kinds.

Every case writes its step once. Run on numbers it is the serial walk; read as a map it is composed into blocks. The
two must agree at every block size, and the products the block form runs say what the rule found in each fold."""

from collections import Counter

import numpy as np
import pytest

from emmy.compiler.ir.tile.blocked import BOOL, MAX_PLUS, REAL, Fold, NotAffine, compose_all

RNG = np.random.default_rng(0)
BLOCKS = (1, 2, 4, 16, 64)


def _map(steps=32, width=8):
    """Each step writes one row from its inputs: no step reads another."""
    x = RNG.standard_normal((steps, width))

    def step(t, state, inputs):
        state.out[t] = inputs.x[[t]] * inputs.x[[t]] + inputs.x[[t]]

    return Fold(steps, step, {}, {"x": x}, {"out": (steps, width)}), {"out": x * x + x}


def _sum(steps=32, width=8, sr=REAL):
    x = RNG.standard_normal((steps, width))

    def step(t, state, inputs):
        state.acc[0] = state.acc[0] + inputs.x[[t]]

    seed = np.full((1, width), sr.zero)
    return Fold(steps, step, {"acc": seed}, {"x": x}, sr=sr), {"acc": (x.sum(0) if sr is REAL else x.max(0))[None]}


def _product(rows=12, columns=10, depth=32):
    """A matrix product as a fold over its reduced axis: a step adds one column times one row."""
    a, b = RNG.standard_normal((rows, depth)), RNG.standard_normal((depth, columns))

    def step(k, state, inputs):
        state.c[:] = state.c[:] + inputs.a[:, [k]] @ inputs.b[[k]]

    return Fold(depth, step, {"c": np.zeros((rows, columns))}, {"a": a, "b": b}), {"c": a @ b}


def _prefix_sum(steps=32, width=8):
    """Every step's value is kept, and each reads the one before it."""
    x = RNG.standard_normal((steps, width))

    def step(t, state, inputs):
        state.out[t] = state.out[t - 1] + inputs.x[[t]] if t else inputs.x[[t]]

    return Fold(steps, step, {}, {"x": x}, {"out": (steps, width)}), {"out": x.cumsum(0)}


def _dense_recurrence(steps=16, width=6, sr=REAL):
    """``S <- A S + b`` with a dense ``A``: each step reads and rewrites the whole state."""
    a = RNG.standard_normal((steps, width, width)) * 0.5
    b, seed = RNG.standard_normal((steps, width, 3)), RNG.standard_normal((width, 3))

    def step(t, state, inputs):
        state.s[:] = inputs.a[t] @ state.s[:] + inputs.b[t]

    expected = seed
    for t in range(steps):
        if sr is REAL:
            expected = a[t] @ expected + b[t]
        else:
            expected = np.maximum((a[t][:, :, None] + expected[None]).max(1), b[t])
    return Fold(steps, step, {"s": seed}, {"a": a, "b": b}, sr=sr), {"s": expected}


def _triangular_inverse(n=32):
    """The inverse of a unit lower-triangular matrix by forward substitution in place, as Transformers writes it:
    a step finishes one row from the rows above it, and multiplies two reads of its state to do so."""
    system = np.tril(RNG.standard_normal((n, n)) * 0.3, -1)

    def step(t, state, inputs):
        row = t + 1
        state.inv[row] = state.inv[row] + state.inv[row, :row] @ state.inv[:row]

    return Fold(n - 1, step, {"inv": -system}, {}), {"inv": np.linalg.inv(np.eye(n) + system) - np.eye(n)}


def _delta_rule(steps=32, keys=24, values=20):
    """One head of the gated delta rule, one row a step, as Transformers' recurrent rule writes it."""
    q, k = RNG.standard_normal((steps, keys)) / keys**0.5, RNG.standard_normal((steps, keys)) / keys**0.5
    v, beta, decay = RNG.standard_normal((steps, values)), RNG.uniform(0.2, 0.9, steps), RNG.uniform(0.7, 1.0, steps)

    def step(t, state, inputs):
        s = inputs.decay[t] * state.s[:]
        delta = inputs.beta[t] * (inputs.v[[t]] - inputs.k[[t]] @ s)
        s = s + inputs.k[[t]].T @ delta
        state.s[:] = s
        state.out[t] = inputs.q[[t]] @ s

    seed = RNG.standard_normal((keys, values))
    s, out = seed, np.zeros((steps, values))
    for t in range(steps):
        s = s * decay[t]
        s = s + np.outer(k[t], (v[t] - k[t] @ s) * beta[t])
        out[t] = q[t] @ s
    inputs = {"q": q, "k": k, "v": v, "beta": beta, "decay": decay}
    return Fold(steps, step, {"s": seed}, inputs, {"out": (steps, values)}), {"s": s, "out": out}


CASES = {
    "map": _map,
    "sum": _sum,
    "max": lambda: _sum(sr=MAX_PLUS),
    "product": _product,
    "prefix_sum": _prefix_sum,
    "dense_recurrence": _dense_recurrence,
    "max_plus_recurrence": lambda: _dense_recurrence(sr=MAX_PLUS),
    "triangular_inverse": _triangular_inverse,
    "delta_rule": _delta_rule,
}


def _shapes(log, *kinds):
    return Counter((p.kind, p.m, p.n, p.k) for p in log if not kinds or p.kind in kinds)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("block", BLOCKS)
def test_the_block_form_computes_what_the_serial_walk_does(name, block):
    fold, expected = CASES[name]()
    serial, blocked = fold.serial(), fold.blocked(block)
    for array, value in expected.items():
        np.testing.assert_allclose(serial[array], value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(blocked[array], value, rtol=1e-9, atol=1e-9)


def test_a_map_blocks_into_scatters_with_no_arithmetic():
    fold, _ = _map()
    log = []
    fold.blocked(16, log)
    assert log == []


@pytest.mark.parametrize("sr", [REAL, MAX_PLUS], ids=lambda sr: sr.name)
def test_a_reduction_blocks_into_one_reduce_per_block(sr):
    fold, _ = _sum(steps=32, width=8, sr=sr)
    log = []
    fold.blocked(16, log)
    assert _shapes(log) == {("reduce", 1, 8, 16): 2}


def test_a_product_blocks_into_one_product_per_chunk_of_the_reduced_axis():
    fold, _ = _product(rows=12, columns=10, depth=32)
    log = []
    fold.blocked(16, log)
    assert _shapes(log) == {("write", 12, 10, 16): 2}


def test_a_prefix_sum_blocks_into_a_triangle_of_ones_and_one_carried_row():
    fold, _ = _prefix_sum(steps=32, width=8)
    first, second = fold.blocks(16)
    assert not first.unit and second.unit == {"out": (15,)}
    np.testing.assert_array_equal(second.coef[(("rows", "out"), ("in", ""))], np.tril(np.ones((16, 16))))
    np.testing.assert_array_equal(second.coef[(("rows", "out"), ("unit", "out"))], np.ones((16, 1)))


def test_a_dense_recurrence_blocks_into_products_of_its_step_matrices():
    fold, _ = _dense_recurrence(steps=16, width=6)
    log = []
    compose_all(fold.leaves(), log)
    # Fifteen compositions of sixteen steps, each one product of two matrices and one of a matrix with the offsets.
    assert _shapes(log) == {("merge", 6, 6, 6): 15, ("merge", 6, 3, 12): 15}


def test_a_triangular_inverse_blocks_into_merges_of_two_products_and_its_coefficient_is_the_inverse():
    n = 32
    fold, expected = _triangular_inverse(n)
    log = []
    whole = compose_all(fold.leaves(log), log)
    inverse = whole.coef[(("rows", "inv"), ("dense", "inv"))]
    np.testing.assert_allclose(inverse, (expected["inv"] + np.eye(n))[1:, 1:], atol=1e-9)
    assert not any(p.kind == "gram" for p in log)
    # Two blocks of 8 rows merge through two 8 x 8 x 8 products; the last row has no pair at the first level.
    assert _shapes(log)[("merge", 8, 8, 8)] == 2
    assert _shapes(log)[("merge", 16, 16, 16)] == 0 and _shapes(log)[("merge", 15, 16, 16)] == 1


def test_the_delta_rule_blocks_into_its_chunk_rule():
    steps, keys, values = 32, 24, 20
    fold, _ = _delta_rule(steps, keys, values)
    log = []
    fold.blocked(16, log)
    shapes = _shapes(log)
    # The state is read once a block through the keys and the queries, and written once through the keys.
    assert shapes[("read", 32, values, keys)] == 2 and shapes[("write", keys, values, 16)] == 2
    # The keys and queries of the later half of a block meet the keys of the earlier half in one product.
    assert shapes[("gram", 16, 8, keys)] == 2
    block = fold.blocks(16)[1]
    assert block.full["s"].shape == (keys, 16) and block.dense["s"].shape == (32, keys)


@pytest.mark.parametrize("name", ["triangular_inverse", "delta_rule", "prefix_sum"])
def test_the_coefficient_of_a_block_is_lower_triangular_in_step_order(name):
    """The same composition over the Boolean semiring says which coefficients can be nonzero."""
    fold, _ = CASES[name]()
    block = compose_all(fold.over(BOOL).leaves()[:16])
    left = ("rows", next(iter(block.rows)))
    reader = ("in", "") if name != "triangular_inverse" else ("dense", "inv")
    pattern = block.coef[(left, reader)]
    assert pattern.shape[0] == pattern.shape[1] == 16
    np.testing.assert_array_equal(pattern, np.tril(np.ones((16, 16), dtype=bool)))


def test_a_map_and_a_product_have_no_coupling_between_steps():
    for case in (_map, _product, _sum):
        block = compose_all(case()[0].over(BOOL).leaves()[:16])
        for pattern in block.coef.values():
            assert pattern.sum(axis=0).max() <= 1 or pattern.sum(axis=1).max() <= 1


def test_a_step_that_is_not_affine_in_its_state_is_refused():
    def squares(t, state, inputs):
        state.s[:] = state.s[:] @ state.s[:]

    with pytest.raises(NotAffine):
        Fold(2, squares, {"s": np.eye(3)}, {}).leaves()

    def reads_a_finished_row_as_a_coefficient(t, state, inputs):
        state.s[1] = state.s[1] + state.s[1, :1] @ state.s[:1]

    with pytest.raises(NotAffine):
        Fold(2, reads_a_finished_row_as_a_coefficient, {"s": np.eye(3)}, {}).leaves()
