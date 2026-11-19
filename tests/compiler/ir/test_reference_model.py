"""The value form of Loop IR, held to the form it replaces on five small kernels.

Each kernel is written twice. ``today`` is the form the compiler lowers now, run on the Loop IR runner: a condition
on a coordinate is a select, a shared value is computed at each read, and a recurrence is a carried state. ``target``
states the same computation with a range, a value defined once and read by index, and a recurrence that reads itself
(``reference_model``): one that is a walk over steps keeps its step as a coordinate and reads one step back, and one
whose every cell has a single writer is an array filled in order. The two agree on random inputs, which is what
fixes the meaning of the target form: a read sees the last definition at or before the coordinate read, and the seed
where there is none.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import parse_body
from tests.compiler.ir import reference_model

#: A sum under a causal mask: the keys at or before the row.
CAUSAL_SUM = (
    """
for i in 0..6:
  for k in 0..6:
    x = load X[i, k]
    let zero = 0.0
    m = select:
      x when k < i + 1
      zero when true
    s <- add(s, m) over k
  O[i] = s
""",
    """
for i in 0..6:
  for k in 0..i + 1 of 6:
    x = load X[i, k]
    s <- add(s, x) over k
  O[i] = s
""",
    {"X": (6, 6)},
    {"O": (6,)},
)

#: Forward substitution, ``X[r] = B[r] - sum(L[r, m] * X[m] for m < r)``. Today a step defines one row and copies the
#: others, and the sum is masked to the rows already solved. The target fills the rows in order: each cell has one
#: writer, so the step is gone, a read of an earlier row reads that row, and a row nothing defines holds the seed.
SOLVE = (
    """
for t in 0..5:
  for h in 0..2:
    for r in 0..6:
      for c in 0..4:
        for m in 0..6:
          l = load L[h, t + 1, m]
          p = pre acc[h, m, c]
          q = multiply(l, p)
          let zero = 0.0
          qm = select:
            q when m < t + 1
            zero when true
          s <- add(s, qm) over m
        b = pre acc[h, t + 1, c]
        v = subtract(b, s)
        old = pre acc[h, r, c]
        w = select:
          v when r == t + 1
          old when true
        acc[h, r, c] <- w seed B
        O[h, r, c] = w
""",
    """
for h in 0..2:
  for r in 1..6 of 6:
    for c in 0..4:
      for m in 0..r of 6:
        l = load L[h, r, m]
        p = pre X[h, m, c]
        q = multiply(l, p)
        s <- add(s, q) over m
      b = load B[h, r, c]
      v = subtract(b, s)
      X[h, r, c] <- v seed B
for h2 in 0..2:
  for r2 in 0..6:
    for c2 in 0..4:
      x = pre X[h2, r2, c2]
      O[h2, r2, c2] = x
""",
    {"L": (2, 6, 6), "B": (2, 6, 4)},
    {"O": (2, 6, 4)},
)

#: Two steps of a delta rule. The normalized key is read at the cell and again inside the sum: today it is computed
#: at both, the target defines it once. The state is a walk over steps, read one step back.
DELTA_RULE = (
    """
for t in 0..2:
  for i in 0..4:
    for j in 0..4:
      for a in 0..4:
        k = load K[t, a]
        w = load W[t]
        ka = multiply(k, w)
        sp = pre S[a, j]
        q = multiply(ka, sp)
        pr <- add(pr, q) over a
      v = load V[t, j]
      u = subtract(v, pr)
      k2 = load K[t, i]
      w2 = load W[t]
      ki = multiply(k2, w2)
      d = multiply(ki, u)
      so = pre S[i, j]
      sn = add(so, d)
      S[i, j] <- sn seed S0
      O[t, i, j] = sn
""",
    """
for t in 0..2:
  for b in 0..4:
    k = load K[t, b]
    w = load W[t]
    kw = multiply(k, w)
    kn[t, b] <- kw
  for i in 0..4:
    for j in 0..4:
      for a in 0..4:
        ka = pre kn[t, a]
        sp = pre S[t - 1, a, j]
        q = multiply(ka, sp)
        pr <- add(pr, q) over a
      v = load V[t, j]
      u = subtract(v, pr)
      ki = pre kn[t, i]
      d = multiply(ki, u)
      so = pre S[t - 1, i, j]
      sn = add(so, d)
      S[t, i, j] <- sn seed S0
      O[t, i, j] = sn
""",
    {"K": (2, 4), "W": (2,), "V": (2, 4), "S0": (4, 4)},
    {"O": (2, 4, 4)},
)

#: A walk over chunks: each chunk reads the state the chunk before left and writes its own rows and the next state.
WALK = (
    """
for c in 0..3:
  for r in 0..2:
    for i in 0..4:
      q = load Q[c, r, i]
      sp = pre S[i]
      qs = multiply(q, sp)
      o <- add(o, qs) over i
    O[c, r] = o
  for j in 0..4:
    for r2 in 0..2:
      k = load K[c, r2, j]
      u = load U[c, r2]
      ku = multiply(k, u)
      g <- add(g, ku) over r2
    d = load D[c]
    so = pre S[j]
    ds = multiply(d, so)
    sn = add(ds, g)
    S[j] <- sn seed S0
""",
    """
for c in 0..3:
  for r in 0..2:
    for i in 0..4:
      q = load Q[c, r, i]
      sp = pre S[c - 1, i]
      qs = multiply(q, sp)
      o <- add(o, qs) over i
    O[c, r] = o
  for j in 0..4:
    for r2 in 0..2:
      k = load K[c, r2, j]
      u = load U[c, r2]
      ku = multiply(k, u)
      g <- add(g, ku) over r2
    d = load D[c]
    so = pre S[c - 1, j]
    ds = multiply(d, so)
    sn = add(ds, g)
    S[c, j] <- sn seed S0
""",
    {"Q": (3, 2, 4), "K": (3, 2, 4), "U": (3, 2), "D": (3,), "S0": (4,)},
    {"O": (3, 2)},
)

#: A concatenation. Its condition picks what a cell computes, on a free axis, so it stays a select in both forms.
CAT = (
    """
for i in 0..6:
  a = load A[i < 3 ? i : 0]
  b = load B[i < 3 ? 0 : i - 3]
  y = select:
    a when i < 3
    b when true
  O[i] = y
""",
    """
for i in 0..6:
  a = load A[i < 3 ? i : 0]
  b = load B[i < 3 ? 0 : i - 3]
  y = select:
    a when i < 3
    b when true
  O[i] = y
""",
    {"A": (3,), "B": (3,)},
    {"O": (6,)},
)

KERNELS = {"causal_sum": CAUSAL_SUM, "solve": SOLVE, "delta_rule": DELTA_RULE, "walk": WALK, "cat": CAT}


def _inputs(shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in shapes.items()}


@pytest.mark.parametrize("name", KERNELS)
def test_the_target_form_computes_what_todays_form_computes(name: str) -> None:
    today, target, shapes, outputs = KERNELS[name]
    inputs = _inputs(shapes)
    op = LoopOp(body=parse_body(today))
    want = execute_loop_op_cpp(op, {buffer: inputs[buffer] for buffer in op.inputs}, outputs)
    want = dict(zip(op.outputs, want if isinstance(want, tuple) else (want,), strict=True))
    got = reference_model.run(target, inputs, outputs)
    for buffer in outputs:
        np.testing.assert_allclose(got[buffer], want[buffer], rtol=1e-5, atol=1e-5)


def test_a_read_sees_the_last_definition_at_or_before_its_coordinate_and_the_seed_where_there_is_none() -> None:
    """The three cases of one rule. A step that defines one cell leaves the others to the step before; a cell no step
    defined holds the seed; and a read one step back never sees what its own step stores, wherever it stands."""
    body = """
for t in 0..3:
  for i in 0..4:
    x = load X[t, i]
    p = pre S[t - 1, i]
    v = add(p, x)
    if i == t:
      S[t, i] <- v seed S0
    q = pre S[t - 1, i]
    O[t, i] = q
for t2 in 0..3:
  for i2 in 0..4:
    last = pre S[t2, i2]
    P[t2, i2] = last
"""
    rng = np.random.default_rng(1)
    inputs = {"X": rng.standard_normal((3, 4)).astype(np.float32), "S0": rng.standard_normal((4,)).astype(np.float32)}
    got = reference_model.run(body, inputs, {"O": (3, 4), "P": (3, 4)})
    state = inputs["S0"].copy()
    for t in range(3):
        np.testing.assert_allclose(got["O"][t], state, err_msg="a read one step back is the state the step started from")
        state[t] += inputs["X"][t, t]
        np.testing.assert_allclose(got["P"][t], state, rtol=1e-6, err_msg="a cell the step left alone keeps its last value")
    assert got["P"][2, 3] == inputs["S0"][3], "no step defines the last cell: it holds the seed"


def test_a_value_without_a_seed_is_read_only_where_it_was_defined() -> None:
    with pytest.raises(ValueError, match="before any statement defined it"):
        reference_model.run("for i in 0..2:\n  x = pre v[i]\n  O[i] = x\n  v[i] <- x", {}, {"O": (2,)})
