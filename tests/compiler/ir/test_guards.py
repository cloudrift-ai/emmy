"""What the steps of a carried state read and write, each with its guard (``ir/guards.py``)."""

from emmy.compiler.ir.constraints import always
from emmy.compiler.ir.expression import parse
from emmy.compiler.ir.guards import carried
from emmy.compiler.ir.stmt import parse_body

# A row substitution the way the roll writes it: every cell takes a next value each step, the selects say which one
# is new, and the reads that are not used are steered to cell 0 by conditionals inside their indices.
_SOLVE = """
let zero = 0.0
for t in 0..6:
  for i in 0..8:
    for j in 0..8:
      for k in 0..6:
        coefficient = pre S[i < t + 3 && t + 2 <= i ? t + 2 : 0, k]
        above = pre S[k, (i < t + 3 && t + 2 <= i ? j : 0) < t + 2 ? j : 0]
        product = multiply(coefficient, above)
        term = select:
          product when k < t + 2
          zero when true
        total <- add(total, term) over k
      mine = pre S[t + 2, (i < t + 3 && t + 2 <= i ? j : 0) < t + 2 ? j : 0]
      new = add(total, mine)
      kept = pre S[i, j]
      row = select:
        new when (i < t + 3 && t + 2 <= i ? j : 0) < t + 2
        kept when true
      next = select:
        row when i < t + 3 && t + 2 <= i
        kept when true
      S[i, j] <- next seed A
      out[i, j] = next
"""


def test_a_carried_kernel_reads_as_guarded_reads_and_writes():
    facts = carried(parse_body(_SOLVE))
    # One row a step, the columns left of the diagonal: every other cell keeps its own value and is not written.
    (written,) = facts.written
    assert always(parse("i == t + 2"), *written, *facts.bounds) and always(parse("j < t + 2"), *written, *facts.bounds)
    assert facts.moving() == (0, 2)
    reads = {read.name: read for read in facts.reads}
    assert set(reads) == {"coefficient", "above", "mine"}, "the read of a cell's own value feeds no written cell"
    # Every conditional inside an index is settled by the guard the read counts under.
    assert [str(index) for index in reads["coefficient"].index] == ["t + 2", "k"]
    assert [str(index) for index in reads["above"].index] == ["k", "j"]
    assert [str(index) for index in reads["mine"].index] == ["t + 2", "j"]
    assert always(parse("k < t + 2"), *reads["above"].guard, *facts.bounds)


def test_a_state_every_step_rewrites_moves_along_no_axis():
    facts = carried(
        parse_body(
            """
            for t in 0..4:
              scale = load D[t]
              for i in 0..3:
                for j in 0..3:
                  own = pre S[i, j]
                  next = multiply(own, scale)
                  S[i, j] <- next seed 1.0
                  out[t, i, j] = next
            """
        )
    )
    assert facts.written == ((),) and facts.moving() is None
    assert [str(index) for read in facts.reads for index in read.index] == ["i", "j"]
