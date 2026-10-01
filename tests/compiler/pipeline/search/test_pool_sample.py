"""The candidate-pool draw — determinism, exact membership, and the whole pool where sampling has nothing to do.

Three properties, and each is load-bearing somewhere else:

- the draw is a pure function of the tree and ``(sample size, seed, keep)`` — the descents are seeded on the
  sample's own identity and every expansion is deterministic — so two byte-identical pools draw byte-identical
  samples. That is what keeps ``emmy fit`` reproducible and what keeps two goldens over one pool merging into one
  training case rather than two;
- a row in ``keep`` survives the draw wherever it sits, reached by its own directed descent, and a row the tree
  does not offer stays absent. The fit locates a golden in its pool and DROPS the golden on a miss, and ``eval
  golden`` reads the same miss as a pin or dtype mismatch — a draw that could lose the row would turn a real
  defect signal into noise;
- a pool whose declared bound fits the draw is taken whole, so the sampled and unsampled paths agree wherever
  sampling has nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.pipeline.fork import DeferredFork, Fork, leaf_knobs
from emmy.compiler.pipeline.search.pool import PoolSample


@dataclass(frozen=True)
class _Branch(Fork):
    """A synthetic branch: the knobs it pins, the options below it, and the pool bound it declares."""

    knobs: dict
    children: tuple
    pool_bound: int | None = field(default=None)

    def expand(self):
        return list(self.children)


def _rows(n_tile: int = 20, n_stage: int = 25) -> list[dict]:
    return [{"TILE": str(t), "STAGE": str(s)} for t in range(n_tile) for s in range(n_stage)]


def _tree(rows: list[dict], bound: int | None = None) -> _Branch:
    """A two-level tree over ``rows``: one branch per ``TILE`` value, one deferred leaf per row below it."""
    by_tile: dict[str, list[dict]] = {}
    for row in rows:
        by_tile.setdefault(row["TILE"], []).append(row)
    branches = []
    for tile, group in by_tile.items():
        branches.append(_Branch({"TILE": tile}, tuple(DeferredFork(lambda: None, dict(row)) for row in group)))
    return _Branch({}, tuple(branches), bound)


def _drawn(sample: PoolSample, tree: _Branch) -> list[dict]:
    return [leaf_knobs(leaf) for leaf in sample.draw([tree])]


def test_the_draw_reads_only_the_tree_its_size_and_its_seed() -> None:
    rows = _rows()
    drawn = _drawn(PoolSample(rows=10, seed=0), _tree(rows))
    assert drawn == _drawn(PoolSample(rows=10, seed=0), _tree(rows)), "an equal tree draws an equal sample"
    assert drawn != _drawn(PoolSample(rows=10, seed=1), _tree(rows)), "the seed must move the draw"
    assert len(drawn) <= 10 and len({tuple(sorted(r.items())) for r in drawn}) == len(drawn), "no row twice"
    assert all(row in rows for row in drawn), "every drawn row is a leaf of the tree"
    assert len({row["TILE"] for row in drawn}) > 1, "descents reach several branches, not an emission-order prefix"


def test_a_kept_row_survives_wherever_it_sits() -> None:
    rows = _rows()
    without = _drawn(PoolSample(rows=10, seed=0), _tree(rows))
    missing = next(row for row in rows if row not in without)
    kept = PoolSample(rows=10, seed=0, keep=(tuple(sorted(missing.items())),))
    drawn = _drawn(kept, _tree(rows))
    assert drawn[0] == missing, "the kept row leads the draw"
    assert drawn[1:] == [row for row in without if row != missing], "and adds exactly itself to it"


def test_a_kept_row_the_tree_does_not_offer_stays_absent() -> None:
    rows = _rows()
    kept = PoolSample(rows=10, seed=0, keep=((("STAGE", "999"), ("TILE", "999")),))
    assert _drawn(kept, _tree(rows)) == _drawn(PoolSample(rows=10, seed=0), _tree(rows))


def test_a_pool_whose_bound_fits_the_draw_is_taken_whole() -> None:
    rows = _rows()
    assert _drawn(PoolSample(rows=500), _tree(rows, bound=500)) == rows
    assert _drawn(PoolSample(rows=0), _tree(rows, bound=500)) == rows, "0 is 'enumerate everything', the live and unsampled default"
    assert _drawn(PoolSample(rows=0), _tree(rows)) == rows, "and so with no bound declared"
    assert len(_drawn(PoolSample(rows=10), _tree(rows, bound=500))) <= 10, "a bound past the draw is sampled"


def test_the_sample_identity_ignores_the_size_sink_and_the_keep_set_order() -> None:
    """The sample identity must be stable across processes and spell one keep-set one way."""
    a = PoolSample(rows=10, seed=1, keep=((("TILE", "x"),), (("TILE", "y"),)))
    b = PoolSample(rows=10, seed=1, keep=((("TILE", "y"),), (("TILE", "x"),)))
    b.totals["some-pool"] = 999
    assert a.key == b.key, "a size sink and the keep order are not part of a sample's identity"
    assert a.key != PoolSample(rows=10, seed=2, keep=a.keep).key
    assert a.key != PoolSample(rows=11, seed=1, keep=a.keep).key
    assert a.key != PoolSample(rows=10, seed=1, keep=((("TILE", "x"),),)).key
