"""``group_measured`` — measured ``perf`` rows packed into comparison sets.

A group here is the set of configs that genuinely competed: one op, one card, one nvcc regime. What makes
that non-trivial is that three plausible spellings of "one op" are wrong, and two kinds of row carry a
number that is not a measurement. Each is pinned below, because every one of them fails silently — a
misgrouped or mislabelled pool still produces a confident-looking correlation.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.data.group import GoldenGroup, Group, group_measured
from tests.compiler.pipeline.search.helpers import F16_MATMUL_FEATS
from tests.compiler.pipeline.search.helpers import GPU_5090 as _GPU
from tests.compiler.pipeline.search.helpers import perf_row as _row

_GPU2 = "NVIDIA GeForce RTX 4090"


def _feats(**knobs) -> dict:
    """A row the admission filter accepts — the shared plausible fixture. Inventing a smaller dict here
    would sidestep the very gate these tests are about."""
    return {**F16_MATMUL_FEATS, **knobs}


def test_configs_that_competed_land_in_one_group():
    rows = [_row(f"n{i}", us=200.0 + 100 * i, knobs=_feats(TILE=f"f2x{2 + i}")) for i in range(4)]
    (group,), dropped = group_measured(rows)
    assert not dropped
    assert group.latency_us.tolist() == [200.0, 300.0, 400.0, 500.0]
    assert len(group.feats) == 4 and group.gpu == _GPU


def test_cards_never_pool_and_a_non_deployable_regime_never_arrives():
    """Two axes, two fates. Cards never pool — they are different tuning problems. A row from a
    non-deployable regime does not reach a pool at all: the shared admission filter
    (``freeze_reason``) drops it, because a measurement taken where nothing deploys answers a
    question nothing asks. The ``H_opt`` key stays part of the grouping key anyway — it is the
    pool's honest label of what it was measured under, a report cell reads it, and it is what would
    catch a regression that let a second regime back in."""
    rows = [
        _row("a", us=200.0, knobs=_feats(), opt=3),
        _row("b", us=300.0, knobs=_feats(), opt=1),  # same kernel, non-deployable regime
        _row("d", us=500.0, knobs=_feats(), opt=3, gpu=_GPU2),
    ]
    groups, dropped = group_measured(rows)
    assert dropped == {"non-deployable regime (H_opt=1)": 1}
    assert len(groups) == 2 and all(len(g.feats) == 1 for g in groups)


def test_the_same_kernel_from_two_sites_is_one_tuning_problem():
    """The key is the KERNEL's structure, not where it came from.

    A kernel minted by a cross-CTA split has its own structural identity, so tuning it is the same
    question as tuning an identical standalone kernel. The deploy path already joins their evidence
    that way (``Prior.evidence_pick`` indexes on ``S_*``). Keyed on the site that offered them the two
    land in different pools and get searched twice: on the RTX 5090 measurement freeze 73 structures
    were fragmented like that, the losing pool's best coming in a median 1.46x behind the winning
    pool's."""
    rows = [
        _row("from-a-split", us=200.0, knobs=_feats(TILE="f2x2")),
        _row("standalone", us=150.0, knobs=_feats(TILE="f4x4")),
    ]
    (group,), dropped = group_measured(rows)
    assert not dropped
    assert sorted(group.latency_us.tolist()) == [150.0, 200.0]


def test_one_offer_site_over_different_work_is_not_one_group():
    """The over-merging half of the same mistake. A site names where a decision was OFFERED, not what got
    computed, so one site can be realized as a single kernel or as several — and then both rows are honest
    measurements, of a whole op and of a piece of one, while comparing their latencies is not.

    Left merged this cost a real number: nine pools of the RTX 5090 freeze paired a fused rms_norm->linear
    megakernel with one kernel of the same op's unfused realization, and the report priced a 5.9 µs norm
    kernel against a 131 ms whole-op row as a 22 221x miss."""
    small = _feats(S_ext_free_prod=30720.0, S_ext_reduce_prod=3840.0, REDUCE="coop")
    large = _feats(S_ext_free_prod=69632.0, S_ext_reduce_prod=14745600.0, REDUCE="coop")
    # Both latencies are plausible for their OWN extents — the point is the grouping, so neither row may be
    # one the plausibility gate would have dropped anyway.
    rows = [_row("small", us=2000.0, knobs=small), _row("large", us=131496.0, knobs=large)]

    groups, dropped = group_measured(rows)
    assert not dropped
    assert sorted(g.latency_us.tolist() for g in groups) == [[2000.0], [131496.0]]


def test_alternative_schedules_of_one_kernel_stay_one_group():
    """Two schedules of one kernel share every ``S_*`` stamp, so they share a pool — which is what makes a
    pool a comparison at all.

    They share them by construction, not by luck: the identity strategy stamps a kernel at BIRTH, in
    recognition, before ``040_schedule`` offers the first fork — that pass's own error text says so. Nothing
    a schedule fork decides can move an ``S_*`` value, which is what makes keying on them safe."""
    rows = [
        _row("a", us=200.0, knobs=_feats(TILE="f2x2", WORK="w1x8")),
        _row("b", us=400.0, knobs=_feats(TILE="f8x8", WORK="w4x2")),
    ]
    (group,), _ = group_measured(rows)
    assert len(group.feats) == 2


def test_a_failed_bench_carries_a_sentinel_not_a_latency():
    """The watchdog sentinel is a huge POSITIVE number, so nothing downstream rejects it on sign — left in a
    group it is a row every model ranks last for free, inflating every correlation computed over the pool."""
    rows = [
        _row("ok", us=500.0, knobs=_feats()),
        _row("fail", us=1e9, knobs=_feats(TILE="f4x4"), status="bench_fail"),
    ]
    (group,), dropped = group_measured(rows)
    assert dropped == {"bench_fail": 1}
    assert group.latency_us.tolist() == [500.0]


def test_every_row_is_either_grouped_or_counted():
    """The counts are returned rather than logged because a report must publish them: "ρ=0.6 over 340 groups"
    means something different when a tenth of the corpus was dropped than when none was."""
    rows = [
        _row("a", us=500.0, knobs=_feats()),
        _row("b", us=1e9, knobs=_feats(TILE="f4x4"), status="bench_fail"),
        _row("c", us=200.0, knobs=_feats(TILE="f8x8"), flags="-lineinfo"),
        _row("d", us=300.0, knobs=_feats(TILE="f8x8"), flags="--use_fast_math"),  # the other precision lane: its own pool
    ]
    groups, dropped = group_measured(rows)
    assert sum(len(g.feats) for g in groups) + sum(dropped.values()) == len(rows)
    assert dropped == {"bench_fail": 1, "non-default compiler flags": 1}
    assert [g.latency_us.tolist() for g in groups] == [[500.0], [300.0]]


# --- the two label kinds -----------------------------------------------------------


def test_only_a_golden_pool_can_be_asked_which_rows_are_the_answer():
    """The two kinds differ in what can be ASKED of them, which is why they are two types. A measured pool
    has no ``golden_ids`` at all — not an empty one, not a raising one — so a rank metric that needs them
    says so by taking :class:`GoldenGroup`, and nothing has to check a flag at runtime. The supervision lives
    on the subclasses for the same reason: on the base it would be one field with two meanings."""
    rows = [_row(f"n{i}", us=200.0 + 100 * i, knobs=_feats(TILE=f"f2x{2 + i}")) for i in range(3)]
    (measured,), _ = group_measured(rows)
    golden = GoldenGroup.from_dicts("g/x", "x", "warp", "g", "x", 1, [{"D_a": float(i)} for i in range(3)])

    assert measured.latency_us.tolist() == [200.0, 300.0, 400.0]  # per row, in microseconds
    assert golden.golden_ids == (1,)
    assert isinstance(golden, Group) and not hasattr(measured, "golden_ids")
    assert not hasattr(golden, "latency_us")  # and the reverse: a golden pool has no measurement to report


def test_a_measured_pool_carries_the_regime_it_was_measured_under():
    """``h_opt`` rides on the group rather than being recovered from its packed columns: a report keys
    a cell on it, so it has to survive the trip from the row to the pool. Only the deployable regime
    reaches a pool, so today every cell reads the same value — the field says which one, instead of
    leaving a reader to assume."""
    rows = [
        _row("a", us=200.0, knobs=_feats(), opt=3),
        _row("b", us=300.0, knobs=_feats(), opt=1),
    ]
    groups, _ = group_measured(rows)
    assert [g.h_opt for g in groups] == [3.0]


def test_goldens_go_in_as_row_indices_and_come_back_as_row_indices():
    """A group is built ONCE, already knowing every golden it holds, so there is no amend-after-construction
    path — several arrive together, in any order, with duplicates, and the marker encoding never leaves the
    class."""
    feats = [{"D_a": float(i)} for i in range(5)]
    assert GoldenGroup.from_dicts("g/x", "x", "warp", "g", "x", 2, feats).golden_ids == (2,)
    assert GoldenGroup.from_dicts("g/y", "y", "warp", "g", "y", (4, 2, 4), feats).golden_ids == (2, 4)


def test_a_row_the_freeze_would_refuse_is_refused_here_too():
    """Admission is ``freeze_reason``, not a second list of rules — which matters most for the row it adds.

    An implausibly fast row is far worse than the ``bench_fail`` sentinel this module excludes by hand: the
    sentinel makes ONE row rank last, while a phantom optimum becomes the group's ``min`` and makes every
    other row in the pool look bad. It is invisible on freeze input — a freeze passed this filter at write
    time — so only an imported tune DB can produce it, and only a test like this one can show it."""
    rows = [
        _row("honest", us=500.0, knobs=_feats()),
        _row("phantom", us=0.05, knobs=_feats(TILE="f8x8")),
    ]
    groups, dropped = group_measured(rows)
    assert [g.latency_us.tolist() for g in groups] == [[500.0]]
    assert sorted(dropped) == ["implausible value"]
