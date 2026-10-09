"""The golden pools of a DB instance (``db/export.golden_pools`` / ``ranking.build_golden_groups``): one pool per
card, regime, exact kernel and sizes holding a golden file's rows, enumerated from the kernel's own definition — the
same candidates the golden's stored program opens through the whole tile pipeline, at unit scale — and the export
that writes them as a dataset the readers load back unchanged."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.dataset import Dataset
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.db.export import export_dataset, golden_pools, placement_pools
from emmy.compiler.pipeline.search.features import tile_signature
from emmy.compiler.pipeline.search.golden import GoldenFile, import_rows, regime_context
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.ranking import build_golden_groups, build_placement_groups, enumerate_graph, enumerate_pool, pool_context
from tests.compiler.pipeline.search.helpers import CARDS, F16_MATMUL_STAMPS, GPU_5090, StubKernel, kernel_row, perf_row, tuned_db
from tests.compiler.realization import helpers as corpus

_MATMUL = "matmul/f16-mma-m128n128k128-f32.json"
_SPLIT = "matmul/f16-mma-splitk-deferred.json"
_CUT = "fused/linear-add-place-cut-sm70.json"


def test_a_kernel_pool_opens_the_candidates_its_golden_program_opens(tmp_path):
    """The DB pool is enumerated from the kernel's definition through the tile lowering alone; the golden
    file's program goes through the whole tile pipeline. Same candidates in the same order, the same golden
    row among them — so a rank over the DB is the rank the file-side fit computed. A pool holds only what a
    golden file sourced, and the builder's ``kernel`` narrows on the kernel's C name."""
    db = tuned_db(None, (_MATMUL,), source="golden:case")
    [pool], dropped = golden_pools(db)
    assert pool.kernel.formed and pool.name.startswith("k_matmul_") and (pool.regime, pool.bindings) == ("", {}) and dropped == {}
    assert build_golden_groups([pool], kernel="k_reduce") == ([], [])
    groups, skipped = build_golden_groups([pool], kernel="k_matmul")
    assert skipped == []
    [group] = groups
    assert (group.key, group.name, group.tier, group.gpu) == (f"{GPU_5090}/{pool.name}", pool.name, "warp", GPU_5090)

    case = corpus.load_case(corpus.CASES_DIR / _MATMUL)
    ctx = Context.from_target(case.compute_cap, gpu_name=CARDS[case.compute_cap], compile_flags="")
    with pinned_knobs(case.row.pins):
        file_side = enumerate_graph(case.program(), ctx)
    db_side = enumerate_pool(pool, pool_context(pool))
    assert group.total == file_side.total == len(group.feats) == len(db_side.rows)
    assert list(map(tile_signature, db_side.rows)) == list(map(tile_signature, file_side.rows))
    want = tile_signature(case.row.knobs)
    assert group.golden_ids == (next(i for i, row in enumerate(file_side.rows) if tile_signature(row) == want),)

    assert golden_pools(tuned_db(None, (_MATMUL,), source="measured")) == ([], {})

    # The export carries the group, its pool (rows and kernel definition) and the provenance to a directory and back.
    dataset = export_dataset(db, source="test", pool_sample=0, seed=0)
    back = Dataset.load(dataset.dump(tmp_path / "dataset"))
    [loaded] = back.golden
    assert (loaded.golden_ids, loaded.feat_names, loaded.total) == (group.golden_ids, group.feat_names, group.total)
    assert np.array_equal(loaded.feats, group.feats, equal_nan=True) and loaded.pools == (pool,)
    assert dataset.provenance["sources"] == {"golden:case": 1} and back.provenance == dataset.provenance


def test_the_placement_space_is_one_pool_per_fork_with_the_golden_arm_marked(tmp_path):
    """A golden that cut its kernel is, in the placement space, the parent's placement fork: the arms the cut
    pass offers unpinned — keep fused, one kernel; the cut, two — featurized from the kernels each leaves, the
    cut arm marked and the fused one not. The parent's pool holds the routing decision as its one row. A piece the
    golden left unsplit is a split fork with the unsplit arm marked; a piece with no kernel-set fork is skipped by
    name. The dataset round-trips with its space."""
    db = tuned_db(None, (_CUT,), source="golden:case")
    pools = placement_pools(db, golden_pools(db)[0])
    [parent] = [pool for pool in pools if pool.rows]
    assert len(pools) == 3 and [row.knobs for row in parent.rows] == [{"PLACE": "cut"}]

    dataset = export_dataset(db, source="test", pool_sample=0, seed=0, space="placement")
    [group] = [group for group in dataset.golden if group.tier == "place"]
    assert (group.key, group.total, group.golden_ids) == (f"{parent.gpu}/{parent.name}", 2, (1,))
    assert list(group.feats[:, group.feat_names.index("P_n_pieces")]) == [1.0, 2.0]
    splits = [group for group in dataset.golden if group.tier == "split"]
    assert splits and all(split.golden_ids == (0,) and split.feats[0, split.feat_names.index("P_n_pieces")] == 1.0 for split in splits)
    assert {reason for *_, reason in dataset.skipped} == {"no kernel-set fork"} and dataset.measured == []

    back = Dataset.load(dataset.dump(tmp_path / "placement"))
    [loaded] = [group for group in back.golden if group.tier == "place"]
    assert back.provenance["space"] == "placement" and loaded.golden_ids == (1,) and np.array_equal(loaded.feats, group.feats)
    assert [pool.kernel.exact_identity for pool in loaded.pools] == [parent.kernel.exact_identity]


def test_symbolic_placement_fork_exports_in_the_dynamic_tier():
    """A recorded symbolic cut must export with the same regime its arm features carry."""
    golden = GoldenFile.load(Path(__file__).parents[5] / "recipes" / "DeepSeek-V4-Flash-0731" / "golden" / "v100_sm70.json")
    target = "k_linear_matmul_reduce_b61b5c"
    row = next(row for row in golden.rows if any(golden.kernel(route.parent).name == target for route in golden.path_to(row.kernel)))
    db = SearchDB()
    import_rows(db, regime_context(golden, row.pins), golden, [row], source="golden:case")
    pools = placement_pools(db, golden_pools(db)[0])
    pool = next(pool for pool in pools if pool.kernel.name == target)

    groups, _skipped = build_placement_groups([pool])

    assert groups and all(group.tier == "dyn" and group.dynamic for group in groups)


def test_a_golden_over_a_kernel_set_is_one_pool_per_piece():
    """A golden that lowers to several kernels — a deferred split-K, its main kernel and its combine — is one
    pool per piece, each holding the receipt measured on that kernel, where the file-side builder had one pool
    for the whole target's forks."""
    pools, _dropped = golden_pools(tuned_db(None, (_SPLIT,), source="golden:case"))
    assert len(pools) == 2 and [len(pool.rows) for pool in pools] == [1, 1]
    assert len({pool.kernel.exact_identity for pool in pools}) == 2
    assert all(row.source == "golden:case" for pool in pools for row in pool.rows)


def test_a_split_a_golden_took_is_the_label_at_the_split_fork():
    """A golden that split its kernel across CTAs is, in the placement space, the kernel's split fork: the unsplit
    arm, one kernel, beside every split width, two — the width the golden recorded marked. The pieces it minted
    split nothing further and are skipped by name."""
    from emmy.compiler.pipeline.search.ranking import placement_decisions, pool_context, walk_placement

    db = tuned_db(None, (_SPLIT,), source="golden:case")
    pools = placement_pools(db, golden_pools(db)[0])
    [parent] = [pool for pool in pools if pool.rows]
    assert [row.knobs for row in parent.rows] == [{"REDUCE": "g2k"}]
    [fork], unmatched = walk_placement(parent, pool_context(parent), placement_decisions(pools, parent), own=True)
    assert unmatched == [] and fork.tier == "split" and fork.labels[0] == "unsplit" and [fork.labels[i] for i in fork.positives] == ["g2k"]

    [group] = export_dataset(db, source="test", pool_sample=0, seed=0, space="placement").golden
    assert (group.key, group.tier, group.golden_ids) == (f"{parent.gpu}/{parent.name}", "split", tuple(fork.positives))
    assert group.feats[0, group.feat_names.index("P_n_pieces")] == 1.0
    assert set(group.feats[1:, group.feat_names.index("P_n_pieces")]) == {2.0}


def test_a_layout_decision_on_another_kernel_steers_no_cut():
    """A layout decision spells several keys, like a several-seam cut, but cuts nothing: one recorded on another
    kernel of the card leaves this kernel's walk as it was."""
    from emmy.compiler.pipeline.search.ranking import placement_decisions, pool_context, walk_placement

    db = tuned_db(None, (_CUT,), source="golden:case")
    pools = placement_pools(db, golden_pools(db)[0])
    [parent] = [pool for pool in pools if pool.rows]
    decisions = placement_decisions(pools, parent)
    forks, _unmatched = walk_placement(parent, pool_context(parent), decisions, own=True)
    layout = {**decisions, "another": {"LAYOUT@w1": "source", "LAYOUT@w2": "source"}}
    assert walk_placement(parent, pool_context(parent), layout, own=True) == (forks, _unmatched)


def test_a_placement_label_is_what_the_golden_of_that_card_did():
    """One card's golden cut the kernel; another's only measured it whole. Each card keeps its own label, and no
    price decides it: the cut stays the first card's label though the kernel is measured faster whole there too."""
    db = tuned_db(None, (_CUT,), source="golden:case")
    [parent] = [pool for pool in placement_pools(db, golden_pools(db)[0]) if pool.rows]
    other = "NVIDIA Tesla V100 SXM3 32GB"
    whole = dict(us=1.0, flags=parent.regime, bindings=parent.bindings, knobs={"WORK": "t1"})
    db.record_perf_row(perf_row(parent.kernel.exact_identity, gpu=parent.gpu, cc=70, source="golden:case", **whole))
    db.record_perf_row(perf_row(parent.kernel.exact_identity, gpu=other, cc=70, source="golden:other", **whole))
    assert {pool.gpu for pool in golden_pools(db)[0] if pool.kernel == parent.kernel} == {parent.gpu, other}
    dataset = export_dataset(db, source="test", pool_sample=0, seed=0, space="placement")
    assert {group.gpu: group.golden_ids for group in dataset.golden if group.tier == "place"} == {parent.gpu: (1,), other: (0,)}


def test_a_cut_a_golden_took_is_the_label_with_no_piece_measured():
    """What a golden did is the label, not what it measured: a file whose rows lost their times still says where
    it cut the kernel, and the placement space marks that cut though the DB holds no measurement of any piece."""
    case = corpus.load_case(corpus.CASES_DIR / _CUT)
    ctx = Context.from_target(case.compute_cap, gpu_name=CARDS[case.compute_cap], compile_flags="")
    db = SearchDB()
    assert import_rows(db, ctx, case.document, [replace(row, measurements=None) for row in case.rows], source="golden:case") == 0
    pools, _dropped = golden_pools(db)
    [parent] = placement_pools(db, pools)
    assert pools == [] and (parent.gpu, [row.knobs for row in parent.rows]) == (CARDS[case.compute_cap], [{"PLACE": "cut"}])


def test_a_pool_of_a_kernel_formed_from_no_loop_op_is_skipped_by_name():
    """A piece carved from a twisted tree has no definition the lowering takes back to it; its rows are counted
    out under their kernel's name rather than enumerated through a body that would make another kernel."""
    db = SearchDB()
    db.record_kernel(kernel_row("twisted", name="k_piece", formed=False))
    db.record_perf_row(perf_row("twisted", us=500.0, source="golden:case"))  # plausible: the freeze admits it
    groups, skipped = build_golden_groups(golden_pools(db, lambda row: StubKernel(F16_MATMUL_STAMPS))[0])
    assert groups == [] and skipped == [(GPU_5090, "k_piece.twisted", "kernel formed from no loop op")]
