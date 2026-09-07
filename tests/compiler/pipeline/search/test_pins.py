"""``spelled_arm`` — the one reading of a measured row at a kernel-set fork, shared by the deploy's
evidence pick and the golden replay."""

from __future__ import annotations

from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.search.pins import spelled_arm


def _arm(knobs: dict, *, structural: bool = False) -> DeferredFork:
    return DeferredFork(materialize=lambda: None, knobs=knobs, structural=structural)


def test_a_placement_fork_reads_cut_fuse_and_stale_rows() -> None:
    fuse, cut_a, cut_b = (
        _arm({"PLACE": "fuse"}),
        _arm({"PLACE@map.1/map": "cut"}, structural=True),
        _arm({"PLACE@map.2/map": "cut"}, structural=True),
    )
    options = [fuse, cut_a, cut_b]

    assert spelled_arm(options, {"PLACE@map.2/map": "cut", "WORK": "t8"}) == (cut_b, {"PLACE@map.2/map": "cut"})
    assert spelled_arm(options, {"PLACE": "cut"}) == (cut_a, {"PLACE@map.1/map": "cut"}), "a bare cut is the root-most offered seam"
    assert spelled_arm(options, {"WORK": "t8", "TILE": "f2"}) == (fuse, {"PLACE": "fuse"}), "a schedule row says the kernel ran fused"
    assert spelled_arm(options, {"PLACE@map.1/map": "fuse"}) == (fuse, {"PLACE": "fuse"})
    assert spelled_arm(options, {"PLACE@map.9/twist": "cut"}) is None, "a cut this kernel does not offer decides nothing"


def test_a_split_fork_reads_the_cross_cta_half_alone() -> None:
    unsplit, g2k, g2a = (
        _arm({"REDUCE@inner": ""}),
        _arm({"REDUCE@inner": "g2k"}, structural=True),
        _arm({"REDUCE@inner": "g2a"}, structural=True),
    )
    options = [unsplit, g2k, g2a]

    assert spelled_arm(options, {"REDUCE": "g2a/coop", "WORK": "t32"}) == (g2a, {"REDUCE@inner": "g2a"})
    assert spelled_arm(options, {"REDUCE@inner": "g2k"}) == (g2k, {"REDUCE@inner": "g2k"})
    assert spelled_arm(options, {"REDUCE": "coop"}) == (unsplit, {"REDUCE@inner": ""}), "no cross-CTA half: the kernel ran whole"
    assert spelled_arm(options, {"WORK": "t32", "TILE": "f2"}) == (unsplit, {"REDUCE@inner": ""}), (
        "a schedule row measured the kernel whole"
    )
    assert spelled_arm(options, {"REDUCE": "g8k"}) is None, "a split this kernel does not offer decides nothing"


def test_a_row_marking_several_offered_seams_spells_the_composed_arm() -> None:
    """A pinned compile consumes every scoped PLACE pin that resolves on a kernel as ONE composed
    decision, and ``run --record-greedy`` writes it as one row naming every seam: where the fork
    offers that composed arm beside its single seams, the row spells the composed arm — never the
    first single seam it happens to mark. A single-seam row still spells its single arm, a seam
    another kernel offers is free, and a fork offering no composed arm reads as before."""
    fuse, cut_a, cut_b, cut_ab = (
        _arm({"PLACE": "fuse"}),
        _arm({"PLACE@map.1/map": "cut"}, structural=True),
        _arm({"PLACE@map.2/map": "cut"}, structural=True),
        _arm({"PLACE@map.1/map": "cut", "PLACE@map.2/map": "cut"}, structural=True),
    )
    both = {"PLACE@map.1/map": "cut", "PLACE@map.2/map": "cut"}
    options = [fuse, cut_a, cut_b, cut_ab]

    assert spelled_arm(options, both) == (cut_ab, both)
    assert spelled_arm(options, {**both, "WORK": "t8"}) == (cut_ab, both)
    assert spelled_arm(options, {**both, "PLACE@map.9/twist": "cut"}) == (cut_ab, both), "a seam this kernel does not offer is free"
    assert spelled_arm(options, {"PLACE@map.1/map": "cut"}) == (cut_a, {"PLACE@map.1/map": "cut"}), "one seam: its single arm"
    assert spelled_arm([fuse, cut_a, cut_b], both) == (cut_a, {"PLACE@map.1/map": "cut"}), "no composed arm offered: as before"
