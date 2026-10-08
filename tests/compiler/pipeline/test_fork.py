"""``pipeline/fork.py``: the deferred leaf, the iterative leaf walk, the fork point's typed partition and walk, and
the schedule branch's descent rule (``admits``) and the cold-pool draw — on synthetic forks, no pass and no tracing."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from emmy.compiler.pipeline.fork import DeferredFork, Fork, _ScheduleFork, iter_leaves, parallel_descent_rows, parallel_expand


@dataclass(frozen=True)
class _Branch(Fork):
    """A synthetic branch: the knobs it pins and the options below it."""

    knobs: dict
    children: tuple

    def expand(self):
        return list(self.children)


def _leaf(tag: str, made: list) -> DeferredFork:
    return DeferredFork(lambda: made.append(tag) or tag, {"TAG": tag})


def test_deferred_structural_leaf_materializes_only_when_selected() -> None:
    made: list[str] = []
    leaf = DeferredFork(lambda: made.append("built") or "graph", {"PLACE": "cut"}, structural=True)
    assert leaf.is_leaf and leaf.structural and leaf.knobs == {"PLACE": "cut"}
    assert made == []
    assert leaf.expand() == ["graph"]
    assert made == ["built"]


def test_leaf_walk_does_not_use_the_python_call_stack() -> None:
    """Maximal fused layers can have more schedule sites than Python's recursion limit."""

    class Chain(Fork):
        knobs = {}

        def __init__(self, depth: int):
            self.depth = depth

        def expand(self):
            return [Chain(self.depth - 1)] if self.depth else [DeferredFork(lambda: "op")]

    (leaf,) = iter_leaves([Chain(2_000)])
    assert leaf.is_leaf


def test_iter_leaves_streams_leaves_depth_first_in_emission_order() -> None:
    """Each option's leaves precede the next's, a branch expanding in place — the order a score tie falls back to
    (option-0 first) — and the walk materializes nothing."""
    made: list[str] = []
    tree = [
        _Branch({"A": 1}, (_leaf("a1", made), _Branch({"A": 1, "B": 2}, (_leaf("a1b2", made),)))),
        _leaf("top", made),
        _Branch({"A": 2}, ()),
    ]
    assert [option.knobs["TAG"] for option in iter_leaves(tree)] == ["a1", "a1b2", "top"]
    assert made == []


def _grid(made: list) -> list[Fork]:
    """A 4 × 4 × 4 tree whose leaves spell their path."""
    return [
        _Branch({"A": a}, tuple(_Branch({"A": a, "B": b}, tuple(_leaf(f"{a}{b}{c}", made) for c in range(4))) for b in range(4)))
        for a in range(4)
    ]


def test_a_parallel_draw_takes_the_same_rows_at_any_worker_count() -> None:
    """The draw is cut into fixed seeded pieces, so forked workers change how fast it runs, never what it takes; a
    worker returns rows, so nothing is built in the parent."""
    made: list[str] = []
    serial = parallel_descent_rows(_grid(made), draw=50, seed="pool", workers=1)
    assert len(serial) == 50 and len({row["TAG"] for row in serial}) > 1
    assert parallel_descent_rows(_grid(made), draw=50, seed="pool", workers=3) == serial
    assert parallel_descent_rows(_grid(made), draw=50, seed="other", workers=1) != serial
    assert made == []


def _draw_in_a_daemon(queue) -> None:
    queue.put(parallel_descent_rows(_grid([]), draw=50, seed="pool", workers=3))


def test_a_daemonic_process_draws_in_place() -> None:
    """A vLLM worker is a daemonic process, which may not start children: a serving boot that compiles a kernel
    with no evidence draws its cold pool in that process, and takes the same rows."""
    from multiprocessing import get_context

    context = get_context("fork")
    queue = context.Queue()
    worker = context.Process(target=_draw_in_a_daemon, args=(queue,), daemon=True)
    worker.start()
    rows = queue.get(timeout=20)
    worker.join(timeout=20)
    assert rows == parallel_descent_rows(_grid([]), draw=50, seed="pool", workers=1)


def test_fork_point_partitions_offers_and_walks_them() -> None:
    """The engine's typed offer partition — ``splices`` / ``variants`` classify top-level options once — and the
    walk the fork point owns: ``leaves()`` streams every complete leaf in emission order, ``find(row)`` descends
    to the one leaf a row names. A fork outside a schedule enumeration carries no pool identity."""
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline.pipeline import ForkPoint

    made: list[str] = []
    tree = _Branch({}, (_leaf("a1", made), _Branch({"A": 2}, (_leaf("a2", made),))))
    splice = Graph()
    fp = ForkPoint(match=None, options=[splice, tree], root_op=None, ctx=None)
    assert fp.splices == (splice,)
    assert fp.structural  # derived from the partition, not a second classification
    assert fp.variants == (tree,)
    assert tree.pool_id is None
    leaves = fp.flat()
    assert leaves[0] is splice and [leaf.knobs["TAG"] for leaf in leaves[1:]] == ["a1", "a2"]
    variants = ForkPoint(match=None, options=[tree], root_op=None, ctx=None)
    found = variants.find({"TAG": "a2"})
    assert found is not None and found[1] == {"TAG": "a2"} and variants.find({"TAG": "none"}) is None
    assert made == []


def test_admits_reads_a_schedule_prefix() -> None:
    """``Fork.admits`` is the one descent rule for a row that names a leaf: a schedule branch spells each decided
    knob as a prefix of what its leaves will spell, so the row's value must extend it at a segment boundary."""
    branch = _ScheduleFork(
        tree=SimpleNamespace(branch_knobs={"S_warp_eligible": 1.0, "STAGE": ""}), context=None, row={"TILE": "mma/f2x2", "WORK": "w2x2"}
    )
    assert branch.admits({"TILE": "mma/f2x2/k2", "WORK": "w2x2+p1", "STAGE": "d2/smem-tma"})
    assert branch.admits({"TILE": "mma/f2x2", "RASTER": "gm8"})
    assert not branch.admits({"TILE": "mma/f2x2/k2", "WORK": "w2x8"})
    assert not branch.admits({"TILE": "mma/f2x20/k2", "WORK": "w2x2"}), "an extension must start at a segment boundary"
    sited = _ScheduleFork(
        tree=SimpleNamespace(branch_knobs={}), context=None, row={"REDUCE@map.1/twist": "coop/r2", "REDUCE@map.1/twist.1/inner": ""}
    )
    assert not sited.admits({"REDUCE": "coop"}), "a bare row key prunes a site that decided another non-OFF value"
    assert _ScheduleFork(tree=SimpleNamespace(branch_knobs={}), context=None, row={"REDUCE@map.1/twist": "coop"}).admits({"REDUCE": "coop"})


def test_admits_prunes_a_site_this_branch_already_decided_OFF() -> None:
    """A branch that DECIDED a site OFF cannot reach a leaf carrying a value there.

    ``admits`` reads a branch's value as a PREFIX of what its leaves will spell, and every string
    extends the empty one — so an OFF the branch had already settled admitted every request and the
    descent only failed at leaf matching. With one such site per fork that is a factor of two, and
    the kernels this bites have dozens: a 16-rank serving boot sat in the schedule search for 45
    minutes without compiling a single kernel. An OFF the branch merely INHERITED is still
    undecided and still admits, and so does a site the row names only by its bare family key.
    """
    decided = _ScheduleFork(tree=SimpleNamespace(branch_knobs={}), context=None, row={"STAGE@map.1/inner": ""})
    assert not decided.admits({"STAGE@map.1/inner": "d1/smem"})
    assert decided.admits({"STAGE@map.1/inner": ""})

    inherited = _ScheduleFork(tree=SimpleNamespace(branch_knobs={"STAGE@map.1/inner": ""}), context=None, row={})
    assert inherited.admits({"STAGE@map.1/inner": "d1/smem"}), "an inherited OFF is undecided, not a decision"

    bare = _ScheduleFork(tree=SimpleNamespace(branch_knobs={}), context=None, row={"REDUCE@map.1/inner": ""})
    assert bare.admits({"REDUCE": "coop"}), "a bare family key still reads as a bare pin: OFF or the value"


def test_a_parallel_expansion_builds_each_unbuilt_arm_on_a_worker() -> None:
    """A kernel-set fork's arms are built on forked workers and memoized on the arm where its own expansion
    would build, so the parent builds none of them and a later expansion finds them; an arm already built stays
    as it is, and one worker leaves them lazy."""
    made: list[str] = []
    arms = [DeferredFork(lambda tag=tag: made.append(tag) or {"arm": tag}, {"PLACE": tag}, structural=True) for tag in "abcd"]
    parallel_expand(arms, workers=1)
    assert made == [] and all("_built" not in arm.__dict__ for arm in arms)
    assert arms[0].expand() == [{"arm": "a"}] and made == ["a"]
    parallel_expand(arms, workers=3)
    assert made == ["a"], "the children built the rest, in their own memory"
    assert [arm.expand() for arm in arms] == [[{"arm": tag}] for tag in "abcd"]
    assert made == ["a"]
