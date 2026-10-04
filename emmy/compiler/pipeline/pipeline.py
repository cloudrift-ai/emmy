"""Pipeline value types and compile driver: ``Pattern``, ``Match``,
``Rule``, ``RuleSkipped``, ``Pass``, ``Cursor``, ``Pipeline``, ``Run``.

Bundled together because they form a tight chain — ``Pattern`` defines
what a rule matches, ``Rule`` carries the pattern + rewrite, ``Pass``
groups rules, ``Pipeline`` is the frozen pass layout + matcher, ``Run``
owns ONE drive of that layout (ctx / db / backend / dump / rejections +
the engine loop), ``Cursor`` tracks per-candidate resume state inside a
Run, and ``Match`` carries ``Rule`` (which backref-resolves to ``Pass``).
``Run.resolve`` is the one entry point over the rule-batch body
(``Run._step``): a deterministic resolution — a ``decide`` callback picks
at each ``ForkPoint`` and the fold returns the terminal graph plus a
``Decision`` trace.

``Pipeline`` also owns the compile entry points — :meth:`build` and
:meth:`run` (which constructs a :class:`Run` and resolves it).
The per-rule logging, rewrite-kwarg dispatch, and snapshot rendering live
on :class:`Candidate` (see :mod:`..search.candidate`).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.compiler.graph import Graph, Node
from emmy.compiler.pipeline.fork import Fork, iter_leaves, leaf_for
from emmy.compiler.pipeline.knob import Knob, apply_off_defaults, family_of
from emmy.compiler.pipeline.strategy import PassEndEvent, RunStartEvent, discovered_strategies

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.base import Op
    from emmy.compiler.pipeline.dump import CompilerDump
    from emmy.compiler.pipeline.search.candidate import Candidate
    from emmy.compiler.pipeline.search.db import SearchDB

logger = logging.getLogger("emmy.compiler.pipeline")

_PASSES_DIR = Path(__file__).resolve().parent / "passes"
# The pass that closes lowering: after it every compute node is a ``CudaOp``. It is the last
# entry of ``CUDA_PASSES`` (the package assembles the canonical lists from it), and the marker
# :attr:`Pipeline.lowers_to_cuda` reads to tell a full compile from a truncated one.
FINAL_LOWERING_PASS = "lowering/cuda"
_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")
_REWRITE_APPLIED = object()


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


@dataclass
class Pattern:
    """One node in a chain-match pattern.

    ``constraints`` is a dict of ``field_name → expected_value`` checks
    applied to ``node.op`` (e.g. ``{"fn": "softmax"}``).
    """

    name: str
    op_type: type
    constraints: dict = field(default_factory=dict)


@dataclass
class Rule:
    """One rewrite rule loaded from a ``passes/<dir>/NNN_<name>.py``
    module.

    * ``name`` — the file stem (engine display + dump filenames).
    * ``pattern`` — the chain-match pattern the rule fires on.
    * ``rewrite`` — the rule's ``rewrite`` function. ``None`` for the
      no-rewrite stubs :meth:`Pipeline.from_pattern` builds for
      pattern-matching-only callers.
    * ``param_names`` — captured at load time so the dispatcher can
      bind each rewrite param via signature inspection. The binding
      rules (kept here so docstring + dataclass live together):

      - ``graph`` — the current ``Graph``
      - ``match`` — the full ``Match`` (escape hatch)
      - ``root`` — ``graph.nodes[match.root_node_id]``
      - ``out`` — ``root.output``
      - ``ctx`` — the engine's ``Context``
      - any ``Pattern.name`` declared in ``pattern`` — that pattern
        entry's matched ``Node``
      - anything else — bound positionally to the input ``Node`` at
        slot ``i``, ``None`` past the input count or for deleted
        source nodes.

    * ``fixpoint`` — repeat this rule after every successful rewrite and advance only when a
      complete match batch is quiescent.

    * ``pass_`` — backref to the owning ``Pass``. Stamped by ``Pass``
      at construction time; ``None`` only on stray ``Rule`` instances
      built outside a pipeline (none exist in production paths).
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | Op | None] | None = None
    param_names: tuple[str, ...] = field(default_factory=tuple)
    fixpoint: bool = False
    pass_: Pass | None = field(default=None, repr=False, compare=False)


class RuleSkipped(Exception):
    """Raised by a rule's ``rewrite()`` to signal that the match was
    considered but skipped, with a human-readable reason for why no
    rewrite was applied. The engine catches it, logs the reason at
    DEBUG (visible at ``compile -vv``), and treats the result the same
    as ``return None`` with no in-place mutation. Use this in place of
    a bare ``return None`` whenever the skip reason would help debug
    why a rule didn't fire on a given match.

    ``reject=True`` marks the skip as this node's LOWERING declining the offered row (the
    materializer's ``UnbindableProjection`` decline): it is recorded into the run's rejection
    sink so the greedy blocklist retry moves past the row. An ordinary skip records nothing."""

    def __init__(self, reason: str, *, reject: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.reject = reject


class LoweringError(Exception):
    """Raised by :meth:`Pipeline.run` when a deterministic (greedy)
    compile of a pipeline that runs to :data:`FINAL_LOWERING_PASS`
    finishes with a node still holding a ``LoopOp`` / ``TileOp``.

    Two shapes strand a node, and both raise here. Every option of its
    only lowering rule failed ``validate(ctx)`` — e.g. a tile shape whose
    materialized kernel exceeds the device smem cap — so the rule
    recorded a rejection and the error names the pass and the reason.
    Or no rule lowered it at all: the row's materializer declined with an
    ordinary ``RuleSkipped``, or no rule matched, and nothing was
    recorded; the error then names the node and the op it is stuck on.

    This converts the old silent leak (an un-lowered ``TileOp`` surviving
    every pass until ``CudaBackend`` raises the cryptic ``non-CudaOp``
    ``TypeError``) into an actionable, early error."""


@dataclass
class Pass:
    """One pipeline pass: a named, indexed list of rules.

    * ``name`` — pass directory name (e.g. ``"frontend/decomposition"``),
      or ``""`` for empty / nameless pass slots (early-exit stubs).
    * ``rules`` — the rules in this pass, in load order.
    * ``index`` — 0-based position in the pipeline.

    Stamps each rule's ``pass_`` backref on construction so a ``Match``
    that carries a ``Rule`` can resolve pass metadata without holding a
    separate index.
    """

    name: str
    rules: list[Rule]
    index: int = 0
    # Every ``Knob`` declared (or imported) by this pass's rule modules — the
    # knobs this pass "owns". Populated by :meth:`load` (scans each rule
    # module's ``vars()`` for ``Knob`` instances, so imported knobs like the
    # planner's ``_enumeration`` set count too). :meth:`Cursor.advance` stamps
    # any of these with a defined ``off`` onto the variant at the pass boundary.
    declared_knobs: tuple[Knob, ...] = ()

    def __post_init__(self) -> None:
        for r in self.rules:
            r.pass_ = self

    @classmethod
    def load(cls, name: str, index: int, select: set[str] | None = None) -> Pass:
        """Discover, import, and (optionally) filter the rule modules
        under ``passes/<name>/``. ``select``, when given, keeps only
        rules whose file stem — or stem with the numeric prefix
        stripped — appears in the set."""
        pass_dir = _PASSES_DIR / name
        rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
        rules: list[Rule] = []
        declared: dict[str, Knob] = {}  # knob name → Knob, deduped across rule modules
        for path in rule_files:
            if select is not None and path.stem not in select and _strip_rule_prefix(path.stem) not in select:
                continue
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load rule from {path}")
            module = importlib.util.module_from_spec(spec)
            # Register before exec so any ``@dataclass`` defined in the
            # rule module can resolve its own module via ``sys.modules``
            # — ``dataclasses._is_type`` looks up ``cls.__module__``
            # there to check for ``KW_ONLY`` and raises
            # ``AttributeError`` on a missing entry.
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            pattern = getattr(module, "PATTERN", None)
            rewrite_fn = getattr(module, "rewrite", None)
            if pattern is None:
                raise ValueError(f"Rule {path} missing PATTERN")
            if rewrite_fn is None:
                raise ValueError(f"Rule {path} missing rewrite() function")
            param_names = tuple(inspect.signature(rewrite_fn).parameters.keys())
            rules.append(
                Rule(
                    name=path.stem,
                    pattern=pattern,
                    rewrite=rewrite_fn,
                    param_names=param_names,
                    fixpoint=bool(getattr(module, "FIXPOINT", False)),
                )
            )
            # Collect the knobs this rule module declares OR imports (e.g. the
            # planner imports the ``_enumeration`` tier knobs) — ``Cursor.advance``
            # uses them to OFF-fill the pass's variants.
            for v in vars(module).values():
                if isinstance(v, Knob):
                    declared.setdefault(v.name, v)
        return cls(name=name, rules=rules, index=index, declared_knobs=tuple(declared.values()))


@dataclass
class Match:
    """Result of matching a pattern against a graph.

    ``graph`` is the graph this match was built against (rules access
    it via ``match.graph`` for ad-hoc lookups). ``nodes`` maps each
    pattern entry's name to the matched node id. ``consumed`` and
    ``output`` may be overwritten by the rewrite function to control
    which nodes the rewriter removes and which node its edges get
    redirected to. ``output`` defaults to ``root_node_id`` when left
    as ``None``.

    ``rule`` locates this match in the pipeline; the rewriter reaches
    pass metadata via ``match.rule.pass_``. Stamped by
    :meth:`Pipeline.match` at construction time. Run-scoped sinks (dump,
    rejections) live on the :class:`Run` — reached through the candidate,
    not the match. ``is_last`` is stamped on the last live match returned
    by :meth:`Pipeline.match` so ``Candidate.try_rewrite`` knows when to
    advance the cursor.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    rule: Rule
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    # ``str`` redirects one node's consumers to the fragment's sole output;
    # ``dict[old_id, frag_output_id]`` redirects several at once (multi-output
    # splice — see ``Graph.splice``). ``None`` defaults to ``root_node_id``.
    output: str | dict[str, str] | None = None
    is_last: bool = False
    # Strong snapshot of every consumed node. The ``is_alive`` check uses
    # object identity to detect removal followed by a different node at the
    # same graph id. Holding the object itself prevents CPython from recycling
    # its integer ``id()`` before the check.
    _identities: dict[str, Node] = field(default_factory=dict, repr=False)

    @property
    def root(self) -> Node:
        """The root ``Node`` (matched by the first ``Pattern`` entry)."""
        return self.graph.nodes[self.root_node_id]

    def node(self, name_or_id: str) -> Node:
        """Resolve a pattern name (e.g. ``"producer"``) OR a raw node id
        to the current ``Node`` in ``graph``. Raises ``KeyError`` if the
        node has been removed."""
        nid = self.nodes.get(name_or_id, name_or_id)
        return self.graph.nodes[nid]

    def input(self, i: int) -> Node | None:
        """The node producing the root's ``i``-th input buffer, or ``None``
        when ``i`` exceeds the input count or the producer was removed."""
        root = self.root
        if i >= len(root.inputs):
            return None
        return self.graph.producer(root.inputs[i])

    def is_alive(self) -> bool:
        """``True`` when every watched node still resolves to the same
        ``Node`` object captured at match time. Catches both removal and
        the "removed-then-re-added under same id" case."""
        for nid in self._identities:
            n = self.graph.nodes.get(nid)
            if n is not self._identities[nid]:
                return False
        return True

    def remap(self, graph: Graph) -> Match:
        """Build a fresh ``Match`` against ``graph`` (a copy of the
        original) that mirrors this match's ids. Re-snapshots
        ``_identities`` against the new graph's nodes so ``is_alive``
        still works after the copy. Used when materializing a lazy
        fork — the fork copies the parent's snapshot, then needs a
        match anchored on the copy."""
        identities = {nid: graph.nodes[nid] for nid in self._identities if nid in graph.nodes}
        return Match(
            graph=graph,
            root_node_id=self.root_node_id,
            rule=self.rule,
            nodes=dict(self.nodes),
            consumed=set(self.consumed),
            output=self.output,
            is_last=self.is_last,
            _identities=identities,
        )


def _off_fill_pass(graph: Graph, pass_: Pass) -> None:
    """Stamp every OFF-declared knob of ``pass_`` that a variant left unspecified
    onto that variant — the "every emitted variant carries an explicit value for
    every knob the pass declares" rule, realized once at the pass boundary (so
    all the pass's rules — including a declined / no-variant rule — have had
    their turn). Rebuilds the op via :func:`dataclasses.replace` (a fresh
    ``knobs`` dict, never an in-place mutation) so a structurally shared op isn't
    corrupted across sibling candidates. Only ops that already carry tuning
    knobs (a realized kernel variant) are touched — inputs / constants with an
    empty ``knobs`` are left alone."""
    if not pass_.declared_knobs:
        return
    for node in graph.nodes.values():
        knobs = getattr(node.op, "knobs", None)
        if not knobs:
            continue
        filled = apply_off_defaults(dict(knobs), pass_.declared_knobs)
        if filled != knobs:
            node.op = replace(node.op, knobs=filled)


@dataclass
class Cursor:
    """Pipeline resume state for a candidate. Owns the entire advance
    logic: ``advance(graph)`` moves past the current rule batch,
    wrapping to the next pass — logging "compile: <pass> done" and
    flushing ``run.dump.on_pass`` — when the scan completes with
    no functional rewrites.

    * ``run`` — the :class:`Run` being driven; resolves the pipeline
      (current pass / rule by index) and the per-run dump sink.
    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass.
    """

    run: Run
    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    @property
    def is_done(self) -> bool:
        return self.pass_idx >= len(self.run.pipeline.passes)

    @property
    def current_pass(self) -> Pass:
        assert not self.is_done, f"cursor is done (pass_idx={self.pass_idx} >= {len(self.run.pipeline.passes)})"
        return self.run.pipeline.passes[self.pass_idx]

    @property
    def current_rule(self) -> Rule:
        pass_ = self.current_pass
        assert self.rule_idx < len(pass_.rules), f"rule_idx={self.rule_idx} out of range for pass {pass_.name!r} ({len(pass_.rules)} rules)"
        return pass_.rules[self.rule_idx]

    def advance(self, graph: Graph) -> None:
        """Move past the just-finished rule batch. Wraps to the next
        pass (logging done + flushing ``run.dump.on_pass``) when
        the scan completes with no functional rewrites; otherwise
        restarts the scan from rule 0 to apply newly-spawned matches.
        ``graph`` is the candidate's current graph — passed in so the
        on-pass dump and node-count debug line have something to
        report. Raises if the cursor is already done."""
        pass_ = self.current_pass  # asserts not is_done
        self.rule_idx += 1
        if self.rule_idx < len(pass_.rules):
            return
        finished = self.n_applied == 0
        self.rule_idx = 0
        self.n_applied = 0
        if finished:
            if pass_.name:
                _off_fill_pass(graph, pass_)
                logger.debug("compile: %-18s done (%d nodes)", pass_.name, len(graph.nodes))
                if self.run.dump is not None:
                    self.run.dump.on_pass(pass_, graph)
                event = PassEndEvent(
                    pass_name=pass_.name,
                    graph=graph,
                    ctx=self.run.ctx,
                    passes=tuple(p.name for p in self.run.pipeline.passes),
                )
                for strat in self.run.pipeline.strategies:
                    strat.on_pass_end(event)
            self.pass_idx += 1


@dataclass(frozen=True)
class Pipeline:
    """Frozen, shareable pass layout of the rewrite pipeline — nothing
    run-scoped lives here (that's :class:`Run`), so one Pipeline can
    drive any number of concurrent runs.

    :meth:`match` is the only entry point for pattern matching: it
    walks the graph for one rule and stamps the rule onto every Match.
    Tests / standalone callers that just want pattern matching can
    build a one-rule Pipeline via :meth:`from_pattern`.
    """

    passes: list[Pass]
    # Engine-event strategies (``PipelineStrategy`` instances): the discovered,
    # stateless set (``strategy.discovered_strategies``) plus whatever a caller
    # composed in via :meth:`with_strategies`. Empty for ``from_pattern`` test
    # shims. A pipeline composed with STATEFUL strategies (e.g. the two-level
    # kernel inventory) serves ONE run — sharing across runs is
    # only safe when every strategy is stateless.
    strategies: tuple = ()

    @property
    def lowers_to_cuda(self) -> bool:
        """Whether this pass layout runs to :data:`FINAL_LOWERING_PASS`, i.e. promises a terminal
        whose every compute node is a ``CudaOp``. Only such a pipeline may treat a surviving
        ``LoopOp`` / ``TileOp`` as a stranded node: a truncated build (``TILE_PASSES``,
        ``LOOP_PASSES``, a one-rule test shim) terminates in an earlier dialect by design."""
        return bool(self.passes) and self.passes[-1].name == FINAL_LOWERING_PASS

    def match(self, graph: Graph, rule: Rule) -> list[Match]:
        """Enumerate every live pattern match for ``rule`` against
        ``graph``. Stamps ``is_last=True`` on the last surviving match
        so the rewriter knows which apply closes out the rule batch
        (cursor advance flows through ``Candidate.try_rewrite`` /
        ``Candidate.apply``). Drops matches that fail
        :meth:`Match.is_alive` — an earlier match in the same batch
        may have removed a consumed node."""
        results: list[Match] = []
        for nid in graph.topological_order():
            m = _match_at(graph, nid, rule)
            if m is not None and m.is_alive():
                results.append(m)
        if results:
            results[-1].is_last = True
        return results

    @classmethod
    def from_pattern(cls, pattern: list[Pattern]) -> Pipeline:
        """Test/standalone helper: build a single-pass, single-rule
        Pipeline whose only rule wraps ``pattern`` (no ``rewrite``).
        Lets pattern-matching tests drive :meth:`match` without
        setting up the full engine pipeline."""
        rule = Rule(name="__test__", pattern=pattern)
        return cls(passes=[Pass(name="__test__", rules=[rule], index=0)])

    @classmethod
    def build(cls, passes: list[str], *, select: Iterable[str] | None = None) -> Pipeline:
        """Load each named pass directory into a :class:`Pass` and
        assemble them into a Pipeline. ``select``, when given, filters
        rules whose stem (with or without numeric prefix) appears in
        the set."""
        select_set = set(select) if select is not None else None
        return cls(passes=[Pass.load(name, i, select_set) for i, name in enumerate(passes)], strategies=discovered_strategies())

    def with_strategies(self, *extra) -> Pipeline:
        """This pipeline with ``extra`` engine-event strategies composed after the existing
        set — how a caller installs PER-RUN strategies (e.g. the kernel inventory). The returned
        pipeline serves one run when any composed strategy holds state."""
        return replace(self, strategies=(*self.strategies, *extra))

    def run(
        self,
        graph: Graph,
        *,
        ctx: Context | None = None,
        backend=None,
        db: SearchDB | None = None,
        dump: CompilerDump | None = None,
    ) -> Graph:
        """Single-shot greedy compile — a deterministic resolution
        (:meth:`Run.resolve`) with the greedy pick
        (:func:`~emmy.compiler.pipeline.search.policy.greedy.greedy_decide`):
        at every fork point, flatten to complete leaves and take the
        ``Prior``'s ``mean_scores`` argmin. Not a search — no frontier,
        no tree, no benching. The input ``graph`` is copied once per
        attempt and resolved in place — no per-fork graph copies.

        ``ctx`` is built once (probing the live device if not provided)
        and passed to every rule that takes a ``ctx`` parameter.

        ``backend`` (typically :class:`CudaBackend`) opts the run into
        real GPU measurement: the terminal graph's per-kernel latency is
        recorded to ``db`` once after the resolution settles. ``db``
        defaults to a fresh in-memory store;
        pass an explicit :class:`SearchDB` to persist measurements
        across runs.

        Retries are ``decide`` wrappers over a deterministic re-resolve
        — cheap non-chronological backtracking with no graph snapshots
        or undo log, since every other choice replays identically:

        * **Validity fallback** — the prior ranks by predicted latency
          and can rank a tile that fails ``validate(ctx)`` (smem /
          thread budget) first; greedy benches nothing, so on a left-un-lowered node we
          blocklist its tile and re-resolve, falling back to the next
          prior-ranked leaf. Bounded retries (each adds ≥1 block or
          stops).
        * **Structural retirement** — a refused row on a fragment
          kernel blocklists at that piece's own schedule fork and the
          composed route replays; only once no row of the piece binds
          is the cut that minted it (the ``Graph`` decision whose
          ``minted`` ids name the piece) blocklisted at its own fork,
          so that one splice is withdrawn and every other kernel-set
          decision stays priced by the evidence.

        Retry orchestration, the ``rejections`` sink, and the loud
        :class:`LoweringError` on an un-lowered node are greedy search
        POLICY, owned by
        :class:`~emmy.compiler.pipeline.search.strategy.greedy.GreedyStrategy` —
        this method is the thin engine entry point."""
        from emmy.compiler.ir.schedule.base import clear_pin_refusals  # noqa: PLC0415
        from emmy.compiler.pipeline.search.strategy import GreedyStrategy  # noqa: PLC0415

        clear_pin_refusals()  # a refusal names the compile it happened in, not an earlier one
        return GreedyStrategy(self, backend=backend, db=db, dump=dump).run(graph, ctx)


@dataclass
class ForkPoint:
    """What a :meth:`Run.resolve` ``decide`` callback sees at one
    multi-option rewrite: the live :class:`Match`, the raw ``options``
    list exactly as ``Candidate.try_rewrite`` returned it (concrete
    ``Op``/``Graph`` leaves and lazy ``Fork``s — branch Forks included,
    unexpanded), the pre-decision root op, and the run's ``ctx``.
    ``resolve`` holds one live graph and applies the chosen option in place.

    ``score`` is the decide callback's one output channel besides its
    return value: a decide that ranks options with a prior stamps the
    chosen option's predicted µs here, and ``resolve`` copies it onto the
    fork's :class:`Decision` trace entry (where e.g. the structural
    pricing probe reads a kernel's price off the partition fork)."""

    match: Match
    options: list
    root_op: Op
    ctx: Context
    score: float | None = None

    @property
    def node_id(self) -> str:
        """The graph node this fork is rewriting — the blocklist / trace key."""
        return self.match.root_node_id

    @property
    def structural(self) -> bool:
        """Whether this fork can change the kernel SET — derived from the typed partition, so the
        fact has one definition."""
        return bool(self.splices)

    # The offer's TYPED partition. Structural options are top-level siblings by construction
    # (``_is_structural_option``: schedule-product branches contain only ``TileOp`` leaves), so
    # the engine can classify without expanding anything, and consumers read the partition
    # instead of each re-deriving it from the raw list.
    # The offer's LEAVES. Every consumer that walks, flattens or searches an offer goes through these three, so
    # the depth-first emission order a score tie falls back to (option-0 first) and the one row-directed descent
    # have a single home; the ``fork`` module's free functions stay for option sequences that are no fork point
    # (a rule's forks before they are offered, a filtered sibling list).
    def leaves(self) -> Iterator:
        """The offer's complete leaves, depth-first in emission order, streamed without retaining the expanded
        tree — ``next(fp.leaves())`` is what emission order decides."""
        return iter_leaves(self.options)

    def flat(self) -> list:
        """:meth:`leaves` as a list, for the small non-schedule forks whose alternatives are compared together."""
        return list(self.leaves())

    def find(self, row: Mapping, *, skip: Callable[[dict], bool] | None = None):
        """The first leaf a (possibly partial) knob ``row`` vouches for, as ``(leaf, its knobs)``, or ``None`` —
        the one row-directed descent the evidence pick, the decision memo's replay and the golden replay share
        (:func:`~emmy.compiler.pipeline.fork.leaf_for`: a schedule root is narrowed to the row, so the walk below
        it is one path). ``skip`` drops a leaf by its knobs (a blocklisted tile)."""
        return leaf_for(self.options, row, skip=skip)

    @cached_property
    def splices(self) -> tuple:
        """The structural (``Graph``-splicing, kernel-set-changing) offers."""
        return tuple(o for o in self.options if _is_structural_option(o))

    @cached_property
    def variants(self) -> tuple:
        """The op-variant offers — the schedule pool this fork ranks in place."""
        return tuple(o for o in self.options if not _is_structural_option(o))


#: A deterministic policy found no complete option below a lazy fork. Resolve uses this explicit
#: result to continue the current rule batch without applying a rewrite.
NO_OPTION = object()


@dataclass(frozen=True)
class Decision:
    """One :meth:`Run.resolve` trace entry — what a deterministic
    resolution decided at one fork point. The trace is the resolution's
    only output channel besides the terminal graph: process facts
    (structural picks taken, per-fork predicted cost) are trace queries,
    never policy-object state.

    * ``rule_name`` / ``node_id`` — where the fork was offered.
    * ``chosen_kind`` — ``"graph"`` for a structural (``Graph``-splicing)
      pick, ``"op"`` for an in-place rebind.
    * ``knob_delta`` — the chosen option's knob identity: a ``Fork``'s
      pinned row, an ``Op``'s own knobs, a ``Graph``'s decision-knob delta
      vs the offer op (:func:`_option_decision`).
    * ``score`` — the decide callback's predicted µs for the pick
      (``None`` when the decide didn't rank, e.g. option-0 fallback).
    * ``n_options`` — raw option count at the fork (a lazy fork tree
      counts as one — its leaves are the decide callback's to expand).
    * ``minted`` — the graph ids of the kernels a ``"graph"`` pick spliced in (the splice
      receipt's post-promotion ids), so a piece that later fails to lower can be traced back
      to the cut that minted it; empty for an ``"op"`` pick."""

    rule_name: str
    node_id: str
    chosen_kind: str
    knob_delta: dict
    score: float | None
    n_options: int
    minted: tuple[str, ...] = ()


@dataclass
class Run:
    """Mutable per-run state of ONE drive of a pipeline — everything
    scoped to a single compile invocation lives here, so
    :class:`Pipeline` stays a frozen, shareable pass layout and nothing
    run-scoped is ever smuggled onto shared objects.

    * ``pipeline`` — the frozen pass layout being driven.
    * ``ctx`` — the resolved hardware context, shared by every candidate
      (reached as ``cand.ctx``).
    * ``db`` — the tune DB the resolution reads its measured evidence from.
    * ``backend`` — optional measurement backend (``None`` = stub bench,
      no persistence).
    * ``dump`` — optional artifact collector: :meth:`Candidate._log_apply`
      routes per-rule diffs through ``dump.on_rule``, :meth:`Cursor.advance`
      routes post-pass graphs through ``dump.on_pass``.
    * ``rejections`` — optional sink for rewrites whose every option
      failed ``validate(ctx)`` (installed by :meth:`Pipeline.run` so
      greedy compiles can raise :class:`LoweringError`).

    Candidates and cursors hold a back-reference to their Run, so
    engine-adjacent code reads run state off the object at hand
    (``cand.run.dump``) instead of threading six arguments around."""

    pipeline: Pipeline
    ctx: Context
    db: SearchDB | None = None
    backend: object | None = None
    dump: CompilerDump | None = None
    rejections: list[tuple[str, str, str]] | None = None

    def _step(
        self,
        cand: Candidate,
        decide: Callable[[ForkPoint], object] | None = None,
        trace: list[Decision] | None = None,
    ) -> tuple[Match, list, bool] | None:
        """Run one rule batch against ``cand`` — the per-candidate engine
        body of :meth:`resolve`. Single-option
        rewrites apply inline (via ``Candidate.try_rewrite``), empty /
        quiescent batches advance the cursor, and a structural fork whose
        offer site was already decided on this trajectory replays that
        side inline (:func:`_replay_structural_decision`). Returns ``None``
        when the batch completed with nothing left to decide, or
        ``(match, options, structural)`` at the first undecided multi-option fork when ``decide``
        is absent. With ``decide``, resolve the fork in place; :data:`NO_OPTION` skips an empty
        lazy subtree and continues with the remaining matches in the same batch."""
        cur = cand.cursor
        pass_ = cur.current_pass
        # Empty pass (e.g. all rules filtered out) OR no live matches →
        # no apply fires → advance the cursor directly so the caller's
        # loop doesn't re-run the same rule batch forever. ``advance``
        # handles both cases uniformly: with ``n_applied == 0`` it wraps
        # to the next pass and fires the post-pass log + dump.
        if not pass_.rules:
            cur.advance(cand.graph)
            return None
        matches = self.pipeline.match(cand.graph, cur.current_rule)
        if not matches:
            cur.advance(cand.graph)
            return None
        for match in matches:
            options = cand.try_rewrite(match)
            if options is _REWRITE_APPLIED:
                if match.rule.fixpoint:
                    return None
                continue
            if options is None:
                continue
            # The fork is classified here, where the raw ``options`` list
            # is concrete (no thunk fired): any ``Graph`` option makes the
            # fork **structural** (kernel-set-changing); pure ``Op``
            # rebinds (and the partition planner's branch Forks) are
            # op-variant.
            structural = any(_is_structural_option(o) for o in options)
            # A structurally identical offer site already decided in the SAME cut domain on this
            # trajectory takes the exact same knob receipt inline. Domain is part of the key:
            # placement and cross-CTA cuts may both mint two kernels, but neither can replay the
            # other's choice.
            domain = _structural_domain(options) if structural else None
            if structural and (chosen := _replay_structural_decision(cand.structural_decisions, match.root.op, options)) is not None:
                option = _concrete_option(chosen)
                cand.apply(match, option, knobs=_choice_knobs(chosen, option, match.root.op), aliases=getattr(chosen, "aliases", None))
                if match.rule.fixpoint:
                    return None
                continue
            if decide is not None:
                root_op = match.root.op
                fp = ForkPoint(match=match, options=options, root_op=root_op, ctx=self.ctx)
                choice = decide(fp)
                if choice is NO_OPTION:
                    cand._advance_if_last(match)
                    continue
                option = _concrete_option(choice)
                if option is None:
                    raise ValueError(f"decide returned a branch Fork at {match.rule.name!r} — return a concrete option or a leaf Fork")
                knob_delta = _choice_knobs(choice, option, root_op)
                minted = cand.apply(match, option, knobs=knob_delta, aliases=getattr(choice, "aliases", None))
                if domain is not None:
                    _remember_structural_decision(cand.structural_decisions, root_op, domain, knob_delta)
                assert trace is not None
                trace.append(
                    Decision(
                        rule_name=match.rule.name,
                        node_id=fp.node_id,
                        chosen_kind="graph" if isinstance(option, Graph) else "op",
                        knob_delta=knob_delta,
                        score=fp.score,
                        n_options=len(options),
                        minted=minted or (),
                    )
                )
                return None
            return match, options, structural
        return None

    def resolve(self, graph: Graph, decide: Callable[[ForkPoint], object]) -> tuple[Graph, list[Decision]]:
        """Deterministic resolution — fold the pipeline over ``graph``
        IN PLACE, asking ``decide`` at every undecided fork point, and
        return ``(terminal_graph, trace)`` — greedy compile, structural
        pricing probes, assembled-graph lowering: one live graph, no
        per-fork graph copies — the returned terminal IS the seeded
        ``graph`` object.

        ``decide`` receives a :class:`ForkPoint` and returns the option
        to apply — a concrete ``Op`` / ``Graph`` from the fork's raw
        options, or a **leaf** ``Fork`` (a decide that wants a lazy fork
        tree's complete rows expands branch Forks itself; returning a
        branch Fork is an error). It may stamp ``fp.score`` with the
        pick's predicted µs — copied onto the trace entry.

        The trace (one :class:`Decision` per decided fork, in resolution
        order) is the only output channel besides the terminal graph:
        "did this compile take a structural pick", "what did the
        partition fork predict for this kernel" are trace queries, not
        accumulated policy state. Inline replays of an already-decided
        structural offer site (see :meth:`_step`) are not decisions and
        don't trace."""
        from emmy.compiler.pipeline.search.candidate import Candidate  # noqa: PLC0415

        event = RunStartEvent(graph=graph, ctx=self.ctx, passes=tuple(p.name for p in self.pipeline.passes))
        for strat in self.pipeline.strategies:
            strat.on_run_start(event)
        cand = Candidate(run=self, graph=graph, cursor=Cursor(run=self))
        trace: list[Decision] = []
        while not cand.cursor.is_done:
            self._step(cand, decide=decide, trace=trace)
        return cand.graph, trace


def _is_structural_option(option: object) -> bool:
    """Classify one raw rewrite option by its effect: a ``Graph`` splice
    changes which ops exist — **structural**; an ``Op`` rebind is in-place —
    **op-variant**. A deferred leaf declares the same fact through ``Fork.structural`` so graph
    construction stays lazy. A branch Fork reads op-variant: schedule-product branches contain
    only TileOp leaves, and typing them would require the expansion their lazy structure avoids."""
    return isinstance(option, Graph) or (isinstance(option, Fork) and option.structural)


def _concrete_option(option: object) -> object | None:
    """Unwrap one raw rewrite option to the concrete ``Op`` / ``Graph`` a
    replayed structural decision can apply inline: leaf Forks fire their
    single-element thunk, concrete options pass through, and a *branch* Fork
    returns ``None`` (un-applyable without a full expand — the caller falls
    back to forking normally; no structural branch Fork exists today)."""
    if isinstance(option, Fork):
        return option.expand()[0] if option.is_leaf else None
    return option


def _option_decision(option: object, root_knobs: dict) -> dict | None:
    """The decision-knob delta one raw structural-fork option would stamp vs the offer op: the knobs
    the option's op / fork knobs **add or change** vs the offer. A ``Graph`` option reads the union
    over its nodes' op knobs: a fragment's kernels are brand-new ones carrying only whatever decision
    the rule stamped, so that union IS the decision. A *changed value* on an existing key counts, not
    only a brand-new key. ``None`` when the option stamps nothing new."""
    if isinstance(option, Graph):
        knobs: dict = {}
        for node in option.nodes.values():
            knobs.update(getattr(node.op, "knobs", None) or {})
    else:
        knobs = getattr(option, "knobs", None) or {}
    delta = {k: v for k, v in knobs.items() if root_knobs.get(k) != v}
    return delta or None


def _choice_knobs(choice: object, option: object, root_op) -> dict:
    """The chosen option's knob identity for a :class:`Decision` trace entry:
    a ``Fork``'s pinned row when it carries one, a ``Graph``'s decision-knob
    delta vs the offer op (:func:`_option_decision`), an ``Op``'s own knobs.
    ``choice`` is what ``decide`` returned (possibly a leaf Fork); ``option``
    is its unwrapped concrete ``Op`` / ``Graph``."""
    if isinstance(choice, Fork) and choice.knobs:
        return dict(choice.knobs)
    if isinstance(option, Graph):
        return _option_decision(option, root_op.knobs) or {}
    return dict(getattr(option, "knobs", None) or {})


def _option_receipt(option: object, root_knobs: dict) -> dict:
    """The exact decided-knob receipt carried by one raw fork option, without materializing it."""
    return dict(option.knobs) if isinstance(option, Fork) else _option_decision(option, root_knobs) or {}


def _structural_domain(options: list) -> tuple[str, ...] | None:
    """The knob families that identify one structural fork domain."""
    families = {family_of(key) for option in options for key in _option_receipt(option, {}).keys()}
    return tuple(sorted(families)) or None


def _remember_structural_decision(decisions: list, root_op, domain: tuple[str, ...], receipt: dict) -> None:
    """Record the first exact structural choice for an identical offer in one cut domain."""
    key = root_op.identity_key(with_io=True, with_knobs=True)
    if key is None or not receipt or any(prior_key == key and prior_domain == domain for prior_key, prior_domain, _ in decisions):
        return
    decisions.append((key, domain, dict(receipt)))


def _replay_structural_decision(decisions: list, root_op, options: list) -> object | None:
    """The offered option (a leaf Fork, or a concrete ``Op`` / ``Graph``) a structurally identical,
    already-decided offer site on this trajectory took — or ``None`` (undecided / unmatchable → fork
    normally).

    A candidate keeps the first decision as ``(root identity, cut domain, exact knob receipt)``.
    Replay therefore preserves the old one-decision-per-identical-kernel bound without inferring a
    semantic decision from its output count. If the earlier receipt is not an exact option here,
    this site remains a real fork."""
    key = root_op.identity_key(with_io=True, with_knobs=True)
    domain = _structural_domain(options)
    if key is None or domain is None:
        return None
    receipt = next((prior for prior_key, prior_domain, prior in decisions if prior_key == key and prior_domain == domain), None)
    if receipt is None:
        return None
    matches = [option for option in options if _option_receipt(option, root_op.knobs) == receipt]
    if len(matches) != 1 or (isinstance(matches[0], Fork) and not matches[0].is_leaf):
        return None
    return matches[0]


def _match_at(graph: Graph, start: str, rule: Rule) -> Match | None:
    nid: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, Node] = {}
    matched_nodes: list[Node] = []
    for prod in rule.pattern:
        if nid is None:
            return None
        node = graph.nodes.get(nid)
        if node is None or not isinstance(node.op, prod.op_type):
            return None
        if not all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items()):
            return None
        nodes[prod.name] = nid
        consumed.add(nid)
        identities[nid] = node
        matched_nodes.append(node)
        consumers = graph.consumers(nid)
        nid = consumers[0] if len(consumers) == 1 else None
    # Snap ``op.inputs`` / ``op.outputs`` to the surrounding graph for
    # every matched node so rule rewrites can read per-buffer Tensors
    # straight off the op without re-querying the graph.
    for node in matched_nodes:
        node.op = node.op.with_io(graph, node)
    return Match(
        graph=graph,
        root_node_id=start,
        rule=rule,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


__all__ = ["Decision", "ForkPoint", "LoweringError", "Match", "Pass", "Pattern", "Pipeline", "Rule", "RuleSkipped"]
