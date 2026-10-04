"""The :class:`Candidate` — the concrete graph state one :class:`Run` resolves in place."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from emmy.compiler.graph import Graph, Tensor, _fmt_op
from emmy.compiler.ir.base import ConstantOp, InputOp, Op
from emmy.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.pipeline import _REWRITE_APPLIED, Cursor, RuleSkipped
from emmy.compiler.pipeline.rule_diff import display_name, emit, format_skipped, render_rule_diff
from emmy.compiler.pipeline.strategy import RebindEvent, SplicedEvent, SpliceEvent

# Use the engine logger so the existing debug-emit toggles (rule-
# skipped lines under ``compile -vv``) keep working without callers
# having to also bump this module's level.
_logger = logging.getLogger("emmy.compiler.pipeline")

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.pipeline import Match, Run


@dataclass
class Candidate:
    """A concrete point in the search space — owns a real ``graph``.

    ``run`` is the :class:`Run` this candidate belongs to, shared by
    reference across siblings — it resolves the hardware ``ctx``
    (exposed as :attr:`ctx`) and the run-scoped sinks the candidate
    reports into (``run.dump`` inside :meth:`_log_apply`, the
    ``run.rejections`` list inside :meth:`try_rewrite`). ``cursor``
    tracks pipeline resume state."""

    run: Run
    graph: Graph
    cursor: Cursor
    # Exact structural decisions already made on this trajectory. The domain is part of each key,
    # so two kernel-set rewrites over the same op never substitute for one another.
    structural_decisions: list[tuple[object, tuple[str, ...], dict]] = field(default_factory=list)

    @property
    def ctx(self) -> Context:
        return self.run.ctx

    def try_rewrite(self, match: Match) -> list[Op | Graph] | object | None:
        """Eager mode (called by the engine loop): invoke
        ``match.rule.rewrite`` against this candidate's graph,
        validate the result, and either apply the single chosen
        option or — for a multi-option fork — return the option list
        for the caller to decide over.
        Returns ``None`` when no rewrite was applied (``RuleSkipped`` or empty options after
        validation). A single concrete rewrite returns the private applied sentinel so a fixpoint
        rule can restart immediately.

        A skip or invalid option advances unconditionally on ``match.is_last``, so quiescent
        batches terminate. A successful fixpoint rewrite keeps the cursor on the same rule; other
        successful rewrites advance normally. A multi-option return leaves advancement to the
        eventual fork's apply on resolve."""
        if not match.is_alive():
            # Earlier applies in this batch invalidated the match's
            # consumed nodes. Skip the rewrite, but still advance the
            # cursor when this was the last match — otherwise the
            # search loop would re-pop the same rule batch forever.
            self._advance_if_last(match)
            return None
        # Refresh each consumed node's op I/O against the graph as it stands
        # NOW: an earlier apply in this batch (e.g. the flash fragment splice)
        # may have swapped a consumed node's op for a rebuilt instance whose
        # ``inputs`` are still ``_seed_io_placeholders``' ``(f32, ())`` stubs —
        # ``is_alive``'s node-identity check cannot see an op swap, and a rule
        # reading placeholder dtypes mis-schedules (gemma o_proj deployed a
        # scalar tile 16x its own measured mma rows because the warp atom gate
        # read the placeholder f32). Same contract as the match-time refresh;
        # idempotent when nothing changed.
        for nid in match.consumed:
            node = match.graph.nodes.get(nid)
            if node is not None:
                node.op = node.op.with_io(match.graph, node)
        rule = match.rule
        try:
            result = rule.rewrite(**_build_rewrite_kwargs(rule, match, self.ctx))
        except RuleSkipped as exc:
            if _logger.isEnabledFor(logging.DEBUG):
                emit(format_skipped(display_name(rule.pass_.name if rule.pass_ else None, rule.name), match.root_node_id, exc.reason))
            # A REJECTING skip — the node's lowering declining the offered row (the
            # materializer's ``UnbindableProjection`` decline) — records into the run's rejection
            # sink like the all-options-filtered case below, so the greedy blocklist retry moves
            # past the row instead of dying downstream. Ordinary skips record nothing: passes
            # skip benignly on nodes that legitimately outlive them.
            if exc.reject and self.run.rejections is not None:
                self.run.rejections.append(
                    (match.root_node_id, display_name(rule.pass_.name if rule.pass_ else None, rule.name), exc.reason)
                )
            self._advance_if_last(match)
            return None
        raw_options = list(result) if isinstance(result, (list, tuple)) else [result]
        # ``Fork`` options pass through unconditionally — they're deferred
        # expansions with no graph to validate yet. Concrete ``Op`` leaves
        # still get the per-ctx validate filter; ``Graph`` splices are
        # validated at splice time, not here.
        options = [o for o in raw_options if not isinstance(o, Op) or o.validate(self.ctx)]
        if not options:
            # Validation-filtered rewrite: the rule produced output but
            # every option failed ``validate(ctx)`` — most commonly the
            # ``KernelOp.validate`` smem-cap check after
            # ``100_materialize_tile`` produces a kernel that exceeds
            # ``ctx.max_dynamic_smem``. In a *fork* this is legitimate
            # pruning (sibling branches carry other tile shapes), but in a
            # deterministic single-path compile it leaves the node
            # un-lowered with no recourse — so we both (a) emit a debug
            # "filtered" line and (b) record the rejection into the
            # run's optional sink. Which nodes are un-lowered is read off
            # the settled terminal rather than off this sink (greedy's
            # ``_stuck``); what the sink adds is the pass and the reason
            # the loud ``LoweringError`` names, turning the old
            # "CudaBackend: non-CudaOp TileOp" mystery into an actionable
            # error. The sink is absent under ``tune`` so the fork-pruning
            # path keeps its zero-overhead silent behavior.
            sink = self.run.rejections
            if raw_options and (sink is not None or _logger.isEnabledFor(logging.DEBUG)):
                rejected = [o for o in raw_options if isinstance(o, Op)]
                reasons = [r for r in (_validate_reason(o, self.ctx) for o in rejected) if r]
                reason_str = "; ".join(reasons) if reasons else "validate(ctx)=False"
                pass_label = display_name(rule.pass_.name if rule.pass_ else None, rule.name)
                if sink is not None:
                    sink.append((match.root_node_id, pass_label, reason_str))
                if _logger.isEnabledFor(logging.DEBUG):
                    emit(
                        format_skipped(pass_label, match.root_node_id, f"all {len(rejected)} option(s) failed validate(ctx): {reason_str}")
                    )
            self._advance_if_last(match)
            return None
        if len(options) > 1 or isinstance(options[0], Fork):
            # Defer to a fork — the caller decides over the options.
            # Single-option ``Fork`` also goes through this path: the
            # decide callback expands the thunk, which can't happen via
            # the inline apply path below. Cursor advance for both cases
            # happens via the eventual leaf's apply on resolve.
            return options
        self.apply(match, options[0])
        return _REWRITE_APPLIED

    def apply(self, match: Match, option: Op | Graph, *, knobs: dict | None = None, aliases: dict | None = None) -> tuple[str, ...] | None:
        """Apply mode (called by ``Run.resolve`` for a decided fork and
        internally by :meth:`try_rewrite` for single-option matches):
        apply the specific ``option`` to this candidate's graph.
        Mutates the graph, logs the rewrite (debug diff +
        ``run.dump.on_rule`` snapshot), bumps cursor ``n_applied`` for functional
        splices, and advances the rule-batch cursor when
        ``match.is_last``. Returns the graph ids a ``Graph`` splice minted
        (the receipt's post-promotion compute ids), ``None`` for an ``Op`` rebind.

        ``Op`` rebinds ``root.op`` (id / inputs / hints kept);
        ``Graph`` is a fragment spliced via ``Graph.splice``. On the
        ``Op`` path the chain ``Op.source`` is stamped with the op
        being replaced and the predecessor's ``knobs`` are merged
        forward — so the rewrite chain threads through every in-place
        rebind for free. The stamp is UNCONDITIONAL (engine-owned):
        "the op this op replaced at this node" is a fact about the
        rewrite, not the rule's to set — a rule building its option
        via ``dataclasses.replace(root.op, ...)`` copies the root's
        own ancestor into ``source``, and honoring that copy would
        skip the replaced op in the chain (and silently disable the
        knob merge, which is idempotent for rules that already merged
        manually). Knobs are NOT merged forward on the ``Graph`` path —
        fragment kernels are kernels of their own. The selected
        fork's delta instead rides ``SpliceEvent.knobs`` (with the fork's
        ``aliases``) for strategies that need the consumed parent's route identity.

        What a splice MEANS in any dialect is strategy business, not
        the engine's: ``on_splice`` fires before the splice (fragment
        op identities stable — where the provenance strategy threads
        attribution) and ``on_spliced`` after it, carrying the splice's
        receipt (where it threads op provenance). See ``pipeline.strategy``."""
        self._log_apply(match, option)
        minted = None
        if isinstance(option, Op):
            node = self.graph.nodes[match.root_node_id]
            old_op = node.op
            if option is not old_op:
                option = replace(option, source=old_op, knobs={**old_op.knobs, **option.knobs})
            node.op = option
            pass_ = match.rule.pass_
            event = RebindEvent(
                match=match, node=node, replaced=old_op, pass_name=pass_.name if pass_ is not None else "", graph=self.graph
            )
            for strat in self.run.pipeline.strategies:
                strat.on_rebind(event)
        else:
            assert isinstance(option, Graph), f"expected Graph or Op; got {type(option).__name__}"
            pass_ = match.rule.pass_
            pass_name = pass_.name if pass_ is not None else ""
            strategies = self.run.pipeline.strategies
            event = SpliceEvent(
                match=match,
                fragment=option,
                root_op=self.graph.nodes[match.root_node_id].op,
                pass_name=pass_name,
                graph=self.graph,
                knobs=dict(knobs or {}),
                aliases=dict(aliases or {}),
            )
            for strat in strategies:
                strat.on_splice(event)
            receipt = self.graph.splice(option, consumed=match.consumed, output=match.output or match.root_node_id)
            spliced = SplicedEvent(graph=self.graph, pass_name=pass_name, receipt=receipt)
            for strat in strategies:
                strat.on_spliced(spliced)
            self.cursor.n_applied += 1
            minted = receipt.new_compute_ids
        self._advance_if_last(match, applied=True)
        return minted

    def _log_apply(self, match: Match, option: Op | Graph) -> None:
        """Render a per-rule diff at DEBUG and route a structured
        record to ``run.dump`` when set. Returns early when
        neither sink is active."""
        rule = match.rule
        pass_ = rule.pass_
        dump = self.run.dump
        debug_on = _logger.isEnabledFor(logging.DEBUG)
        if not (debug_on or dump is not None):
            return
        fragment = _wrap_op_as_fragment(self.graph, match.root_node_id, option) if isinstance(option, Op) else option
        pass_name = pass_.name if pass_ is not None else None
        text = _format_rule_application(rule.name, self.graph, match, fragment, pass_name=pass_name)
        if debug_on:
            emit(text)
        if dump is not None and pass_ is not None and pass_.name:
            record = _record_rule_application(self.graph, match, fragment)
            dump.on_rule(pass_, rule, record, text)

    def _advance_if_last(self, match: Match, *, applied: bool = False) -> None:
        if match.is_last and not (applied and match.rule.fixpoint):
            self.cursor.advance(self.graph)


def _format_rule_application(name: str, graph: Graph, match: Match, fragment: Graph, *, pass_name: str | None = None) -> str:
    """Render a one-rule-application snapshot as a unified diff bracketed
    by ``>>> name`` / ``<<< name`` markers (see ``rule_diff``). Kernel
    ops (LoopOp/TileOp/KernelOp/CudaOp) are pretty-printed via their
    dedicated printers rather than dumped as a body repr."""
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    matched_nodes = [graph.nodes[nid] for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes]
    before = _format_nodes(matched_nodes, graph)
    frag_nodes = [fragment.nodes[nid] for nid in fragment.topological_order()]
    after = _format_nodes(frag_nodes, fragment)
    return render_rule_diff(display_name(pass_name, name), before, after, header=f"matched at {match.root_node_id}")


def _wrap_op_as_fragment(graph: Graph, root_id: str, new_op: Op) -> Graph:
    """Build a single-node fragment that mirrors ``graph.nodes[root_id]``
    with ``new_op`` substituted. Lets the engine render an in-place op
    rebind through the same diff/dump path as a functional fragment splice
    (the engine then assigns ``root.op = new_op`` directly, bypassing the
    splicer — node id, inputs list, hints, and output Tensor are kept)."""
    root = graph.nodes[root_id]
    frag = Graph()
    for inp_id in root.inputs:
        if inp_id in frag.nodes:
            continue
        inp_t = graph.buffer(inp_id)
        shape = inp_t.shape if inp_t is not None else ()
        dtype = inp_t.dtype if inp_t is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    out_id = frag.add_node(new_op, list(root.inputs), root.output, node_id=root.id)
    frag.outputs = [out_id]
    return frag


def _record_rule_application(graph: Graph, match: Match, fragment: Graph) -> dict:
    """Structured analog of ``_format_rule_application`` for JSON dumps.

    Captures the matched-subgraph nodes and the fragment's nodes as plain
    dicts so post-hoc scripts (and the article-side analysis) can iterate
    rule applications without re-parsing the text snapshot.
    """
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    return {
        "root": match.root_node_id,
        "matched_pattern_nodes": dict(match.nodes),
        "before": [_node_to_dict(graph.nodes[nid]) for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes],
        "after": [_node_to_dict(fragment.nodes[nid]) for nid in fragment.topological_order()],
    }


def _node_to_dict(node) -> dict:
    return {
        "id": node.output.name,
        "op_class": type(node.op).__name__,
        "inputs": list(node.inputs),
        "output_shape": list(node.output.shape),
        "output_dtype": node.output.dtype,
    }


def _format_nodes(nodes: list, graph: Graph) -> str:
    """Render a list of nodes as readable text. Body-carrying ops use
    their own ``pretty_body``; everything else falls back to a
    ``name: ClsName(args)`` one-liner. Scalar ``ConstantOp`` inputs are
    inlined as literals (same treatment as ``format_kernels`` — see
    ``_inline_scalar_loads``). The surrounding
    ``<output> = TileOp(<inputs>)`` label is emitted here, one line
    above the body — ``BodyOp.pretty_body`` no longer prepends its own
    kernel-name / I/O header to keep the two from duplicating."""
    lines: list[str] = []
    for node in nodes:
        op = node.op
        if isinstance(op, (InputOp, ConstantOp)):
            continue
        body = op.pretty_body()
        if body is None:
            lines.append(f"{node.output.name} = {_fmt_op(node, graph)}")
            continue
        arg_names = [t.name for inp in node.inputs if (t := graph.buffer(inp)) is not None]
        lines.append(f"{node.output.name} = {type(op).__name__}({', '.join(arg_names)})")
        scalar_inputs = _scalar_constant_inputs(graph, node, ConstantOp)
        if scalar_inputs:
            body = _inline_scalar_loads(body, scalar_inputs)
        lines.extend(f"  {line}" for line in body.splitlines())
    return "\n".join(lines)


def _build_rewrite_kwargs(rule, match: Match, ctx: Context | None) -> dict:
    """Bind each ``rewrite`` param to its source.

    Reserved-name params (``match`` / ``root`` / ``out`` / ``ctx``) and
    ``PATTERN``-name params bind by name; every remaining param binds
    positionally to ``root.inputs[i]`` (in declaration order, ``None``
    when the position exceeds the available inputs)."""
    pattern_names = {p.name for p in rule.pattern}
    root_node = match.root
    graph = match.graph
    kwargs: dict = {}
    input_slot = 0
    for pname in rule.param_names:
        if pname == "match":
            kwargs[pname] = match
        elif pname == "root":
            kwargs[pname] = root_node
        elif pname == "out":
            kwargs[pname] = root_node.output
        elif pname == "ctx":
            kwargs[pname] = ctx
        elif pname in pattern_names:
            kwargs[pname] = match.node(pname)
        else:
            if input_slot < len(root_node.inputs):
                kwargs[pname] = graph.producer(root_node.inputs[input_slot])
            else:
                kwargs[pname] = None
            input_slot += 1
    return kwargs


def _validate_reason(op: Op, ctx: Context) -> str:
    """Best-effort introspection of *why* ``op.validate(ctx)`` returned
    ``False``. Returns a short reason like ``smem 106496 > cap 101376``
    when the op exposes the right introspection hooks (``KernelOp``
    today). Empty string when no reason can be derived — the caller
    treats that as a generic ``validate(ctx)=False`` line."""
    # Best-effort: keep the import local so a missing kernel-IR module
    # (e.g. minimal harness in tests) doesn't break the engine itself.
    try:
        from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    except ImportError:
        return ""
    if not isinstance(op, KernelOp):
        return ""
    reasons: list[str] = []
    try:
        smem = op.smem_bytes()
        if smem > ctx.max_dynamic_smem:
            reasons.append(f"smem {smem} > max_dynamic_smem {ctx.max_dynamic_smem}")
    except Exception:  # noqa: BLE001 — best-effort introspection
        pass
    return "; ".join(reasons)
