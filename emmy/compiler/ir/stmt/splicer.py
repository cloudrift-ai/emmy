"""Shared demand reconstruction for body fusion and compact subroutine expansion.

The engine consumes statement analyses and splice edges, never graph nodes or LoopOps. The Loop
IR adapter chooses graph regions and wraps the resulting Body in a validated LoopOp.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from graphlib import TopologicalSorter

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, Interval, Literal, SimplifyCtx, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.analysis import BodyAnalysis, Scope
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.builder import BodyBuilder
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Select, Write
from emmy.compiler.ir.stmt.subroutine import Call, Subroutine, definitions, pretty_subroutines, reduction_depths
from emmy.utils import cached_method

logger = logging.getLogger(__name__)


class NotSupported(Exception):
    """Pattern not handled — caller converts to ``None`` return.

    Always raised with a human-readable reason: ``splice_loops`` logs
    ``type(exc).__name__: <reason>`` at DEBUG, so ``compile -vv`` shows
    *which* unsupported pattern a given producer→consumer edge hit (σ-solve
    failure, missing Write, scope shortfall, …) without re-instrumenting the
    splicer."""


class UnfusableStmt(NotSupported):
    """One named loop's statement cannot be spliced, and no other loop in the region is implicated.
    Two reasons reach it: ``origin``'s statements multiply σ-bindings past the construction bound
    instead of deduplicating (a recurrence the roller did not roll), and a Write of ``origin`` that
    observes a running accumulator (a scan), whose order no merged body preserves. Raised, never
    folded into ``None``: fusion decides its regions before it splices, so a region it cannot build
    is a compiler bug to fix, and the roller declines a step it cannot splice."""

    def __init__(self, message: str, origin: str) -> None:
        super().__init__(message)
        self.origin = origin


# Unified binding key: ``(origin, name, emit_scope, sigma.restrict(enclosing))``.
# ``emit_scope`` is where the stmt lands in the merged body; ``sigma`` is
# restricted to the stmt's own enclosing axis names — the only bindings that
# affect its rewrite (Load.index / Select.select) or its dep resolution.
_BindKey = tuple[str, str, Scope, Sigma]


@dataclass
class _Demand:
    """A pending dep in the worklist.

    ``bound_as`` is the fresh name the dep's def will bind in the merged
    body — allocated at queue time so callers can reference it without
    waiting for resolution.
    """

    name: str
    origin: str  # tag identifying which loop this def came from
    sigma: Sigma
    demand_scope: Scope
    bound_as: str


def observes_running_accumulator(meta: BodyAnalysis, write: Write, scope: Scope) -> bool:
    """Whether ``write`` observes an accumulator before its reduce loop completes."""
    defining = meta.defs.get(write.value)
    reduce_axis = meta.reduce_axes.get(write.value)
    return isinstance(defining, Accum) and reduce_axis is not None and reduce_axis in scope.enclosing


def expand_calls(body: Body) -> Body:
    """Expand compact SSA bodies through shared demands and reduction-axis unification."""
    targets = definitions(body)
    if not targets:
        return body
    tags = {target: f"sub{index}" for index, target in enumerate(targets)}
    loops = {tags[target]: BodyAnalysis.from_body(target.body, target.axes) for target in targets}
    loops["root"] = BodyAnalysis.from_body(body)
    builder = Splicer(Program(loops, {}), roots=tuple(("root", w.output) for w, _ in loops["root"].writes), expand=tags)
    builder._seed()
    # A statement body may end in values rather than writes. Keep every terminal definition
    # live, including unused values beside output writes; normalization is not dead-code removal.
    root = loops["root"]
    for name in root.defs:
        if name in body.ssa_uses:
            continue
        scope = root.scopes[name]
        value = builder._ensure_dep(name, "root", Sigma.IDENTITY, scope)
        builder.insert(Assign(name, "copy", (value,)), scope)
    builder.resolve()
    return Body.coerce(builder.finish())


@dataclass(frozen=True)
class Program:
    """The source DAG and its shared reduction definitions for one complete splice."""

    loops: dict[str, BodyAnalysis]
    splice_edges: dict[tuple[str, str], tuple[str, str]]

    @cached_property
    def used_names(self) -> set[str]:
        return set().union(
            *(
                set(meta.body.ssa_defs | meta.body.free_ssa | meta.body.axis_names)
                | {axis.name for scope in meta.scopes.values() for axis in scope.enclosing}
                for meta in self.loops.values()
            )
        )

    @cached_property
    def reduction_depth(self) -> dict[str, dict[str, int]]:
        dependencies = {tag: {} for tag in self.loops}
        for (origin, source), target in self.splice_edges.items():
            dependencies[origin][source] = target
        depths: dict[str, dict[str, int]] = {}
        order = TopologicalSorter({tag: {origin for origin, _ in edges.values()} for tag, edges in dependencies.items()})
        for tag in order.static_order():
            inputs = {
                source: max((depths[origin][w.value] for w, _ in self.loops[origin].writes if w.output == output), default=0)
                for source, (origin, output) in dependencies[tag].items()
            }
            depths[tag] = reduction_depths(self.loops[tag].body, inputs)
        return depths

    @cached_property
    def source_stmts(self) -> int:
        return sum(1 for meta in self.loops.values() for _ in meta.body.iter())

    @cached_method
    def subroutine(self, origin: str, name: str) -> Subroutine:
        meta = self.loops[origin]
        scope = meta.scopes[name]
        params = tuple(axis for axis in scope.enclosing if axis.name in meta.live_axes[name])
        builder = Splicer(self, roots=(), outline=(origin, name))
        result = builder._ensure_dep(name, origin, Sigma.IDENTITY, scope)
        builder.resolve()
        from emmy.compiler.ir.stmt.normalize import prepare_body

        # A temporary output keeps the returned value live while aliases and unit reductions
        # simplify. It is removed before forming the read-only subroutine.
        body = prepare_body(Body((*builder.finish(), Write(output="_return", index=(), value=result))))
        returned = next(stmt for stmt in body if isinstance(stmt, Write))
        return Subroutine(f"{origin}_{name}", params, Body(stmt for stmt in body if stmt is not returned), returned.value)


class Splicer(BodyBuilder):
    """Build a maximal region, keeping reduction cones as shared calls until final CSE.

    Worklist dep-resolution is reverse-topological — producers demanded after consumers — so
    the builder's prepend-at-leaf behavior yields defined-before-use ordering. Reusing an already
    emitted producer can invert siblings; normalization restores their topological order.
    """

    def __init__(
        self,
        program: Program,
        *,
        roots: tuple[tuple[str, str], ...],
        outline: tuple[str, str] | None = None,
        expand: dict[Subroutine, str] | None = None,
    ) -> None:
        super().__init__(used_names=program.used_names)
        self.program = program
        self.loops = program.loops
        self.splice_edges = program.splice_edges
        self.roots = roots
        self.outline = outline
        self.expand = expand
        self.bound = frozenset(axis.name for axis in self.loops[outline[0]].scopes[outline[1]].enclosing) if outline else frozenset()
        self._pending: deque[_Demand] = deque()
        self._binding: dict[_BindKey, str] = {}
        self._reduce_axes: dict[tuple[Scope, Expr, int], Axis] = {}
        self._free_vars_by_expr_id: dict[int, tuple[Expr, frozenset[str]]] = {}

    # Construction bound: how many DISTINCT bindings the merged body may take per source
    # statement. The dedup table shares each (stmt, emit scope, σ) binding, and a legitimate
    # splice emits about one binding per input statement — a value read at a few offsets a
    # few. A recurrence left unrolled breaks that sharing: each stage is re-demanded under
    # COMPOSITIONS of σs, so bindings multiply per stage instead of deduplicating (DeepSeek-V4's
    # 20-iteration Sinkhorn chain drove 4.5M distinct bindings from 2,287 input statements and
    # never finished). Such a merge cannot be constructed at any budget; the first binding past
    # the bound raises, and the answer is the roller (``loop/fusion/005_roll_recurrence``), never
    # a smaller region — a termination bound, not a fusion-quality gate: placement still owns
    # every cut on a merge that CAN be built.
    # Whole layers legitimately take over a hundred bindings per source statement: the tracer
    # unrolls per-head work into copies that each re-derive the shared input. Hence 256, rather
    # than a bound that would reject those layers; a chain left unrolled doubles per stage.
    _BINDING_RATIO = 256

    def insert(self, stmt: Stmt, enclosure: Scope) -> None:
        # A definition's parameters are bound by its call, not by loops in its body.
        super().insert(stmt, Scope(tuple(axis for axis in enclosure.enclosing if axis.name not in self.bound)))

    def resolve(self) -> None:
        while self._pending:
            self._resolve(self._pending.popleft())

    def run(self) -> Body:
        self._seed()
        self.resolve()
        # Body normalization restores sibling order when a consumer prepends above
        # an already-emitted producer. Reconstruction itself only orders new demands.
        body = Body.coerce(self.finish())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Compact fusion:\n%s", "\n".join(pretty_subroutines(body)))
        return body

    # -- Seed: every selected root Write, with its value queued -------------

    def _seed(self) -> None:
        for root_tag, output in self.roots:
            root = self.loops.get(root_tag)
            if root is None:
                raise NotSupported(f"root names unknown loop {root_tag!r}")
            found = next(((write, scope) for write, scope in root.writes if write.output == output), None)
            if found is None:
                raise NotSupported(f"root loop {root_tag!r} has no Write to {output!r}")
            w, scope = found
            if observes_running_accumulator(root, w, scope):
                raise UnfusableStmt(
                    f"root Write to {w.output!r} observes running accumulator {w.value!r}; ordered loop cannot be spliced",
                    origin=root_tag,
                )
            v_bound = self._ensure_dep(w.value, root_tag, Sigma(), scope)
            self.insert(
                Write(
                    output=w.output,
                    index=w.index,
                    value=v_bound,
                    value_dtype=w.value_dtype,
                    atomic=w.atomic,
                    swizzle=w.swizzle,
                ),
                scope,
            )

    # -- Dep binding: look up or queue --------------------------------------

    def _ensure_dep(self, name: str, origin: str, sigma: Sigma, ref_scope: Scope) -> str:
        """Return the merged-body name for ``(origin, name)`` at the emit
        scope induced by ``ref_scope`` and σ. Queue a new demand the first
        time the key is seen.
        """
        meta = self.loops[origin]
        if name not in meta.defs:
            raise NotSupported(f"_ensure_dep: {name!r} is not defined in loop {origin!r}")

        required_axes = tuple(
            mapped
            for axis in meta.scopes[name].enclosing
            for mapped in _remap_axis_names(axis, sigma, ref_scope, free_vars=self._expr_free_vars)
        )
        emit_scope = _scope_for_axes(ref_scope, required_axes)

        # σ restricted to axes transitively used in Expr subtrees reachable
        # from this stmt. Bindings outside this set don't affect any emitted
        # stmt, so keeping them in the key would cause spurious duplicate
        # emissions.
        restricted = sigma.restrict(meta.live_axes[name])
        key = (origin, name, emit_scope, restricted)
        existing = self._binding.get(key)
        if existing is not None:
            return existing
        if len(self._binding) >= self._BINDING_RATIO * self.program.source_stmts:
            raise UnfusableStmt(
                f"the merged body takes over {self._BINDING_RATIO} bindings per source statement, at {name!r} of loop "
                f"{origin!r} — the region's σ-bindings multiply instead of deduplicating: a recurrence the roller did not roll",
                origin=origin,
            )
        bound = self.fresh(name)
        self._binding[key] = bound
        self._pending.append(_Demand(name=name, origin=origin, sigma=sigma, demand_scope=emit_scope, bound_as=bound))
        return bound

    def _expr_free_vars(self, expr: Expr) -> frozenset[str]:
        """Memoize one expression's variables for this splice by object identity."""
        key = id(expr)
        cached = self._free_vars_by_expr_id.get(key)
        if cached is not None and cached[0] is expr:
            return cached[1]
        variables = expr.free_vars()
        self._free_vars_by_expr_id[key] = (expr, variables)
        return variables

    # -- Resolution dispatch -------------------------------------------------

    def _resolve(self, d: _Demand) -> None:
        stmt = self.loops[d.origin].defs[d.name]

        if isinstance(stmt, Load):
            edge = self.splice_edges.get((d.origin, stmt.input))
            if edge is not None:
                target_tag, target_output_buf = edge
                self._resolve_splice_load(stmt, d, target_tag, target_output_buf)
            else:
                self._resolve_plain(stmt, d)
        elif isinstance(stmt, Accum):
            if self.expand is not None or (d.origin, d.name) == self.outline:
                self._resolve_accum(stmt, d)
            else:
                target = self.program.subroutine(d.origin, d.name)
                args = tuple(d.sigma.apply(Var(name)) for name in target.params)
                self.insert(Call(d.bound_as, target, args), d.demand_scope)
        elif isinstance(stmt, Call):
            rename = {
                arg: Var(self._ensure_dep(arg, d.origin, d.sigma, d.demand_scope))
                for arg in stmt.deps()
                if arg in self.loops[d.origin].defs
            }
            args = tuple(d.sigma.apply(arg).substitute(rename) for arg in stmt.args)
            sigma = _canonical(Sigma(dict(zip(stmt.target.params, args, strict=True))), d.demand_scope)
            value = self._ensure_dep(stmt.target.result, self.expand[stmt.target], sigma, d.demand_scope)
            self.insert(Assign(name=d.bound_as, op="copy", args=(value,)), d.demand_scope)
        elif isinstance(stmt, (Assign, Select)):
            self._resolve_plain(stmt, d)
        else:
            raise NotSupported(f"_resolve: unsupported stmt type {type(stmt).__name__} for {d.name!r} in loop {d.origin!r}")

    def _resolve_plain(self, stmt: Stmt, d: _Demand) -> None:
        """Resolve SSA operands, leaving coordinates to σ, for an ordinary scalar binding."""
        meta = self.loops[d.origin]
        rename = {v: self._ensure_dep(v, d.origin, d.sigma, d.demand_scope) for v in stmt.deps() if v in meta.defs}
        rename[stmt.name] = d.bound_as
        self.insert(stmt.rewrite(lambda n: rename.get(n, n), d.sigma), d.demand_scope)

    def _resolve_splice_load(self, stmt: Load, d: _Demand, target_tag: str, target_output_buf: str) -> None:
        """A Load that's a splice edge to another registered loop — emit a
        copy alias and queue the target loop's ``Write.value`` under the
        solved σ. The target's expression chain reconstructs piecemeal over
        subsequent iterations. ``target_output_buf`` selects which ``Write``
        of the target is the splice source when the target has multiple outputs."""
        target = self.loops[target_tag]
        found = next(((w, scope) for w, scope in target.writes if w.output == target_output_buf), None)
        if found is None:
            raise NotSupported(
                f"splice edge into {target_tag!r}: no Write with output={target_output_buf!r} "
                f"(target writes {[w.output for w, _ in target.writes]}) — usually a buf-name != node-id mismatch on the producer"
            )
        target_write, target_scope = found
        if observes_running_accumulator(target, target_write, target_scope):
            raise UnfusableStmt(
                f"splice edge into {target_tag!r} observes running accumulator {target_write.value!r}; ordered loop cannot be spliced",
                origin=target_tag,
            )
        source_meta = self.loops[d.origin]
        index_rename = {
            name: Var(self._ensure_dep(name, d.origin, d.sigma, d.demand_scope)) for name in stmt.deps() if name in source_meta.defs
        }
        effective_index = tuple(d.sigma.apply(e).substitute(index_rename) for e in stmt.index)
        sigma = _solve_sigma(target_write.index, effective_index, target.body.axis_names)
        if sigma is None:
            raise NotSupported(f"σ-solve failed pairing target write index {target_write.index} against reader index {effective_index}")
        v_bound = self._ensure_dep(target_write.value, target_tag, _canonical(sigma, d.demand_scope), d.demand_scope)
        self.insert(Assign(name=d.bound_as, op="copy", args=(v_bound,)), d.demand_scope)

    def _resolve_accum(self, stmt: Accum, d: _Demand) -> None:
        """Queue the value under a shared iteration scope for independent reductions of equal extent."""
        orig_axis = self.loops[d.origin].reduce_axes[stmt.name]
        key = (d.demand_scope, orig_axis.extent.expr, self.program.reduction_depth[d.origin][stmt.name])
        reduce_axis = self._reduce_axes.get(key)
        if reduce_axis is None:
            reduce_axis = self._reduce_axes[key] = Axis(name=self.fresh(orig_axis.name), extent=orig_axis.extent)
        fresh_name = reduce_axis.name
        inner_sigma = d.sigma.extend(orig_axis.name, Var(fresh_name))
        inner_scope = Scope(enclosing=d.demand_scope.enclosing + (reduce_axis,))
        value_bound = self._ensure_dep(stmt.value, d.origin, inner_sigma, inner_scope)
        self.insert(Accum(name=d.bound_as, value=value_bound, op=stmt.op, axes=(fresh_name,)), inner_scope)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _canonical(sigma: Sigma, scope: Scope) -> Sigma:
    """``sigma`` with each binding simplified under the ranges of ``scope``'s axes, so two readers
    that spell one address differently — the block of the even nibble, ``2·a / 16``, and of the odd,
    ``(2·a + 1) / 16`` — demand one binding, not two: the dedup table keys on the spelling."""
    ctx = SimplifyCtx.empty()
    for axis in scope.enclosing:
        if axis.extent.is_static:
            extent = axis.extent.as_static()
            ctx = ctx.extend(axis.name, Interval(0, extent - 1), Literal(extent, "int"))
    return Sigma({name: expr.simplify(ctx) for name, expr in sigma.mapping.items()})


def _scope_for_axes(ref_scope: Scope, required: tuple[str, ...]) -> Scope:
    """Shortest prefix of ``ref_scope`` whose axis set contains ``required``.

    Used two ways:
    - For Accums: places the reduce ``Loop`` at the innermost consumer
      scope where all σ-mapped producer enclosing axes are visible (today's
      behavior — further hoisting is left to later passes).
    - For plain producer stmts: picks the emit scope from the consumer's
      nest, tolerating producer's free-axis order differing from consumer's.
      A matmul producer ``(a0, a1, a2)`` σ-maps to consumer ``(a0, a2, a1)``
      (shuffled), but the consumer's scope ``(a0, a1, a2)`` covers the same
      axis set; emitting at the consumer's nest avoids a duplicate Loop tree.
    """
    names = tuple(a.name for a in ref_scope.enclosing)
    remaining = set(required)
    k = 0
    while remaining and k < len(names):
        remaining.discard(names[k])
        k += 1
    if remaining:
        raise NotSupported(f"emit scope {names} is missing required axes {sorted(remaining)}")
    return Scope(enclosing=ref_scope.enclosing[:k])


def _remap_axis_names(
    axis: Axis,
    sigma: Sigma,
    ref_scope: Scope,
    *,
    free_vars: Callable[[Expr], frozenset[str]] | None = None,
) -> tuple[str, ...]:
    """Pick the merged-kernel axes that ``axis``'s σ target depends on.

    Every occurrence of the producer axis is substituted with the complete target expression by
    the caller's σ rewrite.  Placement therefore needs the shortest consumer scope containing
    *all* variables read by that expression: one for an offset/stride, several for a flatten /
    tile-coordinate map, and none when the reader fixes the producer axis to a constant.  The
    old single-variable restriction unnecessarily materialized a pure producer before layouts
    such as ``(tile_k, tile_n, lane) -> (k, n)`` even though substitution is exact.

    ``_scope_for_axes`` already accepts a set of required axes and chooses the common enclosing
    prefix, so multi-axis targets need no new loop representation.
    """
    target = sigma.get(axis.name)
    if target is None:
        return (axis.name,)
    variables = free_vars(target) if free_vars is not None else target.free_vars()
    scope_axes = tuple(a.name for a in ref_scope.enclosing)
    if any(name not in scope_axes for name in variables):
        # A non-axis variable is an SSA gather index. Keep the producer at the reader's current
        # scope, where the defining load/assign is available, instead of treating the SSA name as
        # a missing loop axis.
        return scope_axes
    return tuple(sorted(variables))


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> Sigma | None:
    """Solve per-dim pairing ``writer[k] == reader[k]``. Supported writer
    forms: ``Var(a)`` (``a`` in ``producer_axes``) → bind ``a → reader[k]``;
    ``Literal(c)`` → no binding. Anything else → ``None``."""
    if len(writer) != len(reader):
        return None
    mapping: dict[str, Expr] = {}
    for w, r in zip(writer, reader, strict=True):
        if isinstance(w, Literal):
            continue
        if isinstance(w, Var) and w.name in producer_axes:
            existing = mapping.get(w.name)
            if existing is not None and existing != r:
                return None
            mapping[w.name] = r
            continue
        return None
    return Sigma(mapping)
