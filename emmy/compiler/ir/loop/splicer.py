"""Worklist-driven splicer for a DAG of ``LoopOp``s.

Two entry points wrap the same underlying statement reconstruction engine:

- :func:`splice_loops` — tag-generic N-way: caller supplies ``loops``
  (tag → ``LoopOp``), ``splice_edges`` ((origin_tag, src) →
  (target_tag, target_output)), and optional output roots.
- :func:`splice_graph` — consumes a ``Graph`` fragment directly;
  classifies each Load by its node.inputs edge (LoopOp → splice,
  otherwise → external slot in first-seen order).

Before seeding roots, ``splice_graph`` collapses each output equivalence
cluster: a single-owner chain of same-dtype copies proven to be reshape/axis-
permutation bijections. The computed source's Write retargets through the
layout chain, so a terminal layout does not force reduction reconstruction at
its loads.

Algorithm. Seed: every selected root ``Write``. Each iteration pops
one pending dep and emits its def, queueing that def's own deps.
Resolution dispatches on stmt kind:

- **Load on a splice edge** — emit a copy alias at the demand scope;
  σ is solved by pairing target's ``Write.index`` against the reader's
  σ-substituted index, and the target's ``Write.value`` is queued under
  the solved σ. The alias is dtype-free: a narrowing store spells its rounding
  as an ordinary ``Assign`` conversion ahead of the Write
  (``loop/lifting/090_spell_store_rounding``), so inlining the value chain
  carries it with no special case here. The target's expression
  chain reconstructs piecemeal.
- **Accum** — form one shared subroutine per source reduction and emit a call under the
  demanded coordinates. Building that definition, or expanding calls before full CSE, shares
  equal-extent axes between independent reductions at one scope and queues the contribution
  under σ extended with the reduce binding. Nested reductions remain calls during construction.
- **Plain Assign / Select / Load** (non-splice source) — ``rewrite``
  the original stmt through ``(rename_ssa, sigma)`` and insert at the
  demand scope.

Unified dedup. A single table keyed on
``(origin, name, emit_scope, σ.restrict(live_axes))`` decides whether
to share an existing emission or allocate a fresh one. ``live_axes``
comes from ``BodyAnalysis`` and is the set of axes transitively reachable
through the stmt's Expr subtrees — σ bindings outside that set are
irrelevant and collapsed. Same key → share; different emit scope or
different live-σ → emit twice. This handles plain-stmt and subroutine-call sharing, Accum
scope multiplicity (SDPA QK^T at softmax-max vs softmax-output), and
multi-output splice targets uniformly.

``BodyBuilder.insert`` is pure tree-splicing: descend the body along
the enclosure path, creating ``Loop`` nodes if missing, prepend at the
leaf. The worklist resolves deps in reverse-topological order so
consumers demand before producers — that *usually* yields defined-
before-use. The exception is the dedup case: when a stmt's operand
hits an existing binding emitted earlier in the worklist, the new stmt
still prepends above the existing one — landing above its own dep.
That sibling inversion is fixed up by the generic
``topo_sort_siblings`` pass in :mod:`emmy.compiler.ir.stmt.normalize`,
which runs inside ``LoopOp.__post_init__`` — so the splicer doesn't
need its own ordering pass; constructing the ``LoopOp`` is enough.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

from emmy.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.loop.ir import Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt.splicer import NotSupported, Program, Splicer, UnfusableStmt
from emmy.compiler.ir.stmt.splicer import observes_running_accumulator as _observes_running_accumulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OutputEquivalenceCluster:
    """A single-owner chain of bijective output-layout copies.

    ``buffers`` runs from the computed source to the live graph output;
    ``copy_nodes`` are the intervening LoopOp nodes in the same order.
    """

    buffers: tuple[str, ...]
    copy_nodes: tuple[str, ...]
    inverse_strides: tuple[tuple[int, ...], ...]


def observes_running_accumulator(op: LoopOp) -> bool:
    """Whether a Write of ``op`` observes an accumulator before its reduce loop completes — an
    ordered prefix output (a scan) no merged body preserves, so the loop is a kernel of its own."""
    meta = op.analyze()
    return any(_observes_running_accumulator(meta, write, scope) for write, scope in meta.writes)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def splice_loops(
    loops: dict[str, LoopOp],
    splice_edges: dict[tuple[str, str], tuple[str, str]],
    *,
    roots: tuple[tuple[str, str], ...] | None = None,
) -> LoopOp | None:
    """Splice a DAG of ``LoopOp``s into one merged kernel.

    ``loops`` maps an opaque tag to each participating ``LoopOp``.
    ``splice_edges`` identifies which Loads are inlined from another
    registered loop: key ``(origin_tag, source_buf)`` → value
    ``(target_tag, target_output_buf)`` meaning "this loop's Load whose
    ``source`` is ``source_buf`` reads ``target_tag``'s Write whose
    ``output`` is ``target_output_buf`` and should be inlined."
    Non-splice Loads keep their original ``source`` buf names — buf
    identity is global, no remap needed. ``roots`` selects the observable
    Writes as ``(loop_tag, output_buffer)`` pairs. When omitted, every Write
    of the unique loop that never appears as a splice target is selected.
    Returns ``None`` if roots cannot be derived or any splice edge hits an
    unsupported pattern.
    """
    if roots is None:
        target_tags = {tag for tag, _out in splice_edges.values()}
        candidates = [tag for tag in loops if tag not in target_tags]
        if len(candidates) != 1:
            return None
        root_tag = candidates[0]
        roots = tuple((root_tag, write.output) for write in loops[root_tag].writes)
    if not roots:
        return None
    try:
        body = Splicer(Program({tag: op.analyze() for tag, op in loops.items()}, splice_edges), roots=roots).run()
        return LoopOp(body=body)
    except (NotSupported, ValueError) as exc:
        # NotSupported = splicer hit an unsupported pattern (σ-solve, scope).
        # ValueError = LoopOp construction validation rejected the emitted body.
        # Both surface to callers as None; debug log preserves which one for
        # future investigation without polluting normal output. A doomed
        # statement is raised: fusion answers it with an error, the roller by declining.
        if isinstance(exc, UnfusableStmt):
            raise
        logger.debug("splice_loops rejected pattern: %s: %s", type(exc).__name__, exc)
        return None


def splice_graph(graph) -> tuple[LoopOp, list[str]] | None:
    """Splice a subgraph of ``LoopOp`` nodes into one merged kernel.

    Each ``LoopOp`` node in ``graph`` becomes a registered loop tagged
    by its node id. Within each LoopOp node, a Load whose source points
    at another ``LoopOp`` node becomes a splice edge; a Load whose
    source points at a non-``LoopOp`` node (e.g. ``InputOp``) becomes
    an external read, assigned a slot in first-seen order.

    Every graph output is a root, so separate terminal branches become one
    multi-output LoopOp. A single-owner chain of bijective layout copies
    ending at a root is an output equivalence cluster: its source Write is
    retargeted through the chain before ordinary dependency reconstruction.
    Returns ``(merged_op, external_buffer_ids)`` where the ids are the
    non-``LoopOp`` inputs in merged first-use order. Returns ``None`` if an
    output is not produced by a ``LoopOp`` or any splice edge hits an
    unsupported pattern.
    """
    if not graph.outputs:
        return None

    loop_nodes = {n.id: n for n in graph.nodes.values() if isinstance(n.op, LoopOp)}
    loops = {nid: node.op for nid, node in loop_nodes.items()}
    root_overrides: dict[str, tuple[str, str]] = {}
    collapsed: set[str] = set()
    for cluster in _output_equivalence_clusters(graph, loop_nodes):
        source, output = cluster.buffers[0], cluster.buffers[-1]
        source_node = graph.producer(source)
        if source_node is None:
            continue
        retargeted = _retarget_equivalent_output(graph, loops[source_node.id], cluster)
        if retargeted is None:
            continue
        loops[source_node.id] = retargeted
        collapsed.update(cluster.copy_nodes)
        root_overrides[output] = (source_node.id, output)
    for nid in collapsed:
        loops.pop(nid, None)

    roots: list[tuple[str, str]] = []
    for output in graph.outputs:
        if output in root_overrides:
            roots.append(root_overrides[output])
            continue
        root_node = graph.producer(output)
        if root_node is None or root_node.id not in loops:
            return None
        roots.append((root_node.id, output))
    splice_edges: dict[tuple[str, str], tuple[str, str]] = {}
    external_order: list[str] = []
    seen_external: set[str] = set()

    for node_id, op in loops.items():
        for ld in op.body.loads:
            inp = ld.input
            # A Load is a splice edge if its source buf names another LoopOp node;
            # otherwise it's an external input. We key edges off the buf name
            # (Load.source is the producing node's id), not a positional input
            # index — so a single edge entry covers every Load that reads the
            # same producer.
            input_producer = graph.producer(inp)
            if input_producer is not None and input_producer.id in loops:
                producer_id = input_producer.id
                splice_edges[(node_id, inp)] = (producer_id, inp)
            elif inp not in seen_external:
                seen_external.add(inp)
                external_order.append(inp)

    merged = splice_loops(loops=loops, splice_edges=splice_edges, roots=tuple(roots))
    if merged is None:
        return None
    return merged, external_order


def _output_equivalence_clusters(graph, loop_nodes: dict[str, object]) -> tuple[_OutputEquivalenceCluster, ...]:
    """Find single-owner bijective layout-copy chains ending at graph outputs."""
    clusters: list[_OutputEquivalenceCluster] = []
    for output in graph.outputs:
        if graph.buffer_users(output):
            continue
        buffers = [output]
        copy_nodes: list[str] = []
        inverse_strides: list[tuple[int, ...]] = []
        current = output
        while True:
            copy = graph.producer(current)
            if copy is None or copy.id not in loop_nodes:
                break
            inverse = _layout_copy_inverse(graph, copy, current)
            if inverse is None:
                break
            source, strides = inverse
            if source in graph.outputs or graph.buffer_users(source) != {copy.id}:
                break
            source_node = graph.producer(source)
            if source_node is None or source_node.id not in loop_nodes:
                break
            buffers.append(source)
            copy_nodes.append(copy.id)
            inverse_strides.append(strides)
            current = source
        if copy_nodes:
            clusters.append(
                _OutputEquivalenceCluster(
                    buffers=tuple(reversed(buffers)),
                    copy_nodes=tuple(reversed(copy_nodes)),
                    inverse_strides=tuple(reversed(inverse_strides)),
                )
            )
    return tuple(clusters)


def _layout_copy_inverse(graph, node, output: str) -> tuple[str, tuple[int, ...]] | None:
    """Return ``(source, destination-flat stride per source dimension)``.

    A reshape/axis permutation spells every non-unit source coordinate as one mixed-radix digit
    of the destination's dense flat address. Matching digits from the innermost stride outward
    recovers that permutation symbolically; no tensor coordinates are enumerated.
    """
    if not isinstance(node.op, LoopOp) or node.buffer_names() != (output,):
        return None
    if any(not isinstance(stmt, (Loop, Load, Write)) for stmt in node.op.body.iter()):
        return None
    loads = node.op.body.loads
    writes = node.op.body.writes
    if len(loads) != 1 or len(writes) != 1:
        return None
    load, write = loads[0], writes[0]
    if not load.is_scalar or not write.is_scalar or write.value != load.name or write.output != output:
        return None
    if write.atomic or write.swizzle != "NONE":
        return None

    source = graph.buffer(load.input)
    destination = graph.buffer(output)
    if source is None or destination is None or source.dtype != destination.dtype:
        return None
    # A shared leading symbolic dimension — the token axis of a serving twin — is carried through
    # as itself: the copy reads and writes it by one coordinate, and the static dimensions behind
    # it are the layout the proof is about.
    leading = _shared_leading_symbolic(source.shape, destination.shape)
    load_index, write_index = load.index, write.index
    if leading:
        if len(load_index) < 2 or len(write_index) < 2 or load_index[0] != write_index[0] or not isinstance(load_index[0], Var):
            return None
        load_index, write_index = load_index[1:], write_index[1:]
    source_shape, destination_shape = source.shape[leading:], destination.shape[leading:]
    source_dims = tuple(dim.as_static() for dim in source_shape if dim.is_static)
    destination_strides = _static_strides(destination_shape)
    if len(source_dims) != len(source_shape) or destination_strides is None:
        return None
    source_numel = math.prod(source_dims)
    destination_numel = math.prod(dim.as_static() for dim in destination_shape)
    if source_numel != destination_numel:
        return None
    extents = _loop_extents(node.op, leading=load.index[0].name if leading else None)
    if extents is None or math.prod(extents.values()) != destination_numel:
        return None
    if len(load_index) != len(source_dims) or len(write_index) != len(destination_strides):
        return None

    ctx = _extent_ctx(extents)
    destination_flat = _dense_flat_address(write_index, destination_strides, extents, ctx)
    if destination_flat is None:
        return None
    actual = tuple(expr.simplify(ctx) for expr in load_index)
    inverse = [0] * len(source_dims)
    if any(actual[i] != Literal(0, "int") for i, dim in enumerate(source_dims) if dim == 1):
        return None
    remaining = {i for i, dim in enumerate(source_dims) if dim > 1}
    stride = 1
    while remaining:
        matches = [i for i in remaining if actual[i] == _flat_digit(destination_flat, stride, source_dims[i], ctx)]
        if len(matches) != 1:
            return None
        index = matches[0]
        inverse[index] = stride
        stride *= source_dims[index]
        remaining.remove(index)
    return load.input, tuple(([-1] if leading else []) + inverse)


def _shared_leading_symbolic(source_shape, destination_shape) -> int:
    """1 when both shapes open with one symbolic dimension and are static behind it, else 0."""
    if not source_shape or not destination_shape:
        return 0
    lead, other = source_shape[0], destination_shape[0]
    if lead.is_static or other.is_static or lead != other:
        return 0
    if all(dim.is_static for dim in source_shape[1:]) and all(dim.is_static for dim in destination_shape[1:]):
        return 1
    return 0


def _extent_ctx(extents: dict[str, int]) -> SimplifyCtx:
    """Build the static loop-range context used by layout proofs and retargeting."""
    ctx = SimplifyCtx.empty()
    for name, extent in extents.items():
        ctx = ctx.extend(name, Interval(0, extent - 1), Literal(extent, "int"))
    return ctx


def _flat_expr(index: tuple[Expr, ...], strides: list[int], ctx: SimplifyCtx) -> Expr:
    """Flatten one row-major index and simplify it under ``ctx``."""
    flat: Expr = Literal(0, "int")
    for expression, stride in zip(index, strides, strict=True):
        term = expression if stride == 1 else BinaryExpr("*", expression, Literal(stride, "int"))
        flat = BinaryExpr("+", flat, term)
    return flat.simplify(ctx)


def _dense_flat_address(index: tuple[Expr, ...], strides: list[int], extents: dict[str, int], ctx: SimplifyCtx) -> Expr | None:
    """Return the flat address when it densely enumerates the loop domain."""
    flat = _flat_expr(index, strides, ctx)
    affine = affine_form(flat, set(extents))
    if affine is None:
        return None
    anchor = affine[0].simplify(ctx)
    if not isinstance(anchor, Literal) or anchor.value != 0:
        return None
    active = {name for name, extent in extents.items() if extent > 1}
    coefficients = {name: coefficient for name, coefficient in affine[1].items() if coefficient}
    if set(coefficients) != active or any(coefficient <= 0 for coefficient in coefficients.values()):
        return None
    stride = 1
    for name in sorted(active, key=coefficients.get):
        if coefficients[name] != stride:
            return None
        stride *= extents[name]
    return flat


def _flat_digit(flat: Expr, stride: int, extent: int, ctx: SimplifyCtx) -> Expr:
    """Return one mixed-radix digit of ``flat``; unit dimensions are literal zero."""
    if extent == 1:
        return Literal(0, "int")
    digit = flat if stride == 1 else BinaryExpr("/", flat, Literal(stride, "int"))
    return BinaryExpr("%", digit, Literal(extent, "int")).simplify(ctx)


def _unflatten(flat: Expr, shape, ctx: SimplifyCtx) -> tuple[Expr, ...] | None:
    """Decompose ``flat`` into one static row-major coordinate tuple."""
    strides = _static_strides(shape)
    if strides is None:
        return None
    return tuple(_flat_digit(flat, stride, dim.as_static(), ctx) for stride, dim in zip(strides, shape, strict=True))


def _retarget_equivalent_output(graph, op: LoopOp, cluster: _OutputEquivalenceCluster) -> LoopOp | None:
    """Retarget the source Writes through a chain of equivalent output layouts."""
    source, output = cluster.buffers[0], cluster.buffers[-1]
    source_tensor = graph.buffer(source)
    source_writes = [write for write in op.body.writes if write.output == source]
    if source_tensor is None or not source_writes:
        return None
    # A leading symbolic dimension the chain carries through (``_layout_copy_inverse``) is the
    # producer's own leading coordinate at every step; the proof runs on the static rest.
    leading = 1 if any(inverse and inverse[0] == -1 for inverse in cluster.inverse_strides) else 0
    if leading and any(not inverse or inverse[0] != -1 for inverse in cluster.inverse_strides):
        return None
    lead_axis = None
    if leading:
        heads = {write.index[0] for write in source_writes if write.index}
        if len(heads) != 1 or not isinstance(next(iter(heads)), Var):
            return None
        lead_axis = next(iter(heads)).name
    extents = _loop_extents(op, leading=lead_axis)
    if extents is None or any(isinstance(stmt, Load) and stmt.input == source for stmt in op.body.iter()):
        return None
    if any(not write.is_scalar or len(write.index) != len(source_tensor.shape) for write in source_writes):
        return None

    steps: list[tuple[tuple[int, ...], object]] = []
    for inverse, destination in zip(cluster.inverse_strides, cluster.buffers[1:], strict=True):
        tensor = graph.buffer(destination)
        if tensor is None:
            return None
        steps.append((inverse[leading:], tensor.shape[leading:]))

    ctx = _extent_ctx(extents)
    replacement: dict[int, Write] = {}
    for write in source_writes:
        coordinates = write.index[leading:]
        for inverse, shape in steps:
            if len(coordinates) != len(inverse):
                return None
            flat: Expr = Literal(0, "int")
            for expression, stride in zip(coordinates, inverse, strict=True):
                if not stride:
                    continue
                term = expression if stride == 1 else BinaryExpr("*", expression, Literal(stride, "int"))
                flat = BinaryExpr("+", flat, term)
            coordinates = _unflatten(flat.simplify(ctx), shape, ctx)
            if coordinates is None:
                return None
        replacement[id(write)] = replace(write, output=output, index=(*write.index[:leading], *coordinates))

    return LoopOp(
        body=op.body.map(lambda stmt: replacement.get(id(stmt), stmt)),
        name=op.name,
        source=op.source,
        knobs=dict(op.knobs),
    )


def _static_strides(shape) -> list[int] | None:
    """Return row-major element strides, or ``None`` for a symbolic shape."""
    strides: list[int] = []
    step = 1
    for dim in reversed(tuple(shape)):
        if not dim.is_static:
            return None
        strides.append(step)
        step *= dim.as_static()
    return list(reversed(strides))


def _loop_extents(op: LoopOp, *, leading: str | None = None) -> dict[str, int] | None:
    """Return each distinct static loop-axis extent, declining conflicting reuse. ``leading``
    names the one loop axis allowed a symbolic extent, the shared leading dimension a layout
    proof carries through; it takes no part in the static extents."""
    extents: dict[str, int] = {}
    for stmt in op.body.iter():
        if not isinstance(stmt, Loop):
            continue
        if not stmt.axis.extent.is_static:
            if stmt.axis.name == leading:
                continue
            return None
        extent = stmt.axis.extent.as_static()
        if stmt.axis.name in extents and extents[stmt.axis.name] != extent:
            return None
        extents[stmt.axis.name] = extent
    return extents
