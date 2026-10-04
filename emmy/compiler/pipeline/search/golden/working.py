"""The working golden's writers: trace inventories, and the write-back of what a record run measures.

A trace inventory is a golden with no measurement: every kernel a traced program lowers to, and one unmeasured row
per kernel and input regime — the targets ``run --golden PATH --bench`` benches. ``--record-greedy`` writes the
greedy pick's kernel set back as kernels, routing rows and measured rows; ``--record`` writes a row's per-card
latencies. Every write-back is one locked read-modify-write (:meth:`GoldenFile.edit`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from emmy import gpu
from emmy.compiler import pipeline, provenance
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.search.db import RoutingRow, is_placement_knob
from emmy.compiler.pipeline.search.pins import measured_precision_pins
from emmy.compiler.specialize import specialize_program
from emmy.compiler.wire import intern, kernel_bindings, kernel_tile

from .format import GoldenFile, Latency, Measurements, Row, prepare_traced_graph
from .repository import is_repository_golden_path
from .restamp import definition


@dataclass(frozen=True)
class TraceInventoryResult:
    """Artifacts written for one trace-generated working inventory."""

    path: Path
    target_count: int


def preflight_trace_inventory(path: str | Path) -> Path:
    """Resolve a fresh trace-inventory destination and reject replacement."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing golden artifact: {destination}")
    return destination


def inventory(graph, ctx=None, *, model=None, model_quant_digest=None, realizations=None, name_prefix=None) -> GoldenFile:
    """A traced program's inventory as a document: every kernel it lowers to at ``ctx`` (the live card when none is
    given) and one unmeasured row per kernel and template."""
    document, ctx = _empty_inventory(ctx, model=model, model_quant_digest=model_quant_digest)
    _append(graph, ctx=ctx, document=document, name_prefix=name_prefix, realizations=realizations)
    return document


def write_trace_inventory(graph, path, *, model=None, ctx=None, realizations=None, model_quant_digest=None) -> TraceInventoryResult:
    """Lower a trace through fusion and write a self-contained target inventory."""
    destination = preflight_trace_inventory(path)
    document = inventory(graph, ctx, model=model, model_quant_digest=model_quant_digest, realizations=realizations)
    document.dump(destination)
    return TraceInventoryResult(path=destination, target_count=len(document.kernels))


def append_trace_inventory(graph, path, *, model=None, ctx=None, model_quant_digest=None) -> TraceInventoryResult:
    """Add one more traced program to an existing working inventory: a model whose whole-graph export is not
    bounded is traced one distinct path at a time, and those are one model's inventory. A kernel already covered is
    recorded once."""
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    with GoldenFile.edit(path) as document:
        validate_working_gpu(document, ctx)
        document = replace(document, model=model or document.model, model_quant_digest=model_quant_digest or document.model_quant_digest)
        added = _append(graph, ctx=ctx, document=document)
        document.dump(path, overwrite=True)
    return TraceInventoryResult(path=Path(path), target_count=added)


def write_trace_inventories(graphs: dict[str, object], path, *, model=None, ctx=None, realizations=None, model_quant_digest=None):
    """Combine named traces into one inventory: serving capture emits many independent pre/post/expert graphs, and a
    directory of one-file-per-graph inventories is easy to promote only partially. Identical kernels are recorded
    once. ``realizations`` is the row template every kernel takes, or a mapping from graph name to the rows that
    graph's kernels take — a static twin its own width's, a symbolic twin the dynamic ones."""
    destination = preflight_trace_inventory(path)
    if not graphs:
        raise ValueError("cannot write an empty trace inventory")
    document, ctx = _empty_inventory(ctx, model=model, model_quant_digest=model_quant_digest)
    added = 0
    for name in sorted(graphs):
        rows = realizations.get(name) if isinstance(realizations, Mapping) else realizations
        added += _append(graphs[name], ctx=ctx, document=document, name_prefix=name, realizations=rows)
    document.dump(destination)
    return TraceInventoryResult(path=destination, target_count=added)


def _empty_inventory(ctx, *, model, model_quant_digest) -> tuple[GoldenFile, object]:
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    document = GoldenFile(
        gpu_name=ctx.gpu_name or None,
        compute_cap=tuple(ctx.compute_capability),
        model=model or None,
        model_quant_digest=model_quant_digest or None,
    )
    return document, ctx


def whole_origins(coverage: Mapping, program) -> tuple[str, ...]:
    """The traced ops a kernel computes, when it computes every one of them whole — the frontend slice that is the
    kernel's exact Torch twin. Empty for a kernel holding part of an op."""
    origins = tuple(sorted(origin for origin in coverage if origin in program.nodes))
    return origins if origins and all(coverage[origin][2] for origin in origins) else ()


def _append(graph, *, ctx, document: GoldenFile, name_prefix: str | None = None, realizations: list[dict] | None = None) -> int:
    """Append one traced program's kernels to ``document`` — at every set of sizes the row templates bind, since a
    kernel specialized at a size is a kernel of its own — and a row per kernel and template. Returns the kernels
    added."""
    from .restamp import lift_targets  # noqa: PLC0415

    prepare_traced_graph(graph)
    traced = intern(document.programs, graph)
    templates = realizations if realizations is not None else [{"name": "", "bindings": {}, "pins": measured_precision_pins()}]
    by_bindings: dict[tuple, list[dict]] = {}
    for template in templates:
        by_bindings.setdefault(tuple(sorted(template.get("bindings", {}).items())), []).append(template)
    added = 0
    for bindings, group in by_bindings.items():
        program = specialize_program(graph, dict(bindings))
        fused = Pipeline.build(pipeline.LOOP_PASSES).run(program.copy(), ctx=ctx, db=None)
        totals = provenance.totals(fused)
        origins = {
            frozenset(node.buffer_names()): whole_origins(provenance.coverage(provenance.get(node), totals), graph)
            for node in fused.nodes.values()
            if isinstance(node.op, LoopOp)
        }
        for outputs, tile in lift_targets(program, ctx).items():
            kernel = definition(tile, tile.name, traced=traced, origins=origins.get(outputs, ()), bindings=dict(bindings))
            stored_before = len(document.kernels)
            stored = document.add_kernel(kernel)
            added += len(document.kernels) - stored_before
            for template in group:
                pins = dict(template.get("pins", {}))
                if any(row.kernel == stored.ref and row.pins == pins for row in document.rows):
                    continue
                name = f"{name_prefix}.{tile.name}" if name_prefix else tile.name
                if template.get("name"):
                    name = f"{name}.{template['name']}"
                if any(row.name == name and row.kernel != stored.ref for row in document.rows):
                    name = f"{name}.{stored.exact_identity[:12]}"
                document.rows.append(Row(name=name, kernel=stored.ref, bindings=kernel_bindings(tile), pins=pins))
    return added


def validate_working_gpu(document: GoldenFile, ctx) -> None:
    """Reject a working file recorded for a different concrete GPU."""
    file_cap, file_gpu = document.compute_cap, document.gpu_name
    if file_cap is not None and tuple(file_cap) not in ((0, 0), tuple(ctx.compute_capability)):
        raise ValueError(
            f"working golden targets compute capability {tuple(file_cap)}, but the live GPU is {tuple(ctx.compute_capability)}"
        )
    if file_gpu and ctx.gpu_name and gpu.canonical_name(file_gpu) != gpu.canonical_name(ctx.gpu_name):
        raise ValueError(f"working golden targets {file_gpu}, but the live GPU is {ctx.gpu_name}")


def _refuse_repository(path: Path) -> None:
    if is_repository_golden_path(path):
        raise ValueError(f"refusing to write measurements into a canonical repository golden: {path}")


def record_latency(path, name: str, *, hardware_id: str, emmy_us: float, tcompile_us=None, eager_us=None, knobs=None, pins=None) -> None:
    """Write one card's measured latencies into the row ``name`` (narrowed by ``knobs`` / ``pins`` when given, which
    must select exactly one). The block is keyed by ``Context.hardware_id``, which separates same-die SKUs a card
    name does not; each torch number is omitted rather than faked when the run did not time it."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    destination = Path(path)
    _refuse_repository(destination)
    wanted_knobs = canonical_row_key(knobs) if knobs is not None else None
    wanted_pins = {key: str(value) for key, value in pins.items()} if pins is not None else None
    with GoldenFile.edit(destination) as document:
        matches = [
            index
            for index, row in enumerate(document.rows)
            if row.name == name
            and (wanted_knobs is None or canonical_row_key(row.knobs or {}) == wanted_knobs)
            and (wanted_pins is None or {key: str(value) for key, value in row.pins.items()} == wanted_pins)
        ]
        if len(matches) != 1:
            raise ValueError(f"{destination} resolves {name!r} to {len(matches)} latency rows; exact knobs and pins must select one")
        torch_us = {field: float(us) for field, us in (("tcompile_us", tcompile_us), ("eager_us", eager_us)) if us}
        row = document.rows[matches[0]]
        document.rows[matches[0]] = replace(row, latency={**(row.latency or {}), hardware_id: Latency(emmy_us=float(emmy_us), **torch_us)})


def record_greedy_pick(path, name: str, *, decisions, kernels, reference_backend: str) -> list[str]:
    """Write the greedy pick's kernel set back into the working golden as the DB would hold it. ``decisions`` are the
    kernel-set decisions the compile took, ``(parent, arm, pieces)`` as the splice watcher reports them — each a
    routing row and the kernels it names; ``kernels`` are the CUDA kernels it produced, ``(op, emmy_us,
    reference_us)`` — each a measured row of its kernel at the seed row's regime (a precision gate the seed leaves
    open is the one the compile enumerated under), named ``<seed>.<identity prefix>``. A row of the same kernel, sizes, regime and
    schedule takes the new timings. Returns the names written, in order."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    destination = Path(path)
    _refuse_repository(destination)
    with GoldenFile.edit(destination) as document:
        seeds = document.rows_of(name)
        if not seeds:
            raise ValueError(f"{destination} has no realization named {name!r}")
        regime = {**measured_precision_pins(), **seeds[0].pins}
        for parent, arm, pieces in decisions:
            stored = document.add_kernel(definition(parent, parent.name))
            children = [document.add_kernel(definition(piece, piece.name)) for piece in pieces]
            document.add_routing(RoutingRow(stored.ref, {str(k): str(v) for k, v in arm.items()}, tuple(c.ref for c in children)))
        written = []
        for op, emmy_us, reference_us in kernels:
            tile = kernel_tile(op)
            if tile is None:
                raise ValueError(f"kernel {op.kernel_name} lowered from no tile kernel, so no row can name it")
            stored = document.add_kernel(definition(tile, op.kernel_name))
            row = Row(
                name=f"{name}.{stored.exact_identity[:12]}",
                kernel=stored.ref,
                bindings=kernel_bindings(tile),
                pins=regime,
                knobs=dict(canonical_row_key({k: v for k, v in (op.knobs or {}).items() if not is_placement_knob(k, v)})),
                measurements=Measurements(emmy_us=float(emmy_us), reference_us=float(reference_us), reference_backend=reference_backend),
            )
            written.append(document.upsert_row(row).name)
        return written
