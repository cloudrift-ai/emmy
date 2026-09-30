"""Mutable working-golden inventories, candidates, and ranking feedback.

This module owns the untrusted side of the golden file workflow: trace inventory
generation, target reconstruction, exact proposal measurement, and atomic ranking
persistence. CLI commands only validate argument combinations and report errors.
"""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from emmy import gpu
from emmy.compiler.pipeline.search.golden import (
    Config,
    GoldenFile,
    Latency,
    Measurements,
    Realization,
    Target,
    is_repository_golden_path,
    prepare_traced_graph,
)


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


def write_trace_inventory(
    graph,
    path: str | Path,
    *,
    model: str | None = None,
    ctx=None,
    realizations: list[dict] | None = None,
    model_quant_digest: str | None = None,
) -> TraceInventoryResult:
    """Lower a trace through fusion and write a self-contained target inventory."""
    destination = preflight_trace_inventory(path)
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    programs: list[dict] = []
    loops: list[dict] = []
    entries: list[Config] = []
    _append_trace_inventory(
        graph,
        ctx=ctx,
        programs=programs,
        loops=loops,
        entries=entries,
        realizations=realizations,
    )
    _dump_trace_inventory(
        destination,
        ctx=ctx,
        model=model,
        model_quant_digest=model_quant_digest,
        programs=programs,
        loops=loops,
        entries=entries,
    )
    return TraceInventoryResult(path=destination, target_count=len(entries))


def append_trace_inventory(
    graph,
    path: str | Path,
    *,
    model: str | None = None,
    ctx=None,
    model_quant_digest: str | None = None,
) -> TraceInventoryResult:
    """Add one more traced program to an existing working inventory.

    A model whose whole-graph export is not bounded is traced one distinct path at a
    time -- each decoder-layer kind, then the embedding, final normalization and output
    seams. Those are one model's inventory, so they belong in one file: a directory of
    one-file-per-path is easy to promote only partially. Interning is shared with
    :func:`write_trace_inventories`, so a kernel already covered is recorded once.
    """
    destination = Path(path)
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    document = GoldenFile.load(destination)
    validate_working_gpu(document, ctx)
    programs, loops, entries = document.programs, document.loops, document.configs
    seen_loops = {entry.target.loop for entry in entries}
    before = len(entries)
    _append_trace_inventory(
        graph,
        ctx=ctx,
        programs=programs,
        loops=loops,
        entries=entries,
        seen_loops=seen_loops,
    )
    _dump_trace_inventory(
        destination,
        ctx=ctx,
        model=model or document.model,
        model_quant_digest=model_quant_digest or document.model_quant_digest,
        programs=programs,
        loops=loops,
        entries=entries,
        overwrite=True,
    )
    return TraceInventoryResult(path=destination, target_count=len(entries) - before)


def write_trace_inventories(
    graphs: dict[str, object],
    path: str | Path,
    *,
    model: str | None = None,
    ctx=None,
    realizations: list[dict] | Mapping[str, list[dict]] | None = None,
    model_quant_digest: str | None = None,
) -> TraceInventoryResult:
    """Combine named traces into one exact-Loop-IR working inventory.

    Serving capture emits many independent pre/post/expert graphs.  A directory of
    one-file-per-graph inventories is awkward to tune and, more importantly, easy to
    promote only partially.  This writer interns all of their programs and Loop IR
    targets into one self-contained artifact.  Identical Loop programs are recorded
    once: they are the same tuning target even when several serving twins consult it.
    ``realizations`` is the row template every target takes, or a mapping from graph name
    to the rows that graph's targets take — a static twin its own width's, a symbolic twin
    the dynamic ones.
    """
    destination = preflight_trace_inventory(path)
    if not graphs:
        raise ValueError("cannot write an empty trace inventory")
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    programs: list[dict] = []
    loops: list[dict] = []
    entries: list[Config] = []
    seen_loops: set[int] = set()
    for name in sorted(graphs):
        _append_trace_inventory(
            graphs[name],
            ctx=ctx,
            programs=programs,
            loops=loops,
            entries=entries,
            name_prefix=name,
            seen_loops=seen_loops,
            realizations=realizations.get(name) if isinstance(realizations, Mapping) else realizations,
        )
    _dump_trace_inventory(
        destination,
        ctx=ctx,
        model=model,
        model_quant_digest=model_quant_digest,
        programs=programs,
        loops=loops,
        entries=entries,
    )
    return TraceInventoryResult(path=destination, target_count=len(entries))


def whole_origins(coverage: Mapping, program) -> tuple[str, ...]:
    """The traced ops a kernel computes, when it computes every one of them whole — the frontend
    slice that is the kernel's exact Torch twin. Empty for a kernel holding part of an op."""
    origins = tuple(sorted(origin for origin in coverage if origin in program.nodes))
    return origins if origins and all(coverage[origin][2] for origin in origins) else ()


def kernel_programs(fused) -> list[tuple[str, object]]:
    """One standalone Loop IR program per kernel of a lowered graph, ``(kernel node id, program)``
    in topological order — the targets an inventory stores and ``emmy compile --wire`` prints."""
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    kernels = [node_id for node_id in fused.topological_order() if isinstance(fused.nodes[node_id].op, LoopOp)]
    return [(node_id, single_node_graph(fused, node_id)) for node_id in kernels]


def lowered_kernels(graph, *, ctx) -> tuple[object, list[tuple[str, object]]]:
    """``graph`` through the loop passes at ``ctx``: ``(fused graph, kernel programs)``. The one
    lowering a trace inventory, ``emmy golden check`` and a restamp share, so a stored target and
    its fresh lowering can only differ where the compiler differs."""
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415

    fused = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)
    return fused, kernel_programs(fused)


def _append_trace_inventory(
    graph,
    *,
    ctx,
    programs: list[dict],
    loops: list[dict],
    entries: list[Config],
    name_prefix: str | None = None,
    seen_loops: set[int] | None = None,
    realizations: list[dict] | None = None,
) -> None:
    """Append one lowered graph to shared trace-inventory pools."""
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import measured_precision_pins  # noqa: PLC0415
    from emmy.compiler.wire import intern  # noqa: PLC0415  # noqa: PLC0415

    prepare_traced_graph(graph)
    input_graph = graph.copy()
    fused, kernels = lowered_kernels(graph, ctx=ctx)

    # Persist the pristine program once.  Per-target frontend slices are useful
    # ephemeral tuning views, but they can change fusion when independently
    # lowered (especially sibling linears and computed-A cones).  Provenance
    # selectors must therefore resolve against the original trace context.
    program_ref: int | None = None
    inventory = []
    totals = provenance.totals(fused)
    for node_id, program in kernels:
        node = fused.nodes[node_id]
        inventory.append((node_id, node, program, whole_origins(provenance.coverage(provenance.get(node), totals), input_graph)))
    used_names = {realization.name for entry in entries for realization in entry.realizations}

    for node_id, node, program, origins in inventory:
        # The entry's name is a label, never re-derived: the kernel's provenance name (the ops it
        # realizes, as the backend and the profiler show it), so a reader can tell what a row is.
        name = node.op.name or node_id
        if name_prefix:
            name = f"{name_prefix}.{name}"
        if name in used_names:
            # One kernel name can occur at multiple exact Loop targets whose boundary shapes or
            # checkpoint sources differ. ``emmy run --golden`` resolves by name, so retaining the
            # bare duplicate makes the generated file impossible to replay. Node ids are
            # deterministic within the persisted source program and distinguish these sites.
            base = f"{name}.{node_id}"
            name = base
            duplicate = 2
            while name in used_names:
                name = f"{base}.{duplicate}"
                duplicate += 1
        used_names.add(name)
        # The target IS the kernel's Loop IR: a replay starts from the stored kernel and never re-lowers
        # the program. The traced ops it computes whole ride beside it as provenance — the frontend
        # slice a benchmark compares the kernel against.
        loop_ref = intern(loops, program)
        if seen_loops is not None and loop_ref in seen_loops:
            continue
        if seen_loops is not None:
            seen_loops.add(loop_ref)
        target = Target(loop=loop_ref, origins=tuple(origins))
        if program_ref is None:
            program_ref = intern(programs, input_graph)
        if realizations is None:
            rows = [Realization(name=name, bindings={}, pins=measured_precision_pins())]
        else:
            rows = [
                Realization.from_wire({**template, "name": f"{name}.{template['name']}" if template["name"] else name})
                for template in realizations
            ]
        entries.append(Config(program=program_ref, target=target, realizations=rows))


def _dump_trace_inventory(
    destination: Path,
    *,
    ctx,
    model: str | None,
    model_quant_digest: str | None,
    programs: list[dict],
    loops: list[dict],
    entries: list[Config],
    overwrite: bool = False,
) -> None:
    """Write shared trace-inventory pools with their card and model provenance."""
    document = GoldenFile(
        gpu_name=ctx.gpu_name or None,
        compute_cap=tuple(ctx.compute_capability),
        model=model or None,
        model_quant_digest=model_quant_digest or None,
        programs=programs,
        configs=entries,
        loops=loops,
    )
    document.dump(destination, overwrite=overwrite)


def validate_working_gpu(document: GoldenFile, ctx) -> None:
    """Reject a working file recorded for a different concrete GPU."""
    file_cap = document.compute_cap
    file_gpu = document.gpu_name
    if file_cap is not None and tuple(file_cap) != (0, 0) and tuple(file_cap) != tuple(ctx.compute_capability):
        raise ValueError(
            f"working golden targets compute capability {tuple(file_cap)}, but the live GPU is {tuple(ctx.compute_capability)}"
        )
    if file_gpu and ctx.gpu_name and gpu.canonical_name(file_gpu) != gpu.canonical_name(ctx.gpu_name):
        raise ValueError(f"working golden targets {file_gpu}, but the live GPU is {ctx.gpu_name}")


def record_latency(
    path: str | Path,
    name: str,
    *,
    hardware_id: str,
    emmy_us: float,
    tcompile_us: float | None,
    eager_us: float | None = None,
    knobs: dict | None = None,
    pins: dict | None = None,
) -> None:
    """Write one card's measured latencies back into a working golden's realization.

    The per-card block is keyed by ``Context.hardware_id`` — the identity that already separates
    same-die SKUs like H100 from H200, which a free-text card name does not. It is separate from
    the flat ``measurements`` block because a model golden is one file per card (so the card is
    implied by the file) while a file measured on several cards needs a row each.

    Both numbers where both exist, because the block answers two questions and only one of them is
    a ratchet: ``emmy_us`` against its own stored value says *did we regress*, and ``tcompile_us`` /
    ``eager_us`` beside it say *are we ahead of or behind torch*, per case, per card. Each is
    omitted rather than faked when the run did not time it.

    Read and written inside one :func:`exclusive_golden`, like every measurement this module writes
    back: a run passing both ``--record`` and ``--record-greedy`` writes twice, and a stale second
    document would drop whatever landed between the two.
    """
    destination = Path(path)
    if is_repository_golden_path(destination):
        raise ValueError(f"refusing to write measurements into a canonical repository golden: {destination}")
    with exclusive_golden(destination):
        torch_us = {"tcompile_us": tcompile_us, "eager_us": eager_us}
        _record_latency_row(destination, name, hardware_id=hardware_id, emmy_us=emmy_us, torch_us=torch_us, knobs=knobs, pins=pins)


def _record_latency_row(destination: Path, name: str, *, hardware_id, emmy_us, torch_us, knobs, pins) -> None:
    """One card's latencies written into the file as it stands NOW. Runs under the lock."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    document = GoldenFile.load(destination)
    wanted_knobs = canonical_row_key(knobs) if knobs is not None else None
    wanted_pins = tuple(sorted((key, str(value)) for key, value in pins.items())) if pins is not None else None
    matches = []
    for entry in document.configs:
        for realization in entry.realizations:
            if realization.name != name:
                continue
            if wanted_knobs is not None and canonical_row_key(realization.knobs or {}) != wanted_knobs:
                continue
            got_pins = tuple(sorted((key, str(value)) for key, value in realization.pins.items()))
            if wanted_pins is not None and got_pins != wanted_pins:
                continue
            matches.append(realization)
    if len(matches) != 1:
        raise ValueError(f"{destination} resolves {name!r} to {len(matches)} latency rows; exact knobs and pins must select one")
    timings = Latency(emmy_us=float(emmy_us), **{field: float(us) for field, us in torch_us.items() if us})
    matches[0].latency = {**(matches[0].latency or {}), hardware_id: timings}
    document.dump(destination, overwrite=True)


def kernel_set_prices(kernel_sets: list[tuple[str, tuple[str, ...]]], launch_us: dict[str, float]) -> list[float | None]:
    """The measured price of each kernel-set decision: the summed launch timings of the kernels it
    produced, in the same units as the schedule receipt of the kernel it replaced — the two arms a
    kernel-set fork ranks against each other (``policy.greedy._route_candidates``). ``kernel_sets``
    pairs each decision, in decision order, with the graph ids its splice consumed and minted
    (``(consumed root id, minted ids)``, as the splice watcher ``inventory.KernelInventory`` reports
    them): a later decision that consumed one
    of an earlier decision's kernels stands in for it with its own kernels. ``launch_us`` maps the
    terminal graph's CUDA kernel ids to their launch timings. ``None`` where a kernel of the set is
    not among the launches, so the caller can fall back to the whole graph's timing."""
    consumed = {root: index for index, (root, _) in enumerate(kernel_sets)}
    memo: dict[int, set[str]] = {}

    def kernels(index: int) -> set[str]:
        if index not in memo:
            out: set[str] = set()
            for node_id in kernel_sets[index][1]:
                later = consumed.get(node_id)
                out |= kernels(later) if later is not None and later > index else {node_id}
            memo[index] = out
        return memo[index]

    return [
        sum(launch_us[node_id] for node_id in ids) if ids and all(node_id in launch_us for node_id in ids) else None
        for ids in (kernels(index) for index in range(len(kernel_sets)))
    ]


def greedy_pick_rows(graph) -> list[tuple[str, dict[str, str]]]:
    """Each CUDA kernel of a compiled graph, in launch order, as ``(identity, schedule row)``: the
    deploy identity of the tile kernel it lowered from — what a child-identity schedule receipt
    names — and the schedule row it realized (the schedule families only; a forkless kernel's row
    is its OFF anchors, which is what its one enumerated row spells)."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415
    from emmy.compiler.wire import kernel_tile  # noqa: PLC0415

    rows: list[tuple[str, dict[str, str]]] = []
    for node_id in graph.topological_order():
        op = graph.nodes[node_id].op
        if not isinstance(op, CudaOp):
            continue
        tile = kernel_tile(op)
        identity = tile.identity_key(with_io=True) if tile is not None else None
        if identity is None:
            raise ValueError(f"kernel {op.kernel_name} lowered from no tile kernel, so no receipt can name it")
        rows.append((identity, dict(schedule_row_key(dict(op.knobs or {})))))
    return rows


@contextlib.contextmanager
def exclusive_golden(path: Path):
    """Hold one working golden's read-modify-write against every other process on this machine.

    A recorder loads the file, adds its rows and writes the whole document back, so two runs that
    both load before either writes each dump a document missing the other's rows — 18 of 151
    realizations recorded in one parallel round were lost that way. The reload belongs INSIDE this
    lock: locking a stale document would serialize the loss, not stop it. ``flock`` releases with
    the descriptor, on a killed process too."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as handle:  # noqa: PTH123, SIM115 — flock takes a descriptor
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def record_greedy_pick(
    path: str | Path,
    name: str,
    *,
    decisions: list[tuple[str, dict, float, float]],
    kernels: list[tuple[str, dict, float, float]],
    reference_backend: str,
) -> list[str]:
    """Write the greedy pick's kernel set back into ``name``'s target as measured realizations.

    ``decisions`` are the kernel-set decisions the compile took — ``(identity, arm knobs, emmy_us,
    reference_us)``, the identity being the kernel the fork was offered on — and each becomes a
    routing row: the measured price of that decision, carrying the kernel set's whole-graph
    timings. ``kernels`` are the CUDA kernels it produced — ``(identity, schedule row, emmy_us,
    reference_us)`` — and each becomes a child-identity schedule receipt with its own launch
    timings. Every row takes the seed realization's bindings and input regime and no route: seam
    spellings are kernel-local, so a cut key copied onto every receipt would re-cut any piece that
    offers a same-spelled seam; the replay follows the routing rows, each naming its kernel by
    identity. A row already recorded for the same input regime, kernel, and knobs takes the new
    timings; anything else is appended, so a re-record never duplicates or aliases measurements
    from another width or pin regime. The file is read and written back inside one
    :func:`exclusive_golden`, so the rows a concurrent recorder wrote meanwhile survive.
    Returns the names written, in order.
    """
    destination = Path(path)
    if is_repository_golden_path(destination):
        raise ValueError(f"refusing to write measurements into a canonical repository golden: {destination}")
    with exclusive_golden(destination):
        return _record_rows(destination, name, decisions=decisions, kernels=kernels, reference_backend=reference_backend)


def _record_rows(destination: Path, name: str, *, decisions, kernels, reference_backend: str) -> list[str]:
    """The rows of one greedy pick, added to the file as it stands NOW. Runs under the lock."""
    from emmy.compiler.pipeline.knob import canonical_row_key, family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import measured_precision_pins  # noqa: PLC0415

    document = GoldenFile.load(destination)
    seeds = [(entry, realization) for entry in document.configs for realization in entry.realizations if realization.name == name]
    if not seeds:
        raise ValueError(f"{destination} has no realization named {name!r}")

    def regime_of(seed: Realization) -> dict:
        # The seed's regime, with the precision gates the compile ACTUALLY enumerated under laid over
        # it: a row measured with the reduced-accumulate cell offered must say so, or a replay
        # republishes a regime that no longer offers it (``measured_precision_pins``).
        return {**{key: value for key, value in seed.pins.items() if family_of(str(key)) != "PLACE"}, **measured_precision_pins()}

    if decisions and len(seeds) > 1:
        # One name can hold several routing rows for one target. The seed is the one whose route the
        # compile took — the row its decision lands on; any other ends up with knobs spelling one
        # route and a kernel set naming another.
        identity, knobs = decisions[0][:2]
        route = (identity, canonical_row_key(knobs))
        seeds = [(e, r) for e, r in seeds if r.pins == regime_of(r) and (r.identity, canonical_row_key(r.knobs or {})) == route]
        if len(seeds) != 1:
            raise ValueError(
                f"{destination} holds several realizations named {name!r} and {len(seeds)} of them record the route the "
                f"compile took ({identity[:12]} {dict(knobs)}) under its pins; exactly one must, to carry the kernel set"
            )
    entry, seed = seeds[0]
    regime = regime_of(seed)
    written: list[str] = []
    for identity, knobs, emmy_us, reference_us in (*decisions, *kernels):
        row = Realization(
            name=f"{name}.{identity[:12]}",
            bindings=dict(seed.bindings),
            pins=dict(regime),
            knobs={str(key): str(value) for key, value in knobs.items()},
            identity=identity,
            measurements=Measurements(emmy_us=float(emmy_us), reference_us=float(reference_us), reference_backend=reference_backend),
        )
        key = (row.bindings, row.pins, identity, canonical_row_key(row.knobs))
        recorded = next(
            (r for r in entry.realizations if (r.bindings, r.pins, r.identity, canonical_row_key(r.knobs or {})) == key),
            None,
        )
        if recorded is None:
            entry.realizations.append(row)
        else:
            recorded.measurements = row.measurements
        # The name of the row that CARRIES the measurement — the existing row's wherever the write
        # landed on one. A route whose seam the seed itself records matches the seed on every key,
        # so the decision lands there, and naming the row that was not written leaves
        # ``kernel_set`` pointing at nothing: the file is refused on the way out and the
        # measurement just taken is lost.
        written.append(row.name if recorded is None else recorded.name)
    if decisions:
        # A realization's rows describe ONE kernel set. Re-recording the same realization under a
        # different route rewrites the listing, and leaving the superseded set's rows behind makes
        # the file self-contradictory: a schedule row does not store the route it was measured
        # under, so the replay reads every row against whichever listing the seed now carries. The
        # A100's softmax x V target replayed at 67 ms instead of 166 us that way. Scoped to THIS
        # seed's family and regime — one config entry can hold several seeds side by side — and
        # never to the seed itself, which carries no measurement of its own.
        superseded = set(written[: len(decisions)])
        entry.realizations = [
            r
            for r in entry.realizations
            if not (
                r.name.startswith(f"{name}.")
                and r.bindings == seed.bindings
                and r.pins == regime
                and r.name not in written
                and r.name not in superseded
            )
        ]
        seed.kernel_set = tuple(written[: len(decisions)])
    document.dump(destination, overwrite=True)
    return written
