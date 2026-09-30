"""A repository golden against the fresh lowering of its own programs — the check and the fix.

A golden's rows are evidence for the kernels its stored Loop IR names, and a deploy keys them by
the kernels it lowers FRESH from the model. When the compiler starts lowering a program differently
the two drift apart: every row still decodes against its stored kernel while serving builds kernels
no row describes, and a strict boot refuses. :func:`stale_targets` names the stored targets a fresh
lowering no longer writes; :func:`restamp` rewrites the golden onto that lowering, which is what
the ``refresh-golden`` skill runs after a compiler change moves a lowering. Both are GPU-free:
lowering is the loop passes alone, and a kernel's CUDA source renders without a card.

What a restamp keeps is decided per row, never guessed:

- a target a fresh kernel writes (same output set) takes that kernel's Loop IR; one no fresh kernel
  writes — the layer regrouped — is dropped with its rows, since a row is a schedule of one kernel;
- a row whose stored identity is the target's own takes the fresh target's; a row naming a piece of
  the target's kernel set (a receipt, or a piece row beside its routing row) keeps its identity, and
  survives only if the fresh set still mints that piece under the set's own rows;
- a route key no kernel of the fresh lowering resolves makes the file stale too — a node that changed
  only its kind (a reduce read as a contraction) re-spells every route through it while every stored
  target keeps its Loop IR. Restamp re-spells such a key onto the seam its operand positions reach —
  a slab-and-computed pair that swapped its order counting as the same position — and drops the row
  when they reach none;
- a row that no longer decodes on the fresh kernel is dropped;
- a measurement stays only when the kernel it timed is the kernel the fresh Loop IR renders, byte
  for byte, under the row as the compile's only evidence. Otherwise the row keeps its schedule and
  loses its microseconds: a proposal, no evidence until a record run on the card measures it again.

A file nothing survives in is not written: deleting a golden or re-recording it on the card is a
decision, not a restamp.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.ir.tile.path import family_sites, parse_key, sites
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.pins import composed_routes, pinned_knobs, unpinned_decisions
from emmy.compiler.structural import digest

from .decode import Spelling, _replay, decode_record
from .format import Config, GoldenEntryState, GoldenFile, Realization
from .record import GoldenRecord, GoldenRecords, _lifted_target
from .working import lowered_kernels


def fresh_kernels(document: GoldenFile, programs: Sequence[int] | None = None) -> dict[int, dict[frozenset, dict]]:
    """The kernels a fresh lowering of each stored program writes, keyed by output set, per program
    index — ``emmy compile --golden PATH --program N --ir loop -o fresh.json``, as data."""

    ctx = Context.from_target(tuple(document.compute_cap), gpu_name=document.gpu_name)
    fresh: dict[int, dict[frozenset, dict]] = {}
    for index in programs if programs is not None else sorted({entry.program for entry in document.configs}):
        _fused, kernels = lowered_kernels(document.program(index), ctx=ctx)
        fresh[index] = {frozenset(wire["outputs"]): wire for wire in (program.to_wire() for _, program in kernels)}
    return fresh


def _wire_digest(wire: Mapping) -> str:
    return digest(json.dumps(wire, sort_keys=True, default=str))


def fresh_kernel_digests(document: GoldenFile, program: int) -> dict[str, str]:
    """The fresh lowering of traced ``program`` as ``{sorted output set: Loop IR digest}`` — what the
    check compares a stored target against."""
    fresh = fresh_kernels(document, [program])[program]
    return {",".join(sorted(outputs)): _wire_digest(wire) for outputs, wire in fresh.items()}


def _entry_name(entry: Config) -> str:
    return entry.realizations[0].name if entry.realizations else f"loop {entry.target.loop}"


def _stale_reason(document: GoldenFile, entry: Config, fresh: Mapping[str, str]) -> str | None:
    stored = document.loops[entry.target.loop]
    fresh_digest = fresh.get(",".join(sorted(stored["outputs"])))
    if fresh_digest == _wire_digest(stored):
        return None
    return f"{_entry_name(entry)}: " + (
        "no fresh kernel writes its outputs" if fresh_digest is None else "the fresh kernel's Loop IR differs"
    )


def stale_reasons(document: GoldenFile, program: int, fresh: Mapping[str, str]) -> list[str]:
    """The stored targets of traced ``program`` that ``fresh`` (:func:`fresh_kernel_digests`) does not
    write, and the route keys of its current targets that name no seam of that lowering, ``name: reason``."""
    entries = [entry for entry in document.configs if entry.program == program]
    reasons = [reason for reason in (_stale_reason(document, entry, fresh) for entry in entries) if reason is not None]
    for entry in entries:
        if _stale_reason(document, entry, fresh) is None:
            reasons += _stale_route_reasons(document, entry)
    return reasons


def _route_keys(record: GoldenRecord) -> frozenset[str]:
    """The scoped ``PLACE`` keys a record marks cut — the seams its route names."""
    return frozenset(key for key, value in record.route.items() if family_of(key) == "PLACE" and key != "PLACE" and value == "cut")


def unresolved_route_keys(record: GoldenRecord, keys: frozenset[str], records: Sequence[GoldenRecord] = ()) -> list[str]:
    """The cut keys no decision consumes when this record's kernel set replays through ``tile/cut``.
    Parent routes must run first: a child's seam belongs to the kernel its parent cut creates,
    and publishing all keys as global pins can consume it on the wrong kernel."""
    records = GoldenRecords.of(records)
    entries = GoldenRecords((record, *records.siblings(record)))
    own = _lifted_target(record).identity_key(with_io=True)
    lead = next((entry for entry in entries if entry.identity == own), records.lead(record))
    spelling = Spelling(entries, lead, own=record)
    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)

    def decide(fp):
        identity = fp.root_op.identity_key(with_io=True)
        arm = spelling.structural(fp, spelling.decider(identity))
        return arm[0] if arm is not None else next(fp.leaves())

    with unpinned_decisions(), pinned_knobs(record.regime), composed_routes(spelling.composed()):
        Run(Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]), ctx).resolve(record.target_program.copy(), decide)
    return sorted(keys - spelling.consumed)


def _stale_route_reasons(document: GoldenFile, entry: Config) -> list[str]:
    checked = set()
    reasons = []
    records = GoldenRecords(document.record(entry, row) for row in entry.realizations)
    for record in records:
        keys = _route_keys(record)
        identity = (record.identity, record.pins, keys)
        if not keys or identity in checked:
            continue
        checked.add(identity)
        reasons += [
            f"{record.name}: route key {key!r} names no seam of the fresh lowering" for key in unresolved_route_keys(record, keys, records)
        ]
    return reasons


def stale_targets(document: GoldenFile, program: int | None = None) -> Iterator[str]:
    """Every stored target the fresh lowering no longer writes, ``name: reason`` — of one traced
    ``program``, or of every program smallest first, so a caller that only needs to know whether the
    file is stale stops at the first one for the cost of one small program."""
    programs = {entry.program for entry in document.configs if program is None or entry.program == program}
    for index in sorted(programs, key=lambda index: len(document.programs[index]["nodes"])):
        yield from stale_reasons(document, index, fresh_kernel_digests(document, index))


@dataclass
class RestampReport:
    targets: int = 0
    restamped: int = 0
    dropped_targets: list[str] = field(default_factory=list)
    rows_kept: int = 0
    rows_demoted: list[str] = field(default_factory=list)
    rows_dropped: list[str] = field(default_factory=list)
    rows_respelled: list[str] = field(default_factory=list)
    rows_rekeyed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.restamped or self.dropped_targets or self.rows_dropped or self.rows_respelled or self.rows_rekeyed)

    def lines(self) -> list[str]:
        out = [f"{self.restamped} of {self.targets} targets restamped, {len(self.dropped_targets)} dropped; {self.rows_kept} rows kept"]
        out.extend(f"dropped target {reason}" for reason in self.dropped_targets)
        out.extend(f"re-spelled the route of {reason}" for reason in self.rows_respelled)
        out.extend(f"re-keyed {name} onto the target's identity under the current compiler" for name in self.rows_rekeyed)
        out.extend(f"demoted to a proposal {name}" for name in self.rows_demoted)
        out.extend(f"dropped row {reason}" for reason in self.rows_dropped)
        return out


def restamp(document: GoldenFile) -> tuple[GoldenFile | None, RestampReport]:
    """The golden rewritten onto the fresh lowering of its programs, and what that decided. The
    document comes back ``None`` when no target survives."""
    fresh = fresh_kernels(document)
    report = RestampReport(targets=len(document.configs))
    loops: list[dict] = []
    loop_index: dict[str, int] = {}

    def intern(wire: dict) -> int:
        key = json.dumps(wire, sort_keys=True)
        if key not in loop_index:
            loop_index[key] = len(loops)
            loops.append(wire)
        return loop_index[key]

    configs = []
    for entry in document.configs:
        stored = document.loops[entry.target.loop]
        wire = fresh[entry.program].get(frozenset(stored["outputs"]))
        if wire is None:
            report.dropped_targets.append(f"{_entry_name(entry)}: no fresh kernel writes its outputs")
            continue
        # An unchanged target still re-decodes its rows: a piece of its kernel set can take another
        # identity while the target's own Loop IR stays the same.
        rows = _rekeyed_rows(document, entry, wire, report)
        if not rows:
            report.dropped_targets.append(f"{_entry_name(entry)}: no row survives on the fresh kernel")
            continue
        report.restamped += wire != stored
        configs.append(replace(entry, target=replace(entry.target, loop=intern(wire)), realizations=rows))
    if not configs:
        return None, report
    used = sorted({entry.program for entry in configs})
    renumbered = {old: new for new, old in enumerate(used)}
    out = replace(
        document,
        programs=[document.programs[index] for index in used],
        loops=loops,
        configs=[replace(entry, program=renumbered[entry.program]) for entry in configs],
    )
    return out, report


def _rekeyed_rows(document: GoldenFile, entry: Config, wire: dict, report: RestampReport) -> list[Realization]:
    """The entry's realizations re-keyed to the fresh kernel ``wire``: identities moved, rows that
    stop decoding dropped, measurements of a kernel that renders differently demoted."""
    scratch = replace(document, loops=[*document.loops, wire])
    fresh_entry = replace(entry, target=replace(entry.target, loop=len(document.loops)))
    kept = [row for row in (_respelled_route(scratch, fresh_entry, row, report) for row in entry.realizations) if row is not None]
    entry = replace(entry, realizations=kept)
    old_records = [document.record(entry, row) for row in entry.realizations]
    new_records = [scratch.record(fresh_entry, row) for row in entry.realizations]

    # A row naming the target itself takes the fresh target's identity. Any other stored identity
    # names a piece of the target's kernel set (a receipt, or a piece row beside its routing row):
    # it stays as stored, and the decode below keeps the row only if the fresh set still mints that
    # piece under the set's own rows. One identity shared by every row of the entry, and not the
    # target's as the compiler computes it today, is the target's own under the compiler that
    # recorded it — a piece row comes beside its routing row, never alone — so it moves too: that is
    # the case of a kernel whose Loop IR stayed while what the lift makes of it changed.
    old_key = replace(old_records[0], identity=None).kernel_identity
    new_key = replace(new_records[0], identity=None).kernel_identity
    target_identity = old_records[0].identity if old_records[0].is_routing or len({old.identity for old in old_records}) == 1 else old_key
    report.rows_rekeyed.extend(old.name for old in old_records if old.identity == target_identity and target_identity != old_key)
    survivors = GoldenRecords(
        replace(new, identity=new_key) if old.identity in (old_key, target_identity) else new
        for old, new in zip(old_records, new_records, strict=True)
    )

    rows = []
    demoted_rows: set[int] = set()
    for realization, old, new in zip(entry.realizations, old_records, survivors, strict=True):
        reason = decode_record(new, survivors)
        if reason is not None:
            report.rows_dropped.append(f"{old.name}: {reason}")
            continue
        row = replace(realization, identity=new.identity if new.identity is not None else realization.identity)
        if row.measurements is not None or row.latency is not None:
            if wire != document.loops[entry.target.loop] and (
                (old_sources := _kernel_sources(old, old_records)) is None
                or (new_sources := _kernel_sources(new, survivors)) is None
                or old_sources != new_sources
            ):
                row = replace(row, measurements=None, latency=None)
                report.rows_demoted.append(old.name)
                demoted_rows.add(id(row))
            else:
                report.rows_kept += 1
        else:
            report.rows_kept += 1
        rows.append(row)
    # A dropped piece can invalidate another row that decoded while the piece was still present.
    while rows:
        current = GoldenRecords(scratch.record(fresh_entry, row) for row in rows)
        rejected = []
        for row, record in zip(rows, current, strict=True):
            reason = decode_record(record, current)
            if reason is None and row.kernel_set_state(rows) is GoldenEntryState.INVENTORY:
                reason = "its kernel set lost its measurements"
            if reason is not None:
                rejected.append((row, reason))
        if not rejected:
            break
        for row, reason in rejected:
            rows.remove(row)
            if id(row) in demoted_rows:
                report.rows_demoted.remove(row.name)
            else:
                report.rows_kept -= 1
            report.rows_dropped.append(f"{row.name}: {reason}")
    return rows


def _respelled_route(document: GoldenFile, entry: Config, row: Realization, report: RestampReport) -> Realization | None:
    """``row`` with every route key the fresh lowering no longer resolves re-spelled onto the seam the
    same operand positions reach — a seam whose node changed only its kind (a reduce that became a
    contraction reads ``inner`` where it read ``reduce``). ``None``, reported as dropped, when a key's
    positions reach no seam or the re-spelled route still names one nothing resolves: a row whose
    route decides nothing would replay as another kernel set under its name."""
    record = document.record(entry, row)
    keys = _route_keys(record)
    records = GoldenRecords(document.record(entry, row) for row in entry.realizations)
    stale = unresolved_route_keys(record, keys, records) if keys else []
    if not stale:
        return row
    tile = _lifted_target(record)
    renamed: dict[str, str] = {}
    for key in stale:
        site = seam_at_positions(tile.op, key)
        if site is None:
            report.rows_dropped.append(f"{row.name}: route key {key!r} names no seam of the fresh lowering, and its positions reach none")
            return None
        renamed[key] = f"PLACE@{site.path}"
    respelled = replace(
        row,
        pins={renamed.get(key, key): value for key, value in row.pins.items()},
        knobs=None if row.knobs is None else {renamed.get(key, key): value for key, value in row.knobs.items()},
    )
    if still := unresolved_route_keys(document.record(entry, respelled), frozenset(renamed.get(key, key) for key in keys), records):
        report.rows_dropped.append(f"{row.name}: route key {still[0]!r} names no seam of the fresh lowering, re-spelled or not")
        return None
    report.rows_respelled.append(f"{row.name}: " + ", ".join(f"{old} -> {new}" for old, new in renamed.items()))
    return respelled


def seam_at_positions(op, key: str):
    """The ``PLACE`` site of ``op`` the operand positions of ``key`` reach, whatever kinds its hops name,
    or ``None``. A slab-and-computed pair that swapped its order (the computed operand became A) is
    still one seam: a key names a computed operand, never a slab, so a hop that lands on the slab
    names the other operand."""
    placeable = {id(site.node): site for site in family_sites("PLACE", sites(op))}
    node = op
    for _label, index in parse_key(key).hops:
        operands = node.operands
        node = operands[index - 1] if index <= len(operands) else None
        if node is not None and node.as_slab() is not None and len(operands) == 2:
            node = next((edge for edge in operands if edge is not node and edge.as_slab() is None), node)
        if node is None or node.as_slab() is not None:
            return None
    return placeable.get(id(node))


def _kernel_sources(record: GoldenRecord, records: Sequence[GoldenRecord]) -> tuple[str, ...] | None:
    """The CUDA sources the record's route and schedule render with its sibling rows. The strict
    golden replay selects those rows before lowering, so source comparison does not search an
    unrelated schedule pool. ``None`` when replay or lowering refuses."""

    siblings = [other for other in GoldenRecords.of(records).siblings(record) if other.is_routing or other.identity != record.identity]
    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    try:
        replay = _replay(record, siblings)
        graph = Pipeline.build(CUDA_PASSES[len(TILE_PASSES) :]).run(replay.graph.copy(), ctx=ctx, db=None)
    except Exception:  # noqa: BLE001 — a compile the row cannot steer is not the kernel it measured
        return None
    return tuple(sorted(node.op.kernel_source for node in graph.nodes.values() if isinstance(node.op, CudaOp)))
