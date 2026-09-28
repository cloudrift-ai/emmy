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
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_of
from emmy.compiler.pipeline.search.golden import (
    Config,
    GoldenEntryState,
    GoldenFile,
    GoldenRecord,
    Measurements,
    Realization,
    decode_record,
    siblings_of,
    sole_evidence,
)
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.working_golden import lowered_kernels
from emmy.compiler.structural import digest


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
    write, ``name: reason``."""
    entries = [entry for entry in document.configs if entry.program == program]
    return [reason for reason in (_stale_reason(document, entry, fresh) for entry in entries) if reason is not None]


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

    @property
    def changed(self) -> bool:
        return bool(self.restamped or self.dropped_targets)

    def lines(self) -> list[str]:
        out = [f"{self.restamped} of {self.targets} targets restamped, {len(self.dropped_targets)} dropped; {self.rows_kept} rows kept"]
        out.extend(f"dropped target {reason}" for reason in self.dropped_targets)
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
        if wire == stored:
            report.rows_kept += len(entry.realizations)
            configs.append(replace(entry, target=replace(entry.target, loop=intern(stored))))
            continue
        rows = _rekeyed_rows(document, entry, wire, report)
        if not rows:
            report.dropped_targets.append(f"{_entry_name(entry)}: no row survives on the fresh kernel")
            continue
        report.restamped += 1
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
    old_records = [document.record(entry, row) for row in entry.realizations]
    new_records = [scratch.record(fresh_entry, row) for row in entry.realizations]

    # A row naming the target itself takes the fresh target's identity. Any other stored identity
    # names a piece of the target's kernel set (a receipt, or a piece row beside its routing row):
    # it stays as stored, and the decode below keeps the row only if the fresh set still mints that
    # piece under the set's own rows.
    old_key = replace(old_records[0], identity=None).kernel_identity
    new_key = replace(new_records[0], identity=None).kernel_identity
    survivors = [
        replace(new, identity=new_key) if old.identity == old_key else new for old, new in zip(old_records, new_records, strict=True)
    ]

    rows = []
    for realization, old, new in zip(entry.realizations, old_records, survivors, strict=True):
        reason = decode_record(new, siblings_of(new, survivors))
        if reason is not None:
            report.rows_dropped.append(f"{old.name}: {reason}")
            continue
        row = replace(realization, identity=new.identity if new.identity is not None else realization.identity)
        if row.measurements is not None or row.latency is not None:
            if _kernel_sources(old, old_records) != _kernel_sources(new, survivors):
                row = replace(row, measurements=None, latency=None)
                report.rows_demoted.append(old.name)
            else:
                report.rows_kept += 1
        else:
            report.rows_kept += 1
        rows.append(row)
    # A kernel-set row carries no schedule of its own: once its members lose their measurements it
    # spells nothing, and a repository golden refuses a row that spells nothing.
    for row in [row for row in rows if row.kernel_set_state(rows) is GoldenEntryState.INVENTORY]:
        rows.remove(row)
        report.rows_dropped.append(f"{row.name}: its kernel set lost its measurements")
    return rows


def _kernel_sources(record: GoldenRecord, records: Sequence[GoldenRecord]) -> tuple[str, ...] | None:
    """The CUDA sources the record's target renders with the record as the compile's only evidence —
    beside the rows of its set that decide other kernels (its route, the other pieces), never an
    alternate schedule of its own kernel. ``None`` when the compile refuses, which a caller reads
    as "not the same kernel"."""

    siblings = [other for other in siblings_of(record, records) if other.is_routing or other.identity != record.identity]
    stand_in = Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="restamp")
    evidence = [entry if entry.measurements is not None else replace(entry, measurements=stand_in) for entry in (record, *siblings)]
    regime = {key: value for key, value in record.pin_map.items() if family_of(str(key)) not in KERNEL_DECISION_FAMILIES}
    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    try:
        with sole_evidence(evidence), pinned_knobs(regime):
            graph = Pipeline.build(CUDA_PASSES).run(record.target_program.copy(), ctx=ctx, db=None)
    except Exception:  # noqa: BLE001 — a compile the row cannot steer is not the kernel it measured
        return None
    return tuple(sorted(node.op.kernel_source for node in graph.nodes.values() if isinstance(node.op, CudaOp)))
