"""Loading, regeneration and the four realization oracles for the corpus cases.

A case file is a working golden document carrying exactly one config whose realizations are the
authored ``pins`` / ``knobs`` the compiler is expected to realize: one entry per kernel of the set
the target compiles to, each addressed by the ``identity`` of the kernel it decides (the first
entry is the target's own). ``offered`` strictly decodes each entry, as a golden row is decoded; ``realized``,
``built`` and ``correct`` ask the whole set of the compile the way a deploy would — the case's
entries are the compile's only evidence, strict, and no hand pin rides beside them
(:func:`evidence_scope`). Everything here is GPU-free except :func:`built` and :func:`correct`.
"""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from emmy.compiler.context import Context
from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_of, validate_family_value
from emmy.compiler.pipeline.search.golden import (
    Config,
    GoldenFile,
    GoldenRecord,
    Latency,
    Measurements,
    Realization,
    decode_record,
    kernel_identity,
    siblings_of,
    sole_evidence,
)
from emmy.compiler.pipeline.search.pins import parse_reduce, pinned_knobs, unreproducible_pin_flag
from emmy.compiler.pipeline.strategy import PipelineStrategy

CASES_DIR = Path(__file__).parent / "cases"

#: The four assertions a case walks, in order. A case's filename may name one of them as the
#: stage it is expected to fail at; the walker stops there.
STAGES = ("offered", "realized", "built", "correct")

_XFAIL = re.compile(r"_xfail_(?P<stage>[a-z_]+)$")

#: Families whose pin is consumed structurally rather than stamped on a kernel, so "the pinned
#: family is stamped" cannot be asked of them. ``PLACE`` is consumed by a splice; a ``REDUCE``
#: cross-CTA split replaces the kernel outright. :func:`unreproducible_pin_flag` already reads
#: both the same way — this is the same rule for the complementary stamping check.
_UNSTAMPABLE = ("PLACE", "REDUCE")


class CaseError(Exception):
    """A case file is not a usable corpus case — a hard error, never a skip."""


@dataclass(frozen=True)
class Case:
    """One corpus case: its file, its document, its records (the target's own entry first, then
    one per further kernel of the set), and its expectation."""

    path: Path
    document: GoldenFile
    records: tuple[GoldenRecord, ...]
    #: The stage this case is expected to fail at, or ``None`` when every stage must pass.
    xfail_stage: str | None

    @property
    def record(self) -> GoldenRecord:
        """The target's own entry — the one the perf lane benches by name."""
        return self.records[0]

    @property
    def id(self) -> str:
        """The pytest parameter id — the case's path relative to ``cases/``, which is its identity."""
        return self.path.relative_to(CASES_DIR).as_posix()

    @property
    def compute_cap(self) -> tuple[int, int]:
        return tuple(self.document.compute_cap)

    def context(self) -> Context:
        """The case's own context — its declared capability, never the live card's. This is what
        makes stages 1 and 2 machine-independent, so an sm_70 lockout is exercised on any box."""
        return Context.from_target(self.compute_cap)


def case_files() -> list[Path]:
    return sorted(CASES_DIR.rglob("*.yaml"))


def expectation(path: Path) -> str | None:
    """The stage named by the filename suffix, or ``None`` for a closed case.

    An ``_xfail``-shaped token that is not one of the four stages is a hard error: the filename is
    semantic, so a typo would otherwise quietly strengthen the assertion into a closed case.
    """
    stem = path.stem
    if "_xfail" not in stem:
        return None
    match = _XFAIL.search(stem)
    if match is None or match["stage"] not in STAGES:
        raise CaseError(f"{path.name}: expected a _xfail_<stage> suffix naming one of {', '.join(STAGES)}")
    return match["stage"]


def load_case(path: Path) -> Case:
    """Load one case, enforcing the one-config / one-realization invariant the harness relies on."""
    try:
        document = GoldenFile.load(path)
    except ValueError as exc:
        raise CaseError(str(exc)) from exc
    configs = document.configs
    if len(configs) != 1 or not configs[0].realizations:
        raise CaseError(f"{path.name}: a case holds exactly one config with at least one realization")
    for realization in configs[0].realizations:
        if realization.knobs is None:
            raise CaseError(f"{path.name}: every entry must carry a knobs mapping, empty only for a forkless kernel")
    stage = expectation(path)
    if stage is not None and not evidence_line(path):
        raise CaseError(f"{path.name}: an open case must carry a leading '# evidence:' comment naming why it should realize")
    records = tuple(document.record(configs[0], realization) for realization in configs[0].realizations)
    return Case(path=path, document=document, records=records, xfail_stage=stage)


def pin_of(record: GoldenRecord) -> dict:
    """One entry as the hand pin ``offered`` publishes: its input pins plus its authored row."""
    return {**record.pin_map, **record.knobs}


def evidence_line(path: Path) -> str | None:
    """The case's ``# evidence:`` citation, read out of its leading comment block."""
    for line in leading_comment(path).splitlines():
        body = line.lstrip("#").strip()
        if body.lower().startswith("evidence:"):
            return body
    return None


def leading_comment(path: Path) -> str:
    """The file's leading ``#`` block. ``dump_golden_file`` is a plain YAML dump and drops
    comments, so regeneration captures this and re-prepends it."""
    lines: list[str] = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.startswith("#"):
            break
        lines.append(line)
    return "".join(lines)


# --- the derived half -------------------------------------------------------------------------
#
# Program wire, target and the target's identity are *derived* from the stored program by the
# compiler in front of you; the authored pins, knobs and names are not, and regeneration
# structurally cannot produce them. Recomputing the first group and comparing is what keeps a
# stored case from rotting into a phantom lockout when a kernel identity or a schedule codec
# changes. A name is a label written once: the kernel's provenance name for the target's entry,
# that name plus the piece's identity prefix for a further entry. Nothing re-derives it, so no
# compiler change moves it and a pointer to a row (``--realization``) keeps landing.


def regenerate(document: GoldenFile) -> GoldenFile:
    """The case document as the current compiler would derive it, authored fields preserved.

    Runs the inventory writer through the library under an explicit ``Context.from_target`` — not
    through ``emmy trace``, which stamps ``gpu_name`` from the live card and needs torch. The
    result is machine-independent, so this check fires and its fix works on any box.
    """
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.pipeline.search.working_golden import write_trace_inventory  # noqa: PLC0415

    entry = document.configs[0]
    ctx = Context.from_target(tuple(document.compute_cap))
    graph = Graph.from_wire(document.programs[entry.program])
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "regenerated.yaml"
        write_trace_inventory(graph, destination, ctx=ctx, model=document.model)
        fresh = GoldenFile.load(destination)

    matched = _matching_entry(fresh, entry, document.loops[entry.target.loop])
    # A case keeps its own kernel only: the regenerated pool holds every kernel of the program.
    rebuilt = replace(fresh, loops=[fresh.loops[matched.target.loop]], configs=[matched])
    matched.target = replace(matched.target, loop=0)
    rows = []
    for index, realization in enumerate(entry.realizations):
        # Every entry keeps the name it was authored with; a further entry decides another kernel
        # of the set and keeps its identity too. A latency is measured on a card, never derived
        # from the program: a regeneration on a CPU box must not erase a 4090's recorded timings.
        row = Realization(
            name=realization.name,
            bindings=dict(realization.bindings),
            pins=dict(realization.pins),
            knobs=canonical_knobs(realization.knobs),
            latency=dict(realization.latency) if realization.latency is not None else None,
        )
        row.identity = realization.identity if index else kernel_identity(rebuilt.record(matched, row))
        rows.append(row)
    matched.realizations = rows
    return rebuilt


def complete(document: GoldenFile) -> GoldenFile:
    """The case with an entry for every kernel of its set, each named by identity.

    The set is replayed the way the deploy reads it (``golden._replay`` with the entries as one
    another's siblings); a scheduled kernel no entry names by identity gets an entry of its own:
    that kernel's identity, the input regime, and the schedule row the replay realized on it. A
    further entry naming a kernel the compiler no longer mints is dropped first: its row was
    authored for a kernel that no longer exists, and the kernel standing in its place gets a fresh
    entry. Strict evidence then has a row at every fork, each kernel's own — the rows the golden
    import files in the DB are per kernel, and nothing stands in for a kernel no entry names.
    Authoring, not derivation — the added rows are enumerable schedules of those kernels, and the
    case pins them from then on."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import lead_of, siblings_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.decode import _replay  # noqa: PLC0415

    entry = document.configs[0]
    records = [document.record(entry, realization) for realization in entry.realizations]
    primary = records[0]
    replay = _replay(primary, siblings=siblings_of(primary, records), lead=lead_of(primary, records))
    kernels = set(replay.kernels)
    # The target's entry names the kernel the set was cut from; a routing entry names the kernel its
    # decision replaced. Neither is a kernel the set ran as, and both stay.
    kept = [
        realization
        for record, realization in zip(records, entry.realizations, strict=True)
        if record is primary or record.is_routing or record.identity in kernels
    ]
    covered = {record.identity for record in records if record.identity in kernels}
    regime = {key: value for key, value in primary.pin_map.items() if family_of(str(key)) != "PLACE"}
    added = [
        Realization(
            name=f"{primary.name}.{identity[:12]}",
            bindings=dict(primary.bindings),
            pins=dict(regime),
            knobs=dict(replay.realized.get(identity, {})),
            identity=identity,
        )
        for identity in sorted(kernels - covered)
    ]
    if added or len(kept) != len(records):
        entry.realizations = [*kept, *added]
    return document


def _matching_entry(fresh: GoldenFile, entry: Config, kernel: dict) -> Config:
    """The regenerated config for the stored kernel: the one from the same traced ops, or, for a
    kernel that keeps none, the one with the same exact typed Loop identity."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415

    def identity(wire):
        graph = Graph.from_wire(wire)
        return tuple(
            node.op.with_io(graph, node).identity_key(structural=False, with_io=True)
            for node in graph.nodes.values()
            if isinstance(node.op, LoopOp)
        )

    origins = entry.target.origins
    key = identity(kernel) if not origins else None
    for candidate in fresh.configs:
        target = candidate.target
        if (target.origins == origins) if origins else identity(fresh.loops[target.loop]) == key:
            return replace(candidate)
    offered = ", ".join(repr(candidate.target.origins) for candidate in fresh.configs)
    raise CaseError(f"no kernel of the program matches the stored one (traced ops {origins!r}); the program now forms {offered}")


def canonical_knobs(knobs: dict) -> dict:
    """The authored knobs after exact classic codec validation."""
    canonical = {}
    for name, value in knobs.items():
        try:
            canonical[name] = validate_family_value(name, value)
        except ValueError as exc:
            raise CaseError(f"knob {name}={value!r} is not a spelling this compiler's codec accepts: {exc}") from exc
    return canonical


def write_case(path: Path, document: GoldenFile) -> None:
    """Persist a regenerated case, restoring the leading comment block the YAML dump drops."""
    comment = leading_comment(path)
    document.dump(path, overwrite=True)
    if comment:
        path.write_text(comment + path.read_text())


# --- the four oracles -------------------------------------------------------------------------


@contextmanager
def evidence_scope(case: Case):
    """The case as a compile's ONLY evidence — how its schedule reaches a deploy.

    Its entries are the whole golden scope, strictly (``golden.sole_evidence``: the machine-local
    online prior is out of the way), and its input pins — the regime it was measured under, never
    its route or its schedule row — are the environment. No hand pin rides beside it: the route and the schedule reach the compile as
    measured rows of the kernels they decide, through the same evidence pick every ``compile`` /
    ``run`` / ``serve`` uses, or they do not reach it at all. A case authors schedules rather than
    measuring them, and a proposal is no evidence, so each entry stands in as a measured row: with
    one case in scope the microseconds only have to exist, not rank. Strict: a fork no entry
    decides is an ``EvidenceError`` naming the kernel, never a prior's guess.
    """
    records = [
        replace(record, measurements=Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="corpus"))
        if record.measurements is None
        else record
        for record in case.records
    ]
    regime = {name: value for name, value in case.record.pin_map.items() if family_of(str(name)) not in KERNEL_DECISION_FAMILIES}
    with sole_evidence(records), pinned_knobs(regime):
        yield


class _Splices(PipelineStrategy):
    """Every kernel-set decision a compile took, as the arm knobs each splice carried."""

    def __init__(self) -> None:
        self.taken: list[dict[str, str]] = []

    def on_splice(self, e) -> None:
        self.taken.append({str(key): str(value) for key, value in e.knobs.items()})


def lowered(case: Case, ctx: Context):
    """The case's target lowered through ``CUDA_PASSES`` at ``ctx`` with the case as its only
    evidence. Returns ``(graph, kernel-set decisions taken)``; the tune DB is not consulted."""
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415

    splices = _Splices()
    with evidence_scope(case):
        graph = Pipeline.build(CUDA_PASSES).with_strategies(splices).run(case.record.target_program.copy(), ctx=ctx, db=None)
    return graph, splices.taken


def offered(case: Case) -> str | None:
    """Stage 1 — does the compiler still enumerate every entry's schedule?

    The golden decode is the one question a recorded row and a corpus entry both answer
    (:func:`~emmy.compiler.pipeline.search.golden.decode_record`): the entry's route resolves to
    offered seams, and its row equals an enumerated leaf of the kernel its ``identity`` names,
    decided beside the case's other entries exactly as a deploy reads the set.
    """
    for record in case.records:
        if (reason := decode_record(record, siblings_of(record, case.records))) is not None:
            return f"{record.name}: {reason}"
    return None


def realized(case: Case) -> str | None:
    """Stage 2 — with the case as its only evidence, does the compile realize its schedule?

    The deploy contract, asked of the case: no hand pin, the record as the whole golden scope
    (:func:`evidence_scope`), and the same lowering ``compile`` runs. Four questions, because each
    catches a different way the row is lost: the lowering itself may refuse; a kernel may realize
    a *different* value; the pinned family may reach no kernel at all, which
    ``unreproducible_pin_flag`` deliberately treats as ungateable; and a kernel-set decision the
    case spells — a placement cut, a cross-CTA split — may not have been taken, which no stamp
    on the resulting kernels can show.
    """
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    try:
        graph, taken = lowered(case, case.context())
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        return f"{type(exc).__name__}: {exc}"
    rows = [dict(node.op.knobs or {}) for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    if not rows:
        return "lowering produced no CUDA kernel"
    for record in case.records:
        flag = unreproducible_pin_flag(pin_of(record), rows)
        if flag is not None:
            return f"{record.name}: {flag}"
        unstamped = _unstamped_families(record.knobs, rows)
        if unstamped:
            return f"{record.name}: pinned but unstamped: {', '.join(sorted(unstamped))}"
    untaken = _untaken_decisions(case, taken)
    if untaken:
        return f"kernel-set decision not taken: {', '.join(untaken)}"
    return None


def _untaken_decisions(case: Case, taken: list[dict[str, str]]) -> list[str]:
    """The kernel-set decisions the case spells that no splice of the compile carried: each
    ``PLACE`` key marked ``cut`` (a bare one is any cut), and each ``REDUCE`` value with a
    cross-CTA ``g<n>`` half (matched on that half alone — the rest is a piece's own schedule)."""
    missing: list[str] = []
    for record in case.records:
        for key, value in record.route.items():
            if value != "cut":
                continue
            if not any(v == "cut" and (k == key or key == "PLACE") for arm in taken for k, v in arm.items() if family_of(k) == "PLACE"):
                missing.append(f"{key}=cut")
        for key, value in pin_of(record).items():
            want = parse_reduce(value) if family_of(str(key)) == "REDUCE" else None
            if want is None or not want.needs_split:
                continue
            got = (parse_reduce(v) for arm in taken for k, v in arm.items() if family_of(k) == "REDUCE")
            if not any(plan is not None and plan.cta == want.cta and plan.finalize == want.finalize for plan in got):
                missing.append(f"{key}={value}")
    return missing


def _unstamped_families(knobs: dict, rows: list[dict]) -> set[str]:
    """The authored schedule families no kernel carries a key for.

    Only the authored ``knobs`` are asked, never ``pins``: an input pin like ``FAST_MATH`` is an
    umbrella that gates which forks are offered and is never a stamped kernel property itself.
    """
    stamped = {family_of(key) for row in rows for key in row}
    wanted = {family_of(name) for name in knobs} - set(_UNSTAMPABLE)
    return wanted - stamped


def built(case: Case):
    """Stage 3 — nvcc accepts the kernel the evidence picks. Returns the compiled graph, raising
    on refusal."""
    from emmy.compiler.backend.cuda.program import CompiledProgram  # noqa: PLC0415
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    graph, _taken = lowered(case, Context.probe())
    with gpu_lock():
        CompiledProgram.build(graph, seeded_inputs(case.record.target_program))
    return graph


def correct(case: Case, compiled) -> None:
    """Stage 4 — the kernel the evidence picks computes the reference answer.

    The reference is the kernel's traced ops run on the numpy backend
    (:attr:`~emmy.compiler.pipeline.search.golden.GoldenRecord.reference_program`); a kernel with no
    exact frontend twin compares against the same-input greedy execution of the same program.
    """
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

    program = case.record.target_program
    sources = {}
    feed = seeded_inputs(program, sources=sources)
    result, _ = CudaBackend().run(compiled, input_data=dict(feed))
    if (twin := case.record.reference_program) is not None:
        reference = NumpyBackend()
        # The twin reads the kernel's inputs, plus any checkpoint-backed weight of its own.
        twin_feed = {**seeded_inputs(twin, sources=sources), **{name: feed[name] for name in twin.inputs}}
        want, _ = reference.run(reference.compile(twin.copy()), input_data=twin_feed)
    else:
        greedy = CudaBackend()
        want, _ = greedy.run(greedy.compile(program.copy()), input_data=dict(feed))
    narrow = _has_narrow_operand(program)
    for name in program.outputs:
        got, reference = _comparable(program, name, np.asarray(result.outputs[name]), np.asarray(want.outputs[name]))
        np.testing.assert_allclose(got, reference, err_msg=f"{case.id}: output {name}", **_tolerance(narrow, reference))


def _comparable(program, name: str, got: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two sides of one output as the VALUES they stand for. A packed-pair output (two e2m1
    codes to the byte) is decoded first: its bytes are not values, and the same value has two
    spellings at zero — the block-scaled cell rounds a tiny negative product to the negative zero
    code where the numpy reference lands on the positive one, and a byte comparison counts that
    as a mismatch of every element of an all-zero output."""
    from emmy.compiler.dtype import decode_f4x2  # noqa: PLC0415

    tensor = program.buffer(name)
    if tensor is not None and tensor.dtype.logical_elems == 2 and got.dtype == np.uint8 and reference.dtype == np.uint8:
        return decode_f4x2(got), decode_f4x2(reference)
    return got, reference


def seeded_inputs(program, *, sources: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Deterministic inputs for the target's declared shapes, scaled so an fp16 reduction of a
    model-sized K does not saturate.

    A symbolic axis resolves to its own ``Dim`` hint — the size ``emmy run`` already resolves a
    symbolic reproducer to. The corpus therefore exercises a symbolic kernel at its hint; it has no
    spelling for "compile at the hint, run at some other size", because binding a symbol in a case
    file SPECIALIZES the program rather than sizing a run of it.

    An input is a BUFFER name, not a node id: a Loop target's slice boundary mirrors every buffer of
    the producer it stands in for, and a multi-buffer producer (an NVFP4 encode, which emits packed
    codes beside their block scales) names its second buffer after the tensor rather than the node.
    """
    from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim  # noqa: PLC0415
    from emmy.compiler.ir.base import ConstantOp  # noqa: PLC0415
    from emmy.compiler.loader.binder import bind_constants  # noqa: PLC0415

    rng = np.random.default_rng(0)
    sources = {} if sources is None else sources

    def seeded(dims) -> np.ndarray:  # noqa: ANN001
        dims = tuple(Dim(dim) for dim in dims)
        shape = tuple(dim.as_static() if dim.is_static else (dim.hint or DEFAULT_SEQ_HINT) for dim in dims)
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    feed: dict[str, np.ndarray] = {name: seeded(program.buffer(name).shape) for name in program.inputs}
    for node_id, node in program.nodes.items():
        # A constant the runtime derives from a symbolic extent (a dynamic mean's count) is the
        # backend's to fill; seeding it would divide by noise.
        if not isinstance(node.op, ConstantOp) or node_id in feed or node.op.context_value is not None:
            continue
        op = node.op
        parts = op.source_parts or (((op.source_path, op.source_shape or node.output.shape),) if op.source_path else ())
        for path, shape in parts:
            if path not in sources:
                sources[path] = seeded(shape)
        if not parts:
            feed[node_id] = np.array([op.value], dtype=np.float32) if op.value is not None else seeded(node.output.shape)
    # Both graphs bind the same source weights through their own transpose / reshape chains.
    feed.update(bind_constants(program, sources))
    return feed


#: Dtypes narrow enough that rounding the OPERANDS dominates the comparison.
_NARROW_DTYPES = ("f16", "bf16", "e4m3", "e5m2")


def _has_narrow_operand(program) -> bool:
    """Whether any buffer in the target is narrow enough to set the drift bound.

    Read across the whole program, not off the output. An f16-by-f16 contraction that STORES f32
    still carries a full f16 ulp of operand rounding, and a bound picked from the store alone sits
    below it — no amount of correct codegen can pass that.
    """
    for node in program.nodes.values():
        tensor = getattr(node, "output", None)
        if tensor is not None and tensor.dtype.name in _NARROW_DTYPES:
            return True
    return False


def _tolerance(narrow: bool, reference: np.ndarray) -> dict[str, float]:
    """Comparison bounds, scaled by the reference's own peak when an operand is narrow.

    A fixed absolute bound cannot serve both a 128-long f16 reduction and a 4096-long one: the
    drift bound is roughly K times the peak times the operand epsilon, so a constant either fails
    the long case or stops asserting anything about the short one. This is the same peak-relative
    form the e2e coverage matrices this corpus replaces already use.
    """
    if not narrow:
        return {"rtol": 1e-4, "atol": 1e-5}
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    return {"rtol": 0.05, "atol": max(5e-3, 0.05 * peak)}


# --- stage 5: latency ---------------------------------------------------------------------------
#
# Stages 1 to 4 answer "can this schedule be realized and is it correct". This answers "and is it
# still as fast as it was" — the failure a lockout leaves behind when the compiler quietly stops
# selecting a schedule and falls back to a slower tier.
#
# It cannot ride the correctness walker. `make test` compiles at `-Xcicc -O1`, which by the
# glossary's own definition is not a measurement lane, so a latency assertion there would measure
# the wrong regime entirely.

#: Where a stored latency stops being a match. MEASURED, not guessed: ten estimates per case over
#: four repeats on an idle RTX 5090, spanning 1.5 us to 579 us, put the best-of-three estimator's
#: own run-to-run spread at a median of 0.17% and a maximum of 0.74%. Five percent is roughly seven
#: times that worst case — enough headroom for a card that throttles under a sustained sweep, and
#: still tight enough to see a five-percent regression rather than only a cliff.
#:
#: The ~7% gap `run --json` documents between two timing semantics does not apply here: a stored
#: latency and a fresh one are the same measurement of the same pinned row, compared like with
#: like. And a case outside the band only REPORTS, so a false positive costs a line of output
#: rather than a red build — which is itself an argument for the tighter bound.
LATENCY_BAND = 0.05

#: Interference is one-sided — a busy machine makes a kernel slower, never faster than the hardware
#: can go — so the minimum of several runs is the honest estimator, and requiring every run to be
#: slow is what keeps a developer box that is also compiling something from crying wolf.
#:
#: Taken lazily: a run inside the band ends the case, because the minimum can only fall and no
#: later run could lift it back out. Only a case that looks slow pays for the extra runs, which is
#: exactly the case where interference is the question. On an idle card, where the measured spread
#: is under 1%, that makes the lane three times cheaper at identical strength.
LATENCY_REPEATS = 3


def live_hardware_id() -> str:
    return Context.probe().hardware_id()


def recorded_latency(case: Case, hardware_id: str) -> Latency | None:
    return (case.record.latency or {}).get(hardware_id)


def bench_command(case: Case, output: Path) -> list[str]:
    """The one way to bench a case: `emmy run`, replaying the realization the case authors.

    Naming the realization makes the case's record the deploy tier's whole scope for that compile,
    measurement state notwithstanding — a corpus case is deliberately not `VERIFIED`, so it never
    enters the replay tooling's trusted evidence, and it still benches as a pinned row beside the
    greedy pick because it was asked for by name.
    """
    import sys  # noqa: PLC0415

    return [
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
        "--golden",
        str(case.path),
        "--realization",
        case.record.name,
        "--bench",
        "--bench-backends",
        "eager,tcompile,emmy",
        "--json",
        str(output),
    ]


def measure(case: Case, *, within: float | None = None) -> tuple[list[float], float]:
    """Best-of-N emmy microseconds for the case's own schedule, and the torch.compile number.

    Stops early once a sample lands at or below ``within``: the minimum can only fall, so a run
    already inside the band settles the case and the remaining runs would change nothing.

    Deployable optimization is forced: the correctness lane's `-O1` changes runtime performance,
    and a timing measured there is not one a deploy would ever see.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    environment = {**os.environ, "EMMY_NVCC_FLAGS": ""}
    samples: list[float] = []
    tcompile = 0.0
    with tempfile.TemporaryDirectory(prefix="emmy_corpus_bench_") as directory:
        for repeat in range(LATENCY_REPEATS):
            output = Path(directory) / f"{repeat}.json"
            result = subprocess.run(bench_command(case, output), capture_output=True, text=True, env=environment, timeout=1800)
            if result.returncode != 0 or not output.exists():
                raise AssertionError(f"{case.id}: bench failed (exit {result.returncode})\n{result.stderr[-2000:]}")
            record = json.loads(output.read_text())
            rows = [row for row in record.get("pinned", []) if row.get("status") == "ok" and row.get("total_us")]
            if not rows:
                raise AssertionError(f"{case.id}: the pinned row measured nothing — {record.get('pinned')}")
            samples.append(float(rows[0]["total_us"]))
            tcompile = float(record["backends"].get("torch.compile", {}).get("latency_us") or tcompile)
            if within is not None and samples[-1] <= within:
                break
    return samples, tcompile


def describe(case: Case) -> dict[str, str]:
    """What a case is, for a report: its operations, its input shapes and its dtype.

    Derived from the stored program rather than carried as metadata. A curated case list has to
    keep these in sync by hand — which is how the list it replaces came to claim a dtype the
    compiler had outgrown.
    """
    program = case.record.program_wire
    by_id = {node["id"]: node for node in program["nodes"]}
    ops = [by_id[origin]["op"] for origin in case.record.origins if origin in by_id]
    inputs = [by_id[name]["outputs"][0] for name in program["inputs"] if name in by_id]
    shapes = " x ".join("(" + ",".join(str(dim) for dim in spec[2]) + ")" for spec in inputs)
    dtypes = {spec[1] for spec in inputs} or {"?"}
    return {
        "op": "+".join(dict.fromkeys(op.rsplit(".", 1)[-1] for op in ops)) or "loop",
        "shape": shapes,
        "dtype": "/".join(sorted(dtypes)),
        "family": case.path.parent.name,
    }
