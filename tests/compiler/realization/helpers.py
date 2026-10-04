"""Loading, regeneration and the realization oracles for the corpus cases.

A case file is a golden with one traced program and one target kernel; its rows are the authored schedules the
compiler is expected to realize — one per kernel of the set the target compiles to — and its routing rows the
kernel-set decisions that mint the pieces. ``realized``, ``built`` and ``correct`` ask the whole set of the compile
the way a deploy would: the case is the compile's only evidence, strict, and no hand pin rides beside it
(:func:`evidence_scope`). Everything here is GPU-free except :func:`built` and :func:`correct`.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from emmy.compiler.context import Context
from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_of, validate_family_value
from emmy.compiler.pipeline.search.db import RoutingRow
from emmy.compiler.pipeline.search.golden import GoldenFile, Kernel, Latency, Measurements, Row, definition, restamp, sole_evidence
from emmy.compiler.pipeline.search.inventory import KernelInventory
from emmy.compiler.pipeline.search.pins import parse_reduce, pinned_knobs, unreproducible_pin_flag

CASES_DIR = Path(__file__).parent / "cases"

#: The three assertions a case walks, in order. A case's filename may name one of them as the stage it is expected
#: to fail at; the walker stops there.
STAGES = ("realized", "built", "correct")

_XFAIL = re.compile(r"_xfail_(?P<stage>[a-z_]+)$")

#: Families whose pin is consumed structurally rather than stamped on a kernel, so "the pinned family is stamped"
#: cannot be asked of them. ``PLACE`` is consumed by a splice; a ``REDUCE`` cross-CTA split replaces the kernel.
_UNSTAMPABLE = ("PLACE", "REDUCE")

#: The stand-in measurement a case's rows carry as evidence: a case authors schedules rather than measuring them,
#: and a proposal is no evidence, so each row stands in as a measured row — with one case in scope the microseconds
#: only have to exist, not rank.
STAND_IN = Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="corpus")


class CaseError(Exception):
    """A case file is not a usable corpus case — a hard error, never a skip."""


@dataclass(frozen=True)
class Case:
    """One corpus case: its file, its document and its expectation."""

    path: Path
    document: GoldenFile
    #: The stage this case is expected to fail at, or ``None`` when every stage must pass.
    xfail_stage: str | None

    @property
    def target(self) -> Kernel:
        """The one kernel the case's traced program lowers to — what every stage compiles."""
        [target] = self.document.targets()
        return target

    @property
    def rows(self) -> list[Row]:
        return self.document.rows

    @property
    def row(self) -> Row:
        """The row the perf lane benches by name — the first of the file. A cut target never runs, so a case whose
        target is cut has no row of its own: any row of the set names the compile."""
        return self.rows[0]

    @property
    def regime(self) -> dict:
        """The input regime the case was authored under — its pins less any kernel decision."""
        return {str(name): value for name, value in self.row.pins.items() if family_of(str(name)) not in KERNEL_DECISION_FAMILIES}

    @property
    def id(self) -> str:
        """The pytest parameter id — the case's path relative to ``cases/``, which is its identity."""
        return self.path.relative_to(CASES_DIR).as_posix()

    @property
    def compute_cap(self) -> tuple[int, int]:
        return tuple(self.document.compute_cap)

    def context(self) -> Context:
        """The case's own context — its declared capability, never the live card's. This is what makes the GPU-free
        stages machine-independent, so an sm_70 lockout is exercised on any box."""
        return Context.from_target(self.compute_cap)

    def program(self):
        """The target kernel as a standalone program, its weights bound beside its twin's — what every stage starts from."""
        return self.document.executable(self.target, {})


def case_files() -> list[Path]:
    return sorted(CASES_DIR.rglob("*.json"))


def expectation(path: Path) -> str | None:
    """The stage named by the filename suffix, or ``None`` for a closed case. An ``_xfail``-shaped token that is not
    one of the stages is a hard error: the filename is semantic, so a typo would otherwise quietly strengthen the
    assertion into a closed case."""
    stem = path.stem
    if "_xfail" not in stem:
        return None
    match = _XFAIL.search(stem)
    if match is None or match["stage"] not in STAGES:
        raise CaseError(f"{path.name}: expected a _xfail_<stage> suffix naming one of {', '.join(STAGES)}")
    return match["stage"]


def load_case(path: Path) -> Case:
    """Load one case, enforcing the one-program / one-target invariant the harness relies on and the exact classic
    spelling of every authored knob."""
    try:
        document = GoldenFile.load(path)
    except ValueError as exc:
        raise CaseError(str(exc)) from exc
    if len(document.programs) != 1 or len(document.targets()) != 1 or not document.rows:
        raise CaseError(f"{path.name}: a case holds one traced program, one target kernel and at least one row")
    parents = {route.parent for route in document.routing}
    for row in document.rows:
        if row.knobs is None:
            raise CaseError(f"{path.name}: every row must carry a knobs mapping, empty only for a forkless kernel")
        if row.kernel in parents:
            raise CaseError(f"{path.name}: {row.name} schedules a kernel a decision cuts, which never runs")
        canonical_knobs(row.knobs)
    stage = expectation(path)
    if stage is not None and not evidence_line(document):
        raise CaseError(f"{path.name}: an open case must carry a note with an 'evidence:' paragraph naming why it should realize")
    return Case(path=path, document=document, xfail_stage=stage)


def pin_of(case: Case, row: Row) -> dict:
    """One row as the hand pin its bench publishes: its input pins, the decisions that mint its kernel, its row."""
    route = {str(key): str(value) for step in case.document.path_to(row.kernel) for key, value in step.arm.items()}
    return {**row.pins, **route, **(row.knobs or {})}


def evidence_line(document: GoldenFile) -> str | None:
    """The case's ``evidence:`` citation: the paragraph of its note that starts with it."""
    for paragraph in (document.note or "").split("\n\n"):
        if paragraph.lower().startswith("evidence:"):
            return paragraph
    return None


def canonical_knobs(knobs: dict) -> dict:
    """The authored knobs after exact classic codec validation."""
    canonical = {}
    for name, value in knobs.items():
        try:
            canonical[name] = validate_family_value(name, value)
        except ValueError as exc:
            raise CaseError(f"knob {name}={value!r} is not a spelling this compiler's codec accepts: {exc}") from exc
    return canonical


# --- the derived half -------------------------------------------------------------------------
#
# A case's kernels are DERIVED from its program by the compiler in front of you — the restamp every golden gets —
# while the authored rows, names and note are not, and regeneration structurally cannot produce them. Recomputing
# the kernels and comparing is what keeps a stored case from rotting into a phantom lockout when a kernel identity
# or a body spelling changes.


def regenerate(document: GoldenFile) -> GoldenFile:
    """The case document as the current compiler would derive it: its kernels restamped, its authored rows and note
    preserved (``golden.restamp``, the same rewrite every repository golden gets)."""
    fresh, _report = restamp(document)
    return replace(fresh, rows=[replace(row, knobs=canonical_knobs(row.knobs or {})) for row in fresh.rows])


def complete(document: GoldenFile) -> GoldenFile:
    """The document with a row for every kernel of each target's set and a routing row for every decision that
    mints one.

    Each target is compiled with the document as its evidence; a kernel no row schedules gets a row of its own — that
    kernel, the input regime, and the schedule row the compile realized on it (a traced-only row of it takes that
    row); each kernel-set decision the compile took is recorded as the splice watcher reports it. A row naming a
    kernel the set does not run — one a decision cuts, or one the compiler no longer mints — is dropped, and the
    kernel standing in its place gets a fresh row. Strict evidence then has a row at every fork, each kernel's own.
    Authoring, not derivation."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import evidence_scope  # noqa: PLC0415
    from emmy.compiler.wire import kernel_bindings, kernel_tile  # noqa: PLC0415

    ctx = Context.from_target(tuple(document.compute_cap))
    ran: set[str] = set()
    for target in document.targets():
        seed = next((row for row in document.rows if row.kernel == target.exact_identity), None)
        seed = seed or next(
            row
            for row in document.rows
            if (document.path_to(row.kernel) or [None])[0] and document.path_to(row.kernel)[0].parent == target.exact_identity
        )
        regime = {str(name): value for name, value in seed.pins.items() if family_of(str(name)) not in KERNEL_DECISION_FAMILIES}
        taken: list = []
        watcher = KernelInventory(on_routing=lambda parent, arm, pieces, _ids, taken=taken: taken.append((parent, arm, pieces)))
        with evidence_scope([_measured(document)]), pinned_knobs(regime):
            graph = Pipeline.build(CUDA_PASSES).with_strategies(watcher).run(target.program({}), ctx=ctx, db=None)
        for parent, arm, pieces in taken:
            stored = document.add_kernel(definition(parent, parent.name))
            children = [document.add_kernel(definition(piece, piece.name)) for piece in pieces]
            arm = {str(k): str(v) for k, v in arm.items()}
            document.add_routing(RoutingRow(stored.exact_identity, arm, tuple(c.exact_identity for c in children)))
        for node in graph.nodes.values():
            if not isinstance(node.op, CudaOp) or (tile := kernel_tile(node.op)) is None:
                continue
            stored = document.add_kernel(definition(tile, node.op.kernel_name))
            ran.add(stored.exact_identity)
            realized = dict(schedule_row_key(dict(node.op.knobs or {})))
            existing = [index for index, row in enumerate(document.rows) if row.kernel == stored.exact_identity and row.pins == regime]
            if any(document.rows[index].knobs is not None for index in existing):
                continue
            if existing:
                document.rows[existing[0]] = replace(document.rows[existing[0]], knobs=realized)
                continue
            name = seed.name if seed.kernel == stored.exact_identity else f"{seed.name}.{stored.exact_identity[:12]}"
            document.rows.append(
                Row(name=name, kernel=stored.exact_identity, bindings=kernel_bindings(tile), pins=dict(regime), knobs=realized)
            )
    document.rows[:] = [row for row in document.rows if row.kernel in ran]
    return document


# --- the oracles -------------------------------------------------------------------------------


def _measured(document: GoldenFile) -> GoldenFile:
    """The case with every unmeasured row standing in as a measured one (:data:`STAND_IN`)."""
    rows = [replace(row, measurements=STAND_IN) if row.measurements is None and row.knobs is not None else row for row in document.rows]
    return replace(document, rows=rows)


@contextmanager
def evidence_scope(case: Case):
    """The case as a compile's ONLY evidence — how its schedule reaches a deploy.

    Its file is the whole golden scope, strictly (``golden.sole_evidence``: the machine-local tune DB and the prior are
    out of the way), and its input pins — the regime it was authored under, never a route or a schedule row — are
    the environment. No hand pin rides beside it: the route and the row reach the compile as rows of the kernels
    they decide, through the same evidence pick every ``compile`` / ``run`` / ``serve`` uses, or they do not reach
    it at all. Strict: a fork no row decides is an ``EvidenceError`` naming the kernel, never a prior's guess."""
    with sole_evidence([_measured(case.document)]), pinned_knobs(case.regime):
        yield


def lowered(case: Case, ctx: Context):
    """The case's target lowered through ``CUDA_PASSES`` at ``ctx`` with the case as its only evidence. Returns
    ``(graph, kernel-set decisions taken)`` — each decision as the arm knobs its splice carried; the tune DB is not
    consulted."""
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415

    taken: list[dict[str, str]] = []
    watcher = KernelInventory(on_routing=lambda _parent, arm, _pieces, _ids: taken.append({str(k): str(v) for k, v in arm.items()}))
    with evidence_scope(case):
        graph = Pipeline.build(CUDA_PASSES).with_strategies(watcher).run(case.program(), ctx=ctx, db=None)
    return graph, taken


def realized(case: Case) -> str | None:
    """Stage 1 — with the case as its only evidence, does the compile realize its schedule?

    The deploy contract, asked of the case: no hand pin, the file as the whole golden scope (:func:`evidence_scope`),
    and the same lowering ``compile`` runs. Four questions, because each catches a different way a row is lost: the
    lowering itself may refuse — strict evidence names a kernel whose row the enumeration no longer offers, the
    lockout a case exists to catch; a kernel may realize a *different* value; the authored family may reach no
    kernel at all, which ``unreproducible_pin_flag`` deliberately treats as ungateable; and a kernel-set decision the
    case records may not have been taken, which no stamp on the resulting kernels can show."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    try:
        graph, taken = lowered(case, case.context())
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        return f"{type(exc).__name__}: {exc}"
    rows = [dict(node.op.knobs or {}) for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    if not rows:
        return "lowering produced no CUDA kernel"
    for row in case.rows:
        flag = unreproducible_pin_flag(pin_of(case, row), rows)
        if flag is not None:
            return f"{row.name}: {flag}"
        unstamped = _unstamped_families(row.knobs or {}, rows)
        if unstamped:
            return f"{row.name}: authored but unstamped: {', '.join(sorted(unstamped))}"
    untaken = _untaken_decisions(case, taken)
    if untaken:
        return f"kernel-set decision not taken: {', '.join(untaken)}"
    return None


def _untaken_decisions(case: Case, taken: list[dict[str, str]]) -> list[str]:
    """The kernel-set decisions the case records that no splice of the compile carried: each ``PLACE`` key marked
    ``cut`` (a bare one is any cut), and each ``REDUCE`` value with a cross-CTA ``g<n>`` half (matched on that half
    alone — the rest is a piece's own schedule)."""
    missing: list[str] = []
    for route in case.document.routing:
        for key, value in route.arm.items():
            if family_of(key) == "PLACE" and value == "cut":
                if not any(v == "cut" and (k == key or key == "PLACE") for arm in taken for k, v in arm.items() if family_of(k) == "PLACE"):
                    missing.append(f"{key}=cut")
            elif family_of(key) == "REDUCE" and (want := parse_reduce(value)) is not None and want.needs_split:
                got = (parse_reduce(v) for arm in taken for k, v in arm.items() if family_of(k) == "REDUCE")
                if not any(plan is not None and plan.cta == want.cta and plan.finalize == want.finalize for plan in got):
                    missing.append(f"{key}={value}")
    return missing


def _unstamped_families(knobs: dict, rows: list[dict]) -> set[str]:
    """The authored schedule families no kernel carries a key for. Only the authored ``knobs`` are asked, never
    ``pins``: an input pin like ``FAST_MATH`` gates which forks are offered and is never a stamped kernel property."""
    stamped = {family_of(key) for row in rows for key in row}
    wanted = {family_of(name) for name in knobs} - set(_UNSTAMPABLE)
    return wanted - stamped


def built(case: Case):
    """Stage 2 — nvcc accepts the kernel the evidence picks. Returns the compiled graph, raising on refusal."""
    from emmy.compiler.backend.cuda.program import CompiledProgram  # noqa: PLC0415
    from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    graph, _taken = lowered(case, Context.probe())
    with gpu_lock():
        CompiledProgram.build(graph, seeded_inputs(case.program()))
    return graph


def correct(case: Case, compiled) -> None:
    """Stage 3 — the kernel the evidence picks computes the reference answer: the kernel's traced ops run on the
    numpy backend (``GoldenFile.reference_program``); a kernel with no exact frontend twin compares against the
    same-input greedy execution of the same program."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415

    program = case.program()
    sources = {}
    feed = seeded_inputs(program, sources=sources)
    result, _ = CudaBackend().run(compiled, input_data=dict(feed))
    if (twin := case.document.reference_program(case.target)) is not None:
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
    """The two sides of one output as the VALUES they stand for. A packed-pair output (two e2m1 codes to the byte)
    is decoded first: its bytes are not values, and the same value has two spellings at zero."""
    from emmy.compiler.dtype import decode_f4x2  # noqa: PLC0415

    tensor = program.buffer(name)
    if tensor is not None and tensor.dtype.logical_elems == 2 and got.dtype == np.uint8 and reference.dtype == np.uint8:
        return decode_f4x2(got), decode_f4x2(reference)
    return got, reference


def seeded_inputs(program, *, sources: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Deterministic inputs for the target's declared shapes, scaled so an fp16 reduction of a model-sized K does
    not saturate. A symbolic axis resolves to its own ``Dim`` hint — the size ``emmy run`` already resolves a
    symbolic reproducer to. An input is a BUFFER name, not a node id."""
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
        # A constant the runtime derives from a symbolic extent (a dynamic mean's count) is the backend's to fill.
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
    """Whether any buffer in the target is narrow enough to set the drift bound — read across the whole program, not
    off the output: an f16-by-f16 contraction that STORES f32 still carries a full f16 ulp of operand rounding."""
    for node in program.nodes.values():
        tensor = getattr(node, "output", None)
        if tensor is not None and tensor.dtype.name in _NARROW_DTYPES:
            return True
    return False


def _tolerance(narrow: bool, reference: np.ndarray) -> dict[str, float]:
    """Comparison bounds, scaled by the reference's own peak when an operand is narrow: the drift bound is roughly K
    times the peak times the operand epsilon, so a constant either fails the long case or stops asserting anything
    about the short one."""
    if not narrow:
        return {"rtol": 1e-4, "atol": 1e-5}
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    return {"rtol": 0.05, "atol": max(5e-3, 0.05 * peak)}


# --- latency ------------------------------------------------------------------------------------
#
# The stages above answer "can this schedule be realized and is it correct". This answers "and is it still as fast
# as it was" — the failure a lockout leaves behind when the compiler quietly stops selecting a schedule and falls
# back to a slower tier. It cannot ride the correctness walker: `make test` compiles at `-Xcicc -O1`, which is not a
# measurement lane.

#: Where a stored latency stops being a match. MEASURED, not guessed: ten estimates per case over four repeats on an
#: idle RTX 5090, spanning 1.5 us to 579 us, put the best-of-three estimator's own run-to-run spread at a median of
#: 0.17% and a maximum of 0.74%. Five percent is roughly seven times that worst case.
LATENCY_BAND = 0.05

#: Interference is one-sided — a busy machine makes a kernel slower, never faster — so the minimum of several runs
#: is the honest estimator. Taken lazily: a run inside the band ends the case.
LATENCY_REPEATS = 3


def live_hardware_id() -> str:
    return Context.probe().hardware_id()


def recorded_latency(case: Case, hardware_id: str) -> Latency | None:
    return (case.row.latency or {}).get(hardware_id)


def bench_command(case: Case, output: Path) -> list[str]:
    """The one way to bench a case: `emmy run`, replaying the row the case authors for its target. Naming the row
    makes the case's file the compile's whole golden scope, and the row benches as a pinned row beside the greedy
    pick because it was asked for by name."""
    import sys  # noqa: PLC0415

    return [
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
        "--golden",
        str(case.path),
        "--realization",
        case.row.name,
        "--bench",
        "--bench-backends",
        "eager,tcompile,emmy",
        "--json",
        str(output),
    ]


def measure(case: Case, *, within: float | None = None) -> tuple[list[float], float]:
    """Best-of-N emmy microseconds for the case's own schedule, and the torch.compile number. Stops early once a
    sample lands at or below ``within``. Deployable optimization is forced: the correctness lane's `-O1` changes
    runtime performance."""
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
    """What a case is, for a report: its operations, its input shapes and its dtype — derived from the stored
    program rather than carried as metadata."""
    target = case.target
    program = case.document.programs[target.traced]
    by_id = {node["id"]: node for node in program["nodes"]}
    ops = [by_id[origin]["op"] for origin in target.origins if origin in by_id]
    inputs = [by_id[name]["outputs"][0] for name in program["inputs"] if name in by_id]
    shapes = " x ".join("(" + ",".join(str(dim) for dim in spec[2]) + ")" for spec in inputs)
    dtypes = {spec[1] for spec in inputs} or {"?"}
    return {
        "op": "+".join(dict.fromkeys(op.rsplit(".", 1)[-1] for op in ops)) or "loop",
        "shape": shapes,
        "dtype": "/".join(sorted(dtypes)),
        "family": case.path.parent.name,
    }
