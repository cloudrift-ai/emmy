"""Golden records become tune DB rows before a compile picks (``golden.evidence``).

Every kind of entry a golden file holds lands where the deploy reads it: a plain entry as its one
kernel's perf row, a receipt as the row of the kernel its identity names, a cut as routing rows with
the pieces' receipts pricing it, a split timed as a whole as routing rows only. The realization corpus
stands in for a card's file: its cases are authored without a GPU, so nothing here needs one. The
compile's seam imports a scope once per golden digest and lets a re-recorded file's rows go."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.pipeline.knob import KERNEL_DECISION_FAMILIES, family_of
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements, records_override
from emmy.compiler.pipeline.search.golden.evidence import evidence_db, import_goldens
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.realization import helpers as corpus

_MEASURED = Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="corpus")


def _records(case):
    """The case's entries as measured rows — the corpus authors schedules, so a stand-in µs makes them evidence."""
    return [replace(record, measurements=_MEASURED) if record.measurements is None else record for record in case.records]


def _regime(record) -> dict:
    return {str(name): value for name, value in record.pin_map.items() if family_of(str(name)) not in KERNEL_DECISION_FAMILIES}


def _imported(case_path: str, records=None):
    case = corpus.load_case(corpus.CASES_DIR / case_path)
    records = _records(case) if records is None else records
    db, ctx = SearchDB(), case.context()
    with pinned_knobs(_regime(case.record)):
        counts = import_goldens(db, ctx, records, source="golden:test")
    return db, ctx, records, counts


def _schedule(row) -> dict:
    return {key: str(value) for key, value in row.knobs.items() if not key.startswith(("S_", "I_"))}


def test_a_plain_entry_is_its_kernels_row() -> None:
    """One target, one kernel, one entry: the row is the entry's schedule row as recorded, keyed on the
    kernel the target lowers to, captured, under the golden's source."""
    db, ctx, [record], counts = _imported("fused/norm-linear-f16-scalar-reduce.yaml")

    assert counts == {"perf rows": 1}
    [row] = db.iter_perf_rows()
    assert _schedule(row) == {key: str(value) for key, value in record.schedule_row.items()}
    assert (row.stats.median, row.captured, row.source, row.gpu) == (1.0, True, "golden:test", ctx.hardware_id())
    assert db.perf_sources(ctx) == {"golden:test": 1}
    assert next(iter(db.iter_kernels())).structural_identity == record.identity


def test_a_cut_is_routing_rows_priced_by_the_receipts_of_its_pieces() -> None:
    """The leading entry pins the cut and names the parent, which ran as no kernel and gets no row; each
    piece's receipt is that piece's row. At the parent's fork the decision is priced as the sum of the
    pieces' rows, the read the deploy pick uses."""
    db, ctx, records, counts = _imported("fused/linear-add-place-cut-sm70.yaml")
    receipts = [record for record in records if record.identity != records[0].identity]

    assert counts["routing rows"] >= 1 and counts["kernels a decision replaced"] == 1
    assert counts["perf rows"] == len(receipts) >= 2
    decisions = list(db.iter_routing())
    parent = next(decision for decision in decisions if decision.arm == records[0].route)
    assert {row.kernel for row in db.iter_perf_rows()} <= {child for decision in decisions for child in decision.children}
    [(arm, us)] = db.priced_arms(ctx, parent.parent, bindings={})
    assert arm == records[0].route and us == pytest.approx(float(len(parent.children)))
    deploy = {kernel.exact_identity: kernel.structural_identity for kernel in db.iter_kernels()}
    assert {deploy[row.kernel] for row in db.iter_perf_rows()} <= {record.identity for record in receipts}


def test_a_split_timed_as_a_whole_is_routing_rows_only() -> None:
    """A split entry is a routing row: its time is the set's, which is no piece's, so nothing lands in
    perf for it and the arm is priced only by receipts of the pieces (none here: the case's receipts
    name kernels the current compiler no longer mints, and a stale identity writes nothing)."""
    db, _ctx, records, counts = _imported("reduce/cross-cta-matmul-kernel.yaml")

    assert records[0].is_routing and counts["routing rows"] >= 1
    assert counts["perf rows"] + counts.get("identities no kernel carries", 0) == len(records) - 1
    assert all(row.kernel != next(iter(db.iter_routing())).parent for row in db.iter_perf_rows())


def test_an_unmeasured_or_foreign_regime_entry_writes_nothing() -> None:
    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    db, ctx = SearchDB(), case.context()
    [record] = _records(case)

    with pinned_knobs(_regime(record)):
        assert import_goldens(db, ctx, case.records, source="golden:test") == {"unmeasured or in another regime": 1}
    with pinned_knobs({**_regime(record), "FAST_MATH": True}):
        assert import_goldens(db, ctx, [record], source="golden:test") == {"unmeasured or in another regime": 1}
    assert not list(db.iter_perf_rows())


def test_an_entry_naming_a_kernel_the_compiler_no_longer_mints_writes_nothing() -> None:
    """A stored identity the compiler has re-keyed is a golden to fix, not a row to guess a kernel
    for: on a one-kernel target as behind a cut, the entry writes nothing and is counted, so no
    time is ever filed under a kernel the entry did not measure."""
    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    db, ctx = SearchDB(), case.context()
    [record] = _records(case)
    with pinned_knobs(_regime(record)):
        counts = import_goldens(db, ctx, [replace(record, identity="0" * 64)], source="golden:test")
    assert counts == {"identities no kernel carries": 1} and not list(db.iter_perf_rows())
    cut = corpus.load_case(corpus.CASES_DIR / "fused/linear-add-place-cut-sm70.yaml")
    lead, first, *rest = _records(cut)
    _db, _ctx, _records_, counts = _imported("fused/linear-add-place-cut-sm70.yaml", [lead, replace(first, identity="0" * 64), *rest])
    assert counts["identities no kernel carries"] == 1 and counts["perf rows"] == len(rest)


def test_a_compile_imports_its_scope_once_and_lets_a_re_recorded_files_rows_go(tmp_path) -> None:
    """The tune DB imports a golden scope once per digest; a scope the DB has not seen replaces the
    earlier golden rows of that card and regime (keep-best would keep a stale faster row), and an
    empty scope deletes nothing."""
    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    [record] = _records(case)
    ctx = case.context()
    db = SearchDB(tmp_path / "tune.db")
    with pinned_knobs(_regime(record)):
        with records_override([record]):
            assert evidence_db(db, ctx) is db
            [row] = db.iter_perf_rows()
            assert row.stats.median == 1.0
            first = db.perf_sources()
            assert evidence_db(db, ctx) is db and db.perf_sources() == first
        slower = replace(record, measurements=replace(_MEASURED, emmy_us=2.0))
        with records_override([slower]):
            assert evidence_db(db, ctx) is db
            [row] = db.iter_perf_rows()
            assert row.stats.median == 2.0 and db.perf_sources() != first
        with records_override([]):
            assert evidence_db(db, ctx) is db
            assert [row.stats.median for row in db.iter_perf_rows()] == [2.0]
        # Back to the first scope: its rows were let go by the second's import, so it imports again
        # whatever this process remembers having imported.
        with records_override([record]):
            assert evidence_db(db, ctx) is db
            assert [row.stats.median for row in db.iter_perf_rows()] == [1.0]


def test_a_compile_without_a_db_picks_from_an_instance_holding_its_scope() -> None:
    """A scope is its records' content, target included: a case and its symbolic twin spell the same names,
    pins and knobs over different programs, and each compile picks from its own rows."""
    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    [record] = _records(case)
    ctx = case.context()
    with pinned_knobs(_regime(record)):
        with records_override([record]):
            assert len(list(evidence_db(None, ctx).iter_perf_rows())) == 1
        with records_override([replace(record, measurements=replace(_MEASURED, emmy_us=2.0))]):
            assert [row.stats.mean for row in evidence_db(None, ctx).iter_perf_rows()] == [2.0]
        with records_override([]):
            assert not list(evidence_db(None, ctx).iter_perf_rows())
    twins = [corpus.load_case(corpus.CASES_DIR / f"reduce/combine-amax-ilp-coop{suffix}.yaml") for suffix in ("", "-symbolic")]
    assert [record.name for record in twins[0].records] == [record.name for record in twins[1].records]
    kernels = []
    for twin in twins:
        with pinned_knobs(_regime(twin.record)), records_override(_records(twin)):
            kernels.append({row.kernel for row in evidence_db(None, twin.context()).iter_perf_rows()})
    assert kernels[0] and kernels[1] and kernels[0].isdisjoint(kernels[1])


@pytest.mark.xdist_group("golden_evidence_rtx5090")
def test_the_rtx_5090_hardware_golden_deploys_from_the_db(tmp_path) -> None:
    """The card's repository golden, imported into a fresh DB: every measured record in the standard regime
    is rows — a single-kernel record its kernel's row (an attention record whose stored identity the
    compiler has since re-keyed the same), a cross-CTA split winner its routing rows and nothing in perf —
    the tables agree with themselves, and a compile of each single-kernel record's target with that DB as
    its only evidence deploys a measured row of its kernel: the recorded variant, or a faster one the same
    file holds. The split winners are off the measured ballot until their pieces are benched."""
    from emmy import config
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.knob import schedule_row_key
    from emmy.compiler.pipeline.search.db import is_placement_knob
    from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR
    from emmy.compiler.pipeline.search.pins import regime_live
    from emmy.compiler.wire import kernel_tile
    from tests.compiler.pipeline.search.helpers import GPU_5090

    def splits(record) -> bool:
        return not record.is_routing and any(is_placement_knob(key, value) for key, value in record.schedule_row.items())

    records = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.yaml").records()
    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    db = SearchDB()
    with pinned_knobs({"FAST_MATH": False}):
        counts = import_goldens(db, ctx, records, source="golden:rtx5090")
        live = [record for record in records if record.measurements is not None and regime_live(record)]
        # A routing entry is its decision (routing rows, no perf row); a split winner is a schedule row
        # that spells a split, timed as a whole; every other live record is one kernel's row.
        routing = [record for record in live if record.is_routing]
        whole = [record for record in live if splits(record)]
        single = [record for record in live if record not in routing and record not in whole]
        assert len(single) >= 30 and len(whole) >= 5
        assert counts["perf rows"] == len(single) and counts["kernel sets timed as a whole"] == len(whole)
        assert counts["routing rows"] >= len(whole) + len(routing)
        assert counts["routing rows"] >= 1 and not counts["did not lower"] and not counts["identities no kernel carries"]
        assert not any(db.drift().values())
        measured: dict[str, list[dict]] = {}
        for row in db.iter_perf(ctx, backend="cuda"):
            measured.setdefault(row.kernel, []).append({k: str(v) for k, v in dict(schedule_row_key(dict(row.knobs))).items()})
        with records_override([]), config.online_file_override(tmp_path / "absent-online.json"):
            for record in single:
                graph = Pipeline.build(CUDA_PASSES).run(record.target_program.copy(), ctx=ctx, db=db)
                ops = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
                # A receipt names its kernel inside the set the target compiles to; any other record's target is one kernel.
                by_deploy = {kernel_tile(op).identity_key(with_io=True): op for op in ops}
                assert record.identity is not None or len(ops) == 1, record.name
                op = by_deploy[record.identity] if record.identity is not None else ops[0]
                picked = {k: str(v) for k, v in dict(schedule_row_key(dict(op.knobs or {}))).items()}
                assert picked in measured[kernel_tile(op).identity_key(structural=False, with_io=True)], record.name


def test_a_measurement_taken_here_is_never_replaced_by_an_import(tmp_path) -> None:
    """The tune DB is a cache the import fills, and a row this machine measured is the one copy of that
    measurement: a golden row of the same kernel and schedule, captured or faster, leaves it alone, and a
    scope change lets golden rows go, never a local one."""
    from emmy.compiler.pipeline.search.db import PerfStats
    from emmy.compiler.pipeline.search.policy.terminal_bench import point_stats

    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.yaml")
    [record] = _records(case)
    ctx = case.context()
    scratch, db = SearchDB(), SearchDB(tmp_path / "tune.db")
    with pinned_knobs(_regime(record)):
        # The kernel and its row as the import would file them, measured here first at a slower time.
        import_goldens(scratch, ctx, [record], source="golden:probe")
        [key] = scratch.iter_perf_rows()
        [kernel] = scratch.iter_kernels()
        db.record_kernel(kernel)
        local = PerfStats(median=9.0, min=9.0, max=9.0, mean=9.0, variance=0.0, n_samples=30)
        db.record_perf(ctx, key.kernel, bindings=key.bindings, knobs=key.knobs, backend="cuda", status="ok", stats=local)
        with records_override([record]):
            evidence_db(db, ctx)
        [row] = db.iter_perf_rows()
        assert (row.stats.median, row.source) == (9.0, "measured")
        faster = point_stats(1.0)
        db.record_perf(
            ctx,
            row.kernel,
            bindings=row.bindings,
            knobs=row.knobs,
            backend="cuda",
            status="ok",
            stats=faster,
            captured=True,
            source="golden:x",
        )
        with records_override([replace(record, measurements=replace(_MEASURED, emmy_us=0.5))]):
            evidence_db(db, ctx)
        [row] = db.iter_perf_rows()
        assert (row.stats.median, row.source) == (9.0, "measured")
