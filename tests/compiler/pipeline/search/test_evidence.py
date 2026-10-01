"""Golden rows become tune DB rows before a compile picks (``golden.evidence``).

A golden file is the DB's shape, so the import is a copy: every kernel a ``kernel`` row, every decision a ``routing``
row, every measured row a ``perf`` row in its own regime. The realization corpus stands in for a card's file: its
cases are authored without a GPU, so nothing here needs one. The compile's seam imports a scope once per digest and
lets a re-recorded file's rows go."""

from __future__ import annotations

import multiprocessing
from dataclasses import replace

import pytest

from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.golden import evidence_scope, import_rows, regime_live
from emmy.compiler.pipeline.search.golden.evidence import evidence_db
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.realization import helpers as corpus


def _case(path: str):
    case = corpus.load_case(corpus.CASES_DIR / path)
    return case, corpus._measured(case.document)


def _schedule(row) -> dict:
    return {key: str(value) for key, value in row.knobs.items() if not key.startswith(("S_", "I_"))}


def test_a_plain_row_is_its_kernels_row() -> None:
    """One target, one kernel, one row: the perf row is the row's schedule as recorded, keyed on the kernel, captured,
    under the source given."""
    case, document = _case("fused/norm-linear-f16-scalar-reduce.json")
    db, ctx = SearchDB(), case.context()
    assert import_rows(db, ctx, document, document.rows, source="golden:test") == 1
    [row] = db.iter_perf_rows()
    [stored] = document.rows
    assert _schedule(row) == {key: str(value) for key, value in stored.knobs.items()}
    assert (row.kernel, row.stats.median, row.captured, row.source, row.gpu) == (stored.kernel, 1.0, True, "golden:test", ctx.hardware_id())
    assert db.perf_sources(ctx) == {"golden:test": 1}
    [kernel] = db.iter_kernels()
    assert kernel.exact_identity == case.target.exact_identity and kernel.stamps == case.target.stamps


def test_a_cut_is_a_routing_row_priced_by_its_pieces_rows() -> None:
    """A decision is a routing row and each piece's row is that piece's perf row; at the parent's fork the decision
    is priced as the sum of the pieces' rows, the read the deploy pick uses. The parent ran as no kernel and has no
    row."""
    case, document = _case("fused/linear-add-place-cut-sm70.json")
    db, ctx = SearchDB(), case.context()
    [route] = document.routing
    assert route.parent == case.target.exact_identity and len(route.children) >= 2
    assert {row.kernel for row in document.rows} == set(route.children)
    import_rows(db, ctx, document, document.rows, source="golden:test")
    [stored] = db.iter_routing()
    assert stored == route
    assert {row.kernel for row in db.iter_perf_rows()} == set(route.children)
    [(arm, us)] = db.priced_arms(ctx, route.parent, bindings={})
    assert arm == route.arm and us == pytest.approx(float(len(route.children)))


def test_an_unmeasured_row_writes_nothing() -> None:
    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.json")
    db = SearchDB()
    assert import_rows(db, case.context(), case.document, case.document.rows, source="golden:test") == 0
    assert not list(db.iter_perf_rows()) and len(list(db.iter_kernels())) == 1


def test_a_row_is_evidence_in_its_own_regime_only() -> None:
    """A row measured under one precision regime is no evidence for a compile in the other, both ways."""
    with pinned_knobs({"FAST_MATH": False}):
        assert regime_live({"FAST_MATH": False}) and not regime_live({"FAST_MATH": True})
    with pinned_knobs({"FAST_MATH": True}):
        assert regime_live({"FAST_MATH": True}) and not regime_live({"FAST_MATH": False})


def test_a_compile_imports_its_scope_once_and_lets_a_re_recorded_files_rows_go(tmp_path) -> None:
    """The tune DB imports a golden scope once per digest; a scope the DB has not seen replaces the earlier golden
    rows of that card and regime (keep-best would keep a stale faster row), and an empty scope deletes nothing."""
    case, document = _case("fused/norm-linear-f16-scalar-reduce.json")
    ctx = case.context()
    db = SearchDB(tmp_path / "tune.db")
    [row] = document.rows
    slower = replace(document, rows=[replace(row, measurements=replace(row.measurements, emmy_us=2.0))])
    with pinned_knobs(case.regime):
        with evidence_scope([document]):
            assert evidence_db(db, ctx) is db
            [stored] = db.iter_perf_rows()
            assert stored.stats.median == 1.0
            first = db.perf_sources()
            assert evidence_db(db, ctx) is db and db.perf_sources() == first
        with evidence_scope([slower]):
            assert evidence_db(db, ctx) is db
            [stored] = db.iter_perf_rows()
            assert stored.stats.median == 2.0 and db.perf_sources() != first
        with evidence_scope([]):
            assert evidence_db(db, ctx) is db
            assert [stored.stats.median for stored in db.iter_perf_rows()] == [2.0]
        with evidence_scope([document]):
            assert evidence_db(db, ctx) is db
            assert [stored.stats.median for stored in db.iter_perf_rows()] == [1.0]


def test_a_compile_without_a_db_picks_from_an_instance_holding_its_scope() -> None:
    """A scope is its files' content: a case and its symbolic twin spell the same names, pins and knobs over
    different kernels, and each compile picks from its own rows."""
    case, document = _case("fused/norm-linear-f16-scalar-reduce.json")
    ctx = case.context()
    with pinned_knobs(case.regime):
        with evidence_scope([document]):
            assert len(list(evidence_db(None, ctx).iter_perf_rows())) == 1
        with evidence_scope([]):
            assert not list(evidence_db(None, ctx).iter_perf_rows())
    twins = [_case(f"reduce/combine-amax-ilp-coop{suffix}.json") for suffix in ("", "-symbolic")]
    assert [row.name for row in twins[0][1].rows] == [row.name for row in twins[1][1].rows]
    kernels = []
    for twin, measured in twins:
        with pinned_knobs(twin.regime), evidence_scope([measured]):
            kernels.append({row.kernel for row in evidence_db(None, twin.context()).iter_perf_rows()})
    assert kernels[0] and kernels[1] and kernels[0].isdisjoint(kernels[1])


@pytest.mark.xdist_group("golden_evidence_rtx5090")
def test_the_rtx_5090_hardware_golden_deploys_from_the_db() -> None:
    """The card's repository golden, imported into a fresh DB, is what a compile of each of its targets deploys from:
    with that DB as the only evidence, every target whose kernel has a measured row in the standard regime deploys a
    measured row of its kernel — the recorded variant, or a faster one the same file holds. A split winner is a
    routing row until its pieces are benched, so its target stays with the prior."""
    from emmy.compiler.context import Context
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.knob import schedule_row_key
    from emmy.compiler.pipeline.search.golden import GoldenFile
    from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR
    from emmy.compiler.wire import kernel_tile
    from tests.compiler.pipeline.search.helpers import GPU_5090

    document = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.json")
    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    db = SearchDB()
    with pinned_knobs({"FAST_MATH": False}):
        live = [row for row in document.rows if row.measured and regime_live(row.pins)]
        assert import_rows(db, ctx, document, live, source="golden:rtx5090") == len(live) >= 30
        assert not any(db.drift().values())
        measured: dict[str, list[dict]] = {}
        for row in db.iter_perf(ctx, backend="cuda"):
            measured.setdefault(row.kernel, []).append({k: str(v) for k, v in dict(schedule_row_key(dict(row.knobs))).items()})
        with evidence_scope([]):
            for target in document.targets():
                if target.exact_identity not in measured:
                    continue
                graph = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=ctx, db=db)
                [op] = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
                picked = {k: str(v) for k, v in dict(schedule_row_key(dict(op.knobs or {}))).items()}
                assert picked in measured[kernel_tile(op).identity_key(structural=False, with_io=True)], target.name


def test_a_measurement_taken_here_is_never_replaced_by_an_import(tmp_path) -> None:
    """The tune DB is a cache the import fills, and a row this machine measured is the one copy of that measurement: a
    golden row of the same kernel and schedule, captured or faster, leaves it alone, and a scope change lets golden
    rows go, never a local one."""
    from emmy.compiler.pipeline.search.bench_record import point_stats
    from emmy.compiler.pipeline.search.db import PerfStats

    case, document = _case("fused/norm-linear-f16-scalar-reduce.json")
    ctx = case.context()
    [row] = document.rows
    db = SearchDB(tmp_path / "tune.db")
    with pinned_knobs(case.regime):
        db.record_kernel(case.target)
        local = PerfStats(median=9.0, min=9.0, max=9.0, mean=9.0, variance=0.0, n_samples=30)
        db.record_perf(ctx, row.kernel, bindings=row.bindings, knobs=row.knobs, backend="cuda", status="ok", stats=local)
        with evidence_scope([document]):
            evidence_db(db, ctx)
        [stored] = db.iter_perf_rows()
        assert (stored.stats.median, stored.source) == (9.0, "measured")
        db.record_perf(
            ctx,
            row.kernel,
            bindings=row.bindings,
            knobs=row.knobs,
            backend="cuda",
            status="ok",
            stats=point_stats(1.0),
            captured=True,
            source="golden:x",
        )
        with evidence_scope([replace(document, rows=[replace(row, measurements=replace(row.measurements, emmy_us=0.5))])]):
            evidence_db(db, ctx)
        [stored] = db.iter_perf_rows()
        assert (stored.stats.median, stored.source) == (9.0, "measured")


def _import_in_a_worker(path, barrier) -> None:
    """One serving worker's first compile: open the shared tune DB and import the golden scope."""
    cut, document = _case("fused/linear-add-place-cut-sm70.json")
    with pinned_knobs(cut.regime), evidence_scope([document]):
        db = SearchDB(path)
        barrier.wait()
        evidence_db(db, cut.context())


def test_workers_sharing_a_tune_db_import_the_scope_once(tmp_path) -> None:
    """A tensor- and pipeline-parallel boot starts one process per card, and every one of them imports the golden
    scope into the same fresh tune DB at its first compile. The check, the forget and the import must be one step."""
    ctx = multiprocessing.get_context("spawn")
    workers = 6
    barrier = ctx.Barrier(workers)
    path = tmp_path / "tune.db"
    procs = [ctx.Process(target=_import_in_a_worker, args=(path, barrier)) for _ in range(workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(120)
    assert [proc.exitcode for proc in procs] == [0] * workers, "a worker raised (its traceback is on stderr)"
