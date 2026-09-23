"""The tune DB's tables: measurements of compilable kernels keyed by context, kernel, bindings and
schedule row; the kernel rows with their stamps; the decisions that mint pieces; and what happens to a
file another emmy wrote — a writer re-creates every table, a reader refuses it, nothing migrates."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.db import PerfStats, RoutingRow, SearchDB, knobs_json
from tests.compiler.pipeline.search.helpers import kernel_row, perf_row

_5090 = "NVIDIA GeForce RTX 5090"
_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
_STAMPS = {"S_x": 1.0}
_ROW = {"S_x": 1.0, "WORK": "t16"}


def _stats(us: float) -> PerfStats:
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30)


def _ctx(gpu_name: str, flags: str = "") -> Context:
    return replace(Context.from_target((12, 0), gpu_name=gpu_name), compile_flags=flags)


def _db(*kernels: str, **symbolic: tuple[str, ...]) -> SearchDB:
    """An in-memory instance holding a kernel row per name (a ``symbolic`` keyword names a kernel's dims)."""
    db = SearchDB()
    for k in kernels:
        db.record_kernel(kernel_row(k, stamps=_STAMPS, symbolic=symbolic.get(k, ())))
    return db


def _record(db: SearchDB, ctx: Context, kernel: str, us: float, *, bindings: dict | None = None, knobs: dict | None = None, **kw) -> None:
    kw.setdefault("status", "ok")
    db.record_perf(ctx, kernel, bindings=bindings or {}, knobs=_ROW if knobs is None else knobs, backend="cuda", stats=_stats(us), **kw)


def test_two_cards_sharing_a_capability_keep_their_own_rows() -> None:
    """Both cards are sm_120, so the regime alone cannot tell them apart. Keyed on it, the faster
    card's row would replace the slower one's under the keep-best upsert, and the slower card would
    deploy a latency it never measured."""
    db = _db("k")
    a, b = _ctx(_5090), _ctx(_PRO_6000)
    _record(db, a, "k", 100.0)
    _record(db, b, "k", 80.0)

    assert db.lookup_perf(a, "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(b, "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 80.0
    assert [r.stats.median for r in db.iter_perf(a, backend="cuda")] == [100.0]
    rows = {r.gpu: r for r in db.iter_perf_rows()}
    assert set(rows) == {_5090, _PRO_6000}
    assert (rows[_5090].cc, rows[_5090].opt, rows[_5090].flags, rows[_5090].backend) == (120, 3, "", "cuda")


def test_the_regime_is_spelled_in_columns_and_extra_flags_are_another_regime() -> None:
    """``""`` and ``-Xcicc -O3`` are one regime and read each other's rows; a fast-math compile is
    another context and reads none of them."""
    db = _db("k")
    _record(db, _ctx(_5090), "k", 100.0)
    assert db.lookup_perf(_ctx(_5090, "-Xcicc -O3"), "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(_ctx(_5090, "--use_fast_math"), "k", bindings={}, knobs=_ROW, backend="cuda") is None
    _record(db, _ctx(_5090, "--use_fast_math"), "k", 90.0)
    assert {r.flags for r in db.iter_perf_rows()} == {"", "--use_fast_math"}
    assert db._conn.execute("SELECT COUNT(*) FROM context").fetchone() == (2,)


def test_one_dynamic_kernel_benched_at_two_sizes_is_two_rows() -> None:
    """A kernel's identity ignores its hint, so without the sizes in the key the rows of one dynamic
    kernel at two sizes would meet under keep-best and the smaller size would win every time."""
    db, ctx = _db("k"), _ctx(_5090)
    _record(db, ctx, "k", 100.0, bindings={"seq_len": 512})
    _record(db, ctx, "k", 800.0, bindings={"seq_len": 4096})
    assert db.lookup_perf(ctx, "k", bindings={"seq_len": 512}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(ctx, "k", bindings={"seq_len": 4096}, knobs=_ROW, backend="cuda").stats.median == 800.0
    assert db.lookup_perf(ctx, "k", bindings={}, knobs=_ROW, backend="cuda") is None


def test_a_schedule_row_is_shared_and_a_read_row_reassembles_the_kernel_stamps() -> None:
    """Two measurements taken with the same choices share one schedule row, whatever kernel they are
    of. The ``S_*`` entries a writer passes are the kernel's, not the row's: what a reader gets back
    is the kernel row's stamps plus the schedule row, the flat dict the featurizer always read."""
    db, ctx = _db("a", "b"), _ctx(_5090)
    _record(db, ctx, "a", 100.0, knobs={"S_x": 1.0, "WORK": "t16", "H_opt": 3.0})
    _record(db, ctx, "b", 200.0, knobs={"S_x": 7.0, "WORK": "t16"})
    assert db._conn.execute("SELECT COUNT(*) FROM schedule").fetchone() == (1,)
    assert db._conn.execute("SELECT COUNT(*) FROM schedule_knob").fetchone() == (1,)
    rows = {r.kernel: r.knobs for r in db.iter_perf_rows()}
    assert rows == {"a": {"S_x": 1.0, "WORK": "t16"}, "b": {"S_x": 1.0, "WORK": "t16"}}
    # The empty row is a schedule too: a forkless kernel's one measurement.
    _record(db, ctx, "a", 50.0, knobs={})
    assert db.lookup_perf(ctx, "a", bindings={}, knobs={}, backend="cuda").stats.median == 50.0


def test_a_placement_knob_is_refused_as_a_measurement() -> None:
    """A row spelling a cut or a cross-CTA split is a kernel-set decision — a routing row — never a
    measurement of one kernel; the in-kernel half of ``REDUCE`` is a schedule knob like any other."""
    db, ctx = _db("k"), _ctx(_5090)
    with pytest.raises(ValueError, match="placement knob"):
        _record(db, ctx, "k", 1.0, knobs={"PLACE@map.1/inner": "cut"})
    with pytest.raises(ValueError, match="placement knob"):
        _record(db, ctx, "k", 1.0, knobs={"REDUCE": "g2k"})
    _record(db, ctx, "k", 1.0, knobs={"REDUCE": "coop"})
    assert db.lookup_perf(ctx, "k", bindings={}, knobs={"REDUCE": "coop"}, backend="cuda") is not None


def test_a_kernel_row_is_written_once_and_its_stamps_replaced_on_a_restamp() -> None:
    db = SearchDB()
    db.record_kernel(kernel_row("k", stamps={"S_x": 1.0}, name="k_first"))
    db.record_kernel(kernel_row("k", stamps={"S_x": 1.0}, name="k_second"))
    assert db.kernel_names() == {"k": "k_first"}
    db.record_kernel(kernel_row("k", stamps={"S_x": 2.0, "S_y": 1.0}, name="k_third"))
    [row] = list(db.iter_kernels())
    assert (row.name, row.stamps) == ("k_first", {"S_x": 2.0, "S_y": 1.0})
    assert row.structural_identity == "deploy:k" and row.loop_ir == row.normalized_loop_ir


def test_a_routing_row_holds_one_piece_per_position_and_a_later_splice_replaces_it() -> None:
    db = _db("p", "c1", "c2")
    db.record_routing(RoutingRow(parent="p", arm={"PLACE@map.1/inner": "cut"}, children=("c1", "c2")))
    assert list(db.iter_routing()) == [RoutingRow(parent="p", arm={"PLACE@map.1/inner": "cut"}, children=("c1", "c2"))]
    db.record_routing(RoutingRow(parent="p", arm={"PLACE@map.1/inner": "cut"}, children=("c2",)))
    db.record_routing(RoutingRow(parent="p", arm={"REDUCE": "g2k"}, children=("c1", "c2")))
    assert [(r.arm, r.children) for r in db.iter_routing()] == [({"PLACE@map.1/inner": "cut"}, ("c2",)), ({"REDUCE": "g2k"}, ("c1", "c2"))]
    with pytest.raises(ValueError, match="placement knobs only"):
        db.record_routing(RoutingRow(parent="p", arm={"WORK": "t16"}, children=("c1",)))
    with pytest.raises(sqlite3.IntegrityError):
        db.record_routing(RoutingRow(parent="p", arm={"PLACE": "cut"}, children=("ghost",)))


def test_best_per_op_time_prices_a_leaf_or_the_sum_of_its_pieces() -> None:
    """A kernel's best time is its fastest ok row, or — when a decision cut it — the sum of its pieces'
    fastest rows, each piece at its own projection of the parent's bindings, all-or-nothing."""
    db, ctx = _db("p", "c1", "c2", c1=("seq_len",)), _ctx(_5090)
    at = {"seq_len": 512}
    assert db.best_per_op_time(ctx, "p", bindings=at) is None
    db.record_routing(RoutingRow(parent="p", arm={"PLACE@map.1/inner": "cut"}, children=("c1", "c2")))
    _record(db, ctx, "c1", 30.0, bindings=at)  # c1 kept the symbolic dim: benched at the parent's size
    assert db.best_per_op_time(ctx, "p", bindings=at) is None, "c2 has no row: the cut is unpriced"
    _record(db, ctx, "c2", 50.0)  # c2 went static: its bindings are {}
    assert db.best_per_op_time(ctx, "p", bindings=at) == 80.0
    _record(db, ctx, "p", 100.0, bindings=at)  # the fuse arm measured as a leaf
    assert db.best_per_op_time(ctx, "p", bindings=at) == 80.0
    _record(db, ctx, "p", 70.0, bindings=at, knobs={"WORK": "t32"})
    assert db.best_per_op_time(ctx, "p", bindings=at) == 70.0
    assert db.best_per_op_time(ctx, "p", bindings={"seq_len": 4096}) is None, "nothing measured at that size"
    _record(db, ctx, "c1", 1.0, bindings=at, status="bench_fail", knobs={"WORK": "t8"})
    assert db.best_per_op_time(ctx, "c1", bindings=at) == 30.0


def test_a_knob_row_has_one_spelling_and_no_lossy_fallback() -> None:
    """The spelling identifies a row, so two writers must agree on it, and a value json cannot spell
    raises rather than turning into a string that would key a second row for the same kernel. An int
    and a float are two spellings on purpose: a row's ``S_*`` stamps are floats, and a writer that
    passed ints would be a different vocabulary, not the same row."""
    assert knobs_json({"b": 1.0, "a": "t16"}) == knobs_json({"a": "t16", "b": 1.0}) == '{"a":"t16","b":1.0}'
    assert knobs_json({"S_shape": 128}) != knobs_json({"S_shape": 128.0})
    with pytest.raises(TypeError):
        knobs_json({"S_x": object()})
    with pytest.raises(ValueError):
        knobs_json({"S_x": float("nan")})


def _write_older_emmy_file(path) -> None:
    """A tune DB as the emmy of PR #839 wrote it: a ``perf`` table with other columns, beside tables
    nothing reads any more."""
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE perf (gpu TEXT NOT NULL, context_key TEXT NOT NULL, op_key TEXT NOT NULL, backend TEXT NOT NULL, "
        "status TEXT NOT NULL, "
        "latency_us_median REAL NOT NULL, latency_us_min REAL NOT NULL, latency_us_max REAL NOT NULL, latency_us_mean REAL NOT NULL, "
        "latency_us_variance REAL NOT NULL, n_samples INTEGER NOT NULL, measured_at TEXT NOT NULL, knobs TEXT NOT NULL DEFAULT '{}', "
        "captured INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (gpu, context_key, op_key, backend))"
    )
    old.execute(
        "INSERT INTO perf VALUES ('g', 'ctx', 'k', 'cuda', 'ok', 50.0, 50.0, 50.0, 50.0, 0.0, 30, '2026-01-01T00:00:00+00:00', '{}', 1)"
    )
    old.execute("CREATE TABLE kernel_set (parent TEXT, decision TEXT, children TEXT)")
    old.execute("CREATE TABLE cuda_op (key TEXT PRIMARY KEY, kernel_source TEXT NOT NULL)")
    old.commit()
    old.close()


def test_a_file_another_emmy_wrote_is_re_created_whole_by_a_writer_and_refused_by_a_reader(tmp_path) -> None:
    """Nothing migrates: the rows are regenerable, and a migration would be a second schema to carry.
    A writer open re-creates EVERY table (dropping one would orphan the rows that reference it) and
    the tables nothing reads; a read-only open cannot re-create anything, so it refuses the file and
    names the fix. Foreign keys are enforced from then on."""
    path = tmp_path / "autotune.db"
    _write_older_emmy_file(path)

    with pytest.raises(RuntimeError, match="written by another emmy"):
        SearchDB.open_readonly(path)

    db = SearchDB(path)
    assert list(db.iter_perf_rows()) == []
    tables = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"kernel", "kernel_feature", "context", "schedule", "schedule_knob", "placement", "placement_knob", "routing", "perf"}
    with pytest.raises(sqlite3.IntegrityError):
        db.record_perf_rows([perf_row("ghost", us=1.0)])  # no kernel row behind it
    db.record_kernel(kernel_row("k"))
    db.record_perf_rows([perf_row("k", us=60.0, captured=True)])
    db.close()

    ro = SearchDB.open_readonly(path)
    assert [(r.stats.median, r.knobs["WORK"]) for r in ro.iter_perf_rows()] == [(60.0, "w1x8")]
    ro.close()
