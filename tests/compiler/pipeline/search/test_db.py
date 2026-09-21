"""The ``perf`` key — the card, the regime, the kernel, the sizes it was benched at and its knobs —
the ``kernel`` table beside it, and what happens to a file another emmy wrote: a writer re-creates the
table, a reader refuses it, nothing migrates."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.db import KernelRow, PerfStats, RoutingRow, SearchDB, knobs_json

_5090 = "NVIDIA GeForce RTX 5090"
_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
_ROW = {"S_x": 1.0, "WORK": "t16"}


def _stats(us: float) -> PerfStats:
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30)


def _ctx(gpu_name: str, flags: str = "") -> Context:
    return replace(Context.from_target((12, 0), gpu_name=gpu_name), compile_flags=flags)


def _record(db: SearchDB, ctx: Context, kernel: str, us: float, *, bindings: dict | None = None, knobs: dict | None = None, **kw) -> None:
    db.record_perf(
        ctx, kernel, bindings=bindings or {}, knobs=_ROW if knobs is None else knobs, backend="cuda", status="ok", stats=_stats(us), **kw
    )


def test_two_cards_sharing_a_capability_keep_their_own_rows() -> None:
    """Both cards are sm_120, so the regime alone cannot tell them apart. Keyed on it, the faster
    card's row would replace the slower one's under the keep-best upsert, and the slower card would
    deploy a latency it never measured."""
    db = SearchDB()
    a, b = _ctx(_5090), _ctx(_PRO_6000)
    _record(db, a, "k", 100.0)
    _record(db, b, "k", 80.0)

    assert db.lookup_perf(a, "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(b, "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 80.0
    assert [r.stats.median for r in db.iter_perf(a, backend="cuda")] == [100.0]
    rows = {r.gpu: r for r in db.iter_perf_rows()}
    assert set(rows) == {_5090, _PRO_6000}
    assert (rows[_5090].cc, rows[_5090].opt, rows[_5090].flags) == (120, 3, "")


def test_the_regime_is_spelled_in_columns_and_extra_flags_are_another_regime() -> None:
    """``""`` and ``-Xcicc -O3`` are one regime and read each other's rows; a fast-math compile is
    another and reads none of them."""
    db = SearchDB()
    _record(db, _ctx(_5090), "k", 100.0)
    assert db.lookup_perf(_ctx(_5090, "-Xcicc -O3"), "k", bindings={}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(_ctx(_5090, "--use_fast_math"), "k", bindings={}, knobs=_ROW, backend="cuda") is None
    _record(db, _ctx(_5090, "--use_fast_math"), "k", 90.0)
    assert {r.flags for r in db.iter_perf_rows()} == {"", "--use_fast_math"}


def test_one_dynamic_kernel_benched_at_two_sizes_is_two_rows() -> None:
    """A kernel's identity ignores its hint, so without the sizes in the key the rows of one dynamic
    kernel at two sizes would meet under keep-best and the smaller size would win every time."""
    db, ctx = SearchDB(), _ctx(_5090)
    _record(db, ctx, "k", 100.0, bindings={"seq_len": 512})
    _record(db, ctx, "k", 800.0, bindings={"seq_len": 4096})
    assert db.lookup_perf(ctx, "k", bindings={"seq_len": 512}, knobs=_ROW, backend="cuda").stats.median == 100.0
    assert db.lookup_perf(ctx, "k", bindings={"seq_len": 4096}, knobs=_ROW, backend="cuda").stats.median == 800.0
    assert db.lookup_perf(ctx, "k", bindings={}, knobs=_ROW, backend="cuda") is None


def test_a_kernel_row_is_written_once_and_names_the_kernel() -> None:
    db = SearchDB()
    db.record_kernel(KernelRow(identity="k", wire={"nodes": []}, name="k_first"))
    db.record_kernel(KernelRow(identity="k", wire={"nodes": []}, name="k_second"))
    assert db.kernel_names() == {"k": "k_first"}
    assert [row.wire for row in db.iter_kernels()] == [{"nodes": []}]


def test_a_routing_row_is_replaced_by_a_later_splice_of_the_same_decision() -> None:
    """One decision on one parent mints one set; a compiler that mints other pieces for the same
    decision replaces the row rather than keeping a stale one beside it."""
    db = SearchDB()
    db.record_routing(RoutingRow(parent="p", decision={"PLACE@a": "cut"}, children=("c1", "c2")))
    db.record_routing(RoutingRow(parent="p", decision={"PLACE@a": "cut"}, children=("c3",)))
    db.record_routing(RoutingRow(parent="p", decision={"REDUCE": "g2k"}, children=("c4", "c5")))
    assert [(r.parent, r.decision, r.children) for r in db.iter_routing_rows()] == [
        ("p", {"PLACE@a": "cut"}, ("c3",)),
        ("p", {"REDUCE": "g2k"}, ("c4", "c5")),
    ]


def test_best_per_op_time_prefers_the_whole_slice_total_over_the_kernel_rows() -> None:
    """The two-level tuner records a slice's Σ under the kernel with no knobs; where it never did,
    the kernel's fastest ok row stands in."""
    db, ctx = SearchDB(), _ctx(_5090)
    assert db.best_per_op_time(ctx, "k", bindings={}) is None
    _record(db, ctx, "k", 120.0, knobs={"S_x": 1.0, "WORK": "t16"})
    _record(db, ctx, "k", 90.0, knobs={"S_x": 1.0, "WORK": "t32"})
    db.record_perf(ctx, "k", bindings={}, knobs={"S_x": 1.0, "WORK": "t8"}, backend="cuda", status="bench_fail", stats=_stats(1.0))
    assert db.best_per_op_time(ctx, "k", bindings={}) == 90.0
    _record(db, ctx, "k", 200.0, knobs={}, captured=True)  # the Σ over a split's pieces is more than one piece
    assert db.best_per_op_time(ctx, "k", bindings={}) == 200.0


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
    """A tune DB as an emmy before the card key wrote it: a ``perf`` table with other columns, beside
    the inventory and ``lowering`` tables nothing reads any more."""
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE perf (context_key TEXT NOT NULL, op_key TEXT NOT NULL, backend TEXT NOT NULL, status TEXT NOT NULL, "
        "latency_us_median REAL NOT NULL, latency_us_min REAL NOT NULL, latency_us_max REAL NOT NULL, latency_us_mean REAL NOT NULL, "
        "latency_us_variance REAL NOT NULL, n_samples INTEGER NOT NULL, measured_at TEXT NOT NULL, knobs TEXT NOT NULL DEFAULT '{}', "
        "captured INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (context_key, op_key, backend))"
    )
    old.execute("INSERT INTO perf VALUES ('ctx', 'k', 'cuda', 'ok', 50.0, 50.0, 50.0, 50.0, 0.0, 30, '2026-01-01T00:00:00+00:00', '{}', 1)")
    old.execute("CREATE TABLE lowering (parent_key TEXT PRIMARY KEY, child_key TEXT NOT NULL)")
    old.execute("CREATE TABLE cuda_op (key TEXT PRIMARY KEY, kernel_source TEXT NOT NULL)")
    old.commit()
    old.close()


def test_a_file_another_emmy_wrote_is_re_created_by_a_writer_and_refused_by_a_reader(tmp_path) -> None:
    """Nothing migrates: the rows are regenerable, and a migration would be a second schema to carry.
    A writer open re-creates ``perf`` empty and drops the tables nothing reads; a read-only open
    cannot re-create anything, so it refuses the file and names the fix."""
    path = tmp_path / "autotune.db"
    _write_older_emmy_file(path)

    with pytest.raises(RuntimeError, match="written by another emmy"):
        SearchDB.open_readonly(path)

    db = SearchDB(path)
    assert list(db.iter_perf_rows()) == []
    tables = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"perf", "kernel", "routing"}
    _record(db, _ctx(_5090), "k", 60.0, captured=True)
    db.close()

    ro = SearchDB.open_readonly(path)
    assert [r.stats.median for r in ro.iter_perf_rows()] == [60.0]
    ro.close()
