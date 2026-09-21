"""The ``perf`` table's card key — two cards never share a row — and what happens to a file another
emmy wrote: a writer re-creates the table, a reader refuses it, nothing migrates."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.db import PerfStats, SearchDB

_5090 = "NVIDIA GeForce RTX 5090"
_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"


def _stats(us: float) -> PerfStats:
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=30)


def _ctx(gpu_name: str) -> Context:
    return replace(Context.from_target((12, 0), gpu_name=gpu_name), compile_flags="")


def test_two_cards_sharing_a_capability_keep_their_own_rows() -> None:
    """Both cards are sm_120, so the regime key alone cannot tell them apart. Keyed on it, the faster
    card's row would replace the slower one's under the keep-best upsert, and the slower card would
    deploy a latency it never measured."""
    db = SearchDB()
    a, b = _ctx(_5090), _ctx(_PRO_6000)
    assert a.structural_key() == b.structural_key()
    db.record_perf(a, "k", backend="cuda", status="ok", stats=_stats(100.0), knobs={"S_x": 1.0})
    db.record_perf(b, "k", backend="cuda", status="ok", stats=_stats(80.0), knobs={"S_x": 1.0})

    assert db.lookup_perf(a, "k", backend="cuda").stats.median == 100.0
    assert db.lookup_perf(b, "k", backend="cuda").stats.median == 80.0
    assert [r.stats.median for r in db.iter_perf(a, backend="cuda")] == [100.0]
    rows = {r.gpu: r for r in db.iter_perf_rows()}
    assert set(rows) == {_5090, _PRO_6000}
    assert (rows[_5090].cc, rows[_5090].opt) == (120, 3)


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
    old.execute("CREATE TABLE loop_op (key TEXT PRIMARY KEY, body_json TEXT NOT NULL, pretty TEXT NOT NULL)")
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
    assert tables == {"perf", "cuda_op"}
    ctx = _ctx(_5090)
    db.record_perf(ctx, "k", backend="cuda", status="ok", stats=_stats(60.0), captured=True)
    db.close()

    ro = SearchDB.open_readonly(path)
    assert [r.stats.median for r in ro.iter_perf_rows()] == [60.0]
    ro.close()
