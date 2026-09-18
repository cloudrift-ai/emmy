"""The ``perf`` table's card key — two cards never share a row — and the one-time migration of a table
written before the card joined the key."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

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


def test_a_pre_card_table_migrates_and_keeps_serving_its_machine(tmp_path) -> None:
    """An old ``perf`` table is rebuilt card-keyed on the first writer open. Its rows cannot say which
    card measured them, so they take an empty card: the machine that owns the file still deploys them,
    a fresh measurement of the same kernel lands beside them as the card's own row and wins the lookup,
    and a read-only open of an unmigrated file reads them the same way."""
    path = tmp_path / "autotune.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE perf (context_key TEXT NOT NULL, op_key TEXT NOT NULL, backend TEXT NOT NULL, status TEXT NOT NULL, "
        "latency_us_median REAL NOT NULL, latency_us_min REAL NOT NULL, latency_us_max REAL NOT NULL, latency_us_mean REAL NOT NULL, "
        "latency_us_variance REAL NOT NULL, n_samples INTEGER NOT NULL, measured_at TEXT NOT NULL, knobs TEXT NOT NULL DEFAULT '{}', "
        "captured INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (context_key, op_key, backend))"
    )
    ctx = _ctx(_5090)
    old.execute(
        "INSERT INTO perf VALUES (?, 'k', 'cuda', 'ok', 50.0, 50.0, 50.0, 50.0, 0.0, 30, '2026-01-01T00:00:00+00:00', '{}', 1)",
        (ctx.structural_key(),),
    )
    old.commit()
    old.close()

    ro = SearchDB.open_readonly(path)
    [unmigrated] = ro.iter_perf_rows()
    ro.close()
    assert (unmigrated.gpu, unmigrated.cc, unmigrated.feat_ver) == ("", None, 1)

    db = SearchDB(path)
    [migrated] = db.iter_perf_rows()
    assert migrated == unmigrated
    assert db.lookup_perf(ctx, "k", backend="cuda").stats.median == 50.0

    db.record_perf(ctx, "k", backend="cuda", status="ok", stats=_stats(60.0), captured=True)
    assert db.lookup_perf(ctx, "k", backend="cuda").gpu == _5090
    assert sorted(r.gpu for r in db.iter_perf(ctx, backend="cuda")) == ["", _5090]
    db.close()
