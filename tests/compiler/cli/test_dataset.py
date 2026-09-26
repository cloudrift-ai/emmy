"""``emmy dataset import`` — a dataset DB filled from a measurement freeze, golden files and tune DBs, every
kernel re-lowered by the current compiler — the readers' refusal of a default dataset DB that does not hold the
checked-in freeze, and ``emmy dataset check``, the checks that an instance's tables agree with themselves."""

from __future__ import annotations

import dataclasses
from argparse import Namespace

import pytest

from emmy.commands.dataset import dataset_db, handle_dataset_check, handle_dataset_import
from emmy.compiler.pipeline.search.data.freeze import freeze_source, write_freeze
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB, knobs_json
from emmy.compiler.structural import digest
from tests.compiler.pipeline.search.helpers import tuned_db

_CASE = "fused/norm-linear-f16-scalar-reduce.yaml"


def _freeze(tmp_path, name: str, us: float):
    """A freeze of one measured kernel at ``us``, written under ``name``."""
    tuned_db(tmp_path / f"{name}.db", (_CASE,), us=us).close()
    write_freeze(tmp_path / f"{name}.db", tmp_path / name)
    return {freeze_source(path) for path in (tmp_path / name).glob("*.yaml")}


def test_the_default_dataset_holds_the_checked_in_freeze_or_is_refused(tmp_path, monkeypatch):
    """A report over the default dataset DB carries the checked-in freeze's name as its data. A dataset
    built from another freeze — or from an older one, after the checked-in freeze moved on — would put
    yesterday's numbers under today's label, so the reader refuses it and names the command that fixes
    it."""
    current = _freeze(tmp_path, "current", 500.0)
    _freeze(tmp_path, "older", 400.0)
    monkeypatch.setenv("EMMY_DATASET_DB", str(tmp_path / "dataset.db"))
    monkeypatch.setenv("EMMY_FREEZE_DIR", str(tmp_path / "current"))

    with pytest.raises(SystemExit):
        dataset_db(None)  # nothing imported yet

    handle_dataset_import(Namespace(sources=[str(tmp_path / "older")], db=None, fresh=False))
    with pytest.raises(SystemExit):
        dataset_db(None)

    handle_dataset_import(Namespace(sources=[], db=None, fresh=True))
    assert dataset_db(None) == tmp_path / "dataset.db"
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert db.perf_sources() == dict.fromkeys(current, 1)
    assert [row.stats.median for row in db.iter_perf_rows()] == [500.0]
    db.close()


def test_a_tune_db_is_frozen_and_re_lowered_on_import(tmp_path):
    """A tune DB's rows reach the dataset the way a freeze of it would: re-lowered from each kernel's
    definition and sourced by the frozen file's digest. A golden row a compile imported into the tune DB
    stays behind — the golden file holds it — and a file the instance already holds is not imported twice."""
    tune = tuned_db(tmp_path / "autotune.db", (_CASE, "fused/linear-add-place-cut-sm70.yaml"))
    [plain] = [row for row in tune.iter_perf_rows() if row.cc == 120]
    tune.record_perf_row(dataclasses.replace(plain, knobs={**plain.knobs, "WORK": "t8"}, source="golden:abcdef012345"))
    measured = sorted((row.kernel, knobs_json(row.bindings), row.stats.median) for row in tune.iter_perf_rows() if row.source == "measured")
    tune.close()

    handle_dataset_import(Namespace(sources=[str(tmp_path / "autotune.db")], db=str(tmp_path / "dataset.db"), fresh=False))
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert sorted((row.kernel, knobs_json(row.bindings), row.stats.median) for row in db.iter_perf_rows()) == measured
    assert all(source.startswith("freeze:") for source in db.perf_sources()) and len(db.perf_sources()) == 2
    held = db.perf_sources()
    db.close()

    handle_dataset_import(Namespace(sources=[str(tmp_path / "autotune.db")], db=str(tmp_path / "dataset.db"), fresh=False))
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert db.perf_sources() == held
    db.close()


def _instance(path):
    """A DB as the tuner writes it: two real tile kernels with their measurements, one cut on the first that
    minted the second, on a registry card."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.db import PerfStats
    from emmy.compiler.pipeline.search.policy.terminal_bench import kernel_row as tile_row
    from emmy.compiler.wire import kernel_bindings
    from tests.compiler.helpers import case_target_tile
    from tests.compiler.pipeline.search.helpers import GPU_5090

    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    parent = case_target_tile("fused/norm-linear-f16-scalar-reduce.yaml")
    piece = case_target_tile("matmul/f16-mma-f16acc-gmem.yaml")
    db = SearchDB(path)
    for tile, name in ((parent, "k_parent"), (piece, "k_piece")):
        db.record_kernel(tile_row(tile, name))
    stats = PerfStats(median=10.0, min=10.0, max=10.0, mean=10.0, variance=0.0, n_samples=3)
    for tile in (parent, piece):
        identity = tile.identity_key(structural=False, with_io=True)
        db.record_perf(ctx, identity, bindings=kernel_bindings(tile), knobs={"WORK": "t16"}, backend="cuda", status="ok", stats=stats)
    db.record_routing(
        RoutingRow(
            parent=parent.identity_key(structural=False, with_io=True),
            arm={"PLACE": "cut"},
            children=(piece.identity_key(structural=False, with_io=True),),
        )
    )
    return db, parent, piece


_TAMPERS = {
    "schedule and placement digests match their knob rows": "UPDATE placement SET digest = 'moved'",
    "every row names the rows it references": "DELETE FROM kernel WHERE kernel_name = 'k_piece'",
    "every context names a registry card": "UPDATE context SET gpu_name = 'Mystery GPU'",
    "schedule knobs and placement knobs stay apart": (
        f"INSERT INTO schedule (id, digest) VALUES (99, '{digest(knobs_json({'PLACE': 'cut'}))}'); "
        "INSERT INTO schedule_knob VALUES (99, 'PLACE', 'cut')"
    ),
}


def test_a_fresh_instance_has_no_drift(tmp_path):
    """What the tuner writes agrees with itself: every knob row's digest, every reference, the card, the two
    knob vocabularies."""
    db, _parent, _piece = _instance(tmp_path / "tune.db")
    assert db.drift() == dict.fromkeys(_TAMPERS, 0)
    db.close()
    handle_dataset_check(Namespace(db=str(tmp_path / "tune.db")))


@pytest.mark.parametrize("check", list(_TAMPERS))
def test_each_kind_of_drift_is_counted_by_its_own_check(check):
    """One tamper per check, each the shape a code change would leave behind: a knob row whose digest
    moved, a kernel row deleted from under its rows, a card the registry dropped, a placement knob filed as
    a schedule knob. The check names it; the others stay quiet."""
    db, _parent, _piece = _instance(None)
    db._conn.execute("PRAGMA foreign_keys = OFF")
    db._conn.executescript(_TAMPERS[check])
    counts = db.drift()
    assert counts[check] >= 1, counts
    assert all(n == 0 for name, n in counts.items() if name != check), counts


def test_check_exits_nonzero_on_drift(tmp_path):
    db, _parent, _piece = _instance(tmp_path / "tune.db")
    db._conn.execute("UPDATE context SET gpu_name = 'Mystery GPU'")
    db._conn.commit()
    db.close()
    with pytest.raises(SystemExit):
        handle_dataset_check(Namespace(db=str(tmp_path / "tune.db")))
