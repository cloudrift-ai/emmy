"""``emmy dataset import`` — a dataset DB filled from a measurement freeze and from tune DBs — the readers'
refusal of a default dataset DB that does not hold the checked-in freeze, and ``emmy dataset check``, the
drift checks the stored definitions allow."""

from __future__ import annotations

from argparse import Namespace

import pytest

from emmy.commands.dataset import dataset_db, handle_dataset_check, handle_dataset_import
from emmy.compiler.pipeline.search.data.freeze import write_freeze
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB, knobs_json
from emmy.compiler.structural import digest
from tests.compiler.pipeline.search.helpers import kernel_row, perf_row


def _freeze(tmp_path, name: str, us: float):
    db = SearchDB(tmp_path / f"{name}.db")
    db.record_kernel(kernel_row(name))
    db.record_perf_rows([perf_row(name, us=us)])
    db.close()
    return write_freeze(tmp_path / f"{name}.db", tmp_path / name)


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
    assert db.perf_sources() == {f"freeze:{current['sha256'][:12]}": 1}
    db.close()


def test_a_tune_db_imports_its_rows_and_definitions(tmp_path):
    """A tune DB's CUDA rows arrive as they are, keeping the source they were written with, and so do
    its kernel and routing rows — the definitions its rows are of."""
    tune = SearchDB(tmp_path / "autotune.db")
    tune.record_kernels([kernel_row("k", name="k_test", symbolic=("seq_len",)), kernel_row("p")])
    tune.record_perf_rows([perf_row("k", us=500.0), perf_row("k", us=300.0, bindings={"seq_len": 128})])
    tune.record_routing(RoutingRow(parent="p", arm={"PLACE": "cut"}, children=("k",)))
    tune.close()

    handle_dataset_import(Namespace(sources=[str(tmp_path / "autotune.db")], db=str(tmp_path / "dataset.db"), fresh=False))
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert sorted((r.kernel, tuple(r.bindings.items())) for r in db.iter_perf_rows()) == [("k", ()), ("k", (("seq_len", 128),))]
    assert db.perf_sources() == {"measured": 2}
    assert db.kernel_names() == {"k": "k_test", "p": "k_p"}
    assert [(s.parent, s.arm, s.children) for s in db.iter_routing()] == [("p", {"PLACE": "cut"}, ("k",))]
    db.close()


def _instance(path):
    """A DB as the tuner writes it: two real tile kernels with their measurements, one cut on the first that
    minted the second, on a registry card."""
    from emmy.compiler.context import Context
    from emmy.compiler.loop_wire import kernel_bindings
    from emmy.compiler.pipeline.search.db import PerfStats
    from emmy.compiler.pipeline.search.policy.terminal_bench import kernel_row as tile_row
    from tests.compiler.helpers import case_target_tile
    from tests.compiler.pipeline.search.helpers import GPU_5090

    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    parent = case_target_tile("fused/norm-linear-f16-scalar-reduce.yaml")
    piece = case_target_tile("matmul/f16-mma-f16acc-gmem.yaml")
    db = SearchDB(path)
    db.record_kernels([tile_row(parent, "k_parent"), tile_row(piece, "k_piece")])
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
    "loop_ir normalizes to normalized_loop_ir": "UPDATE kernel SET loop_ir = (SELECT loop_ir FROM kernel WHERE kernel_name = 'k_piece') "
    "WHERE kernel_name = 'k_parent'",
    "normalized_loop_ir lifts to the stored identities": "UPDATE kernel SET structural_identity = 'moved' WHERE kernel_name = 'k_parent'",
    "normalized_loop_ir stamps to kernel_feature": "UPDATE kernel_feature SET value = value + 1 WHERE name = 'S_loop_depth'",
    "perf bindings name the kernel's symbolic dims": "UPDATE perf SET bindings = '{\"ghost\":4}'",
    "schedule and placement digests match their knob rows": "UPDATE placement SET digest = 'moved'",
    "every routing child and perf row names a kernel row": "DELETE FROM kernel WHERE kernel_name = 'k_piece'",
    "every context names a registry card": "UPDATE context SET gpu_name = 'Mystery GPU'",
    "schedule knobs and placement knobs stay apart": (
        f"INSERT INTO schedule (id, digest) VALUES (99, '{digest(knobs_json({'PLACE': 'cut'}))}'); "
        "INSERT INTO schedule_knob VALUES (99, 'PLACE', 'cut')"
    ),
}


def test_a_fresh_instance_has_no_drift(tmp_path):
    """Every definition the tuner stores re-derives under the code that stored it: the raw wire normalizes
    to the stored one, the stored one lifts to both identities and stamps to the feature rows, the bindings
    name the dims, and the tables agree with themselves."""
    from emmy.compiler.pipeline.search.data.check import drift

    db, _parent, _piece = _instance(tmp_path / "tune.db")
    assert drift(db) == dict.fromkeys(_TAMPERS, 0)
    db.close()
    handle_dataset_check(Namespace(db=str(tmp_path / "tune.db")))


@pytest.mark.parametrize("check", list(_TAMPERS))
def test_each_kind_of_drift_is_counted_by_its_own_check(check):
    """One tamper per check, each the shape a code change would leave behind: another kernel's raw wire,
    an identity the digest no longer produces, a re-stamped feature, a binding of a dim the IR lost, a knob
    row whose digest moved, a kernel row deleted from under its rows, a card the registry dropped, a
    placement knob filed as a schedule knob. The check names it; the others stay quiet."""
    from emmy.compiler.pipeline.search.data.check import drift

    db, _parent, _piece = _instance(None)
    db._conn.execute("PRAGMA foreign_keys = OFF")
    db._conn.executescript(_TAMPERS[check])
    counts = drift(db)
    assert counts[check] >= 1, counts
    assert all(n == 0 for name, n in counts.items() if name != check), counts


def test_check_exits_nonzero_on_drift(tmp_path):
    db, _parent, _piece = _instance(tmp_path / "tune.db")
    db._conn.execute("UPDATE context SET gpu_name = 'Mystery GPU'")
    db._conn.commit()
    db.close()
    with pytest.raises(SystemExit):
        handle_dataset_check(Namespace(db=str(tmp_path / "tune.db")))
