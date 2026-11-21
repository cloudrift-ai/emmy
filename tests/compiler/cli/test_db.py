"""``emmy db import`` — a DB instance filled from golden files and tune DBs, each a copy of the source's rows;
``emmy db export``, the instance's rows as a dataset; the readers' refusal of a missing instance; and
``emmy db check``, the checks that an instance's tables agree with themselves."""

from __future__ import annotations

import dataclasses
from argparse import Namespace

import pytest

from emmy.commands.db import db_path, handle_db_check, handle_db_export, handle_db_import
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.golden import Measurements
from emmy.compiler.pipeline.search.golden.evidence import file_source
from tests.compiler.pipeline.search.helpers import CARDS, tuned_db
from tests.compiler.realization import helpers as corpus

_CASE = "fused/norm-linear-f16-scalar-reduce.json"


def _golden(tmp_path, name: str, us: float, case: str = _CASE):
    """A golden file of one corpus case, under ``name``, every row measured at ``us``."""
    document = corpus.load_case(corpus.CASES_DIR / case).document
    measured = Measurements(emmy_us=us, reference_us=us, reference_backend="corpus")
    rows = [dataclasses.replace(row, measurements=measured) for row in document.rows]
    card = CARDS[tuple(document.compute_cap)]  # the card ``tuned_db`` files the same case under
    return dataclasses.replace(document, gpu_name=card, rows=rows).dump(tmp_path / f"{name}.json")


def test_a_reader_refuses_a_missing_dataset_db_and_reads_a_filled_one(tmp_path):
    """Nothing fills a dataset DB but an import naming its sources and the file: a reader over an instance that
    does not exist exits with the command that fills it, and reads whatever an import put there — the sources are
    the report's to name, not the reader's to check."""
    current = _golden(tmp_path, "current", 500.0)

    with pytest.raises(SystemExit):
        db_path(str(tmp_path / "dataset.db"))  # nothing imported yet

    handle_db_import(Namespace(sources=[str(current)], db=str(tmp_path / "dataset.db"), fresh=True))
    assert db_path(str(tmp_path / "dataset.db")) == tmp_path / "dataset.db"
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert db.perf_sources() == {file_source("golden", current): 1}
    assert [row.us for row in db.iter_perf_rows()] == [500.0]
    db.close()


def test_a_golden_file_named_on_the_command_line_is_held_under_its_own_source(tmp_path):
    """A ``.json`` source is a golden file: its rows land under a ``golden:`` source of its own digest — where the
    golden readers read them — and naming it again is a no-op, so the documented import can be re-run without
    doubling a file's rows."""
    mine = _golden(tmp_path, "mine", 500.0)
    handle_db_import(Namespace(sources=[str(mine)], db=str(tmp_path / "dataset.db"), fresh=True))
    handle_db_import(Namespace(sources=[str(mine)], db=str(tmp_path / "dataset.db"), fresh=False))
    db = SearchDB.open_readonly(db_path(str(tmp_path / "dataset.db")))
    assert db.perf_sources() == {file_source("golden", mine): 1}
    db.close()


def test_a_golden_whose_rows_yield_no_measurement_is_still_held(tmp_path):
    """A source is a fact of its own, not a count of rows: a golden file none of whose rows becomes a measurement
    (a corpus case carries no timings; a restamped golden keeps its schedules and loses its microseconds) is held
    after the import — a report names it among its sources — and the dataset holds no row for it."""
    case = corpus.CASES_DIR / _CASE
    handle_db_import(Namespace(sources=[str(case)], db=str(tmp_path / "dataset.db"), fresh=True))
    assert db_path(str(tmp_path / "dataset.db")) == tmp_path / "dataset.db"
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert db.sources() == {file_source("golden", case)} and db.perf_sources() == {}
    db.close()


def test_export_writes_a_dataset_the_readers_load(tmp_path):
    """``export`` writes the instance's rows as a dataset directory — here an instance holding one measured row and no
    golden row, so the dataset carries one measured group, no golden group, and a provenance naming the DB — and a
    second export over the same rows writes the same manifest."""
    from emmy.compiler.pipeline.search.dataset import Dataset

    tuned_db(tmp_path / "dataset.db", (_CASE,), us=500.0).close()
    args = Namespace(
        out=str(tmp_path / "dataset"), db=str(tmp_path / "dataset.db"), pool_sample=0, seed=0, space="schedule", jobs=1, candidates=[]
    )
    handle_db_export(args)
    dataset = Dataset.load(tmp_path / "dataset")
    assert (len(dataset.golden), len(dataset.measured)) == (0, 1) and dataset.provenance["source"] == str(tmp_path / "dataset.db")
    assert dataset.measured[0].latency_us.tolist() == [500.0] and dataset.provenance["feat_ver"] == dataset.provenance["feat_ver"]
    first = (tmp_path / "dataset" / "manifest.json").read_bytes()
    handle_db_export(args)
    assert (tmp_path / "dataset" / "manifest.json").read_bytes() == first


def test_placement_export_reads_the_instance_alone(tmp_path):
    """A placement export opens no golden file: what a layout fork is offered on came in with the import."""
    from emmy.compiler.pipeline.search.dataset import Dataset

    golden = _golden(tmp_path, "cut", 500.0, case="fused/linear-add-place-cut-sm70.json")
    database = tmp_path / "dataset.db"
    handle_db_import(Namespace(sources=[str(golden)], db=str(database), fresh=True))
    args = Namespace(out=str(tmp_path / "placement"), db=str(database), pool_sample=0, seed=0, space="placement", jobs=1, candidates=[])
    handle_db_export(args)
    assert Dataset.load(args.out).golden


def test_a_tune_db_is_imported_row_for_row(tmp_path):
    """A tune DB is an instance of the same schema: the rows it measured itself reach the dataset under the keys it
    filed them under, sourced by the file's digest. A golden row a compile imported into the tune DB stays behind —
    the golden file holds it — and a file the instance already holds is not imported twice."""
    tune = tuned_db(tmp_path / "autotune.db", (_CASE, "fused/linear-add-place-cut-sm70.json"))
    [plain] = [row for row in tune.iter_perf_rows() if row.cc == 120]
    tune.record_perf_row(dataclasses.replace(plain, knobs={**plain.knobs, "WORK": "t8"}, source="golden:abcdef012345"))
    measured = sorted((row.kernel, knobs_json(row.bindings), row.us) for row in tune.iter_perf_rows() if row.source == "measured")
    tune.close()

    for _ in range(2):
        handle_db_import(Namespace(sources=[str(tmp_path / "autotune.db")], db=str(tmp_path / "dataset.db"), fresh=False))
        db = SearchDB.open_readonly(tmp_path / "dataset.db")
        assert sorted((row.kernel, knobs_json(row.bindings), row.us) for row in db.iter_perf_rows()) == measured
        assert db.perf_sources() == {file_source("tune", tmp_path / "autotune.db"): len(measured)}
        assert {kernel.exact_identity for kernel in db.iter_kernels()} == {kernel for kernel, _sizes, _us in measured}
        db.close()


def _instance(path):
    """A DB as a tune writes it: two real tile kernels with their measurements, one cut on the first that
    minted the second, on a registry card."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.bench_record import kernel_row as tile_row
    from emmy.compiler.wire import kernel_bindings
    from tests.compiler.helpers import case_target_tile
    from tests.compiler.pipeline.search.helpers import GPU_5090

    ctx = Context.from_target((12, 0), gpu_name=GPU_5090)
    parent = case_target_tile("fused/norm-linear-f16-scalar-reduce.json")
    piece = case_target_tile("matmul/f16-mma-f16acc-gmem.json")
    db = SearchDB(path)
    for tile in (parent, piece):
        db.record_kernel(tile_row(tile))
    for tile in (parent, piece):
        identity = tile.identity_key(structural=False, with_io=True)
        db.record_perf(ctx, identity, bindings=kernel_bindings(tile), knobs={"WORK": "t16"}, status="ok", us=10.0)
    db.record_routing(
        RoutingRow(
            parent=parent.identity_key(structural=False, with_io=True),
            arm={"PLACE": "cut"},
            children=(piece.identity_key(structural=False, with_io=True),),
        )
    )
    return db, parent, piece


_TAMPERS = {
    "every row names the rows it references": "DELETE FROM kernel WHERE exact_identity = (SELECT child FROM routing LIMIT 1)",
    "every CUDA context names a registry card": "UPDATE context SET gpu_name = 'Mystery GPU'",
}


def test_a_fresh_instance_has_no_drift(tmp_path):
    """What a tune writes agrees with itself: every reference and the card."""
    db, _parent, _piece = _instance(tmp_path / "tune.db")
    assert db.drift() == dict.fromkeys(_TAMPERS, 0)
    db.close()
    handle_db_check(Namespace(db=str(tmp_path / "tune.db")))


@pytest.mark.parametrize("check", list(_TAMPERS))
def test_each_kind_of_drift_is_counted_by_its_own_check(check):
    """One tamper per check, each the shape a code change would leave behind: a kernel row deleted from under its
    rows, a card the registry dropped. The check names it; the others stay quiet."""
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
        handle_db_check(Namespace(db=str(tmp_path / "tune.db")))


def test_repository_import_reads_the_repository_goldens(tmp_path, monkeypatch):
    """``--repository`` names the repository goldens, and nothing where there is none."""
    from contextlib import nullcontext

    from emmy.compiler.pipeline.search.golden import repository

    monkeypatch.setattr(repository, "repository_golden_paths", lambda: nullcontext([]))
    args = Namespace(sources=[], db=str(tmp_path / "dataset.db"), fresh=True, repository=True)
    with pytest.raises(SystemExit):
        handle_db_import(args)

    golden = _golden(tmp_path, "card", 500.0)
    monkeypatch.setattr(repository, "repository_golden_paths", lambda: nullcontext([golden]))
    handle_db_import(args)
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert db.perf_sources() == {file_source("golden", golden): 1}
    db.close()


def test_a_tune_db_row_never_replaces_a_golden_row(tmp_path):
    """A golden file's row is the file's: a tune DB that measured the same schedule faster — a tune re-benching the
    golden's own pick — does not take it over, or the golden readers, which read ``golden:`` rows alone, would lose
    the pool from the fit."""
    golden = _golden(tmp_path, "golden", 500.0)
    tuned_db(tmp_path / "faster.db", (_CASE,), us=400.0).close()
    handle_db_import(Namespace(sources=[str(golden), str(tmp_path / "faster.db")], db=str(tmp_path / "dataset.db"), fresh=True))
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert [(row.source.split(":")[0], row.us) for row in db.iter_perf_rows()] == [("golden", 500.0)]
    db.close()
