"""``emmy dataset import`` — a dataset DB filled from a measurement freeze and from tune DBs — and the
readers' refusal of a default dataset DB that does not hold the checked-in freeze."""

from __future__ import annotations

from argparse import Namespace

import pytest

from emmy.commands.dataset import dataset_db, handle_dataset_import
from emmy.compiler.pipeline.search.data.freeze import write_freeze
from emmy.compiler.pipeline.search.db import SearchDB
from tests.compiler.pipeline.search.helpers import perf_row


def _freeze(tmp_path, name: str, us: float):
    db = SearchDB(tmp_path / f"{name}.db")
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


def test_a_tune_db_imports_its_card_keyed_rows(tmp_path):
    """A tune DB's CUDA rows arrive as they are, keeping the source they were written with; the rows it
    holds from before the card joined the key cannot say which card measured them and stay behind."""
    tune = SearchDB(tmp_path / "autotune.db")
    tune.record_perf_rows([perf_row("keyed", us=500.0), perf_row("pre-card", us=300.0, gpu="", cc=None, opt=None, context_key="old")])
    tune.close()

    handle_dataset_import(Namespace(sources=[str(tmp_path / "autotune.db")], db=str(tmp_path / "dataset.db"), fresh=False))
    db = SearchDB.open_readonly(tmp_path / "dataset.db")
    assert [r.op_key for r in db.iter_perf_rows()] == ["keyed"]
    assert db.perf_sources() == {"measured": 1}
    db.close()
