"""The serving lane's evidence is its golden alone, whatever other tests import into the session's tune DB (CPU-only)."""

from __future__ import annotations

from emmy import config
from tests.serving import helpers


def test_the_lane_picks_from_a_tune_db_no_other_worker_imports_into(monkeypatch):
    """Every xdist worker shares the session's tune DB, and a compile importing another golden scope (the set of
    golden files it reads evidence from) deletes the lane's golden rows there; the lane's compiles pick from a file of
    this process's own, and the session's DB is back once ``evidence_scope`` exits."""
    monkeypatch.setattr(helpers, "golden_path", lambda compute_capability=None: helpers.GOLDENS / "serving_sm120.golden.json")
    session = config.tune_db_path()
    with helpers.evidence_scope():
        lane = config.tune_db_path()
    assert lane != session and lane.parent != session.parent
    assert config.tune_db_path() == session
    with helpers.evidence_scope():
        assert config.tune_db_path() == lane
