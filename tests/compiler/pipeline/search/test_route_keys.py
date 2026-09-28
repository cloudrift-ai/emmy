"""A route key the fresh lowering no longer resolves is loud: ``emmy golden check`` names it, restamp
re-spells it onto the seam the same operand positions reach, and a compile that pins it warns.

The case is the RMSNorm cut from the realization corpus: its route cuts ``PLACE@map.1/map``. A key
that walks the same positions under another kind (``reduce``) is what a lowering that re-reads a
node's kind leaves behind in every stored route."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.pipeline.search.restamp import restamp, stale_targets, unresolved_route_keys

_CASE = Path(__file__).parents[2] / "realization" / "cases" / "reduce" / "rms-norm-cut-sweep-work.json"
_KEY = "PLACE@map.1/map"
_STALE = "PLACE@map.1/reduce"


@pytest.fixture
def stale(tmp_path) -> GoldenFile:
    path = tmp_path / "case.json"
    path.write_text(_CASE.read_text().replace(f'"{_KEY}"', f'"{_STALE}"'))
    return GoldenFile.load(path)


def _route_row(document: GoldenFile):
    entry = document.configs[0]
    return document.record(entry, entry.realizations[0])


def test_a_current_route_resolves_every_key() -> None:
    document = GoldenFile.load(_CASE)
    assert unresolved_route_keys(_route_row(document), frozenset({_KEY})) == []
    assert list(stale_targets(document)) == []


def test_check_names_a_route_key_that_resolves_nowhere(stale: GoldenFile) -> None:
    reasons = list(stale_targets(stale))
    assert reasons == [f"k_rms_norm_3fbe25: route key {_STALE!r} names no seam of the fresh lowering"]


def test_restamp_respells_the_key_onto_the_seam_its_positions_reach(stale: GoldenFile) -> None:
    document, report = restamp(stale)
    assert report.rows_respelled == [f"k_rms_norm_3fbe25: {_STALE} -> {_KEY}"]
    assert document.configs[0].realizations[0].pins[_KEY] == "cut"
    assert list(stale_targets(document)) == []


def test_restamp_drops_a_route_whose_key_positions_reach_no_seam(tmp_path) -> None:
    path = tmp_path / "case.json"
    path.write_text(_CASE.read_text().replace(f'"{_KEY}"', '"PLACE@map.9/map"'))
    _, report = restamp(GoldenFile.load(path))
    assert report.rows_respelled == []
    # The route row goes, and with it the pieces only its cut minted.
    reason = "route key 'PLACE@map.9/map' names no seam of the fresh lowering, and its positions reach none"
    assert report.rows_dropped[0] == f"k_rms_norm_3fbe25: {reason}"


def test_a_pinned_key_that_resolves_nowhere_warns(caplog) -> None:
    record = _route_row(GoldenFile.load(_CASE))
    with caplog.at_level(logging.WARNING), pinned_knobs({"FAST_MATH": False, _STALE: "cut"}):
        Pipeline.build([*LOOP_PASSES, "tile/lift", "tile/cut"]).run(record.target_program.copy(), db=None)
    assert f"PLACE pin {_STALE} names no seam of any kernel in this compile" in caplog.text


def test_a_golden_route_key_that_resolves_nowhere_warns_at_import(tmp_path, caplog) -> None:
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden.evidence import import_goldens  # noqa: PLC0415

    document = json.loads(_CASE.read_text().replace(f'"{_KEY}"', f'"{_STALE}"'))
    for row in document["configs"][0]["realizations"]:
        row["measurements"] = {"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "test"}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    golden = GoldenFile.load(path)
    records = [golden.record(entry, row) for entry in golden.configs for row in entry.realizations]
    with caplog.at_level(logging.WARNING), pinned_knobs({"FAST_MATH": False}):
        import_goldens(SearchDB(), Context.from_target(tuple(golden.compute_cap)), records, source="test")
    assert f"{_STALE} names no seam of the fresh lowering" in caplog.text
