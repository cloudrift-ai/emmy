"""``emmy golden check`` — the stored targets a fresh lowering of a golden's own programs no longer
writes — and ``emmy golden restamp``, the rewrite onto that lowering: a target takes the fresh Loop
IR, a row measured on a kernel the fresh Loop IR does not render keeps its schedule and loses its
microseconds, and a target no fresh kernel writes is dropped with its rows."""

from __future__ import annotations

import json
import shutil
from argparse import Namespace

import pytest

from emmy.commands.golden import handle_golden_check, handle_golden_restamp
from emmy.compiler.pipeline.search.golden import GoldenEntryState, GoldenFile
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR

#: Eight square matmuls with one measured row each — the smallest repository golden, and current.
_SMALLEST = _RECORDS_DIR / "rtx4080_sm89.json"


@pytest.fixture
def golden(tmp_path):
    path = tmp_path / "golden.json"
    shutil.copy(_SMALLEST, path)
    return path


def _check(path) -> int:
    try:
        handle_golden_check(Namespace(paths=[str(path)]))
    except SystemExit as exc:
        return int(exc.code)
    return 0


def _rename_output(document, index: int) -> None:
    """Rename program ``index``'s traced output, so its fresh kernel writes an output set no stored loop names."""
    program = document["programs"][index]
    program["outputs"] = ["product"]
    node = next(node for node in program["nodes"] if node["id"] == "matmul")
    node["id"] = "product"
    node["outputs"][0][0] = "product"
    document["configs"][index]["target"]["origins"] = ["product"]


def _make_stale(path) -> None:
    document = json.loads(path.read_text())
    document["configs"][0]["target"]["loop"] = 1  # the 512 target claims the 1024 kernel's Loop IR
    _rename_output(document, 2)  # the 2048 target's stored loop writes an output no fresh kernel does
    path.write_text(json.dumps(document))


def test_check_passes_a_current_golden_and_names_each_stale_target(golden, caplog):
    assert _check(golden) == 0
    _make_stale(golden)
    with caplog.at_level("ERROR"):
        assert _check(golden) == 1
    assert "2 of 8 stored targets are not the fresh lowering" in caplog.text
    assert "matmul.square.512: the fresh kernel's Loop IR differs" in caplog.text
    assert "matmul.square.2048: no fresh kernel writes its outputs" in caplog.text


def test_restamp_rewrites_the_golden_onto_the_fresh_lowering(golden, caplog):
    _make_stale(golden)
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert _check(golden) == 0, "the restamped file is the fresh lowering"

    document = GoldenFile.load(golden)
    rows = {entry.realizations[0].name: entry.realizations[0] for entry in document.configs}
    assert "matmul.square.2048" not in rows, "a target no fresh kernel writes is dropped"
    assert len(rows) == 7 and len(document.loops) == 7 and len(document.programs) == 7, "and its pools go with it"
    states = {name: row.state for name, row in rows.items()}
    assert states.pop("matmul.square.512") is GoldenEntryState.PROPOSAL, "measured on another kernel: the row keeps its schedule only"
    assert set(states.values()) == {GoldenEntryState.VERIFIED}, "a target that was already the fresh lowering keeps its measurement"
    assert "1 of 8 targets restamped, 1 dropped; 7 rows kept" in caplog.text  # nine rows: one target holds two

    caplog.clear()
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "already the fresh lowering" in caplog.text


def test_restamp_demotes_a_measurement_when_source_compile_fails(golden, monkeypatch):
    _make_stale(golden)
    monkeypatch.setattr("emmy.compiler.pipeline.search.golden.restamp._kernel_sources", lambda *_: None)
    handle_golden_restamp(Namespace(paths=[str(golden)]))
    document = GoldenFile.load(golden)
    row = next(row for entry in document.configs for row in entry.realizations if row.name == "matmul.square.512")
    assert row.state is GoldenEntryState.PROPOSAL


def test_restamp_drops_a_kernel_set_row_whose_members_lose_their_measurements(golden, caplog):
    """A kernel-set row carries no schedule of its own; once its member is demoted it spells nothing."""
    _make_stale(golden)
    document = json.loads(golden.read_text())
    realizations = document["configs"][0]["realizations"]
    lead = {key: value for key, value in realizations[0].items() if key not in ("knobs", "measurements", "identity")}
    realizations.append({**lead, "name": "matmul.square.512.set", "kernel_set": [realizations[0]["name"]]})
    golden.write_text(json.dumps(document))
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "matmul.square.512.set: its kernel set lost its measurements" in caplog.text
    names = [row.name for row in GoldenFile.load(golden).configs[0].realizations]
    assert names == ["matmul.square.512"], "the file stays a valid repository golden"


def test_restamp_keeps_the_piece_rows_of_a_kernel_set_whose_target_only_reordered_its_inputs(tmp_path, caplog):
    """A piece row names a kernel of its set, not the target: re-keying it to the fresh target's
    identity made it decode against the whole kernel, and every piece of a cut or split set was lost."""
    document = json.loads((_RECORDS_DIR / "rtx5090_sm120.json").read_text())
    entry = next(entry for entry in document["configs"] if entry["realizations"][0]["name"] == "attention.hd128.gqa.decode.split")
    loop = document["loops"][entry["target"]["loop"]]
    loop["inputs"] = loop["inputs"][::-1]  # the stale lowering: the same kernel with its inputs in another order
    document.update(programs=[document["programs"][entry["program"]]], loops=[loop])
    document["configs"] = [{**entry, "program": 0, "target": {**entry["target"], "loop": 0}}]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(document))
    assert _check(path) == 1
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(path)]))
    assert "1 of 1 targets restamped, 0 dropped; 3 rows kept" in caplog.text
    rows = GoldenFile.load(path).configs[0].realizations
    assert [row.identity for row in rows] == [row.get("identity") for row in entry["realizations"]], "each piece keeps its own"


def test_restamp_rekeys_a_mixed_set_lead_without_dropping_its_pieces(tmp_path):
    document = json.loads((_RECORDS_DIR / "rtx5090_sm120.json").read_text())
    entry = next(entry for entry in document["configs"] if entry["realizations"][0]["name"] == "attention.hd128.gqa.decode.split")
    loop = document["loops"][entry["target"]["loop"]]
    loop["inputs"] = loop["inputs"][::-1]
    document.update(programs=[document["programs"][entry["program"]]], loops=[loop])
    document["configs"] = [{**entry, "program": 0, "target": {**entry["target"], "loop": 0}}]
    lead, *pieces = document["configs"][0]["realizations"]
    lead["identity"] = "0" * len(lead["identity"])
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(document))

    handle_golden_restamp(Namespace(paths=[str(path)]))

    rows = GoldenFile.load(path).configs[0].realizations
    assert len(rows) == 1 + len(pieces)
    assert rows[0].identity != lead["identity"]
    assert [row.identity for row in rows[1:]] == [piece["identity"] for piece in pieces]


def test_restamp_drops_a_piece_row_the_fresh_set_no_longer_mints_under_a_current_target(tmp_path, caplog):
    """A cut can re-form a piece while the target's own Loop IR stays the same: the piece row then names no
    kernel, and restamp drops it though the target is already the fresh lowering."""
    document = json.loads((_RECORDS_DIR / "rtx5090_sm120.json").read_text())
    entry = next(entry for entry in document["configs"] if entry["realizations"][0]["name"] == "attention.hd128.gqa.decode.split")
    piece = next(row for row in entry["realizations"][1:] if row["name"] != entry["realizations"][0]["name"])
    entry["realizations"][0]["kernel_set"] = [piece["name"]]
    piece["identity"] = "0" * len(piece["identity"])
    document.update(programs=[document["programs"][entry["program"]]], loops=[document["loops"][entry["target"]["loop"]]])
    document["configs"] = [{**entry, "program": 0, "target": {**entry["target"], "loop": 0}}]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(document))
    assert _check(path) == 0
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(path)]))
    assert f"dropped row {piece['name']}: stored identity equals none" in caplog.text
    assert "its kernel set lost its measurements" in caplog.text
    assert "1 rows kept" in caplog.text
    rows = GoldenFile.load(path).configs[0].realizations
    assert piece["identity"] not in [row.identity for row in rows]
    assert all(piece["name"] not in row.kernel_set for row in rows)


def test_restamp_refuses_to_write_a_golden_nothing_survives_in(golden, caplog):
    document = json.loads(golden.read_text())
    for index in range(len(document["programs"])):
        _rename_output(document, index)
    golden.write_text(json.dumps(document))
    before = golden.read_bytes()
    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "no target survives" in caplog.text
    assert golden.read_bytes() == before, "deleting or re-recording the file is a decision, not a restamp"


def test_kernels_and_a_json_compile_output_are_the_same_pool_for_a_current_golden(golden, run_cli, tmp_path):
    """``emmy golden check`` in two commands: the stored pool and the fresh lowering's pool are one text
    while the file is current, and they part once a stored target is not the fresh lowering."""
    fresh = tmp_path / "fresh.json"
    compiled = run_cli("compile", "--golden", str(golden), "--program", "0", "--ir", "loop", "-o", str(fresh))
    stored = run_cli("golden", "kernels", str(golden), "--program", "0")
    assert compiled[0] == 0 and stored[0] == 0, (compiled[2], stored[2])
    assert stored[1] == fresh.read_text() and '"op":"loop"' in stored[1]
    program = tmp_path / "program.json"
    assert run_cli("compile", "--golden", str(golden), "--program", "0", "--ir", "torch", "-o", str(program))[0] == 0
    assert json.loads(program.read_text()) == json.loads(golden.read_text())["programs"][0], "the torch stage is the stored program"
    _make_stale(golden)
    assert run_cli("golden", "kernels", str(golden), "--program", "0")[1] != fresh.read_text(), "the 512 target now claims the 1024 kernel"
