"""``emmy golden check`` — what a restamp onto the fresh lowering of a golden's own programs would change — and
``emmy golden restamp``, the rewrite: a kernel takes the fresh identity and body, a row measured on a kernel whose
body changed keeps its schedule and loses its microseconds, a kernel no fresh kernel writes is dropped with its
rows, a decision the fresh parent no longer takes the same way goes with its pieces' rows."""

from __future__ import annotations

import json
import shutil
from argparse import Namespace
from dataclasses import replace

import pytest

from emmy.commands.golden import handle_golden_check, handle_golden_restamp
from emmy.compiler.pipeline.search.golden import GoldenFile
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


def _rename_output(document: GoldenFile, index: int) -> GoldenFile:
    """Program ``index``'s traced output under another name: its fresh kernel writes an output set no stored kernel names."""
    program = {**document.programs[index], "outputs": ["product"], "nodes": [dict(node) for node in document.programs[index]["nodes"]]}
    for node in program["nodes"]:
        if node["id"] == "matmul":
            node["id"] = "product"
            node["outputs"] = [["product", *node["outputs"][0][1:]]]
    programs = [program if i == index else stored for i, stored in enumerate(document.programs)]
    kernels = [replace(kernel, origins=("product",)) if kernel.traced == index else kernel for kernel in document.kernels]
    return replace(document, programs=programs, kernels=kernels)


def _shrink(document: GoldenFile, index: int) -> GoldenFile:
    """Program ``index`` at half its sizes: the fresh kernel writes the stored output set from another body."""
    program = {**document.programs[index], "nodes": [dict(node) for node in document.programs[index]["nodes"]]}
    for node in program["nodes"]:
        node["outputs"] = [[name, dtype, [dim // 2 for dim in dims]] for name, dtype, dims in node["outputs"]]
    return replace(document, programs=[program if i == index else stored for i, stored in enumerate(document.programs)])


def _make_stale(path) -> None:
    document = GoldenFile.load(path)
    _rename_output(_shrink(document, 0), 2).dump(path, overwrite=True)


def test_check_passes_a_current_golden_and_names_what_a_restamp_changes(golden, caplog):
    assert _check(golden) == 0
    _make_stale(golden)
    with caplog.at_level("ERROR"):
        assert _check(golden) == 1
    assert "not the fresh lowering" in caplog.text
    assert "re-keyed k_matmul" in caplog.text and "no fresh kernel writes its outputs" in caplog.text
    assert "demoted to a proposal matmul.square.512" in caplog.text


def test_restamp_rewrites_the_golden_onto_the_fresh_lowering(golden, caplog):
    _make_stale(golden)
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert _check(golden) == 0, "the restamped file is the fresh lowering"

    document = GoldenFile.load(golden)
    names = {row.name for row in document.rows}
    assert "matmul.square.2048" not in names, "a target no fresh kernel writes is dropped with its rows"
    assert len(document.kernels) == 7 and len(document.programs) == 8, "the kernel goes; the traced program is provenance"
    proposals = {row.name for row in document.rows if not row.measured}
    assert proposals == {"matmul.square.512"}, "measured on another kernel: the row keeps its schedule only"
    assert "1 of 8 kernels re-keyed, 1 dropped" in caplog.text

    caplog.clear()
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "already the fresh lowering" in caplog.text


def test_restamp_takes_a_decision_again_on_the_re_keyed_parent(tmp_path, caplog):
    """A routing row's parent re-keyed by a program change: the decision is taken again on the fresh parent, the
    pieces take the identities the fresh splice mints, and their rows follow — each row keeps its measurement only
    while its kernel kept its body."""
    document = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.json")
    route = next(route for route in document.routing if document.kernel(route.parent).traced is not None)
    parent = document.kernel(route.parent)
    stale = _shrink(document, parent.traced)
    path = tmp_path / "golden.json"
    stale.dump(path, overwrite=True)
    assert _check(path) == 1
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(path)]))
    fresh = GoldenFile.load(path)
    assert _check(path) == 0
    assert len(fresh.routing) == len(document.routing) and len(fresh.kernels) == len(document.kernels)
    moved = {kernel.ref for kernel in document.kernels if fresh.kernel(kernel.ref).exact_identity != kernel.exact_identity}
    assert route.parent in moved and set(route.children) <= moved, "the parent and the pieces it mints are re-keyed"
    assert {kernel.ref for kernel in fresh.kernels} == {kernel.ref for kernel in document.kernels}, "a re-keyed kernel keeps its ref"
    assert f"re-keyed {parent.name}" in caplog.text
    assert len(fresh.rows) == len(document.rows)


def test_restamp_drops_a_decision_the_fresh_parent_does_not_take(golden, caplog):
    """A decision whose arm names no seam of the fresh parent mints nothing: it is dropped with the rows of its pieces."""
    from emmy.compiler.pipeline.search.db import RoutingRow
    from emmy.compiler.pipeline.search.golden import Row

    document = GoldenFile.load(golden)
    parent = document.kernels[0]
    orphan = replace(document.kernels[1], key="orphan", traced=None, origins=(), bindings={})
    document.kernels.append(orphan)
    document.routing.append(RoutingRow(parent.ref, {"PLACE@missing": "cut"}, (orphan.ref,)))
    document.rows.append(Row(name="orphan", kernel=orphan.ref, pins={"FAST_MATH": False}, knobs={"WORK": "t16x8"}))
    document.dump(golden, overwrite=True)
    with caplog.at_level("INFO"):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "dropped decision" in caplog.text
    fresh = GoldenFile.load(golden)
    assert fresh.routing == [] and len(fresh.kernels) == len(document.kernels) - 1
    assert all(row.kernel != orphan.ref for row in fresh.rows), "the piece's rows go with the decision"


def test_restamp_refuses_to_write_a_golden_nothing_survives_in(golden, caplog):
    document = GoldenFile.load(golden)
    for index in range(len(document.programs)):
        document = _rename_output(document, index)
    document.dump(golden, overwrite=True)
    before = golden.read_bytes()
    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        handle_golden_restamp(Namespace(paths=[str(golden)]))
    assert "no kernel survives" in caplog.text
    assert golden.read_bytes() == before, "deleting or re-recording the file is a decision, not a restamp"


def test_list_reads_each_measured_row_beside_torch_compile(golden, capsys):
    from emmy.commands.golden import handle_golden_list
    from emmy.compiler.pipeline.search.golden import Latency

    with GoldenFile.edit(golden) as document:
        first, second = document.rows[0], document.rows[1]
        document.rows[0] = replace(
            first, latency={"NVIDIA GeForce RTX 4080": Latency(emmy_us=30.0, tcompile_us=10.0)}, note="needs a wider tile"
        )
        document.rows[1] = replace(second, measurements=replace(second.measurements, tried=12))

    def listed(**options) -> list[dict]:
        handle_golden_list(
            Namespace(
                **{"paths": [str(golden)], "gpu": None, "kernel": None, "behind": False, "missing": False, "json_out": "-", **options}
            )
        )
        return json.loads(capsys.readouterr().out)

    entries = listed()
    assert len(entries) == len(GoldenFile.load(golden).rows), "one entry per measured row, the timed card merged in"
    assert entries[0]["row"] == first.name and entries[0]["vs_tcompile"] == 3.0 and entries[0]["note"] == "needs a wider tile"
    assert next(entry for entry in entries if entry["row"] == second.name)["tried"] == 12
    assert [entry["row"] for entry in listed(behind=True)] == [first.name]
    assert listed(gpu="H100") == []


def test_list_missing_names_each_proposal_and_each_target_with_no_torch_compile_time(golden, capsys):
    from emmy.commands.golden import handle_golden_list
    from emmy.compiler.pipeline.search.golden import Latency

    with GoldenFile.edit(golden) as document:
        proposal, timed = document.rows[0], document.rows[1]
        document.rows[0] = replace(proposal, measurements=None)
        document.rows[1] = replace(timed, latency={"NVIDIA GeForce RTX 4080": Latency(emmy_us=3.0, tcompile_us=2.0)})
    targets = len(GoldenFile.load(golden).target_rows())

    handle_golden_list(Namespace(paths=[str(golden)], gpu=None, kernel=None, behind=False, missing=True, json_out="-"))
    entries = json.loads(capsys.readouterr().out)

    assert [(entry["row"], entry["missing"]) for entry in entries if entry["missing"] == "emmy"] == [(proposal.name, "emmy")]
    tcompile = {entry["row"] for entry in entries if entry["missing"] == "tcompile"}
    assert len(tcompile) == targets - 2, "the proposal's target is measured by its record, the timed target needs nothing"
    assert proposal.name not in tcompile and timed.name not in tcompile
