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
from emmy.compiler.pipeline.search.db import RoutingRow
from emmy.compiler.pipeline.search.golden import GoldenFile, restamp
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


def _one_program(document: GoldenFile, index: int) -> GoldenFile:
    """Program ``index`` alone, as program 0: its target kernels, the decisions below them and their rows."""
    targets = [replace(kernel, traced=0) for kernel in document.targets() if kernel.traced == index]
    scope = {kernel.ref for kernel in targets}
    routing = []
    for route in document.routing:
        if route.parent in scope:
            routing.append(route)
            scope.update(route.children)
    pieces = [kernel for kernel in document.kernels if kernel.ref in scope and kernel.traced is None]
    rows = [row for row in document.rows if row.kernel in scope]
    return replace(document, programs=[document.programs[index]], kernels=targets + pieces, routing=routing, rows=rows)


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
    full = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.json")
    route = next(route for route in full.routing if full.kernel(route.parent).traced is not None)
    document = _one_program(full, full.kernel(route.parent).traced)
    parent = document.kernel(route.parent)
    stale = _shrink(document, 0)
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


def _beside_its_half(document: GoldenFile) -> GoldenFile:
    """``document``'s one program, and as traced program 1 the same program at half its sizes (``_shrink``): both
    current, the second's kernels, decisions and rows those of a restamp of the shrunk copy, each ref and row name
    suffixed ``@half``."""
    half, _ = restamp(_shrink(document, 0))

    def ref(name: str) -> str:
        return f"{name}@half"

    kernels = [replace(kernel, key=ref(kernel.ref), traced=None if kernel.traced is None else 1) for kernel in half.kernels]
    routing = [RoutingRow(ref(route.parent), route.arm, tuple(ref(child) for child in route.children)) for route in half.routing]
    rows = [replace(row, name=ref(row.name), kernel=ref(row.kernel)) for row in half.rows]
    return replace(
        document,
        programs=[*document.programs, *half.programs],
        kernels=[*document.kernels, *kernels],
        routing=[*document.routing, *routing],
        rows=[*document.rows, *rows],
    )


@pytest.mark.parametrize("half_first", [False, True])
def test_restamp_splits_a_piece_two_decisions_mint_as_two_kernels(half_first):
    """Two programs' decisions name one stored piece, but the fresh lowering mints it as two kernels: the piece splits
    in two, each decision naming the kernel it mints, and the piece's measured row stays with the decision whose kernel
    it measured — whatever order the file stores the decisions in. A whole-file restamp and a restamp of each program
    alone agree on which program's kernel set is stale."""
    full = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.json")
    route = next(route for route in full.routing if full.kernel(route.parent).traced is not None)
    one = _one_program(full, full.kernel(route.parent).traced)
    piece = route.children[0]
    measured = next(row for row in one.rows if row.kernel == route.parent and row.measured)
    current = _beside_its_half(replace(one, rows=[*one.rows, replace(measured, name="piece", kernel=piece)]))
    assert restamp(current)[0] == current
    assert current.kernel(f"{piece}@half").exact_identity != current.kernel(piece).exact_identity

    # The file a lowering that minted one kernel for both programs' piece wrote: program 1's decision names program 0's.
    halved = next(stored for stored in current.routing if stored.parent == f"{route.parent}@half")
    shared = replace(halved, children=tuple(piece if child == f"{piece}@half" else child for child in halved.children))
    routing = [stored for stored in current.routing if stored != halved]
    stale = replace(
        current,
        kernels=[kernel for kernel in current.kernels if kernel.ref != f"{piece}@half"],
        routing=[shared, *routing] if half_first else [*routing, shared],
        rows=[row for row in current.rows if row.kernel != f"{piece}@half"],
    )

    fresh, report = restamp(stale)
    [split] = {kernel.ref for kernel in fresh.kernels} - {kernel.ref for kernel in stale.kernels}
    assert any(line.startswith(f"split shared piece {piece} ") and split in line for line in report.lines()), report.lines()
    assert fresh.kernel(split).exact_identity == current.kernel(f"{piece}@half").exact_identity
    assert fresh.kernel(piece) == stale.kernel(piece), "the decision that still mints the stored kernel keeps it"
    children = {stored.parent: stored.children for stored in fresh.routing}
    assert piece in children[route.parent] and split in children[f"{route.parent}@half"]
    assert next(row for row in fresh.rows if row.name == "piece") == next(row for row in stale.rows if row.name == "piece")

    assert restamp(stale, traced=0)[0] == stale, "program 0's kernel set is current"
    assert restamp(stale, traced=1)[1].changed, "program 1's decision mints another kernel"
    for traced in (None, 0, 1):
        assert restamp(fresh, traced=traced)[0] == fresh, "the split file is current, as a whole and per program"


def _measured_pieces() -> tuple[GoldenFile, RoutingRow]:
    """One program of the RTX 5090 golden with a cross-CTA split under its target, and a measured row on each piece."""
    full = GoldenFile.load(_RECORDS_DIR / "rtx5090_sm120.json")
    route = next(route for route in full.routing if full.kernel(route.parent).traced is not None)
    one = _one_program(full, full.kernel(route.parent).traced)
    measured = next(row for row in one.rows if row.kernel == route.parent and row.measured)
    rows = [replace(measured, name=f"piece.{position}", kernel=child) for position, child in enumerate(route.children)]
    return replace(one, rows=[*one.rows, *rows]), route


def test_restamp_matches_a_decision_s_pieces_by_identity_before_position():
    """A piece is the stored piece of its exact identity wherever the fresh decision orders it: pieces stored in
    another order are put in the fresh order and keep their bodies and measurements. A stored piece no fresh piece
    is loses its rows — they measured a kernel the decision no longer mints — and the fresh piece no stored one is
    joins the file with none."""
    current, route = _measured_pieces()
    assert restamp(current)[0] == current
    first, second = route.children

    def stored(children: tuple[str, ...], kernels) -> GoldenFile:
        routing = [replace(stored, children=children) if stored == route else stored for stored in current.routing]
        return replace(current, kernels=kernels, routing=routing)

    swapped = stored((second, first), current.kernels)
    fresh, report = restamp(swapped)
    assert fresh == current, report.lines()
    assert report.reordered and not report.rekeyed and not report.demoted

    parent = current.kernel(route.parent)
    other = replace(current.kernel(second), loop_ir=parent.loop_ir, formed=parent.formed)  # a body the decision never mints
    stale = stored((second, first), [other if kernel.ref == second else kernel for kernel in current.kernels])
    fresh, report = restamp(stale)
    [new] = report.added
    [added] = {kernel.ref for kernel in fresh.kernels} - {kernel.ref for kernel in stale.kernels}
    assert new.startswith(added) and fresh.kernel(added).exact_identity == current.kernel(second).exact_identity
    assert next(r for r in fresh.routing if r.parent == route.parent).children == (first, added)
    assert second not in {kernel.ref for kernel in fresh.kernels}
    assert {row.name for row in stale.rows} - {row.name for row in fresh.rows} == {"piece.1"}, "dropped, not re-attached"
    assert next(row for row in fresh.rows if row.name == "piece.0").measured, "the piece of the same identity keeps its measurement"


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
