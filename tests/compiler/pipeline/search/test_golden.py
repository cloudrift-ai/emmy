"""The checked-in golden corpus is current, and it imports whole.

Every repository golden — the model-agnostic hardware goldens and each recipe's model golden — is held to the fresh
lowering of its own traced programs: a restamp (``emmy golden restamp``, the one rewrite a stale golden gets) must
leave the file unchanged. One node per traced program, so the work scatters over the workers instead of queueing
behind the widest file, and a failure names the kernels, decisions and rows the compiler now disagrees with. There
is no list of expected failures: a stale golden is red until the restamp rewrites it, which needs no card. The
import — a copy of the file's tables into a DB — files every measured row.
"""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search import golden
from emmy.compiler.pipeline.search.golden import GoldenFile, restamp, scope_digest
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, document_of, repository_golden_paths


def _golden_id(path: Path) -> str:
    """A golden file's id: its name for a hardware golden, ``<recipe>/<name>`` for a model golden."""
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


def _program_parameters():
    parameters = []
    with repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            for index in sorted({kernel.traced for kernel in document_of(path).targets()}):
                parameters.append(pytest.param(path, index, id=f"{_golden_id(path)}/program-{index}"))
    return parameters


@pytest.mark.parametrize(("path", "traced"), _program_parameters())
def test_a_repository_golden_is_the_fresh_lowering(path: Path, traced: int) -> None:
    """A restamp onto the fresh lowering of one traced program changes nothing: every target kernel keeps its
    identity, stamps and body, every decision taken on one is taken the same way and mints the same pieces, every row
    keeps its measurement. Lowering is GPU-free, so this holds on any machine; a card is needed to re-record a stale
    row, not to detect one."""
    document = document_of(path)
    fresh, report = restamp(document, traced=traced)
    assert fresh == document, "\n".join(
        report.lines() if report.changed else ["the kernels' stamps or bodies are not the fresh lowering's"]
    )


def _file_parameters():
    with repository_golden_paths() as paths:
        return [pytest.param(path, id=_golden_id(path)) for path in sorted(paths, key=_golden_id)]


@pytest.mark.parametrize("path", _file_parameters())
def test_every_measured_row_imports(path: Path) -> None:
    """The import is a copy: every kernel, every decision and every measured row of a repository golden lands in the
    DB it is imported into, so a compile that reads the file deploys from all of it."""
    from emmy.compiler.pipeline.search.db import SearchDB
    from emmy.compiler.pipeline.search.golden import file_source, import_file

    document = document_of(path)
    db = SearchDB()
    counts = import_file(db, path, file_source("golden", path))
    assert counts["perf rows"] == sum(row.measured for row in document.rows)
    assert sum(1 for _ in db.iter_kernels()) == len(document.kernels)
    assert sum(1 for _ in db.iter_routing()) == len(document.routing)
    assert not any(db.drift().values())


def test_scope_digest_follows_the_cards_rows_only(tmp_path, monkeypatch) -> None:
    """The digest a serving pack keys on moves with the rows this card's compile reads and with nothing else: another
    card's file, or a file scope that names a different file."""
    mine = tmp_path / "mine.json"
    other = tmp_path / "other.json"
    mine.write_text('{"gpu_name": "NVIDIA H100 80GB HBM3",\n "rows": 1}\n')
    other.write_text('{"gpu_name": "NVIDIA GeForce RTX 5090",\n "rows": 1}\n')
    monkeypatch.setattr(golden.repository, "repository_golden_paths", lambda: nullcontext([mine, other]))
    monkeypatch.delenv("EMMY_GOLDEN_FILE", raising=False)
    card = "NVIDIA H100 80GB"
    base = scope_digest(card)
    other.write_text('{"gpu_name": "NVIDIA GeForce RTX 5090",\n "rows": 2}\n')
    assert scope_digest(card) == base
    mine.write_text('{"gpu_name": "NVIDIA H100 80GB HBM3",\n "rows": 2}\n')
    changed = scope_digest(card)
    assert changed != base
    monkeypatch.setenv("EMMY_GOLDEN_FILE", str(other))
    scoped = scope_digest(card)
    assert scoped not in (base, changed)
    monkeypatch.setenv("EMMY_GOLDEN_FILE", "")
    assert scope_digest(card) not in (base, changed, scoped)


def test_a_scope_reads_a_file_only_on_the_card_that_measured_it(caplog) -> None:
    """An explicit scope reads a file only on the card that measured it, or everywhere when the file names no card,
    and says which files it drops: a working golden seeded from another card's file and recorded here kept every row
    under that card, and its replay built the fused kernel with nothing said."""
    sxm2, sxm3 = "NVIDIA Tesla V100 SXM2 16GB", "NVIDIA Tesla V100 SXM3 32GB"
    documents = [GoldenFile(gpu_name=name, compute_cap=(7, 0)) for name in (sxm2, sxm3, None)]
    with golden.evidence_scope(documents), caplog.at_level("WARNING"):
        kept = golden.documents_for_card(sxm3, (7, 0))
        assert golden.documents_for_card(sxm3, (8, 0)) == []
    assert kept == documents[1:]
    assert f"measured on {sxm2} are no evidence on {sxm3}" in caplog.text


def test_restamp_drops_what_the_fresh_lowering_no_longer_writes(tmp_path) -> None:
    """A target no fresh kernel writes is dropped with its rows; a kernel whose body changed keeps its rows as
    proposals; a decision whose parent is gone goes with it. Staged on the smallest hardware golden."""
    document = GoldenFile.load(_RECORDS_DIR / "rtx4080_sm89.json")
    programs = [dict(program) for program in document.programs]
    # Program 2's traced output takes another name: no fresh kernel writes the stored output set.
    renamed = {**programs[2], "outputs": ["product"], "nodes": [dict(node) for node in programs[2]["nodes"]]}
    for node in renamed["nodes"]:
        if node["id"] == "matmul":
            node["id"] = "product"
            node["outputs"] = [["product", *node["outputs"][0][1:]]]
    programs[2] = renamed
    # Program 0 shrinks: the fresh kernel writes the same output set from another body.
    shrunk = {**programs[0], "nodes": [dict(node) for node in programs[0]["nodes"]]}
    for node in shrunk["nodes"]:
        node["outputs"] = [[name, dtype, [dim // 2 for dim in dims]] for name, dtype, dims in node["outputs"]]
    programs[0] = shrunk
    stale = replace(
        document, programs=programs, kernels=[replace(k, origins=("product",) if k.traced == 2 else k.origins) for k in document.kernels]
    )

    fresh, report = restamp(stale)
    gone = [kernel for kernel in document.kernels if kernel.traced == 2]
    moved = [kernel for kernel in document.kernels if kernel.traced == 0]
    assert len(report.dropped_kernels) == 1 and gone[0].name in report.dropped_kernels[0]
    assert len(report.rekeyed) == 1 and moved[0].name in report.rekeyed[0]
    assert len(fresh.kernels) == len(document.kernels) - 1
    assert [row.name for row in document.rows if row.kernel == gone[0].exact_identity] == [
        reason.split(":")[0] for reason in report.dropped_rows
    ]
    demoted = [row for row in fresh.rows if not row.measured]
    assert [row.name for row in demoted] == report.demoted and len(demoted) == sum(
        row.kernel == moved[0].exact_identity for row in document.rows
    )
    assert all(row.knobs for row in demoted), "a demoted row keeps its schedule"
    again, report2 = restamp(fresh)
    assert again == fresh and not report2.changed, "a restamped file is current"
    fresh.dump(tmp_path / "golden.json", repository=True)
