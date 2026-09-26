"""Strict decode of the checked-in golden corpus.

A recorded row is evidence a deploy can use only while it still equals an enumerated leaf of its
own target. Every repository golden — the model-agnostic hardware goldens and each recipe's model
golden — is asked that ROW BY ROW, on the default lane, so the nodes scatter over the workers
instead of queueing behind the widest file, and a failure names the row instead of a count. There
is no list of expected failures: a row that stops decoding, or a file whose stored targets stop being
the fresh lowering, is red until ``emmy golden restamp`` rewrites the file, which needs no card.
"""

import difflib
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search import golden
from emmy.compiler.pipeline.search.golden import decode_record, scope_digest, siblings_of
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, _records_of, _repository_golden_paths


def _decode(record, records) -> str | None:
    """The strict verdict for one row — ``None`` when it still decodes, else the reason."""
    try:
        return decode_record(record, siblings_of(record, records))
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        return f"{type(exc).__name__}: {exc}"


def _labels(records) -> list[str]:
    """One stable label per record, in file order: the row's name, numbered when the file records
    that name more than once — alternate input pins or alternate schedules of one realization."""
    total = Counter(record.name for record in records)
    seen: Counter = Counter()
    labels = []
    for record in records:
        seen[record.name] += 1
        labels.append(record.name if total[record.name] == 1 else f"{record.name}#{seen[record.name]}")
    return labels


def _golden_id(path: Path) -> str:
    """A golden file's id: its name for a hardware golden, ``<recipe>/<name>`` for a model golden."""
    return path.name if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.name}"


def _row_parameters():
    """One parameter per recorded row of every repository golden."""
    parameters = []
    with _repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            file_id = _golden_id(path)
            for label in _labels(_records_of(path)):
                parameters.append(pytest.param(path, label, id=f"{file_id}/{label}"))
    return parameters


@pytest.mark.parametrize(("path", "label"), _row_parameters())
def test_recorded_row_decodes(path: Path, label: str) -> None:
    """One row of a repository golden must still equal an enumerated leaf of its own target.

    A row that does not is no evidence a deploy can use: the compile that reads it either picks a
    schedule the enumeration no longer offers, or falls through to the prior. Decoding replays the
    persisted program at the record's DECLARED capability, so this holds on any machine — a card is
    needed to re-record a stale row, not to detect one.
    """
    records = _records_of(path)
    record = records[_labels(records).index(label)]
    assert (reason := _decode(record, records)) is None, reason


def test_scope_digest_follows_the_cards_rows_only(tmp_path, monkeypatch) -> None:
    """The digest a serving pack keys on moves with the rows this card's compile reads and with nothing else: another
    card's file, or a file scope that names a different file."""
    mine = tmp_path / "mine.json"
    other = tmp_path / "other.json"
    mine.write_text('{"gpu_name": "NVIDIA H100 80GB HBM3",\n "rows": 1}\n')
    other.write_text('{"gpu_name": "NVIDIA GeForce RTX 5090",\n "rows": 1}\n')
    monkeypatch.setattr(golden.repository, "_repository_golden_paths", lambda: nullcontext([mine, other]))
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


def test_decode_ignores_off_anchors_but_not_a_decided_value() -> None:
    """A row's OFF anchors are not part of what it is compared by, and its decided values are.

    Which anchors a spelling writes down says where it came from, not what it decided: a resolved
    kernel carries every declared OFF value because the pipeline stamps them at the pass boundary,
    while a fork offers its leaves carrying only the families the kernel's sites give it. Both spell
    the same schedule. Requiring them to agree is what left two thirds of the recorded rows matching
    nothing, so the decode compares them blind to the anchors — and stays strict about every value a
    row actually decided.
    """
    from dataclasses import replace

    # The smallest target on the card, named rather than searched for: every assertion below decodes
    # the record again, so the row this stands on decides what the test costs.
    records = _records_of(_RECORDS_DIR / "rtx5090_sm120.json")
    named = [r for r in records if r.name == "matmul.square.512" and any(v == "" for v in r.knobs.values())]
    record = next((r for r in named if _decode(r, records) is None), None)
    assert record is not None, "matmul.square.512 records no decoding row carrying an OFF anchor to compare against"

    stripped = replace(record, knobs={key: value for key, value in record.knobs.items() if value != ""})
    assert _decode(stripped, records) is None, "a row must decode without the anchors a fork's leaf would not spell"

    anchored = replace(record, knobs={"STAGE": "", "REDUCE": "", "RASTER": "", **record.knobs})
    assert _decode(anchored, records) is None, "and with the anchors a resolved kernel would be stamped with"

    decided = next(key for key, value in record.knobs.items() if value not in ("", "0"))
    assert _decode(replace(record, knobs={**record.knobs, decided: "not-a-real-value"}), records) is not None


def test_a_pool_that_holds_the_recorded_row_is_not_walked_whole(monkeypatch) -> None:
    """The strict decode asks one question of a schedule pool, and pays for one answer.

    A recorded row decodes through the schedule's codec and compatibility context, instead of
    keying every leaf of a pool that runs to millions on a fused attention target. A row that names
    a site outside the codec is rejected before decode; another miss keeps only the requested keys
    and values needed to explain it.
    """
    from dataclasses import replace

    from emmy.compiler.pipeline import fork
    from emmy.compiler.pipeline.knob import schedule_match_key
    from emmy.compiler.pipeline.search.golden import piece_row, unmatched_reason
    from emmy.compiler.pipeline.search.golden.decode import _replay, _unmatched_reason

    # The same smallest target the anchor test stands on: every assertion here replays it.
    records = _records_of(_RECORDS_DIR / "rtx5090_sm120.json")
    record = next(r for r in records if r.name == "matmul.square.512" and _decode(r, records) is None)
    siblings = siblings_of(record, records)

    def rows(entry, wanted=None):
        return frozenset().union(*_replay(entry, siblings=siblings, exhaustive=True, wanted=wanted).rows.values())

    wanted = schedule_match_key(piece_row(record.knobs))
    assert wanted in rows(record), "the whole pool holds the recorded row"

    old_leaf_for = fork.leaf_for

    def reject_schedule_descent(options, row, *, skip=None):
        assert not any(option.pool_id is not None for option in options), "an exact schedule row must use the codec"
        return old_leaf_for(options, row, skip=skip)

    monkeypatch.setattr(fork, "leaf_for", reject_schedule_descent)
    assert wanted in rows(record, wanted), "and asking for that one row still finds it"
    assert len(rows(record, wanted)) < len(rows(record)), "having filed it without keying the pool's every leaf"

    empty = replace(record, knobs={key: "" for key in record.knobs})
    assert () in rows(empty, ()), "an empty requested row still finds a nonstructural empty leaf"

    decided = next(key for key, value in record.knobs.items() if value not in ("", "0"))
    missing = replace(record, knobs={**record.knobs, decided: "not-a-real-value"})
    absent = schedule_match_key(piece_row(missing.knobs))
    assert absent not in rows(missing), "a row no leaf spells equals nothing in the pool"
    full = rows(missing)
    miss = _replay(missing, siblings=siblings, exhaustive=True, wanted=absent, explain=True)
    keys = set().union(*(summary[0] for summary in miss.offered.values()))
    pairs = set().union(*(summary[1] for summary in miss.offered.values()))
    assert not miss.rows, "a miss retains no candidate rows"
    assert _unmatched_reason(absent, keys, pairs) == unmatched_reason(absent, full)

    respelled = replace(record, knobs={**record.knobs, "WORK@missing": record.knobs["WORK"]})
    reason = _decode(respelled, records)
    assert reason is not None and "WORK@missing" in reason and "re-spelling" in reason


def test_a_row_with_only_an_invalid_offered_value_reports_a_semantic_miss() -> None:
    """An offered key whose requested value is invalid is a normal decode miss, not an error.

    The replay tracks offered keys separately from validated key/value pairs.  When every requested
    value is invalid, that second set is deliberately empty; the diagnostic still has enough
    information to report narrowing without indexing a pair set that was never populated.
    """
    from dataclasses import replace

    records = _records_of(_RECORDS_DIR / "rtx5090_sm120.json")
    record = next(r for r in records if r.name == "matmul.square.512" and _decode(r, records) is None)
    decided = next(key for key, value in record.knobs.items() if value not in ("", "0"))
    invalid = replace(record, knobs={decided: "not-a-real-value"})

    reason = decode_record(invalid, siblings_of(invalid, records))
    assert reason is not None and "NARROWING" in reason and decided in reason


def test_a_row_whose_every_site_is_re_spelled_still_gets_a_verdict() -> None:
    """A row can lose every key it decided at once — an identity re-key moves a kernel's sites, and
    the codec then declares none of the keys the recording spelled. That is a re-spelling like any
    other and the decode has to say so. Two expert rows of the DeepSeek V4 golden raised instead
    after #804 re-keyed the file: the replay filed the kernel's declared keys with no offered pair
    behind them, then indexed the pair it never filed."""
    from dataclasses import replace

    records = _records_of(_RECORDS_DIR / "rtx5090_sm120.json")
    record = next(r for r in records if r.name == "matmul.square.512" and _decode(r, records) is None)
    respelled = replace(record, knobs={f"{key}@missing": value for key, value in record.knobs.items() if value not in ("", "0")})
    reason = _decode(respelled, records)
    assert reason is not None and "re-spelling" in reason and "@missing" in reason, reason


def test_a_sibling_sharing_the_target_identity_cannot_silence_the_lead_cut() -> None:
    """A receipt decodes behind its lead's cut. The replay finds the entry that decides a fork by
    the fork root's identity, so a plain receipt stamped with its lead's identity — what the #804
    re-key did to 145 rows of the DeepSeek V4 golden — stands in for the lead at the cut fork,
    fuses the kernel, and every receipt of the set reads as a re-spelling. Where entries share an
    identity, the one that spells a route decides the cut. The set is the golden's own, the M=16
    pre-attention statistic: a cut lead, a cross-CTA split of one piece, and three receipts."""
    from dataclasses import replace

    from emmy.compiler.pipeline.knob import schedule_match_key
    from emmy.compiler.pipeline.search.golden import piece_row
    from emmy.compiler.pipeline.search.golden.decode import _replay

    def decodes(record, siblings) -> bool:  # the replay itself: the decode's verdict is memoized per record
        wanted = schedule_match_key(piece_row(record.knobs))
        return any(_replay(record, siblings=siblings, exhaustive=True, wanted=wanted).rows.values())

    golden = Path(__file__).parents[4] / "recipes" / "DeepSeek-V4-Flash-0731" / "golden" / "v100_sm70.json"
    records = _records_of(golden)
    lead = next(r for r in records if r.name == "pre16.k_linear_mean_reduce_03c479.8caa25e24052.m16.dc6db94ec8ea.dc6db94ec8ea")
    siblings = siblings_of(lead, records)
    receipt = next(m for m in siblings if m.name.endswith(".5d9b14249e94"))
    assert decodes(receipt, [lead, *(m for m in siblings if m is not receipt)]), "the receipt decodes behind its lead's cut"

    other = next(m for m in siblings if m.name.endswith(".c607711d8ef8"))
    impostor = replace(other, identity=lead.identity)
    beside = [lead, *(impostor if m is other else m for m in siblings if m is not receipt)]
    assert decodes(receipt, beside), "and beside a receipt stamped with the lead's identity"


def test_a_red_row_says_which_of_the_three_kinds_of_churn_moved_it() -> None:
    """A compiler change moves recorded rows in bulk and only one reading of that is a loss, so the
    decode has to say which. Before this the reason was a candidate COUNT, and telling a re-spelling
    from a narrowing meant writing a probe: the Qwen family's 126 red rows across three goldens were
    classified by hand before any of them could be re-recorded."""
    from emmy.compiler.pipeline.search.golden import unmatched_reason

    offered = frozenset({(("WORK", "t128"), ("TILE", "f4")), (("WORK", "t256"), ("TILE", "f2"))})

    respelled = unmatched_reason(((("REDUCE@reduce"), "coop"), ("WORK", "t128")), offered)
    assert "re-spelling or an identity change" in respelled and "REDUCE@reduce" in respelled

    narrowed = unmatched_reason((("WORK", "t512"), ("TILE", "f4")), offered)
    assert narrowed.startswith("NARROWING") and "WORK='t512'" in narrowed

    regrouped = unmatched_reason((("WORK", "t128"), ("TILE", "f2")), offered)
    assert "no one candidate carries them together" in regrouped

    gained = unmatched_reason((), offered)
    assert "takes a schedule where the recording spelled none" in gained


def test_the_narrowing_reading_outranks_the_respelling_one() -> None:
    """A row can lose a key AND a value at once. The re-spelling reading is reported first because
    a key that is gone explains the value that went with it, and calling that a narrowing would
    report a lost capability that was only renamed."""
    from emmy.compiler.pipeline.search.golden import unmatched_reason

    offered = frozenset({(("WORK", "t128"),)})
    both = unmatched_reason(((("TILE"), "f4"), ("WORK", "t512")), offered)
    assert "re-spelling" in both and "NARROWING" not in both


def _kernel_parameters():
    """One parameter per stored kernel of every repository golden, named by the first row that targets it."""
    from emmy.compiler.pipeline.search.golden.repository import _document_of

    parameters = []
    with _repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            document, _ = _document_of(path)
            for index in range(len(document.loops)):
                name = next((entry.realizations[0].name for entry in document.configs if entry.target.loop == index), f"loop-{index}")
                parameters.append(pytest.param(path, index, id=f"{_golden_id(path)}/{name}"))
    return parameters


@pytest.mark.parametrize(("path", "index"), _kernel_parameters())
def test_stored_kernel_is_a_fixed_point_of_normalization(path: Path, index: int) -> None:
    """A stored kernel must come back byte for byte from a decode: decoding builds a Loop op, whose
    constructor normalizes the body, so a kernel that decodes to different wire is one the current
    normalizer spells differently. A load keeps the pools as wires and never decodes, and this is
    the premise that lets it: until ``emmy golden restamp`` rewrites such a kernel, every consumer
    reads a body the compiler would not produce, and its rows are evidence for a kernel that does
    not exist. Cheaper than the fresh-lowering test above, which lowers the whole program, and
    narrower: it says whether the normalizer moved, not whether the lowering did."""
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline.search.golden import kernel_pool_text
    from emmy.compiler.pipeline.search.golden.repository import _document_of

    document, _ = _document_of(path)
    stored = document.loops[index]
    rewritten = Graph.from_wire(stored).to_wire()
    diff = difflib.unified_diff(
        kernel_pool_text([stored]).splitlines(), kernel_pool_text([rewritten]).splitlines(), "stored", "normalized", lineterm=""
    )
    assert rewritten == stored, "\n".join(diff)


def _golden_parameters():
    parameters = []
    with _repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            file_id = _golden_id(path)
            for index in sorted({record.program_index for record in _records_of(path)}):
                parameters.append(pytest.param(path, index, id=f"{file_id}/program-{index}"))
    return parameters


@pytest.mark.parametrize(("path", "program"), _golden_parameters())
def test_stored_targets_are_the_fresh_lowering(path: Path, program: int) -> None:
    """Every stored target of a repository golden must be a kernel the current compiler lowers the
    golden's own traced program to — byte for byte: per traced program, ``emmy golden kernels PATH
    --program N`` against ``emmy compile --golden PATH --program N --ir loop -o fresh.json``, restricted to
    the targets the file stores (``emmy golden check``; ``emmy golden restamp`` is the fix).

    A golden's rows are evidence for the kernels its stored Loop IR names, and a deploy keys them by
    the kernels it lowers FRESH from the model. The decode test above replays the stored target, so
    it stays green when the two drift apart: after #863 the DeepSeek V4 V100 golden decoded row by
    row while serving lowered kernels no row of it described and the strict boot refused. Lowering
    is the loop passes alone, GPU-free, so this holds on any machine; one node per program, so a
    whole-layer trace costs its own minutes and nothing queues behind the widest file.
    """
    from emmy.compiler.pipeline.search.golden import kernel_pool_text
    from emmy.compiler.pipeline.search.golden.repository import _document_of
    from emmy.compiler.pipeline.search.restamp import fresh_kernel_digests, fresh_kernels, stale_reasons

    document, _ = _document_of(path)
    stale = stale_reasons(document, program, fresh_kernel_digests(document, program))
    if stale:
        fresh = fresh_kernels(document, [program])[program]
        stored = document.kernels(program)
        matched = [fresh[frozenset(kernel["outputs"])] for kernel in stored if frozenset(kernel["outputs"]) in fresh]
        diff = difflib.unified_diff(
            kernel_pool_text(stored).splitlines(),
            kernel_pool_text(matched).splitlines(),
            f"emmy golden kernels {path} --program {program}",
            f"emmy compile --golden {path} --program {program} --ir loop -o fresh.json",
            lineterm="",
        )
        pytest.fail("targets not the fresh lowering:\n  " + "\n  ".join(stale) + "\n" + "\n".join(diff), pytrace=False)


def test_a_stored_identity_the_compiler_re_keyed_is_refused() -> None:
    """A row's stored identity must be one kernel the replay resolves under its pins — the target's own
    as much as a receipt's child. A compiler change that re-keys the kernel turns the row red on the
    commit that causes it; re-keying the file is the fix, and no import stands in for it meanwhile."""
    from dataclasses import replace

    from tests.compiler.realization import helpers as corpus

    case = corpus.load_case(corpus.CASES_DIR / "fused/norm-linear-f16-scalar-reduce.json")
    record = case.record
    assert _decode(record, case.records) is None
    stale = replace(record, identity="0" * 64)
    assert "stored identity equals none of the kernel identities" in (_decode(stale, [stale]) or "")
