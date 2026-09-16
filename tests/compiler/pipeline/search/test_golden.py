"""Strict decode of the checked-in golden corpus.

A recorded row is evidence a deploy can use only while it still equals an enumerated leaf of its
own target. The model-agnostic hardware goldens are asked that ROW BY ROW, on the default lane:
parsing one of those files costs milliseconds and deciding one row costs about a second, so the
nodes scatter over the workers instead of four files queueing behind the widest, and a failure
names the row instead of a count. Rows that no longer decode are listed in ``golden_xfails.yaml``
and asked strictly, so the list can only shrink.

Recipe-local model goldens stay off the lane behind the ``goldens`` marker: they are an order of
magnitude more rows, and the widest of them is a multi-megabyte parse. Run those with
``make test-goldens`` after a tuning round has re-recorded a card's rows.
"""

import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from emmy.compiler.pipeline.search.golden import (
    _HARDWARE_GOLDENS_DIR,
    _records_of,
    _repository_golden_paths,
    decode_record,
    flush_identity_store,
    siblings_of,
)

#: The rows whose recorded schedule equals no enumerated leaf today, ``{file name: [row label]}``.
#: They are asked as STRICT xfails: closing one turns its node red until the line is deleted, which
#: is what keeps the hole shrinking. Never add a line to make a red row green — a recorded row that
#: stops decoding is a regression in the enumeration, and listing it enshrines that as the reference.
#: Re-record the card's rows instead, or fix the compiler.
_XFAILS_FILE = Path(__file__).parent / "golden_xfails.yaml"


@pytest.fixture(scope="module", autouse=True)
def _persist_derivations():
    """Persist what this module derived once per worker rather than once per row: the memo is keyed
    by compiler fingerprint and record content, so the next run re-derives only what changed."""
    yield
    flush_identity_store()


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


def _row_parameters():
    """One parameter per recorded row of the hardware goldens, plus one per registry line naming a
    row the file no longer holds. The stale line carries NO xfail: marked, its own failure would be
    the expected one and the dead entry would sit there forever."""
    listed_by_file = yaml.safe_load(_XFAILS_FILE.read_text()) or {}
    parameters = []
    for path in sorted(_HARDWARE_GOLDENS_DIR.glob("*.yaml")):
        labels = _labels(_records_of(path))
        listed = set(listed_by_file.get(path.name, ()))
        for label in labels:
            marks = [pytest.mark.xfail(strict=True, reason="row equals no enumerated leaf")] if label in listed else []
            parameters.append(pytest.param(path, label, id=f"{path.name}/{label}", marks=marks))
        for stale in sorted(listed - set(labels)):
            parameters.append(pytest.param(path, stale, id=f"{path.name}/{stale}"))
    return parameters


@pytest.mark.parametrize(("path", "label"), _row_parameters())
def test_recorded_row_decodes(path: Path, label: str) -> None:
    """One row of a hardware golden must still equal an enumerated leaf of its own target.

    A row that does not is no evidence a deploy can use: the compile that reads it either picks a
    schedule the enumeration no longer offers, or falls through to the prior. Decoding replays the
    persisted program at the record's DECLARED capability, so this holds on any machine — a card is
    needed to re-record a stale row, not to detect one.
    """
    records = _records_of(path)
    labels = _labels(records)
    assert label in labels, f"{_XFAILS_FILE.name} lists {label!r}, which {path.name} no longer records"
    record = records[labels.index(label)]
    assert (reason := _decode(record, records)) is None, reason


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
    records = _records_of(_HARDWARE_GOLDENS_DIR / "rtx5090_sm120.yaml")
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
    from emmy.compiler.pipeline.search import golden
    from emmy.compiler.pipeline.search.golden import _replay, _unmatched_reason, piece_row, unmatched_reason

    monkeypatch.setattr(golden, "_REPLAY_CACHE", {})

    # The same smallest target the anchor test stands on: every assertion here replays it.
    records = _records_of(_HARDWARE_GOLDENS_DIR / "rtx5090_sm120.yaml")
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
    miss = _replay(missing, siblings=siblings, exhaustive=True, wanted=absent)
    keys = set().union(*(summary[0] for summary in miss.offered.values()))
    pairs = set().union(*(summary[1] for summary in miss.offered.values()))
    assert not miss.rows, "a miss retains no candidate rows"
    assert _unmatched_reason(absent, keys, pairs) == unmatched_reason(absent, full)
    assert not golden._REPLAY_CACHE, "requested-row results cannot serve a different recording and must not accumulate"

    respelled = replace(record, knobs={**record.knobs, "WORK@missing": record.knobs["WORK"]})
    reason = _decode(respelled, records)
    assert reason is not None and "WORK@missing" in reason and "re-spelling" in reason


def _recipe_paths() -> list[Path]:
    """The recipe-local model goldens — the repository set minus the hardware files above."""
    with _repository_golden_paths() as paths:
        return [path for path in paths if path.parent.name == "golden"]


@pytest.mark.goldens
@pytest.mark.parametrize("path", _recipe_paths(), ids=lambda path: f"{path.parent.parent.name}/{path.name}")
def test_every_recorded_row_of_a_model_golden_decodes(path: Path) -> None:
    """Every row a model golden records must still decode — one node per file, because a model
    inventory is hundreds of rows whose per-row nodes would cost more to collect than to run."""
    records = _records_of(path)
    failures = [f"  {record.name}: {' '.join(reason.split())[:120]}" for record in records if (reason := _decode(record, records))]
    listed = "\n".join(failures[:20])
    more = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
    assert not failures, f"{len(failures)}/{len(records)} recorded rows equal no enumerated leaf:\n{listed}{more}"


def test_a_row_whose_every_site_is_re_spelled_still_gets_a_verdict() -> None:
    """A row can lose every key it decided at once — an identity re-key moves a kernel's sites, and
    the codec then declares none of the keys the recording spelled. That is a re-spelling like any
    other and the decode has to say so. Two expert rows of the DeepSeek V4 golden raised instead
    after #804 re-keyed the file: the replay filed the kernel's declared keys with no offered pair
    behind them, then indexed the pair it never filed."""
    from dataclasses import replace

    records = _records_of(_HARDWARE_GOLDENS_DIR / "rtx5090_sm120.yaml")
    record = next(r for r in records if r.name == "matmul.square.512" and _decode(r, records) is None)
    respelled = replace(record, knobs={f"{key}@missing": value for key, value in record.knobs.items() if value not in ("", "0")})
    reason = _decode(respelled, records)
    assert reason is not None and "re-spelling" in reason and "@missing" in reason, reason


def test_compiler_fingerprint_ignores_mtime_so_two_checkouts_share_one_memo(tmp_path):
    """Two byte-identical trees fingerprint alike however their mtimes differ.

    The identity memo is one file per machine, and every checkout of the same revision reads it:
    an agent worktree beside the main tree, the re-exported host tree a serving container mounts.
    Keyed by mtime those checkouts disagreed, so each discarded the other's derivations and the
    next process re-derived every identity from scratch.
    """
    from emmy.compiler.pipeline.search.golden import _tree_fingerprint

    first, second = tmp_path / "a", tmp_path / "b"
    for root in (first, second):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "rule.py").write_text("VALUE = 1\n")
    stamp = (1, 1)
    os.utime(second / "pkg" / "rule.py", stamp)

    assert _tree_fingerprint(first) == _tree_fingerprint(second)

    # Same length and the SAME mtime as the equal case: only content differs, so a fingerprint that
    # went back to hashing metadata would fail here instead of passing unnoticed.
    (second / "pkg" / "rule.py").write_text("VALUE = 2\n")
    os.utime(second / "pkg" / "rule.py", stamp)
    assert _tree_fingerprint(first) != _tree_fingerprint(second)


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
