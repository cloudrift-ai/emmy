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
