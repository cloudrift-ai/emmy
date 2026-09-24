"""Strict decode of the checked-in golden corpus.

A recorded row is evidence a deploy can use only while it still equals an enumerated leaf of its
own target. Every repository golden — the model-agnostic hardware goldens and each recipe's model
golden — is asked that ROW BY ROW, on the default lane, so the nodes scatter over the workers
instead of queueing behind the widest file, and a failure names the row instead of a count. Rows
that no longer decode are listed in ``golden_xfails.yaml`` and asked strictly, so the list can only
shrink.
"""

import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from emmy.compiler.pipeline.search import golden
from emmy.compiler.pipeline.search.golden import (
    _HARDWARE_GOLDENS_DIR,
    _records_of,
    _repository_golden_paths,
    decode_record,
    flush_identity_store,
    scope_digest,
    siblings_of,
)

#: The rows whose recorded schedule equals no enumerated leaf today, ``{file id: [row label]}`` (see
#: :func:`_golden_id`).
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


def _golden_id(path: Path) -> str:
    """A golden file's id: its name for a hardware golden, ``<recipe>/<name>`` for a model golden."""
    return path.name if path.parent == _HARDWARE_GOLDENS_DIR else f"{path.parent.parent.name}/{path.name}"


def _row_parameters():
    """One parameter per recorded row of every repository golden, plus one per registry line naming
    a row the file no longer holds. The stale line carries NO xfail: marked, its own failure would be
    the expected one and the dead entry would sit there forever."""
    listed_by_file = yaml.safe_load(_XFAILS_FILE.read_text()) or {}
    parameters = []
    with _repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            file_id = _golden_id(path)
            labels = _labels(_records_of(path))
            listed = set(listed_by_file.get(file_id, ()))
            for label in labels:
                marks = [pytest.mark.xfail(strict=True, reason="row equals no enumerated leaf")] if label in listed else []
                parameters.append(pytest.param(path, label, id=f"{file_id}/{label}", marks=marks))
            for stale in sorted(listed - set(labels)):
                parameters.append(pytest.param(path, stale, id=f"{file_id}/{stale}"))
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
    labels = _labels(records)
    assert label in labels, f"{_XFAILS_FILE.name} lists {label!r}, which {path.name} no longer records"
    record = records[labels.index(label)]
    assert (reason := _decode(record, records)) is None, reason


def test_scope_digest_follows_the_cards_rows_only(tmp_path, monkeypatch) -> None:
    """The digest a serving pack keys on moves with the rows this card's compile reads and with nothing else: another
    card's file, or a file scope that names a different file."""
    mine = tmp_path / "mine.yaml"
    other = tmp_path / "other.yaml"
    mine.write_text("gpu_name: NVIDIA H100 80GB HBM3\nrows: 1\n")
    other.write_text("gpu_name: NVIDIA GeForce RTX 5090\nrows: 1\n")
    monkeypatch.setattr(golden, "_repository_golden_paths", lambda: nullcontext([mine, other]))
    monkeypatch.delenv("EMMY_GOLDEN_FILE", raising=False)
    card = "NVIDIA H100 80GB"
    base = scope_digest(card)
    other.write_text("gpu_name: NVIDIA GeForce RTX 5090\nrows: 2\n")
    assert scope_digest(card) == base
    mine.write_text("gpu_name: NVIDIA H100 80GB HBM3\nrows: 2\n")
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
    miss = _replay(missing, siblings=siblings, exhaustive=True, wanted=absent, explain=True)
    keys = set().union(*(summary[0] for summary in miss.offered.values()))
    pairs = set().union(*(summary[1] for summary in miss.offered.values()))
    assert not miss.rows, "a miss retains no candidate rows"
    assert _unmatched_reason(absent, keys, pairs) == unmatched_reason(absent, full)
    assert not golden._REPLAY_CACHE, "requested-row results cannot serve a different recording and must not accumulate"

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

    records = _records_of(_HARDWARE_GOLDENS_DIR / "rtx5090_sm120.yaml")
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

    records = _records_of(_HARDWARE_GOLDENS_DIR / "rtx5090_sm120.yaml")
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
    from emmy.compiler.pipeline.search.golden import _replay, piece_row

    def decodes(record, siblings) -> bool:  # the replay itself: the decode's verdict is memoized per record
        wanted = schedule_match_key(piece_row(record.knobs))
        return any(_replay(record, siblings=siblings, exhaustive=True, wanted=wanted).rows.values())

    golden = Path(__file__).parents[4] / "recipes" / "DeepSeek-V4-Flash-0731" / "golden" / "v100_sm70.yaml"
    records = _records_of(golden)
    lead = next(r for r in records if r.name == "pre16.k_linear_mean_reduce_03c479.8caa25e24052.m16.dc6db94ec8ea.dc6db94ec8ea")
    siblings = siblings_of(lead, records)
    receipt = next(m for m in siblings if m.name.endswith(".5d9b14249e94"))
    assert decodes(receipt, [lead, *(m for m in siblings if m is not receipt)]), "the receipt decodes behind its lead's cut"

    other = next(m for m in siblings if m.name.endswith(".c607711d8ef8"))
    impostor = replace(other, identity=lead.identity)
    beside = [lead, *(impostor if m is other else m for m in siblings if m is not receipt)]
    assert decodes(receipt, beside), "and beside a receipt stamped with the lead's identity"


def test_compiler_fingerprint_ignores_mtime_so_two_checkouts_share_one_memo(tmp_path):
    """Two byte-identical trees fingerprint alike however their mtimes differ.

    The identity memo is one file per fingerprint, and every checkout of the same revision reads it:
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


def test_a_flush_from_another_compiler_tree_keeps_this_trees_derivations(tmp_path, monkeypatch):
    """Two compiler revisions sharing one cache directory each keep what they derived.

    The memo was one file holding one fingerprint, so a process from any other tree replaced it
    whole. On the serving host a one-row replay from an older tree, run between two boots of the
    same tree, cost the second boot every replay the first had derived: 851 s, then 859 s.
    """
    from emmy import config
    from emmy.compiler.pipeline.search import golden

    monkeypatch.setattr(config, "_CACHE_ROOT", tmp_path)

    def derive(fingerprint: str, key: str) -> dict:
        monkeypatch.setattr(golden, "_compiler_fingerprint", lambda: fingerprint)
        monkeypatch.setattr(golden, "_IDENTITY_STORE", None)
        kept = dict(golden._identity_store()["entries"])
        golden._identity_store()["entries"][key] = None
        monkeypatch.setattr(golden, "_IDENTITY_STORE_DIRTY", True)
        flush_identity_store()
        return kept

    derive("serving tree", "boot")
    derive("another tree", "replay")
    assert "boot" in derive("serving tree", "next boot")


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


#: The goldens whose stored targets no longer come out of a fresh lowering of their own programs,
#: ``[file id]`` (see :func:`_golden_id`). Strict xfails, like the row list above: the line goes
#: when the file is restamped.
_LOWERING_XFAILS_FILE = Path(__file__).parent / "golden_lowering_xfails.yaml"


def _golden_parameters():
    listed = set(yaml.safe_load(_LOWERING_XFAILS_FILE.read_text()) or ())
    parameters = []
    with _repository_golden_paths() as paths:
        for path in sorted(paths, key=_golden_id):
            file_id = _golden_id(path)
            marks = [pytest.mark.xfail(strict=True, reason="stored targets are not the fresh lowering")] if file_id in listed else []
            parameters.append(pytest.param(path, id=file_id, marks=marks))
    return parameters


@pytest.mark.parametrize("path", _golden_parameters())
def test_stored_targets_are_the_fresh_lowering(path: Path) -> None:
    """Every stored target of a repository golden must be a kernel the current compiler lowers the
    golden's own traced program to — byte for byte.

    A golden's rows are evidence for the kernels its stored Loop IR names, and a deploy keys them by
    the kernels it lowers FRESH from the model. The decode test above replays the stored target, so
    it stays green when the two drift apart: after #863 the DeepSeek V4 V100 golden decoded row by
    row while serving lowered kernels no row of it described and the strict boot refused. Lowering
    is the loop passes alone, GPU-free, so this holds on any machine; a file that fails needs a
    restamp on the card that recorded it, not a card to detect it.
    """
    from emmy.compiler import provenance
    from emmy.compiler.context import Context
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.loop_wire import loop_graph_to_wire
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.pipeline.search.golden import load_golden_file
    from emmy.compiler.pipeline.search.slice import single_node_graph
    from emmy.compiler.torch_wire import graph_from_wire

    document = load_golden_file(path)
    ctx = Context.from_target(tuple(document["compute_cap"]), gpu_name=document.get("gpu_name"))
    fresh: dict[int, dict[frozenset, dict]] = {}
    for index in sorted({entry["program"] for entry in document["configs"]}):
        graph = graph_from_wire(document["programs"][index])
        for node in graph.nodes.values():
            if node.hints.get("trace.materialize") and node.id not in graph.outputs:
                graph.outputs.append(node.id)
        for node in graph.nodes.values():
            node.hints.remove(provenance.PROV)
        provenance.seed(graph)
        fused = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)
        kernel_ids = [nid for nid in fused.topological_order() if isinstance(fused.nodes[nid].op, LoopOp)]
        kernels = (loop_graph_to_wire(single_node_graph(fused, nid)) for nid in kernel_ids)
        fresh[index] = {frozenset(wire["outputs"]): wire for wire in kernels}
    stale = []
    for entry in document["configs"]:
        stored = document["loops"][entry["target"]["loop"]]
        wire = fresh[entry["program"]].get(frozenset(stored["outputs"]))
        if wire != stored:
            name = entry["realizations"][0]["name"] if entry.get("realizations") else f"loop {entry['target']['loop']}"
            stale.append(f"{name}: " + ("no fresh kernel writes its outputs" if wire is None else "the fresh kernel's Loop IR differs"))
    assert not stale, f"{len(stale)} of {len(document['configs'])} targets are not the fresh lowering:\n  " + "\n  ".join(stale[:12])
