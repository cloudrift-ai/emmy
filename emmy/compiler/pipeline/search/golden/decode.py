"""The strict decode of a record against the current compiler: the replay of its target through the tile passes
under its pins, the match of its spelled row against what the replay offers, and the reason when it misses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from emmy.compiler.pipeline.knob import family_of

from .record import GoldenRecord, _lifted_target, kernel_set_pins


def unmatched_reason(row: Sequence[tuple[str, str]], candidates) -> str:
    """Why a recorded row equals no enumerated leaf — the three readings golden churn has.

    A compiler change moves recorded rows in bulk and only one of the readings is a loss, so a
    re-record that does not tell them apart can enshrine one. A key the replay offers NOWHERE is a
    re-spelling or an identity change: the site the row addressed is not on this kernel any more. A
    value gone while its key survives is a NARROWING — the family is still offered and no longer
    reaches that value, which is a capability the compiler used to have and is the one reading to
    report rather than overwrite. Everything offered but never together is a kernel whose fork SET
    moved, of which the row that spelled no decision at all against a kernel that now takes one is
    the common case, and a gain."""
    seen_keys: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        seen_keys.update(key for key, _ in candidate)
        seen_pairs.update(candidate)
    return _unmatched_reason(row, seen_keys, seen_pairs)


def _unmatched_reason(
    row: Sequence[tuple[str, str]], seen_keys: set[str] | frozenset[str], seen_pairs: set[tuple[str, str]] | frozenset[tuple[str, str]]
) -> str:
    """Classify a miss from the bounded facts the candidates offered."""
    absent = sorted({key for key, _ in row if key not in seen_keys})
    if absent:
        return f"the replay offers no {', '.join(absent)} — a re-spelling or an identity change"
    narrowed = sorted(f"{key}={value!r}" for key, value in row if (key, value) not in seen_pairs)
    if narrowed:
        return f"NARROWING, the key is offered and the value is not: {', '.join(narrowed)}"
    if not row:
        return "the kernel takes a schedule where the recording spelled none"
    return "every key and value is offered, no one candidate carries them together"


def decode_record(record: GoldenRecord, siblings: Sequence[GoldenRecord] = ()) -> str | None:
    """STRICTLY decode one record against the current compiler — ``None`` on success, else the
    failure reason. This is the replayability contract the nightly onboarding job gates: the persisted
    program selects exactly one kernel, except that a child-identity schedule receipt may select its
    kernel from a multi-kernel target by stored identity; a routing record's every cut key names a
    seam the cut pass offers on the replay (:func:`_replay`); a SCHEDULE record's spelled row equals
    one enumerated leaf (``schedule_match_key`` equality under the record's own pins) — no prefix
    matching, no any-of, no classified shape. That equality is blind to the two sides' OFF anchors:
    which of them a spelling writes down depends on whether it came from a resolved kernel or from a
    fork's offered leaf, and neither carries schedule content. A stored identity — a receipt's, or
    a target's own — must equal one kernel resolved under the record's pins, and the spelled row must
    equal one of THAT kernel's rows — a sibling child's row must not vouch for it; a compiler change
    that re-keys the kernel turns the row red until the file is re-keyed."""
    from emmy.compiler.pipeline.knob import schedule_match_key  # noqa: PLC0415

    tile = None
    try:
        tile = _lifted_target(record)
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        if not record.is_receipt:
            return f"{type(exc).__name__}: {exc}"
    # The piece row, not the recorded one: a ``g<n>`` cross-CTA half names the kernel-set arm the
    # replay already resolved, and the pieces it mints cannot stamp it, so comparing it to a leaf
    # asks a piece to spell its parent's decision.
    row = schedule_match_key(piece_row(record.knobs))
    # The set's leading entry is the target's own — the one naming the kernel the target lifts to —
    # and it decides every fork no entry names, a residual's further cut or split included; a
    # receipt replayed as its own lead would read those forks as fused and never mint its kernel.
    own = tile.identity_key(with_io=True) if tile is not None else None
    lead = next((entry for entry in (record, *siblings) if own is not None and entry.identity == own), None)
    replay = _replay(record, siblings=siblings, lead=lead, exhaustive=True, wanted=row)
    if record.is_routing:
        reason = f"routing key {replay.unresolved[0]!r} does not resolve to an offered cut seam" if replay.unresolved else None
        return reason

    def verdict(replay: _Replay) -> str | None:
        candidates = replay.rows
        if record.identity is not None and (tile is None or record.identity != tile.identity_key(with_io=True)):
            child_rows = candidates.get(record.identity)
            offered = replay.offered.get(record.identity, (frozenset(), frozenset()))
            if record.identity not in replay.kernels:
                return "stored identity equals none of the kernel identities resolved under the record's pins"
            if child_rows is not None and row in child_rows:
                return None
            return f"no enumerated row of the identified kernel equals the recording: {_unmatched_reason(row, *offered)}"
        pooled = frozenset().union(*candidates.values()) if candidates else frozenset()
        offered_keys = set().union(*(summary[0] for summary in replay.offered.values())) if replay.offered else set()
        offered_pairs = set().union(*(summary[1] for summary in replay.offered.values())) if replay.offered else set()
        return None if row in pooled else f"no enumerated row equals the recording: {_unmatched_reason(row, offered_keys, offered_pairs)}"

    reason = verdict(replay)
    if reason is not None:
        # A miss: replay again walking every fork, so the reason names what the kernels offer.
        reason = verdict(_replay(record, siblings=siblings, lead=lead, exhaustive=True, wanted=row, explain=True))
    return reason


class _Replay(NamedTuple):
    """One replay of a record's target through the tile passes under the record's pins, following
    the record's knobs at every kernel-set fork (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`).

    ``rows`` — the EXHAUSTIVE replay's answer when no row is requested: every schedule-row identity each kernel can realize
    as a :func:`~emmy.compiler.pipeline.knob.schedule_match_key`, bucketed by the kernel's deploy
    identity (``identity_key(with_io=True)``; ``None`` for forks
    whose root is not a recognized ``TileOp``): the fork leaves' rows, PLUS each resolved kernel's
    own realized row — a forkless kernel (the schedule space collapsed to one row, often the all-OFF
    anchor) never opens a fork, so its one row is read off the resolved op instead. Behind a cut the
    buckets are exactly the pieces, which is what lets a child-identity receipt decode against its
    own kernel only. A requested-row replay keeps only an exact match here. ``arms`` — the arm the
    record's route and knobs spelled at each kernel-set fork it decided (a cut seam, a cross-CTA
    plan), keyed by the signature of the kernel that fork was offered on.
    ``unresolved`` — the record's scoped cut keys no offered seam carried, the strict decode's
    routing failure. ``offered`` — for a requested-row replay, the bounded set of keys and
    key/value pairs offered per kernel identity; a miss uses it to explain re-spelling, narrowing
    or regrouping without retaining every candidate row."""

    rows: dict[str | None, frozenset]
    #: The kernels the replay scheduled — those that reached a schedule fork or were resolved
    #: without one; a kernel a cut or split consumed is not among them.
    kernels: frozenset[str]
    arms: tuple[tuple[frozenset, dict[str, str]], ...]
    unresolved: tuple[str, ...]
    #: Each scheduled kernel's realized schedule row (``schedule_row_key`` families), by identity —
    #: what a per-kernel entry for a kernel the set leaves undescribed would record.
    realized: dict[str, dict[str, str]]
    offered: dict[str | None, tuple[frozenset[str], frozenset[tuple[str, str]]]]


def piece_row(row: Mapping[str, str]) -> dict[str, str]:
    """A record's schedule row as a piece of its kernel set can carry it: a ``REDUCE`` value reduced
    to what a piece can still stamp (:func:`~emmy.compiler.pipeline.search.pins.stampable_reduce`),
    since the cross-CTA split it names was the parent's decision.

    A value whose whole content was the split reduces to the OFF ``''``, and the key STAYS at it:
    the piece decided to fold nothing, and an enumerated leaf spells that decision rather than
    omitting the family (:attr:`GoldenRecord.schedule_row`). Dropping the key instead read as
    "free", which no leaf equals — the whole split half of a card's rows decoded to nothing and
    joined no kernel in the evidence index."""
    from emmy.compiler.pipeline.search.pins import stampable_reduce  # noqa: PLC0415

    out = {str(key): str(value) for key, value in row.items()}
    for key, value in list(out.items()):
        if family_of(key) == "REDUCE" and (rest := stampable_reduce(value)) is not None:
            out[key] = rest
    return out


def siblings_of(record: GoldenRecord, records: Sequence[GoldenRecord]) -> tuple[GoldenRecord, ...]:
    """The other records of ``record``'s target among ``records`` — same config entry, bindings and
    input regime: the entries that walk one kernel set together (a case's per-kernel entries, a
    golden config's receipts). The first of them in ``records`` order is the set's lead."""
    key = _set_key(record)
    return tuple(other for other in records if other is not record and _set_key(other) == key)


def lead_of(record: GoldenRecord, records: Sequence[GoldenRecord]) -> GoldenRecord:
    """The set's leading entry — the first record of ``record``'s target in ``records`` order, the
    target's own entry: it decides every fork no entry names by identity."""
    key = _set_key(record)
    return next(other for other in records if _set_key(other) == key)


def _record_cache_key(record: GoldenRecord) -> tuple:
    return (id(record.loop_wire), record.target_key, record.compute_cap, record.bindings)


def _set_key(record: GoldenRecord) -> tuple:

    regime = tuple(sorted((str(k), str(v)) for k, v in record.pin_map.items() if family_of(str(k)) != "PLACE"))
    return (_record_cache_key(record), record.config_index, regime)


def _replay(
    record: GoldenRecord,
    *,
    siblings: Sequence[GoldenRecord] = (),
    lead: GoldenRecord | None = None,
    exhaustive: bool = False,
    wanted: tuple[tuple[str, str], ...] | None = None,
    explain: bool = False,
) -> _Replay:
    """Replay ``record``'s target through the tile passes — see :class:`_Replay`. The record's input
    pins are the regime it was measured under and go to the environment; its route (the ``PLACE``
    keys of its pins and knobs) and its knobs are the decisions it took at forks and are followed
    fork by fork. A piece a cut or split mints is a brand-new kernel: it inherits nothing, and the
    record's remaining keys are read against its own offers, exactly as the deploy reads a row of
    its signature.

    ``siblings`` are the other entries of the same target (:func:`siblings_of`) and ``lead`` the
    set's leading entry (:func:`lead_of`; its explicit kernel-set entry, or the record itself, when absent). A fork offered on a kernel
    one entry names by ``identity`` is decided by THAT entry's spelling; every other fork by the
    lead's — never by an entry that does not own it, whose row would say "fused" or "unsplit" of a
    kernel it never described. So a set of per-kernel entries — the parent's cut, each piece's
    row — walks one path together, and the record's own rows are what this replay reports.
    ``exhaustive`` streams every schedule pool for ``rows``; a plain replay descends to each
    kernel's realized row. ``wanted`` names the ONE match key the caller will ask ``rows`` about.
    An unsampled schedule answers by decoding that complete row through its codec and compatibility
    context, without enumerating candidates. Other forks use lazy descent and keep only the wanted
    keys and values needed to classify a miss, never the candidate rows. A schedule fork that cannot
    hold ``wanted`` is left undecided: the kernel it decides is not the one the row describes, and
    giving it a schedule anyway walks its fork to a first leaf, the bulk of a cold decode. ``explain``
    walks such forks instead, to name what they offer when the row is found nowhere."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import exact_schedule_leaf, fork_signature, iter_leaves, leaf_for, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import (  # noqa: PLC0415
        schedule_match_key,
        schedule_row_key,
        validate_family_value,
    )
    from emmy.compiler.pipeline.pipeline import NO_OPTION, Run, _is_structural_option  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import composed_routes, pinned_knobs, spelled_arm, unpinned_decisions  # noqa: PLC0415

    def _spelling(entry: GoldenRecord) -> dict[str, str]:
        # A routed realization measures nothing itself and carries no row, so read alone it would
        # say "this kernel ran whole" — the fuse reading, which is right for a row that genuinely
        # took no kernel-set decision and wrong for this one. It spells the route it names instead.
        referenced = kernel_set_pins(entry, (record, *siblings))
        return {**referenced, **entry.route, **{str(key): str(value) for key, value in entry.knobs.items()}}

    if lead is None:
        lead = record if record.is_routing else next((entry for entry in (record, *siblings) if entry.kernel_set), record)
    # Entries can share an identity — a routing row and a plain row of one target. The one that
    # spells a route decides the cut fork (it sorts last, and last wins); a row spelling none would
    # read the kernel as fused.
    named = {entry.identity: entry for entry in sorted(siblings, key=lambda entry: bool(entry.route)) if entry.identity is not None}
    if record.identity is not None:
        named[record.identity] = record
    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    spelled = _spelling(record)
    regime = {key: value for key, value in record.pin_map.items() if family_of(str(key)) != "PLACE"}
    pending = {key for key, value in spelled.items() if family_of(key) == "PLACE" and value == "cut"}
    piece = piece_row(record.schedule_row)
    buckets: dict[str | None, set] = {}
    offered_keys: dict[str | None, set[str]] = {}
    offered_pairs: dict[str | None, set[tuple[str, str]]] = {}
    wanted_keys = frozenset(key for key, _ in wanted or ())
    wanted_pairs = frozenset(wanted or ())
    kernels: set[str] = set()
    realized: dict[str, dict[str, str]] = {}
    arms: list[tuple[frozenset, dict[str, str]]] = []
    declined: set[str] = set()  # the schedule forks left undecided: none of them can hold ``wanted``

    def _identity_of(op) -> str | None:
        return op.identity_key(with_io=True) if isinstance(op, TileOp) else None

    def _offer(identity: str | None, row: tuple[tuple[str, str], ...]) -> None:
        offered_keys.setdefault(identity, set()).update(key for key, _ in row if key in wanted_keys)
        offered_pairs.setdefault(identity, set()).update(pair for pair in row if pair in wanted_pairs)

    def decide(fp):
        # The kernel's signature as the deploy reads it at this fork — an op resolved without a
        # fork is keyed below, off its own stamp.
        signature = fork_signature(fp.root_op, fp.options, ctx)
        identity = _identity_of(fp.root_op)
        owner = named.get(identity) if identity is not None else None
        decider = owner if owner is not None else lead
        if fp.structural:
            arm = spelled_arm(fp.options, spelled if decider is record else _spelling(decider))
            if arm is not None:
                option, knobs = arm
                if _is_structural_option(option) and decider is record:
                    # A cut consumes the key that spelled it (a bare ``PLACE=cut`` its one root-most
                    # cut), so the pieces are read against what the record has left to say.
                    arms.append((signature, dict(knobs)))
                    for key in (*(pending & set(knobs)), *(("PLACE",) if spelled.get("PLACE") == "cut" else ())):
                        pending.discard(key)
                        spelled.pop(key, None)
                return option
        if identity is not None:
            kernels.add(identity)
        if not exhaustive:
            # A receipt can name the TileOp produced by its schedule row rather than the op that
            # owns the schedule fork. Decode that one row without enumerating the pool, but select
            # it only after materializing the leaf and refreshing its graph I/O proves that it
            # produces the receipt's deploy identity; another unowned kernel that accepts the
            # same keys must retain the lead's decision.
            asked = piece if decider is record else piece_row(decider.schedule_row)
            exact = exact_schedule_leaf(fp.options, piece, frozenset(piece)) if owner is None and piece else None
            if exact is not None:
                _declared, option = exact
                materialized = option.expand() if option is not None else ()
                candidate = materialized[0].with_io(fp.match.graph, fp.match.root) if len(materialized) == 1 else None
                hit = (option, leaf_knobs(option)) if candidate is not None and _identity_of(candidate) == record.identity else None
            else:
                hit = None
            if hit is None:
                hit = leaf_for(fp.options, asked) if asked else None
            return hit[0] if hit is not None else next(iter_leaves(fp.options))
        # An unsampled semantic schedule decodes the complete wanted row through the same codec and
        # compatibility context that validates a direct schedule. Other forks retain the generic
        # lazy descent; ``skip`` keeps that answer exact where a partial row vouches for more than
        # one leaf. Structural options never enter ``buckets``.
        proved_miss = False
        if wanted is not None:
            exact = exact_schedule_leaf(fp.options, piece, wanted_keys)
            if exact is not None:
                declared, hit = exact
                offered_keys.setdefault(identity, set()).update(wanted_keys & declared)
                offered_pairs.setdefault(identity, set())
                for key, value in wanted_pairs:
                    try:
                        if key in declared and validate_family_value(key, value) == value:
                            offered_pairs.setdefault(identity, set()).add((key, value))
                    except ValueError:
                        pass
                if hit is not None and schedule_match_key(leaf_knobs(hit)) == wanted:
                    buckets.setdefault(identity, set()).add(wanted)
                    _offer(identity, wanted)
                    return hit
                if not explain and not fp.structural:
                    declined.add(fp.node_id)
                    return NO_OPTION
                return next(iter_leaves(fp.options))
            hit = leaf_for(fp.options, piece, skip=lambda knobs: schedule_match_key(knobs) != wanted)
            if hit is not None and not _is_structural_option(hit[0]):
                buckets.setdefault(identity, set()).add(wanted)
                _offer(identity, wanted)
                return hit[0]
            proved_miss = hit is None
            if not explain and not fp.structural:
                declined.add(fp.node_id)
                return NO_OPTION
        first_leaf = None
        first_op = None
        for leaf in iter_leaves(fp.options):
            if first_leaf is None:
                first_leaf = leaf
            if _is_structural_option(leaf):
                continue
            if first_op is None:
                first_op = leaf
            row = leaf_knobs(leaf)
            key = schedule_match_key(row)
            if wanted is None:
                if row:
                    buckets.setdefault(identity, set()).add(key)
            else:
                if key == wanted:
                    buckets.setdefault(identity, set()).add(wanted)
                _offer(identity, key)
                if proved_miss and wanted_pairs and wanted_pairs <= offered_pairs[identity]:
                    break
        assert first_leaf is not None
        return first_op if first_op is not None else first_leaf

    # The seams an entry marks cut together are one composed decision where they resolve on one
    # kernel (a pinned compile consumed them so, and ``run --record-greedy`` wrote them so); the cut
    # pass offers that arm on this replay's kernels so whichever entry decides a fork — the record,
    # the lead, a sibling naming the kernel — can spell it.
    composed: list[tuple[frozenset | None, tuple[str, ...]]] = []
    for entry in (record, lead, *named.values()):
        keys = tuple(sorted(key for key, value in _spelling(entry).items() if family_of(key) == "PLACE" and value == "cut"))
        if len(keys) > 1 and (None, keys) not in composed:
            composed.append((None, keys))
    with unpinned_decisions(), pinned_knobs(regime), composed_routes(composed):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(record.target_program.copy(), decide)
    for node_id, node in out.nodes.items():
        if isinstance(node.op, TileOp):
            identity = _identity_of(node.op)
            knobs = dict(node.op.knobs or {})
            row = schedule_row_key(knobs)
            if exhaustive and node_id not in declined:
                row_key = schedule_match_key(knobs)
                if wanted is None or row_key == wanted:
                    buckets.setdefault(identity, set()).add(row_key)
                if wanted is not None:
                    _offer(identity, row_key)
            if identity is not None:
                kernels.add(identity)
                realized[identity] = dict(row)
    result = _Replay(
        {identity: frozenset(rows) for identity, rows in buckets.items()},
        frozenset(kernels),
        tuple(arms),
        tuple(sorted(pending)),
        realized,
        {identity: (frozenset(keys), frozenset(offered_pairs.get(identity, ()))) for identity, keys in offered_keys.items()},
    )
    return result
