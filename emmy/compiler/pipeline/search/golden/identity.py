"""A record's kernel identity under the current compiler, and the persisted derivation memo — identities, decode
verdicts and fresh-lowering digests — keyed by the compiler tree's fingerprint."""

from __future__ import annotations

import tempfile
from pathlib import Path

from emmy.compiler.structural import digest

from .record import GoldenRecord, _lifted_target, _record_cache_key

_IDENTITY_CACHE: dict[tuple, str | None] = {}
#: The persisted identity memo: {record fingerprint: identity | None}, valid only under one
#: compiler fingerprint. Purely derived data — a stale or missing store just re-derives.
_IDENTITY_STORE: dict | None = None
_IDENTITY_STORE_DIRTY: bool = False
#: Wire payload digests, memoized per payload OBJECT. The value holds the wire itself, not just
#: its digest: an ``id()``-keyed memo whose entry outlives the object it describes answers for
#: whatever later lands at that address, and this memo feeds a record's identity fingerprint.
#: Keeping the reference is what makes the address stable, and it matches how every sibling
#: cache in this module (``_PROGRAM_GRAPH_CACHE``, ``_LOOP_GRAPH_CACHE``) is written.
_WIRE_DIGESTS: dict[int, tuple[dict, str]] = {}


def _tree_fingerprint(root: Path) -> str:
    """Path plus CONTENT digest of every ``*.py`` under ``root``.

    Content, not mtime: the memo this keys is one file per fingerprint and every checkout of the same
    revision reads it — an agent worktree beside the main tree, the re-exported tree a serving
    container mounts. Byte-identical sources with different mtimes fingerprinted differently, so
    each checkout discarded the other's derivations and the next process re-derived every identity
    from scratch. Hashing the 3.7 MB the compiler occupies costs about 6 ms, once per process.
    """
    return digest("\n".join(f"{path.relative_to(root)}:{digest(path.read_bytes())}" for path in sorted(root.rglob("*.py"))))


def _compiler_fingerprint() -> str:
    """The compiler tree's fingerprint. Any edit invalidates the persisted identity memo, so a
    derivation can never be replayed across compiler versions."""
    import emmy.compiler as _pkg  # noqa: PLC0415

    return _tree_fingerprint(Path(_pkg.__file__).parent)


def _identity_store() -> dict:
    global _IDENTITY_STORE
    if _IDENTITY_STORE is None:
        import json  # noqa: PLC0415

        from emmy import config  # noqa: PLC0415

        fingerprint = _compiler_fingerprint()
        sections: dict = {"entries": {}, "verdicts": {}, "lowerings": {}}
        try:
            payload = json.loads(config.golden_identity_cache_path(fingerprint).read_text())
            sections = {name: payload.get(name, {}) for name in sections}
        except (OSError, ValueError):
            pass
        _IDENTITY_STORE = {"fingerprint": fingerprint, **sections}
    return _IDENTITY_STORE


def flush_identity_store() -> None:
    """Persist newly derived identities, decode verdicts and fresh-lowering digests (atomic replace;
    concurrent writers merge — a lost write only re-derives later), so the next process on this machine and
    compiler reads the derivations instead of lifting every record again."""
    global _IDENTITY_STORE_DIRTY
    if not _IDENTITY_STORE_DIRTY or _IDENTITY_STORE is None:
        return
    import json  # noqa: PLC0415

    from emmy import config  # noqa: PLC0415

    path = config.golden_identity_cache_path(_IDENTITY_STORE["fingerprint"])
    try:
        # MERGE with the on-disk state before writing: concurrent processes (xdist workers each
        # walking one golden set) flush independently, and overwrite-last-wins silently dropped
        # every other worker's derivations.
        try:
            on_disk = json.loads(path.read_text())
        except (OSError, ValueError):
            on_disk = None
        if on_disk is not None:
            for section in ("entries", "verdicts", "lowerings"):
                merged = dict(on_disk.get(section, {}))
                merged.update(_IDENTITY_STORE.get(section, {}))
                _IDENTITY_STORE[section] = merged
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            json.dump(_IDENTITY_STORE, out)
            temporary = Path(out.name)
        temporary.replace(path)
        _IDENTITY_STORE_DIRTY = False
    except OSError:
        pass  # the store is a memo; failing to persist only costs a re-derivation


def _record_fingerprint(record: GoldenRecord) -> str:
    """A stable content digest for one record's TARGET (identity depends on nothing else): the
    persisted wire payload, the target selector, bindings, card. Wire digests are memoized per
    payload object — one document's records share their program pool."""
    import json  # noqa: PLC0415

    wire = record.loop_wire
    cached = _WIRE_DIGESTS.get(id(wire))
    if cached is None or cached[0] is not wire:
        cached = (wire, digest(json.dumps(wire, sort_keys=True, default=str)))
        _WIRE_DIGESTS[id(wire)] = cached
    return digest(cached[1], str(record.target_key), str(record.bindings), str(record.compute_cap), record.gpu_name or "")


def remember(section: str, key: str, value):
    """Write one derivation into the memo — an identity, a decode verdict, a fresh lowering — and mark it for
    :func:`flush_identity_store`. Returns ``value``."""
    global _IDENTITY_STORE_DIRTY
    _identity_store().setdefault(section, {})[key] = value
    _IDENTITY_STORE_DIRTY = True
    return value


def kernel_identity(record: GoldenRecord) -> str | None:
    """The record's kernel identity under the CURRENT compiler — the strict decode's and the drift
    key (``identity_key(with_io=True)``). A STORED identity is returned as-is: it is how a
    child-identity receipt names the one split child its schedule decorates (the target's own lift
    stops at the pre-cut kernel and cannot say), and a stale stored identity selects nothing — the
    strict decode is where that fails loudly. Without one, the identity is derived as the lift of the
    record's ONE target kernel, through the exact total lift the live compile uses
    (``_fromloop.lift_loop_op``). ``None`` when the record cannot carry a deploy identity: the target
    lowers to several kernels (a schedule row decorates exactly one), or selection/lifting fails —
    best-effort here (a corpus row must never break a compile); nightly strict decoding is where
    failure is loud. Deploy never joins on this key: a record deploys as measured rows, matched by
    ``S_*`` features plus the exact ``I_kernel`` stamp, off the rows the golden import files
    (``golden.evidence``)."""
    if record.identity is not None:
        return record.identity
    key = _record_cache_key(record)
    if key in _IDENTITY_CACHE:
        return _IDENTITY_CACHE[key]
    store = _identity_store()
    fingerprint = _record_fingerprint(record)
    if fingerprint in store["entries"]:
        identity = store["entries"][fingerprint]
        _IDENTITY_CACHE[key] = identity
        return identity
    try:
        identity = _lifted_target(record).identity_key(with_io=True)
    except Exception:  # noqa: BLE001 — see the docstring; the decode tripwire re-derives loudly
        identity = None
    _IDENTITY_CACHE[key] = identity
    return remember("entries", fingerprint, identity)
