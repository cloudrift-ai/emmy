"""The repository's goldens: where they live, which of them a card reads, and the evidence scope a compile installs
over them."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from emmy import config, gpu
from emmy.recipe.bundled import default_recipe_root

from .format import GoldenFile
from .record import GoldenRecord, GoldenRecords

logger = logging.getLogger("emmy.compiler.pipeline")

#: The maintained golden records, one file per card: model-agnostic rows the offline prior trains on, the tests
#: decode, and a compile on that card picks from. They ship inside this package.
_RECORDS_DIR = Path(__file__).parent / "records"
_RECIPE_GOLDEN_DIR = "golden"


def is_repository_golden_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    records_root = _RECORDS_DIR.resolve()
    if resolved == records_root or records_root in resolved.parents:
        return True
    with default_recipe_root() as recipe_root:
        if recipe_root is None:
            return False
        try:
            relative = resolved.relative_to(recipe_root.resolve())
        except ValueError:
            return False
        return len(relative.parts) >= 2 and relative.parts[1] == _RECIPE_GOLDEN_DIR


@contextmanager
def repository_golden_paths():
    """Yield model-agnostic hardware goldens plus recipe-local model goldens."""
    with default_recipe_root() as recipe_root:
        paths = list(_RECORDS_DIR.glob("*.json"))
        if recipe_root is not None:
            paths.extend(recipe_root.glob(f"*/{_RECIPE_GOLDEN_DIR}/*.json"))
        yield sorted(paths)


def _file_gpu_name(path: Path) -> str | None:
    """The document's ``gpu_name`` read off the file's first line without parsing the body — the dump
    writes it there alone, so a card-scoped consumer can skip foreign multi-megabyte files (the
    whole-corpus parse is the dominant first-evidence cost). ``None`` when the head does not carry
    it — the caller falls back to the full parse."""
    try:
        with path.open() as handle:
            head = handle.readline().rstrip()
        return gpu.canonical_name(str(json.loads(head.removesuffix(",") + "}")["gpu_name"]))
    except (OSError, ValueError, KeyError):
        return None


#: Optional scope override for :func:`records_for_card` — the golden rows the evidence index loads.
#: ``None`` (the default) reads ``EMMY_GOLDEN_FILE`` when set, else the repository files. Every
#: command that names a golden scopes the rows here: ``run`` / ``compile`` install the selected
#: records in-process, the release gate (``eval golden --serving-config``) one precision lane's
#: records through :func:`sole_evidence`, and ``serve --golden`` reaches the same loader through
#: the env var because the vLLM child is another process. Set it through :func:`records_override`,
#: never by hand.
RECORDS_OVERRIDE: GoldenRecords | None = None


@contextmanager
def records_override(records: Iterable[GoldenRecord] | None):
    """Scope the golden rows :func:`records_for_card` supplies to the evidence index, restoring
    the previous scope after. ``[]`` hides every record — how a caller that must measure without
    golden evidence says so; ``None`` is a no-op, leaving whatever scope is already installed.

    **The body must not ``await``.** This swaps a module global, so it is only atomic with respect
    to other coroutines while the block stays synchronous."""
    global RECORDS_OVERRIDE  # noqa: PLW0603 — the documented scope seam, one owner
    if records is None:
        yield
        return
    prev = RECORDS_OVERRIDE
    RECORDS_OVERRIDE = GoldenRecords.of(records)
    try:
        yield
    finally:
        RECORDS_OVERRIDE = prev


@contextmanager
def sole_evidence(records: Iterable[GoldenRecord]):
    """``records`` as a compile's ONLY evidence, strictly: the golden scope is these rows
    (:func:`records_override`) and strict evidence is on, so a fork none of the rows decides is an
    ``EvidenceError`` naming the kernel instead of a prediction; a ``Pipeline.run`` given no ``db``
    consults no tune DB either. The release gate (``eval golden --serving-config``) and the
    realization corpus ask their question inside this, which is what makes the answer the same on
    every machine that holds the same rows."""
    with records_override(records), config.strict_evidence_override(True):
        yield


def scope_explicit() -> bool:
    """Whether a caller scoped the golden evidence to records of its own choosing — an in-process
    override or ``EMMY_GOLDEN_FILE`` — rather than the repository corpus."""
    return RECORDS_OVERRIDE is not None or config.golden_scope() is not None


@contextmanager
def _scope(gpu_name: str):
    """Yield where the golden rows a compile on ``gpu_name`` reads come from, ``(records, files)``: the installed scope
    (:data:`RECORDS_OVERRIDE`) and no file, else the ``EMMY_GOLDEN_FILE`` file — none when set empty, which is no
    golden evidence — else the card's repository files, sniffed by header (a wheel's live only inside this block)."""
    if RECORDS_OVERRIDE is not None:
        yield RECORDS_OVERRIDE, []
    elif (scope := config.golden_scope()) is not None:
        yield None, [Path(scope)] if scope else []
    else:
        with _card_golden_paths(gpu_name) as paths:
            yield None, paths


def records_for_card(gpu_name: str, compute_cap: tuple[int, int]) -> GoldenRecords:
    """The golden records the evidence index loads for ONE card (:func:`_scope`, then the card's rows among them).
    ``golden_records()`` stays the full corpus for the eval consumers."""
    gpu_name = gpu.canonical_name(gpu_name)
    with _scope(gpu_name) as (records, paths):
        if records is None:
            records = GoldenRecords(record for path in paths for record in _records_of(path))
    return records.for_card(gpu_name, compute_cap)


@contextmanager
def _card_golden_paths(gpu_name: str):
    """The repository golden files that can hold ``gpu_name``'s rows: a file whose header names another card is skipped
    unparsed, one whose header names none is kept for the parse to decide."""
    with repository_golden_paths() as paths:
        yield [path for path in paths if (head := _file_gpu_name(path)) is None or head == gpu_name]


def scope_digest(gpu_name: str) -> str:
    """A digest of the golden rows :func:`records_for_card` would load for ``gpu_name`` (:func:`_scope`) — taken over
    file bytes where they come from files, so it costs no parse. A serving pack keys on it: plans compiled from other
    rows are not what this compile would deploy."""
    sha = hashlib.sha256()
    with _scope(gpu_name) as (records, paths):
        if records is not None:
            # The target and its bindings too: a case and its symbolic twin spell the same names, pins and knobs
            # over different programs, and a compile of one must not pick from the other's rows.
            rows = (
                json.dumps(
                    [
                        r.name,
                        r.gpu_name,
                        r.target_key,
                        r.bindings,
                        r.identity,
                        r.pins,
                        r.knobs,
                        r.measurements and r.measurements.to_wire(),
                    ],
                    sort_keys=True,
                    default=str,
                )
                for r in records
            )
            sha.update("\n".join(sorted(rows)).encode())
        for path in paths:
            data = path.read_bytes()
            sha.update(f"{path.name}:{len(data)}:".encode() + data)
    return sha.hexdigest()[:16]


_DOCUMENT_MEMO: dict[Path, tuple[GoldenFile, GoldenRecords]] = {}


def _document_of(path: Path) -> tuple[GoldenFile, GoldenRecords]:
    """A golden parsed once per process: ``(document, records)``. The parse is the whole cost of a load — the
    36 MB FP8 golden takes 16 s — and the test collection reads every file three times, every row test once
    more; nothing derived is kept here, so nothing here can go stale within a process."""
    path = Path(path)
    cached = _DOCUMENT_MEMO.get(path)
    if cached is None:
        document = GoldenFile.load(path)
        cached = _DOCUMENT_MEMO.setdefault(path, (document, document.records()))
    return cached


def _records_of(path: Path) -> GoldenRecords:
    return _document_of(path)[1]


@cache
def golden_records() -> GoldenRecords:
    """Every row of every repository golden, loaded on first use — the corpus the eval consumers read."""
    with repository_golden_paths() as paths:
        return GoldenRecords(record for path in paths for record in _records_of(path))


def goldens_for_live_gpu() -> GoldenRecords:
    """The live card's own rows, or every row when no CUDA card is visible or none are recorded for it."""
    key = live_gpu_key()
    records = golden_records()
    if key is None:
        return records
    return GoldenRecords(record for record in records if record.gpu_name == key[0] and record.compute_cap == key[1]) or records


def live_gpu_key() -> tuple[str, tuple[int, int]] | None:
    try:
        import torch  # noqa: PLC0415 — heavy, and only the live-card path needs it

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        return gpu.canonical_name(name), tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None
