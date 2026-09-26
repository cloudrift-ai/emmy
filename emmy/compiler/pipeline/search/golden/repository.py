"""The repository's goldens: where they live, which of them a card reads, and the evidence scope a compile installs
over them."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import yaml

from emmy import config, gpu
from emmy.recipe.bundled import default_recipe_root

from .format import _SAFE_LOADER, GoldenFile
from .record import GoldenRecord

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
def _repository_golden_paths():
    """Yield model-agnostic hardware goldens plus recipe-local model goldens."""
    with default_recipe_root() as recipe_root:
        paths = list(_RECORDS_DIR.glob("*.yaml"))
        if recipe_root is not None:
            paths.extend(recipe_root.glob(f"*/{_RECIPE_GOLDEN_DIR}/*.yaml"))
        yield sorted(paths)


def _file_gpu_name(path: Path) -> str | None:
    """The document's ``gpu_name`` read off the file HEAD without parsing the body — the dump
    writes it as the first key, so a card-scoped consumer can skip foreign multi-megabyte files
    (the whole-corpus parse is the dominant first-evidence cost). ``None`` when the head does not
    carry it — the caller falls back to the full parse."""
    try:
        head = path.open("r").read(256)
    except OSError:
        return None
    for line in head.splitlines():
        if line.startswith("gpu_name:"):
            return gpu.canonical_name(str(yaml.load(line, Loader=_SAFE_LOADER)["gpu_name"]))
    return None


#: Optional scope override for :func:`records_for_card` — the golden rows the evidence index loads.
#: ``None`` (the default) reads ``EMMY_GOLDEN_FILE`` when set, else the repository files. Every
#: command that names a golden scopes the rows here: ``run`` / ``compile`` install the selected
#: records in-process, the release gate (``eval golden --serving-config``) one precision lane's
#: records through :func:`sole_evidence`, and ``serve --golden`` reaches the same loader through
#: the env var because the vLLM child is another process. Set it through :func:`records_override`,
#: never by hand.
RECORDS_OVERRIDE: list[GoldenRecord] | None = None


@contextmanager
def records_override(records: list[GoldenRecord] | None):
    """Scope the golden rows :func:`records_for_card` supplies to the evidence index, restoring
    the previous scope after. ``[]`` hides every record — how a caller that must measure without
    golden evidence (the tuner) says so; ``None`` is a no-op, leaving whatever scope is already
    installed.

    **The body must not ``await``.** This swaps a module global, so it is only atomic with respect
    to other coroutines while the block stays synchronous — and it is used inside concurrently
    gathered tune targets, which share one event loop."""
    global RECORDS_OVERRIDE  # noqa: PLW0603 — the documented scope seam, one owner
    if records is None:
        yield
        return
    prev = RECORDS_OVERRIDE
    RECORDS_OVERRIDE = records
    try:
        yield
    finally:
        RECORDS_OVERRIDE = prev


@contextmanager
def sole_evidence(records: list[GoldenRecord]):
    """``records`` as a compile's ONLY evidence, strictly: the golden scope is these rows
    (:func:`records_override`), the machine-local online prior and its reservoir are out of the
    way (``EMMY_ONLINE_FILE`` at a nonexistent path) and strict evidence is on, so a fork none of
    the rows decides is an ``EvidenceError`` naming the kernel instead of a prediction; a
    ``Pipeline.run`` given no ``db`` consults no tune DB either. The release gate (``eval golden
    --serving-config``) and the realization corpus ask their question inside this, which is what
    makes the answer the same on every machine that holds the same rows."""
    with tempfile.TemporaryDirectory(prefix="emmy-evidence-") as tmp:
        with (
            records_override(records),
            config.online_file_override(Path(tmp) / "absent-online.json"),
            config.strict_evidence_override(True),
        ):
            yield


def scope_explicit() -> bool:
    """Whether a caller scoped the golden evidence to records of its own choosing — an in-process
    override or ``EMMY_GOLDEN_FILE`` — rather than the repository corpus."""
    return RECORDS_OVERRIDE is not None or config.golden_scope() is not None


def _scoped(records: Sequence[GoldenRecord], gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenRecord]:
    """An explicit scope's records for one card: the capability must agree; a record that names
    no card (a working golden traced off-GPU) applies to whichever card compiles it."""
    return [r for r in records if tuple(r.compute_cap) == tuple(compute_cap) and (not r.gpu_name or r.gpu_name == gpu_name)]


def records_for_card(gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenRecord]:
    """The golden records the evidence index loads for ONE card: the installed scope when one is set
    (:data:`RECORDS_OVERRIDE`, else ``EMMY_GOLDEN_FILE`` — a file, or none when set empty), otherwise
    the repository files, loading only that card's (header sniff). ``golden_records()`` stays the full corpus for the eval / fit
    consumers."""
    gpu_name = gpu.canonical_name(gpu_name)
    if RECORDS_OVERRIDE is not None:
        return _scoped(RECORDS_OVERRIDE, gpu_name, compute_cap)
    if (scope := config.golden_scope()) is not None:
        # A path scopes the evidence to that file; the empty form (``EMMY_GOLDEN_FILE=``) is no golden evidence.
        return _scoped(_records_of(Path(scope)), gpu_name, compute_cap) if scope else []
    records: list[GoldenRecord] = []
    with _card_golden_paths(gpu_name) as paths:
        for path in paths:
            records.extend(r for r in _records_of(path) if r.gpu_name == gpu_name and tuple(r.compute_cap) == tuple(compute_cap))
    return records


@contextmanager
def _card_golden_paths(gpu_name: str):
    """The repository golden files that can hold ``gpu_name``'s rows: a file whose header names another card is skipped
    unparsed, one whose header names none is kept for the parse to decide."""
    with _repository_golden_paths() as paths:
        yield [path for path in paths if (head := _file_gpu_name(path)) is None or head == gpu_name]


def scope_digest(gpu_name: str) -> str:
    """A digest of the golden rows :func:`records_for_card` would load for ``gpu_name`` — the installed override's, the
    ``EMMY_GOLDEN_FILE`` file's, or the card's repository files' — taken over file bytes, so it costs no parse. A
    serving pack keys on it: plans compiled from other rows are not what this compile would deploy."""
    sha = hashlib.sha256()
    if RECORDS_OVERRIDE is not None:
        # The target and its bindings too: a case and its symbolic twin spell the same names, pins and knobs
        # over different programs, and a compile of one must not pick from the other's rows.
        rows = (
            json.dumps(
                [r.name, r.gpu_name, r.target_key, r.bindings, r.identity, r.pins, r.knobs, r.measurements and r.measurements.to_wire()],
                sort_keys=True,
                default=str,
            )
            for r in RECORDS_OVERRIDE
        )
        sha.update("\n".join(sorted(rows)).encode())
    elif (scope := config.golden_scope()) is not None:
        sha.update(Path(scope).read_bytes() if scope else b"no golden evidence")
    else:
        with _card_golden_paths(gpu_name) as paths:
            for path in paths:
                data = path.read_bytes()
                sha.update(f"{path.name}:{len(data)}:".encode() + data)
    return sha.hexdigest()[:16]


_DOCUMENT_MEMO: dict[Path, tuple[GoldenFile, list[GoldenRecord]]] = {}


def _document_of(path: Path) -> tuple[GoldenFile, list[GoldenRecord]]:
    """A golden parsed once per process: ``(document, records)``. The parse is the whole cost of a load — the
    36 MB FP8 golden takes 16 s — and the test collection reads every file three times, every row test once
    more; nothing derived is kept here, so nothing here can go stale within a process."""
    path = Path(path)
    cached = _DOCUMENT_MEMO.get(path)
    if cached is None:
        document = GoldenFile.load(path)
        cached = _DOCUMENT_MEMO.setdefault(path, (document, document.records()))
    return cached


def _records_of(path: Path) -> list[GoldenRecord]:
    return _document_of(path)[1]


@cache
def golden_records() -> tuple[GoldenRecord, ...]:
    """Every row of every repository golden, loaded on first use — the corpus the eval and fit consumers read."""
    with _repository_golden_paths() as paths:
        return tuple(record for path in paths for record in _records_of(path))


def goldens_for_live_gpu() -> list[GoldenRecord]:
    """The live card's own rows, or every row when no CUDA card is visible or none are recorded for it."""
    key = _live_gpu_key()
    records = list(golden_records())
    if key is None:
        return records
    return [record for record in records if record.gpu_name == key[0] and record.compute_cap == key[1]] or records


def _live_gpu_key() -> tuple[str, tuple[int, int]] | None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        return gpu.canonical_name(name), tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None
