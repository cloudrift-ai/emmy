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

import yaml

from emmy import config, gpu
from emmy.recipe.bundled import default_recipe_root
from emmy.recipe.lifecycle import MAINTAINED_TAG, recipe_lifecycle

from .format import GoldenFile

logger = logging.getLogger("emmy.compiler.pipeline")

#: The maintained hardware goldens, one file per exact card: rows the offline prior trains on, the tests
#: check, and a compile on that card picks from. They ship inside this package.
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
def repository_golden_paths(*, maintained: bool = False):
    """Yield hardware goldens plus recipe-local model goldens — with ``maintained``, only maintained recipes' goldens,
    the set the default test suite checks."""
    with default_recipe_root() as recipe_root:
        paths = list(_RECORDS_DIR.glob("*.json"))
        if recipe_root is not None:
            paths.extend(
                path
                for path in recipe_root.glob(f"*/{_RECIPE_GOLDEN_DIR}/*.json")
                if not maintained or _is_maintained(path.parent.parent / "recipe.yaml")
            )
        yield sorted(paths)


def _is_maintained(recipe: Path) -> bool:
    return recipe_lifecycle(yaml.safe_load(recipe.read_text()) or {}) == MAINTAINED_TAG


def _file_gpu_name(path: Path) -> str | None:
    """The document's ``gpu_name`` read off the file's first line without parsing the body — the dump writes it
    there alone, so a card-scoped consumer can skip foreign multi-megabyte files."""
    try:
        with Path(path).open() as handle:
            head = handle.readline().rstrip()
        return gpu.canonical_name(str(json.loads(head.removesuffix(",") + "}")["gpu_name"]))
    except (OSError, ValueError, KeyError):
        return None


#: The scope override: the golden files a compile reads instead of the repository's. ``None`` (the default) reads
#: ``EMMY_GOLDEN_FILE`` when set, else the repository files of the card. ``run`` / ``compile`` install the file
#: they were given in-process, the release gate one precision lane of it, and ``serve --golden`` reaches the same
#: loader through the env var because the vLLM child is another process. Set it through :func:`evidence_scope`.
SCOPE: list[GoldenFile] | None = None


@contextmanager
def evidence_scope(documents: Iterable[GoldenFile] | None):
    """Scope the golden files :func:`documents_for_card` supplies to the evidence index, restoring the previous
    scope after. ``[]`` hides every file — how a caller that must measure without golden evidence says so; ``None``
    is a no-op. The body must not ``await``: this swaps a module global."""
    global SCOPE  # noqa: PLW0603 — the documented scope seam, one owner
    if documents is None:
        yield
        return
    prev = SCOPE
    SCOPE = list(documents)
    try:
        yield
    finally:
        SCOPE = prev


@contextmanager
def sole_evidence(documents: Iterable[GoldenFile]):
    """``documents`` as a compile's ONLY evidence, strictly: the golden scope is these files and strict evidence is
    on, so a fork none of their rows decides is an ``EvidenceError`` naming the kernel instead of a prediction; a
    ``Pipeline.run`` given no ``db`` consults no tune DB either."""
    with evidence_scope(documents), config.strict_evidence_override(True):
        yield


def scope_explicit() -> bool:
    """Whether a caller scoped the golden evidence to files of its own choosing — an in-process override or
    ``EMMY_GOLDEN_FILE`` — rather than the repository corpus."""
    return SCOPE is not None or config.golden_scope() is not None


@contextmanager
def _scope(gpu_name: str):
    """Where the golden files a compile on ``gpu_name`` reads come from: ``(documents, paths)`` — the installed scope
    and no path, else the ``EMMY_GOLDEN_FILE`` file (none when set empty), else the card's repository files sniffed
    by header (a wheel's live only inside this block)."""
    if SCOPE is not None:
        yield SCOPE, []
    elif (scope := config.golden_scope()) is not None:
        yield None, [Path(scope)] if scope else []
    else:
        with repository_golden_paths() as paths:
            yield None, [path for path in paths if (head := _file_gpu_name(path)) is None or head == gpu_name]


def documents_for_card(gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenFile]:
    """The golden files that are evidence on one card: the files in scope whose capability agrees and that name
    this card or none (a working golden traced off-GPU applies to whichever card compiles it). A file another card of
    the same capability recorded is no evidence here, and the compile says so."""
    gpu_name = gpu.canonical_name(gpu_name)
    with _scope(gpu_name) as (documents, paths):
        if documents is None:
            documents = [document_of(path) for path in paths]
    kept = []
    for document in documents:
        if tuple(document.compute_cap) != tuple(compute_cap):
            continue
        card = gpu.canonical_name(document.gpu_name or "")
        if card and gpu_name and card != gpu_name:
            logger.warning("golden scope: %d row(s) measured on %s are no evidence on %s", len(document.rows), card, gpu_name)
            continue
        kept.append(document)
    return kept


def scope_digest(gpu_name: str) -> str:
    """A digest of the golden files :func:`documents_for_card` would load for ``gpu_name`` — over file bytes where
    they come from files, so it costs no parse. A serving pack keys on it: plans compiled from other rows are not
    what this compile would deploy."""
    sha = hashlib.sha256()
    with _scope(gpu.canonical_name(gpu_name)) as (documents, paths):
        for document in documents or []:
            sha.update(json.dumps(document.to_wire(), sort_keys=True, default=str).encode())
        for path in paths:
            data = path.read_bytes()
            sha.update(f"{path.name}:{len(data)}:".encode() + data)
    return sha.hexdigest()[:16]


@cache
def document_of(path: Path) -> GoldenFile:
    """A repository golden parsed once per process: the parse is the whole cost of a load, and the suite reads every
    file many times."""
    return GoldenFile.load(Path(path))


def repository_documents(gpu_name: str | None = None, compute_cap: tuple[int, int] | None = None) -> list[GoldenFile]:
    """Every repository golden — of one card when it is named and any file names it, else all of them."""
    with repository_golden_paths() as paths:
        documents = [document_of(path) for path in paths]
    if gpu_name is None:
        return documents
    card = gpu.canonical_name(gpu_name)
    mine = [d for d in documents if gpu.canonical_name(d.gpu_name or "") == card and tuple(d.compute_cap) == tuple(compute_cap)]
    return mine or documents


def live_gpu_key() -> tuple[str, tuple[int, int]] | None:
    try:
        import torch  # noqa: PLC0415 — heavy, and only the live-card path needs it

        if not torch.cuda.is_available():
            return None
        return gpu.canonical_name(torch.cuda.get_device_name(0)), tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None
