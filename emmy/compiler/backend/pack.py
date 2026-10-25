"""Pack — an on-disk bundle of execution plans for one model × GPU × serving shape.

Layout::

    pack/
      manifest.json        # validity key + environment tags + provenance + program index + each program's evidence
      plan/<program>.json  # one serialized ExecutionPlan per compiled program (plan.py)

Kernel binaries are NOT stored here: each plan references its cubins by content-addressed
cache key into the shared ``EMMY_CUBIN_CACHE``, so multiple packs (and repeated layers within
one pack) dedupe to the same cubin files, and the docker bake ships pack + cubin cache + model
snapshot together. Weights are external too (the HF checkpoint), rebound by
``WeightSpec.source_path`` at load.

Validity: a pack loads only when its manifest matches the caller's ``key`` (model identity +
serving shape — composed by the runner; "identity" has to include whatever the compiled programs
read off the CHECKPOINT, not just the architecture config — see the serving runners' keys) AND
the current environment (backend, device arch, nvcc toolkit tag + flags — the same tags the cubin
cache keys on — the precision pins, and the version kernel identities are computed under) AND every
referenced cubin still exists. **Any mismatch or error returns ``None`` and the caller falls back to
the full compile path** — a stale or damaged pack costs a recompile, never a wrong result. Compiler
version is deliberately NOT part of validity (a pack keeps serving its frozen snapshot);
``PLAN_FORMAT_VERSION`` gates the runtime contract instead.

The golden rows are part of validity per program, because a re-recorded golden exists precisely to
change what the compile deploys: each entry keeps the exact identities of the kernels whose evidence
its compile read (``policy.greedy.evidence_reads``) and a digest of the golden rows in scope those
reach (``golden.rows_digests``). After a golden edit (the card's scope digest moved) a load computes
the digests again; an entry whose rows moved maps to no plan and compiles again, and so does every
twin of its wrapper, which settled its weights' layouts with it. The rest load.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from emmy.compiler.backend.plan import (
    PLAN_FORMAT_VERSION,
    ExecutionPlan,
    KernelSpec,
    plan_from_dict,
    plan_to_dict,
)

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


class PackEntry(NamedTuple):
    """One program of a loaded pack: its plan, or ``None`` when the golden rows its compile read moved since the
    pack was written, and the exact identities of the kernels whose evidence that compile read."""

    plan: ExecutionPlan | None
    reads: frozenset[str]


def _environment() -> dict:
    """The environment half of the validity key — the same tags the cubin cache keys on
    (arch / toolkit / flags), so a pack never references cubins the current toolchain
    wouldn't have produced, PLUS the precision-gate state: the ``FAST_MATH``-family pins
    change which kernel forks the compile enumerates (not the nvcc flags), so a pack
    warmed in one precision lane must never serve a boot in the other. Found live
    (2026-07-31 article repro): an ``EMMY_FAST_MATH=1`` boot at the baked serving shape
    pack-hit the std plans and silently served std kernels — the fm lane's numbers were
    the std lane's. A pack whose manifest predates the field mismatches and falls back
    to the full compile, which is the conservative reading (its lane is unrecorded).
    And the card, whose own rows and specs the compile picks from even where no golden row
    decides, and the version of the kernel identities an entry's reads name (the tune DB's
    key version): identities computed under two versions do not compare.
    Probes the live GPU."""
    from emmy.compiler.backend.cuda import nvcc  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import _VERSION  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import live_gpu_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import (
        # noqa: PLC0415,
        F16_MMA_F32_ACC,
        FAST_EXP,
        FP8_MMA,
        precision_pin,
    )

    return {
        "backend": "cuda",
        "arch": nvcc.device_arch(False),
        "toolkit": nvcc._toolkit_tag(),
        "nvcc_flags": nvcc.effective_flags(),
        "precision": {k.name: precision_pin(k) for k in (FAST_EXP, F16_MMA_F32_ACC, FP8_MMA)},
        "card": (live_gpu_key() or ("",))[0],
        "kernel_key": _VERSION,
    }


def _scope_digest() -> str:
    """The live card's golden scope digest (``golden.scope_digest``): file bytes only, so it costs no lift."""
    from emmy.compiler.pipeline.search.golden import live_gpu_key, scope_digest  # noqa: PLC0415

    return scope_digest((live_gpu_key() or ("",))[0])


def _rows(reads: dict[str, frozenset[str]]) -> dict[str, str]:
    """The digest of each program's golden rows in the live card's scope (``golden.rows_digests``, which lifts the
    files)."""
    from emmy.compiler.pipeline.search.golden import documents_for_card, live_gpu_key, rows_digests  # noqa: PLC0415

    card = live_gpu_key()
    return rows_digests(documents_for_card(*card) if card else [], reads)


def _bound(plan: ExecutionPlan) -> frozenset[str]:
    """The checkpoint tensors a plan binds: the twins of one wrapper bind the same ones."""
    return frozenset(path for w in plan.weights.values() for path in (w.source_path, *(p for p, _ in w.source_parts)) if path)


def _safe_name(program: str) -> str:
    return re.sub(r"[^-\w.]", "_", program)


def pack_path(root: Path | str, key: dict) -> Path:
    """The directory for one pack under the ``EMMY_PACK_DIR`` root: a human-readable model
    label plus a digest of BOTH halves of the validity key — the serving shape (``key``) and
    the environment (:func:`_environment`) — so every (shape, environment) combination gets
    its own directory.

    The environment half is load-bearing in the PATH, not just in the manifest: two lanes that
    differ only in environment (the ``FAST_MATH`` precision gate is the live case) build the
    same ``key`` and would otherwise share one directory, where the second warm silently
    overwrites the first. Found by baking a multi-shape image (2026-08-02): the three
    fast-math shapes clobbered their standard-lane twins, leaving 5 directories for 8 shapes
    and a pinned-shape boot that mismatched its own pack and fell back to a full compile."""
    env_digest = hashlib.sha1(json.dumps(_environment(), sort_keys=True).encode()).hexdigest()[:8]
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    label = _safe_name(str(key.get("model", "model")).split("/")[-1])
    return Path(root) / f"{label}-{digest}-{env_digest}"


def save_pack(
    pack_dir: Path | str,
    plans: dict[str, ExecutionPlan],
    *,
    key: dict,
    reads: dict[str, frozenset[str]],
    provenance: dict | None = None,
) -> Path:
    """Write a pack: resolve every kernel to its content-addressed cubin key (compiling into
    the cache on a miss — normally a hit, the caller just built these programs; a plan loaded
    from a pack keeps its keys) and store the plans binary-keyed with no sources, plus the
    manifest. Requires the live GPU (arch probe).

    ``key`` is the caller-composed validity dict (model id / config hash / serving shape);
    ``reads`` names, for every program, the kernels whose evidence its compile read (a loaded
    program's :attr:`PackEntry.reads`); ``provenance`` is informational only (compiler rev, tune
    state) and never checked."""
    from emmy.compiler.backend.cuda import nvcc  # noqa: PLC0415

    root = Path(pack_dir)
    (root / "plan").mkdir(parents=True, exist_ok=True)
    names = {p: _safe_name(p) for p in plans}
    if len(set(names.values())) != len(names):
        raise ValueError(f"pack: program names collide after sanitization: {sorted(plans)}")

    rows = _rows(reads)
    index: dict[str, str] = {}
    for program, plan in plans.items():
        kernels: dict[str, KernelSpec] = {}
        for kname, spec in plan.kernels.items():
            if spec.source is None and spec.binary_key is None:
                raise ValueError(f"pack: kernel {kname!r} of program {program!r} carries no source — cannot resolve a cubin key")
            binary_key = (
                spec.binary_key
                if spec.source is None
                else nvcc.compile_to_cubin(spec.source, kname, arch=nvcc.device_arch(spec.arch_specific)).stem
            )
            kernels[kname] = KernelSpec(
                source=None, binary_key=binary_key, arch_specific=spec.arch_specific, dependent_launch=spec.dependent_launch
            )
        stored = dataclasses.replace(plan, kernels=kernels)
        rel = f"plan/{names[program]}.json"
        (root / rel).write_text(json.dumps(plan_to_dict(stored)))
        index[program] = rel

    manifest = {
        "format": PLAN_FORMAT_VERSION,
        "environment": _environment(),
        "key": json.loads(json.dumps(key)),
        "programs": index,
        "golden": _scope_digest(),
        "evidence": {program: {"reads": sorted(reads[program]), "rows": rows[program]} for program in plans},
        "provenance": provenance or {},
    }
    (root / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    logger.info("[pack] saved %d program plan(s) to %s", len(index), root)
    return root


def load_pack(pack_dir: Path | str, *, key: dict) -> dict[str, PackEntry] | None:
    """Load a pack's entries, or ``None`` when anything disqualifies it — no manifest, format /
    environment / key mismatch, an unparsable plan, or a referenced cubin gone from the cache.
    ``None`` means "boot the full compile path"; the reason is logged. When the card's golden
    scope moved since the save, an entry whose golden rows moved — or that shares its wrapper
    with one — comes back with no plan, and with none of them moved the manifest takes the new
    scope digest, so the next load skips the comparison."""
    from emmy import config  # noqa: PLC0415

    root = Path(pack_dir)
    try:
        manifest = json.loads((root / _MANIFEST).read_text())
    except (OSError, ValueError):
        logger.info("[pack] no readable manifest at %s — full compile", root)
        return None
    try:
        if manifest.get("format") != PLAN_FORMAT_VERSION:
            logger.info("[pack] format %r != runtime %r — full compile", manifest.get("format"), PLAN_FORMAT_VERSION)
            return None
        env = _environment()
        if manifest.get("environment") != env:
            logger.info("[pack] environment mismatch (pack %r vs current %r) — full compile", manifest.get("environment"), env)
            return None
        want = json.loads(json.dumps(key))
        if manifest.get("key") != want:
            logger.info("[pack] key mismatch (pack %r vs wanted %r) — full compile", manifest.get("key"), want)
            return None
        plans: dict[str, ExecutionPlan] = {}
        cache = config.cubin_cache_dir()
        evidence = manifest["evidence"]
        for program, rel in manifest["programs"].items():
            plan = plan_from_dict(json.loads((root / rel).read_text()))
            for kname, spec in plan.kernels.items():
                if spec.binary_key is None or not (cache / f"{spec.binary_key}.cubin").exists():
                    logger.info("[pack] cubin for kernel %r of %r missing from cache — full compile", kname, program)
                    return None
            plans[program] = plan
        golden = _scope_digest()
        stale: set[str] = set()
        if manifest["golden"] != golden:
            rows = _rows({program: evidence[program]["reads"] for program in plans})
            moved = {program for program in plans if rows[program] != evidence[program]["rows"]}
            wrappers = {_bound(plans[program]) for program in moved} - {frozenset()}
            stale = moved | {program for program, plan in plans.items() if _bound(plan) in wrappers}
            logger.info(
                "[pack] golden scope moved: the rows of %d of %d program(s) moved, %d compile again%s",
                len(moved),
                len(plans),
                len(stale),
                f" ({', '.join(sorted(stale))})" if stale else "",
            )
            if not stale:
                (root / _MANIFEST).write_text(json.dumps({**manifest, "golden": golden}, indent=2))
    except Exception:  # noqa: BLE001 — a damaged pack must cost a recompile, never a crash
        logger.warning("[pack] failed to load %s — full compile", root, exc_info=True)
        return None
    logger.info("[pack] loaded %d program plan(s) from %s", len(plans) - len(stale), root)
    return {
        program: PackEntry(None if program in stale else plan, frozenset(evidence[program]["reads"])) for program, plan in plans.items()
    }


def save_executable(
    pack_dir: Path | str,
    plans: dict[str, ExecutionPlan],
    *,
    bindings: dict[str, dict[str, bytes | Callable[[], bytes]]],
    key: dict,
    reads: dict[str, frozenset[str]],
    provenance: dict | None = None,
    pages: dict | None = None,
) -> Path:
    """Bundle a pack with cubins and resolved constant/input bytes for independent execution; ``key``,
    ``reads`` and ``provenance`` are :func:`save_pack`'s.

    Bindings are contiguous little-endian storage bytes in each buffer's declared dtype;
    bf16 uses its uint16 bit carrier. Every constant must be supplied, including scalars and
    generated weights. Input bindings are optional. No checkpoint or cache is needed at runtime.
    A binding is its bytes or a function returning them, called once, as its file is written, so a
    caller can read a trunk's weights in turn and never hold them all; one binding object several
    programs hold is one file, which the runtime reads once and lends.
    ``pages`` holds, for each buffer the runtime's pager holds, a function returning the same bytes
    and how many of its first pages are resident: paged along its leading axis with fewer device slots
    than pages, so its storage is its pages one after another. Each function is called once, as its
    file is written, so a caller can read them in turn and never hold them all. The pack keeps them
    once for every program that declares the buffer, and the runtime, which checks them against the
    plans, reads the resident pages straight onto the device and the rest straight into pinned host
    memory. The destination must not exist; a failed export never publishes a partial artifact.
    """
    from emmy import config  # noqa: PLC0415

    root = Path(pack_dir)
    if root.exists():
        raise FileExistsError(root)
    if set(bindings) != set(plans):
        raise ValueError("executable bindings must name every program exactly once")
    sizes = {}
    for name, plan in plans.items():
        buffers = {b.name: b for b in plan.buffers}
        required = {b.name for b in plan.buffers if b.role == "constant"}
        if not required <= bindings[name].keys():
            raise ValueError(f"missing constant bindings for {name}: {sorted(required - bindings[name].keys())}")
        for buffer_name in bindings[name]:
            buffer = buffers.get(buffer_name)
            if buffer is None or buffer.role not in ("input", "constant"):
                raise ValueError(f"invalid binding: {name}/{buffer_name}")
            if buffer.is_symbolic:
                raise ValueError("standalone bindings require static shapes")
            sizes[name, buffer_name] = buffer.dtype.nbytes * math.prod(buffer.resolve_shape({}))
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".emmy-pack-", dir=root.parent) as temporary:
        stage = Path(temporary) / "bundle"
        save_pack(stage, plans, key=key, reads=reads, provenance=provenance)
        manifest = json.loads((stage / _MANIFEST).read_text())
        (stage / "cubin").mkdir()
        (stage / "bindings").mkdir()
        manifest["standalone"] = 1
        manifest["bindings"] = {}
        written = {}  # id(binding) -> its file
        for index, (program, rel) in enumerate(manifest["programs"].items()):
            stored = json.loads((stage / rel).read_text())
            for kernel in stored["kernels"].values():
                filename = f"{kernel['binary_key']}.cubin"
                destination = stage / "cubin" / filename
                if not destination.exists():
                    shutil.copyfile(config.cubin_cache_dir() / filename, destination)
            manifest["bindings"][program] = {}
            for slot, (name, binding) in enumerate(bindings[program].items()):
                if id(binding) not in written:
                    data = binding() if callable(binding) else binding
                    if not isinstance(data, bytes) or len(data) != sizes[program, name]:
                        raise ValueError(f"binding size mismatch: {program}/{name}")
                    written[id(binding)] = f"bindings/{index}-{slot}.bin"
                    (stage / written[id(binding)]).write_bytes(data)
                manifest["bindings"][program][name] = written[id(binding)]
        if pages:
            (stage / "pages").mkdir()
            manifest["pages"] = {}
            for index, (name, (source, resident)) in enumerate(pages.items()):
                manifest["pages"][name] = {"path": (rel := f"pages/{index}.bin"), "resident": resident}
                (stage / rel).write_bytes(memoryview(source()))
        (stage / _MANIFEST).write_text(json.dumps(manifest, indent=2))
        stage.rename(root)
    return root
